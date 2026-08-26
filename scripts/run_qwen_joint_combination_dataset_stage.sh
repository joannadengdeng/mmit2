#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-/root/autodl-tmp/venvs/vlmintune-torch211}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$ROOT_DIR/experiments}"
GPU_LOCK_FILE="${GPU_LOCK_FILE:-$EXPERIMENTS_DIR/qwen_textvqa_combinations.lock}"
MODEL="${MODEL:-qwen25vl_3b_instruct}"
DATASET="${DATASET:-}"
RUN_PREFIX="${RUN_PREFIX:-}"
STAGE_SAMPLES="${STAGE_SAMPLES:-8}"
EVAL_SAMPLES="${EVAL_SAMPLES:-$STAGE_SAMPLES}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-42}"
LEARNING_RATE_DEFAULT="${LEARNING_RATE_DEFAULT:-2e-4}"
FORCE="${FORCE:-0}"
METHODS="mores_lora mores_dora reft_lora"

case "$DATASET" in
  lmms-lab/textvqa)
    EVAL_SPLIT=validation
    METRIC=vqa_accuracy
    ;;
  ebrukilic/vizwiz_vqa_dataset)
    EVAL_SPLIT=validation
    METRIC=vqa_accuracy
    ;;
  scienceqa_image)
    EVAL_SPLIT=validation
    METRIC=normalized_exact_match
    ;;
  *)
    echo "This strict joint stage only accepts TextVQA, VizWiz, or ScienceQA image-only." >&2
    exit 2
    ;;
esac

case "$MODEL" in
  qwen25vl_3b_instruct|llava15_7b) ;;
  *)
    echo "Unsupported model for strict joint stage: $MODEL" >&2
    exit 2
    ;;
esac

if [[ -z "$RUN_PREFIX" ]]; then
  echo "RUN_PREFIX is required." >&2
  exit 2
fi
if [[ "$FORCE" != "0" ]]; then
  echo "FORCE must remain 0 for the progressive joint-combination queue." >&2
  exit 2
fi
if [[ "$STAGE_SAMPLES" -le 8 ]]; then
  DEFAULT_GRAD_ACC=1
else
  DEFAULT_GRAD_ACC=4
fi
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GRAD_ACC}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing experiment Python: $VENV_DIR/bin/python" >&2
  exit 2
fi
mkdir -p "$EXPERIMENTS_DIR"
EXPERIMENTS_DIR="$(cd "$EXPERIMENTS_DIR" && pwd -P)"
GPU_LOCK_FILE="$(cd "$(dirname "$GPU_LOCK_FILE")" && pwd -P)/$(basename "$GPU_LOCK_FILE")"

if [[ "${VLMINTUNE_GPU_LOCK_HELD:-0}" == "1" ]]; then
  expected_lock="$(readlink -f "$GPU_LOCK_FILE")"
  inherited_lock="$(readlink "/proc/$$/fd/9" 2>/dev/null || true)"
  if [[ -z "$expected_lock" || "$inherited_lock" != "$expected_lock" ]]; then
    echo "VLMINTUNE_GPU_LOCK_HELD=1 was set without inherited GPU lock fd 9." >&2
    exit 5
  fi
  if ! flock -n 9; then
    echo "Inherited GPU lock fd 9 is not held." >&2
    exit 5
  fi
else
  exec 9>>"$GPU_LOCK_FILE"
  if ! flock -n 9; then
    echo "Another combination pipeline holds $GPU_LOCK_FILE" >&2
    exit 5
  fi
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

audit_method() {
  local scope="$1"
  local method="$2"
  "$VENV_DIR/bin/python" scripts/audit_qwen_joint_combination_dataset_stage.py \
    --experiments-dir "$EXPERIMENTS_DIR" \
    --run-prefix "$RUN_PREFIX" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --eval-split "$EVAL_SPLIT" \
    --metric "$METRIC" \
    --train-samples "$STAGE_SAMPLES" \
    --eval-samples "$EVAL_SAMPLES" \
    --grad-acc "$GRADIENT_ACCUMULATION_STEPS" \
    --max-length "$MAX_LENGTH" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --learning-rate "$LEARNING_RATE_DEFAULT" \
    --scope "$scope" \
    "$method"
}

has_files() {
  local path="$1"
  [[ -d "$path" ]] && [[ -n "$(find "$path" -type f -print -quit)" ]]
}

pending_methods=()
for method in $METHODS; do
  run_name="${RUN_PREFIX}_${method}_n${STAGE_SAMPLES}_s${SEED}"
  experiment_dir="$EXPERIMENTS_DIR/$run_name"
  checkpoint_dir="$experiment_dir/checkpoint"
  eval_dir="$experiment_dir/eval_trained"
  eval_config="$experiment_dir/eval_trained_config.yaml"

  if has_files "$checkpoint_dir"; then
    echo "Found checkpoint artifacts for $run_name; running strict classification."
    if ! audit_method checkpoint "$method"; then
      echo "Checkpoint artifacts are incomplete or invalid; preserving them and stopping this stage." >&2
      echo "No training or evaluation output was overwritten for $run_name." >&2
      exit 6
    fi
    echo "REUSE strict checkpoint: $run_name"
  elif has_files "$experiment_dir"; then
    echo "Partial run exists without a valid checkpoint: $experiment_dir" >&2
    echo "Preserving partial artifacts and stopping before retraining." >&2
    exit 6
  fi

  if has_files "$eval_dir" || [[ -f "$eval_config" ]]; then
    echo "Found evaluation artifacts for $run_name; running strict classification."
    if audit_method all "$method"; then
      echo "SKIP strict completed run: $run_name"
      continue
    fi

    # A valid checkpoint may safely be re-evaluated, but the old evaluation
    # remains recoverable and is never overwritten in place.
    if ! has_files "$checkpoint_dir" || ! audit_method checkpoint "$method"; then
      echo "Evaluation is invalid and no strict checkpoint is available; preserving all artifacts." >&2
      exit 6
    fi
    invalid_stamp="$(date -u +%Y%m%dT%H%M%SZ)_$$"
    if [[ -d "$eval_dir" ]]; then
      invalid_eval_dir="${eval_dir}.invalid_${invalid_stamp}"
      mv "$eval_dir" "$invalid_eval_dir"
      echo "Preserved prior evaluation directory at $invalid_eval_dir"
    fi
    if [[ -f "$eval_config" ]]; then
      invalid_eval_config="${eval_config}.invalid_${invalid_stamp}"
      mv "$eval_config" "$invalid_eval_config"
      echo "Preserved prior evaluation config at $invalid_eval_config"
    fi
  fi
  pending_methods+=("$method")
done

if [[ "${#pending_methods[@]}" -gt 0 ]]; then
  echo "Running strict pending methods: ${pending_methods[*]}"
  DATASET="$DATASET" \
  MODEL="$MODEL" \
  STAGE_SAMPLES="$STAGE_SAMPLES" \
  EVAL_SAMPLES="$EVAL_SAMPLES" \
  METHODS="${pending_methods[*]}" \
  RUN_PREFIX="$RUN_PREFIX" \
  MAX_LENGTH="$MAX_LENGTH" \
  EPOCHS="$EPOCHS" \
  SEED="$SEED" \
  GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS" \
  LEARNING_RATE_DEFAULT="$LEARNING_RATE_DEFAULT" \
  EXPERIMENTS_DIR="$EXPERIMENTS_DIR" \
  FORCE=0 \
  bash scripts/run_qwen_dataset_stage.sh

  for method in "${pending_methods[@]}"; do
    audit_method all "$method"
  done
fi

echo "All joint combinations passed strict dataset=$DATASET train=$STAGE_SAMPLES eval=$EVAL_SAMPLES."
