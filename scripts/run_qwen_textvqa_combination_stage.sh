#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-/root/autodl-tmp/venvs/vlmintune-torch211}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/root/autodl-tmp/hf_cache}"
STAGE_SAMPLES="${STAGE_SAMPLES:-8}"
EVAL_SAMPLES="${EVAL_SAMPLES:-$STAGE_SAMPLES}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-42}"
METHODS="${METHODS:-mores_lora mores_dora reft_lora}"
RUN_PREFIX="${RUN_PREFIX:-qwen_textvqa_combo}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$ROOT_DIR/experiments}"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-experiment_setup/paper_benchmark/run_paper_benchmark.sh}"

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
if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
  echo "Training benchmark script is missing: $BENCHMARK_SCRIPT" >&2
  exit 2
fi
for numeric_value in \
  "$STAGE_SAMPLES" "$EVAL_SAMPLES" "$MAX_LENGTH" "$EPOCHS" \
  "$GRADIENT_ACCUMULATION_STEPS"; do
  if [[ ! "$numeric_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Stage sizes, max length, epochs, and gradient accumulation must be positive integers." >&2
    exit 2
  fi
done

mkdir -p "$EXPERIMENTS_DIR"
EXPERIMENTS_DIR="$(cd "$EXPERIMENTS_DIR" && pwd -P)"

export PATH="$VENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
export HF_XET_CACHE="$HF_CACHE_ROOT/xet"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export VLMINTUNE_FAST_EXIT=1
export TOKENIZERS_PARALLELISM=false

MODEL_CACHE="$HF_HUB_CACHE/models--Qwen--Qwen2.5-VL-3B-Instruct"
if [[ ! -e "$MODEL_CACHE" ]]; then
  echo "Qwen cache is not visible at $MODEL_CACHE" >&2
  exit 2
fi

read -r -a method_list <<< "$METHODS"
if [[ "${#method_list[@]}" -eq 0 ]]; then
  echo "METHODS must contain at least one fixed combination." >&2
  exit 2
fi
for method in "${method_list[@]}"; do
  case "$method" in
    mores_lora|mores_dora|reft_lora) ;;
    *)
      echo "Unsupported TextVQA combination: $method" >&2
      exit 2
      ;;
  esac
done

method_learning_rate() {
  echo "${LEARNING_RATE_DEFAULT:-2e-4}"
}

audit_method() {
  local scope="$1"
  local method="$2"
  "$VENV_DIR/bin/python" scripts/audit_qwen_textvqa_combination_stage.py \
    --experiments-dir "$EXPERIMENTS_DIR" \
    --run-prefix "$RUN_PREFIX" \
    --train-samples "$STAGE_SAMPLES" \
    --eval-samples "$EVAL_SAMPLES" \
    --grad-acc "$GRADIENT_ACCUMULATION_STEPS" \
    --max-length "$MAX_LENGTH" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --scope "$scope" \
    "$method"
}

has_files() {
  local path="$1"
  [[ -d "$path" ]] && [[ -n "$(find "$path" -type f -print -quit)" ]]
}

for method in "${method_list[@]}"; do
  learning_rate="$(method_learning_rate "$method")"
  run_name="${RUN_PREFIX}_${method}_n${STAGE_SAMPLES}_s${SEED}"
  experiment_dir="$EXPERIMENTS_DIR/$run_name"
  checkpoint_dir="$experiment_dir/checkpoint"
  train_dir="$experiment_dir/train"
  eval_dir="$experiment_dir/eval_trained"
  eval_config="$experiment_dir/eval_trained_config.yaml"

  checkpoint_valid=0
  if has_files "$checkpoint_dir"; then
    echo "Found checkpoint artifacts for $run_name; running strict classification."
    if audit_method checkpoint "$method"; then
      checkpoint_valid=1
      echo "REUSE strict checkpoint: $run_name"
    else
      echo "Checkpoint artifacts are incomplete or invalid; preserving them and stopping." >&2
      echo "No training or evaluation output was overwritten for $run_name." >&2
      exit 6
    fi
  elif has_files "$experiment_dir"; then
    echo "Partial run exists without a valid checkpoint: $experiment_dir" >&2
    echo "Preserving partial artifacts and stopping before retraining." >&2
    exit 6
  fi

  if [[ "$checkpoint_valid" -eq 0 ]]; then
    mkdir -p "$train_dir" "$checkpoint_dir"
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    train_log="$train_dir/run_${timestamp}.log"

    echo "================================================================================"
    echo "TRAIN $run_name"
    echo "method=$method samples=$STAGE_SAMPLES eval_samples=$EVAL_SAMPLES"
    echo "lr=$learning_rate max_length=$MAX_LENGTH grad_acc=$GRADIENT_ACCUMULATION_STEPS"
    echo "================================================================================"

    MODEL=qwen25vl_3b_instruct \
    DATASET=lmms-lab/textvqa \
    METHOD="$method" \
    MAX_SAMPLES="$STAGE_SAMPLES" \
    EPOCHS="$EPOCHS" \
    LEARNING_RATE="$learning_rate" \
    BATCH_SIZE=1 \
    GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS" \
    MAX_LENGTH="$MAX_LENGTH" \
    SEED="$SEED" \
    RUN_NAME="$run_name" \
    OUTPUT_DIR="$checkpoint_dir" \
    CONFIG_PATH="$checkpoint_dir/train_config.yaml" \
    DRY_RUN=0 \
    bash "$BENCHMARK_SCRIPT" \
      2>&1 | tee "$train_log"

    audit_method checkpoint "$method"
  fi

  if has_files "$eval_dir" || [[ -f "$eval_config" ]]; then
    echo "Found evaluation artifacts for $run_name; running strict classification."
    if audit_method all "$method"; then
      echo "SKIP strict completed run: $run_name"
      continue
    fi

    invalid_stamp="$(date -u +%Y%m%dT%H%M%SZ)_$$"
    echo "Evaluation is incomplete or invalid; checkpoint passed strict validation."
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

  cat > "$eval_config" <<YAML
model:
  name: qwen25vl_3b_instruct
experiment:
  name: "$run_name"
  base_dir: "$EXPERIMENTS_DIR"
eval:
  source: trained
  dataset_name: lmms-lab/textvqa
  split: validation
  max_samples: $EVAL_SAMPLES
  sample_seed: $SEED
  shuffle_buffer_size: 10000
  max_new_tokens: 16
  temperature: 0.0
YAML

  echo "EVAL $run_name"
  python -m vlmintune.eval --config "$eval_config"
  audit_method all "$method"
  echo "PASS $run_name"
done

echo "All fixed combinations passed strict stage n=$STAGE_SAMPLES eval=$EVAL_SAMPLES."
