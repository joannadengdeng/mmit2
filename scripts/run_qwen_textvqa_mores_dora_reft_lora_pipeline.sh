#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$ROOT_DIR/experiments}"
mkdir -p "$EXPERIMENTS_DIR"
EXPERIMENTS_DIR="$(cd "$EXPERIMENTS_DIR" && pwd -P)"
STATUS_FILE="${STATUS_FILE:-$EXPERIMENTS_DIR/qwen_textvqa_mores_dora_reft_lora_s42.status}"
PHASE_FILE="${PHASE_FILE:-$EXPERIMENTS_DIR/qwen_textvqa_mores_dora_reft_lora_s42.phase}"
LOG_FILE="${LOG_FILE:-$EXPERIMENTS_DIR/qwen_textvqa_mores_dora_reft_lora_s42.log}"
# Share the lock with every TextVQA combination pipeline so only one owns the
# GPU and combination output namespace at a time.
LOCK_FILE="${LOCK_FILE:-$EXPERIMENTS_DIR/qwen_textvqa_combinations.lock}"
VENV_DIR="${VENV_DIR:-/root/autodl-tmp/venvs/vlmintune-torch211}"
METHODS="mores_dora reft_lora"
STAGES="8:8 256:32 1000:100 34602:5000"
RUN_PREFIX="qwen_textvqa_combo"
MAX_LENGTH=1536
EPOCHS=1
SEED=42
LEARNING_RATE_DEFAULT=2e-4

read -r -a method_list <<< "$METHODS"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ "${VLMINTUNE_COMBO_LOCK_HELD:-0}" == "1" ]]; then
  expected_lock="$(readlink -f "$LOCK_FILE")"
  inherited_lock="$(readlink "/proc/$$/fd/9" 2>/dev/null || true)"
  if [[ -z "$expected_lock" || "$inherited_lock" != "$expected_lock" ]]; then
    echo "VLMINTUNE_COMBO_LOCK_HELD=1 was set without inherited lock fd 9." >&2
    exit 5
  fi
  if ! flock -n 9; then
    echo "Inherited TextVQA combination lock fd 9 is not held." >&2
    exit 5
  fi
else
  exec 9>>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "Another TextVQA combination queue or pipeline holds $LOCK_FILE" >&2
    exit 5
  fi
fi

printf 'RUNNING\n' > "$STATUS_FILE"
printf 'initializing\n' > "$PHASE_FILE"

record_exit() {
  local status=$?
  printf '%s\n' "$status" > "$STATUS_FILE"
  if [[ "$status" -ne 0 ]]; then
    printf 'failed_status_%s\n' "$status" > "$PHASE_FILE"
  fi
}
trap record_exit EXIT

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing experiment Python: $VENV_DIR/bin/python" >&2
  exit 2
fi

# The stage runner exports this path in its child shell, but the strict audit
# below runs back in this parent shell. Keep both entry points bound to the
# checked-out source tree so newly added combination modules are importable.
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

for stage in $STAGES; do
  if [[ ! "$stage" =~ ^[1-9][0-9]*:[1-9][0-9]*$ ]]; then
    echo "Invalid stage '$stage'; expected TRAIN:EVAL positive integers." >&2
    exit 2
  fi
  stage_samples="${stage%%:*}"
  eval_samples="${stage##*:}"
  grad_acc=4
  if [[ "$stage_samples" -le 8 ]]; then
    grad_acc=1
  fi

  printf 'stage_train_%s_eval_%s\n' "$stage_samples" "$eval_samples" > "$PHASE_FILE"
  echo "================================================================================"
  echo "QWEN TEXTVQA MORES+DORA / REFT+LORA train=$stage_samples eval=$eval_samples"
  echo "================================================================================"
  STAGE_SAMPLES="$stage_samples" \
  EVAL_SAMPLES="$eval_samples" \
  METHODS="$METHODS" \
  RUN_PREFIX="$RUN_PREFIX" \
  MAX_LENGTH="$MAX_LENGTH" \
  EPOCHS="$EPOCHS" \
  SEED="$SEED" \
  GRADIENT_ACCUMULATION_STEPS="$grad_acc" \
  LEARNING_RATE_DEFAULT="$LEARNING_RATE_DEFAULT" \
  BENCHMARK_SCRIPT=experiment_setup/paper_benchmark/run_paper_benchmark.sh \
  FORCE=0 \
  EXPERIMENTS_DIR="$EXPERIMENTS_DIR" \
  bash scripts/run_qwen_textvqa_combination_stage.sh

  "$VENV_DIR/bin/python" scripts/audit_qwen_textvqa_combination_stage.py \
    --experiments-dir "$EXPERIMENTS_DIR" \
    --run-prefix "$RUN_PREFIX" \
    --train-samples "$stage_samples" \
    --eval-samples "$eval_samples" \
    --grad-acc "$grad_acc" \
    --max-length "$MAX_LENGTH" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    "${method_list[@]}"
done

printf 'complete\n' > "$PHASE_FILE"
echo "All Qwen TextVQA MoReS + DoRA and ReFT + LoRA stages completed successfully."
