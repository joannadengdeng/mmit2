#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$ROOT_DIR/experiments}"
mkdir -p "$EXPERIMENTS_DIR"
EXPERIMENTS_DIR="$(cd "$EXPERIMENTS_DIR" && pwd -P)"
CONTROL_STEM="qwen_small3_joint_combinations_s42"
STATUS_FILE="${STATUS_FILE:-$EXPERIMENTS_DIR/${CONTROL_STEM}.status}"
PHASE_FILE="${PHASE_FILE:-$EXPERIMENTS_DIR/${CONTROL_STEM}.phase}"
LOG_FILE="${LOG_FILE:-$EXPERIMENTS_DIR/${CONTROL_STEM}.log}"
SELF_LOCK_FILE="${SELF_LOCK_FILE:-$EXPERIMENTS_DIR/${CONTROL_STEM}.lock}"
GPU_LOCK_FILE="${GPU_LOCK_FILE:-$EXPERIMENTS_DIR/qwen_textvqa_combinations.lock}"
VENV_DIR="${VENV_DIR:-/root/autodl-tmp/venvs/vlmintune-torch211}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/root/autodl-tmp/hf_cache}"
TEXTVQA_STATUS_FILE="${TEXTVQA_STATUS_FILE:-$EXPERIMENTS_DIR/qwen_textvqa_mores_dora_reft_lora_s42.status}"
TEXTVQA_PHASE_FILE="${TEXTVQA_PHASE_FILE:-$EXPERIMENTS_DIR/qwen_textvqa_mores_dora_reft_lora_s42.phase}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
RETRY_SECONDS="${RETRY_SECONDS:-60}"

METHODS="mores_lora mores_dora reft_lora"
TEXTVQA_STAGES="8:8 256:32 1000:100 34602:5000"
VIZWIZ_STAGES="8:8 256:32 1000:100 20523:4319"
SCIENCEQA_STAGES="8:8 256:32 1000:100 6218:2097"
VIZWIZ_PREFIX="qwen_vizwiz_px1003520_joint3"
SCIENCEQA_PREFIX="qwen_scienceqa_image_joint3"
MAX_LENGTH=1536
EPOCHS=1
SEED=42
LEARNING_RATE_DEFAULT=2e-4

VIZWIZ_REVISION="8458ff83feb8d782b53b11b391cf1dedd961922e"
SCIENCEQA_REVISION="f18b0a70359ebfb41f658fd564208d0355b013f4"

exec > >(tee -a "$LOG_FILE") 2>&1

exec 8>>"$SELF_LOCK_FILE"
if ! flock -n 8; then
  echo "Another $CONTROL_STEM queue already holds $SELF_LOCK_FILE" >&2
  exit 5
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

# This queue is intentionally cache-only. These settings apply to the queue,
# every stage runner, training, evaluation, and cache preflight.
export PATH="$VENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
export HF_XET_CACHE="$HF_CACHE_ROOT/xet"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=1
export VLMINTUNE_FAST_EXIT=1
export TOKENIZERS_PARALLELISM=false

verify_fixed_caches() {
  "$VENV_DIR/bin/python" - \
    "$HF_HUB_CACHE" "$VIZWIZ_REVISION" "$SCIENCEQA_REVISION" <<'PY'
from pathlib import Path
import sys

import pyarrow.parquet as pq

hub, vizwiz_revision, scienceqa_revision = sys.argv[1:]
hub = Path(hub)


def require_snapshot(repo: str, revision: str, train_shards: int, eval_shards: int):
    snapshot = hub / repo / "snapshots" / revision
    if not snapshot.is_dir():
        raise SystemExit(f"missing fixed dataset snapshot: {snapshot}")
    train = sorted((snapshot / "data").glob("train-*.parquet"))
    validation = sorted((snapshot / "data").glob("validation-*.parquet"))
    if len(train) != train_shards or any(path.stat().st_size <= 0 for path in train):
        raise SystemExit(
            f"wrong/non-empty train shard count in {snapshot}: {len(train)} != {train_shards}"
        )
    if len(validation) != eval_shards or any(path.stat().st_size <= 0 for path in validation):
        raise SystemExit(
            f"wrong/non-empty validation shard count in {snapshot}: "
            f"{len(validation)} != {eval_shards}"
        )
    return snapshot, train, validation


_, vizwiz_train, vizwiz_validation = require_snapshot(
    "datasets--ebrukilic--vizwiz_vqa_dataset", vizwiz_revision, 19, 5
)
_, scienceqa_train, scienceqa_validation = require_snapshot(
    "datasets--derek-thomas--ScienceQA", scienceqa_revision, 1, 1
)


def parquet_rows(paths) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


vizwiz_counts = (parquet_rows(vizwiz_train), parquet_rows(vizwiz_validation))
if vizwiz_counts != (20523, 4319):
    raise SystemExit(f"VizWiz cached row counts are {vizwiz_counts}, expected (20523, 4319)")


def image_count(paths) -> int:
    # ScienceQA stores a path exactly when a row has an image. Reading only
    # this nested Parquet leaf avoids decoding or loading the image bytes.
    paths_column = pq.read_table(paths, columns=["image.path"]).column(0)
    return len(paths_column) - paths_column.null_count


scienceqa_counts = (image_count(scienceqa_train), image_count(scienceqa_validation))
if scienceqa_counts != (6218, 2097):
    raise SystemExit(
        f"ScienceQA image-only counts are {scienceqa_counts}, expected (6218, 2097)"
    )
print(
    "Fixed offline caches OK: VizWiz revision="
    f"{vizwiz_revision} shards=19/5 rows=20523/4319; ScienceQA revision="
    f"{scienceqa_revision} shards=1/1 image_rows=6218/2097"
)
PY
}

audit_textvqa_all() {
  local stage stage_samples eval_samples grad_acc method
  for stage in $TEXTVQA_STAGES; do
    stage_samples="${stage%%:*}"
    eval_samples="${stage##*:}"
    grad_acc=4
    if [[ "$stage_samples" -le 8 ]]; then
      grad_acc=1
    fi
    for method in $METHODS; do
      "$VENV_DIR/bin/python" scripts/audit_qwen_textvqa_combination_stage.py \
        --experiments-dir "$EXPERIMENTS_DIR" \
        --run-prefix qwen_textvqa_combo \
        --train-samples "$stage_samples" \
        --eval-samples "$eval_samples" \
        --grad-acc "$grad_acc" \
        --max-length "$MAX_LENGTH" \
        --epochs "$EPOCHS" \
        --seed "$SEED" \
        "$method"
    done
  done
}

wait_for_textvqa() {
  local upstream_status upstream_phase
  while true; do
    upstream_status="$(tr -d '[:space:]' < "$TEXTVQA_STATUS_FILE" 2>/dev/null || true)"
    upstream_phase="$(tr -d '\r\n' < "$TEXTVQA_PHASE_FILE" 2>/dev/null || true)"
    printf 'waiting_for_textvqa_joint_combinations\n' > "$PHASE_FILE"
    if [[ "$upstream_status" == "0" && "$upstream_phase" == "complete" ]]; then
      printf 'auditing_textvqa_joint_combinations\n' > "$PHASE_FILE"
      if audit_textvqa_all; then
        echo "All twelve TextVQA joint-combination stages passed strict audit."
        return 0
      fi
      echo "TextVQA reports complete but strict audit is not yet clean; waiting without modifying it."
    else
      echo "Waiting for TextVQA: status=${upstream_status:-missing} phase=${upstream_phase:-missing}"
    fi
    sleep "$WAIT_SECONDS"
  done
}

wait_for_gpu_idle() {
  while ps -eo args= | grep -Eq '[p]ython([^ ]*)? .*\-m vlmintune\.(training|eval)'; do
    printf 'waiting_for_gpu_idle\n' > "$PHASE_FILE"
    echo "A vlmintune training/evaluation process still owns the GPU; waiting."
    sleep "$WAIT_SECONDS"
  done
}

run_stage_until_success() {
  local dataset="$1"
  local dataset_slug="$2"
  local run_prefix="$3"
  local stage_samples="$4"
  local eval_samples="$5"
  local grad_acc=4 status
  if [[ "$stage_samples" -le 8 ]]; then
    grad_acc=1
  fi

  while true; do
    printf 'dataset_%s_train_%s_eval_%s\n' \
      "$dataset_slug" "$stage_samples" "$eval_samples" > "$PHASE_FILE"
    set +e
    DATASET="$dataset" \
    RUN_PREFIX="$run_prefix" \
    STAGE_SAMPLES="$stage_samples" \
    EVAL_SAMPLES="$eval_samples" \
    MAX_LENGTH="$MAX_LENGTH" \
    EPOCHS="$EPOCHS" \
    SEED="$SEED" \
    GRADIENT_ACCUMULATION_STEPS="$grad_acc" \
    LEARNING_RATE_DEFAULT="$LEARNING_RATE_DEFAULT" \
    EXPERIMENTS_DIR="$EXPERIMENTS_DIR" \
    GPU_LOCK_FILE="$GPU_LOCK_FILE" \
    VLMINTUNE_GPU_LOCK_HELD=1 \
    FORCE=0 \
    bash scripts/run_qwen_joint_combination_dataset_stage.sh
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      return 0
    fi
    printf 'blocked_%s_train_%s_eval_%s_status_%s\n' \
      "$dataset_slug" "$stage_samples" "$eval_samples" "$status" > "$PHASE_FILE"
    echo "Stage failed with status $status; artifacts and the root-cause log were preserved."
    echo "The queue remains alive in a blocked phase for the 30-minute monitor to diagnose; it will not blindly retry."
    while true; do
      sleep "$RETRY_SECONDS"
    done
  done
}

run_dataset() {
  local dataset="$1"
  local dataset_slug="$2"
  local run_prefix="$3"
  local stages="$4"
  local stage stage_samples eval_samples
  for stage in $stages; do
    stage_samples="${stage%%:*}"
    eval_samples="${stage##*:}"
    run_stage_until_success \
      "$dataset" "$dataset_slug" "$run_prefix" "$stage_samples" "$eval_samples"
  done
}

verify_fixed_caches
wait_for_textvqa

# The current TextVQA pipeline uses this same lock. Acquire it only after the
# upstream reports completion, then hold it across both remaining datasets.
printf 'acquiring_gpu_lock\n' > "$PHASE_FILE"
exec 9>>"$GPU_LOCK_FILE"
flock 9
audit_textvqa_all
wait_for_gpu_idle

run_dataset \
  ebrukilic/vizwiz_vqa_dataset vizwiz "$VIZWIZ_PREFIX" "$VIZWIZ_STAGES"
run_dataset \
  scienceqa_image scienceqa_image "$SCIENCEQA_PREFIX" "$SCIENCEQA_STAGES"

printf 'complete\n' > "$PHASE_FILE"
echo "TextVQA, VizWiz, and ScienceQA image-only joint combinations completed strictly."
