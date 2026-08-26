#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="pingzhili/vqa_v2"
DATASET_REVISION="${DATASET_REVISION:-f3a61102e2e569351e80ec2b9dca59792e5a0ef1}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$ROOT_DIR/experiments}"
STATUS_FILE="${STATUS_FILE:-$EXPERIMENTS_DIR/qwen_vqav2_pipeline_px1003520_s42.status}"
LOG_FILE="${LOG_FILE:-$EXPERIMENTS_DIR/qwen_vqav2_pipeline_px1003520_s42.log}"
PHASE_FILE="${PHASE_FILE:-$EXPERIMENTS_DIR/qwen_vqav2_pipeline_px1003520_s42.phase}"
VENV_DIR="${VENV_DIR:-/root/autodl-tmp/venvs/vlmintune-torch211}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/root/autodl-tmp/hf_cache}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
METHODS="${METHODS:-lora mores reft dora vl_adapter qlora l2t}"
STAGES="${STAGES:-8:8 256:32 1000:100 443757:214354}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
EPOCHS="${EPOCHS:-1}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-1}"
DOWNLOAD_ATTEMPTS="${DOWNLOAD_ATTEMPTS:-5}"
MIN_FREE_KB_AFTER_DOWNLOAD="${MIN_FREE_KB_AFTER_DOWNLOAD:-52428800}"

mkdir -p "$EXPERIMENTS_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

printf 'RUNNING\n' > "$STATUS_FILE"
printf 'initializing\n' > "$PHASE_FILE"

record_status() {
  local status="$1"
  printf '%s\n' "$status" > "$STATUS_FILE"
}
record_exit() {
  local status=$?
  record_status "$status"
}
record_signal() {
  local status="$1"
  trap - EXIT
  record_status "$status"
  exit "$status"
}
trap record_exit EXIT
trap 'record_signal 130' INT
trap 'record_signal 143' TERM

if [[ ! -x "$VENV_DIR/bin/python" || ! -x "$VENV_DIR/bin/hf" ]]; then
  echo "Missing experiment Python or hf CLI in $VENV_DIR/bin" >&2
  exit 2
fi
if [[ ! "$DOWNLOAD_WORKERS" =~ ^[1-9][0-9]*$ \
      || ! "$DOWNLOAD_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DOWNLOAD_WORKERS and DOWNLOAD_ATTEMPTS must be positive integers." >&2
  exit 2
fi

export PATH="$VENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
export HF_XET_CACHE="$HF_CACHE_ROOT/xet"
export HF_ENDPOINT
# The configured mirror can cache anonymous Xet read tokens beyond their
# expiry.  Its regular resolve endpoint still returns fresh signed HTTP URLs,
# so use one HTTP transfer at a time and let completed shards accumulate in
# the Hub cache across snapshot-download retries.
export HF_HUB_DISABLE_XET=1
unset HF_XET_HIGH_PERFORMANCE
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export TOKENIZERS_PARALLELISM=false

model_cache="$HF_HUB_CACHE/models--Qwen--Qwen2.5-VL-3B-Instruct"
if [[ ! -e "$model_cache" ]]; then
  echo "Qwen cache is not visible at $model_cache" >&2
  exit 2
fi

repo_cache="$HF_HUB_CACHE/datasets--pingzhili--vqa_v2"
revision="$DATASET_REVISION"
snapshot="$repo_cache/snapshots/$revision"

count_shards() {
  local pattern="$1"
  find -L "$snapshot/data" -maxdepth 1 -type f -name "$pattern" -size +0c | wc -l | tr -d '[:space:]'
}

printf 'checking_vqav2_cache\n' > "$PHASE_FILE"
cache_ready=0
if [[ -d "$snapshot/data" && -f "$snapshot/README.md" ]]; then
  cached_train_shards="$(count_shards 'train-*.parquet')"
  cached_validation_shards="$(count_shards 'validation-*.parquet')"
  if [[ "$cached_train_shards" == "68" && "$cached_validation_shards" == "34" ]]; then
    cache_ready=1
  fi
fi

if [[ "$cache_ready" == "1" ]]; then
  echo "VQAv2 required cache already present; skipping network download."
else
  printf 'downloading_vqav2\n' > "$PHASE_FILE"
  export HF_HUB_OFFLINE=0
  download_ok=0
  echo "VQAv2 download plan: revision=$DATASET_REVISION workers=$DOWNLOAD_WORKERS splits=train,validation transport=http"
  for ((attempt = 1; attempt <= DOWNLOAD_ATTEMPTS; attempt++)); do
    echo "VQAv2 cache download attempt $attempt/$DOWNLOAD_ATTEMPTS"
    if hf download "$DATASET" \
        --repo-type dataset \
        --revision "$DATASET_REVISION" \
        --cache-dir "$HF_HUB_CACHE" \
        --max-workers "$DOWNLOAD_WORKERS" \
        --exclude 'data/test-*' \
        --quiet; then
      download_ok=1
      break
    fi
    if (( attempt < DOWNLOAD_ATTEMPTS )); then
      echo "Download attempt $attempt failed; retrying in 30 seconds." >&2
      sleep 30
    fi
  done
  if [[ "$download_ok" != "1" ]]; then
    echo "Unable to cache VQAv2 after $DOWNLOAD_ATTEMPTS attempts." >&2
    exit 3
  fi
fi

if [[ ! -d "$snapshot/data" || ! -f "$snapshot/README.md" ]]; then
  echo "VQAv2 snapshot is incomplete: $snapshot" >&2
  exit 3
fi

train_shards="$(count_shards 'train-*.parquet')"
validation_shards="$(count_shards 'validation-*.parquet')"
test_shards="$(count_shards 'test-*.parquet')"
if [[ "$train_shards" != "68" \
      || "$validation_shards" != "34" ]]; then
  echo "Wrong VQAv2 shard counts: train=$train_shards validation=$validation_shards test=$test_shards" >&2
  exit 3
fi
echo "VQAv2 cache verified: revision=$revision train=68 validation=34 test_not_required=$test_shards"

"$VENV_DIR/bin/python" - "$snapshot" <<'PY'
from pathlib import Path
import sys

import pyarrow.parquet as pq

snapshot = Path(sys.argv[1])
expected = {"train": (68, 443757), "validation": (34, 214354)}
for split, (expected_files, expected_rows) in expected.items():
    files = sorted((snapshot / "data").glob(f"{split}-*.parquet"))
    if len(files) != expected_files:
        raise SystemExit(
            f"VQAv2 {split} expected {expected_files} shards, found {len(files)}"
        )
    rows = 0
    for path in files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise SystemExit(f"VQAv2 shard is missing or empty: {path}")
        rows += pq.ParquetFile(path).metadata.num_rows
    if rows != expected_rows:
        raise SystemExit(
            f"VQAv2 {split} expected {expected_rows} rows, found {rows}"
        )
    print(f"VQAv2 Parquet verified: split={split} files={len(files)} rows={rows}")
PY

available_kb="$(df -Pk /root/autodl-tmp | awk 'NR == 2 {print $4}')"
if (( available_kb < MIN_FREE_KB_AFTER_DOWNLOAD )); then
  echo "Insufficient free disk after download: ${available_kb} KiB; require ${MIN_FREE_KB_AFTER_DOWNLOAD} KiB" >&2
  exit 4
fi
echo "Disk guard passed: available_kb=$available_kb"

export VLMINTUNE_VQAV2_SNAPSHOT="$snapshot"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
for stage in $STAGES; do
  stage_samples="${stage%%:*}"
  eval_samples="${stage##*:}"
  if (( stage_samples <= 8 )); then
    grad_acc=1
  else
    grad_acc=4
  fi

  printf 'stage_train_%s_eval_%s\n' "$stage_samples" "$eval_samples" > "$PHASE_FILE"
  echo "================================================================================"
  echo "QWEN VQAV2 STAGE train=$stage_samples eval=$eval_samples methods=$METHODS"
  echo "================================================================================"
  DATASET="$DATASET" \
  STAGE_SAMPLES="$stage_samples" \
  EVAL_SAMPLES="$eval_samples" \
  GRADIENT_ACCUMULATION_STEPS="$grad_acc" \
  MAX_LENGTH="$MAX_LENGTH" \
  EPOCHS="$EPOCHS" \
  SEED="$SEED" \
  METHODS="$METHODS" \
  RUN_PREFIX=qwen_vqav2_px1003520 \
  bash scripts/run_qwen_dataset_stage.sh
done

printf 'complete\n' > "$PHASE_FILE"
echo "Qwen VQAv2 pipeline completed successfully."
