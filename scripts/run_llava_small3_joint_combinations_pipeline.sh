#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$ROOT_DIR/experiments}"
mkdir -p "$EXPERIMENTS_DIR"
EXPERIMENTS_DIR="$(cd "$EXPERIMENTS_DIR" && pwd -P)"

CONTROL_STEM="llava_small3_joint_combinations_s42"
STATUS_FILE="${STATUS_FILE:-$EXPERIMENTS_DIR/${CONTROL_STEM}.status}"
PHASE_FILE="${PHASE_FILE:-$EXPERIMENTS_DIR/${CONTROL_STEM}.phase}"
LOG_FILE="${LOG_FILE:-$EXPERIMENTS_DIR/${CONTROL_STEM}.log}"
SELF_LOCK_FILE="${SELF_LOCK_FILE:-$EXPERIMENTS_DIR/${CONTROL_STEM}.lock}"
GPU_LOCK_FILE="${GPU_LOCK_FILE:-$EXPERIMENTS_DIR/qwen_textvqa_combinations.lock}"

VENV_DIR="${VENV_DIR:-/root/autodl-tmp/venvs/vlmintune-torch211}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/root/autodl-tmp/hf_cache}"
VQ_STATUS_FILE="${VQ_STATUS_FILE:-/root/autodl-tmp/vlmintune_codex/experiments/qwen_vqav2_pipeline_px1003520_s42.status}"
VQ_PHASE_FILE="${VQ_PHASE_FILE:-/root/autodl-tmp/vlmintune_codex/experiments/qwen_vqav2_pipeline_px1003520_s42.phase}"

MODEL="llava15_7b"
METHODS="mores_lora mores_dora reft_lora"
TEXTVQA_STAGES="8:8 256:32 1000:100 34602:5000"
VIZWIZ_STAGES="8:8 256:32 1000:100 20523:4319"
SCIENCEQA_STAGES="8:8 256:32 1000:100 6218:2097"
TEXTVQA_PREFIX="llava_textvqa_joint3_progressive20260822"
VIZWIZ_PREFIX="llava_vizwiz_joint3_progressive20260822"
SCIENCEQA_PREFIX="llava_scienceqa_image_joint3_progressive20260822"
MAX_LENGTH=1536
EPOCHS=1
SEED=42
LEARNING_RATE_DEFAULT=2e-4

TEXTVQA_REVISION="9c0699cd19768ac5ab97568f6b3cbac4c0062884"
VIZWIZ_REVISION="8458ff83feb8d782b53b11b391cf1dedd961922e"
SCIENCEQA_REVISION="f18b0a70359ebfb41f658fd564208d0355b013f4"
LLAVA_REVISION="b234b804b114d9e37bb655e11cbbb5f5e971b7a9"

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
  local phase=""
  printf '%s\n' "$status" > "$STATUS_FILE"
  if [[ "$status" -ne 0 ]]; then
    phase="$(tr -d '\r\n' < "$PHASE_FILE" 2>/dev/null || true)"
    if [[ "$phase" != blocked_* ]]; then
      printf 'failed_status_%s\n' "$status" > "$PHASE_FILE"
    fi
  fi
}
trap record_exit EXIT

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing experiment Python: $VENV_DIR/bin/python" >&2
  exit 2
fi

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

verify_vqav2_paused() {
  local status phase
  status="$(tr -d '\r\n' < "$VQ_STATUS_FILE" 2>/dev/null || true)"
  phase="$(tr -d '\r\n' < "$VQ_PHASE_FILE" 2>/dev/null || true)"
  if [[ "$status" != "PAUSED" \
        || "$phase" != "paused_by_user_dora_step_11788_of_110940" ]]; then
    echo "VQAv2 pause guard failed: status=${status:-missing} phase=${phase:-missing}" >&2
    exit 7
  fi
  echo "VQAv2 remains manually paused: $phase"
}

verify_idle_gpu_processes() {
  local conflicts
  conflicts="$(ps -eo pid=,args= | grep -E '[p]ython([^ ]*)? .*\-m vlmintune\.(training|eval)' || true)"
  if [[ -n "$conflicts" ]]; then
    echo "Refusing to start while another vlmintune training/evaluation process exists:" >&2
    echo "$conflicts" >&2
    exit 5
  fi
}

verify_fixed_caches() {
  "$VENV_DIR/bin/python" - \
    "$HF_HUB_CACHE" "$TEXTVQA_REVISION" "$VIZWIZ_REVISION" \
    "$SCIENCEQA_REVISION" "$LLAVA_REVISION" <<'PY'
from pathlib import Path
import sys

import pyarrow.parquet as pq

(
    hub,
    textvqa_revision,
    vizwiz_revision,
    scienceqa_revision,
    llava_revision,
) = sys.argv[1:]
hub = Path(hub)


def require_snapshot(repo: str, revision: str, train_shards: int, eval_shards: int):
    repo_dir = hub / repo
    ref = (repo_dir / "refs" / "main").read_text(encoding="utf-8").strip()
    if ref != revision:
        raise SystemExit(f"wrong cached revision for {repo}: {ref} != {revision}")
    snapshot = repo_dir / "snapshots" / revision
    if not snapshot.is_dir():
        raise SystemExit(f"missing fixed snapshot: {snapshot}")
    broken = [path for path in snapshot.rglob("*") if path.is_symlink() and not path.exists()]
    if broken:
        raise SystemExit(f"broken cache links in {snapshot}: {broken[:3]}")
    train = sorted((snapshot / "data").glob("train-*.parquet"))
    validation = sorted((snapshot / "data").glob("validation-*.parquet"))
    if len(train) != train_shards or any(path.stat().st_size <= 0 for path in train):
        raise SystemExit(f"wrong/non-empty train shard count in {snapshot}: {len(train)}")
    if len(validation) != eval_shards or any(path.stat().st_size <= 0 for path in validation):
        raise SystemExit(f"wrong/non-empty validation shard count in {snapshot}: {len(validation)}")
    return train, validation


def parquet_rows(paths) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


text_train, text_eval = require_snapshot(
    "datasets--lmms-lab--textvqa", textvqa_revision, 20, 3
)
viz_train, viz_eval = require_snapshot(
    "datasets--ebrukilic--vizwiz_vqa_dataset", vizwiz_revision, 19, 5
)
science_train, science_eval = require_snapshot(
    "datasets--derek-thomas--ScienceQA", scienceqa_revision, 1, 1
)
if (parquet_rows(text_train), parquet_rows(text_eval)) != (34602, 5000):
    raise SystemExit("TextVQA cached row counts do not match 34602/5000")
if (parquet_rows(viz_train), parquet_rows(viz_eval)) != (20523, 4319):
    raise SystemExit("VizWiz cached row counts do not match 20523/4319")


def image_count(paths) -> int:
    column = pq.read_table(paths, columns=["image.path"]).column(0)
    return len(column) - column.null_count


if (image_count(science_train), image_count(science_eval)) != (6218, 2097):
    raise SystemExit("ScienceQA image-only counts do not match 6218/2097")

model_repo = hub / "models--llava-hf--llava-1.5-7b-hf"
model_ref = (model_repo / "refs" / "main").read_text(encoding="utf-8").strip()
model_snapshot = model_repo / "snapshots" / llava_revision
if model_ref != llava_revision or not model_snapshot.is_dir():
    raise SystemExit(
        f"wrong/missing LLaVA snapshot: ref={model_ref} expected={llava_revision}"
    )
broken = [path for path in model_snapshot.rglob("*") if path.is_symlink() and not path.exists()]
if broken:
    raise SystemExit(f"broken LLaVA cache links: {broken[:3]}")
if not any(path.name.startswith("model-") and path.suffix == ".safetensors" for path in model_snapshot.iterdir()):
    raise SystemExit("LLaVA snapshot has no model safetensors shards")

print(
    "Fixed offline caches OK: LLaVA-1.5-7B; TextVQA=34602/5000; "
    "VizWiz=20523/4319; ScienceQA-image=6218/2097"
)
PY

  local available_bytes
  available_bytes="$(df --output=avail -B1 /root/autodl-tmp | tail -n 1 | tr -d '[:space:]')"
  if [[ ! "$available_bytes" =~ ^[0-9]+$ || "$available_bytes" -lt 85899345920 ]]; then
    echo "Less than 80 GiB is available on /root/autodl-tmp; refusing to start." >&2
    exit 8
  fi
  echo "Disk preflight OK: available_bytes=$available_bytes"
}

run_stage() {
  local dataset="$1"
  local dataset_slug="$2"
  local run_prefix="$3"
  local train_samples="$4"
  local eval_samples="$5"
  local grad_acc=4
  if [[ "$train_samples" -le 8 ]]; then
    grad_acc=1
  fi

  printf 'dataset_%s_train_%s_eval_%s\n' \
    "$dataset_slug" "$train_samples" "$eval_samples" > "$PHASE_FILE"
  set +e
  MODEL="$MODEL" \
      DATASET="$dataset" \
      RUN_PREFIX="$run_prefix" \
      STAGE_SAMPLES="$train_samples" \
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
  local stage_status=$?
  set -e
  if [[ "$stage_status" -ne 0 ]]; then
    printf 'blocked_%s_train_%s_eval_%s_status_%s\n' \
      "$dataset_slug" "$train_samples" "$eval_samples" "$stage_status" > "$PHASE_FILE"
    return "$stage_status"
  fi
}

run_dataset() {
  local dataset="$1"
  local dataset_slug="$2"
  local run_prefix="$3"
  local stages="$4"
  local stage train_samples eval_samples
  for stage in $stages; do
    verify_vqav2_paused
    train_samples="${stage%%:*}"
    eval_samples="${stage##*:}"
    run_stage "$dataset" "$dataset_slug" "$run_prefix" "$train_samples" "$eval_samples"
  done
}

verify_vqav2_paused
verify_idle_gpu_processes
verify_fixed_caches

printf 'acquiring_gpu_lock\n' > "$PHASE_FILE"
exec 9>>"$GPU_LOCK_FILE"
if ! flock -n 9; then
  echo "Another experiment pipeline holds $GPU_LOCK_FILE" >&2
  exit 5
fi

run_dataset \
  "lmms-lab/textvqa" "textvqa" "$TEXTVQA_PREFIX" "$TEXTVQA_STAGES"
run_dataset \
  "ebrukilic/vizwiz_vqa_dataset" "vizwiz" "$VIZWIZ_PREFIX" "$VIZWIZ_STAGES"
run_dataset \
  "scienceqa_image" "scienceqa_image" "$SCIENCEQA_PREFIX" "$SCIENCEQA_STAGES"

verify_vqav2_paused
printf 'complete\n' > "$PHASE_FILE"
echo "LLaVA three-dataset joint-combination pipeline completed successfully."
