#!/usr/bin/env bash
set -euo pipefail

# LoRA smoke-training script for JarvisLabs.ai.
#
# Official JarvisLabs flows this script is designed for:
#   1. SSH into an instance, then run this script directly.
#   2. Use `jl run . --script scripts/jarvislabs_lora_100.sh --gpu A100 --keep`.
#
# Defaults:
#   - model: Qwen/Qwen2.5-VL-3B-Instruct
#   - dataset: lmms-lab/textvqa
#   - samples: 100
#   - method: LoRA
#
# Common overrides:
#   MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct MAX_SAMPLES=100 ./scripts/jarvislabs_lora_100.sh
#   EXPERIMENT_NAME=my_lora_smoke NUM_EPOCHS=2 ./scripts/jarvislabs_lora_100.sh
#   ./scripts/jarvislabs_lora_100.sh --hf-token-file /root/.hf_token
#   ./scripts/jarvislabs_lora_100.sh --hf-token hf_xxx
#   DRY_RUN=1 ./scripts/jarvislabs_lora_100.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

normalize_slug() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-+/-/g'
}

dataset_slug() {
  local raw="${DATASET_NAME##*/}"
  local slug
  slug="$(normalize_slug "$raw")"
  case "$slug" in
    vizwiz-vqa)
      echo "vizwiz"
      ;;
    *)
      echo "$slug"
      ;;
  esac
}

model_size_slug() {
  local base="${MODEL_PATH##*/}"
  local token

  for token in ${base//[-_\/]/ }; do
    if [[ "$token" =~ ^[0-9]+([.][0-9]+)?[BbMm]$ ]]; then
      printf '%s\n' "$token" | tr '[:upper:]' '[:lower:]'
      return
    fi
  done

  if [[ "$base" =~ ([0-9]+([.][0-9]+)?[BbMm]) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}" | tr '[:upper:]' '[:lower:]'
    return
  fi

  echo "model"
}

sample_scope_slug() {
  if [[ -z "${MAX_SAMPLES:-}" || "${MAX_SAMPLES}" == "0" ]]; then
    echo "full"
  else
    echo "$(normalize_slug "${MAX_SAMPLES}samples")"
  fi
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATASET_NAME="${DATASET_NAME:-lmms-lab/textvqa}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SAVE_STEPS="${SAVE_STEPS:-0}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
EXPERIMENT_BASE_DIR="${EXPERIMENT_BASE_DIR:-$ROOT_DIR/experiments}"
DEFAULT_EXPERIMENT_NAME="$(date +%Y%m%d)_lora_$(dataset_slug)_$(model_size_slug)_$(sample_scope_slug)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
DRY_RUN="${DRY_RUN:-0}"
HF_TOKEN_VALUE="${HF_TOKEN:-}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-token)
      if [[ $# -lt 2 ]]; then
        echo "[mmit2] --hf-token requires a value" >&2
        exit 1
      fi
      HF_TOKEN_VALUE="$2"
      shift 2
      ;;
    --hf-token-file)
      if [[ $# -lt 2 ]]; then
        echo "[mmit2] --hf-token-file requires a path" >&2
        exit 1
      fi
      HF_TOKEN_FILE="$2"
      shift 2
      ;;
    *)
      echo "[mmit2] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$HF_TOKEN_VALUE" && -n "$HF_TOKEN_FILE" ]]; then
  if [[ ! -f "$HF_TOKEN_FILE" ]]; then
    echo "[mmit2] HF token file not found: $HF_TOKEN_FILE" >&2
    exit 1
  fi
  HF_TOKEN_VALUE="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
fi

if [[ -n "$HF_TOKEN_VALUE" ]]; then
  export HF_TOKEN="$HF_TOKEN_VALUE"
fi

export MODEL_PATH
export DATASET_NAME
export TRAIN_SPLIT
export MAX_SAMPLES
export NUM_EPOCHS
export PER_DEVICE_BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS
export LEARNING_RATE
export WARMUP_RATIO
export WEIGHT_DECAY
export MAX_GRAD_NORM
export SAVE_STEPS
export LORA_R
export LORA_ALPHA
export LORA_DROPOUT
export TARGET_MODULES
export EXPERIMENT_BASE_DIR
export EXPERIMENT_NAME

CONFIG_JSON="$("$PYTHON_BIN" - <<PY
import json
import os

target_modules = [
    item.strip()
    for item in os.environ["TARGET_MODULES"].split(",")
    if item.strip()
]

config = {
    "model": {
        "model_path": os.environ["MODEL_PATH"],
    },
    "experiment": {
        "name": os.environ["EXPERIMENT_NAME"],
        "base_dir": os.environ["EXPERIMENT_BASE_DIR"],
    },
    "data": {
        "adapter": "hf_datasets",
        "data_path": os.environ["DATASET_NAME"],
        "split": os.environ["TRAIN_SPLIT"],
        "max_samples": int(os.environ["MAX_SAMPLES"]),
    },
    "training_method": "lora",
    "method_params": {
        "lora_r": int(os.environ["LORA_R"]),
        "lora_alpha": int(os.environ["LORA_ALPHA"]),
        "lora_dropout": float(os.environ["LORA_DROPOUT"]),
        "target_modules": target_modules,
    },
    "training": {
        "num_epochs": int(os.environ["NUM_EPOCHS"]),
        "per_device_batch_size": int(os.environ["PER_DEVICE_BATCH_SIZE"]),
        "gradient_accumulation_steps": int(os.environ["GRADIENT_ACCUMULATION_STEPS"]),
        "learning_rate": float(os.environ["LEARNING_RATE"]),
        "warmup_ratio": float(os.environ["WARMUP_RATIO"]),
        "weight_decay": float(os.environ["WEIGHT_DECAY"]),
        "max_grad_norm": float(os.environ["MAX_GRAD_NORM"]),
        "save_steps": int(os.environ["SAVE_STEPS"]),
        "output_dir": os.environ["EXPERIMENT_BASE_DIR"],
    },
}

print(json.dumps(config, ensure_ascii=False))
PY
)"

if [[ "$DRY_RUN" == "1" ]]; then
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[mmit2] HF token: enabled"
  else
    echo "[mmit2] HF token: not set"
  fi
  printf '%s\n' "$CONFIG_JSON" | "$PYTHON_BIN" -m json.tool
  exit 0
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if ! python -m pip --version >/dev/null 2>&1; then
  python -m ensurepip --upgrade
fi

python -m pip install --upgrade pip
if [[ "$SKIP_INSTALL" != "1" ]]; then
  python -m pip install -e ".[finetune]"
fi

export PYTHONUNBUFFERED=1

echo "[mmit2] Starting JarvisLabs LoRA run"
echo "[mmit2] Model: $MODEL_PATH"
echo "[mmit2] Dataset: $DATASET_NAME ($TRAIN_SPLIT)"
echo "[mmit2] Samples: $MAX_SAMPLES"
echo "[mmit2] Experiment: $EXPERIMENT_NAME"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[mmit2] HF token: enabled"
fi
echo

python -m mmit2.training --config-json "$CONFIG_JSON"
