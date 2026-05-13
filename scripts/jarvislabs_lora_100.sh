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
#   DRY_RUN=1 ./scripts/jarvislabs_lora_100.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

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
DEFAULT_EXPERIMENT_NAME="jarvislabs_lora_100_$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
DRY_RUN="${DRY_RUN:-0}"

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
echo

python -m mmit2.training --config-json "$CONFIG_JSON"
