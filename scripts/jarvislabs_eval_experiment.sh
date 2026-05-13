#!/usr/bin/env bash
set -euo pipefail

# Evaluate a previously trained mmit2 experiment on JarvisLabs.
#
# Intended usage:
#   1. SSH into the JarvisLabs machine after training.
#   2. Run this script from the repo root or anywhere.
#
# Defaults:
#   - experiment: latest directory under ./experiments
#   - eval dataset: lmms-lab/textvqa
#   - split: dataset default
#   - samples: 100
#
# Common overrides:
#   EXPERIMENT_NAME=jarvislabs_lora_100_20260513_123456 ./scripts/jarvislabs_eval_experiment.sh
#   EVAL_DATASET_NAME=lmms-lab/VQAv2 EVAL_MAX_SAMPLES=200 ./scripts/jarvislabs_eval_experiment.sh
#   ./scripts/jarvislabs_eval_experiment.sh --hf-token-file /root/.hf_token
#   ./scripts/jarvislabs_eval_experiment.sh --hf-token hf_xxx
#   DRY_RUN=1 ./scripts/jarvislabs_eval_experiment.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
EXPERIMENT_BASE_DIR="${EXPERIMENT_BASE_DIR:-$ROOT_DIR/experiments}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-lmms-lab/textvqa}"
EVAL_SPLIT="${EVAL_SPLIT:-}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
EVAL_NAME="${EVAL_NAME:-}"
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

if [[ -z "$EXPERIMENT_NAME" ]]; then
  if [[ ! -d "$EXPERIMENT_BASE_DIR" ]]; then
    echo "[mmit2] No experiments directory found at $EXPERIMENT_BASE_DIR" >&2
    exit 1
  fi
  EXPERIMENT_NAME="$(basename "$(ls -td "$EXPERIMENT_BASE_DIR"/* 2>/dev/null | head -n 1)")"
fi

if [[ -z "$EXPERIMENT_NAME" ]]; then
  echo "[mmit2] Could not determine experiment name. Set EXPERIMENT_NAME explicitly." >&2
  exit 1
fi

SUMMARY_PATH="$EXPERIMENT_BASE_DIR/$EXPERIMENT_NAME/summary.json"
if [[ ! -f "$SUMMARY_PATH" ]]; then
  echo "[mmit2] Experiment summary not found: $SUMMARY_PATH" >&2
  exit 1
fi

export EXPERIMENT_BASE_DIR
export EXPERIMENT_NAME
export EVAL_DATASET_NAME
export EVAL_SPLIT
export EVAL_MAX_SAMPLES
export MAX_NEW_TOKENS
export TEMPERATURE
export EVAL_NAME

CONFIG_JSON="$("$PYTHON_BIN" - <<PY
import json
import os

config = {
    "experiment": {
        "name": os.environ["EXPERIMENT_NAME"],
        "base_dir": os.environ["EXPERIMENT_BASE_DIR"],
    },
    "eval": {
        "dataset_name": os.environ["EVAL_DATASET_NAME"],
        "max_samples": int(os.environ["EVAL_MAX_SAMPLES"]),
        "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
        "temperature": float(os.environ["TEMPERATURE"]),
    },
}

eval_split = os.environ.get("EVAL_SPLIT", "").strip()
if eval_split:
    config["eval"]["split"] = eval_split

eval_name = os.environ.get("EVAL_NAME", "").strip()
if eval_name:
    config["eval"]["name"] = eval_name

print(json.dumps(config, ensure_ascii=False))
PY
)"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[mmit2] Experiment summary: $SUMMARY_PATH"
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

echo "[mmit2] Starting JarvisLabs eval run"
echo "[mmit2] Experiment: $EXPERIMENT_NAME"
echo "[mmit2] Summary: $SUMMARY_PATH"
if [[ -n "$EVAL_SPLIT" ]]; then
  echo "[mmit2] Eval dataset: $EVAL_DATASET_NAME ($EVAL_SPLIT)"
else
  echo "[mmit2] Eval dataset: $EVAL_DATASET_NAME (dataset default split)"
fi
echo "[mmit2] Eval samples: $EVAL_MAX_SAMPLES"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[mmit2] HF token: enabled"
fi
echo

python -m mmit2.eval --config-json "$CONFIG_JSON"
