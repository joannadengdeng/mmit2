#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

MODEL="${MODEL:-qwen25vl_3b_instruct}"
DATASET="${DATASET:-lmms-lab/textvqa}"
METHOD="${METHOD:-lora}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2.0e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

case "$METHOD" in
  lora|qlora|dora|mole|reft|mores|vl_adapter|l2t) ;;
  *)
    echo "Unsupported METHOD='$METHOD'. Expected one of: lora qlora dora mole reft mores vl_adapter l2t" >&2
    exit 2
    ;;
esac

case "$MODEL" in
  qwen25vl_3b_instruct|llava15_7b) ;;
  *)
    echo "Unsupported MODEL='$MODEL'. Expected qwen25vl_3b_instruct or llava15_7b." >&2
    exit 2
    ;;
esac

if [[ "$METHOD" == "mole" && "$MODEL" != "llava15_7b" ]]; then
  echo "METHOD=mole requires MODEL=llava15_7b." >&2
  exit 2
fi

if [[ "$METHOD" == "vl_adapter" && "$MODEL" != "qwen25vl_3b_instruct" ]]; then
  echo "METHOD=vl_adapter requires MODEL=qwen25vl_3b_instruct." >&2
  exit 2
fi

DATASET_SLUG="${DATASET##*/}"
DATASET_SLUG="${DATASET_SLUG//[^[:alnum:]_-]/_}"
RUN_NAME="${RUN_NAME:-${MODEL}_${METHOD}_${DATASET_SLUG}}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/${RUN_NAME}/checkpoint}"
CONFIG_PATH="${CONFIG_PATH:-${OUTPUT_DIR}/train_config.yaml}"

mkdir -p "$(dirname "$CONFIG_PATH")"

cat > "$CONFIG_PATH" <<YAML
model: "$MODEL"
dataset: "$DATASET"
method: "$METHOD"
epochs: $EPOCHS
learning_rate: $LEARNING_RATE
batch_size: $BATCH_SIZE
gradient_accumulation_steps: $GRADIENT_ACCUMULATION_STEPS
max_length: $MAX_LENGTH
max_samples: $MAX_SAMPLES
seed: $SEED
output_dir: "$OUTPUT_DIR"
YAML

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
python - "$CONFIG_PATH" <<'PY'
import sys

import yaml

from vlmintune.config.training_config import (
    PUBLIC_CONFIG_FIELDS,
    config_to_trainer_dict,
    load_config,
)

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as handle:
    raw_config = yaml.safe_load(handle) or {}
if set(raw_config) != PUBLIC_CONFIG_FIELDS:
    raise ValueError(
        "Runner must write exactly the public training fields: "
        f"expected={sorted(PUBLIC_CONFIG_FIELDS)}, actual={sorted(raw_config)}"
    )
config = config_to_trainer_dict(load_config(config_path))
print(f"Validated strict v1 config: {config_path}")
print(
    "Run: "
    f"model={config['model']} dataset={config['dataset']} method={config['method']} "
    f"max_samples={config['max_samples'] or 'full'} "
    f"output_dir={config['output_dir']}"
)
PY

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; configuration validated without loading a model or dataset."
  exit 0
fi

HF_TOKEN_ARGS=()
if [[ -n "${HF_TOKEN_FILE:-}" ]]; then
  HF_TOKEN_ARGS=(--hf-token-file "$HF_TOKEN_FILE")
elif [[ -s "$ROOT_DIR/.hf_token" ]]; then
  HF_TOKEN_ARGS=(--hf-token-file "$ROOT_DIR/.hf_token")
elif [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_ARGS=(--hf-token "$HF_TOKEN")
fi

python -m vlmintune.training "${HF_TOKEN_ARGS[@]}" --config "$CONFIG_PATH"
