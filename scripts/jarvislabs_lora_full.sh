#!/usr/bin/env bash
set -euo pipefail

# Full-dataset LoRA training wrapper for JarvisLabs.ai.
#
# This keeps the implementation tiny by reusing the smoke-training script and
# only overriding the defaults that define a full training run.
#
# Defaults:
#   - model: Qwen/Qwen2.5-VL-3B-Instruct
#   - dataset: lmms-lab/textvqa
#   - split: train
#   - samples: full split (MAX_SAMPLES=0)
#
# Common overrides:
#   DATASET_NAME=lmms-lab/VQAv2 ./scripts/jarvislabs_lora_full.sh
#   NUM_EPOCHS=2 ./scripts/jarvislabs_lora_full.sh
#   ./scripts/jarvislabs_lora_full.sh --hf-token-file /root/.hf_token

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

export MAX_SAMPLES="${MAX_SAMPLES:-0}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-jarvislabs_lora_full_$(date +%Y%m%d_%H%M%S)}"

exec "$ROOT_DIR/scripts/jarvislabs_lora_100.sh" "$@"
