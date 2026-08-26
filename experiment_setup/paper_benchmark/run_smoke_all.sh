#!/usr/bin/env bash
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vlmintune-config-smoke.XXXXXX")"
trap 'rm -rf "$SMOKE_DIR"' EXIT

DATASET="${DATASET:-lmms-lab/textvqa}"
GENERAL_MODEL="${MODEL:-qwen25vl_3b_instruct}"

for method in \
  lora qlora dora reft mores vl_adapter l2t \
  mores_lora mores_dora reft_lora; do
  model="$GENERAL_MODEL"
  if [[ "$method" == "vl_adapter" ]]; then
    model="qwen25vl_3b_instruct"
  fi

  echo "Validating method=$method model=$model dataset=$DATASET"
  MODEL="$model" \
  DATASET="$DATASET" \
  METHOD="$method" \
  RUN_NAME="smoke_${method}" \
  OUTPUT_DIR="$SMOKE_DIR/$method/checkpoint" \
  CONFIG_PATH="$SMOKE_DIR/$method/train_config.yaml" \
  DRY_RUN=1 \
    bash "$SETUP_DIR/run_paper_benchmark.sh"
done

echo "Strict configuration smoke passed for all ten release recipes."
