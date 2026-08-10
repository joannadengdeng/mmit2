#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS="${DATASETS:-scienceqa_image ebrukilic/vizwiz_vqa_dataset pingzhili/vqa_v2 Mineru/GQA}"
STAGE_SAMPLES="${STAGE_SAMPLES:-8}"
EVAL_SAMPLES="${EVAL_SAMPLES:-8}"
SEED="${SEED:-42}"
METHODS="${METHODS:-lora mores reft dora vl_adapter qlora l2t}"

read -r -a dataset_list <<< "$DATASETS"
for dataset in "${dataset_list[@]}"; do
  DATASET="$dataset" \
  STAGE_SAMPLES="$STAGE_SAMPLES" \
  EVAL_SAMPLES="$EVAL_SAMPLES" \
  SEED="$SEED" \
  METHODS="$METHODS" \
  bash scripts/run_qwen_dataset_stage.sh
done

echo "All requested non-TextVQA Qwen smoke stages passed."
