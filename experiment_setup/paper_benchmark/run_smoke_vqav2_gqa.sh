#!/usr/bin/env bash
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SETUP_DIR/../.." && pwd)"

cd "$ROOT_DIR"
mkdir -p run_logs

export RUN_STAMP="${RUN_STAMP:-vqav2_gqa_smoke_$(date +%Y%m%d_%H%M%S)}"
export MODELS="${MODELS:-qwen25vl_3b_instruct llava15_7b}"
export DATASETS="${DATASETS:-vqav2 gqa}"
export METHODS="${METHODS:-base qlora lora dora freeze l2t mole reft mores lora_layer}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-100}"
export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-30}"
export MAX_LENGTH="${MAX_LENGTH:-1536}"
export QWEN_VIZWIZ_MAX_LENGTH="${QWEN_VIZWIZ_MAX_LENGTH:-4096}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
export EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-20260611}"
export EVAL_SHUFFLE_BUFFER_SIZE="${EVAL_SHUFFLE_BUFFER_SIZE:-10000}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
export VLMINTUNE_FAST_EXIT="${VLMINTUNE_FAST_EXIT:-1}"
export REFT_LAYERS="${REFT_LAYERS:-16 24 31}"
export LORA_LAYER_TRAIN_LAYER_RANGE="${LORA_LAYER_TRAIN_LAYER_RANGE:-16:31}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-false}"
export DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"

echo "VQAv2/GQA smoke matrix:"
echo "  MODELS=$MODELS"
echo "  DATASETS=$DATASETS"
echo "  METHODS=$METHODS"
echo "  TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES"
echo "  EVAL_MAX_SAMPLES=$EVAL_MAX_SAMPLES"
echo "  RUN_STAMP=$RUN_STAMP"

bash "$SETUP_DIR/run_paper_benchmark.sh" \
  2>&1 | tee "run_logs/smoke_vqav2_gqa_${RUN_STAMP}.log"
