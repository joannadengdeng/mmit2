#!/usr/bin/env bash
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SETUP_DIR/../.." && pwd)"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT_DIR"
mkdir -p run_logs

export RUN_STAMP
export MODELS="${MODELS:-qwen25vl_3b_instruct llava15_7b}"
export DATASETS="${DATASETS:-textvqa vqav2 vizwiz gqa scienceqa}"
export METHODS="${METHODS:-base qlora lora dora freeze l2t mole reft mores lora_layer}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-8}"
export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-8}"
export MAX_LENGTH="${MAX_LENGTH:-1536}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
export VLMINTUNE_FAST_EXIT="${VLMINTUNE_FAST_EXIT:-1}"
export REFT_LAYERS="${REFT_LAYERS:-0}"

echo "Smoke matrix:"
echo "  MODELS=$MODELS"
echo "  DATASETS=$DATASETS"
echo "  METHODS=$METHODS"
echo "  TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES"
echo "  EVAL_MAX_SAMPLES=$EVAL_MAX_SAMPLES"
echo "  REFT_LAYERS=$REFT_LAYERS"
if [[ -n "${VISNEC_SCORE_FILE:-}" ]]; then
  echo "  VISNEC_SCORE_FILE=$VISNEC_SCORE_FILE"
  echo "  VISNEC_TOP_RATIO=${VISNEC_TOP_RATIO:-1.0}"
fi
echo "  RUN_STAMP=$RUN_STAMP"

bash "$SETUP_DIR/run_paper_benchmark.sh" \
  2>&1 | tee "run_logs/smoke_all_${RUN_STAMP}.log"
