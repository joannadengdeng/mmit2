#!/usr/bin/env bash
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METHODS="${METHODS:-qlora lora dora freeze l2t mores}" \
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-0}" \
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}" \
MAX_LENGTH="${MAX_LENGTH:-1536}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}" \
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}" \
"$SETUP_DIR/run_llava_textvqa_train_eval.sh"
