#!/usr/bin/env bash
set -u

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_KEY="${1:-}"
MAX_SAMPLES="${2:-}"
MAX_LENGTH="${3:-}"

if [[ -z "$MODEL_KEY" || -z "$MAX_SAMPLES" || -z "$MAX_LENGTH" ]]; then
  echo "Usage: bash $0 <qwen|llava> <max_samples|0> <max_length>" >&2
  exit 2
fi

FAILED=0
for DATASET_KEY in textvqa vizwiz vqav2 gqa; do
  echo
  echo "### Running $MODEL_KEY $DATASET_KEY samples=$MAX_SAMPLES max_length=$MAX_LENGTH"
  bash "$SETUP_DIR/run_probe.sh" "$MODEL_KEY" "$DATASET_KEY" "$MAX_SAMPLES" "$MAX_LENGTH"
  STATUS=$?
  if [[ "$STATUS" -ne 0 ]]; then
    echo "### FAILED $MODEL_KEY $DATASET_KEY samples=$MAX_SAMPLES max_length=$MAX_LENGTH exit=$STATUS" >&2
    FAILED=1
  fi
done

exit "$FAILED"
