#!/usr/bin/env bash
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default: a small all-dataset check for RTX 5090 planning.
# Override these env vars when you want a wider or longer probe:
#   SAMPLE_SIZES="100" bash .../run_5090_check.sh
#   DATASETS="textvqa" SAMPLE_SIZES="100 1000" bash .../run_5090_check.sh
#   SAMPLE_SIZES="50" MAX_LENGTHS="1024 1536" bash .../run_5090_check.sh
MODELS="${MODELS:-qwen llava}"
DATASETS="${DATASETS:-textvqa vizwiz vqav2 gqa}"
SAMPLE_SIZES="${SAMPLE_SIZES:-50}"
MAX_LENGTHS="${MAX_LENGTHS:-1536}"

echo "=== MoReS RTX 5090 fit check ==="
echo "models=$MODELS"
echo "datasets=$DATASETS"
echo "sample_sizes=$SAMPLE_SIZES"
echo "max_lengths=$MAX_LENGTHS"
echo

FAILED=0
for MODEL_KEY in $MODELS; do
  for DATASET_KEY in $DATASETS; do
    for MAX_SAMPLES in $SAMPLE_SIZES; do
      for MAX_LENGTH in $MAX_LENGTHS; do
        echo
        echo "### Running model=$MODEL_KEY dataset=$DATASET_KEY samples=$MAX_SAMPLES max_length=$MAX_LENGTH"
        set +e
        bash "$SETUP_DIR/run_probe.sh" "$MODEL_KEY" "$DATASET_KEY" "$MAX_SAMPLES" "$MAX_LENGTH"
        STATUS=$?
        set -e
        if [[ "$STATUS" -ne 0 ]]; then
          echo "### FAILED model=$MODEL_KEY dataset=$DATASET_KEY samples=$MAX_SAMPLES max_length=$MAX_LENGTH exit=$STATUS" >&2
          FAILED=1
        fi
      done
    done
  done
done

echo
echo "=== Probe summaries ==="
find "$SETUP_DIR/../.." -path "*/memory_probe_logs/*_summary.txt" -type f -maxdepth 3 -print 2>/dev/null | sort | tail -40 | while read -r SUMMARY; do
  echo
  echo "--- $SUMMARY"
  cat "$SUMMARY"
done

exit "$FAILED"
