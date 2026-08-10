#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$ROOT_DIR/experiments}"
STATUS_FILE="${STATUS_FILE:-$EXPERIMENTS_DIR/llava_textvqa_pipeline.status}"
LOG_FILE="${LOG_FILE:-$EXPERIMENTS_DIR/llava_textvqa_pipeline.log}"
VENV_DIR="${VENV_DIR:-/root/autodl-tmp/venvs/vlmintune-torch211}"
METHODS="${METHODS:-lora mores reft dora qlora l2t}"
STAGES="${STAGES:-8:8 256:32 1000:100 34602:5000}"

mkdir -p "$EXPERIMENTS_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing experiment Python: $VENV_DIR/bin/python" >&2
  exit 2
fi

printf 'RUNNING\n' > "$STATUS_FILE"

record_exit() {
  local status=$?
  printf '%s\n' "$status" > "$STATUS_FILE"
}
trap record_exit EXIT

"$VENV_DIR/bin/python" - "$EXPERIMENTS_DIR" <<'PY'
import json
import os
import sys

experiments_dir = sys.argv[1]
for method in ("lora", "mores", "reft", "dora", "vl_adapter", "qlora", "l2t"):
    run_dir = os.path.join(experiments_dir, f"qwen_textvqa_{method}_n34602_s42")
    meta_path = os.path.join(run_dir, "checkpoint", "vlmintune_meta.json")
    eval_path = os.path.join(run_dir, "eval_trained", "eval.json")
    predictions_path = os.path.join(run_dir, "eval_trained", "predictions.jsonl")
    for path in (meta_path, eval_path, predictions_path):
        if not os.path.isfile(path):
            raise SystemExit(f"Qwen prerequisite is incomplete: {path}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    with open(eval_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if meta.get("model_name") != "qwen25vl_3b_instruct" or meta.get("ft_method") != method:
        raise SystemExit(f"Qwen prerequisite metadata mismatch for {method}: {meta}")
    if int(summary.get("num_predictions", -1)) != 5000:
        raise SystemExit(f"Qwen prerequisite eval count mismatch for {method}: {summary}")
    with open(predictions_path, "r", encoding="utf-8") as handle:
        prediction_count = sum(1 for line in handle if line.strip())
    if prediction_count != 5000:
        raise SystemExit(
            f"Qwen prerequisite prediction count mismatch for {method}: {prediction_count}"
        )
print("Qwen prerequisite validated: seven methods completed with 5000 predictions each.")
PY

for stage in $STAGES; do
  stage_samples="${stage%%:*}"
  eval_samples="${stage##*:}"
  echo "================================================================================"
  echo "LLAVA TEXTVQA STAGE train=$stage_samples eval=$eval_samples methods=$METHODS"
  echo "================================================================================"
  STAGE_SAMPLES="$stage_samples" \
  EVAL_SAMPLES="$eval_samples" \
  METHODS="$METHODS" \
  RUN_PREFIX=llava_textvqa \
  bash scripts/run_llava_textvqa_stage.sh
done

echo "LLaVA TextVQA pipeline completed successfully."
