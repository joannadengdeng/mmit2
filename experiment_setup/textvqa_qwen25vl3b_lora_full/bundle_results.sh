#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-textvqa_qwen25vl3b_lora_full}"
EXPERIMENT_BASE_DIR="${EXPERIMENT_BASE_DIR:-experiments}"
EVAL_OUTPUTS_DIR="${EVAL_OUTPUTS_DIR:-eval_outputs}"
RUN_LOGS_DIR="${RUN_LOGS_DIR:-run_logs}"
BUNDLE_BASE_DIR="${BUNDLE_BASE_DIR:-experiment_results}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

cd "$ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if ! python -m pip --version >/dev/null 2>&1; then
  python -m ensurepip --upgrade
fi

python -m pip install --upgrade pip
if [[ "$SKIP_INSTALL" != "1" ]]; then
  python -m pip install -e ".[finetune]"
fi

python -m vlmintune.results.bundle \
  --experiment-name "$EXPERIMENT_NAME" \
  --experiment-base-dir "$EXPERIMENT_BASE_DIR" \
  --eval-outputs-dir "$EVAL_OUTPUTS_DIR" \
  --run-logs-dir "$RUN_LOGS_DIR" \
  --bundle-base-dir "$BUNDLE_BASE_DIR" \
  --setup-dir "$SETUP_DIR" \
  "$@"
