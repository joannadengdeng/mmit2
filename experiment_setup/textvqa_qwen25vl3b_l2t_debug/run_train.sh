#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
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

python -m vlmintune.training --config "$SETUP_DIR/train_config.yaml" "$@"

