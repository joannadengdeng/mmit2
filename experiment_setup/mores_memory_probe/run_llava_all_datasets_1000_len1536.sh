#!/usr/bin/env bash
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SETUP_DIR/run_matrix.sh" llava 1000 1536
