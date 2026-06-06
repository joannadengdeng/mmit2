#!/usr/bin/env bash
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SETUP_DIR/run_probe.sh" llava textvqa 0 1024
