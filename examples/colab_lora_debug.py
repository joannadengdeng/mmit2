"""Compatibility wrapper for the package-level debug entry point.

Preferred command:
    python -m mmit2.debug --config configs/colab_lora_debug.yaml

This file stays in ``examples/`` so existing Colab notebooks and teaching notes
continue to work unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmit2.debug.colab import main


if __name__ == "__main__":
    main()
