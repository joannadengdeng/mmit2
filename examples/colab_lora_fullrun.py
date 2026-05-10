"""Compatibility wrapper for the package-level full LoRA run entry point.

Preferred command:
    python -m mmit2.fullrun --config configs/colab_lora_full_eval.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmit2.fullrun import main


if __name__ == "__main__":
    main()
