"""Compatibility wrapper for the package-level training runner.

Preferred command:
    python -m mmit2.trainrun --config configs/colab_lora_full_train.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmit2.trainrun import main


if __name__ == "__main__":
    main()
