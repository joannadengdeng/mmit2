"""Package-level training runner entry point.

Usage:
    python -m mmit2.trainrun --config configs/colab_lora_full_train.yaml
"""
from __future__ import annotations

import argparse

from mmit2.training.runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a training config through the runtime-aware runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
