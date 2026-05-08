"""Run the package-level debug entry point.

Usage:
    python -m mmit2.debug --config configs/colab_lora_debug.yaml
"""

from .colab import main


if __name__ == "__main__":
    main()
