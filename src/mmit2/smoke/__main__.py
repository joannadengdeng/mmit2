"""Run the package-level smoke entry point.

Usage:
    python -m mmit2.smoke --suite quick
"""

from .matrix import main


if __name__ == "__main__":
    main()
