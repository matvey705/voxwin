"""PyInstaller entry point (see packaging/voxwin.spec)."""

import sys

from voxwin.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
