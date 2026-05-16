"""Run mocap movement alignment from the organized package."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from jazzhands.mocap.movement_alignment import main


if __name__ == "__main__":
    raise SystemExit(main())
