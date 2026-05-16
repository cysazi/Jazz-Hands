"""Run the FL Studio visualizer by itself for manual testing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from jazzhands.visualizer.fl_studio_debug_visualizer import main


if __name__ == "__main__":
    raise SystemExit(main())
