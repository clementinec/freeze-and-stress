#!/usr/bin/env python3
"""Run all paper-facing figure scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_SCRIPTS = [
    "figure_1_climate_driver_trajectories.py",
    "figure_2_sentinel_matrix.py",
    "figure_3_driver_block_decomposition.py",
    "figure_4_driver_sentinel_action_map.py",
]


def main() -> None:
    for script in FIGURE_SCRIPTS:
        subprocess.run([sys.executable, str(SCRIPT_DIR / script)], check=True)


if __name__ == "__main__":
    main()
