#!/usr/bin/env python3
"""Write brief periodic status snapshots for the standalone campaign."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path


DEFAULT_SCENARIOS = [
    "CORDEX_CMIP5_REMO2015_rcp85",
    "CORDEX_CMIP5_RegCM_rcp85",
    "CMIP5_rcp85",
]

TOTAL_SIMS_PER_SCENARIO = 4 * 3 * 76


def latest_result_leaf(result_root: Path) -> str:
    latest = None
    latest_mtime = -1.0
    for path in result_root.rglob("eplusout.sql"):
        mtime = path.stat().st_mtime
        if mtime > latest_mtime:
            latest = path.parent
            latest_mtime = mtime
    if latest is None:
        return "no-results-yet"
    return str(latest.relative_to(result_root))


def build_snapshot(root: Path, scenarios: list[str]) -> list[str]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"[{timestamp}]"]

    for scenario in scenarios:
        result_root = root / "epw_pipeline" / "energyplus_runner" / "results" / scenario
        epw_root = root / "epw_pipeline" / "epw_out" / scenario

        sim_count = len(list(result_root.rglob("eplusout.sql"))) if result_root.exists() else 0
        epw_count = len(list(epw_root.rglob("*.epw"))) if epw_root.exists() else 0
        latest_leaf = latest_result_leaf(result_root) if sim_count else "no-results-yet"

        lines.append(
            f"{scenario}: sims={sim_count}/{TOTAL_SIMS_PER_SCENARIO}, "
            f"epws={epw_count}/304, latest={latest_leaf}"
        )

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Write periodic campaign status snapshots.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Standalone campaign root.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario names.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=3600,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("run_status/hourly_status.log"),
        help="Status log file, relative to --root unless absolute.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    log_file = args.log_file if args.log_file.is_absolute() else root / args.log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]

    while True:
        lines = build_snapshot(root, scenarios)
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
