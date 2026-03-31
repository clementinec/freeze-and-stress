#!/usr/bin/env python3
"""
Run the standalone verified weather pipeline through EPW generation.

This wrapper does three steps:
  1. Generate RS-VAR baseline+anomaly outputs.
  2. Apply the verified REMO2015 daily climate delta.
  3. Convert hourly V6 outputs into year-by-year EPWs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CITY_NAME_MAP = {
    "Los_Angeles": "Los Angeles",
    "Miami": "Miami",
    "Montreal": "Montreal",
    "Toronto": "Toronto",
}


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"\n[{cwd.name}] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone V6 weather generation through EPW creation.")
    parser.add_argument(
        "--cities",
        type=str,
        default="Los_Angeles,Miami,Montreal,Toronto",
        help="Comma-separated city slugs.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="CORDEX_CMIP5_REMO2015_rcp85",
        help="Scenario name from weather/config.yaml.",
    )
    parser.add_argument("--delta-mode", type=str, default="daily", choices=["daily", "monthly"])
    parser.add_argument("--start-year", type=int, default=2025)
    parser.add_argument("--end-year", type=int, default=2100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    weather_dir = root / "weather"
    epw_dir = root / "epw_pipeline"

    city_slugs = [city.strip() for city in args.cities.split(",") if city.strip()]
    invalid = [city for city in city_slugs if city not in CITY_NAME_MAP]
    if invalid:
        raise SystemExit(f"Unsupported cities: {invalid}. Supported: {sorted(CITY_NAME_MAP)}")

    weather_cities = [CITY_NAME_MAP[city] for city in city_slugs]

    rsvar_dir = weather_dir / "output" / "RSVAR"
    climate_dir = weather_dir / "output" / ("RSVAR_cd_monthly" if args.delta_mode == "monthly" else "RSVAR_cd")

    rsvar_cmd = [
        sys.executable,
        "run_v5_save_rsvar_batch.py",
        "--output",
        str(rsvar_dir),
        "--city",
        *weather_cities,
    ]
    _run(rsvar_cmd, weather_dir)

    climate_cmd = [
        sys.executable,
        "apply_climate_delta_batch.py",
        "--scenario",
        args.scenario,
        "--rsvar-dir",
        str(rsvar_dir),
        "--out-dir",
        str(climate_dir),
        "--delta-mode",
        args.delta_mode,
        "--city",
        *weather_cities,
    ]
    if args.overwrite:
        climate_cmd.append("--overwrite")
    _run(climate_cmd, weather_dir)

    epw_cmd = [
        sys.executable,
        "generate_epws_from_v6.py",
        "--weather-root",
        str(climate_dir),
        "--scenario",
        args.scenario,
        "--cities",
        ",".join(city_slugs),
        "--out-root",
        str(epw_dir / "epw_out"),
        "--start-year",
        str(args.start_year),
        "--end-year",
        str(args.end_year),
    ]
    _run(epw_cmd, epw_dir)


if __name__ == "__main__":
    main()
