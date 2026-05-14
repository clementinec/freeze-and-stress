#!/usr/bin/env python3
"""Compute derived climate-driver metrics from EPW files."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = SCRIPT_DIR / "annual_metrics.csv"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "annual_metrics_extended.csv"
DEFAULT_EPW_ROOT = SCRIPT_DIR.parent.parent / "epw_out"

HEADER_LINES = 8
DRYBULB_IDX = 6
RH_IDX = 8
PRESSURE_IDX = 9
GHI_IDX = 13

NEW_COLUMNS = [
    "hdd_18",
    "cdd_18",
    "summer_mean_drybulb_c",
    "outdoor_humidity_ratio_mean",
    "high_wetbulb_hours",
    "high_rh_hours",
    "annual_ghi_kwh_m2",
    "summer_ghi_kwh_m2",
    "peak_ghi_w_m2",
    "extreme_solar_hours",
    "hot_days_30c",
    "hot_days_35c",
    "heatwave_days",
    "hot_nights",
    "max_consec_hot_days_30c",
    "max_consec_hot_days_35c",
]

SUMMER_MONTHS = {6, 7, 8}
HIGH_WETBULB_C = 28.0
HIGH_RH_PCT = 70.0
EXTREME_GHI_W_M2 = 800.0
HOT_NIGHT_C = 25.0
HEATWAVE_DAILY_MAX_C = 35.0
HEATWAVE_MIN_RUN_DAYS = 3


def iter_progress(total: int, desc: str):
    if HAS_TQDM:
        return tqdm(total=total, desc=desc, unit="epw")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add HDD/CDD, GHI, and persistence drivers to annual metrics.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_CSV), help="Input annual metrics CSV.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_CSV), help="Output extended annual metrics CSV.")
    parser.add_argument("--epw-root", default=str(DEFAULT_EPW_ROOT), help="Root directory containing scenario/city EPWs.")
    return parser.parse_args()


def saturation_vapor_pressure_kpa(drybulb_c: float) -> float:
    return 0.61094 * math.exp((17.625 * drybulb_c) / (drybulb_c + 243.04))


def humidity_ratio(drybulb_c: float, rh_pct: float, pressure_pa: float) -> float | None:
    pressure_kpa = pressure_pa / 1000.0
    vapor_pressure = max(0.0, min(rh_pct, 100.0)) / 100.0 * saturation_vapor_pressure_kpa(drybulb_c)
    if pressure_kpa <= vapor_pressure:
        return None
    return 0.62198 * vapor_pressure / (pressure_kpa - vapor_pressure)


def wetbulb_stull_c(drybulb_c: float, rh_pct: float) -> float:
    rh = max(0.0, min(rh_pct, 100.0))
    return (
        drybulb_c * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(drybulb_c + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
        - 4.686035
    )


def parse_epw_hourly(epw_path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with open(epw_path, encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if line_number < HEADER_LINES:
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) <= max(DRYBULB_IDX, RH_IDX, PRESSURE_IDX, GHI_IDX):
                continue
            try:
                month = int(parts[1])
                day = int(parts[2])
                drybulb = float(parts[DRYBULB_IDX])
                rh = float(parts[RH_IDX])
                pressure = float(parts[PRESSURE_IDX])
                ghi = max(float(parts[GHI_IDX]), 0.0)
            except (IndexError, ValueError):
                continue

            wetbulb = wetbulb_stull_c(drybulb, rh)
            humidity = humidity_ratio(drybulb, rh, pressure)
            rows.append(
                {
                    "month": month,
                    "day": day,
                    "drybulb": drybulb,
                    "rh": rh,
                    "pressure": pressure,
                    "ghi": ghi,
                    "wetbulb": wetbulb,
                    "humidity_ratio": humidity if humidity is not None else float("nan"),
                }
            )
    return rows


def longest_run_above(values: list[float], threshold: float) -> int:
    best = 0
    current = 0
    for value in values:
        if value >= threshold:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def count_days_above(values: list[float], threshold: float) -> int:
    return sum(1 for value in values if value >= threshold)


def count_heatwave_days(values: list[float], threshold: float, min_run_days: int) -> int:
    total = 0
    current = 0
    for value in values:
        if value >= threshold:
            current += 1
            continue
        if current >= min_run_days:
            total += current
        current = 0
    if current >= min_run_days:
        total += current
    return total


def mean(values: list[float]) -> float | None:
    clean = [value for value in values if not math.isnan(value)]
    return (sum(clean) / len(clean)) if clean else None


def compute_drivers(hourly: list[dict[str, float | int]]) -> dict[str, float | int]:
    base_temp = 18.0
    hdd_18 = 0.0
    cdd_18 = 0.0
    daily_max: dict[tuple[int, int], float] = defaultdict(lambda: float("-inf"))
    daily_min: dict[tuple[int, int], float] = defaultdict(lambda: float("inf"))

    for row in hourly:
        drybulb = float(row["drybulb"])
        if drybulb < base_temp:
            hdd_18 += (base_temp - drybulb) / 24.0
        else:
            cdd_18 += (drybulb - base_temp) / 24.0

        day_key = (int(row["month"]), int(row["day"]))
        daily_max[day_key] = max(daily_max[day_key], drybulb)
        daily_min[day_key] = min(daily_min[day_key], drybulb)

    ordered_days = sorted(daily_max)
    ordered_daily_max = [daily_max[key] for key in ordered_days]
    ordered_daily_min = [daily_min[key] for key in ordered_days]
    annual_ghi_kwh_m2 = sum(float(row["ghi"]) for row in hourly) / 1000.0
    summer_rows = [row for row in hourly if int(row["month"]) in SUMMER_MONTHS]
    summer_mean_drybulb = mean([float(row["drybulb"]) for row in summer_rows])
    outdoor_humidity_ratio_mean = mean([float(row["humidity_ratio"]) for row in hourly])

    return {
        "hdd_18": round(hdd_18, 2),
        "cdd_18": round(cdd_18, 2),
        "summer_mean_drybulb_c": round(summer_mean_drybulb, 4) if summer_mean_drybulb is not None else pd.NA,
        "outdoor_humidity_ratio_mean": round(outdoor_humidity_ratio_mean, 8) if outdoor_humidity_ratio_mean is not None else pd.NA,
        "high_wetbulb_hours": sum(1 for row in hourly if float(row["wetbulb"]) >= HIGH_WETBULB_C),
        "high_rh_hours": sum(1 for row in hourly if float(row["rh"]) >= HIGH_RH_PCT),
        "annual_ghi_kwh_m2": round(annual_ghi_kwh_m2, 2),
        "summer_ghi_kwh_m2": round(sum(float(row["ghi"]) for row in summer_rows) / 1000.0, 2),
        "peak_ghi_w_m2": round(max(float(row["ghi"]) for row in hourly), 2),
        "extreme_solar_hours": sum(1 for row in hourly if float(row["ghi"]) >= EXTREME_GHI_W_M2),
        "hot_days_30c": count_days_above(ordered_daily_max, 30.0),
        "hot_days_35c": count_days_above(ordered_daily_max, 35.0),
        "heatwave_days": count_heatwave_days(ordered_daily_max, HEATWAVE_DAILY_MAX_C, HEATWAVE_MIN_RUN_DAYS),
        "hot_nights": count_days_above(ordered_daily_min, HOT_NIGHT_C),
        "max_consec_hot_days_30c": longest_run_above(ordered_daily_max, 30.0),
        "max_consec_hot_days_35c": longest_run_above(ordered_daily_max, 35.0),
    }


def resolve_epw_path(epw_root: Path, row: pd.Series) -> Path:
    scenario = str(row["scenario"])
    city = str(row["city"])
    year = int(row["year"])
    return epw_root / scenario / city / f"{city}_{year}_climatedelta.epw"


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input).resolve()
    output_csv = Path(args.output).resolve()
    epw_root = Path(args.epw_root).resolve()

    df = pd.read_csv(input_csv)
    for column in NEW_COLUMNS:
        df[column] = pd.NA

    processed = 0
    missing_epw = 0
    empty_epw = 0

    progress = iter_progress(len(df), "Deriving drivers")
    for index, row in df.iterrows():
        epw_path = resolve_epw_path(epw_root, row)
        if not epw_path.exists():
            missing_epw += 1
            if progress is not None:
                progress.update(1)
            elif (index + 1) % 100 == 0 or (index + 1) == len(df):
                print(f"Deriving drivers: {index + 1}/{len(df)}")
            continue

        hourly = parse_epw_hourly(epw_path)
        if not hourly:
            empty_epw += 1
            if progress is not None:
                progress.update(1)
            elif (index + 1) % 100 == 0 or (index + 1) == len(df):
                print(f"Deriving drivers: {index + 1}/{len(df)}")
            continue

        derived = compute_drivers(hourly)
        for column, value in derived.items():
            df.at[index, column] = value

        processed += 1
        if progress is not None:
            progress.update(1)
        elif (index + 1) % 100 == 0 or (index + 1) == len(df):
            print(f"Deriving drivers: {index + 1}/{len(df)}")

    if progress is not None:
        progress.close()

    df.to_csv(output_csv, index=False)
    print(f"Loaded input rows : {len(df)}")
    print(f"Processed EPWs    : {processed}")
    print(f"Missing EPWs      : {missing_epw}")
    print(f"Empty EPWs        : {empty_epw}")
    print(f"Saved             : {output_csv}")
    print(f"Added columns     : {', '.join(NEW_COLUMNS)}")


if __name__ == "__main__":
    main()
