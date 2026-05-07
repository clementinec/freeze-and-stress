#!/usr/bin/env python3
"""
Analyze annual EnergyPlus metrics under climate stress.

Selected metrics
----------------
1) Comfort failure
   - unmet_hours
   - ULH ratio (%)
   - break_year based on ULH ratio threshold

2) System stress
   - stress_per_1000kwh = unmet_hours / annual_kwh * 1000

3) Capacity pressure
   - peak_kw
   - peak_ratio = peak_year / peak_2025

4) Resilience proxy
   - years_to_failure = break_year - 2025

5) Failure mode
   - control_limited
   - capacity_pressured
   - mixed_or_unclear
   - no_break_by_end_year
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import pandas as pd


# Annual occupied hours by archetype
OCCUPIED_HOURS_MAP: Dict[str, int] = {
    "office": 2349,
    "retail": 2349,
    "apartment": 6047,
}

# Candidate unmet-hour columns in priority order
UNMET_COLUMNS: List[str] = [
    "facility_cooling_setpoint_not_met_occupied_time_total_hours",
    "abups_occupied_cooling_not_met_hours",
]

# Required columns besides unmet column
BASE_REQUIRED_COLUMNS: List[str] = [
    "scenario",
    "building",
    "city",
    "year",
    "electricity_facility_annual_kwh",
    "facility_total_electricity_demand_rate_max_kw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute filtered climate-stress metrics and plots from annual metrics CSV."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to annual metrics CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory to save summary CSV and plots.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Break threshold for ULH ratio (default: 0.10 = 10%%).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Optional scenario filter.",
    )
    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="Optional city filter.",
    )
    parser.add_argument(
        "--building",
        type=str,
        default=None,
        help="Optional building filter.",
    )
    return parser.parse_args()


def choose_unmet_column(df: pd.DataFrame) -> str:
    for col in UNMET_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(
        "No unmet-hours column found. Expected one of: "
        + ", ".join(UNMET_COLUMNS)
    )


def validate_columns(df: pd.DataFrame, unmet_col: str) -> None:
    required = BASE_REQUIRED_COLUMNS + [unmet_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def assign_occupied_hours(building: str) -> int:
    b = str(building).strip().lower()
    if b not in OCCUPIED_HOURS_MAP:
        raise ValueError(
            f"Unknown building type '{building}'. "
            f"Expected one of {list(OCCUPIED_HOURS_MAP.keys())}"
        )
    return OCCUPIED_HOURS_MAP[b]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["building"] = df["building"].astype(str).str.strip().str.lower()
    df["city"] = df["city"].astype(str).str.strip()
    df["scenario"] = df["scenario"].astype(str).str.strip()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    return df


def apply_filters(
    df: pd.DataFrame,
    scenario: Optional[str] = None,
    city: Optional[str] = None,
    building: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()

    if scenario:
        out = out[out["scenario"] == scenario]

    if city:
        out = out[out["city"].str.lower() == city.strip().lower()]

    if building:
        out = out[out["building"] == building.strip().lower()]

    return out


def add_peak_ratio(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("year").copy()

    # Use 2025 as baseline if present, otherwise use first year
    if (group["year"] == 2025).any():
        baseline_peak = group.loc[
            group["year"] == 2025,
            "facility_total_electricity_demand_rate_max_kw",
        ].iloc[0]
        baseline_year = 2025
    else:
        baseline_peak = group["facility_total_electricity_demand_rate_max_kw"].iloc[0]
        baseline_year = int(group["year"].iloc[0])

    if pd.isna(baseline_peak) or baseline_peak == 0:
        group["baseline_peak_kw"] = pd.NA
        group["baseline_peak_year"] = baseline_year
        group["peak_ratio"] = pd.NA
    else:
        group["baseline_peak_kw"] = baseline_peak
        group["baseline_peak_year"] = baseline_year
        group["peak_ratio"] = (
            group["facility_total_electricity_demand_rate_max_kw"] / baseline_peak
        )

    return group


def compute_metrics(df: pd.DataFrame, unmet_col: str) -> pd.DataFrame:
    df = df.copy()

    # Core numeric fields
    df["occupied_hours"] = df["building"].map(assign_occupied_hours)
    df["unmet_hours"] = pd.to_numeric(df[unmet_col], errors="coerce")
    df["annual_kwh"] = pd.to_numeric(df["electricity_facility_annual_kwh"], errors="coerce")
    df["peak_kw"] = pd.to_numeric(
        df["facility_total_electricity_demand_rate_max_kw"],
        errors="coerce"
    )

    # Comfort failure
    df["ulh_ratio"] = df["unmet_hours"] / df["occupied_hours"]
    df["ulh_percent"] = df["ulh_ratio"] * 100.0

    # System stress
    df["stress_index"] = df["unmet_hours"] / df["annual_kwh"]
    df["stress_per_1000kwh"] = df["stress_index"] * 1000.0

    # Capacity pressure
    grouped = []
    for _, g in df.groupby(["scenario", "building", "city"], dropna=False):
        grouped.append(add_peak_ratio(g))
    df = pd.concat(grouped, ignore_index=True)

    return df


def classify_failure_mode(g: pd.DataFrame, threshold: float) -> str:
    g = g.sort_values("year").copy()

    broken = g[g["ulh_ratio"] > threshold]
    if broken.empty:
        return "no_break_by_end_year"

    first_break_year = int(broken["year"].min())
    post = g[g["year"] >= first_break_year].copy()

    baseline_stress = g["stress_per_1000kwh"].iloc[0]
    median_post_stress = post["stress_per_1000kwh"].median()
    median_post_peak_ratio = post["peak_ratio"].median()

    stress_increase = median_post_stress - baseline_stress

    if pd.isna(median_post_peak_ratio):
        return "mixed_or_unclear"

    peak_change = median_post_peak_ratio - 1.0

    if peak_change < 0.10 and stress_increase > 0:
        return "control_limited"
    if peak_change >= 0.10:
        return "capacity_pressured"

    return "mixed_or_unclear"


def summarize_break_years(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    summaries = []

    for keys, g in df.groupby(["scenario", "building", "city"], dropna=False):
        g = g.sort_values("year").copy()

        broken = g[g["ulh_ratio"] > threshold]
        break_year = int(broken["year"].min()) if not broken.empty else None

        start_year = int(g["year"].min())
        end_year = int(g["year"].max())
        baseline_year = int(g["baseline_peak_year"].iloc[0])

        if break_year is None:
            years_to_failure = None
        else:
            years_to_failure = break_year - baseline_year

        summaries.append(
            {
                "scenario": keys[0],
                "building": keys[1],
                "city": keys[2],
                "start_year": start_year,
                "end_year": end_year,
                "baseline_year": baseline_year,
                "break_threshold_ratio": threshold,
                "break_threshold_percent": threshold * 100.0,
                "break_year": break_year,
                "years_to_failure": years_to_failure,
                "baseline_peak_kw": g["baseline_peak_kw"].iloc[0],
                "last_year_ulh_percent": g.loc[g["year"] == end_year, "ulh_percent"].iloc[0],
                "last_year_stress_per_1000kwh": g.loc[g["year"] == end_year, "stress_per_1000kwh"].iloc[0],
                "last_year_peak_ratio": g.loc[g["year"] == end_year, "peak_ratio"].iloc[0],
                "failure_mode": classify_failure_mode(g, threshold),
            }
        )

    return pd.DataFrame(summaries)


def safe_name(*parts: str) -> str:
    return "__".join(str(p).replace(" ", "_").replace("/", "_") for p in parts)


def plot_case(g: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    g = g.sort_values("year").copy()

    scenario = g["scenario"].iloc[0]
    building = g["building"].iloc[0]
    city = g["city"].iloc[0]

    broken = g[g["ulh_ratio"] > threshold]
    break_year = int(broken["year"].min()) if not broken.empty else None

    # 1. ULH ratio
    plt.figure(figsize=(8, 5))
    plt.plot(g["year"], g["ulh_percent"], marker="o")
    plt.axhline(threshold * 100.0, linestyle="--")
    if break_year is not None:
        plt.axvline(break_year, linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("ULH (%)")
    plt.title(f"ULH ratio | {scenario} | {building} | {city}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{safe_name(scenario, building, city)}__ulh_percent.png", dpi=200)
    plt.close()

    # 2. Stress
    plt.figure(figsize=(8, 5))
    plt.plot(g["year"], g["stress_per_1000kwh"], marker="o")
    if break_year is not None:
        plt.axvline(break_year, linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("unmet h / 1000 kWh")
    plt.title(f"System stress | {scenario} | {building} | {city}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{safe_name(scenario, building, city)}__stress_index.png", dpi=200)
    plt.close()

    # 3. Peak ratio
    plt.figure(figsize=(8, 5))
    plt.plot(g["year"], g["peak_ratio"], marker="o")
    plt.axhline(1.0, linestyle="--")
    if break_year is not None:
        plt.axvline(break_year, linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("Peak ratio (-)")
    plt.title(f"Peak ratio | {scenario} | {building} | {city}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{safe_name(scenario, building, city)}__peak_ratio.png", dpi=200)
    plt.close()

    # 4. Combined figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    axes[0, 0].plot(g["year"], g["unmet_hours"], marker="o")
    if break_year is not None:
        axes[0, 0].axvline(break_year, linestyle="--")
    axes[0, 0].set_title("Occupied unmet cooling hours")
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].set_ylabel("Hours")

    axes[0, 1].plot(g["year"], g["ulh_percent"], marker="o")
    axes[0, 1].axhline(threshold * 100.0, linestyle="--")
    if break_year is not None:
        axes[0, 1].axvline(break_year, linestyle="--")
    axes[0, 1].set_title("ULH ratio")
    axes[0, 1].set_xlabel("Year")
    axes[0, 1].set_ylabel("%")

    axes[1, 0].plot(g["year"], g["annual_kwh"], marker="o")
    axes[1, 0].set_title("Annual facility electricity")
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].set_ylabel("kWh")

    axes[1, 1].plot(g["year"], g["peak_kw"], marker="o")
    axes[1, 1].set_title("Peak facility demand")
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("kW")

    axes[2, 0].plot(g["year"], g["stress_per_1000kwh"], marker="o")
    if break_year is not None:
        axes[2, 0].axvline(break_year, linestyle="--")
    axes[2, 0].set_title("System stress")
    axes[2, 0].set_xlabel("Year")
    axes[2, 0].set_ylabel("unmet h / 1000 kWh")

    axes[2, 1].plot(g["year"], g["peak_ratio"], marker="o")
    axes[2, 1].axhline(1.0, linestyle="--")
    if break_year is not None:
        axes[2, 1].axvline(break_year, linestyle="--")
    axes[2, 1].set_title("Peak ratio")
    axes[2, 1].set_xlabel("Year")
    axes[2, 1].set_ylabel("-")

    fig.suptitle(f"{scenario} | {building} | {city}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_name(scenario, building, city)}__combined.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = preprocess(df)

    unmet_col = choose_unmet_column(df)
    validate_columns(df, unmet_col)

    df = apply_filters(
        df,
        scenario=args.scenario,
        city=args.city,
        building=args.building,
    )

    if df.empty:
        raise ValueError("No rows left after filtering.")

    df = compute_metrics(df, unmet_col)

    # Keep only selected analysis columns + identifiers
    selected_cols = [
        "scenario",
        "building",
        "city",
        "year",
        "occupied_hours",
        "unmet_hours",
        "annual_kwh",
        "peak_kw",
        "baseline_peak_year",
        "baseline_peak_kw",
        "ulh_ratio",
        "ulh_percent",
        "stress_index",
        "stress_per_1000kwh",
        "peak_ratio",
    ]
    filtered_df = df[selected_cols].sort_values(["scenario", "building", "city", "year"])

    filtered_csv = args.output_dir / "annual_metrics_filtered.csv"
    filtered_df.to_csv(filtered_csv, index=False)

    summary = summarize_break_years(df, threshold=args.threshold)
    summary_csv = args.output_dir / "break_year_summary.csv"
    summary.to_csv(summary_csv, index=False)

    for _, g in df.groupby(["scenario", "building", "city"], dropna=False):
        plot_case(g, args.output_dir, threshold=args.threshold)

    print(f"[OK] Filtered annual metrics written to: {filtered_csv}")
    print(f"[OK] Break-year summary written to: {summary_csv}")
    print(f"[OK] Plots written to: {args.output_dir}")


if __name__ == "__main__":
    main()