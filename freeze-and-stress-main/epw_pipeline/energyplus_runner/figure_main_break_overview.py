#!/usr/bin/env python3
"""Paper-facing overview figure using main break year logic only.

This figure does not use adaptive re-freeze outputs. It relies on:
  - metric_exports/annual_metrics(_extended).csv
  - paper_metrics_summary.csv

Panels:
  A. Main break year and sustained failure year across six cities.
  B. Representative city trajectory with adjusted main-break logic.
  C. Representative city trajectory with baseline-breached logic.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

import compute_break_years_multi_metric as cbm
from figure_paper_common import CLIMATE_CITY, CLIMATE_ORDER


SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_ROOT = SCRIPT_DIR / "metric_res"
ANNUAL_METRICS = SCRIPT_DIR / "metric_exports" / "annual_metrics_extended.csv"
if not ANNUAL_METRICS.exists():
    ANNUAL_METRICS = SCRIPT_DIR / "metric_exports" / "annual_metrics.csv"

CITY_ORDER = [CLIMATE_CITY[code] for code in CLIMATE_ORDER]
SCENARIO = "CORDEX_CMIP5_REMO2015_rcp85"
BUILDING = "office"
METRIC_PANELS = [
    ("annual_edh_c_h", "Annual EDH", 300.0, "#b44b5f"),
    ("cooling_setpoint_not_met_occupied_hours", "Cooling unmet occupied hours", 500.0, "#4f6d8a"),
]
SEVERITY_COLOR = "#7b5c9d"
BREAK_COLOR = "#1f1f1f"
SUSTAINED_COLOR = "#5c7ea0"


def format_city(city: str) -> str:
    return city.replace("_", " ")


def load_summary(summary_csv: Path, scenario: str, building: str) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df = df[(df["scenario"] == scenario) & (df["building"] == building)].copy()
    for col in ["break_year", "sustained_failure_year", "baseline_failure_severity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_records(annual_csv: Path, scenario: str, building: str) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    with annual_csv.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for source in reader:
            if source.get("scenario") != scenario or source.get("building") != building:
                continue
            row = cbm.normalize_row(source)
            city = str(row["city"])
            grouped.setdefault(city, []).append(row)
    for city, records in grouped.items():
        grouped[city] = sorted(records, key=lambda row: row["year"])
    return grouped


def plot_summary_panel(ax: plt.Axes, summary_df: pd.DataFrame) -> None:
    cities = [city for city in CITY_ORDER if city in set(summary_df["city"])]
    for idx, city in enumerate(cities):
        row = summary_df[summary_df["city"] == city].iloc[0]
        y = len(cities) - 1 - idx
        break_year = row["break_year"]
        sustained = row["sustained_failure_year"]

        if pd.notna(break_year) and pd.notna(sustained):
            ax.plot([break_year, sustained], [y, y], color=SUSTAINED_COLOR, linewidth=2.0, alpha=0.95)
        elif pd.notna(break_year):
            ax.plot([break_year, break_year + 0.8], [y, y], color=SUSTAINED_COLOR, linewidth=2.0, alpha=0.5)

        if pd.notna(break_year):
            ax.scatter(break_year, y, s=52, color=BREAK_COLOR, edgecolor="white", linewidth=0.6, zorder=4)
        else:
            ax.text(2101.2, y, "Never", va="center", ha="left", fontsize=8, color="#666666")

        if pd.notna(sustained):
            ax.scatter(sustained, y, s=108, facecolor="white", edgecolor=SUSTAINED_COLOR, linewidth=2.0, zorder=5)

    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels([format_city(city) for city in cities[::-1]])
    ax.set_ylim(-0.5, len(cities) - 0.5)
    ax.set_xlim(2024, 2105)
    ax.set_xlabel("Year")
    ax.set_title("A  Main break and sustained failure across six cities", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linewidth=0.3, alpha=0.45)
    ax.axvline(2025, color="#999999", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.scatter([], [], s=52, color=BREAK_COLOR, label="Main break year")
    ax.scatter([], [], s=108, facecolor="white", edgecolor=SUSTAINED_COLOR, linewidth=2.0, label="Sustained failure year")
    ax.legend(loc="lower right", frameon=False, fontsize=8)


def plot_city_panel(
    axes: list[plt.Axes],
    records: list[dict[str, object]],
    summary_row: pd.Series,
    city: str,
    panel_letter: str,
) -> None:
    years = [record["year"] for record in records]
    severity = [cbm.failure_severity(record, cbm.FAILURE_METRICS) for record in records]
    baseline_year = int(summary_row["baseline_year"])
    baseline_record = next((record for record in records if record["year"] == baseline_year), records[0])
    baseline_sev = cbm.failure_severity(baseline_record, cbm.FAILURE_METRICS)
    baseline_failed = cbm.any_failure(baseline_record, cbm.FAILURE_METRICS)
    break_year = summary_row["break_year"]
    sustained_year = summary_row["sustained_failure_year"]

    for ax, (metric_key, label, threshold, color) in zip(axes[:2], METRIC_PANELS):
        vals = [cbm.to_float(record.get(metric_key)) for record in records]
        ax.plot(years, vals, color=color, linewidth=1.7)
        breaches = [
            (year, value)
            for year, value in zip(years, vals)
            if value is not None and value > threshold
        ]
        if breaches:
            ax.scatter(
                [year for year, _ in breaches],
                [value for _, value in breaches],
                s=24,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
        ax.axhline(threshold, color=color, linewidth=1.0, linestyle="--", alpha=0.7)
        ax.text(2025.4, threshold, f"threshold {threshold:g}", fontsize=7, color=color, va="bottom")
        if pd.notna(break_year):
            ax.axvline(break_year, color=BREAK_COLOR, linewidth=0.9, linestyle=":")
        if pd.notna(sustained_year):
            ax.axvline(sustained_year, color=SUSTAINED_COLOR, linewidth=0.9, linestyle=":")
        ax.set_xlim(2025, 2100)
        ax.set_ylabel(label, color=color, fontsize=8)
        ax.grid(linewidth=0.3, alpha=0.4)

    sev_ax = axes[2]
    sev_ax.plot(years, severity, color=SEVERITY_COLOR, linewidth=1.8)
    sev_ax.scatter(years, severity, s=10, color=SEVERITY_COLOR, alpha=0.7)
    if baseline_failed:
        sev_ax.axhline(
            baseline_sev + cbm.STRESS_EMERGENCE_DELTA,
            color=SEVERITY_COLOR,
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
        )
        sev_ax.text(
            2025.4,
            baseline_sev + cbm.STRESS_EMERGENCE_DELTA,
            f"baseline severity + {cbm.STRESS_EMERGENCE_DELTA:.2f}",
            fontsize=7,
            color=SEVERITY_COLOR,
            va="bottom",
        )
    else:
        sev_ax.axhline(0.0, color=SEVERITY_COLOR, linewidth=1.0, linestyle="--", alpha=0.55)
        sev_ax.text(2025.4, 0.0, "first failure after baseline", fontsize=7, color=SEVERITY_COLOR, va="bottom")

    if pd.notna(break_year):
        sev_ax.axvline(break_year, color=BREAK_COLOR, linewidth=0.9, linestyle=":")
    if pd.notna(sustained_year):
        sev_ax.axvline(sustained_year, color=SUSTAINED_COLOR, linewidth=0.9, linestyle=":")
    sev_ax.set_xlim(2025, 2100)
    sev_ax.set_xlabel("Year")
    sev_ax.set_ylabel("Failure severity", color=SEVERITY_COLOR, fontsize=8)
    sev_ax.grid(linewidth=0.3, alpha=0.4)

    title = f"{panel_letter}  {format_city(city)}"
    subtitle = "baseline already breached; main break = stress emergence year" if baseline_failed else "baseline not breached; main break = first failure year"
    axes[0].set_title(f"{title}\n{subtitle}", loc="left", fontsize=11, fontweight="bold")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)


def make_figure(
    summary_df: pd.DataFrame,
    records_by_city: dict[str, list[dict[str, object]]],
    city_a: str,
    city_b: str,
    output_png: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig = plt.figure(figsize=(14.5, 8.3), facecolor="white")
    gs = gridspec.GridSpec(3, 3, width_ratios=[1.35, 1.0, 1.0], hspace=0.16, wspace=0.30)

    ax_a = fig.add_subplot(gs[:, 0])
    plot_summary_panel(ax_a, summary_df)

    axes_b = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    axes_c = [fig.add_subplot(gs[i, 2]) for i in range(3)]

    row_b = summary_df[summary_df["city"] == city_a].iloc[0]
    row_c = summary_df[summary_df["city"] == city_b].iloc[0]
    plot_city_panel(axes_b, records_by_city[city_a], row_b, city_a, "B")
    plot_city_panel(axes_c, records_by_city[city_b], row_c, city_b, "C")

    fig.suptitle(
        "Main break year uses baseline-adjusted stress emergence rather than raw first exceedance",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight", facecolor="white")
    output_pdf = output_png.with_suffix(".pdf")
    fig.savefig(output_pdf, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create main-break overview figure from annual metrics.")
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--building", default=BUILDING)
    parser.add_argument("--city-a", default="Los_Angeles")
    parser.add_argument("--city-b", default="Phoenix")
    parser.add_argument("--summary-root", type=Path, default=SUMMARY_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--annual-csv", type=Path, default=ANNUAL_METRICS)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "metric_exports" / "figures" / "fig_main_break_overview.png",
    )
    args = parser.parse_args()

    summary_csv = (
        args.summary_csv
        if args.summary_csv is not None
        else args.summary_root / args.scenario / "paper_metrics_summary.csv"
    )
    summary_df = load_summary(summary_csv, args.scenario, args.building)
    records_by_city = load_records(args.annual_csv, args.scenario, args.building)
    make_figure(summary_df, records_by_city, args.city_a, args.city_b, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
