#!/usr/bin/env python3
"""Plot break-year sensitivity across alternative transition definitions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_DIR = SCRIPT_DIR.parent
SUMMARY_CSV = RUNNER_DIR / "paper_metrics_summary.csv"
FIGURES_DIR = SCRIPT_DIR / "figures"
OUTPUT_PNG = FIGURES_DIR / "break_year_definition_sensitivity.png"

CITY_ORDER = ["Phoenix", "Miami", "Vancouver", "Los_Angeles", "Montreal", "Toronto"]
CITY_LABELS = {
    "Phoenix": "Phoenix",
    "Miami": "Miami",
    "Vancouver": "Vancouver",
    "Los_Angeles": "Los Angeles",
    "Montreal": "Montreal",
    "Toronto": "Toronto",
}
DEFINITIONS = [
    ("break_year", "First severe trigger", "#9e9e9e", "o"),
    ("failure_occurrence_2y_year", "2Y >=25%", "#1f78b4", "s"),
    ("failure_occurrence_3y_year", "3Y >=25%", "#33a02c", "^"),
    ("failure_occurrence_5y_year", "5Y >=25% (primary)", "#e31a1c", "D"),
    ("sustained_failure_year", "Strict 5Y consecutive", "#6a3d9a", "X"),
]
NEVER_YEAR = 2106


def parse_year(value: object) -> int | None:
    if value in (None, "", "Never"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise SystemExit(f"Missing summary CSV: {SUMMARY_CSV}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_CSV)
    summary = summary[summary["city"].isin(CITY_ORDER)].copy()
    summary["city_order"] = summary["city"].map({city: index for index, city in enumerate(CITY_ORDER)})
    summary = summary.sort_values("city_order")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y_positions = {city: len(CITY_ORDER) - index - 1 for index, city in enumerate(CITY_ORDER)}

    for _, row in summary.iterrows():
        city = str(row["city"])
        y = y_positions[city]
        valid_years = [parse_year(row[column]) for column, *_ in DEFINITIONS]
        valid_years = [year for year in valid_years if year is not None]
        if valid_years:
            ax.hlines(y, min(valid_years), max(valid_years), color="#c7c7c7", linewidth=1.0, zorder=1)

        for column, label, color, marker in DEFINITIONS:
            year = parse_year(row[column])
            if year is None:
                ax.scatter(
                    NEVER_YEAR,
                    y,
                    s=55,
                    marker=marker,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=1.4,
                    zorder=4,
                )
            else:
                ax.scatter(
                    year,
                    y,
                    s=55,
                    marker=marker,
                    color=color,
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=5,
                )

    ax.axvline(2025, color="#757575", linewidth=0.8, linestyle="--", alpha=0.8)
    ax.text(2025.4, len(CITY_ORDER) - 0.35, "baseline", fontsize=8, color="#616161")
    ax.axvspan(2101, NEVER_YEAR + 1, color="#f5f5f5", zorder=0)
    ax.text(2103.5, len(CITY_ORDER) - 0.35, "Never", fontsize=8, color="#616161", ha="center")

    ax.set_xlim(2024, NEVER_YEAR + 1)
    ax.set_ylim(-0.7, len(CITY_ORDER) - 0.3)
    ax.set_yticks([y_positions[city] for city in CITY_ORDER])
    ax.set_yticklabels([CITY_LABELS[city] for city in CITY_ORDER])
    ax.set_xlabel("")
    fig.suptitle("Break-Year Definition Sensitivity", fontsize=13, fontweight="bold", x=0.12, ha="left", y=0.98)
    fig.text(
        0.12,
        0.935,
        "Primary paper map uses 5Y >=25%; 2Y/3Y show sensitivity, strict sustained is conservative.",
        fontsize=9,
        color="#424242",
        ha="left",
    )
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.7)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        Line2D([0], [0], marker=marker, linestyle="none", color=color, markerfacecolor=color, markersize=7, label=label)
        for _, label, color, marker in DEFINITIONS
    ]
    ax.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.35), frameon=True, fontsize=8)

    fig.tight_layout(rect=[0, 0.16, 1, 0.9])
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
