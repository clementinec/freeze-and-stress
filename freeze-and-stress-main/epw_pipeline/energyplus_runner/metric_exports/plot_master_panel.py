#!/usr/bin/env python3
"""Plot a paper-facing master panel with drivers and response families."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "annual_metrics_extended.csv"
if not INPUT_CSV.exists():
    INPUT_CSV = SCRIPT_DIR / "annual_metrics.csv"
FIGURES_DIR = SCRIPT_DIR / "figures"
OUTPUT_PNG = FIGURES_DIR / "master_panel_drivers_responses.png"

CITY_COLORS = {
    "Los_Angeles": "#c43c39",
    "Miami": "#dd7d21",
    "Montreal": "#4a6fa5",
    "Phoenix": "#8b5a2b",
    "Toronto": "#2f8f5b",
    "Vancouver": "#6b7aa1",
}

FAMILY_COLORS = {
    "Climate Drivers": "#355c7d",
    "Habitability": "#c06c84",
    "Operational Adequacy": "#6c5b7b",
    "Energy Burden": "#f67280",
}

BREAK_THRESHOLDS = {
    "annual_edh_c_h": 156.0,
    "daily_edh_max_k_h": 6.0,
    "daily_edh_exceed_6kh_count": 10.0,
    "facility_cooling_setpoint_not_met_occupied_time_total_hours": 300.0,
    "abups_occupied_heating_not_met_hours": 300.0,
}

METRIC_FAMILIES: OrderedDict[str, list[str]] = OrderedDict(
    {
        "Climate Drivers": [
            "outdoor_drybulb_c_mean",
            "summer_mean_drybulb_c",
            "cdd_18",
            "hdd_18",
            "outdoor_wetbulb_c_max",
            "annual_ghi_kwh_m2",
            "peak_ghi_w_m2",
            "heatwave_days",
            "max_consec_hot_days_35c",
        ],
        "Habitability": [
            "rep_zone_operative_temp_c_max",
            "delta_t_max_k",
            "annual_edh_c_h",
            "daily_edh_max_k_h",
            "daily_edh_exceed_6kh_count",
        ],
        "Operational Adequacy": [
            "facility_cooling_setpoint_not_met_occupied_time_total_hours",
            "abups_occupied_heating_not_met_hours",
            "cooling_electricity_peak_kw",
            "heating_electricity_peak_kw",
        ],
        "Energy Burden": [
            "cooling_electricity_annual_kwh",
            "heating_electricity_annual_kwh",
        ],
    }
)


def pretty_label(metric: str) -> str:
    label = metric.replace("_", " ")
    label = label.replace(" c h", " C h")
    label = label.replace(" k h", " K h")
    label = label.replace(" ghi", " GHI")
    label = label.replace(" kw", " kW")
    label = label.replace(" kwh", " kWh")
    return label


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    if "building" in df.columns:
        df = df[df["building"] == "office"].copy()
    cities = sorted(df["city"].dropna().unique())

    max_cols = max(len(metrics) for metrics in METRIC_FAMILIES.values())
    fig, axes = plt.subplots(len(METRIC_FAMILIES), max_cols, figsize=(4.2 * max_cols, 3.2 * len(METRIC_FAMILIES)), sharex=True)
    axes = np.atleast_2d(axes)

    for row_index, (family, metrics) in enumerate(METRIC_FAMILIES.items()):
        for col_index in range(max_cols):
            ax = axes[row_index, col_index]
            if col_index >= len(metrics):
                ax.set_visible(False)
                continue

            metric = metrics[col_index]
            if metric not in df.columns:
                ax.set_visible(False)
                continue

            for city in cities:
                city_df = df[df["city"] == city].sort_values("year")
                years = pd.to_numeric(city_df["year"], errors="coerce")
                values = pd.to_numeric(city_df[metric], errors="coerce")
                ax.plot(years, values, "o-", ms=2.2, lw=1.15, color=CITY_COLORS.get(city, "#555555"), label=city)

                mask = years.notna() & values.notna()
                if mask.sum() >= 3:
                    coeffs = np.polyfit(years[mask].astype(float), values[mask].astype(float), 1)
                    ax.plot(years[mask], np.polyval(coeffs, years[mask]), "--", lw=0.7, alpha=0.5, color=CITY_COLORS.get(city, "#555555"))

            if metric in BREAK_THRESHOLDS:
                ax.axhline(BREAK_THRESHOLDS[metric], color="#b22222", linestyle=":", linewidth=1.1)

            ax.set_title(pretty_label(metric), fontsize=8.5, fontweight="bold")
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)
            for spine in ("left", "top"):
                ax.spines[spine].set_color(FAMILY_COLORS[family])
            ax.spines["left"].set_linewidth(2.6)
            ax.spines["top"].set_linewidth(1.0)

        y_position = 1.0 - (row_index + 0.5) / len(METRIC_FAMILIES)
        fig.text(0.008, y_position, family, rotation=90, va="center", ha="left", fontsize=11, fontweight="bold", color=FAMILY_COLORS[family])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:len(cities)], labels[:len(cities)], loc="lower center", ncol=len(cities), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Master Panel: Climate Drivers and Response Families", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.98])
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
