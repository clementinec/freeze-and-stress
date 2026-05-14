#!/usr/bin/env python3
"""Generate Fig. 1: climate-driver trajectories by Koppen prototype."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from figure_paper_common import CLIMATE_CITY, CLIMATE_LABEL, CLIMATE_ORDER, FIGURES_DIR, ensure_figures_dir


MASTER_PANEL_CSV = FIGURES_DIR.parent.parent / "paper_master_panel.csv"
OUTPUT_PNG = FIGURES_DIR / "fig1_climate_driver_trajectories.png"
OUTPUT_DATA = FIGURES_DIR / "fig1_climate_driver_trajectories_data.csv"

DRIVER_ORDER = [
    ("annual_mean_drybulb_c", "Mean dry-bulb"),
    ("cdd_18", "CDD"),
    ("hdd_18", "HDD"),
    ("maximum_wetbulb_c", "Max wet-bulb"),
    ("annual_ghi_kwh_m2", "GHI"),
    ("heatwave_days", "Heatwave days"),
]

COLORS = {
    "annual_mean_drybulb_c": "#b55239",
    "cdd_18": "#d17a22",
    "hdd_18": "#4f78a8",
    "maximum_wetbulb_c": "#7b559c",
    "annual_ghi_kwh_m2": "#d1a33f",
    "heatwave_days": "#4f9a68",
}


def build_data(master: pd.DataFrame) -> pd.DataFrame:
    drivers = master[(master["role"] == "driver") & (master["metric"].isin([metric for metric, _ in DRIVER_ORDER]))].copy()
    drivers["delta_std"] = pd.to_numeric(drivers["delta_std"], errors="coerce")
    return drivers[
        [
            "scenario",
            "building",
            "city",
            "climate_type",
            "year",
            "metric",
            "label",
            "metric_value",
            "baseline_value",
            "delta_abs",
            "delta_std",
        ]
    ]


def plot(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 7.2), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, climate in zip(axes, CLIMATE_ORDER):
        city = CLIMATE_CITY[climate]
        subset = data[data["city"] == city]
        for metric, label in DRIVER_ORDER:
            series = subset[subset["metric"] == metric].sort_values("year")
            if series.empty:
                continue
            ax.plot(
                series["year"],
                series["delta_std"],
                color=COLORS[metric],
                linewidth=1.8,
                alpha=0.92,
                label=label,
            )
        ax.axhline(0, color="#90a4ae", linewidth=0.8)
        ax.set_title(CLIMATE_LABEL[climate].replace("\n", " / "), fontsize=10, fontweight="bold")
        ax.grid(True, color="#e6eaed", linewidth=0.7)
        ax.set_xlim(2025, 2100)
    for ax in axes[3:]:
        ax.set_xlabel("Year")
    for ax in axes[::3]:
        ax.set_ylabel("Change from 2025-2034 baseline (z)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8.5)
    fig.suptitle("Fig. 1 | Climate-driver trajectories by representative Koppen type", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ensure_figures_dir()
    master = pd.read_csv(MASTER_PANEL_CSV)
    data = build_data(master)
    data.to_csv(OUTPUT_DATA, index=False)
    plot(data)
    print(f"Saved -> {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
