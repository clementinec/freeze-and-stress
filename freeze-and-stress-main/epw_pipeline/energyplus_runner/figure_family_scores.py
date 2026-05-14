#!/usr/bin/env python3
"""Plot decadal family-score outputs.

Two figures are produced:
- Main figure: habitability and operational-adequacy score trajectories.
- Supplementary heatmap: all family scores, including cautious/secondary overlays.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_paper_common import CLIMATE_CITY, CLIMATE_LABEL, CLIMATE_ORDER, FIGURES_DIR, SUMMARY_CSV, ensure_figures_dir


matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
    }
)


SCRIPT_DIR = FIGURES_DIR.parent.parent
FAMILY_DECADAL_CSV = SCRIPT_DIR / "paper_family_scores_decadal.csv"
MAIN_OUTPUT = FIGURES_DIR / "family_scores_main_trajectories.png"
HEATMAP_OUTPUT = FIGURES_DIR / "family_scores_all_heatmap.png"
MAIN_DATA_OUTPUT = FIGURES_DIR / "family_scores_main_trajectories_data.csv"
HEATMAP_DATA_OUTPUT = FIGURES_DIR / "family_scores_all_heatmap_data.csv"

DECADE_ORDER = [
    "2025-2034",
    "2035-2044",
    "2045-2054",
    "2055-2064",
    "2065-2074",
    "2075-2084",
    "2085-2094",
    "2095-2100",
]

DECADE_MIDPOINT = {
    "2025-2034": 2029.5,
    "2035-2044": 2039.5,
    "2045-2054": 2049.5,
    "2055-2064": 2059.5,
    "2065-2074": 2069.5,
    "2075-2084": 2079.5,
    "2085-2094": 2089.5,
    "2095-2100": 2097.5,
}

FAMILY_LABEL = {
    "habitability": "Habitability",
    "operational_adequacy": "Operational adequacy",
    "durability_cautious": "Durability cautious",
    "energy_burden_secondary": "Energy burden secondary",
}

FAMILY_COLOR = {
    "habitability": "#b44b5f",
    "operational_adequacy": "#4f6d8a",
    "durability_cautious": "#5b8a72",
    "energy_burden_secondary": "#8f7a3d",
}

MAIN_FAMILIES = ["habitability", "operational_adequacy"]
ALL_FAMILIES = [
    "habitability",
    "operational_adequacy",
    "durability_cautious",
    "energy_burden_secondary",
]


def read_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    family = pd.read_csv(FAMILY_DECADAL_CSV)
    family["family_score_mean"] = pd.to_numeric(family["family_score_mean"], errors="coerce")
    family["family_score_sd"] = pd.to_numeric(family["family_score_sd"], errors="coerce")
    family["decade_midpoint"] = family["decade"].map(DECADE_MIDPOINT)
    family["decade_order"] = family["decade"].map({decade: idx for idx, decade in enumerate(DECADE_ORDER)})
    summary = pd.read_csv(SUMMARY_CSV)
    return family, summary


def plot_main_trajectories(family: pd.DataFrame, summary: pd.DataFrame) -> None:
    main = family[family["score_family"].isin(MAIN_FAMILIES)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.6), sharex=True, sharey=True)
    axes = axes.flatten()
    y_min = min(-1.2, float(main["family_score_mean"].min()) - 0.25)
    y_max = max(3.3, float(main["family_score_mean"].max()) + 0.35)

    for ax, climate in zip(axes, CLIMATE_ORDER):
        city = CLIMATE_CITY[climate]
        city_rows = main[main["city"] == city].sort_values("decade_order")
        for family_name in MAIN_FAMILIES:
            series = city_rows[city_rows["score_family"] == family_name].sort_values("decade_order")
            ax.plot(
                series["decade_midpoint"],
                series["family_score_mean"],
                color=FAMILY_COLOR[family_name],
                linewidth=2.6,
                marker="o",
                markersize=5.6,
                label=FAMILY_LABEL[family_name],
            )

        ax.axhline(0, color="#8d9aa3", linewidth=1.0)
        summary_row = summary[summary["city"] == city]
        if not summary_row.empty:
            break_year = str(summary_row.iloc[0].get("main_break_year", "Never"))
            sustained_year = str(summary_row.iloc[0].get("main_sustained_failure_year", "Never"))
            if break_year != "Never":
                ax.axvline(float(break_year), color="#263238", linewidth=1.25, linestyle="--", alpha=0.75)
                ax.text(float(break_year) + 0.8, y_max - 0.24, f"break {break_year}", fontsize=8.6, color="#263238")
            if sustained_year != "Never":
                ax.axvline(float(sustained_year), color="#111111", linewidth=1.4, linestyle=":", alpha=0.85)
                ax.text(float(sustained_year) + 0.8, y_max - 0.55, f"sust. {sustained_year}", fontsize=8.6, color="#111111")

        ax.set_title(CLIMATE_LABEL[climate].replace("\n", " / "), fontsize=11.5, fontweight="bold")
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(2025, 2100)
        ax.set_box_aspect(1)
        ax.grid(True, color="#e6eaed", linewidth=0.7)
        ax.tick_params(axis="both", labelsize=9.5)

    for ax in axes[3:]:
        ax.set_xlabel("Decade midpoint", fontsize=10.5)
    for ax in axes[::3]:
        ax.set_ylabel("Family score\n(z from 2025-2034)", fontsize=10.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.085, top=0.88, wspace=0.11, hspace=0.22)
    fig.savefig(MAIN_OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    main.to_csv(MAIN_DATA_OUTPUT, index=False)


def plot_all_heatmap(family: pd.DataFrame) -> None:
    rows = []
    for climate in CLIMATE_ORDER:
        city = CLIMATE_CITY[climate]
        for family_name in ALL_FAMILIES:
            rows.append((climate, city, family_name))

    matrix = np.full((len(rows), len(DECADE_ORDER)), np.nan)
    for row_idx, (climate, city, family_name) in enumerate(rows):
        for col_idx, decade in enumerate(DECADE_ORDER):
            match = family[
                (family["city"] == city)
                & (family["score_family"] == family_name)
                & (family["decade"] == decade)
            ]
            if not match.empty:
                matrix[row_idx, col_idx] = float(match.iloc[0]["family_score_mean"])

    vmax = float(np.nanpercentile(np.abs(matrix), 95))
    vmax = max(2.5, min(vmax, 8.0))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")

    fig = plt.figure(figsize=(12.4, 10.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 2.6, 8.7], wspace=0.02)
    bracket_ax = fig.add_subplot(gs[0, 0])
    label_ax = fig.add_subplot(gs[0, 1], sharey=bracket_ax)
    ax = fig.add_subplot(gs[0, 2], sharey=bracket_ax)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                text = "NA"
                color = "#777777"
            else:
                text = f"{value:.1f}"
                color = "white" if abs(value) > 0.55 * vmax else "#182026"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(np.arange(len(DECADE_ORDER)))
    ax.set_xticklabels([decade.replace("-", "\n") for decade in DECADE_ORDER], rotation=0, ha="center", fontsize=10)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    # fig.suptitle(
    #     "All family scores by decade, including cautious and secondary overlays",
    #     x=0.05,
    #     ha="left",
    #     fontsize=13,
    #     fontweight="bold",
    # )

    for boundary in range(len(ALL_FAMILIES), len(rows), len(ALL_FAMILIES)):
        ax.axhline(boundary - 0.5, color="white", linewidth=2.2)

    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(len(rows) - 0.5, -0.5)
    label_ax.axis("off")
    for row_idx, (_, _, family_name) in enumerate(rows):
        label_ax.text(
            0.98,
            row_idx,
            FAMILY_LABEL[family_name],
            ha="right",
            va="center",
            fontsize=10,
            color="black",
        )

    # Bracket-style group labels for each climate type.
    bracket_ax.set_xlim(0, 1)
    bracket_ax.set_ylim(len(rows) - 0.5, -0.5)
    bracket_ax.axis("off")
    bracket_x = 0.92
    tick_x = 1.04
    label_x = 0.82
    for group_index, climate in enumerate(CLIMATE_ORDER):
        start = group_index * len(ALL_FAMILIES) - 0.5
        end = (group_index + 1) * len(ALL_FAMILIES) - 0.5
        mid = (start + end) / 2
        bracket_ax.plot([bracket_x, bracket_x], [start, end], color="#263238", linewidth=1.25, clip_on=False)
        bracket_ax.plot([bracket_x, tick_x], [start, start], color="#263238", linewidth=1.25, clip_on=False)
        bracket_ax.plot([bracket_x, tick_x], [end, end], color="#263238", linewidth=1.25, clip_on=False)
        bracket_ax.text(
            label_x,
            mid,
            CLIMATE_LABEL[climate],
            ha="right",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="#263238",
            clip_on=False,
        )

    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.026, pad=0.018)
    colorbar.set_label("Decadal family score (z from 2025-2034)")
    fig.subplots_adjust(left=0.05, right=0.92, top=0.92, bottom=0.08)
    fig.savefig(HEATMAP_OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    heatmap_data = []
    for row_idx, (climate, city, family_name) in enumerate(rows):
        for col_idx, decade in enumerate(DECADE_ORDER):
            heatmap_data.append(
                {
                    "climate_type": climate,
                    "city": city,
                    "score_family": family_name,
                    "decade": decade,
                    "family_score_mean": matrix[row_idx, col_idx],
                }
            )
    pd.DataFrame(heatmap_data).to_csv(HEATMAP_DATA_OUTPUT, index=False)


def main() -> None:
    ensure_figures_dir()
    family, summary = read_data()
    plot_main_trajectories(family, summary)
    plot_all_heatmap(family)
    print(f"Saved -> {MAIN_OUTPUT}")
    print(f"Saved -> {HEATMAP_OUTPUT}")


if __name__ == "__main__":
    main()
