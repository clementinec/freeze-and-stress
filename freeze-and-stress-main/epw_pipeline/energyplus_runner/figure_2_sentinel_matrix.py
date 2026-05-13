#!/usr/bin/env python3
"""Generate Fig. 2: climate-specific sentinel matrix."""

from __future__ import annotations

from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_paper_common import (
    CLIMATE_CITY,
    CLIMATE_LABEL,
    CLIMATE_ORDER,
    FIGURES_DIR,
    REGISTRY_CSV,
    SCREENING_CSV,
    SUMMARY_CSV,
    SUBFAMILY_COLOR,
    SUBFAMILY_LABEL,
    ensure_figures_dir,
    first_trigger_keys,
    metric_prefix,
    parse_year,
    response_metrics,
    short_metric_label,
)


def build_data(summary: pd.DataFrame, registry: pd.DataFrame, screening: pd.DataFrame) -> pd.DataFrame:
    metrics = response_metrics(registry, screening)
    label_to_metric = dict(zip(registry["label"], registry["metric"]))
    rows: list[dict[str, object]] = []
    for climate in CLIMATE_ORDER:
        city = CLIMATE_CITY[climate]
        source = summary[summary["city"] == city].iloc[0]
        trigger_keys = first_trigger_keys(source, label_to_metric)
        for metric_order, metric_row in metrics.iterrows():
            prefix = metric_prefix(metric_row)
            break_value = source.get(f"{prefix}__break_year", "")
            sustained_value = source.get(f"{prefix}__sustained_failure_year", "")
            break_year = parse_year(break_value)
            rows.append(
                {
                    "climate_type": climate,
                    "city": city,
                    "climate_label": CLIMATE_LABEL[climate].replace("\n", " / "),
                    "metric_order": metric_order,
                    "subfamily": metric_row["subfamily"],
                    "subfamily_label": SUBFAMILY_LABEL.get(metric_row["subfamily"], metric_row["subfamily"]),
                    "metric": metric_row["metric"],
                    "label": metric_row["label"],
                    "unit": metric_row["unit"],
                    "failure_relevant": bool(metric_row["failure_relevant"]),
                    "has_standard_threshold": pd.notna(metric_row["threshold"]),
                    "break_year": break_year if break_year is not None else "Never",
                    "sustained_failure_year": parse_year(sustained_value) or "Never",
                    "is_first_trigger": metric_row["metric"] in trigger_keys,
                    "main_break_year": source.get("main_break_year"),
                    "main_sustained_failure_year": source.get("main_sustained_failure_year"),
                    "first_trigger_metric": source.get("first_trigger_metric"),
                }
            )
    return pd.DataFrame(rows)


def plot(data: pd.DataFrame, registry: pd.DataFrame, screening: pd.DataFrame) -> None:
    metric_rows = response_metrics(registry, screening)
    matrix = np.full((len(CLIMATE_ORDER), len(metric_rows)), np.nan)
    sentinel = np.zeros_like(matrix, dtype=bool)
    for i, climate in enumerate(CLIMATE_ORDER):
        for j, metric in enumerate(metric_rows["metric"]):
            row = data[(data["climate_type"] == climate) & (data["metric"] == metric)].iloc[0]
            year = parse_year(row["break_year"])
            if year is not None and bool(row["has_standard_threshold"]):
                matrix[i, j] = year
            sentinel[i, j] = bool(row["is_first_trigger"])

    fig, ax = plt.subplots(figsize=(14.8, 5.6))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#eeeeee")
    norm = mcolors.Normalize(vmin=2025, vmax=2050)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            year = matrix[i, j]
            row = data[(data["climate_type"] == CLIMATE_ORDER[i]) & (data["metric"] == metric_rows.iloc[j]["metric"])].iloc[0]
            if not bool(row["has_standard_threshold"]):
                text = "diag."
                color = "#8a8a8a"
            elif np.isnan(year):
                text = "Never"
                color = "#6b6b6b"
            else:
                text = str(int(year))
                color = "white" if year <= 2030 or year >= 2040 else "#222222"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.2, color=color, fontweight="bold")
            if sentinel[i, j]:
                ax.add_patch(
                    patches.Rectangle(
                        (j - 0.48, i - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        linewidth=2.6,
                        edgecolor="#111111",
                    )
                )

    ax.set_yticks(np.arange(len(CLIMATE_ORDER)))
    ax.set_yticklabels([CLIMATE_LABEL[c] for c in CLIMATE_ORDER], fontsize=9)
    ax.set_xticks(np.arange(len(metric_rows)))
    ax.set_xticklabels([short_metric_label(label) for label in metric_rows["label"]], rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="both", length=0)
    ax.set_title("Fig. 2 | Climate-specific sentinel matrix", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Building response metrics grouped by family")

    for j, subfamily in enumerate(metric_rows["subfamily"]):
        ax.add_patch(
            patches.Rectangle(
                (j - 0.5, -1.12),
                1.0,
                0.20,
                transform=ax.transData,
                clip_on=False,
                color=SUBFAMILY_COLOR.get(subfamily, "#999999"),
            )
        )

    family_groups: OrderedDict[str, list[int]] = OrderedDict()
    for idx, subfamily in enumerate(metric_rows["subfamily"]):
        family_groups.setdefault(str(subfamily), []).append(idx)
    for subfamily, indices in family_groups.items():
        start = min(indices) - 0.5
        end = max(indices) + 0.5
        if start > -0.5:
            ax.axvline(start, color="white", linewidth=2.2)
        ax.text(
            (start + end) / 2,
            -1.38,
            SUBFAMILY_LABEL.get(subfamily, subfamily),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=SUBFAMILY_COLOR.get(subfamily, "#333333"),
            transform=ax.transData,
            clip_on=False,
        )

    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.026,
        pad=0.018,
    )
    colorbar.set_label("First threshold breach year")
    ax.text(
        1.0,
        1.03,
        "Black outline = first-trigger sentinel metric(s); diag. = diagnostic metric without a standard threshold",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_sentinel_matrix.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ensure_figures_dir()
    summary = pd.read_csv(SUMMARY_CSV)
    registry = pd.read_csv(REGISTRY_CSV)
    screening = pd.read_csv(SCREENING_CSV)
    data = build_data(summary, registry, screening)
    data.to_csv(FIGURES_DIR / "fig2_sentinel_matrix_data.csv", index=False)
    plot(data, registry, screening)
    print(f"Saved -> {FIGURES_DIR / 'fig2_sentinel_matrix.png'}")


if __name__ == "__main__":
    main()
