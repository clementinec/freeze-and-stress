#!/usr/bin/env python3
"""Generate Fig. 3: driver-block fingerprints and attribution."""

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
    BLOCK_COLOR,
    BLOCK_DECOMP_CSV,
    BLOCK_LABEL,
    CLIMATE_CITY,
    CLIMATE_LABEL,
    CLIMATE_ORDER,
    CLIMATE_SCREENING_CSV,
    FIGURES_DIR,
    REGISTRY_CSV,
    SUMMARY_CSV,
    ensure_figures_dir,
    first_trigger_keys,
    short_metric_label,
)


def build_driver_fingerprint(climate_screening: pd.DataFrame) -> pd.DataFrame:
    rows = []
    drivers = climate_screening[climate_screening["role"] == "driver"].copy()
    for climate in CLIMATE_ORDER:
        for block in BLOCK_LABEL:
            block_rows = drivers[(drivers["climate_type"] == climate) & (drivers["subfamily"] == block)]
            rows.append(
                {
                    "climate_type": climate,
                    "city": CLIMATE_CITY[climate],
                    "driver_block": block,
                    "driver_block_label": BLOCK_LABEL[block].replace("\n", " "),
                    "mean_abs_standardized_change": pd.to_numeric(
                        block_rows["mean_abs_standardized_change"], errors="coerce"
                    ).mean(),
                    "mean_abs_temporal_slope": pd.to_numeric(
                        block_rows["mean_abs_temporal_slope"], errors="coerce"
                    ).mean(),
                    "mean_volatility_change": pd.to_numeric(
                        block_rows["mean_volatility_change"], errors="coerce"
                    ).mean(),
                }
            )
    return pd.DataFrame(rows)


def build_data(
    summary: pd.DataFrame,
    registry: pd.DataFrame,
    climate_screening: pd.DataFrame,
    block_decomp: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fingerprint = build_driver_fingerprint(climate_screening)
    label_to_metric = dict(zip(registry["label"], registry["metric"]))
    sentinel_metrics: list[str] = []
    for _, row in summary.iterrows():
        sentinel_metrics.extend(sorted(first_trigger_keys(row, label_to_metric)))
    sentinel_metrics = sorted(set(sentinel_metrics))
    priority = [
        "annual_edh_c_h",
        "maximum_daily_edh_k_h",
        "daily_edh_exceedance_days",
        "cooling_setpoint_not_met_occupied_hours",
        "heating_unmet_hours",
    ]
    metric_order = [metric for metric in priority if metric in set(sentinel_metrics)]
    metric_labels = dict(zip(registry["metric"], registry["label"]))

    rows = []
    for metric in metric_order:
        for climate in CLIMATE_ORDER:
            decomp = block_decomp[(block_decomp["climate_type"] == climate) & (block_decomp["metric"] == metric)]
            if decomp.empty:
                rows.append(
                    {
                        "metric": metric,
                        "label": metric_labels.get(metric, metric),
                        "climate_type": climate,
                        "city": CLIMATE_CITY[climate],
                        "block_dominant": "",
                        "block_dominant_share": np.nan,
                        "block_decomposition_r2": np.nan,
                    }
                )
            else:
                source = decomp.iloc[0]
                rows.append(
                    {
                        "metric": metric,
                        "label": metric_labels.get(metric, metric),
                        "climate_type": climate,
                        "city": CLIMATE_CITY[climate],
                        "block_dominant": source["block_dominant"],
                        "block_dominant_share": source["block_dominant_share"],
                        "block_decomposition_r2": source["block_decomposition_r2"],
                        **{
                            f"share_{block}": source.get(f"block_beta2_share_{block}", np.nan)
                            for block in BLOCK_LABEL
                        },
                    }
                )
    return fingerprint, pd.DataFrame(rows)


def plot(fingerprint: pd.DataFrame, attribution: pd.DataFrame) -> None:
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(15.2, 6.5),
        gridspec_kw={"width_ratios": [0.95, 1.35]},
    )

    fp_matrix = np.full((len(CLIMATE_ORDER), len(BLOCK_LABEL)), np.nan)
    for i, climate in enumerate(CLIMATE_ORDER):
        for j, block in enumerate(BLOCK_LABEL):
            row = fingerprint[(fingerprint["climate_type"] == climate) & (fingerprint["driver_block"] == block)]
            if not row.empty:
                fp_matrix[i, j] = float(row.iloc[0]["mean_abs_standardized_change"])
    fp_norm = mcolors.Normalize(vmin=np.nanmin(fp_matrix), vmax=np.nanmax(fp_matrix))
    fp = ax0.imshow(fp_matrix, cmap="Blues", norm=fp_norm, aspect="auto")
    for i in range(fp_matrix.shape[0]):
        for j in range(fp_matrix.shape[1]):
            value = fp_matrix[i, j]
            label = "NA" if np.isnan(value) else f"{value:.1f}"
            ax0.text(j, i, label, ha="center", va="center", fontsize=8, color="#10243a")
    ax0.set_title("(a) Driver-block stress fingerprints", loc="left", fontsize=11, fontweight="bold")
    ax0.set_xticks(np.arange(len(BLOCK_LABEL)))
    ax0.set_xticklabels([BLOCK_LABEL[b] for b in BLOCK_LABEL], rotation=35, ha="right", fontsize=8)
    ax0.set_yticks(np.arange(len(CLIMATE_ORDER)))
    ax0.set_yticklabels([CLIMATE_LABEL[c] for c in CLIMATE_ORDER], fontsize=9)
    ax0.tick_params(length=0)
    cbar = fig.colorbar(fp, ax=ax0, fraction=0.046, pad=0.03)
    cbar.set_label("Mean absolute standardized change", fontsize=8)

    metrics = list(OrderedDict.fromkeys(attribution["metric"]))
    block_to_id = {block: idx for idx, block in enumerate(BLOCK_LABEL)}
    attr_matrix = np.full((len(metrics), len(CLIMATE_ORDER)), np.nan)
    share_matrix = np.full_like(attr_matrix, np.nan, dtype=float)
    r2_matrix = np.full_like(attr_matrix, np.nan, dtype=float)
    for i, metric in enumerate(metrics):
        for j, climate in enumerate(CLIMATE_ORDER):
            row = attribution[(attribution["metric"] == metric) & (attribution["climate_type"] == climate)]
            if row.empty:
                continue
            dominant = row.iloc[0]["block_dominant"]
            if dominant in block_to_id:
                attr_matrix[i, j] = block_to_id[dominant]
            share_matrix[i, j] = pd.to_numeric(row.iloc[0]["block_dominant_share"], errors="coerce")
            r2_matrix[i, j] = pd.to_numeric(row.iloc[0]["block_decomposition_r2"], errors="coerce")

    attr_cmap = mcolors.ListedColormap([BLOCK_COLOR[block] for block in BLOCK_LABEL])
    attr_cmap.set_bad("#eeeeee")
    attr_norm = mcolors.BoundaryNorm(np.arange(-0.5, len(BLOCK_LABEL) + 0.5, 1), attr_cmap.N)
    ax1.imshow(attr_matrix, cmap=attr_cmap, norm=attr_norm, aspect="auto")
    for i in range(attr_matrix.shape[0]):
        for j in range(attr_matrix.shape[1]):
            if np.isnan(attr_matrix[i, j]):
                text = "NA"
                color = "#777777"
            else:
                text = f"{share_matrix[i, j]:.0%}\nR2 {r2_matrix[i, j]:.2f}"
                color = "white"
            ax1.text(j, i, text, ha="center", va="center", fontsize=7.3, color=color, fontweight="bold")
    label_by_metric = dict(zip(attribution["metric"], attribution["label"]))
    ax1.set_title("(b) Dominant driver block for sentinel-response metrics", loc="left", fontsize=11, fontweight="bold")
    ax1.set_xticks(np.arange(len(CLIMATE_ORDER)))
    ax1.set_xticklabels([CLIMATE_LABEL[c] for c in CLIMATE_ORDER], fontsize=8)
    ax1.set_yticks(np.arange(len(metrics)))
    ax1.set_yticklabels([short_metric_label(label_by_metric[m]).replace("\n", " ") for m in metrics], fontsize=8)
    ax1.tick_params(length=0)

    legend_handles = [
        patches.Patch(facecolor=BLOCK_COLOR[block], edgecolor="none", label=BLOCK_LABEL[block].replace("\n", " "))
        for block in BLOCK_LABEL
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle("Fig. 3 | Same response label, different climate-driver structure", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(FIGURES_DIR / "fig3_driver_block_decomposition.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ensure_figures_dir()
    summary = pd.read_csv(SUMMARY_CSV)
    registry = pd.read_csv(REGISTRY_CSV)
    climate_screening = pd.read_csv(CLIMATE_SCREENING_CSV)
    block_decomp = pd.read_csv(BLOCK_DECOMP_CSV)
    fingerprint, attribution = build_data(summary, registry, climate_screening, block_decomp)
    fingerprint.to_csv(FIGURES_DIR / "fig3_driver_fingerprint_data.csv", index=False)
    attribution.to_csv(FIGURES_DIR / "fig3_driver_block_decomposition_data.csv", index=False)
    plot(fingerprint, attribution)
    print(f"Saved -> {FIGURES_DIR / 'fig3_driver_block_decomposition.png'}")


if __name__ == "__main__":
    main()
