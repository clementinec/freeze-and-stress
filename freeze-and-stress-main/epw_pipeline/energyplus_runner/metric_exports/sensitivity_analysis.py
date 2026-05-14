#!/usr/bin/env python3
"""Sensitivity analysis: rank metrics by their trend against year (climate change signal)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "annual_metrics.csv"
OUTPUT_CSV = SCRIPT_DIR / "sensitivity_ranking.csv"
OUTPUT_HEATMAP = SCRIPT_DIR / "sensitivity_heatmap.png"

EXCLUDE_COLS = {
    "scenario", "building", "city", "year", "sql_path", "extracted_at",
    "interval_minutes", "time_rows", "representative_zone",
    "environment_period_index", "total_building_area_m2",
    "conditioned_building_area_m2", "unconditioned_building_area_m2",
}
EXCLUDE_SUFFIXES = ("_timestamp",)

TOP_N_HEATMAP = 15
TOP_N_TRENDS = 15


def is_numeric_metric(col: str) -> bool:
    if col in EXCLUDE_COLS:
        return False
    if any(col.endswith(s) for s in EXCLUDE_SUFFIXES):
        return False
    return True


def compute_sensitivity(group: pd.DataFrame, metric: str) -> dict | None:
    y = pd.to_numeric(group[metric], errors="coerce")
    mask = y.notna()
    x = group.loc[mask, "year"].values.astype(float)
    y = y[mask].values
    if len(x) < 5 or np.std(y) == 0:
        return None

    slope, intercept, r, p, se = stats.linregress(x, y)
    mean_val = np.mean(y)
    norm_slope = slope / abs(mean_val) if abs(mean_val) > 1e-12 else np.nan

    return {
        "slope_per_year": slope,
        "normalized_slope": norm_slope,
        "r2": r ** 2,
        "p_value": p,
        "mean": mean_val,
        "std": np.std(y),
    }

def plot_city_heatmap(ranking: pd.DataFrame, cities: list[str]) -> None:
    """One horizontal bar chart per city showing top 15 metrics by |normalized_slope|."""
    n_cities = len(cities)
    fig, axes = plt.subplots(1, n_cities, figsize=(5 * n_cities, 8), sharey=False)
    if n_cities == 1:
        axes = [axes]

    for ax, city in zip(axes, cities):
        city_rank = ranking[ranking["city"] == city].head(TOP_N_HEATMAP)
        if city_rank.empty:
            continue
        colors = ["#d73027" if v > 0 else "#4575b4" for v in city_rank["normalized_slope"]]
        ax.barh(range(len(city_rank)), city_rank["normalized_slope"], color=colors)
        ax.set_yticks(range(len(city_rank)))
        ax.set_yticklabels(city_rank["metric"], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Normalized slope", fontsize=9)
        ax.set_title(city, fontsize=11, fontweight="bold")
        ax.axvline(0, color="k", linewidth=0.5)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"Top {TOP_N_HEATMAP} Climate-Sensitive Metrics per City (office)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_HEATMAP, dpi=150)
    plt.close(fig)
    print(f"  Heatmap -> {OUTPUT_HEATMAP}")


def plot_city_trends(df: pd.DataFrame, ranking: pd.DataFrame, city: str) -> None:
    city_rank = ranking[ranking["city"] == city].head(TOP_N_TRENDS)
    if city_rank.empty:
        return
    city_df = df[df["city"] == city].sort_values("year")
    n = len(city_rank)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), sharex=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (_, row) in enumerate(city_rank.iterrows()):
        ax = axes_flat[i]
        metric = row["metric"]
        y = pd.to_numeric(city_df[metric], errors="coerce")
        color = "#d73027" if row["normalized_slope"] > 0 else "#4575b4"
        ax.plot(city_df["year"], y, "o-", markersize=2, linewidth=1, color=color)
        z = np.polyfit(city_df["year"].values.astype(float), y.values, 1)
        ax.plot(city_df["year"], np.polyval(z, city_df["year"]), "--", color="gray", linewidth=0.8)
        ax.set_title(f"{metric}\nR²={row['r2']:.3f}  slope={row['normalized_slope']:.4f}/yr", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Top {TOP_N_TRENDS} Sensitive Metrics — {city} (office)", fontsize=12)
    fig.tight_layout()
    out = SCRIPT_DIR / f"sensitivity_trends_{city}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Trends  -> {out}")

def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df = df[df["building"] == "office"].copy()
    print(f"Loaded {len(df)} rows, cities: {sorted(df['city'].unique())}")

    numeric_cols = [c for c in df.columns if is_numeric_metric(c)]
    cities = sorted(df["city"].unique())

    rows = []
    for city in cities:
        group = df[df["city"] == city]
        for col in numeric_cols:
            result = compute_sensitivity(group, col)
            if result is None:
                continue
            rows.append({"city": city, "metric": col, **result})

    ranking = pd.DataFrame(rows)
    ranking["abs_norm_slope"] = ranking["normalized_slope"].abs()
    ranking = ranking.sort_values("abs_norm_slope", ascending=False)

    try:
        ranking.to_csv(OUTPUT_CSV, index=False, float_format="%.6g")
    except PermissionError:
        alt = OUTPUT_CSV.with_name("sensitivity_ranking_new.csv")
        ranking.to_csv(alt, index=False, float_format="%.6g")
        print(f"  (!) Original CSV locked, wrote to {alt.name}")
    print(f"Ranking CSV -> {OUTPUT_CSV}  ({len(ranking)} rows)")

    plot_city_heatmap(ranking, cities)
    for city in cities:
        plot_city_trends(df, ranking, city)

    for city in cities:
        print(f"\n{'='*60}")
        print(f"  {city} — Top {TOP_N_TRENDS} by |normalized_slope|")
        print(f"{'='*60}")
        sub = ranking[ranking["city"] == city].head(TOP_N_TRENDS)
        print(sub[["metric", "slope_per_year", "normalized_slope", "r2", "p_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
