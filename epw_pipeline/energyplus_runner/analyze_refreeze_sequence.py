#!/usr/bin/env python3
"""
ANALYZE: Visualize and summarize the adaptive re-freeze sequence.

Reads refreeze_sequence.csv produced by run_adaptive_refreeze.py and generates:
  1. Capacity evolution chart — cooling capacity gen0 → gen1 → gen2 per city.
  2. Survival timeline (Gantt) — colored segments per generation per (building, city).
  3. Unmet hours trajectory — 2025–2100 time series with re-freeze staircase.
  4. Summary table — capacities, % uplift, retrofit count per (building, city).

Usage:
    python analyze_refreeze_sequence.py \\
        --sequence refreeze_results/SCENARIO/refreeze_sequence.csv \\
        --output-dir refreeze_results/SCENARIO/plots/

Optional:
    --annual-metrics metric_exports/annual_metrics.csv   (enables unmet-hours trajectory)
    --dpi 150
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["figure.dpi"] = 100
matplotlib.rcParams["font.size"] = 9

GENERATION_COLORS = ["#4878CF", "#E24A33", "#8EBA42", "#988ED5", "#FBC15E", "#FFB5B8"]
BUILDING_ORDER = ["office", "apartment", "retail"]
CITY_ORDER = ["Miami", "Los_Angeles", "Montreal", "Toronto"]
THRESHOLD_LINES = [30, 100]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sequence(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Coerce numeric break_year; keep "Never"/"2100+" as strings
    df["break_year_num"] = pd.to_numeric(df["break_year"], errors="coerce")
    df["generation"] = df["generation"].astype(int)
    if "primary_cooling_capacity_w" in df.columns:
        df["primary_cooling_capacity_kw"] = pd.to_numeric(
            df["primary_cooling_capacity_w"], errors="coerce"
        ) / 1000.0
    else:
        df["primary_cooling_capacity_kw"] = np.nan
    return df


def load_annual_metrics(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Expect columns: building, city, year, scenario/generation, unmet_hours
    if "abups_occupied_cooling_not_met_hours" in df.columns:
        df = df.rename(columns={"abups_occupied_cooling_not_met_hours": "unmet_hours"})
    elif "facility_cooling_setpoint_not_met_occupied_time_total_hours" in df.columns:
        df = df.rename(
            columns={"facility_cooling_setpoint_not_met_occupied_time_total_hours": "unmet_hours"}
        )
    else:
        return None
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["unmet_hours"] = pd.to_numeric(df["unmet_hours"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Plot 1: Capacity evolution
# ---------------------------------------------------------------------------

def plot_capacity_evolution(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Grouped bar chart: cooling capacity at each generation, grouped by city.
    One subplot per building type.
    """
    cap_df = df.dropna(subset=["primary_cooling_capacity_kw"])
    if cap_df.empty:
        print("  [SKIP] No capacity data available for capacity evolution chart.")
        return

    buildings = [b for b in BUILDING_ORDER if b in cap_df["building"].unique()]
    max_gen = int(cap_df["generation"].max())
    n_gens = max_gen + 1
    width = 0.8 / n_gens

    fig, axes = plt.subplots(1, len(buildings), figsize=(5 * len(buildings), 5), sharey=False)
    if len(buildings) == 1:
        axes = [axes]

    for ax, building in zip(axes, buildings):
        bdf = cap_df[cap_df["building"] == building]
        cities = [c for c in CITY_ORDER if c in bdf["city"].unique()]
        x = np.arange(len(cities))

        for gen in range(n_gens):
            gdf = bdf[bdf["generation"] == gen]
            vals = [gdf[gdf["city"] == c]["primary_cooling_capacity_kw"].mean() for c in cities]
            offsets = x + (gen - n_gens / 2 + 0.5) * width
            color = GENERATION_COLORS[gen % len(GENERATION_COLORS)]
            label = "Gen 0 (TMY)" if gen == 0 else f"Gen {gen}"
            ax.bar(offsets, vals, width * 0.9, label=label, color=color, alpha=0.85)

        ax.set_title(f"{building.capitalize()}")
        ax.set_xticks(x)
        ax.set_xticklabels(cities, rotation=20, ha="right")
        ax.set_ylabel("Cooling capacity (kW)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle("HVAC Cooling Capacity by Generation", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_path = output_dir / "capacity_evolution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Survival timeline (Gantt)
# ---------------------------------------------------------------------------

def plot_survival_timeline(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Horizontal Gantt chart: each row = (building, city) pair.
    Colored segments show the operational lifespan per generation.
    Break years are marked with a vertical tick.
    """
    buildings = [b for b in BUILDING_ORDER if b in df["building"].unique()]
    cities = [c for c in CITY_ORDER if c in df["city"].unique()]

    row_labels: list[str] = []
    for building in buildings:
        for city in cities:
            if not df[(df["building"] == building) & (df["city"] == city)].empty:
                row_labels.append(f"{city}\n({building[:3]})")

    n_rows = len(row_labels)
    fig, ax = plt.subplots(figsize=(12, 0.6 * n_rows + 2))

    row_idx = 0
    legend_handles: list = []
    added_gen_labels: set = set()

    for building in buildings:
        for city in cities:
            pair_df = df[(df["building"] == building) & (df["city"] == city)].sort_values("generation")
            if pair_df.empty:
                continue

            y = n_rows - 1 - row_idx

            for _, row in pair_df.iterrows():
                gen = int(row["generation"])
                start = int(row["first_year_simulated"])
                break_yr = row["break_year_num"]
                end = int(break_yr) if not pd.isna(break_yr) else 2100

                color = GENERATION_COLORS[gen % len(GENERATION_COLORS)]
                ax.barh(y, end - start, left=start, height=0.6,
                        color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

                if gen not in added_gen_labels:
                    label = "Gen 0 (TMY)" if gen == 0 else f"Gen {gen}"
                    legend_handles.append(mpatches.Patch(color=color, label=label))
                    added_gen_labels.add(gen)

                if not pd.isna(break_yr):
                    ax.plot([int(break_yr), int(break_yr)], [y - 0.35, y + 0.35],
                            color="black", linewidth=1.5, zorder=5)

            row_idx += 1

    ax.set_xlim(2025, 2101)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels[::-1], fontsize=8)
    ax.set_xlabel("Year")
    ax.set_title("Adaptive Re-Freeze Survival Timeline", fontsize=11, fontweight="bold")
    ax.axvline(2025, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(2100, color="grey", linewidth=0.5, linestyle="--")
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    out_path = output_dir / "survival_timeline.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Unmet hours trajectory
# ---------------------------------------------------------------------------

def plot_unmet_hours_trajectory(
    seq_df: pd.DataFrame,
    annual_metrics: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    """
    Line chart per (building, city): unmet hours 2025–2100.
    Each generation's simulation period is shown as a separate line segment.
    Threshold lines at 30h and 100h.
    Requires annual_metrics with a 'generation' column or per-generation scenario column.
    """
    if annual_metrics is None:
        print("  [SKIP] No annual metrics provided — skipping unmet hours trajectory.")
        return

    # Check if 'generation' column exists; if not, try to infer from 'scenario'
    if "generation" not in annual_metrics.columns:
        print("  [SKIP] annual_metrics has no 'generation' column — trajectory skipped.")
        return

    buildings = [b for b in BUILDING_ORDER if b in seq_df["building"].unique()]
    cities = [c for c in CITY_ORDER if c in seq_df["city"].unique()]
    n_pairs = sum(
        1 for b in buildings for c in cities
        if not seq_df[(seq_df["building"] == b) & (seq_df["city"] == c)].empty
    )

    ncols = min(len(cities), 4)
    nrows = len(buildings)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharey=False)
    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]

    for row_i, building in enumerate(buildings):
        col_j = 0
        for city in cities:
            pair_seq = seq_df[
                (seq_df["building"] == building) & (seq_df["city"] == city)
            ].sort_values("generation")
            if pair_seq.empty:
                continue

            ax = axes[row_i][col_j] if ncols > 1 else axes[row_i]
            col_j += 1

            for _, gen_row in pair_seq.iterrows():
                gen = int(gen_row["generation"])
                start = int(gen_row["first_year_simulated"])
                break_yr = gen_row["break_year_num"]
                end = int(break_yr) if not pd.isna(break_yr) else 2101

                gen_mask = (
                    (annual_metrics["building"] == building)
                    & (annual_metrics["city"] == city)
                    & (annual_metrics["generation"] == gen)
                    & (annual_metrics["year"] >= start)
                    & (annual_metrics["year"] < end)
                )
                gdf = annual_metrics[gen_mask].sort_values("year")
                if gdf.empty:
                    continue

                color = GENERATION_COLORS[gen % len(GENERATION_COLORS)]
                linestyle = ["-", "--", ":", "-."][gen % 4]
                label = "Gen 0 (TMY)" if gen == 0 else f"Gen {gen}"
                ax.plot(gdf["year"], gdf["unmet_hours"], color=color,
                        linestyle=linestyle, linewidth=1.2, label=label)

                if not pd.isna(break_yr):
                    ax.axvline(int(break_yr), color=color, linewidth=0.8,
                               linestyle=":", alpha=0.7)

            for thresh, lstyle in zip(THRESHOLD_LINES, ["--", "-"]):
                ax.axhline(thresh, color="grey", linewidth=0.6, linestyle=lstyle, alpha=0.6)
                ax.text(2025.5, thresh + 2, f"{thresh}h", fontsize=6, color="grey")

            ax.set_title(f"{city} / {building[:3]}", fontsize=8)
            ax.set_xlim(2025, 2100)
            ax.set_xlabel("Year", fontsize=7)
            ax.set_ylabel("Unmet h/yr", fontsize=7)
            ax.legend(fontsize=6)
            ax.grid(linewidth=0.3, alpha=0.4)

    fig.suptitle("Occupied Cooling Unmet Hours — Adaptive Re-Freeze Trajectory",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_path = output_dir / "unmet_hours_trajectory.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Table: capacity uplift summary
# ---------------------------------------------------------------------------

def compute_capacity_uplift_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table: for each (building, city), show capacity at each
    generation and the % increase relative to gen 0.
    """
    if "primary_cooling_capacity_kw" not in df.columns:
        return pd.DataFrame()

    records = []
    for (building, city), grp in df.groupby(["building", "city"]):
        grp = grp.sort_values("generation")
        gen0_cap = grp[grp["generation"] == 0]["primary_cooling_capacity_kw"]
        gen0_cap_val = float(gen0_cap.iloc[0]) if not gen0_cap.empty else None

        row: dict = {"building": building, "city": city}
        for _, gentry in grp.iterrows():
            gen = int(gentry["generation"])
            cap = gentry["primary_cooling_capacity_kw"]
            if pd.isna(cap):
                cap = None
            row[f"gen{gen}_cap_kw"] = round(cap, 1) if cap is not None else None
            if cap is not None and gen0_cap_val and gen > 0:
                row[f"gen{gen}_uplift_pct"] = round(100.0 * (cap - gen0_cap_val) / gen0_cap_val, 1)
            row[f"gen{gen}_break_year"] = gentry["break_year"]

        row["n_retrofits_by_2100"] = int(grp[grp["generation"] > 0].shape[0])
        records.append(row)

    return pd.DataFrame(records)


def print_and_save_summary_table(df: pd.DataFrame, output_dir: Path) -> None:
    table = compute_capacity_uplift_table(df)
    if table.empty:
        print("  [SKIP] No capacity data for summary table.")
        return

    csv_path = output_dir / "capacity_uplift_summary.csv"
    table.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print()
    print(table.to_string(index=False))


# ---------------------------------------------------------------------------
# Plot 4: Unmet hours line chart from SQLite (self-contained)
# ---------------------------------------------------------------------------

_UNMET_QUERY = """
SELECT Value FROM TabularDataWithStrings
WHERE ReportName = 'AnnualBuildingUtilityPerformanceSummary'
  AND TableName = 'Comfort and Setpoint Not Met Summary'
  AND RowName = 'Time Setpoint Not Met During Occupied Cooling'
LIMIT 1
"""


def _read_unmet_sql(sql_path: Path) -> float | None:
    try:
        conn = sqlite3.connect(f"file:{sql_path}?mode=ro", uri=True, timeout=2)
        try:
            row = conn.execute(_UNMET_QUERY).fetchone()
            return float(row[0]) if row and row[0] is not None else None
        finally:
            conn.close()
    except Exception:
        return None


def load_unmet_hours_from_sim(sim_root: Path, seq_df: pd.DataFrame) -> dict:
    """
    Read per-year unmet hours from sim/<genN>/<building>/<city>/<year>/eplusout.sql.

    Returns dict keyed by (building, city) → list of (year, unmet_hours, gen).
    """
    result: dict = {}
    pairs = seq_df[["building", "city"]].drop_duplicates().values.tolist()

    for building, city in pairs:
        pair_seq = seq_df[
            (seq_df["building"] == building) & (seq_df["city"] == city)
        ].sort_values("generation")

        records = []
        for _, row in pair_seq.iterrows():
            gen = int(row["generation"])
            start = int(row["first_year_simulated"])
            break_yr = row["break_year_num"]
            end = int(break_yr) if not pd.isna(break_yr) else 2101

            gen_dir = sim_root / f"gen{gen}" / building / city
            if not gen_dir.exists():
                continue

            for year in range(start, end + 1):
                sql = gen_dir / str(year) / "eplusout.sql"
                if sql.exists():
                    unmet = _read_unmet_sql(sql)
                    if unmet is not None:
                        records.append((year, unmet, gen))

        if records:
            result[(building, city)] = records

    return result


def plot_unmet_hours_line(
    seq_df: pd.DataFrame,
    unmet_data: dict,
    output_dir: Path,
    threshold: float | None = None,
) -> None:
    """
    One subplot per (building, city): continuous unmet-hours line where solid
    segments = one sizing generation, dashed connectors = retrofit transitions.
    Legend is placed to the right of the plot area.
    """
    pairs = [(b, c) for (b, c) in unmet_data]
    if not pairs:
        print("  [SKIP] No unmet-hours data found in sim directories.")
        return

    n = len(pairs)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    # Extra width on the right for the shared legend
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols + 2.2, 4 * nrows), squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    # Shared legend handles (built once)
    legend_handles = []
    legend_built = False

    for idx, (building, city) in enumerate(pairs):
        ax = axes_flat[idx]
        records = unmet_data[(building, city)]
        df_yr = pd.DataFrame(records, columns=["year", "unmet_hours", "gen"])
        df_yr = df_yr.sort_values("year")

        pair_seq = seq_df[
            (seq_df["building"] == building) & (seq_df["city"] == city)
        ].sort_values("generation")

        # Build ordered list of (gen, last_year, first_year_of_next_gen)
        gens = df_yr["gen"].unique()
        gens_sorted = sorted(gens)

        # Per-gen endpoint lookup for connector lines
        gen_last: dict[int, tuple[int, float]] = {}   # gen → (last_year, last_unmet)
        gen_first: dict[int, tuple[int, float]] = {}  # gen → (first_year, first_unmet)

        for gen in gens_sorted:
            gdf = df_yr[df_yr["gen"] == gen].sort_values("year")
            if gdf.empty:
                continue
            color = GENERATION_COLORS[gen % len(GENERATION_COLORS)]
            ax.plot(gdf["year"], gdf["unmet_hours"],
                    color=color, linewidth=1.6, zorder=3, solid_capstyle="round")
            ax.scatter(gdf["year"], gdf["unmet_hours"],
                       color=color, s=14, zorder=4)
            gen_last[gen] = (int(gdf["year"].iloc[-1]), float(gdf["unmet_hours"].iloc[-1]))
            gen_first[gen] = (int(gdf["year"].iloc[0]), float(gdf["unmet_hours"].iloc[0]))

        # Dashed connectors between consecutive gens at retrofit points
        for i, gen in enumerate(gens_sorted[:-1]):
            next_gen = gens_sorted[i + 1]
            if gen in gen_last and next_gen in gen_first:
                x0, y0 = gen_last[gen]
                x1, y1 = gen_first[next_gen]
                ax.plot([x0, x1], [y0, y1], color="grey",
                        linewidth=1.2, linestyle="--", alpha=0.55, zorder=2)

        # Subtle vertical markers at retrofit (break) years
        for _, row in pair_seq.iterrows():
            by = row["break_year_num"]
            if pd.isna(by):
                continue
            by = int(by)
            ax.axvline(by, color="grey", linewidth=0.6, linestyle=":", alpha=0.4, zorder=1)
            ax.text(by + 0.4, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] > 0 else 10,
                    str(by), fontsize=6, color="grey",
                    va="top", ha="left", rotation=90, alpha=0.6)

        # Threshold line
        if threshold is not None:
            tline = ax.axhline(threshold, color="red", linewidth=1.0,
                               linestyle="--", alpha=0.55, zorder=2)
            ax.text(2026, threshold + ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] > 0 else threshold + 2,
                    f"{threshold:.0f} h threshold",
                    fontsize=6.5, color="red", alpha=0.7)

        ax.set_xlim(2025, 2100)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{city} / {building}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel("Unmet cooling hours / yr", fontsize=8)
        ax.grid(linewidth=0.3, alpha=0.4)

        # Build shared legend once from the first subplot
        if not legend_built:
            from matplotlib.lines import Line2D
            legend_handles.append(
                Line2D([0], [0], color="grey", linewidth=1.2,
                       linestyle="--", alpha=0.7, label="Retrofit transition")
            )
            if threshold is not None:
                legend_handles.append(
                    Line2D([0], [0], color="red", linewidth=1.0,
                           linestyle="--", alpha=0.7, label=f"Threshold ({threshold:.0f} h)")
                )
            legend_built = True

    # Hide unused subplots
    for idx in range(len(pairs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Occupied Cooling Unmet Hours — Adaptive Re-Freeze",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.84, 1])
    fig.legend(handles=legend_handles, loc="center right",
               bbox_to_anchor=(0.99, 0.5), fontsize=8,
               title="Sizing period", title_fontsize=8,
               framealpha=0.9, edgecolor="lightgrey")

    out_path = output_dir / "unmet_hours_line.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Retrofit frequency summary
# ---------------------------------------------------------------------------

def print_retrofit_frequency(df: pd.DataFrame) -> None:
    freq = (
        df[df["generation"] > 0]
        .groupby(["building", "city"])
        .size()
        .reset_index(name="n_retrofits")
    )
    if freq.empty:
        print("  No retrofit events detected.")
        return
    print("\nRetrofit events per (building, city):")
    print(freq.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="ANALYZE: Visualize adaptive re-freeze sequence"
    )
    parser.add_argument("--sequence", type=Path, required=True,
                        help="Path to refreeze_sequence.csv")
    parser.add_argument("--sim-root", type=Path, default=None,
                        help="Root sim/ directory (default: <sequence_dir>/sim)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold line to draw on unmet-hours plot")
    parser.add_argument("--annual-metrics", type=Path, default=None,
                        help="Path to annual_metrics.csv with 'generation' column (optional)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for plots (default: same dir as sequence CSV)")
    parser.add_argument("--dpi", type=int, default=150, help="Plot DPI (default: 150)")
    args = parser.parse_args()

    if not args.sequence.exists():
        print(f"ERROR: sequence CSV not found: {args.sequence}")
        return 1

    output_dir = args.output_dir if args.output_dir else args.sequence.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["figure.dpi"] = args.dpi

    print(f"Loading: {args.sequence}")
    seq_df = load_sequence(args.sequence)
    print(f"  {len(seq_df)} rows, {seq_df['building'].nunique()} buildings, "
          f"{seq_df['city'].nunique()} cities, {seq_df['generation'].max()+1} max generations")

    annual_metrics = None
    if args.annual_metrics:
        print(f"Loading: {args.annual_metrics}")
        annual_metrics = load_annual_metrics(args.annual_metrics)
        if annual_metrics is None:
            print("  WARNING: Could not parse annual metrics — trajectory plot skipped.")

    print("\nGenerating plots...")

    plot_capacity_evolution(seq_df, output_dir)
    plot_survival_timeline(seq_df, output_dir)
    plot_unmet_hours_trajectory(seq_df, annual_metrics, output_dir)

    # Unmet hours line chart from SQLite (self-contained, no external CSV needed)
    sim_root = args.sim_root if args.sim_root else args.sequence.parent / "sim"
    if sim_root.exists():
        print(f"  Reading per-year unmet hours from: {sim_root}")
        unmet_data = load_unmet_hours_from_sim(sim_root, seq_df)
        plot_unmet_hours_line(seq_df, unmet_data, output_dir, threshold=args.threshold)
    else:
        print(f"  [SKIP] sim/ directory not found: {sim_root}")

    print("\nSummary table:")
    print_and_save_summary_table(seq_df, output_dir)
    print_retrofit_frequency(seq_df)

    print(f"\nAll outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
