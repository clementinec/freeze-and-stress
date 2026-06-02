#!/usr/bin/env python3
"""
ANALYZE: Visualize and summarize the adaptive re-freeze sequence.

Reads refreeze_sequence.csv produced by run_adaptive_refreeze.py and generates:
  1. Capacity evolution chart — cooling capacity gen0 → gen1 → gen2 per city.
  2. Survival timeline (Gantt) — colored segments per generation per (building, city).
  3. Trigger metric panel — screened break-year metrics for each (building, city).
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
import csv
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
CITY_ORDER = ["Miami", "Los_Angeles", "Phoenix", "Vancouver", "Montreal", "Toronto"]
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


def load_trigger_metrics(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "generation" in df.columns:
        df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def load_screened_metric_specs(metric_res_root: Path, scenario: str) -> list[dict[str, object]]:
    screening_csv = metric_res_root / scenario / "paper_metric_screening.csv"
    if not screening_csv.exists():
        return []

    specs: list[dict[str, object]] = []
    with screening_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("role") != "response":
                continue
            if str(row.get("screening_selected_for_analysis", "")).strip().lower() != "yes":
                continue
            threshold = row.get("threshold")
            if threshold in (None, ""):
                continue
            try:
                threshold_value = float(threshold)
            except ValueError:
                continue
            specs.append(
                {
                    "metric": str(row.get("metric", "")).strip(),
                    "label": str(row.get("label", row.get("metric", ""))).strip(),
                    "threshold": threshold_value,
                    "threshold_direction": str(row.get("threshold_direction", "above")).strip() or "above",
                    "subfamily": str(row.get("subfamily", "")).strip(),
                }
            )
    return specs


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
# Plot 3: Unmet hours trajectory from annual_metrics.csv
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
            if "first_trigger_metric" in grp.columns and pd.notna(gentry.get("first_trigger_metric")):
                row[f"gen{gen}_trigger_metric"] = gentry.get("first_trigger_metric")
            if "trigger_subfamily" in grp.columns and pd.notna(gentry.get("trigger_subfamily")):
                row[f"gen{gen}_trigger_subfamily"] = gentry.get("trigger_subfamily")

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

def plot_trigger_metric_panel(
    seq_df: pd.DataFrame,
    trigger_metrics: pd.DataFrame | None,
    metric_specs: list[dict[str, object]],
    output_dir: Path,
) -> None:
    if trigger_metrics is None or trigger_metrics.empty:
        print("  [SKIP] No annual trigger metrics provided — skipping trigger metric plots.")
        return
    if not metric_specs:
        print("  [SKIP] No screened thresholded metric specs available — skipping trigger metric plots.")
        return

    available_specs = [spec for spec in metric_specs if spec["metric"] in trigger_metrics.columns]
    if not available_specs:
        print("  [SKIP] Trigger metrics CSV has no screened metric columns to plot.")
        return

    pairs = seq_df[["building", "city"]].drop_duplicates().values.tolist()
    for building, city in pairs:
        pair_seq = seq_df[
            (seq_df["building"] == building) & (seq_df["city"] == city)
        ].sort_values("generation")
        pair_trigger = trigger_metrics[
            (trigger_metrics["building"] == building) & (trigger_metrics["city"] == city)
        ].copy()
        if pair_trigger.empty:
            continue

        pair_trigger["generation"] = pd.to_numeric(pair_trigger["generation"], errors="coerce")
        pair_trigger["year"] = pd.to_numeric(pair_trigger["year"], errors="coerce")
        pair_trigger = pair_trigger.sort_values(["generation", "year"])

        fig, axes = plt.subplots(1, len(available_specs), figsize=(5.2 * len(available_specs), 4), squeeze=False)
        axes_row = list(axes[0])

        for ax, spec in zip(axes_row, available_specs):
            metric = str(spec["metric"])
            label = str(spec["label"])
            threshold = float(spec["threshold"])
            pair_trigger[metric] = pd.to_numeric(pair_trigger[metric], errors="coerce")

            for _, gen_row in pair_seq.iterrows():
                gen = int(gen_row["generation"])
                gdf = pair_trigger[pair_trigger["generation"] == gen].dropna(subset=[metric])
                if gdf.empty:
                    continue
                color = GENERATION_COLORS[gen % len(GENERATION_COLORS)]
                ax.plot(
                    gdf["year"],
                    gdf[metric],
                    color=color,
                    linewidth=1.5,
                    marker="o",
                    markersize=3.5,
                )
                triggered = gdf[gdf["break_triggered"].astype(str).str.lower() == "yes"]
                if not triggered.empty:
                    ax.scatter(triggered["year"], triggered[metric], color=color, s=24, zorder=4)

            ax.axhline(threshold, color="red", linewidth=1.0, linestyle="--", alpha=0.55)
            ax.text(2026, threshold, f"{threshold:g}", fontsize=6.5, color="red", alpha=0.7, va="bottom")

            for _, gen_row in pair_seq.iterrows():
                by = gen_row["break_year_num"]
                if not pd.isna(by):
                    ax.axvline(int(by), color="grey", linewidth=0.6, linestyle=":", alpha=0.4)

            ax.set_title(label, fontsize=9, fontweight="bold")
            ax.set_xlabel("Year", fontsize=8)
            ax.set_ylabel(label, fontsize=8)
            ax.set_xlim(2025, 2100)
            ax.grid(linewidth=0.3, alpha=0.4)

        fig.suptitle(f"{city} / {building} — Adaptive Re-Freeze Trigger Metrics", fontsize=12, fontweight="bold")
        fig.tight_layout()
        out_path = output_dir / f"trigger_metrics_{building}_{city}.png"
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


def print_trigger_metric_summary(trigger_metrics: pd.DataFrame | None) -> None:
    if trigger_metrics is None or trigger_metrics.empty:
        return
    if "break_triggered" not in trigger_metrics.columns:
        return

    triggered = trigger_metrics[
        trigger_metrics["break_triggered"].astype(str).str.lower() == "yes"
    ].copy()
    if triggered.empty:
        print("\nNo triggered break rows found in annual trigger metrics.")
        return

    cols = [
        "all_trigger_metric_keys",
        "all_trigger_metrics",
        "all_trigger_subfamilies",
        "first_trigger_metric_key",
        "first_trigger_metric",
        "trigger_subfamily",
    ]
    for col in cols:
        if col not in triggered.columns:
            triggered[col] = ""

    summary = (
        triggered.groupby(cols, dropna=False)
        .size()
        .reset_index(name="n_breaks")
        .sort_values(["n_breaks", "all_trigger_metric_keys", "first_trigger_metric_key"], ascending=[False, True, True])
    )
    print("\nTrigger metric summary:")
    print(summary.to_string(index=False))


def print_generation_cap_notes(seq_df: pd.DataFrame, trigger_metrics: pd.DataFrame | None) -> None:
    if trigger_metrics is None or trigger_metrics.empty or "break_triggered" not in trigger_metrics.columns:
        return

    notes: list[str] = []
    for (building, city), grp in seq_df.groupby(["building", "city"]):
        final_gen = int(grp["generation"].max())
        final_entry = grp[grp["generation"] == final_gen].iloc[0]
        if str(final_entry["break_year"]) != "Never":
            continue
        pair_trigger = trigger_metrics[
            (trigger_metrics["building"] == building)
            & (trigger_metrics["city"] == city)
            & (pd.to_numeric(trigger_metrics["generation"], errors="coerce") == final_gen)
            & (trigger_metrics["break_triggered"].astype(str).str.lower() == "yes")
        ]
        if pair_trigger.empty:
            continue
        first_year = int(pd.to_numeric(pair_trigger["year"], errors="coerce").min())
        trigger_keys = str(
            pair_trigger.iloc[0].get("all_trigger_metric_keys", "")
            or pair_trigger.iloc[0].get("first_trigger_metric_key", "")
            or pair_trigger.iloc[0].get("first_trigger_metric", "")
        )
        notes.append(
            f"{building}/{city}: final generation gen{final_gen} still breached from "
            f"{first_year} onward via {trigger_keys}"
        )

    if notes:
        print("\nGeneration-cap notes:")
        for note in notes:
            print(note)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="ANALYZE: Visualize adaptive re-freeze sequence"
    )
    parser.add_argument("--sequence", type=Path, required=True,
                        help="Path to refreeze_sequence.csv")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold line to draw on legacy unmet-hours trajectory plot")
    parser.add_argument("--annual-metrics", type=Path, default=None,
                        help="Path to annual_metrics.csv with 'generation' column (optional)")
    parser.add_argument("--trigger-metrics", type=Path, default=None,
                        help="Path to annual_trigger_metrics.csv (default: <sequence_dir>/annual_trigger_metrics.csv)")
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
    script_dir = Path(__file__).resolve().parent

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

    trigger_metrics_path = args.trigger_metrics if args.trigger_metrics else args.sequence.parent / "annual_trigger_metrics.csv"
    trigger_metrics = None
    if trigger_metrics_path.exists():
        print(f"Loading: {trigger_metrics_path}")
        trigger_metrics = load_trigger_metrics(trigger_metrics_path)
        if trigger_metrics is None:
            print("  WARNING: Could not parse annual trigger metrics.")

    scenarios = sorted({str(value) for value in seq_df["scenario"].dropna().unique()}) if "scenario" in seq_df.columns else []
    metric_specs: list[dict[str, object]] = []
    if len(scenarios) == 1:
        metric_specs = load_screened_metric_specs(script_dir / "metric_res", scenarios[0])

    print("\nGenerating plots...")

    plot_capacity_evolution(seq_df, output_dir)
    plot_survival_timeline(seq_df, output_dir)
    plot_unmet_hours_trajectory(seq_df, annual_metrics, output_dir)
    plot_trigger_metric_panel(seq_df, trigger_metrics, metric_specs, output_dir)

    stale_unmet_plot = output_dir / "unmet_hours_line.png"
    if stale_unmet_plot.exists():
        stale_unmet_plot.unlink()
        print(f"  Removed stale plot: {stale_unmet_plot}")

    print("\nSummary table:")
    print_and_save_summary_table(seq_df, output_dir)
    print_retrofit_frequency(seq_df)
    print_trigger_metric_summary(trigger_metrics)
    print_generation_cap_notes(seq_df, trigger_metrics)

    print(f"\nAll outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
