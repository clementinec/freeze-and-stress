#!/usr/bin/env python3
"""Paper-oriented overview figure for adaptive re-freeze results.

Creates a 3-panel composite figure:
  A. Six-city survival timeline for one building prototype.
  B. Year-by-year trigger metrics + cooling capacity for a representative city.
  C. Same as B for a second representative city.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from figure_paper_common import CLIMATE_CITY, CLIMATE_ORDER, ensure_figures_dir, parse_year


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCENARIO = "CORDEX_CMIP5_REMO2015_rcp85"
DEFAULT_BUILDING = "office"
GENERATION_COLORS = [
    "#b9c2cb",
    "#8fa6bf",
    "#6f8ba8",
    "#517293",
    "#38567a",
    "#203a56",
]
METRIC_STYLES = {
    "annual_edh_c_h": {"color": "#b44b5f", "label": "Annual EDH"},
    "cooling_setpoint_not_met_occupied_hours": {
        "color": "#4f6d8a",
        "label": "Cooling unmet occupied hours",
    },
}
CAPACITY_COLOR = "#c57b2a"


def load_thresholds(screening_csv: Path) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    with screening_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("role") != "response":
                continue
            if str(row.get("screening_selected_for_analysis", "")).strip().lower() != "yes":
                continue
            metric = str(row.get("metric", "")).strip()
            threshold = str(row.get("threshold", "")).strip()
            if metric and threshold:
                thresholds[metric] = float(threshold)
    return thresholds


def load_sequence(sequence_csv: Path, building: str) -> pd.DataFrame:
    df = pd.read_csv(sequence_csv)
    df = df[df["building"] == building].copy()
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce").astype(int)
    df["first_year_simulated"] = pd.to_numeric(df["first_year_simulated"], errors="coerce")
    df["break_year_num"] = pd.to_numeric(df["break_year"], errors="coerce")
    df["primary_cooling_capacity_kw"] = pd.to_numeric(
        df["primary_cooling_capacity_w"], errors="coerce"
    ) / 1000.0
    return df


def load_trigger_metrics(trigger_csv: Path, building: str) -> pd.DataFrame:
    df = pd.read_csv(trigger_csv)
    df = df[df["building"] == building].copy()
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce").astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    for metric in list(METRIC_STYLES) + ["heating_unmet_hours"]:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def climate_ordered_cities(seq_df: pd.DataFrame) -> list[str]:
    available = set(seq_df["city"].unique())
    ordered = [CLIMATE_CITY[code] for code in CLIMATE_ORDER if CLIMATE_CITY[code] in available]
    extras = sorted(available.difference(ordered))
    return ordered + extras


def format_city(city: str) -> str:
    return city.replace("_", " ")


def plot_timeline(ax: plt.Axes, seq_df: pd.DataFrame) -> None:
    cities = climate_ordered_cities(seq_df)
    legend_handles: list[mpatches.Patch] = []
    seen_gens: set[int] = set()

    for row_idx, city in enumerate(cities):
        pair_df = seq_df[seq_df["city"] == city].sort_values("generation")
        y = len(cities) - 1 - row_idx
        for _, row in pair_df.iterrows():
            gen = int(row["generation"])
            start = int(row["first_year_simulated"])
            end = parse_year(row["break_year"])
            span_end = end if end is not None else 2101
            color = GENERATION_COLORS[min(gen, len(GENERATION_COLORS) - 1)]
            ax.barh(
                y,
                span_end - start,
                left=start,
                height=0.62,
                color=color,
                edgecolor="white",
                linewidth=0.6,
            )
            if end is not None:
                ax.plot([end, end], [y - 0.36, y + 0.36], color="#1f1f1f", linewidth=1.0, zorder=4)

            if gen not in seen_gens and gen <= 4:
                label = "Gen 0 (TMY)" if gen == 0 else f"Gen {gen}"
                legend_handles.append(mpatches.Patch(color=color, label=label))
                seen_gens.add(gen)

        n_retrofits = max(int(pair_df["generation"].max()), 0)
        ax.text(2101.3, y, f"{n_retrofits}", va="center", ha="left", fontsize=8, color="#333333")

    ax.set_xlim(2025, 2106)
    ax.set_ylim(-0.8, len(cities) - 0.2)
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels([format_city(city) for city in cities[::-1]], fontsize=9)
    ax.set_xlabel("Year")
    ax.set_title("A  Six-city adaptive re-freeze timeline", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linewidth=0.35, alpha=0.45)
    ax.axvline(2025, color="#888888", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.axvline(2100, color="#888888", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.text(2101.3, len(cities) - 0.15, "Retrofits", fontsize=8, ha="left", va="bottom", color="#333333")
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=False)


def build_capacity_series(pair_seq: pd.DataFrame, pair_trigger: pd.DataFrame) -> pd.DataFrame:
    cap_map = (
        pair_seq.dropna(subset=["primary_cooling_capacity_kw"])
        .set_index("generation")["primary_cooling_capacity_kw"]
        .to_dict()
    )
    cap_df = pair_trigger[["generation", "year"]].copy()
    cap_df["capacity_kw"] = cap_df["generation"].map(cap_map)
    cap_df = cap_df.dropna(subset=["capacity_kw"]).sort_values(["year", "generation"])
    return cap_df


def plot_city_metric_panel(
    axes: list[plt.Axes],
    seq_df: pd.DataFrame,
    trigger_df: pd.DataFrame,
    thresholds: dict[str, float],
    city: str,
    panel_letter: str,
) -> None:
    pair_seq = seq_df[seq_df["city"] == city].sort_values("generation")
    pair_trigger = trigger_df[trigger_df["city"] == city].sort_values(["generation", "year"])
    break_years = [
        int(year)
        for year in pair_seq["break_year_num"].tolist()
        if pd.notna(year)
    ]

    for ax, metric in zip(axes[:2], METRIC_STYLES):
        style = METRIC_STYLES[metric]
        ax.plot(pair_trigger["year"], pair_trigger[metric], color=style["color"], linewidth=1.6, alpha=0.9)
        triggered = pair_trigger[pair_trigger["break_triggered"].astype(str).str.lower() == "yes"]
        if not triggered.empty:
            ax.scatter(
                triggered["year"],
                triggered[metric],
                s=28,
                color=style["color"],
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )

        threshold = thresholds.get(metric)
        if threshold is not None:
            ax.axhline(threshold, color=style["color"], linewidth=1.0, linestyle="--", alpha=0.75)
            ax.text(2025.5, threshold, f"threshold {threshold:g}", fontsize=7, va="bottom", color=style["color"])

        for break_year in break_years:
            ax.axvline(break_year, color="#999999", linewidth=0.7, linestyle=":", alpha=0.65)

        ax.set_xlim(2025, 2100)
        ax.set_ylabel(style["label"], color=style["color"], fontsize=8)
        ax.grid(linewidth=0.3, alpha=0.4)

    cap_ax = axes[2]
    cap_df = build_capacity_series(pair_seq, pair_trigger)
    if not cap_df.empty:
        cap_ax.step(cap_df["year"], cap_df["capacity_kw"], where="post", color=CAPACITY_COLOR, linewidth=1.8)
        cap_ax.scatter(cap_df["year"], cap_df["capacity_kw"], s=8, color=CAPACITY_COLOR, alpha=0.8)
    for break_year in break_years:
        cap_ax.axvline(break_year, color="#999999", linewidth=0.7, linestyle=":", alpha=0.65)
    cap_ax.set_xlim(2025, 2100)
    cap_ax.set_ylabel("Cooling capacity (kW)", color=CAPACITY_COLOR, fontsize=8)
    cap_ax.set_xlabel("Year")
    cap_ax.grid(linewidth=0.3, alpha=0.4)
    cap_ax.text(
        0.01,
        0.05,
        "Saved outputs do not include gen0 TMY capacity;\nstep line starts at first re-freeze generation.",
        transform=cap_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#5c5c5c",
    )

    axes[0].set_title(
        f"{panel_letter}  {format_city(city)}\nyearly triggers and capacity response",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)


def make_figure(
    seq_df: pd.DataFrame,
    trigger_df: pd.DataFrame,
    thresholds: dict[str, float],
    city_a: str,
    city_b: str,
    output_path: Path,
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
    fig = plt.figure(figsize=(14.5, 8.5), facecolor="white")
    gs = gridspec.GridSpec(
        3,
        3,
        width_ratios=[1.55, 1.0, 1.0],
        height_ratios=[1.0, 1.0, 0.9],
        wspace=0.28,
        hspace=0.16,
    )

    ax_timeline = fig.add_subplot(gs[:, 0])
    plot_timeline(ax_timeline, seq_df)

    axes_b = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    axes_c = [fig.add_subplot(gs[i, 2]) for i in range(3)]
    plot_city_metric_panel(axes_b, seq_df, trigger_df, thresholds, city_a, "B")
    plot_city_metric_panel(axes_c, seq_df, trigger_df, thresholds, city_b, "C")

    fig.suptitle(
        "Adaptive re-freeze outcomes for office buildings across six climate archetypes",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create paper-oriented adaptive re-freeze overview figure.")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--building", default=DEFAULT_BUILDING)
    parser.add_argument("--city-a", default="Los_Angeles")
    parser.add_argument("--city-b", default="Miami")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=SCRIPT_DIR / "refreeze_results",
        help="Root directory containing refreeze_results/<scenario>/...",
    )
    parser.add_argument(
        "--metric-res-root",
        type=Path,
        default=SCRIPT_DIR / "metric_res",
        help="Root directory containing metric_res/<scenario>/paper_metric_screening.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. PDF will be written alongside it.",
    )
    args = parser.parse_args()

    ensure_figures_dir()

    scenario_root = args.results_root / args.scenario
    sequence_csv = scenario_root / "refreeze_sequence.csv"
    trigger_csv = scenario_root / "annual_trigger_metrics.csv"
    screening_csv = args.metric_res_root / args.scenario / "paper_metric_screening.csv"
    output_path = (
        args.output
        if args.output is not None
        else scenario_root / "plots_updated" / f"{args.building}_adaptive_refreeze_overview.png"
    )

    seq_df = load_sequence(sequence_csv, args.building)
    trigger_df = load_trigger_metrics(trigger_csv, args.building)
    thresholds = load_thresholds(screening_csv)
    make_figure(seq_df, trigger_df, thresholds, args.city_a, args.city_b, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
