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
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from figure_paper_common import CLIMATE_CITY, CLIMATE_ORDER, ensure_figures_dir, parse_year


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCENARIO = "CORDEX_CMIP5_REMO2015_rcp85"
DEFAULT_BUILDING = "office"
STRESS_EMERGENCE_DELTA = 0.25
DEFAULT_CITY_PANELS = ("Los_Angeles", "Miami", "Phoenix")
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
RESIZE_YEAR_COLOR = "#7a7a7a"
RESIZE_YEAR_LINEWIDTH = 1.05
RESIZE_YEAR_ALPHA = 0.95
BUILDING_FROZEN_PATTERNS = {
    "office": "office_{city}_frozen_detailed.epJSON",
    "apartment": "apartment_{city}_frozen_detailed.epJSON",
    "retail": "retail_{city}_frozen_detailed.epJSON",
}
CITY_KEYS = {
    "Los_Angeles": "LosAngeles",
    "Miami": "Miami",
    "Montreal": "Montreal",
    "Phoenix": "Phoenix",
    "Toronto": "Toronto",
    "Vancouver": "Vancouver",
}


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


def _sum_numeric_fields(model: dict, object_types: tuple[str, ...], field_names: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for obj_type in object_types:
        objects = model.get(obj_type, {})
        if not isinstance(objects, dict):
            continue
        for obj_data in objects.values():
            if not isinstance(obj_data, dict):
                continue
            for field_name in field_names:
                value = obj_data.get(field_name)
                if isinstance(value, (int, float)):
                    values.append(float(value))
    return sum(values) if values else None


def load_gen0_capacities(frozen_models_dir: Path, building: str, cities: list[str]) -> dict[str, float]:
    pattern = BUILDING_FROZEN_PATTERNS.get(building)
    if pattern is None:
        return {}

    capacities_kw: dict[str, float] = {}
    for city in cities:
        city_key = CITY_KEYS.get(city, city)
        model_path = frozen_models_dir / pattern.format(city=city_key)
        if not model_path.exists():
            continue
        model = json.loads(model_path.read_text())
        total_cooling_w = _sum_numeric_fields(
            model,
            (
                "Coil:Cooling:DX:TwoSpeed",
                "Coil:Cooling:DX:SingleSpeed",
                "Coil:Cooling:DX:MultiSpeed",
                "Coil:Cooling:Water",
            ),
            (
                "high_speed_gross_rated_total_cooling_capacity",
                "gross_rated_total_cooling_capacity",
                "rated_total_cooling_capacity",
            ),
        )
        if total_cooling_w is not None:
            capacities_kw[city] = total_cooling_w / 1000.0
    return capacities_kw


def load_tmy_snapshot_capacities(snapshot_path: Path, cities: list[str]) -> dict[str, float]:
    if not snapshot_path.exists():
        return {}
    data = json.loads(snapshot_path.read_text())
    capacities_kw: dict[str, float] = {}
    for city in cities:
        snap = data.get(city, {})
        capacity_w = snap.get("cooling_capacity_w")
        if isinstance(capacity_w, (int, float)):
            capacities_kw[city] = float(capacity_w) / 1000.0
    return capacities_kw


def climate_ordered_cities(seq_df: pd.DataFrame) -> list[str]:
    available = set(seq_df["city"].unique())
    ordered = [CLIMATE_CITY[code] for code in CLIMATE_ORDER if CLIMATE_CITY[code] in available]
    extras = sorted(available.difference(ordered))
    return ordered + extras


def format_city(city: str) -> str:
    return city.replace("_", " ")


def plot_timeline(ax: plt.Axes, seq_df: pd.DataFrame) -> None:
    cities = climate_ordered_cities(seq_df)

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

    ax.set_xlim(2025, 2101)
    ax.set_ylim(-0.8, len(cities) - 0.2)
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels([format_city(city) for city in cities[::-1]], fontsize=9)
    ax.set_title("A  Adaptive re-freeze timeline", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linewidth=0.35, alpha=0.45)


def build_capacity_series(
    pair_seq: pd.DataFrame,
    pair_trigger: pd.DataFrame,
    gen0_capacity_kw: float | None,
) -> pd.DataFrame:
    cap_map = (
        pair_seq.dropna(subset=["primary_cooling_capacity_kw"])
        .set_index("generation")["primary_cooling_capacity_kw"]
        .to_dict()
    )
    if gen0_capacity_kw is not None:
        cap_map[0] = gen0_capacity_kw
    cap_df = pair_trigger[["generation", "year"]].copy()
    cap_df["capacity_kw"] = cap_df["generation"].map(cap_map)
    cap_df = cap_df.dropna(subset=["capacity_kw"]).sort_values(["year", "generation"])
    return cap_df


def failure_severity(row: pd.Series, thresholds: dict[str, float]) -> float:
    severities: list[float] = []
    for metric, threshold in thresholds.items():
        value = pd.to_numeric(row.get(metric), errors="coerce")
        if pd.isna(value) or threshold <= 0:
            continue
        severities.append(max(0.0, float(value) / threshold - 1.0))
    return max(severities, default=0.0)


def threshold_segments(
    pair_seq: pd.DataFrame,
    pair_trigger: pd.DataFrame,
    thresholds: dict[str, float],
    metric: str,
) -> list[tuple[int, int, float, bool]]:
    segments: list[tuple[int, int, float, bool]] = []
    base_threshold = thresholds.get(metric)
    if base_threshold is None:
        return segments

    for _, seq_row in pair_seq.iterrows():
        gen = int(seq_row["generation"])
        start_year = int(seq_row["first_year_simulated"])
        break_year = parse_year(seq_row["break_year"])
        end_year = break_year if break_year is not None else 2100
        baseline_rows = pair_trigger[
            (pair_trigger["generation"] == gen) & (pair_trigger["year"] == start_year)
        ]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        baseline_severity = failure_severity(baseline, thresholds)
        if baseline_severity > 0:
            value = base_threshold * (1.0 + baseline_severity + STRESS_EMERGENCE_DELTA)
            segments.append((start_year, end_year, value, True))
        else:
            segments.append((start_year, end_year, base_threshold, False))
    return segments


def plot_city_metric_panel(
    axes: list[plt.Axes],
    seq_df: pd.DataFrame,
    trigger_df: pd.DataFrame,
    thresholds: dict[str, float],
    gen0_capacities_kw: dict[str, float],
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

        segments = threshold_segments(pair_seq, pair_trigger, thresholds, metric)
        labeled_fixed = False
        labeled_dynamic = False
        for start_year, end_year, threshold_value, is_dynamic in segments:
            ax.hlines(
                threshold_value,
                start_year,
                min(end_year + 1, 2100),
                color=style["color"],
                linewidth=1.0,
                linestyle="--",
                alpha=0.75,
            )
            if is_dynamic and not labeled_dynamic:
                ax.text(
                    start_year + 0.5,
                    threshold_value,
                    "baseline-relative break line",
                    fontsize=7,
                    va="bottom",
                    color=style["color"],
                )
                labeled_dynamic = True
            elif not is_dynamic and not labeled_fixed:
                ax.text(
                    start_year + 0.5,
                    threshold_value,
                    f"screen threshold {threshold_value:g}",
                    fontsize=7,
                    va="bottom",
                    color=style["color"],
                )
                labeled_fixed = True

        for break_year in break_years:
            ax.axvline(
                break_year,
                color=RESIZE_YEAR_COLOR,
                linewidth=RESIZE_YEAR_LINEWIDTH,
                linestyle=":",
                alpha=RESIZE_YEAR_ALPHA,
            )

        ax.set_xlim(2025, 2100)
        ax.set_ylabel(style["label"], color=style["color"], fontsize=8)
        ax.grid(linewidth=0.3, alpha=0.4)

    cap_ax = axes[2]
    cap_df = build_capacity_series(pair_seq, pair_trigger, gen0_capacities_kw.get(city))
    if not cap_df.empty:
        cap_ax.step(cap_df["year"], cap_df["capacity_kw"], where="post", color=CAPACITY_COLOR, linewidth=1.8)
        cap_ax.scatter(cap_df["year"], cap_df["capacity_kw"], s=8, color=CAPACITY_COLOR, alpha=0.8)
    for break_year in break_years:
        cap_ax.axvline(
            break_year,
            color=RESIZE_YEAR_COLOR,
            linewidth=RESIZE_YEAR_LINEWIDTH,
            linestyle=":",
            alpha=RESIZE_YEAR_ALPHA,
        )
    cap_ax.set_xlim(2025, 2100)
    cap_ax.set_ylabel("Cooling capacity (kW)", color=CAPACITY_COLOR, fontsize=8)
    cap_ax.set_xlabel("Year")
    cap_ax.grid(linewidth=0.3, alpha=0.4)

    axes[0].set_title(
        f"{panel_letter}  {format_city(city)}",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)


def make_figure(
    seq_df: pd.DataFrame,
    trigger_df: pd.DataFrame,
    thresholds: dict[str, float],
    gen0_capacities_kw: dict[str, float],
    cities: list[str],
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
    n_panels = len(cities)
    fig = plt.figure(figsize=(5.0 * n_panels + 3.0, 8.8), facecolor="white")
    gs = gridspec.GridSpec(
        4,
        n_panels + 1,
        width_ratios=[1.0] * n_panels + [0.28],
        height_ratios=[0.82, 1.0, 1.0, 0.92],
        wspace=0.34,
        hspace=0.42,
    )

    ax_timeline = fig.add_subplot(gs[0, :n_panels])
    plot_timeline(ax_timeline, seq_df)

    legend_ax = fig.add_subplot(gs[:, n_panels])
    legend_ax.axis("off")

    gen_handles = [
        mpatches.Patch(
            color=GENERATION_COLORS[min(gen, len(GENERATION_COLORS) - 1)],
            label="Gen 0 (TMY)" if gen == 0 else f"Gen {gen}",
        )
        for gen in sorted(seq_df["generation"].dropna().astype(int).unique())
    ]
    legend_gen = legend_ax.legend(
        handles=gen_handles,
        loc="upper left",
        bbox_to_anchor=(-0.12, 0.98),
        fontsize=8,
        frameon=False,
        borderaxespad=0.0,
    )
    legend_ax.add_artist(legend_gen)

    annotation_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=5.5, color="#666666", label="Main break year"),
        Line2D([0], [0], linestyle="--", linewidth=1.2, color="#666666", label="Break reference line"),
        Line2D(
            [0],
            [0],
            linestyle=":",
            linewidth=RESIZE_YEAR_LINEWIDTH,
            color=RESIZE_YEAR_COLOR,
            alpha=RESIZE_YEAR_ALPHA,
            label="Resize year",
        ),
    ]
    legend_ax.legend(
        handles=annotation_handles,
        loc="upper left",
        bbox_to_anchor=(-0.12, 0.72),
        ncol=1,
        fontsize=8,
        frameon=False,
        handlelength=2.4,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    for col_idx, city in enumerate(cities):
        axes = [fig.add_subplot(gs[row_idx, col_idx]) for row_idx in range(1, 4)]
        panel_letter = chr(ord("B") + col_idx)
        plot_city_metric_panel(axes, seq_df, trigger_df, thresholds, gen0_capacities_kw, city, panel_letter)
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
    parser.add_argument(
        "--cities",
        default=",".join(DEFAULT_CITY_PANELS),
        help="Comma-separated city panels to show.",
    )
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
        "--frozen-models-dir",
        type=Path,
        default=SCRIPT_DIR / "frozen_models",
        help="Directory containing the baseline TMY-frozen models.",
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
    tmy_snapshot_path = scenario_root / args.building / "tmy_sizing_snapshots.json"
    output_path = (
        args.output
        if args.output is not None
        else scenario_root / "plots_updated" / f"{args.building}_adaptive_refreeze_overview.png"
    )

    seq_df = load_sequence(sequence_csv, args.building)
    trigger_df = load_trigger_metrics(trigger_csv, args.building)
    thresholds = load_thresholds(screening_csv)
    cities = [city.strip() for city in args.cities.split(",") if city.strip()]
    gen0_capacities_kw = load_gen0_capacities(
        args.frozen_models_dir,
        args.building,
        cities,
    )
    gen0_capacities_kw.update(
        load_tmy_snapshot_capacities(tmy_snapshot_path, cities)
    )
    make_figure(seq_df, trigger_df, thresholds, gen0_capacities_kw, cities, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
