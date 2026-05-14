#!/usr/bin/env python3
"""Build a paper-facing screening table for climate drivers and responses."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "screening_table.csv"
FIGURES_DIR = SCRIPT_DIR / "figures"
OUTPUT_PNG = FIGURES_DIR / "screening_table.png"

INPUT_CSV = SCRIPT_DIR / "annual_metrics_extended.csv"
if not INPUT_CSV.exists():
    INPUT_CSV = SCRIPT_DIR / "annual_metrics.csv"

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
    label = label.replace(" edh", " EDH")
    label = label.replace(" kw", " kW")
    label = label.replace(" kwh", " kWh")
    return label


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def normalized_slope(group: pd.DataFrame, metric: str) -> tuple[float | None, float | None]:
    years = pd.to_numeric(group["year"], errors="coerce")
    values = numeric_series(group, metric)
    mask = years.notna() & values.notna()
    if mask.sum() < 5:
        return None, None
    slope, _, r_value, _, _ = stats.linregress(years[mask].astype(float), values[mask].astype(float))
    mean_value = float(values[mask].mean())
    normalized = np.nan if abs(mean_value) < 1e-12 else slope / abs(mean_value)
    return normalized, r_value ** 2


def volatility(group: pd.DataFrame, metric: str) -> float | None:
    values = numeric_series(group.sort_values("year"), metric).dropna()
    if len(values) < 5:
        return None
    diffs = values.diff().dropna()
    if diffs.empty:
        return None
    baseline = abs(values.mean())
    if baseline < 1e-12:
        return float(diffs.std(ddof=0))
    return float(diffs.std(ddof=0) / baseline)


def standardized_change(group: pd.DataFrame, metric: str) -> float | None:
    ordered = group.sort_values("year")
    values = numeric_series(ordered, metric).dropna()
    if len(values) < 2:
        return None
    spread = float(values.std(ddof=0))
    if spread <= 1e-12:
        return None
    return float(abs(values.iloc[-1] - values.iloc[0]) / spread)


def breach_frequency(series: pd.Series, threshold: float | None) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty or threshold is None:
        return None
    return 100.0 * float((values > threshold).mean())


def climate_discrimination(df: pd.DataFrame, metric: str) -> float | None:
    values = numeric_series(df, metric)
    if values.notna().sum() < 10:
        return None
    temp = df[["city"]].copy()
    temp["value"] = values
    temp = temp.dropna()
    if temp.empty:
        return None
    city_means = temp.groupby("city")["value"].mean()
    total_variance = temp["value"].var(ddof=0)
    if total_variance is None or total_variance <= 0:
        return None
    between_variance = city_means.var(ddof=0)
    return float(between_variance / total_variance)


def score_series(series: pd.Series, higher_is_better: bool) -> pd.Series:
    filled = series.fillna(series.min() if higher_is_better else series.max())
    ranked = filled.rank(method="average", pct=True)
    return ranked if higher_is_better else 1.0 - ranked + (1.0 / len(filled))


def decision_from_score(score: float) -> str:
    if score >= 0.60:
        return "Keep"
    if score >= 0.40:
        return "Review"
    return "Drop"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    if "building" in df.columns:
        df = df[df["building"] == "office"].copy()

    available_metrics = [metric for metrics in METRIC_FAMILIES.values() for metric in metrics if metric in df.columns]
    numeric_df = df[available_metrics].apply(pd.to_numeric, errors="coerce")
    corr_matrix = numeric_df.corr().abs()

    rows: list[dict[str, object]] = []
    for family, metrics in METRIC_FAMILIES.items():
        for metric in metrics:
            if metric not in df.columns:
                continue

            slopes: list[float] = []
            r2_values: list[float] = []
            vol_values: list[float] = []
            standardized_changes: list[float] = []
            for _, city_group in df.groupby(["scenario", "building", "city"], dropna=False):
                slope_value, r2_value = normalized_slope(city_group, metric)
                if slope_value is not None and not np.isnan(slope_value):
                    slopes.append(abs(slope_value))
                if r2_value is not None and not np.isnan(r2_value):
                    r2_values.append(r2_value)
                volatility_value = volatility(city_group, metric)
                if volatility_value is not None and not np.isnan(volatility_value):
                    vol_values.append(volatility_value)
                standardized_change_value = standardized_change(city_group, metric)
                if standardized_change_value is not None and not np.isnan(standardized_change_value):
                    standardized_changes.append(standardized_change_value)

            redundancy = None
            if metric in corr_matrix.columns:
                related = corr_matrix.loc[metric].drop(index=metric, errors="ignore").dropna()
                if not related.empty:
                    redundancy = float(related.max())

            coverage_pct = 100.0 * float(numeric_series(df, metric).notna().mean())
            rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "label": pretty_label(metric),
                    "coverage_pct": coverage_pct,
                    "mean_abs_norm_slope": float(np.nanmean(slopes)) if slopes else np.nan,
                    "mean_abs_standardized_change": float(np.nanmean(standardized_changes)) if standardized_changes else np.nan,
                    "median_r2": float(np.nanmedian(r2_values)) if r2_values else np.nan,
                    "mean_volatility": float(np.nanmean(vol_values)) if vol_values else np.nan,
                    "breach_frequency_pct": breach_frequency(df[metric], BREAK_THRESHOLDS.get(metric)),
                    "climate_discrimination": climate_discrimination(df, metric),
                    "redundancy_max_abs_r": redundancy,
                }
            )

    screening = pd.DataFrame(rows)
    screening["slope_score"] = score_series(screening["mean_abs_norm_slope"], higher_is_better=True)
    breach_signal = screening["breach_frequency_pct"].where(
        screening["breach_frequency_pct"].notna(),
        screening["mean_abs_standardized_change"],
    )
    screening["breach_score"] = score_series(breach_signal, higher_is_better=True)
    screening["discrimination_score"] = score_series(screening["climate_discrimination"], higher_is_better=True)
    screening["coverage_score"] = score_series(screening["coverage_pct"], higher_is_better=True)
    screening["stability_score"] = score_series(screening["mean_volatility"], higher_is_better=False)
    screening["uniqueness_score"] = score_series(screening["redundancy_max_abs_r"], higher_is_better=False)
    screening["screening_score"] = (
        0.24 * screening["slope_score"]
        + 0.18 * screening["breach_score"]
        + 0.22 * screening["discrimination_score"]
        + 0.16 * screening["coverage_score"]
        + 0.10 * screening["stability_score"]
        + 0.10 * screening["uniqueness_score"]
    )
    screening["decision"] = screening["screening_score"].apply(decision_from_score)
    screening = screening.sort_values(["family", "decision", "screening_score"], ascending=[True, True, False])
    screening.to_csv(OUTPUT_CSV, index=False, float_format="%.6g")

    plot_cols = [
        "slope_score",
        "breach_score",
        "discrimination_score",
        "coverage_score",
        "stability_score",
        "uniqueness_score",
        "screening_score",
    ]
    plot_df = screening.sort_values(["family", "screening_score"], ascending=[True, False]).reset_index(drop=True)
    heat_values = plot_df[plot_cols].to_numpy(dtype=float)

    fig_height = max(8, 0.36 * len(plot_df) + 2.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    image = ax.imshow(heat_values, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(plot_cols)))
    ax.set_xticklabels(
        ["Slope", "Breach", "Discr.", "Coverage", "Stability", "Uniqueness", "Score"],
        fontsize=9,
    )
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(
        [f"{family} | {label}" for family, label in zip(plot_df["family"], plot_df["label"])],
        fontsize=8,
    )

    for row_index, decision in enumerate(plot_df["decision"]):
        ax.text(len(plot_cols) - 0.05, row_index, f"  {decision}", va="center", ha="left", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02, label="Relative screening score")
    ax.set_title("Metric Screening Table: Drivers vs Responses", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded : {INPUT_CSV}")
    print(f"Saved  : {OUTPUT_CSV}")
    print(f"Saved  : {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
