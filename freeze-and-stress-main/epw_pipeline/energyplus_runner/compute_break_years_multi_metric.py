#!/usr/bin/env python3
"""Summarize break years and paper-facing driver/response metrics.

The metric list is intentionally registry-driven. New upstream extraction
columns can be added by extending METRICS aliases without rewriting the
summary logic.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "metric_exports" / "annual_metrics_extended.csv"
if not DEFAULT_INPUT.exists():
    DEFAULT_INPUT = SCRIPT_DIR / "metric_exports" / "annual_metrics.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "paper_metrics_summary.csv"
DEFAULT_SCREENING_OUTPUT = SCRIPT_DIR / "paper_metric_screening.csv"
DEFAULT_REGISTRY_OUTPUT = SCRIPT_DIR / "paper_metric_registry.csv"
DEFAULT_CLIMATE_TYPE_SCREENING_OUTPUT = SCRIPT_DIR / "paper_screening_by_climate_type.csv"
DEFAULT_BLOCK_DECOMPOSITION_OUTPUT = SCRIPT_DIR / "paper_block_decomposition.csv"

DEFAULT_BASELINE_YEAR = 2025

# Köppen climate type mapping for per-climate-type analysis (NCC paper).
# Each city is assigned to its representative Köppen zone.
CITY_KOPPEN_MAP: dict[str, str] = {
    "Phoenix": "BWh",        # hot arid
    "Miami": "Am",           # tropical monsoon / hot-humid
    "Los_Angeles": "Csb",    # warm-summer Mediterranean
    "Toronto": "Dfa",        # humid continental
    "Montreal": "Dfb",       # cold continental
    "Vancouver": "Cfb",      # marine / oceanic
}
SUSTAINED_YEARS = 5
FAILURE_OCCURRENCE_WINDOWS = (2, 3, 5)
FAILURE_OCCURRENCE_RATIO_THRESHOLD = 0.25
REDUNDANCY_THRESHOLD = 0.90
KEEP_SCORE_THRESHOLD = 0.60
REVIEW_SCORE_THRESHOLD = 0.40
STRESS_EMERGENCE_DELTA = 0.25

# Break-year thresholds are intentionally set as severe transition screens,
# not as first-exceedance comfort or compliance limits. Lower first-exceedance
# values made most climates fail at the 2025 baseline and compressed the paper
# break years into the first few simulation years.
ANNUAL_EDH_THRESHOLD = 300.0
DAILY_EDH_THRESHOLD = 12.0
DAILY_EDH_EXCEED_DAYS_THRESHOLD = 15.0
OPER_HEATING_NOT_MET_THRESHOLD = 400.0
OPER_COOLING_SETPOINT_NOT_MET_THRESHOLD = 500.0
ATTRIBUTION_DRIVER_KEYS = (
    "annual_mean_drybulb_c",
    "cdd_18",
    "hdd_18",
    "maximum_wetbulb_c",
    "annual_ghi_kwh_m2",
    "heatwave_days",
)
ATTRIBUTION_RIDGE_ALPHA = 1.0
RELATIVE_STRESS_PERCENTILE = 90.0
DRIVER_BLOCK_REPRESENTATIVES = {
    "thermal_shift": ("annual_mean_drybulb_c", "summer_mean_drybulb_c"),
    "demand_background": ("cdd_18", "hdd_18"),
    "humid_heat_stress": ("maximum_wetbulb_c",),
    "solar_burden": ("annual_ghi_kwh_m2", "peak_ghi_w_m2"),
    "persistence_heatwave": ("heatwave_days", "maximum_consecutive_hot_days"),
}
DECISION_RANK = {"Drop": 0, "Supplementary": 1, "Review": 2, "Priority Review": 3, "Keep": 4}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    role: str  # "driver" or "response"
    family: str
    subfamily: str
    label: str
    unit: str
    aliases: tuple[str, ...] = ()
    threshold: float | None = None
    threshold_direction: str = "above"
    failure_relevant: bool = False
    optional: bool = False
    standard: str = ""
    interpretation: str = ""
    analysis_tier: str = "main"

    @property
    def prefix(self) -> str:
        return f"{self.family}__{self.subfamily}__{self.key}"


def spec(
    key: str,
    role: str,
    family: str,
    subfamily: str,
    label: str,
    unit: str,
    aliases: tuple[str, ...] = (),
    threshold: float | None = None,
    failure_relevant: bool = False,
    optional: bool = False,
    standard: str = "",
    interpretation: str = "",
    analysis_tier: str = "main",
) -> MetricSpec:
    return MetricSpec(
        key=key,
        role=role,
        family=family,
        subfamily=subfamily,
        label=label,
        unit=unit,
        aliases=aliases,
        threshold=threshold,
        failure_relevant=failure_relevant,
        optional=optional,
        standard=standard,
        interpretation=interpretation,
        analysis_tier=analysis_tier,
    )


METRICS: tuple[MetricSpec, ...] = (
    # ── Climate drivers ──────────────────────────────────────────────
    # Thermal shift
    spec(
        "annual_mean_drybulb_c",
        "driver",
        "climate_drivers",
        "thermal_shift",
        "Annual mean dry-bulb temperature",
        "C",
        ("outdoor_drybulb_c_mean", "annual_mean_drybulb_c", "outdoor_drybulb_mean"),
    ),
    spec(
        "summer_mean_drybulb_c",
        "driver",
        "climate_drivers",
        "thermal_shift",
        "Summer mean dry-bulb temperature",
        "C",
        ("summer_mean_drybulb_c", "summer_outdoor_drybulb_c_mean", "outdoor_drybulb_c_summer_mean"),
    ),
    # Cooling/heating demand background
    spec(
        "cdd_18",
        "driver",
        "climate_drivers",
        "demand_background",
        "Cooling degree days",
        "degree-days",
        ("cdd_18", "cooling_degree_days_18c"),
    ),
    spec(
        "hdd_18",
        "driver",
        "climate_drivers",
        "demand_background",
        "Heating degree days",
        "degree-days",
        ("hdd_18", "heating_degree_days_18c"),
    ),
    # Humid heat stress
    spec(
        "maximum_wetbulb_c",
        "driver",
        "climate_drivers",
        "humid_heat_stress",
        "Maximum wet-bulb temperature",
        "C",
        ("outdoor_wetbulb_c_max", "maximum_wetbulb_c", "outdoor_wetbulb_max"),
    ),
    # Solar burden
    spec(
        "annual_ghi_kwh_m2",
        "driver",
        "climate_drivers",
        "solar_burden",
        "Annual GHI",
        "kWh/m2",
        ("annual_ghi_kwh_m2", "ghi_annual_kwh_m2"),
    ),
    spec(
        "peak_ghi_w_m2",
        "driver",
        "climate_drivers",
        "solar_burden",
        "Peak GHI",
        "W/m2",
        ("peak_ghi_w_m2", "ghi_w_m2_max", "outdoor_ghi_w_m2_max"),
    ),
    # Persistence / heatwave
    spec(
        "heatwave_days",
        "driver",
        "climate_drivers",
        "persistence_heatwave",
        "Heatwave days",
        "days",
        ("heatwave_days", "heatwave_days_35c"),
    ),
    spec(
        "maximum_consecutive_hot_days",
        "driver",
        "climate_drivers",
        "persistence_heatwave",
        "Maximum consecutive hot days",
        "days",
        ("max_consec_hot_days_35c", "maximum_consecutive_hot_days", "max_consecutive_hot_days"),
    ),
    # ── Building responses ───────────────────────────────────────────
    # Habitability
    spec(
        "peak_operative_temperature_c",
        "response",
        "building_responses",
        "habitability",
        "Peak operative temperature",
        "C",
        ("rep_zone_operative_temp_c_max", "peak_operative_temperature_c", "tmax"),
    ),
    spec(
        "maximum_delta_t_k",
        "response",
        "building_responses",
        "habitability",
        "Maximum delta_t",
        "K",
        ("delta_t_max_k", "maximum_delta_t_k", "delta_t"),
    ),
    spec(
        "annual_edh_c_h",
        "response",
        "building_responses",
        "habitability",
        "Annual EDH",
        "C h",
        ("annual_edh_c_h", "annual_edh"),
        threshold=ANNUAL_EDH_THRESHOLD,
        failure_relevant=True,
        standard="EN 16798-1 derived transition screen",
        interpretation="Severe annual exceedance degree-hour burden above adaptive comfort threshold",
    ),
    spec(
        "maximum_daily_edh_k_h",
        "response",
        "building_responses",
        "habitability",
        "Maximum daily EDH",
        "K h",
        ("daily_edh_max_k_h", "maximum_daily_edh_k_h", "daily_edh"),
        threshold=DAILY_EDH_THRESHOLD,
        failure_relevant=True,
        standard="CIBSE TM52-derived transition screen",
        interpretation="Severe maximum daily exceedance degree-hour burden",
    ),
    spec(
        "daily_edh_exceedance_days",
        "response",
        "building_responses",
        "habitability",
        "Daily EDH exceedance days",
        "days",
        ("daily_edh_exceed_6kh_count", "daily_edh_exceedance_days"),
        threshold=DAILY_EDH_EXCEED_DAYS_THRESHOLD,
        failure_relevant=True,
        standard="TM52-derived transition screen",
        interpretation="Annual count of days with severe daily EDH exceedance",
    ),
    # Operational adequacy
    spec(
        "cooling_setpoint_not_met_occupied_hours",
        "response",
        "building_responses",
        "operational_adequacy",
        "Cooling setpoint not met occupied hours",
        "h",
        (
            "facility_cooling_setpoint_not_met_occupied_time_total_hours",
            "cooling_setpoint_not_met_occupied",
        ),
        threshold=OPER_COOLING_SETPOINT_NOT_MET_THRESHOLD,
        failure_relevant=True,
        standard="ASHRAE 90.1-derived transition screen",
        interpretation="Severe occupied cooling unmet load hours",
    ),
    spec(
        "heating_unmet_hours",
        "response",
        "building_responses",
        "operational_adequacy",
        "Heating unmet hours",
        "h",
        ("abups_occupied_heating_not_met_hours", "heating_unmet_hours"),
        threshold=OPER_HEATING_NOT_MET_THRESHOLD,
        failure_relevant=True,
        standard="ASHRAE 90.1-derived transition screen",
        interpretation="Severe occupied heating unmet load hours; cold-climate operational adequacy trigger",
    ),
    spec(
        "cooling_peak_demand_kw",
        "response",
        "building_responses",
        "operational_adequacy",
        "Cooling peak demand",
        "kW",
        ("cooling_electricity_peak_kw", "cooling_peak_kw"),
        interpretation="Operational peak demand diagnostic",
    ),
    spec(
        "heating_peak_demand_kw",
        "response",
        "building_responses",
        "operational_adequacy",
        "Heating peak demand",
        "kW",
        ("heating_electricity_peak_kw", "heating_peak_kw"),
        interpretation="Heating peak demand diagnostic; cold-climate supplementary adequacy",
    ),
    # Energy burden (supporting)
    spec(
        "annual_cooling_electricity_kwh",
        "response",
        "building_responses",
        "energy_burden",
        "Annual cooling electricity",
        "kWh",
        ("cooling_electricity_annual_kwh", "annual_cooling_electricity_kwh"),
        analysis_tier="supporting",
    ),
    spec(
        "annual_heating_electricity_kwh",
        "response",
        "building_responses",
        "energy_burden",
        "Annual heating electricity",
        "kWh",
        ("heating_electricity_annual_kwh", "annual_heating_electricity_kwh"),
        analysis_tier="supporting",
    ),
    # Durability
    spec(
        "indoor_rh_max_pct",
        "response",
        "building_responses",
        "durability",
        "Indoor RH max",
        "%",
        ("rep_zone_rh_pct_max", "indoor_rh_max_pct"),
        interpretation="Bulk zone RH diagnostic; not a full moisture-risk model",
        analysis_tier="supplementary",
    ),
)

FAILURE_METRICS = tuple(metric for metric in METRICS if metric.failure_relevant)
THRESHOLD_METRICS = tuple(metric for metric in METRICS if metric.threshold is not None)
SUMMARY_SUFFIXES = (
    "baseline_value",
    "final_value",
    "absolute_change",
    "log_ratio_change",
    "standardized_change",
    "temporal_slope_per_year",
    "temporal_acceleration_per_year2",
    "volatility_change",
    "maximum_jump",
    "breach_status",
    "breach_frequency_pct",
    "break_year",
    "sustained_failure_year",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute break years and paper-facing metric summaries.")
    parser.add_argument("--file", default=str(DEFAULT_INPUT), help="Input annual metrics CSV.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output wide summary CSV.")
    parser.add_argument(
        "--screening-output",
        default=str(DEFAULT_SCREENING_OUTPUT),
        help="Output long-form metric screening CSV.",
    )
    parser.add_argument(
        "--registry-output",
        default=str(DEFAULT_REGISTRY_OUTPUT),
        help="Output metric registry CSV with roles, thresholds, and retention notes.",
    )
    parser.add_argument(
        "--climate-type-screening-output",
        default=str(DEFAULT_CLIMATE_TYPE_SCREENING_OUTPUT),
        help="Output per-climate-type screening CSV (NCC Fig 2).",
    )
    parser.add_argument(
        "--block-decomposition-output",
        default=str(DEFAULT_BLOCK_DECOMPOSITION_OUTPUT),
        help="Output driver-block variance decomposition CSV (NCC Fig 3).",
    )
    parser.add_argument(
        "--baseline-year",
        type=int,
        default=DEFAULT_BASELINE_YEAR,
        help="Preferred baseline year. If missing for a group, the first valid year is used.",
    )
    parser.add_argument(
        "--sustained-years",
        type=int,
        default=SUSTAINED_YEARS,
        help="Consecutive breach years required for sustained failure.",
    )
    parser.add_argument(
        "--occurrence-ratio-threshold",
        type=float,
        default=FAILURE_OCCURRENCE_RATIO_THRESHOLD,
        help="Failure occurrence ratio threshold for rolling 2/3/5-year screening.",
    )
    return parser.parse_args()


def to_float(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        number = float(value)
        if math.isnan(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def to_int(value: object) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def first_float(source: dict[str, object], aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        value = to_float(source.get(alias))
        if value is not None:
            return value
    return None


def slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return numerator / denominator if denominator else None


def corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator_x = sum((x - x_mean) ** 2 for x in xs) ** 0.5
    denominator_y = sum((y - y_mean) ** 2 for y in ys) ** 0.5
    if denominator_x == 0 or denominator_y == 0:
        return None
    return numerator / (denominator_x * denominator_y)


def std(values: list[float]) -> float | None:
    if not values:
        return None
    mean_value = sum(values) / len(values)
    return (sum((value - mean_value) ** 2 for value in values) / len(values)) ** 0.5


def variance(values: list[float]) -> float | None:
    spread = std(values)
    return None if spread is None else spread * spread


def clean(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def mean_or_none(values: list[float | None]) -> float | None:
    filtered = clean(values)
    return (sum(filtered) / len(filtered)) if filtered else None


def value_at_or_first(records: list[dict[str, object]], key: str, preferred_year: int) -> tuple[int | None, float | None]:
    valid = [
        (int(record["year"]), to_float(record.get(key)))
        for record in sorted(records, key=lambda row: row["year"])
        if record.get("year") is not None and to_float(record.get(key)) is not None
    ]
    if not valid:
        return None, None
    for year, value in valid:
        if year == preferred_year:
            return year, value
    return valid[0]


def final_value(records: list[dict[str, object]], key: str) -> tuple[int | None, float | None]:
    valid = [
        (int(record["year"]), to_float(record.get(key)))
        for record in sorted(records, key=lambda row: row["year"])
        if record.get("year") is not None and to_float(record.get(key)) is not None
    ]
    return valid[-1] if valid else (None, None)


def yearly_pairs(records: list[dict[str, object]], key: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for record in sorted(records, key=lambda row: row["year"]):
        year = to_int(record.get("year"))
        value = to_float(record.get(key))
        if year is not None and value is not None:
            pairs.append((float(year), value))
    return pairs


def year_to_year_changes(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    changes: list[tuple[float, float]] = []
    for (left_year, left_value), (right_year, right_value) in zip(pairs[:-1], pairs[1:]):
        year_delta = right_year - left_year
        if year_delta <= 0:
            continue
        midpoint = (left_year + right_year) / 2.0
        changes.append((midpoint, (right_value - left_value) / year_delta))
    return changes


def breaches(value: float | None, metric: MetricSpec) -> bool:
    if value is None or metric.threshold is None:
        return False
    if metric.threshold_direction == "below":
        return value < metric.threshold
    return value > metric.threshold


def metric_break_year(records: list[dict[str, object]], metric: MetricSpec) -> int | str:
    if metric.threshold is None:
        return ""
    for record in sorted(records, key=lambda row: row["year"]):
        if breaches(to_float(record.get(metric.key)), metric):
            return record["year"]
    return "Never"


def metric_sustained_year(records: list[dict[str, object]], metric: MetricSpec, sustained_years: int) -> int | str:
    if metric.threshold is None:
        return ""
    ordered = sorted(records, key=lambda row: row["year"])
    for index in range(len(ordered) - sustained_years + 1):
        window = ordered[index:index + sustained_years]
        if all(breaches(to_float(record.get(metric.key)), metric) for record in window):
            return window[0]["year"]
    return "Never"


def any_failure(record: dict[str, object], metrics: tuple[MetricSpec, ...] = FAILURE_METRICS) -> bool:
    return any(breaches(to_float(record.get(metric.key)), metric) for metric in metrics)


def failure_severity(record: dict[str, object], metrics: tuple[MetricSpec, ...] = FAILURE_METRICS) -> float:
    severities: list[float] = []
    for metric in metrics:
        value = to_float(record.get(metric.key))
        if value is None or metric.threshold is None or metric.threshold <= 0:
            continue
        if metric.threshold_direction == "below":
            severities.append(max(0.0, metric.threshold / value - 1.0) if value > 0 else 0.0)
        else:
            severities.append(max(0.0, value / metric.threshold - 1.0))
    return max(severities, default=0.0)


def absolute_break_year(records: list[dict[str, object]], metrics: tuple[MetricSpec, ...] = FAILURE_METRICS) -> int | str:
    for record in sorted(records, key=lambda row: row["year"]):
        if any_failure(record, metrics):
            return record["year"]
    return "Never"


def stress_emergence_year(
    records: list[dict[str, object]],
    baseline_year: int,
    metrics: tuple[MetricSpec, ...] = FAILURE_METRICS,
    delta: float = STRESS_EMERGENCE_DELTA,
) -> int | str:
    ordered = sorted(records, key=lambda row: row["year"])
    if not ordered:
        return "Never"
    baseline_record = next((record for record in ordered if record["year"] == baseline_year), ordered[0])
    baseline_severity = failure_severity(baseline_record, metrics)
    for record in ordered:
        if record["year"] <= baseline_record["year"]:
            continue
        if failure_severity(record, metrics) >= baseline_severity + delta:
            return record["year"]
    return "Never"


def break_year(records: list[dict[str, object]], baseline_year: int, metrics: tuple[MetricSpec, ...] = FAILURE_METRICS) -> int | str:
    ordered = sorted(records, key=lambda row: row["year"])
    if not ordered:
        return "Never"
    baseline_record = next((record for record in ordered if record["year"] == baseline_year), ordered[0])
    if any_failure(baseline_record, metrics):
        return stress_emergence_year(ordered, baseline_year, metrics)
    for record in ordered:
        if record["year"] < baseline_record["year"]:
            continue
        if any_failure(record, metrics):
            return record["year"]
    return "Never"


def sustained_failure_year(
    records: list[dict[str, object]],
    sustained_years: int,
    baseline_year: int,
    metrics: tuple[MetricSpec, ...] = FAILURE_METRICS,
) -> int | str:
    ordered = sorted(records, key=lambda row: row["year"])
    if not ordered:
        return "Never"
    baseline_record = next((record for record in ordered if record["year"] == baseline_year), ordered[0])
    if any_failure(baseline_record, metrics):
        threshold = failure_severity(baseline_record, metrics) + STRESS_EMERGENCE_DELTA
        for index in range(len(ordered) - sustained_years + 1):
            window = ordered[index:index + sustained_years]
            if window[0]["year"] <= baseline_record["year"]:
                continue
            if all(failure_severity(record, metrics) >= threshold for record in window):
                return window[0]["year"]
        return "Never"

    for index in range(len(ordered) - sustained_years + 1):
        window = ordered[index:index + sustained_years]
        if window[0]["year"] < baseline_record["year"]:
            continue
        if all(any_failure(record, metrics) for record in window):
            return window[0]["year"]
    return "Never"


def failure_occurrence_count(records: list[dict[str, object]], metrics: tuple[MetricSpec, ...] = FAILURE_METRICS) -> int:
    return sum(1 for record in records if any_failure(record, metrics))


def failure_occurrence_ratio(records: list[dict[str, object]], metrics: tuple[MetricSpec, ...] = FAILURE_METRICS) -> float | None:
    if not records:
        return None
    return failure_occurrence_count(records, metrics) / len(records)


def rolling_failure_occurrence_year(
    records: list[dict[str, object]],
    window_years: int,
    baseline_year: int,
    ratio_threshold: float,
    metrics: tuple[MetricSpec, ...] = FAILURE_METRICS,
) -> int | str:
    ordered = [record for record in sorted(records, key=lambda row: row["year"]) if record["year"] >= baseline_year]
    if window_years <= 0 or len(ordered) < window_years:
        return "Never"
    for index in range(len(ordered) - window_years + 1):
        window = ordered[index:index + window_years]
        ratio = failure_occurrence_ratio(window, metrics)
        if ratio is not None and ratio >= ratio_threshold:
            return window[-1]["year"]
    return "Never"


def rolling_failure_occurrence_count(
    records: list[dict[str, object]],
    window_years: int,
    baseline_year: int,
    ratio_threshold: float,
    metrics: tuple[MetricSpec, ...] = FAILURE_METRICS,
) -> int:
    ordered = [record for record in sorted(records, key=lambda row: row["year"]) if record["year"] >= baseline_year]
    if window_years <= 0 or len(ordered) < window_years:
        return 0
    return sum(
        1
        for index in range(len(ordered) - window_years + 1)
        if (failure_occurrence_ratio(ordered[index:index + window_years], metrics) or 0.0) >= ratio_threshold
    )


def failure_probability(records: list[dict[str, object]]) -> float | None:
    if not records:
        return None
    return 100.0 * sum(1 for record in records if any_failure(record)) / len(records)


def first_trigger_metric(records: list[dict[str, object]], target_year: int | str | None = None) -> str:
    if target_year == "Never":
        return "Never"
    for record in sorted(records, key=lambda row: row["year"]):
        if target_year not in (None, "Never") and record["year"] != target_year:
            continue
        triggered = [
            metric.label
            for metric in FAILURE_METRICS
            if breaches(to_float(record.get(metric.key)), metric)
        ]
        if triggered:
            return "; ".join(triggered)
    return "Never"


def first_trigger_year_for_subfamily(records: list[dict[str, object]], subfamily: str) -> int | str:
    metrics = [metric for metric in THRESHOLD_METRICS if metric.subfamily == subfamily]
    if not metrics:
        return ""
    return absolute_break_year(records, tuple(metrics))


def sustained_trigger_year_for_subfamily(records: list[dict[str, object]], subfamily: str, sustained_years: int) -> int | str:
    metrics = [metric for metric in THRESHOLD_METRICS if metric.subfamily == subfamily]
    if not metrics:
        return ""
    ordered = sorted(records, key=lambda row: row["year"])
    for index in range(len(ordered) - sustained_years + 1):
        window = ordered[index:index + sustained_years]
        if all(any(breaches(to_float(record.get(metric.key)), metric) for metric in metrics) for record in window):
            return window[0]["year"]
    return "Never"


def adjusted_trigger_year_for_subfamily(records: list[dict[str, object]], subfamily: str, baseline_year: int) -> int | str:
    metrics = tuple(metric for metric in THRESHOLD_METRICS if metric.subfamily == subfamily)
    if not metrics:
        return ""
    return break_year(records, baseline_year, metrics)


def adjusted_sustained_trigger_year_for_subfamily(
    records: list[dict[str, object]],
    subfamily: str,
    baseline_year: int,
    sustained_years: int,
) -> int | str:
    metrics = tuple(metric for metric in THRESHOLD_METRICS if metric.subfamily == subfamily)
    if not metrics:
        return ""
    return sustained_failure_year(records, sustained_years, baseline_year, metrics)


def metric_stats(records: list[dict[str, object]], metric: MetricSpec, baseline_year: int, sustained_years: int) -> dict[str, object]:
    pairs = yearly_pairs(records, metric.key)
    values = [value for _, value in pairs]
    baseline_actual_year, baseline_value = value_at_or_first(records, metric.key, baseline_year)
    final_actual_year, end_value = final_value(records, metric.key)

    absolute_change = None
    log_ratio_change = None
    standardized_change = None
    if baseline_value is not None and end_value is not None:
        absolute_change = end_value - baseline_value
        if baseline_value > 0 and end_value > 0:
            log_ratio_change = math.log(end_value / baseline_value)
        series_std = std(values)
        if series_std is not None and series_std > 1e-12:
            standardized_change = absolute_change / series_std

    changes = year_to_year_changes(pairs)
    change_values = [change for _, change in changes]
    max_jump = max((abs(change) for change in change_values), default=None)
    breach_flags = [breaches(value, metric) for value in values]

    if metric.threshold is None:
        breach_status: str | None = ""
        breach_frequency_pct: float | None = None
    elif values:
        breach_status = "Breached" if any(breach_flags) else "Not breached"
        breach_frequency_pct = 100.0 * sum(breach_flags) / len(breach_flags)
    else:
        breach_status = ""
        breach_frequency_pct = None

    return {
        "baseline_year": baseline_actual_year,
        "baseline_value": baseline_value,
        "final_year": final_actual_year,
        "final_value": end_value,
        "absolute_change": absolute_change,
        "log_ratio_change": log_ratio_change,
        "standardized_change": standardized_change,
        "temporal_slope_per_year": slope([year for year, _ in pairs], values),
        "temporal_acceleration_per_year2": slope([year for year, _ in changes], change_values),
        "volatility_change": std(change_values),
        "maximum_jump": max_jump,
        "breach_status": breach_status,
        "breach_frequency_pct": breach_frequency_pct,
        "break_year": metric_break_year(records, metric),
        "sustained_failure_year": metric_sustained_year(records, metric, sustained_years),
    }


def normalize_row(source: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario": source.get("scenario"),
        "building": source.get("building"),
        "city": source.get("city"),
        "year": to_int(source.get("year")),
    }

    for metric in METRICS:
        row[metric.key] = first_float(source, metric.aliases)

    return row


def summarize(
    records: list[dict[str, object]],
    baseline_year: int,
    sustained_years: int,
    occurrence_ratio_threshold: float = FAILURE_OCCURRENCE_RATIO_THRESHOLD,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    records = sorted(records, key=lambda row: row["year"])
    absolute_standard_break_year = absolute_break_year(records)
    main_break_year = break_year(records, baseline_year)
    main_sustained_failure_year = sustained_failure_year(records, sustained_years, baseline_year)

    result: dict[str, object] = {
        "baseline_year": baseline_year,
        "first_available_year": records[0]["year"] if records else None,
        "final_available_year": records[-1]["year"] if records else None,
        "absolute_standard_break_year": absolute_standard_break_year,
        "baseline_failure_severity": failure_severity(next((record for record in records if record["year"] == baseline_year), records[0])) if records else None,
        "main_break_year": main_break_year,
        "main_stress_emergence_year": main_break_year,
        "stress_emergence_delta": STRESS_EMERGENCE_DELTA,
        "main_sustained_failure_year": main_sustained_failure_year,
        "break_year": main_break_year,
        "sustained_failure_year": main_sustained_failure_year,
        "first_trigger_metric": first_trigger_metric(records, main_break_year),
        "failure_probability_pct": failure_probability(records),
        "failure_occurrence_count": failure_occurrence_count(records),
        "failure_occurrence_ratio_pct": (failure_occurrence_ratio(records) or 0.0) * 100.0 if records else None,
        "failure_occurrence_ratio_threshold_pct": occurrence_ratio_threshold * 100.0,
        "operational_adequacy_break_year": adjusted_trigger_year_for_subfamily(records, "operational_adequacy", baseline_year),
        "operational_adequacy_sustained_failure_year": adjusted_sustained_trigger_year_for_subfamily(records, "operational_adequacy", baseline_year, sustained_years),
        "habitability_break_year": adjusted_trigger_year_for_subfamily(records, "habitability", baseline_year),
        "habitability_sustained_failure_year": adjusted_sustained_trigger_year_for_subfamily(records, "habitability", baseline_year, sustained_years),
        "operational_adequacy__first_trigger_year": adjusted_trigger_year_for_subfamily(records, "operational_adequacy", baseline_year),
        "habitability__first_trigger_year": adjusted_trigger_year_for_subfamily(records, "habitability", baseline_year),
    }
    for window_years in FAILURE_OCCURRENCE_WINDOWS:
        result[f"failure_occurrence_{window_years}y_year"] = rolling_failure_occurrence_year(
            records,
            window_years,
            baseline_year,
            occurrence_ratio_threshold,
        )
        result[f"failure_occurrence_{window_years}y_window_count"] = rolling_failure_occurrence_count(
            records,
            window_years,
            baseline_year,
            occurrence_ratio_threshold,
        )

    stats_by_metric: dict[str, dict[str, object]] = {}
    for metric in METRICS:
        stats = metric_stats(records, metric, baseline_year, sustained_years)
        stats_by_metric[metric.key] = stats
        for suffix in SUMMARY_SUFFIXES:
            result[f"{metric.prefix}__{suffix}"] = stats.get(suffix)

    return result, stats_by_metric


def climate_discrimination(records: list[dict[str, object]], metric: MetricSpec) -> float | None:
    city_values: defaultdict[str, list[float]] = defaultdict(list)
    all_values: list[float] = []
    for record in records:
        city = record.get("city")
        value = to_float(record.get(metric.key))
        if city is None or value is None:
            continue
        city_values[str(city)].append(value)
        all_values.append(value)

    if len(all_values) < 2 or len(city_values) < 2:
        return None

    total_variance = variance(all_values)
    if total_variance is None or total_variance <= 0:
        return None

    city_means = [sum(values) / len(values) for values in city_values.values() if values]
    between_variance = variance(city_means)
    if between_variance is None:
        return None
    return between_variance / total_variance


def redundancy_score(records: list[dict[str, object]], metric: MetricSpec) -> float | None:
    values_by_key = {
        other.key: [to_float(record.get(other.key)) for record in records]
        for other in METRICS
    }
    target_values = values_by_key[metric.key]
    best: float | None = None
    for other in METRICS:
        if other.key == metric.key or other.subfamily != metric.subfamily:
            continue
        pairs = [
            (left, right)
            for left, right in zip(target_values, values_by_key[other.key])
            if left is not None and right is not None
        ]
        if len(pairs) < 5:
            continue
        value = corr([left for left, _ in pairs], [right for _, right in pairs])
        if value is None:
            continue
        abs_value = abs(value)
        best = abs_value if best is None else max(best, abs_value)
    return 0.0 if best is None else best


def scale_scores(values: list[float | None], higher_is_better: bool) -> list[float]:
    available = [value for value in values if value is not None and not math.isnan(value)]
    if not available:
        return [0.0 for _ in values]

    low = min(available)
    high = max(available)
    scores: list[float] = []
    for value in values:
        if value is None or math.isnan(value):
            scores.append(0.0)
        elif abs(high - low) < 1e-12:
            scores.append(1.0)
        else:
            score_value = (value - low) / (high - low)
            scores.append(score_value if higher_is_better else 1.0 - score_value)
    return scores


def decide_from_score(score_value: float, coverage_pct: float) -> str:
    if coverage_pct <= 0:
        return "Drop"
    if score_value >= KEEP_SCORE_THRESHOLD:
        return "Keep"
    if score_value >= REVIEW_SCORE_THRESHOLD:
        return "Review"
    return "Drop"


def stronger_decision(current: str, floor: str) -> str:
    return floor if DECISION_RANK[floor] > DECISION_RANK.get(current, 0) else current


def apply_retention_rules(rows: list[dict[str, object]]) -> None:
    rows_by_key = {str(row["metric"]): row for row in rows}

    for subfamily, representatives in DRIVER_BLOCK_REPRESENTATIVES.items():
        candidate_rows = [rows_by_key[key] for key in representatives if key in rows_by_key]
        if not candidate_rows:
            continue

        if subfamily == "demand_background":
            selected_rows = candidate_rows
        else:
            selected_rows = [max(candidate_rows, key=lambda row: to_float(row.get("screening_score")) or -1.0)]

        for row in selected_rows:
            row["keep_drop_decision"] = stronger_decision(str(row["keep_drop_decision"]), "Keep")
            rule = f"Driver block representative retained for {subfamily}"
            row["retention_rule"] = rule if not row.get("retention_rule") else f"{row['retention_rule']}; {rule}"

    operational_floors = {
        "cooling_setpoint_not_met_occupied_hours": ("Keep", "Core operational adequacy trigger"),
        "cooling_peak_demand_kw": ("Priority Review", "Peak demand is core operational adequacy evidence"),
        "heating_unmet_hours": ("Keep", "Standard-threshold cold-climate operational adequacy trigger"),
        "heating_peak_demand_kw": ("Review", "Cold-climate peak demand supplementary diagnostic"),
    }
    for key, (floor, rule) in operational_floors.items():
        row = rows_by_key.get(key)
        if row is None:
            continue
        row["keep_drop_decision"] = stronger_decision(str(row["keep_drop_decision"]), floor)
        row["retention_rule"] = rule if not row.get("retention_rule") else f"{row['retention_rule']}; {rule}"

    for row in rows:
        if row.get("analysis_tier") == "supplementary":
            row["keep_drop_decision"] = "Supplementary"
            rule = "Supplementary only; current data are not sufficient for main durability analysis"
            row["retention_rule"] = rule if not row.get("retention_rule") else f"{row['retention_rule']}; {rule}"


def source_column(metric: MetricSpec, fieldnames: list[str]) -> str:
    for alias in metric.aliases:
        if alias in fieldnames:
            return alias
    return ""


def relative_stress_stats(values: list[float], metric: MetricSpec) -> tuple[float | None, float | None, float | None]:
    if metric.threshold is not None or not values:
        return None, None, None
    percentile = RELATIVE_STRESS_PERCENTILE
    threshold = float(np.percentile(np.asarray(values, dtype=float), percentile))
    frequency_pct = 100.0 * sum(value > threshold for value in values) / len(values)
    return percentile, threshold, frequency_pct


def driver_response_attribution(records: list[dict[str, object]], metric: MetricSpec) -> dict[str, object]:
    result: dict[str, object] = {
        "driver_response_n": "",
        "driver_response_r2": None,
        "driver_response_condition_number": None,
        "driver_response_dominant_driver": "",
        "driver_response_dominant_abs_beta": None,
    }
    for driver_key in ATTRIBUTION_DRIVER_KEYS:
        result[f"driver_response_beta_{driver_key}"] = None

    if metric.role != "response":
        return result

    y_values: list[float] = []
    x_values: list[list[float]] = []
    for record in records:
        y_value = to_float(record.get(metric.key))
        driver_values = [to_float(record.get(driver_key)) for driver_key in ATTRIBUTION_DRIVER_KEYS]
        if y_value is None or any(value is None for value in driver_values):
            continue
        y_values.append(y_value)
        x_values.append([float(value) for value in driver_values if value is not None])

    if len(y_values) < len(ATTRIBUTION_DRIVER_KEYS) + 5:
        return result

    y_array = np.asarray(y_values, dtype=float)
    x_array = np.asarray(x_values, dtype=float)

    y_std = float(y_array.std())
    x_stds = x_array.std(axis=0)
    valid_predictors = [index for index, spread in enumerate(x_stds) if float(spread) > 1e-12]
    if y_std <= 1e-12 or not valid_predictors:
        return result

    y_scaled = (y_array - y_array.mean()) / y_std
    x_scaled = x_array[:, valid_predictors]
    x_scaled = (x_scaled - x_scaled.mean(axis=0)) / x_scaled.std(axis=0)

    regularizer = ATTRIBUTION_RIDGE_ALPHA * np.eye(x_scaled.shape[1])
    coefficients = np.linalg.solve(x_scaled.T @ x_scaled + regularizer, x_scaled.T @ y_scaled)
    predicted = x_scaled @ coefficients
    residual_sum_squares = float(np.sum((y_scaled - predicted) ** 2))
    total_sum_squares = float(np.sum((y_scaled - y_scaled.mean()) ** 2))
    r2 = None if total_sum_squares <= 1e-12 else max(0.0, min(1.0, 1.0 - residual_sum_squares / total_sum_squares))

    beta_by_driver = {
        ATTRIBUTION_DRIVER_KEYS[predictor_index]: float(beta)
        for predictor_index, beta in zip(valid_predictors, coefficients)
    }
    if beta_by_driver:
        dominant_driver, dominant_beta = max(beta_by_driver.items(), key=lambda item: abs(item[1]))
        result["driver_response_dominant_driver"] = dominant_driver
        result["driver_response_dominant_abs_beta"] = abs(dominant_beta)
    result["driver_response_n"] = len(y_values)
    result["driver_response_r2"] = r2
    result["driver_response_condition_number"] = float(np.linalg.cond(x_scaled))
    for driver_key, beta in beta_by_driver.items():
        result[f"driver_response_beta_{driver_key}"] = beta
    return result


def driver_block_variance_share(
    records: list[dict[str, object]],
    metric: MetricSpec,
) -> dict[str, object]:
    """Compute driver-block-level variance share for a response metric.

    Aggregates squared standardized betas within each driver block and
    expresses them as fractions of total explained variance. Used for
    NCC Fig 3 (variance decomposition by driver block).
    """
    result: dict[str, object] = {"block_decomposition_n": ""}
    for block_name in DRIVER_BLOCK_REPRESENTATIVES:
        result[f"block_beta2_share_{block_name}"] = None
    result["block_dominant"] = ""
    result["block_dominant_share"] = None
    result["block_decomposition_r2"] = None

    if metric.role != "response":
        return result

    # Reuse the per-key attribution to get betas
    attribution = driver_response_attribution(records, metric)
    if not attribution.get("driver_response_n"):
        return result

    # Collect betas per driver block
    block_beta2: dict[str, float] = {}
    for block_name, block_keys in DRIVER_BLOCK_REPRESENTATIVES.items():
        beta2_sum = 0.0
        for key in block_keys:
            beta = attribution.get(f"driver_response_beta_{key}")
            if beta is not None:
                beta2_sum += float(beta) ** 2
        block_beta2[block_name] = beta2_sum

    total_beta2 = sum(block_beta2.values())
    if total_beta2 <= 1e-12:
        return result

    result["block_decomposition_n"] = attribution["driver_response_n"]
    result["block_decomposition_r2"] = attribution["driver_response_r2"]
    for block_name, beta2 in block_beta2.items():
        result[f"block_beta2_share_{block_name}"] = beta2 / total_beta2

    dominant_block = max(block_beta2, key=block_beta2.get)  # type: ignore[arg-type]
    result["block_dominant"] = dominant_block
    result["block_dominant_share"] = block_beta2[dominant_block] / total_beta2

    return result


def build_screening_rows(
    records: list[dict[str, object]],
    grouped_stats: dict[tuple[str | None, str | None, str | None], dict[str, dict[str, object]]],
    fieldnames: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        all_values = [to_float(record.get(metric.key)) for record in records]
        present_values = [value for value in all_values if value is not None]
        coverage_pct = 100.0 * len(present_values) / len(records) if records else 0.0
        group_metric_stats = [stats[metric.key] for stats in grouped_stats.values()]

        breach_frequency_pct = None
        if metric.threshold is not None and present_values:
            breach_frequency_pct = 100.0 * sum(breaches(value, metric) for value in present_values) / len(present_values)
        relative_stress_percentile, relative_stress_threshold, relative_stress_frequency_pct = relative_stress_stats(present_values, metric)

        slopes = clean([to_float(stats.get("temporal_slope_per_year")) for stats in group_metric_stats])
        accels = clean([to_float(stats.get("temporal_acceleration_per_year2")) for stats in group_metric_stats])
        volatilities = clean([to_float(stats.get("volatility_change")) for stats in group_metric_stats])
        jumps = clean([to_float(stats.get("maximum_jump")) for stats in group_metric_stats])
        baseline_values = clean([to_float(stats.get("baseline_value")) for stats in group_metric_stats])
        absolute_changes = clean([to_float(stats.get("absolute_change")) for stats in group_metric_stats])
        log_ratio_changes = clean([to_float(stats.get("log_ratio_change")) for stats in group_metric_stats])
        standardized_changes = clean([to_float(stats.get("standardized_change")) for stats in group_metric_stats])

        redundancy = redundancy_score(records, metric) if coverage_pct > 0 else None
        row = {
            "role": metric.role,
            "family": metric.family,
            "subfamily": metric.subfamily,
            "metric": metric.key,
            "label": metric.label,
            "unit": metric.unit,
            "analysis_tier": metric.analysis_tier,
            "source_column": source_column(metric, fieldnames),
            "optional": metric.optional,
            "threshold": metric.threshold,
            "threshold_direction": metric.threshold_direction if metric.threshold is not None else "",
            "threshold_standard": metric.standard,
            "failure_relevant": metric.failure_relevant,
            "interpretation": metric.interpretation,
            "coverage_pct": coverage_pct,
            "baseline_value_median": median(baseline_values) if baseline_values else None,
            "mean_abs_absolute_change": mean_or_none([abs(value) for value in absolute_changes]),
            "mean_abs_log_ratio_change": mean_or_none([abs(value) for value in log_ratio_changes]),
            "mean_abs_standardized_change": mean_or_none([abs(value) for value in standardized_changes]),
            "mean_abs_temporal_slope": mean_or_none([abs(value) for value in slopes]),
            "mean_abs_temporal_acceleration": mean_or_none([abs(value) for value in accels]),
            "mean_volatility_change": mean_or_none(volatilities),
            "maximum_jump": max(jumps) if jumps else None,
            "breach_frequency_pct": breach_frequency_pct,
            "failure_probability_pct": breach_frequency_pct if metric.failure_relevant else None,
            "relative_stress_percentile": relative_stress_percentile,
            "relative_stress_threshold": relative_stress_threshold,
            "relative_stress_frequency_pct": relative_stress_frequency_pct,
            "relative_stress_use": "auxiliary_screening_only" if relative_stress_threshold is not None else "",
            "break_year_count": sum(1 for stats in group_metric_stats if stats.get("break_year") not in ("", "Never", None)),
            "sustained_failure_year_count": sum(
                1 for stats in group_metric_stats if stats.get("sustained_failure_year") not in ("", "Never", None)
            ),
            "climate_discrimination_score": climate_discrimination(records, metric),
            "redundancy_max_abs_r": redundancy,
            "redundancy_scope": "same_subfamily",
            "redundancy_flag": "Yes" if redundancy is not None and redundancy >= REDUNDANCY_THRESHOLD else "No",
            "retention_rule": "",
        }
        row.update(driver_response_attribution(records, metric))
        rows.append(row)

    coverage_scores = scale_scores([to_float(row["coverage_pct"]) for row in rows], higher_is_better=True)
    slope_scores = scale_scores([to_float(row["mean_abs_temporal_slope"]) for row in rows], higher_is_better=True)
    breach_signal_scores = scale_scores(
        [
            to_float(row["breach_frequency_pct"])
            if row["breach_frequency_pct"] is not None
            else to_float(row["mean_abs_standardized_change"])
            for row in rows
        ],
        higher_is_better=True,
    )
    discrimination_scores = scale_scores([to_float(row["climate_discrimination_score"]) for row in rows], higher_is_better=True)
    volatility_scores = scale_scores([to_float(row["mean_volatility_change"]) for row in rows], higher_is_better=False)
    uniqueness_scores = scale_scores([to_float(row["redundancy_max_abs_r"]) for row in rows], higher_is_better=False)

    for index, row in enumerate(rows):
        score_value = (
            0.24 * slope_scores[index]
            + 0.18 * breach_signal_scores[index]
            + 0.22 * discrimination_scores[index]
            + 0.16 * coverage_scores[index]
            + 0.10 * volatility_scores[index]
            + 0.10 * uniqueness_scores[index]
        )
        row["screening_decision_raw"] = decide_from_score(score_value, to_float(row["coverage_pct"]) or 0.0)
        row["screening_score"] = score_value
        row["keep_drop_decision"] = row["screening_decision_raw"]

    apply_retention_rules(rows)

    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys

    with open(path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_fieldnames() -> list[str]:
    fields = [
        "scenario",
        "building",
        "city",
        "baseline_year",
        "first_available_year",
        "final_available_year",
        "absolute_standard_break_year",
        "baseline_failure_severity",
        "main_break_year",
        "main_stress_emergence_year",
        "stress_emergence_delta",
        "main_sustained_failure_year",
        "break_year",
        "sustained_failure_year",
        "first_trigger_metric",
        "failure_probability_pct",
        "failure_occurrence_count",
        "failure_occurrence_ratio_pct",
        "failure_occurrence_ratio_threshold_pct",
        *[
            field
            for window_years in FAILURE_OCCURRENCE_WINDOWS
            for field in (
                f"failure_occurrence_{window_years}y_year",
                f"failure_occurrence_{window_years}y_window_count",
            )
        ],
        "operational_adequacy_break_year",
        "operational_adequacy_sustained_failure_year",
        "habitability_break_year",
        "habitability_sustained_failure_year",
        "operational_adequacy__first_trigger_year",
        "habitability__first_trigger_year",
    ]
    for metric in METRICS:
        for suffix in SUMMARY_SUFFIXES:
            fields.append(f"{metric.prefix}__{suffix}")
    return fields


def screening_fieldnames() -> list[str]:
    return [
        "role",
        "family",
        "subfamily",
        "metric",
        "label",
        "unit",
        "analysis_tier",
        "source_column",
        "optional",
        "threshold",
        "threshold_direction",
        "threshold_standard",
        "failure_relevant",
        "interpretation",
        "coverage_pct",
        "baseline_value_median",
        "mean_abs_absolute_change",
        "mean_abs_log_ratio_change",
        "mean_abs_standardized_change",
        "mean_abs_temporal_slope",
        "mean_abs_temporal_acceleration",
        "mean_volatility_change",
        "maximum_jump",
        "breach_frequency_pct",
        "failure_probability_pct",
        "relative_stress_percentile",
        "relative_stress_threshold",
        "relative_stress_frequency_pct",
        "relative_stress_use",
        "break_year_count",
        "sustained_failure_year_count",
        "climate_discrimination_score",
        "driver_response_n",
        "driver_response_r2",
        "driver_response_condition_number",
        "driver_response_dominant_driver",
        "driver_response_dominant_abs_beta",
        *[f"driver_response_beta_{driver_key}" for driver_key in ATTRIBUTION_DRIVER_KEYS],
        "redundancy_max_abs_r",
        "redundancy_scope",
        "redundancy_flag",
        "screening_decision_raw",
        "screening_score",
        "keep_drop_decision",
        "retention_rule",
    ]


def registry_fieldnames() -> list[str]:
    return [
        "role",
        "family",
        "subfamily",
        "metric",
        "label",
        "unit",
        "analysis_tier",
        "source_column",
        "aliases",
        "threshold",
        "threshold_direction",
        "threshold_standard",
        "failure_relevant",
        "break_year_use",
        "relative_stress_use",
        "optional",
        "interpretation",
        "driver_block_representative_rule",
    ]


def build_registry_rows(fieldnames: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        representative_rule = ""
        if metric.role == "driver":
            representatives = DRIVER_BLOCK_REPRESENTATIVES.get(metric.subfamily, ())
            if metric.key in representatives:
                representative_rule = "Retain at least one representative in this driver block"
                if metric.subfamily == "demand_background":
                    representative_rule = "Retain both CDD and HDD in this driver block"
        rows.append(
            {
                "role": metric.role,
                "family": metric.family,
                "subfamily": metric.subfamily,
                "metric": metric.key,
                "label": metric.label,
                "unit": metric.unit,
                "analysis_tier": metric.analysis_tier,
                "source_column": source_column(metric, fieldnames),
                "aliases": "; ".join(metric.aliases),
                "threshold": metric.threshold,
                "threshold_direction": metric.threshold_direction if metric.threshold is not None else "",
                "threshold_standard": metric.standard,
                "failure_relevant": metric.failure_relevant,
                "break_year_use": "standard_threshold_response" if metric.failure_relevant else "",
                "relative_stress_use": "p90 auxiliary only" if metric.threshold is None else "",
                "optional": metric.optional,
                "interpretation": metric.interpretation,
                "driver_block_representative_rule": representative_rule,
            }
        )
    return rows


def block_decomposition_fieldnames() -> list[str]:
    return [
        "climate_type",
        "city",
        "role",
        "family",
        "subfamily",
        "metric",
        "label",
        "block_decomposition_n",
        "block_decomposition_r2",
        *[f"block_beta2_share_{block}" for block in DRIVER_BLOCK_REPRESENTATIVES],
        "block_dominant",
        "block_dominant_share",
    ]


def climate_type_screening_fieldnames() -> list[str]:
    base = screening_fieldnames()
    return ["climate_type"] + base


def main() -> None:
    args = parse_args()
    input_csv = Path(args.file).resolve()
    output_csv = Path(args.output).resolve()
    screening_output_csv = Path(args.screening_output).resolve()
    registry_output_csv = Path(args.registry_output).resolve()
    climate_type_screening_csv = Path(args.climate_type_screening_output).resolve()
    block_decomposition_csv = Path(args.block_decomposition_output).resolve()

    rows: list[dict[str, object]] = []
    with open(input_csv, encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        required_columns = {"scenario", "building", "city", "year"}
        if not reader.fieldnames or not required_columns.issubset(set(reader.fieldnames)):
            raise SystemExit(f"Missing required columns in {input_csv}")

        source_fieldnames = list(reader.fieldnames)
        for source in reader:
            rows.append(normalize_row(source))

    grouped: defaultdict[tuple[str | None, str | None, str | None], list[dict[str, object]]] = defaultdict(list)
    skipped_rows = 0
    for row in rows:
        if row["year"] is None:
            skipped_rows += 1
            continue
        grouped[(row["scenario"], row["building"], row["city"])].append(row)

    summary_rows: list[dict[str, object]] = []
    grouped_stats: dict[tuple[str | None, str | None, str | None], dict[str, dict[str, object]]] = {}
    for group_key, records in grouped.items():
        scenario, building, city = group_key
        result, stats = summarize(records, args.baseline_year, args.sustained_years, args.occurrence_ratio_threshold)
        result.update({"scenario": scenario, "building": building, "city": city})
        summary_rows.append(result)
        grouped_stats[group_key] = stats

    summary_rows.sort(key=lambda row: (str(row.get("scenario")), str(row.get("building")), str(row.get("city"))))
    write_csv(output_csv, summary_rows, summary_fieldnames())

    valid_rows = [row for row in rows if row.get("year") is not None]
    screening_rows = build_screening_rows(valid_rows, grouped_stats, source_fieldnames)
    write_csv(screening_output_csv, screening_rows, screening_fieldnames())
    write_csv(registry_output_csv, build_registry_rows(source_fieldnames), registry_fieldnames())

    # ── Per-climate-type screening (NCC Fig 2) ──────────────────────────
    climate_type_screening_rows: list[dict[str, object]] = []
    by_climate_type: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in valid_rows:
        city = str(row.get("city", ""))
        climate_type = CITY_KOPPEN_MAP.get(city, "Unknown")
        by_climate_type[climate_type].append(row)

    for climate_type, ct_rows in sorted(by_climate_type.items()):
        # Build grouped_stats for this climate type only
        ct_grouped: defaultdict[tuple[str | None, str | None, str | None], list[dict[str, object]]] = defaultdict(list)
        for row in ct_rows:
            ct_grouped[(row["scenario"], row["building"], row["city"])].append(row)
        ct_grouped_stats: dict[tuple[str | None, str | None, str | None], dict[str, dict[str, object]]] = {}
        for group_key, records in ct_grouped.items():
            _, stats = summarize(records, args.baseline_year, args.sustained_years, args.occurrence_ratio_threshold)
            ct_grouped_stats[group_key] = stats

        ct_screening = build_screening_rows(ct_rows, ct_grouped_stats, source_fieldnames)
        for row in ct_screening:
            row["climate_type"] = climate_type
        climate_type_screening_rows.extend(ct_screening)

    write_csv(climate_type_screening_csv, climate_type_screening_rows, climate_type_screening_fieldnames())

    # ── Driver-block variance decomposition (NCC Fig 3) ─────────────────
    block_decomposition_rows: list[dict[str, object]] = []
    for climate_type, ct_rows in sorted(by_climate_type.items()):
        cities_in_type = sorted(set(str(row.get("city", "")) for row in ct_rows))
        for metric in METRICS:
            if metric.role != "response":
                continue
            decomp = driver_block_variance_share(ct_rows, metric)
            decomp.update({
                "climate_type": climate_type,
                "city": "; ".join(cities_in_type),
                "role": metric.role,
                "family": metric.family,
                "subfamily": metric.subfamily,
                "metric": metric.key,
                "label": metric.label,
            })
            block_decomposition_rows.append(decomp)

    write_csv(block_decomposition_csv, block_decomposition_rows, block_decomposition_fieldnames())

    # ── Summary ─────────────────────────────────────────────────────────
    available_count = sum(
        1
        for metric in METRICS
        if any(to_float(row.get(metric.key)) is not None for row in valid_rows)
    )

    print(f"Input rows        : {len(rows)}")
    print(f"Skipped rows      : {skipped_rows}")
    print(f"Summary rows      : {len(summary_rows)}")
    print(f"Metric specs      : {len(METRICS)}")
    print(f"Metrics with data : {available_count}")
    print(f"Climate types     : {len(by_climate_type)}")
    print(f"Saved summary     : {output_csv}")
    print(f"Saved screening   : {screening_output_csv}")
    print(f"Saved registry    : {registry_output_csv}")
    print(f"Saved CT screening: {climate_type_screening_csv}")
    print(f"Saved block decomp: {block_decomposition_csv}")


if __name__ == "__main__":
    main()
