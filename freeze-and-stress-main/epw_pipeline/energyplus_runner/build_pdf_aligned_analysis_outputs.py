#!/usr/bin/env python3
"""Build PDF-aligned analysis tables after multi-metric break-year screening.

This script keeps the threshold-based break-year workflow intact and adds the
paper analysis objects requested by F_S_framework.pdf:

- long master panel with 2025-2034 decadal baseline changes
- decadal metric panel
- annual and decadal family scores
- climate-profile classification table
- lightweight robustness/status checklist
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

import pandas as pd

from compute_break_years_multi_metric import (
    CITY_KOPPEN_MAP,
    DRIVER_BLOCK_REPRESENTATIVES,
    FAILURE_METRICS,
    METRICS,
    MetricSpec,
    breaches,
    normalize_row,
    source_column,
    to_float,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "metric_exports" / "annual_metrics_extended.csv"
if not DEFAULT_INPUT.exists():
    DEFAULT_INPUT = SCRIPT_DIR / "metric_exports" / "annual_metrics.csv"

SUMMARY_CSV = SCRIPT_DIR / "paper_metrics_summary.csv"
REGISTRY_CSV = SCRIPT_DIR / "paper_metric_registry.csv"
SCREENING_CSV = SCRIPT_DIR / "paper_metric_screening.csv"
BLOCK_DECOMP_CSV = SCRIPT_DIR / "paper_block_decomposition.csv"

MASTER_PANEL_CSV = SCRIPT_DIR / "paper_master_panel.csv"
DECADAL_PANEL_CSV = SCRIPT_DIR / "paper_master_panel_decadal.csv"
FAMILY_SCORES_CSV = SCRIPT_DIR / "paper_family_scores.csv"
FAMILY_SCORES_DECADAL_CSV = SCRIPT_DIR / "paper_family_scores_decadal.csv"
PROFILE_CSV = SCRIPT_DIR / "paper_climate_profile_classification.csv"
ROBUSTNESS_CSV = SCRIPT_DIR / "paper_robustness_status.csv"

BASELINE_START = 2025
BASELINE_END = 2034
EPSILON = 1e-6

DECADES = (
    (2025, 2034),
    (2035, 2044),
    (2045, 2054),
    (2055, 2064),
    (2065, 2074),
    (2075, 2084),
    (2085, 2094),
    (2095, 2100),
)

ACTION_MAP = {
    "Am": "Envelope heat control + cooling adequacy check",
    "BWh": "Peak comfort, shading, and operating controls",
    "Csb": "Short-duration overheating controls",
    "Cfb": "Heating adequacy and heat-pump timing",
    "Dfa": "Monitor; no severe-transition retrofit trigger",
    "Dfb": "Intermittent overheating monitoring in cold climate",
}

FAMILY_GROUPS = {
    "operational_adequacy": "operational_adequacy",
    "habitability": "habitability",
    "durability": "durability_cautious",
    "energy_burden": "energy_burden_secondary",
}


def stdev(values: Iterable[float]) -> float | None:
    values = [value for value in values if value is not None and not pd.isna(value)]
    if not values:
        return None
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def decade_label(year: int | float | None) -> str:
    if year is None or pd.isna(year):
        return ""
    year_int = int(year)
    for start, end in DECADES:
        if start <= year_int <= end:
            return f"{start}-{end}"
    return ""


def as_number(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_year(value: object) -> int | str:
    if value is None or pd.isna(value):
        return "Never"
    text = str(value).strip()
    if not text or text.lower() == "never":
        return "Never"
    return int(float(text))


def metric_lookup() -> dict[str, MetricSpec]:
    return {metric.key: metric for metric in METRICS}


def read_normalized_rows() -> tuple[list[dict[str, object]], list[str]]:
    with open(DEFAULT_INPUT, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        rows = [normalize_row(row) for row in reader]
    return rows, fieldnames


def group_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (str(row.get("scenario")), str(row.get("building")), str(row.get("city")))


def build_baseline_stats(rows: list[dict[str, object]]) -> dict[tuple[str, str, str, str], dict[str, object]]:
    by_group_metric: defaultdict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    by_group_metric_all: defaultdict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        year = row.get("year")
        if year is None:
            continue
        scenario, building, city = group_key(row)
        for metric in METRICS:
            value = as_number(row.get(metric.key))
            if value is None:
                continue
            key = (scenario, building, city, metric.key)
            by_group_metric_all[key].append(value)
            if BASELINE_START <= int(year) <= BASELINE_END:
                by_group_metric[key].append(value)

    stats: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for key, all_values in by_group_metric_all.items():
        baseline_values = by_group_metric.get(key, [])
        baseline_mean = mean(baseline_values) if baseline_values else None
        baseline_sd = stdev(baseline_values)
        fallback_sd = stdev(all_values)
        stats[key] = {
            "baseline_mean": baseline_mean,
            "baseline_sd": baseline_sd,
            "fallback_sd": fallback_sd,
            "baseline_n": len(baseline_values),
        }
    return stats


def driver_snapshot(
    row: dict[str, object],
    stats: dict[tuple[str, str, str, str], dict[str, object]],
) -> dict[str, object]:
    scenario, building, city = group_key(row)
    snapshot: dict[str, object] = {}
    for metric in METRICS:
        if metric.role != "driver":
            continue
        value = as_number(row.get(metric.key))
        stat = stats.get((scenario, building, city, metric.key), {})
        baseline = as_number(stat.get("baseline_mean"))
        sd = as_number(stat.get("baseline_sd")) or as_number(stat.get("fallback_sd"))
        snapshot[f"driver_value_{metric.key}"] = value
        snapshot[f"driver_delta_abs_{metric.key}"] = None if value is None or baseline is None else value - baseline
        snapshot[f"driver_delta_std_{metric.key}"] = (
            None if value is None or baseline is None or not sd or abs(sd) < 1e-12 else (value - baseline) / sd
        )
    return snapshot


def build_master_panel(rows: list[dict[str, object]], source_fieldnames: list[str]) -> pd.DataFrame:
    stats = build_baseline_stats(rows)
    records: list[dict[str, object]] = []
    for row in rows:
        year = row.get("year")
        if year is None:
            continue
        scenario, building, city = group_key(row)
        climate_type = CITY_KOPPEN_MAP.get(city, "Unknown")
        drivers = driver_snapshot(row, stats)
        for metric in METRICS:
            value = as_number(row.get(metric.key))
            stat = stats.get((scenario, building, city, metric.key), {})
            baseline = as_number(stat.get("baseline_mean"))
            baseline_sd = as_number(stat.get("baseline_sd"))
            sd_for_z = baseline_sd or as_number(stat.get("fallback_sd"))
            delta_abs = None if value is None or baseline is None else value - baseline
            delta_pct = None
            if value is not None and baseline is not None and abs(baseline) > 1e-12:
                delta_pct = 100.0 * (value - baseline) / baseline
            delta_log = None
            if value is not None and baseline is not None and value + EPSILON > 0 and baseline + EPSILON > 0:
                delta_log = math.log(value + EPSILON) - math.log(baseline + EPSILON)
            delta_std = None
            if value is not None and baseline is not None and sd_for_z and abs(sd_for_z) > 1e-12:
                delta_std = (value - baseline) / sd_for_z
            records.append(
                {
                    "scenario": scenario,
                    "building": building,
                    "city": city,
                    "climate_type": climate_type,
                    "year": int(year),
                    "decade": decade_label(int(year)),
                    "metric": metric.key,
                    "label": metric.label,
                    "role": metric.role,
                    "family": metric.family,
                    "subfamily": metric.subfamily,
                    "unit": metric.unit,
                    "analysis_tier": metric.analysis_tier,
                    "source_column": source_column(metric, source_fieldnames),
                    "metric_value": value,
                    "baseline_period": f"{BASELINE_START}-{BASELINE_END}",
                    "baseline_value": baseline,
                    "baseline_sd": baseline_sd,
                    "baseline_n": stat.get("baseline_n"),
                    "delta_abs": delta_abs,
                    "delta_pct": delta_pct,
                    "delta_log": delta_log,
                    "delta_std": delta_std,
                    "threshold": metric.threshold,
                    "threshold_direction": metric.threshold_direction if metric.threshold is not None else "",
                    "breached": breaches(value, metric) if metric.threshold is not None else "",
                    "failure_relevant": metric.failure_relevant,
                    **drivers,
                }
            )
    return pd.DataFrame.from_records(records)


def build_decadal_panel(master: pd.DataFrame) -> pd.DataFrame:
    grouped = master.groupby(
        [
            "scenario",
            "building",
            "city",
            "climate_type",
            "decade",
            "metric",
            "label",
            "role",
            "family",
            "subfamily",
            "unit",
            "analysis_tier",
            "failure_relevant",
        ],
        dropna=False,
    )
    result = grouped.agg(
        year_start=("year", "min"),
        year_end=("year", "max"),
        n_years=("year", "count"),
        metric_value_mean=("metric_value", "mean"),
        metric_value_sd=("metric_value", "std"),
        delta_abs_mean=("delta_abs", "mean"),
        delta_log_mean=("delta_log", "mean"),
        delta_std_mean=("delta_std", "mean"),
        breach_frequency_pct=("breached", lambda values: 100.0 * sum(value is True for value in values) / len(values)),
    ).reset_index()
    return result


def build_family_scores(master: pd.DataFrame, screening: pd.DataFrame) -> pd.DataFrame:
    response = master[master["role"] == "response"].copy()
    if "screening_selected_for_analysis" in screening.columns:
        keep_screen = screening[
            (screening["role"] == "response")
            & (screening["screening_selected_for_analysis"].astype(str).str.lower() == "yes")
        ]
    else:
        keep_screen = screening[
            (screening["role"] == "response")
            & (
                screening["keep_drop_decision"].isin(["Keep", "Priority Review", "Review"])
                | (
                    screening["subfamily"].eq("durability")
                    & screening["keep_drop_decision"].eq("Supplementary")
                )
            )
        ]
    keep_metrics = set(keep_screen["metric"])
    response = response[response["metric"].isin(keep_metrics)].copy()
    response["score_family"] = response["subfamily"].map(FAMILY_GROUPS)
    response = response[response["score_family"].notna()].copy()

    rows = []
    keys = ["scenario", "building", "city", "climate_type", "year", "decade", "score_family"]
    for key, group in response.groupby(keys, dropna=False):
        valid = pd.to_numeric(group["delta_std"], errors="coerce").dropna()
        scenario, building, city, climate_type, year, decade, score_family = key
        threshold_metrics = group[group["failure_relevant"].astype(bool)]
        breached_metrics = threshold_metrics[threshold_metrics["breached"] == True]  # noqa: E712
        rows.append(
            {
                "scenario": scenario,
                "building": building,
                "city": city,
                "climate_type": climate_type,
                "year": year,
                "decade": decade,
                "score_family": score_family,
                "family_score": valid.mean() if not valid.empty else None,
                "component_count": len(valid),
                "component_metrics": "; ".join(sorted(group["metric"].unique())),
                "breached_component_count": len(breached_metrics),
                "breached_component_metrics": "; ".join(sorted(breached_metrics["metric"].unique())),
                "interpretation_note": (
                    "Bulk-zone RH only; report cautiously"
                    if score_family == "durability_cautious"
                    else ("Secondary overlay, not main resilience family" if score_family == "energy_burden_secondary" else "")
                ),
            }
        )
    return pd.DataFrame(rows)


def build_family_scores_decadal(family_scores: pd.DataFrame) -> pd.DataFrame:
    grouped = family_scores.groupby(
        ["scenario", "building", "city", "climate_type", "decade", "score_family"],
        dropna=False,
    )
    return grouped.agg(
        year_start=("year", "min"),
        year_end=("year", "max"),
        n_years=("year", "count"),
        family_score_mean=("family_score", "mean"),
        family_score_sd=("family_score", "std"),
        max_breached_component_count=("breached_component_count", "max"),
        component_metrics=("component_metrics", "first"),
        interpretation_note=("interpretation_note", "first"),
    ).reset_index()


def dominant_family_from_scores(
    family_scores_decadal: pd.DataFrame,
    city: str,
    decade: str,
    include_cautious_durability: bool = False,
) -> str:
    families = ["operational_adequacy", "habitability"]
    if include_cautious_durability:
        families.append("durability_cautious")
    subset = family_scores_decadal[
        (family_scores_decadal["city"] == city)
        & (family_scores_decadal["decade"] == decade)
        & (family_scores_decadal["score_family"].isin(families))
    ].copy()
    if subset.empty:
        return ""
    subset["family_score_mean"] = pd.to_numeric(subset["family_score_mean"], errors="coerce")
    subset = subset.dropna(subset=["family_score_mean"])
    if subset.empty:
        return ""
    return str(subset.sort_values("family_score_mean", ascending=False).iloc[0]["score_family"])


def build_profile_table(summary: pd.DataFrame, registry: pd.DataFrame, block_decomp: pd.DataFrame, family_decadal: pd.DataFrame) -> pd.DataFrame:
    label_to_metric = dict(zip(registry["label"], registry["metric"]))
    metric_to_subfamily = dict(zip(registry["metric"], registry["subfamily"]))
    rows = []
    for _, source in summary.sort_values(["scenario", "building", "city"]).iterrows():
        city = str(source["city"])
        climate_type = CITY_KOPPEN_MAP.get(city, "Unknown")
        trigger_text = str(source.get("first_trigger_metric", "Never")).strip()
        trigger_metrics = []
        if trigger_text and trigger_text != "Never":
            trigger_metrics = [label_to_metric.get(part.strip(), part.strip()) for part in trigger_text.split(";")]
        trigger_families = sorted({metric_to_subfamily.get(metric, "") for metric in trigger_metrics if metric_to_subfamily.get(metric, "")})

        driver_counter: Counter[str] = Counter()
        driver_share: dict[str, float] = {}
        for metric in trigger_metrics:
            match = block_decomp[(block_decomp["climate_type"] == climate_type) & (block_decomp["metric"] == metric)]
            if match.empty:
                continue
            dominant = str(match.iloc[0].get("block_dominant", ""))
            share = as_number(match.iloc[0].get("block_dominant_share"))
            if dominant and dominant != "nan":
                driver_counter[dominant] += 1
                driver_share[dominant] = max(driver_share.get(dominant, 0.0), share or 0.0)

        dominant_drivers = [
            driver
            for driver, _ in sorted(
                driver_counter.items(),
                key=lambda item: (-item[1], -driver_share.get(item[0], 0.0), item[0]),
            )
        ]
        dominant_driver = " + ".join(dominant_drivers[:2])
        if not dominant_driver and parse_year(source.get("main_break_year")) == "Never":
            dominant_driver = "No severe-transition driver"

        rows.append(
            {
                "scenario": source.get("scenario"),
                "building": source.get("building"),
                "city": city,
                "climate_type": climate_type,
                "main_break_year": parse_year(source.get("main_break_year")),
                "main_sustained_failure_year": parse_year(source.get("main_sustained_failure_year")),
                "first_trigger_metric": trigger_text,
                "first_trigger_metric_keys": "; ".join(trigger_metrics) if trigger_metrics else "Never",
                "first_breached_family": "; ".join(trigger_families) if trigger_families else "No severe transition",
                "dominant_driver_block": dominant_driver,
                "dominant_driver_share_max": max(driver_share.values()) if driver_share else None,
                "dominant_main_score_family_2045_2054": dominant_family_from_scores(family_decadal, city, "2045-2054"),
                "dominant_main_score_family_2095_2100": dominant_family_from_scores(family_decadal, city, "2095-2100"),
                "dominant_score_family_with_durability_cautious_2045_2054": dominant_family_from_scores(
                    family_decadal,
                    city,
                    "2045-2054",
                    include_cautious_durability=True,
                ),
                "dominant_score_family_with_durability_cautious_2095_2100": dominant_family_from_scores(
                    family_decadal,
                    city,
                    "2095-2100",
                    include_cautious_durability=True,
                ),
                "adaptation_priority": ACTION_MAP.get(climate_type, ""),
            }
        )
    return pd.DataFrame(rows)


def build_robustness_status(master: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    scenario_count = master["scenario"].nunique()
    city_count = master["city"].nunique()
    building_count = master["building"].nunique()
    rows = [
        {
            "check": "Decadal baseline",
            "status": "completed",
            "result": f"{BASELINE_START}-{BASELINE_END} baseline used in paper_master_panel and family-score outputs",
        },
        {
            "check": "Pathway sensitivity",
            "status": "blocked" if scenario_count < 2 else "available",
            "result": f"{scenario_count} pathway(s) present: {', '.join(sorted(map(str, master['scenario'].unique())))}",
        },
        {
            "check": "Leave-one-city-out sensitivity",
            "status": "limited",
            "result": (
                f"{city_count} cities but one representative per Koppen type; leave-one-city-out cannot test within-type replication"
            ),
        },
        {
            "check": "Archetype sensitivity",
            "status": "blocked" if building_count < 2 else "available",
            "result": f"{building_count} building archetype(s) present: {', '.join(sorted(map(str, master['building'].unique())))}",
        },
        {
            "check": "Threshold transition coverage",
            "status": "completed",
            "result": (
                f"{sum(str(value) != 'Never' for value in summary['main_break_year'])}/"
                f"{len(summary)} climate profiles have a main break year by 2100"
            ),
        },
    ]
    return pd.DataFrame(rows)


def write_outputs() -> None:
    rows, source_fieldnames = read_normalized_rows()
    master = build_master_panel(rows, source_fieldnames)
    decadal = build_decadal_panel(master)

    screening = pd.read_csv(SCREENING_CSV)
    family_scores = build_family_scores(master, screening)
    family_scores_decadal = build_family_scores_decadal(family_scores)

    summary = pd.read_csv(SUMMARY_CSV)
    registry = pd.read_csv(REGISTRY_CSV)
    block_decomp = pd.read_csv(BLOCK_DECOMP_CSV)
    profiles = build_profile_table(summary, registry, block_decomp, family_scores_decadal)
    robustness = build_robustness_status(master, summary)

    master.to_csv(MASTER_PANEL_CSV, index=False)
    decadal.to_csv(DECADAL_PANEL_CSV, index=False)
    family_scores.to_csv(FAMILY_SCORES_CSV, index=False)
    family_scores_decadal.to_csv(FAMILY_SCORES_DECADAL_CSV, index=False)
    profiles.to_csv(PROFILE_CSV, index=False)
    robustness.to_csv(ROBUSTNESS_CSV, index=False)

    print(f"Saved master panel       : {MASTER_PANEL_CSV}")
    print(f"Saved decadal panel      : {DECADAL_PANEL_CSV}")
    print(f"Saved family scores      : {FAMILY_SCORES_CSV}")
    print(f"Saved decadal scores     : {FAMILY_SCORES_DECADAL_CSV}")
    print(f"Saved profile table      : {PROFILE_CSV}")
    print(f"Saved robustness status  : {ROBUSTNESS_CSV}")


if __name__ == "__main__":
    write_outputs()
