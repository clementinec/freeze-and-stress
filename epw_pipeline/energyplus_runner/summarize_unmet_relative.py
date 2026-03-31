#!/usr/bin/env python3
"""Summarize unmet hours relative to 2025 baseline and occupied-hour share."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


MODEL_CITY_KEYS = {
    "Los_Angeles": "LosAngeles",
    "Miami": "Miami",
    "Montreal": "Montreal",
    "Toronto": "Toronto",
}


def safe_float(value: str | None) -> float | None:
    """Convert a CSV value to float."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def percentile_median(values: list[float]) -> float | None:
    """Return median for a numeric list."""
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def parse_schedule_compact(schedule_data: list[dict]) -> dict[str, list[float]]:
    """Parse EnergyPlus Schedule:Compact entries into per-day-type hourly values."""
    schedule_dict: dict[str, list[float]] = {}
    current_day_type: str | None = None
    current_schedule: list[float | None] = []

    i = 0
    while i < len(schedule_data):
        field = schedule_data[i].get("field")
        if isinstance(field, str):
            field_upper = field.upper()

            if field_upper.startswith("FOR:"):
                if current_day_type and current_schedule:
                    filled = forward_fill_schedule(current_schedule)
                    schedule_dict[current_day_type] = filled
                current_day_type = field.split(":", 1)[1].strip()
                current_schedule = [None] * 24

            elif field_upper.startswith("UNTIL:") and current_day_type is not None:
                time_str = field.split(":", 1)[1].strip()
                hour = int(time_str.split(":")[0])
                target_hour = 23 if hour == 24 else max(hour - 1, 0)

                if i + 1 < len(schedule_data):
                    next_field = schedule_data[i + 1].get("field")
                    if isinstance(next_field, (int, float)):
                        for h in range(target_hour + 1):
                            if current_schedule[h] is None:
                                current_schedule[h] = float(next_field)
                    i += 1
        i += 1

    if current_day_type and current_schedule:
        schedule_dict[current_day_type] = forward_fill_schedule(current_schedule)

    return schedule_dict


def forward_fill_schedule(values: list[float | None]) -> list[float]:
    """Forward-fill one 24-hour schedule."""
    result: list[float] = []
    last_valid = 0.0
    for value in values:
        if value is None:
            result.append(last_valid)
        else:
            last_valid = float(value)
            result.append(last_valid)
    return result


def get_hourly_value(schedule_dict: dict[str, list[float]], dt: datetime) -> float:
    """Resolve the schedule value for one hour."""
    day_of_week = dt.weekday()  # Monday=0
    hour = dt.hour

    for key, values in schedule_dict.items():
        lower = key.lower()
        if ("weekdays" in lower or "weekday" in lower) and "weekend" not in lower and day_of_week < 5:
            return values[hour]
        if "saturday" in lower and day_of_week == 5:
            return values[hour]
        if "sunday" in lower and day_of_week == 6:
            return values[hour]
        if ("weekends" in lower or "weekend" in lower) and day_of_week >= 5:
            return values[hour]
        if "alldays" in lower:
            return values[hour]

    for key, values in schedule_dict.items():
        if "allotherdays" in key.lower():
            return values[hour]

    first_key = next(iter(schedule_dict))
    return schedule_dict[first_key][hour]


def load_model(path: Path) -> dict:
    """Load one epJSON model."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_people_schedule_names(model: dict) -> list[str]:
    """Return unique people schedule names present in the model."""
    names: list[str] = []
    seen: set[str] = set()
    for people_obj in model.get("People", {}).values():
        name = people_obj.get("number_of_people_schedule_name")
        if isinstance(name, str) and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def compute_occupied_hours(model: dict, year: int) -> int:
    """Compute annual occupied hours based on any People schedule being positive."""
    schedule_compact = model.get("Schedule:Compact", {})
    parsed_schedules: list[dict[str, list[float]]] = []

    for sched_name in get_people_schedule_names(model):
        sched = schedule_compact.get(sched_name, {})
        data = sched.get("data", [])
        if data:
            parsed_schedules.append(parse_schedule_compact(data))

    if not parsed_schedules:
        return 0

    dt = datetime(year, 1, 1, 0, 0)
    end = datetime(year + 1, 1, 1, 0, 0)
    occupied_hours = 0

    while dt < end:
        if any(get_hourly_value(schedule, dt) > 0.0 for schedule in parsed_schedules):
            occupied_hours += 1
        dt += timedelta(hours=1)

    return occupied_hours


def model_path_for(model_dir: Path, building: str, city: str) -> Path:
    """Map result-tree building/city keys to the frozen-model filename."""
    city_key = MODEL_CITY_KEYS[city]
    return model_dir / f"{building}_{city_key}_frozen_detailed.epJSON"


def load_metric_rows(metric_files: list[Path]) -> list[dict[str, object]]:
    """Load unmet-hour metric rows from one or more CSV files."""
    rows: list[dict[str, object]] = []
    for metric_file in metric_files:
        with metric_file.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "scenario": row["scenario"],
                        "building": row["building"],
                        "city": row["city"],
                        "year": int(row["year"]),
                        "unmet_hours": safe_float(row.get("abups_occupied_cooling_not_met_hours")),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write CSV rows with stable field ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize unmet hours relative to baseline and occupied hours")
    parser.add_argument("--metrics-files", type=Path, nargs="+", required=True, help="Input annual_metrics.csv files")
    parser.add_argument("--model-dir", type=Path, default=Path("frozen_models"), help="Frozen model directory")
    parser.add_argument("--output-dir", type=Path, default=Path("unmet_relative_summary"), help="Output directory")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    model_dir = (script_dir / args.model_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    rows = load_metric_rows(args.metrics_files)

    occupied_hours_cache: dict[tuple[str, str, int], int] = {}
    model_cache: dict[tuple[str, str], dict] = {}

    for row in rows:
        cache_key = (str(row["building"]), str(row["city"]), int(row["year"]))
        model_key = (str(row["building"]), str(row["city"]))
        if model_key not in model_cache:
            model_cache[model_key] = load_model(model_path_for(model_dir, str(row["building"]), str(row["city"])))
        if cache_key not in occupied_hours_cache:
            occupied_hours_cache[cache_key] = compute_occupied_hours(model_cache[model_key], int(row["year"]))

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        occupied_hours = occupied_hours_cache[(str(row["building"]), str(row["city"]), int(row["year"]))]
        unmet = row["unmet_hours"]
        pct_unmet = (100.0 * unmet / occupied_hours) if occupied_hours and unmet is not None else None
        enriched = dict(row)
        enriched["occupied_hours"] = occupied_hours
        enriched["pct_occupied_hours_unmet"] = pct_unmet
        grouped[(str(row["scenario"]), str(row["building"]), str(row["city"]))].append(enriched)

    all_rows: list[dict[str, object]] = []
    city_snapshot_rows: list[dict[str, object]] = []
    building_summary_rows: list[dict[str, object]] = []

    for key, records in sorted(grouped.items()):
        scenario, building, city = key
        ordered = sorted(records, key=lambda item: int(item["year"]))
        baseline = next((row for row in ordered if int(row["year"]) == 2025), None)
        baseline_unmet = baseline["unmet_hours"] if baseline else None
        baseline_pct = baseline["pct_occupied_hours_unmet"] if baseline else None

        for row in ordered:
            row["baseline_2025_unmet_hours"] = baseline_unmet
            row["delta_vs_2025_unmet_hours"] = (row["unmet_hours"] - baseline_unmet) if row["unmet_hours"] is not None and baseline_unmet is not None else None
            row["baseline_2025_pct_occupied_unmet"] = baseline_pct
            row["delta_vs_2025_pct_points"] = (row["pct_occupied_hours_unmet"] - baseline_pct) if row["pct_occupied_hours_unmet"] is not None and baseline_pct is not None else None
            all_rows.append(row)

        latest = ordered[-1]
        city_snapshot_rows.append(
            {
                "scenario": scenario,
                "building": building,
                "city": city,
                "occupied_hours_per_year": latest["occupied_hours"],
                "baseline_2025_unmet_hours": baseline_unmet,
                "baseline_2025_pct_occupied_unmet": baseline_pct,
                "latest_year": latest["year"],
                "latest_unmet_hours": latest["unmet_hours"],
                "latest_delta_vs_2025_unmet_hours": latest["delta_vs_2025_unmet_hours"],
                "latest_pct_occupied_unmet": latest["pct_occupied_hours_unmet"],
                "latest_delta_vs_2025_pct_points": latest["delta_vs_2025_pct_points"],
                "max_unmet_hours_observed": max(float(row["unmet_hours"]) for row in ordered if row["unmet_hours"] is not None),
                "max_pct_occupied_unmet_observed": max(float(row["pct_occupied_hours_unmet"]) for row in ordered if row["pct_occupied_hours_unmet"] is not None),
                "n_years": len(ordered),
            }
        )

    summary_grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in city_snapshot_rows:
        summary_grouped[(str(row["scenario"]), str(row["building"]))].append(row)

    for (scenario, building), records in sorted(summary_grouped.items()):
        baseline_raw = [float(row["baseline_2025_unmet_hours"]) for row in records if row["baseline_2025_unmet_hours"] is not None]
        latest_raw = [float(row["latest_unmet_hours"]) for row in records if row["latest_unmet_hours"] is not None]
        latest_delta = [float(row["latest_delta_vs_2025_unmet_hours"]) for row in records if row["latest_delta_vs_2025_unmet_hours"] is not None]
        baseline_pct = [float(row["baseline_2025_pct_occupied_unmet"]) for row in records if row["baseline_2025_pct_occupied_unmet"] is not None]
        latest_pct = [float(row["latest_pct_occupied_unmet"]) for row in records if row["latest_pct_occupied_unmet"] is not None]
        latest_delta_pct = [float(row["latest_delta_vs_2025_pct_points"]) for row in records if row["latest_delta_vs_2025_pct_points"] is not None]
        occupied_hours = [int(row["occupied_hours_per_year"]) for row in records]

        building_summary_rows.append(
            {
                "scenario": scenario,
                "building": building,
                "cities_with_data": len(records),
                "median_occupied_hours_per_year": percentile_median([float(value) for value in occupied_hours]),
                "median_2025_unmet_hours": percentile_median(baseline_raw),
                "median_latest_unmet_hours": percentile_median(latest_raw),
                "median_latest_delta_vs_2025_unmet_hours": percentile_median(latest_delta),
                "median_2025_pct_occupied_unmet": percentile_median(baseline_pct),
                "median_latest_pct_occupied_unmet": percentile_median(latest_pct),
                "median_latest_delta_vs_2025_pct_points": percentile_median(latest_delta_pct),
                "latest_year_min": min(int(row["latest_year"]) for row in records),
                "latest_year_max": max(int(row["latest_year"]) for row in records),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "annual_unmet_relative.csv", all_rows)
    write_csv(output_dir / "city_snapshot.csv", city_snapshot_rows)
    write_csv(output_dir / "building_scenario_summary.csv", building_summary_rows)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metric_files": [str(path.resolve()) for path in args.metrics_files],
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "annual_rows": len(all_rows),
        "city_snapshots": len(city_snapshot_rows),
        "building_scenario_rows": len(building_summary_rows),
        "note": "Percent occupied unmet uses occupied-hour counts derived from People schedule unions in frozen models.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
