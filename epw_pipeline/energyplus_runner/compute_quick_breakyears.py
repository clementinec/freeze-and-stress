#!/usr/bin/env python3
"""Compute quick break-year summaries from extracted annual metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median


def safe_float(value: str | None) -> float | None:
    """Parse a CSV cell into float when possible."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_rows(metric_files: list[Path]) -> list[dict[str, object]]:
    """Load and normalize annual-metric rows from one or more CSV files."""
    rows: list[dict[str, object]] = []

    for metric_file in metric_files:
        with metric_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                row = dict(raw)
                row["year"] = int(row["year"])
                row["unmet_hours"] = safe_float(
                    row.get("abups_occupied_cooling_not_met_hours")
                    or row.get("facility_cooling_setpoint_not_met_occupied_time_total_hours")
                )
                rows.append(row)

    return rows


def find_break_year(records: list[dict[str, object]], threshold: float) -> int | str:
    """Return the first year with unmet_hours above threshold."""
    for record in sorted(records, key=lambda item: int(item["year"])):
        unmet = record.get("unmet_hours")
        if unmet is not None and float(unmet) > threshold:
            return int(record["year"])
    return "Never"


def summarize_city_group(
    scenario: str,
    building: str,
    city: str,
    records: list[dict[str, object]],
    mild_threshold: float,
    strong_threshold: float,
) -> dict[str, object]:
    """Summarize one scenario-building-city time series."""
    ordered = sorted(records, key=lambda item: int(item["year"]))
    unmet_values = [float(row["unmet_hours"]) for row in ordered if row.get("unmet_hours") is not None]
    baseline = next((float(row["unmet_hours"]) for row in ordered if int(row["year"]) == 2025 and row.get("unmet_hours") is not None), None)
    final = float(ordered[-1]["unmet_hours"]) if ordered and ordered[-1].get("unmet_hours") is not None else None

    return {
        "scenario": scenario,
        "building": building,
        "city": city,
        "n_years": len(ordered),
        "start_year": int(ordered[0]["year"]),
        "end_year": int(ordered[-1]["year"]),
        "baseline_2025_unmet_hours": baseline,
        "final_unmet_hours": final,
        "max_unmet_hours": max(unmet_values) if unmet_values else None,
        f"break_year_over_{int(mild_threshold)}h": find_break_year(ordered, mild_threshold),
        f"break_year_over_{int(strong_threshold)}h": find_break_year(ordered, strong_threshold),
    }


def summarize_building_scenario(
    scenario: str,
    building: str,
    city_rows: list[dict[str, object]],
    mild_threshold: float,
    strong_threshold: float,
) -> dict[str, object]:
    """Aggregate break years across cities for one scenario-building pair."""
    mild_key = f"break_year_over_{int(mild_threshold)}h"
    strong_key = f"break_year_over_{int(strong_threshold)}h"

    mild_years = [int(row[mild_key]) for row in city_rows if row[mild_key] != "Never"]
    strong_years = [int(row[strong_key]) for row in city_rows if row[strong_key] != "Never"]

    return {
        "scenario": scenario,
        "building": building,
        "cities_with_data": len(city_rows),
        "cities_breaking_mild": len(mild_years),
        "cities_breaking_strong": len(strong_years),
        f"median_break_year_over_{int(mild_threshold)}h": int(median(mild_years)) if mild_years else "Never",
        f"earliest_break_year_over_{int(mild_threshold)}h": min(mild_years) if mild_years else "Never",
        f"latest_break_year_over_{int(mild_threshold)}h": max(mild_years) if mild_years else "Never",
        f"never_break_count_over_{int(mild_threshold)}h": len(city_rows) - len(mild_years),
        f"median_break_year_over_{int(strong_threshold)}h": int(median(strong_years)) if strong_years else "Never",
        f"earliest_break_year_over_{int(strong_threshold)}h": min(strong_years) if strong_years else "Never",
        f"latest_break_year_over_{int(strong_threshold)}h": max(strong_years) if strong_years else "Never",
        f"never_break_count_over_{int(strong_threshold)}h": len(city_rows) - len(strong_years),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV."""
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
    parser = argparse.ArgumentParser(description="Compute quick break-year tables from annual_metrics.csv outputs")
    parser.add_argument(
        "--metrics-files",
        type=Path,
        nargs="+",
        required=True,
        help="One or more annual_metrics.csv files to summarize",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("breakyear_quick"),
        help="Output directory for break-year summary files",
    )
    parser.add_argument(
        "--mild-threshold",
        type=float,
        default=30.0,
        help="Occupied cooling unmet-hours threshold for the early break marker",
    )
    parser.add_argument(
        "--strong-threshold",
        type=float,
        default=100.0,
        help="Occupied cooling unmet-hours threshold for the stronger break marker",
    )
    args = parser.parse_args()

    rows = load_rows(args.metrics_files)

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario"]), str(row["building"]), str(row["city"]))].append(row)

    city_rows = [
        summarize_city_group(
            scenario,
            building,
            city,
            records,
            args.mild_threshold,
            args.strong_threshold,
        )
        for (scenario, building, city), records in sorted(grouped.items())
    ]

    summary_grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in city_rows:
        summary_grouped[(str(row["scenario"]), str(row["building"]))].append(row)

    building_rows = [
        summarize_building_scenario(
            scenario,
            building,
            grouped_rows,
            args.mild_threshold,
            args.strong_threshold,
        )
        for (scenario, building), grouped_rows in sorted(summary_grouped.items())
    ]

    output_dir = args.output_dir.resolve()
    write_csv(output_dir / "city_break_years.csv", city_rows)
    write_csv(output_dir / "building_scenario_break_summary.csv", building_rows)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metric_files": [str(path.resolve()) for path in args.metrics_files],
        "output_dir": str(output_dir),
        "rows_loaded": len(rows),
        "city_series": len(city_rows),
        "building_scenario_groups": len(building_rows),
        "definition": {
            "metric": "occupied cooling setpoint-not-met hours",
            "source_column_priority": [
                "abups_occupied_cooling_not_met_hours",
                "facility_cooling_setpoint_not_met_occupied_time_total_hours",
            ],
            "break_year_over_mild_threshold": args.mild_threshold,
            "break_year_over_strong_threshold": args.strong_threshold,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
