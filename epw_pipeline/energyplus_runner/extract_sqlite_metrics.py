#!/usr/bin/env python3
"""Extract annual metrics from EnergyPlus SQLite outputs for the standalone run."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from output_preset import REPRESENTATIVE_ZONES


METER_ALIASES = {
    "Electricity:Facility": "electricity_facility",
    "Electricity:HVAC": "electricity_hvac",
    "Heating:Electricity": "heating_electricity",
    "Cooling:Electricity": "cooling_electricity",
    "Fans:Electricity": "fans_electricity",
    "InteriorLights:Electricity": "interior_lights_electricity",
    "InteriorEquipment:Electricity": "interior_equipment_electricity",
}

TABULAR_COMFORT_ROWS = {
    "Time Setpoint Not Met During Occupied Heating": "abups_occupied_heating_not_met_hours",
    "Time Setpoint Not Met During Occupied Cooling": "abups_occupied_cooling_not_met_hours",
    "Time Not Comfortable Based on Simple ASHRAE 55-2004": "abups_time_not_comfortable_simple_ashrae55_hours",
}

TABULAR_AREA_ROWS = {
    "Total Building Area": "total_building_area_m2",
    "Net Conditioned Building Area": "conditioned_building_area_m2",
    "Unconditioned Building Area": "unconditioned_building_area_m2",
}

VARIABLE_SPECS = [
    {
        "name": "Site Outdoor Air Drybulb Temperature",
        "preferred_key": "Environment",
        "alias": "outdoor_drybulb_c",
        "kind": "level",
    },
    {
        "name": "Site Outdoor Air Relative Humidity",
        "preferred_key": "Environment",
        "alias": "outdoor_rh_pct",
        "kind": "level",
    },
    {
        "name": "Site Outdoor Air Wetbulb Temperature",
        "preferred_key": "Environment",
        "alias": "outdoor_wetbulb_c",
        "kind": "level",
    },
    {
        "name": "Facility Total Purchased Electricity Rate",
        "preferred_key": "Whole Building",
        "alias": "facility_purchased_electricity_rate",
        "kind": "rate_kw",
    },
    {
        "name": "Facility Total Purchased Electricity Energy",
        "preferred_key": "Whole Building",
        "alias": "facility_purchased_electricity_energy",
        "kind": "energy_kwh",
    },
    {
        "name": "Facility Total HVAC Electricity Demand Rate",
        "preferred_key": "Whole Building",
        "alias": "facility_hvac_electricity_demand_rate",
        "kind": "rate_kw",
    },
    {
        "name": "Facility Total Building Electricity Demand Rate",
        "preferred_key": "Whole Building",
        "alias": "facility_building_electricity_demand_rate",
        "kind": "rate_kw",
    },
    {
        "name": "Facility Total Electricity Demand Rate",
        "preferred_key": "Whole Building",
        "alias": "facility_total_electricity_demand_rate",
        "kind": "rate_kw",
    },
    {
        "name": "Facility Heating Setpoint Not Met Time",
        "preferred_key": "Facility",
        "alias": "facility_heating_setpoint_not_met_time",
        "kind": "time_hours",
    },
    {
        "name": "Facility Cooling Setpoint Not Met Time",
        "preferred_key": "Facility",
        "alias": "facility_cooling_setpoint_not_met_time",
        "kind": "time_hours",
    },
    {
        "name": "Facility Heating Setpoint Not Met While Occupied Time",
        "preferred_key": "Facility",
        "alias": "facility_heating_setpoint_not_met_occupied_time",
        "kind": "time_hours",
    },
    {
        "name": "Facility Cooling Setpoint Not Met While Occupied Time",
        "preferred_key": "Facility",
        "alias": "facility_cooling_setpoint_not_met_occupied_time",
        "kind": "time_hours",
    },
    {
        "name": "Zone Mean Air Temperature",
        "preferred_key": "__REP_ZONE__",
        "alias": "rep_zone_mean_air_temp_c",
        "kind": "level",
    },
    {
        "name": "Zone Air Relative Humidity",
        "preferred_key": "__REP_ZONE__",
        "alias": "rep_zone_rh_pct",
        "kind": "level",
    },
    {
        "name": "Zone Air Humidity Ratio",
        "preferred_key": "__REP_ZONE__",
        "alias": "rep_zone_humidity_ratio",
        "kind": "level",
    },
    {
        "name": "Zone Mean Radiant Temperature",
        "preferred_key": "__REP_ZONE__",
        "alias": "rep_zone_mean_radiant_temp_c",
        "kind": "level",
    },
    {
        "name": "Zone Operative Temperature",
        "preferred_key": "__REP_ZONE__",
        "alias": "rep_zone_operative_temp_c",
        "kind": "level",
    },
    {
        "name": "Zone Thermostat Air Temperature",
        "preferred_key": "__REP_ZONE__",
        "alias": "rep_zone_thermostat_air_temp_c",
        "kind": "level",
    },
    {
        "name": "Zone Air System Sensible Heating Rate",
        "preferred_key": "__REP_ZONE__",
        "alias": "rep_zone_sensible_heating_rate",
        "kind": "rate_kw",
    },
    {
        "name": "Zone Air System Sensible Cooling Rate",
        "preferred_key": "__REP_ZONE__",
        "alias": "rep_zone_sensible_cooling_rate",
        "kind": "rate_kw",
    },
    {
        "name": "Zone Air System Sensible Heating Energy",
        "preferred_key": "__REP_ZONE__",
        "alias": "rep_zone_sensible_heating_energy",
        "kind": "energy_kwh",
    },
    {
        "name": "Zone Air System Sensible Cooling Energy",
        "preferred_key": "__REP_ZONE__",
        "alias": "rep_zone_sensible_cooling_energy",
        "kind": "energy_kwh",
    },
]


def safe_float(value: object) -> float | None:
    """Convert a SQLite/tabular string value to float when possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def discover_sqlite_files(results_root: Path) -> list[dict[str, object]]:
    """Discover simulation SQLite files under results/<scenario>/<building>/<city>/<year>/."""
    discovered: list[dict[str, object]] = []

    for sql_path in sorted(results_root.rglob("eplusout.sql")):
        rel = sql_path.relative_to(results_root)
        parts = rel.parts
        if len(parts) != 5:
            continue

        scenario, building, city, year_text, _ = parts
        if not year_text.isdigit():
            continue

        discovered.append(
            {
                "scenario": scenario,
                "building": building,
                "city": city,
                "year": int(year_text),
                "sql_path": sql_path,
            }
        )

    return discovered


def fetch_all_rows(conn: sqlite3.Connection, query: str, params: tuple = ()) -> list[sqlite3.Row]:
    """Fetch all rows as sqlite Row objects."""
    cur = conn.execute(query, params)
    return cur.fetchall()


def get_active_environment_period_index(conn: sqlite3.Connection) -> int | None:
    """Return the main non-warmup environment period index for the run."""
    row = conn.execute(
        """
        SELECT EnvironmentPeriodIndex, COUNT(*) AS n_rows
        FROM Time
        WHERE WarmupFlag = 0
        GROUP BY EnvironmentPeriodIndex
        ORDER BY n_rows DESC, EnvironmentPeriodIndex ASC
        LIMIT 1
        """
    ).fetchone()
    return int(row["EnvironmentPeriodIndex"]) if row else None


def load_rdd_lookup(conn: sqlite3.Connection) -> tuple[dict[tuple[str, str], sqlite3.Row], dict[str, list[sqlite3.Row]]]:
    """Load the report-data dictionary into exact and by-name lookup maps."""
    rows = fetch_all_rows(
        conn,
        """
        SELECT ReportDataDictionaryIndex, KeyValue, Name, ReportingFrequency, Units
        FROM ReportDataDictionary
        """,
    )

    exact: dict[tuple[str, str], sqlite3.Row] = {}
    by_name: dict[str, list[sqlite3.Row]] = {}

    for row in rows:
        key_value = row["KeyValue"] or ""
        exact[(key_value, row["Name"])] = row
        by_name.setdefault(row["Name"], []).append(row)

    return exact, by_name


def choose_rdd_row(
    exact_lookup: dict[tuple[str, str], sqlite3.Row],
    by_name_lookup: dict[str, list[sqlite3.Row]],
    variable_name: str,
    preferred_key: str,
) -> sqlite3.Row | None:
    """Resolve one variable definition in the SQLite dictionary."""
    if (preferred_key, variable_name) in exact_lookup:
        return exact_lookup[(preferred_key, variable_name)]

    candidates = by_name_lookup.get(variable_name, [])
    if not candidates:
        return None

    for row in candidates:
        key_value = row["KeyValue"] or ""
        if key_value == preferred_key:
            return row

    if len(candidates) == 1:
        return candidates[0]

    for row in candidates:
        key_value = (row["KeyValue"] or "").strip()
        if preferred_key in {"Environment", "Facility", "Whole Building"} and key_value == preferred_key:
            return row

    return candidates[0]


def fetch_series_values(conn: sqlite3.Connection, dictionary_index: int, environment_period_index: int | None) -> list[float]:
    """Fetch report-data values for one dictionary index, excluding warmup timesteps."""
    if environment_period_index is None:
        return []

    rows = fetch_all_rows(
        conn,
        """
        SELECT rd.Value
        FROM ReportData AS rd
        JOIN Time AS t
          ON rd.TimeIndex = t.TimeIndex
        WHERE rd.ReportDataDictionaryIndex = ?
          AND t.WarmupFlag = 0
          AND t.EnvironmentPeriodIndex = ?
        ORDER BY rd.TimeIndex
        """,
        (dictionary_index, environment_period_index),
    )
    return [float(row["Value"]) for row in rows]


def summarize_level(values: list[float]) -> dict[str, float]:
    """Summarize a continuous environmental/zone variable."""
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def summarize_rate_kw(values: list[float]) -> dict[str, float]:
    """Summarize a power/demand rate in kW."""
    converted = [value / 1000.0 for value in values]
    return {
        "min_kw": min(converted),
        "mean_kw": sum(converted) / len(converted),
        "max_kw": max(converted),
    }


def summarize_energy_kwh(values: list[float]) -> dict[str, float]:
    """Summarize a timestep energy variable in kWh."""
    return {
        "total_kwh": sum(values) / 3_600_000.0,
        "peak_timestep_kwh": max(values) / 3_600_000.0,
    }


def summarize_time_hours(values: list[float]) -> dict[str, float]:
    """Summarize a timestep time-accumulation variable in hours."""
    return {
        "total_hours": sum(values),
        "max_timestep_hours": max(values),
    }


def extract_tabular_metrics(conn: sqlite3.Connection) -> dict[str, object]:
    """Extract compact annual metrics from tabular report outputs."""
    row: dict[str, object] = {}

    tabular_rows = fetch_all_rows(
        conn,
        """
        SELECT ReportName, TableName, RowName, ColumnName, Value, Units
        FROM TabularDataWithStrings
        WHERE (ReportName = 'AnnualBuildingUtilityPerformanceSummary'
               AND TableName IN ('Building Area', 'Comfort and Setpoint Not Met Summary'))
           OR (ReportName = 'EnergyMeters'
               AND TableName = 'Annual and Peak Values - Electricity')
        """,
    )

    for record in tabular_rows:
        report_name = record["ReportName"]
        table_name = record["TableName"]
        row_name = record["RowName"]
        column_name = record["ColumnName"]
        value = record["Value"]

        if report_name == "AnnualBuildingUtilityPerformanceSummary" and table_name == "Building Area":
            alias = TABULAR_AREA_ROWS.get(row_name)
            if alias:
                row[alias] = safe_float(value)
            continue

        if report_name == "AnnualBuildingUtilityPerformanceSummary" and table_name == "Comfort and Setpoint Not Met Summary":
            alias = TABULAR_COMFORT_ROWS.get(row_name)
            if alias:
                row[alias] = safe_float(value)
            continue

        if report_name == "EnergyMeters" and table_name == "Annual and Peak Values - Electricity":
            meter_alias = METER_ALIASES.get(row_name)
            if not meter_alias:
                continue

            numeric_value = safe_float(value)
            if column_name == "Electricity Annual Value" and numeric_value is not None:
                row[f"{meter_alias}_annual_gj"] = numeric_value
                row[f"{meter_alias}_annual_kwh"] = numeric_value * 277.7777777778
            elif column_name == "Electricity Maximum Value" and numeric_value is not None:
                row[f"{meter_alias}_peak_w"] = numeric_value
                row[f"{meter_alias}_peak_kw"] = numeric_value / 1000.0
            elif column_name == "Electricity Minimum Value" and numeric_value is not None:
                row[f"{meter_alias}_min_w"] = numeric_value
                row[f"{meter_alias}_min_kw"] = numeric_value / 1000.0
            elif column_name == "Timestamp of Maximum {TIMESTAMP}":
                row[f"{meter_alias}_peak_timestamp"] = str(value).strip()
            elif column_name == "Timestamp of Minimum {TIMESTAMP}":
                row[f"{meter_alias}_min_timestamp"] = str(value).strip()

    return row


def extract_time_metadata(conn: sqlite3.Connection, environment_period_index: int | None) -> dict[str, object]:
    """Extract basic timestep metadata from the Time table."""
    row: dict[str, object] = {}
    if environment_period_index is None:
        row["environment_period_index"] = None
        row["interval_minutes"] = None
        row["time_rows"] = 0
        return row

    interval_row = conn.execute(
        """
        SELECT Interval, COUNT(*) AS n_rows
        FROM Time
        WHERE WarmupFlag = 0 AND EnvironmentPeriodIndex = ?
        GROUP BY Interval
        ORDER BY n_rows DESC
        LIMIT 1
        """,
        (environment_period_index,),
    ).fetchone()

    total_rows = conn.execute(
        """
        SELECT COUNT(*) AS n_rows
        FROM Time
        WHERE WarmupFlag = 0 AND EnvironmentPeriodIndex = ?
        """,
        (environment_period_index,),
    ).fetchone()

    row["environment_period_index"] = environment_period_index
    row["interval_minutes"] = int(interval_row["Interval"]) if interval_row else None
    row["time_rows"] = int(total_rows["n_rows"]) if total_rows else 0
    return row


def extract_report_metrics(conn: sqlite3.Connection, building: str, environment_period_index: int | None) -> dict[str, object]:
    """Extract metrics directly from ReportData/ReportDataDictionary."""
    row: dict[str, object] = {}
    rep_zone = REPRESENTATIVE_ZONES[building]
    row["representative_zone"] = rep_zone

    exact_lookup, by_name_lookup = load_rdd_lookup(conn)

    for spec in VARIABLE_SPECS:
        preferred_key = spec["preferred_key"]
        if preferred_key == "__REP_ZONE__":
            preferred_key = rep_zone

        rdd_row = choose_rdd_row(exact_lookup, by_name_lookup, spec["name"], preferred_key)
        if rdd_row is None:
            continue

        values = fetch_series_values(conn, int(rdd_row["ReportDataDictionaryIndex"]), environment_period_index)
        if not values:
            continue

        alias = spec["alias"]
        kind = spec["kind"]

        if kind == "level":
            summary = summarize_level(values)
            row[f"{alias}_min"] = summary["min"]
            row[f"{alias}_mean"] = summary["mean"]
            row[f"{alias}_max"] = summary["max"]
        elif kind == "rate_kw":
            summary = summarize_rate_kw(values)
            row[f"{alias}_min_kw"] = summary["min_kw"]
            row[f"{alias}_mean_kw"] = summary["mean_kw"]
            row[f"{alias}_max_kw"] = summary["max_kw"]
        elif kind == "energy_kwh":
            summary = summarize_energy_kwh(values)
            row[f"{alias}_total_kwh"] = summary["total_kwh"]
            row[f"{alias}_peak_timestep_kwh"] = summary["peak_timestep_kwh"]
        elif kind == "time_hours":
            summary = summarize_time_hours(values)
            row[f"{alias}_total_hours"] = summary["total_hours"]
            row[f"{alias}_max_timestep_hours"] = summary["max_timestep_hours"]

    return row


def extract_one_sqlite(entry: dict[str, object]) -> dict[str, object]:
    """Extract one simulation's metrics into a flat row."""
    sql_path = Path(entry["sql_path"])
    row = {
        "scenario": entry["scenario"],
        "building": entry["building"],
        "city": entry["city"],
        "year": entry["year"],
        "sql_path": str(sql_path),
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
    }

    conn = sqlite3.connect(sql_path)
    conn.row_factory = sqlite3.Row
    try:
        environment_period_index = get_active_environment_period_index(conn)
        row.update(extract_time_metadata(conn, environment_period_index))
        row.update(extract_tabular_metrics(conn))
        row.update(extract_report_metrics(conn, str(entry["building"]), environment_period_index))
    finally:
        conn.close()

    return row


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    """Build a stable field order from extracted rows."""
    preferred = [
        "scenario",
        "building",
        "city",
        "year",
        "sql_path",
        "extracted_at",
        "interval_minutes",
        "time_rows",
        "representative_zone",
    ]

    seen = set(preferred)
    dynamic: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                dynamic.append(key)

    return preferred + sorted(dynamic)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract annual metrics from standalone EnergyPlus SQLite outputs")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing scenario result folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metric_exports"),
        help="Directory to write extracted CSV/JSON outputs",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Optional comma-separated scenario filter",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of SQLite files to process",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    results_root = (script_dir / args.results_root).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    discovered = discover_sqlite_files(results_root)
    if args.scenarios:
        allowed = {item.strip() for item in args.scenarios.split(",") if item.strip()}
        discovered = [entry for entry in discovered if entry["scenario"] in allowed]
    if args.limit is not None:
        discovered = discovered[: args.limit]

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for entry in discovered:
        try:
            rows.append(extract_one_sqlite(entry))
        except Exception as exc:
            failures.append(
                {
                    "scenario": entry["scenario"],
                    "building": entry["building"],
                    "city": entry["city"],
                    "year": entry["year"],
                    "sql_path": str(entry["sql_path"]),
                    "error": str(exc),
                }
            )

    fieldnames = build_fieldnames(rows) if rows else [
        "scenario",
        "building",
        "city",
        "year",
        "sql_path",
        "extracted_at",
    ]

    write_csv(output_dir / "annual_metrics.csv", rows, fieldnames)
    write_csv(
        output_dir / "extraction_failures.csv",
        failures,
        ["scenario", "building", "city", "year", "sql_path", "error"],
    )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "sqlite_files_found": len(discover_sqlite_files(results_root)),
        "sqlite_files_processed": len(discovered),
        "rows_written": len(rows),
        "failures": len(failures),
        "scenario_counts": {},
    }

    scenario_counts: dict[str, int] = {}
    for row in rows:
        scenario = str(row["scenario"])
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    summary["scenario_counts"] = scenario_counts

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "extraction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
