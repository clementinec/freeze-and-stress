#!/usr/bin/env python3
"""
ADAPTIVE RE-FREEZE: Year-by-year HVAC re-sizing on climate failure.

For each (building, city) pair this script runs a single sequential loop over
2025-2100.  Each year is simulated with the current frozen model.  When occupied
cooling unmet hours exceed --threshold the model is immediately re-frozen using
that year's future-climate EPW (ASHRAE 0.4% design DB derived from EPW
statistics), and simulation continues from the following year with the new model.

This avoids wasting simulation time on years that will be superseded by a
re-sized model, and ensures every result comes from the correct model generation.

Output layout:
  <summary-root>/<scenario>/
    refreeze_sequence.csv           — aggregated sequence for all pairs
    <building>/<city>/
      state.json                    — per-pair restart state (year-level granularity)
      frozen_gen<N>_<year>epw.epJSON
      sizing_snapshots.json
  <sim-root>/<scenario>/sim/gen<N>/<building>/<city>/<year>/eplusout.sql
                                    — simulation outputs

Usage:
    python run_adaptive_refreeze.py \\
        --epw-root ../epw_out/CORDEX_CMIP5_REMO2015_rcp85 \\
        --summary-root refreeze_results/ \\
        --sim-root refreeze_results/ \\
        --local-eplus /usr/local/bin/energyplus \\
        --scenario-name CORDEX_CMIP5_REMO2015_rcp85 \\
        --threshold 100 \\
        --workers 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import compute_break_years_multi_metric as cbm
from output_preset import REPRESENTATIVE_ZONES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUILDINGS = {
    "office": {
        "name": "Office Medium",
        "frozen_pattern": "office_{city}_frozen_detailed.epJSON",
        "source_pattern": "office_medium_{city}_patched.epJSON",
    },
    "apartment": {
        "name": "Apartment MidRise",
        "frozen_pattern": "apartment_{city}_frozen_detailed.epJSON",
        "source_pattern": "apartment_midrise_{city}_patched.epJSON",
    },
    "retail": {
        "name": "Retail Standalone",
        "frozen_pattern": "retail_{city}_frozen_detailed.epJSON",
        "source_pattern": "retail_standalone_{city}_patched.epJSON",
    },
}

CITIES = {
    "Los_Angeles": {"epw_city_name": "Los_Angeles", "city_key": "LosAngeles"},
    "Miami":       {"epw_city_name": "Miami",       "city_key": "Miami"},
    "Montreal":    {"epw_city_name": "Montreal",    "city_key": "Montreal"},
    "Phoenix":     {"epw_city_name": "Phoenix",     "city_key": "Phoenix"},
    "Toronto":     {"epw_city_name": "Toronto",     "city_key": "Toronto"},
    "Vancouver":   {"epw_city_name": "Vancouver",   "city_key": "Vancouver"},
}

YEARS = list(range(2025, 2101))

_UNMET_QUERY = """
SELECT Value
FROM TabularDataWithStrings
WHERE ReportName = 'AnnualBuildingUtilityPerformanceSummary'
  AND TableName = 'Comfort and Setpoint Not Met Summary'
  AND RowName = 'Time Setpoint Not Met During Occupied Cooling'
LIMIT 1
"""

_HEATING_UNMET_QUERY = """
SELECT Value
FROM TabularDataWithStrings
WHERE ReportName = 'AnnualBuildingUtilityPerformanceSummary'
  AND TableName = 'Comfort and Setpoint Not Met Summary'
  AND RowName = 'Time Setpoint Not Met During Occupied Heating'
LIMIT 1
"""

SUPPORTED_SCREENED_METRICS = {
    "annual_edh_c_h",
    "maximum_daily_edh_k_h",
    "daily_edh_exceedance_days",
    "cooling_setpoint_not_met_occupied_hours",
    "heating_unmet_hours",
}
DEFAULT_EDH_BASE_TEMP_C = 26.0


@dataclass(frozen=True)
class ScreenedMetric:
    key: str
    label: str
    threshold: float
    threshold_direction: str
    subfamily: str
    source_column: str


# ---------------------------------------------------------------------------
# EnergyPlus runner
# ---------------------------------------------------------------------------

def _run_sim_local(model: Path, epw: Path, out: Path, eplus: str) -> tuple[bool, str | None]:
    out.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [eplus, "-w", str(epw), "-d", str(out), str(model)],
        capture_output=True, text=True,
    )
    end = out / "eplusout.end"
    if end.exists() and "EnergyPlus Completed Successfully" in end.read_text():
        err = out / "eplusout.err"
        if err.exists() and "**  Fatal  **" in err.read_text():
            return False, "Fatal error in simulation"
        return True, None
    return False, (result.stderr or result.stdout)[-400:] or "Simulation failed"


def _run_sim_docker(model: Path, epw: Path, out: Path, name: str) -> tuple[bool, str | None]:
    out.mkdir(parents=True, exist_ok=True)
    work = "/tmp/eplus_adaptive"
    try:
        subprocess.run(["docker", "exec", name, "rm", "-rf", work], capture_output=True, check=False)
        subprocess.run(["docker", "exec", name, "mkdir", "-p", work], check=True)
        subprocess.run(["docker", "cp", str(model), f"{name}:{work}/model.epJSON"], check=True)
        subprocess.run(["docker", "cp", str(epw),   f"{name}:{work}/weather.epw"],  check=True)
        r = subprocess.run(
            ["docker", "exec", "-w", work, name,
             "energyplus", "-w", "weather.epw", "-d", "output", "model.epJSON"],
            capture_output=True, text=True,
        )
        if "EnergyPlus Completed Successfully" in r.stderr:
            subprocess.run(["docker", "cp", f"{name}:{work}/output/.", str(out)], check=True)
            return True, None
        return False, r.stderr[-400:] if r.stderr else r.stdout[-400:]
    except subprocess.CalledProcessError as exc:
        return False, str(exc)


def _run_simulation(model: Path, epw: Path, out: Path, executor: dict) -> tuple[bool, str | None]:
    if executor["mode"] == "docker":
        return _run_sim_docker(model, epw, out, executor["name"])
    return _run_sim_local(model, epw, out, executor["path"])


# ---------------------------------------------------------------------------
# Screened failure metrics
# ---------------------------------------------------------------------------

def load_screened_failure_metrics(metric_res_root: Path, scenario: str) -> list[ScreenedMetric]:
    screening_csv = metric_res_root / scenario / "paper_metric_screening.csv"
    if not screening_csv.exists():
        raise FileNotFoundError(f"Screening CSV not found: {screening_csv}")

    screening_rows: list[dict[str, object]] = []
    with screening_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        screening_rows = list(reader)

    selected = cbm.selected_metrics_from_screening(screening_rows, "response", require_threshold=True)
    if not selected:
        selected = cbm.FAILURE_METRICS

    metrics: list[ScreenedMetric] = []
    for metric in selected:
        if metric.key not in SUPPORTED_SCREENED_METRICS:
            raise ValueError(
                f"Unsupported screened failure metric '{metric.key}' in {screening_csv}. "
                "Add an explicit SQLite extraction mapping before using it in adaptive refreeze."
            )
        metrics.append(
            ScreenedMetric(
                key=metric.key,
                label=metric.label,
                threshold=float(metric.threshold),
                threshold_direction=metric.threshold_direction,
                subfamily=metric.subfamily,
                source_column=str(metric.aliases[0]) if metric.aliases else metric.key,
            )
        )

    if not metrics:
        raise ValueError(
            f"No thresholded response metrics with screening_selected_for_analysis=Yes found in {screening_csv}"
        )
    return metrics


def _cbm_metrics_from_screened(screened_metrics: list[ScreenedMetric]) -> tuple[cbm.MetricSpec, ...]:
    selected_keys = {metric.key for metric in screened_metrics}
    return tuple(metric for metric in cbm.METRICS if metric.key in selected_keys)


def _record_from_metric_values(year: int, metric_values: dict[str, float | None]) -> dict[str, object]:
    record: dict[str, object] = {"year": year}
    for key, value in metric_values.items():
        record[key] = value
    return record


def _records_for_generation(
    trigger_rows: dict[tuple[int, int], dict],
    generation: int,
    screened_metrics: list[ScreenedMetric],
) -> dict[int, dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    keys = [metric.key for metric in screened_metrics]
    for (gen, year), row in trigger_rows.items():
        if gen != generation:
            continue
        record: dict[str, object] = {"year": year}
        for key in keys:
            record[key] = _safe_float(row.get(key))
        records[year] = record
    return records


# ---------------------------------------------------------------------------
# SQLite metric extraction
# ---------------------------------------------------------------------------

def _extract_unmet_hours(sql: Path) -> float | None:
    try:
        conn = sqlite3.connect(f"file:{sql}?mode=ro", uri=True, timeout=2)
        try:
            row = conn.execute(_UNMET_QUERY).fetchone()
            return float(row[0]) if row and row[0] is not None else None
        finally:
            conn.close()
    except Exception:
        return None


def _safe_float(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_scalar_query(conn: sqlite3.Connection, query: str, params: tuple = ()) -> float | None:
    row = conn.execute(query, params).fetchone()
    return _safe_float(row[0]) if row else None


def _get_active_env_period(conn: sqlite3.Connection) -> int | None:
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
    return int(row[0]) if row else None


def _find_rdd_index(conn: sqlite3.Connection, variable_name: str, key_value: str) -> int | None:
    row = conn.execute(
        """
        SELECT ReportDataDictionaryIndex
        FROM ReportDataDictionary
        WHERE Name = ? AND KeyValue = ?
        LIMIT 1
        """,
        (variable_name, key_value),
    ).fetchone()
    return int(row[0]) if row else None


def _fetch_time_series_with_day(
    conn: sqlite3.Connection,
    dictionary_index: int,
    environment_period_index: int,
) -> list[tuple[float, int, int, int]]:
    rows = conn.execute(
        """
        SELECT rd.Value, t.Month, t.Day, t.Interval
        FROM ReportData AS rd
        JOIN Time AS t
          ON rd.TimeIndex = t.TimeIndex
        WHERE rd.ReportDataDictionaryIndex = ?
          AND t.WarmupFlag = 0
          AND t.EnvironmentPeriodIndex = ?
        ORDER BY rd.TimeIndex
        """,
        (dictionary_index, environment_period_index),
    ).fetchall()
    return [(float(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in rows]


def _fetch_time_series_values(
    conn: sqlite3.Connection,
    dictionary_index: int,
    environment_period_index: int,
) -> list[float]:
    rows = conn.execute(
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
    ).fetchall()
    return [float(r[0]) for r in rows]


def _compute_edh_metrics(
    hourly_data: list[tuple[float, int, int, int]],
    base_temp_c: float = DEFAULT_EDH_BASE_TEMP_C,
) -> dict[str, float]:
    daily_edh: dict[tuple[int, int], float] = {}
    annual_edh = 0.0

    for temp_c, month, day, interval_min in hourly_data:
        exceedance = temp_c - base_temp_c
        if exceedance <= 0:
            continue
        dt_hours = interval_min / 60.0
        edh_step = exceedance * dt_hours
        annual_edh += edh_step
        daily_edh[(month, day)] = daily_edh.get((month, day), 0.0) + edh_step

    daily_max = max(daily_edh.values()) if daily_edh else 0.0
    exceed_days = sum(1 for value in daily_edh.values() if value > 6.0)
    return {
        "annual_edh_c_h": round(annual_edh, 4),
        "maximum_daily_edh_k_h": round(daily_max, 4),
        "daily_edh_exceedance_days": float(exceed_days),
    }


def extract_screened_metric_values(
    sql_path: Path,
    building: str,
    screened_metrics: list[ScreenedMetric],
) -> dict[str, float | None]:
    values: dict[str, float | None] = {metric.key: None for metric in screened_metrics}
    if not sql_path.exists():
        return values

    conn = sqlite3.connect(sql_path)
    try:
        env_period = _get_active_env_period(conn)
        need_edh = any(metric.key in {
            "annual_edh_c_h",
            "maximum_daily_edh_k_h",
            "daily_edh_exceedance_days",
        } for metric in screened_metrics)
        edh_values: dict[str, float] = {}
        if need_edh:
            rep_zone = REPRESENTATIVE_ZONES.get(building)
            if rep_zone is None:
                raise ValueError(f"Unsupported building type for EDH extraction: {building}")
            if env_period is None:
                raise ValueError(f"No active environment period found in {sql_path}")
            op_temp_index = _find_rdd_index(conn, "Zone Operative Temperature", rep_zone)
            if op_temp_index is None:
                raise ValueError(f"Zone Operative Temperature not found for '{rep_zone}' in {sql_path}")
            hourly = _fetch_time_series_with_day(conn, op_temp_index, env_period)
            if not hourly:
                raise ValueError(f"No hourly operative temperature data found in {sql_path}")
            edh_values = _compute_edh_metrics(hourly)

        for metric in screened_metrics:
            if metric.key in edh_values:
                values[metric.key] = edh_values[metric.key]
            elif metric.key == "cooling_setpoint_not_met_occupied_hours":
                if env_period is None:
                    raise ValueError(f"No active environment period found in {sql_path}")
                idx = _find_rdd_index(conn, "Facility Cooling Setpoint Not Met While Occupied Time", "Facility")
                if idx is None:
                    raise ValueError(
                        "Facility Cooling Setpoint Not Met While Occupied Time not found in "
                        f"{sql_path}"
                    )
                series = _fetch_time_series_values(conn, idx, env_period)
                values[metric.key] = round(sum(series), 6) if series else 0.0
            elif metric.key == "heating_unmet_hours":
                values[metric.key] = _extract_scalar_query(conn, _HEATING_UNMET_QUERY)
            else:
                raise ValueError(f"Unsupported screened metric mapping for '{metric.key}'")
    finally:
        conn.close()

    return values


def breached_metrics(
    metric_values: dict[str, float | None],
    screened_metrics: list[ScreenedMetric],
) -> list[ScreenedMetric]:
    breached: list[ScreenedMetric] = []
    for metric in screened_metrics:
        value = metric_values.get(metric.key)
        if value is None:
            continue
        if metric.threshold_direction == "below":
            is_breached = value < metric.threshold
        else:
            is_breached = value > metric.threshold
        if is_breached:
            breached.append(metric)
    return breached


# ---------------------------------------------------------------------------
# State helpers (year-level granularity for restarts)
# ---------------------------------------------------------------------------

def _load_state(path: Path) -> dict:
    """
    State schema:
      gen: int                      current generation index
      freeze_year: str|int          label for current gen's freeze basis
      current_frozen: str           path to current frozen model
      sequence: list[dict]          completed generation records
      completed_years: list[int]    years already simulated in the current gen
    """
    if path.exists():
        return json.loads(path.read_text())
    return {
        "gen": 0,
        "freeze_year": "TMY",
        "current_frozen": None,
        "sequence": [],
        "completed_years": [],
    }


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def _trigger_rows_path(pair_dir: Path) -> Path:
    return pair_dir / "annual_trigger_metrics.csv"


def _load_trigger_rows(pair_dir: Path) -> dict[tuple[int, int], dict]:
    path = _trigger_rows_path(pair_dir)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: dict[tuple[int, int], dict] = {}
        for row in reader:
            try:
                key = (int(row["generation"]), int(row["year"]))
            except (KeyError, TypeError, ValueError):
                continue
            rows[key] = row
        return rows


def _save_trigger_rows(pair_dir: Path, rows_by_key: dict[tuple[int, int], dict]) -> None:
    path = _trigger_rows_path(pair_dir)
    rows = [rows_by_key[key] for key in sorted(rows_by_key, key=lambda item: (item[0], item[1]))]
    if not rows:
        if path.exists():
            path.unlink()
        return

    preferred = [
        "scenario",
        "building",
        "city",
        "generation",
        "year",
        "freeze_year",
        "sql_path",
        "threshold_breached",
        "break_triggered",
        "all_trigger_metrics",
        "all_trigger_metric_keys",
        "all_trigger_subfamilies",
        "first_trigger_metric",
        "first_trigger_metric_key",
        "trigger_subfamily",
    ]
    seen = set(preferred)
    extras: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                extras.append(key)
    fieldnames = preferred + sorted(extras)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Core adaptive loop
# ---------------------------------------------------------------------------

def run_adaptive_refreeze_pair(
    building: str,
    city: str,
    source_models_dir: Path,
    frozen_models_dir: Path,
    epw_root: Path,
    scenario_summary_dir: Path,
    scenario_sim_dir: Path,
    scenario_name: str,
    executor: dict,
    screened_metrics: list[ScreenedMetric],
    threshold: float = 100.0,
    max_gens: int = 6,
    epw_suffix: str = "climatedelta",
    retrofit_lag: int = 0,
    force_restart: bool = False,
) -> list[dict]:
    """
    Sequential year-by-year adaptive re-freeze loop for one (building, city) pair.

    Algorithm:
      for year in 2025..2100:
          simulate year with current frozen model
          if unmet_hours > threshold and gen < max_gens:
              record break for this gen
              re-freeze immediately with EPW[year + retrofit_lag]
              gen += 1, continue with new model from year+1
      record final gen entry (survived or hit 2100)

    Restart-safe: year-level state.json is written after every simulated year.
    """
    from freeze_model import re_freeze  # local import avoids multiprocessing pickling issues

    city_cfg  = CITIES[city]
    city_key  = city_cfg["city_key"]
    epw_city  = city_cfg["epw_city_name"]
    bldg_cfg  = BUILDINGS[building]
    pair_dir  = scenario_summary_dir / building / city
    pair_dir.mkdir(parents=True, exist_ok=True)
    state_path = pair_dir / "state.json"
    trigger_rows_file = _trigger_rows_path(pair_dir)

    if force_restart and state_path.exists():
        state_path.unlink()
    if force_restart and trigger_rows_file.exists():
        trigger_rows_file.unlink()

    state = _load_state(state_path)
    trigger_rows = _load_trigger_rows(pair_dir)

    # Validate / seed initial frozen model
    initial_frozen = frozen_models_dir / bldg_cfg["frozen_pattern"].format(city=city_key)
    source_model   = source_models_dir  / bldg_cfg["source_pattern"].format(city=city_key)
    if not initial_frozen.exists():
        print(f"  [ERROR] Initial frozen model not found: {initial_frozen}")
        return state["sequence"]
    if not source_model.exists():
        print(f"  [ERROR] Source model not found: {source_model}")
        return state["sequence"]

    if state["current_frozen"] is None:
        state["current_frozen"] = str(initial_frozen)

    gen             = state["gen"]
    current_frozen  = Path(state["current_frozen"])
    sequence        = state["sequence"]
    completed_years = set(state["completed_years"])
    cbm_metrics     = _cbm_metrics_from_screened(screened_metrics)

    # Track per-gen simulation start year (for sequence entry)
    # On resume, gen N starts at (previous gen's break_year + 1)
    gen_start_year  = 2025 if gen == 0 else (
        sequence[-1]["break_year"] + 1 if sequence else 2025
    )
    gen_records_by_year = _records_for_generation(trigger_rows, gen, screened_metrics)

    print(f"\n  [{building}/{city}] Resuming: gen={gen}, current_model={current_frozen.name}")

    for year in YEARS:
        # Skip years before this generation's start year
        if year < gen_start_year:
            continue

        # Determine output dir for this (gen, year) simulation
        sim_out = scenario_sim_dir / "sim" / f"gen{gen}" / building / city / str(year)

        # Skip years already completed for this gen
        if year not in completed_years or not (sim_out / "eplusout.sql").exists():
            # Find EPW
            epw_file = epw_root / city / f"{epw_city}_{year}_{epw_suffix}.epw"
            if not epw_file.exists():
                print(f"    [{building}/{city}] [SKIP] EPW not found: {epw_file.name}")
                continue

            ok, err = _run_simulation(current_frozen, epw_file, sim_out, executor)
            if not ok:
                print(f"    [{building}/{city}] [FAIL] {year}: {err}")
                continue

            completed_years.add(year)
            state["completed_years"] = sorted(completed_years)
            _save_state(state_path, state)

        sql_path = sim_out / "eplusout.sql"
        metric_values = extract_screened_metric_values(sql_path, building, screened_metrics)
        breached = breached_metrics(metric_values, screened_metrics)
        first_breached = breached[0] if breached else None
        legacy_unmet = metric_values.get("cooling_setpoint_not_met_occupied_hours")
        all_trigger_metrics = "; ".join(metric.label for metric in breached)
        all_trigger_metric_keys = "; ".join(metric.key for metric in breached)
        all_trigger_subfamilies = "; ".join(
            metric.subfamily for metric in breached if metric.subfamily
        )
        gen_records_by_year[year] = _record_from_metric_values(year, metric_values)
        records_this_gen = [gen_records_by_year[y] for y in sorted(gen_records_by_year)]
        candidate_break_year = cbm.break_year(records_this_gen, gen_start_year, cbm_metrics)
        break_triggered = candidate_break_year == year

        trigger_row = {
            "scenario": scenario_name,
            "building": building,
            "city": city,
            "generation": gen,
            "year": year,
            "freeze_year": state["freeze_year"],
            "sql_path": str(sql_path),
            "threshold_breached": "Yes" if breached else "No",
            "break_triggered": "Yes" if break_triggered else "No",
            "all_trigger_metrics": all_trigger_metrics,
            "all_trigger_metric_keys": all_trigger_metric_keys,
            "all_trigger_subfamilies": all_trigger_subfamilies,
            "first_trigger_metric": first_breached.label if first_breached else "",
            "first_trigger_metric_key": first_breached.key if first_breached else "",
            "trigger_subfamily": first_breached.subfamily if first_breached else "",
        }
        for metric in screened_metrics:
            trigger_row[metric.key] = metric_values.get(metric.key)
        trigger_rows[(gen, year)] = trigger_row
        _save_trigger_rows(pair_dir, trigger_rows)

        if breached:
            trigger_value = metric_values.get(first_breached.key) if first_breached else None
            breach_parts: list[str] = []
            for metric in breached:
                value = metric_values.get(metric.key)
                if value is None:
                    breach_parts.append(metric.key)
                else:
                    breach_parts.append(f"{metric.key}={value:.3f}>{metric.threshold:g}")
            print(
                f"    [{building}/{city}] gen{gen} {year}: "
                + ", ".join(breach_parts)
            )
        elif legacy_unmet is not None:
            print(f"    [{building}/{city}] gen{gen} {year}: unmet={legacy_unmet:.1f}h no screened breach")
        else:
            print(f"    [{building}/{city}] gen{gen} {year}: no screened breach")

        if breached and not break_triggered:
            print(f"    [{building}/{city}] gen{gen} {year}: threshold breach noted, main break still pending")

        # Check for failure under the main-pipeline break-year definition.
        if break_triggered and gen < max_gens:
            # ── Record this generation's sequence entry ──────────────────────
            snap = _read_snapshot(pair_dir, gen)
            entry = _build_entry(building, city, gen, state["freeze_year"],
                                 gen_start_year, break_year=year,
                                 unmet_at_break=legacy_unmet, threshold=threshold,
                                 snap=snap, trigger_metrics=breached,
                                 trigger_value=metric_values.get(first_breached.key) if first_breached else None)
            sequence.append(entry)
            trigger_desc = all_trigger_metric_keys or "unknown trigger"
            print(f"  [{building}/{city}] *** BREAK gen{gen} at {year} via {trigger_desc} ***")

            # ── Re-freeze ───────────────────────────────────────────────────
            trigger_year = year + retrofit_lag
            trigger_epw  = epw_root / city / f"{epw_city}_{trigger_year}_{epw_suffix}.epw"
            if not trigger_epw.exists():
                print(f"  [{building}/{city}] WARNING: trigger EPW {trigger_epw.name} not found. Stopping.")
                break

            gen_next           = gen + 1
            frozen_out         = pair_dir / f"frozen_gen{gen_next}_{trigger_year}epw.epJSON"
            sizing_dir         = pair_dir / f"sizing_gen{gen_next}"

            print(f"  [{building}/{city}] Re-freezing → gen{gen_next} using {trigger_epw.name}...")
            _, sizing_snap = re_freeze(
                source_model_path=source_model,
                epw_path=trigger_epw,
                sizing_output_dir=sizing_dir,
                frozen_output_path=frozen_out,
                building_key=building,
                executor=executor,
                update_design_days=True,
            )
            if sizing_snap is None:
                print(f"  [{building}/{city}] ERROR: re-freeze failed for gen{gen_next}. Stopping.")
                break

            _save_snapshot(pair_dir, gen_next, sizing_snap)

            # ── Advance state ────────────────────────────────────────────────
            gen            = gen_next
            gen_start_year = year + 1   # next gen starts from the year after the break
            current_frozen = frozen_out
            completed_years = set()
            gen_records_by_year = {}

            state["gen"]             = gen
            state["freeze_year"]     = trigger_year
            state["current_frozen"]  = str(current_frozen)
            state["sequence"]        = sequence
            state["completed_years"] = []
            _save_state(state_path, state)

    # ── Final sequence entry for the last (surviving) generation ─────────────
    if not sequence or sequence[-1]["generation"] != gen:
        snap = _read_snapshot(pair_dir, gen)
        entry = _build_entry(building, city, gen, state["freeze_year"],
                             gen_start_year, break_year=None,
                             unmet_at_break=None, threshold=threshold,
                             snap=snap, trigger_metrics=None, trigger_value=None)
        sequence.append(entry)
        state["sequence"] = sequence
        _save_state(state_path, state)

    (pair_dir / "refreeze_sequence.json").write_text(json.dumps(sequence, indent=2))
    print(f"\n  [{building}/{city}] Done. {len(sequence)} generation(s).")
    return sequence


# ---------------------------------------------------------------------------
# Helpers for sequence entries and snapshots
# ---------------------------------------------------------------------------

def _build_entry(
    building: str, city: str, gen: int, freeze_year,
    first_year: int, break_year: int | None,
    unmet_at_break: float | None, threshold: float,
    snap: dict,
    trigger_metrics: list[ScreenedMetric] | None,
    trigger_value: float | None,
) -> dict:
    first_trigger = trigger_metrics[0] if trigger_metrics else None
    survived = (break_year - first_year) if break_year else (2100 - first_year + 1)
    return {
        "building": building,
        "city": city,
        "generation": gen,
        "freeze_year": freeze_year,
        "first_year_simulated": first_year,
        "break_year": break_year if break_year is not None else "Never",
        "years_survived": survived,
        "unmet_hours_at_break": unmet_at_break,
        "threshold_used": threshold,
        "all_trigger_metrics": "; ".join(metric.label for metric in (trigger_metrics or [])),
        "all_trigger_metric_keys": "; ".join(metric.key for metric in (trigger_metrics or [])),
        "all_trigger_subfamilies": "; ".join(
            metric.subfamily for metric in (trigger_metrics or []) if metric.subfamily
        ),
        "first_trigger_metric": first_trigger.label if first_trigger else "",
        "first_trigger_metric_key": first_trigger.key if first_trigger else "",
        "trigger_subfamily": first_trigger.subfamily if first_trigger else "",
        "trigger_threshold": first_trigger.threshold if first_trigger else None,
        "trigger_value": trigger_value,
        "primary_cooling_capacity_w":   snap.get("cooling_capacity_w"),
        "primary_cooling_airflow_m3s":  snap.get("cooling_airflow_m3s"),
        "primary_heating_capacity_w":   snap.get("heating_capacity_w"),
        "design_day_db_temp_c":         snap.get("design_day_db_temp_c"),
        "design_day_wb_temp_c":         snap.get("design_day_wb_temp_c"),
        "n_fields_replaced":            snap.get("n_fields_replaced"),
    }


def _save_snapshot(pair_dir: Path, gen: int, snap: dict) -> None:
    p = pair_dir / "sizing_snapshots.json"
    all_snaps = json.loads(p.read_text()) if p.exists() else {}
    all_snaps[str(gen)] = snap
    p.write_text(json.dumps(all_snaps, indent=2))


def _read_snapshot(pair_dir: Path, gen: int) -> dict:
    p = pair_dir / "sizing_snapshots.json"
    if p.exists():
        return json.loads(p.read_text()).get(str(gen), {})
    return {}


# ---------------------------------------------------------------------------
# Parallel orchestration
# ---------------------------------------------------------------------------

def _pair_worker(kwargs: dict) -> list[dict]:
    return run_adaptive_refreeze_pair(**kwargs)


def write_sequence_csv(all_sequences: list[list[dict]], path: Path, scenario: str) -> None:
    rows = [dict(e, scenario=scenario) for seq in all_sequences for e in seq]
    if not rows:
        return
    fieldnames = [
        "scenario", "building", "city", "generation", "freeze_year",
        "first_year_simulated", "break_year", "years_survived",
        "unmet_hours_at_break", "threshold_used",
        "all_trigger_metrics", "all_trigger_metric_keys", "all_trigger_subfamilies",
        "first_trigger_metric", "first_trigger_metric_key", "trigger_subfamily",
        "trigger_threshold", "trigger_value",
        "primary_cooling_capacity_w", "primary_cooling_airflow_m3s",
        "primary_heating_capacity_w", "design_day_db_temp_c",
        "design_day_wb_temp_c", "n_fields_replaced",
    ]
    extra = [k for k in rows[0] if k not in fieldnames]
    fieldnames.extend(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def aggregate_trigger_csv(
    pair_kwargs: list[dict],
    scenario_summary_dir: Path,
    output_path: Path,
) -> None:
    all_rows: list[dict] = []
    for kw in pair_kwargs:
        pair_dir = scenario_summary_dir / kw["building"] / kw["city"]
        pair_rows = _load_trigger_rows(pair_dir)
        all_rows.extend(pair_rows[key] for key in sorted(pair_rows, key=lambda item: (item[0], item[1])))

    if not all_rows:
        return

    preferred = [
        "scenario",
        "building",
        "city",
        "generation",
        "year",
        "freeze_year",
        "sql_path",
        "threshold_breached",
        "break_triggered",
        "all_trigger_metrics",
        "all_trigger_metric_keys",
        "all_trigger_subfamilies",
        "first_trigger_metric",
        "first_trigger_metric_key",
        "trigger_subfamily",
    ]
    seen = set(preferred)
    extras: list[str] = []
    for row in all_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                extras.append(key)
    fieldnames = preferred + sorted(extras)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="ADAPTIVE RE-FREEZE: Year-by-year HVAC re-sizing on climate failure"
    )
    parser.add_argument("--epw-root", type=Path, required=True,
                        help="Root folder with future-climate EPW files")
    parser.add_argument("--summary-root", type=Path, default=Path("refreeze_results"),
                        help="Root directory for CSV/JSON/state/frozen-model outputs")
    parser.add_argument("--sim-root", type=Path, default=None,
                        help="Root directory for heavy simulation outputs (default: same as --summary-root)")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Deprecated alias for --summary-root when sim and summary live together")
    parser.add_argument("--metric-res-root", type=Path, default=Path("metric_res"),
                        help="Root directory containing metric_res/<scenario>/paper_metric_screening.csv")
    parser.add_argument("--frozen-models", type=Path, default=Path("frozen_models"),
                        help="Directory with TMY-frozen models (gen 0)")
    parser.add_argument("--source-models", type=Path, default=Path("source_models"),
                        help="Directory with patched source models (for re-freezing)")
    parser.add_argument("--local-eplus", type=str, default=None,
                        help="Path to local EnergyPlus executable")
    parser.add_argument("--docker-name", type=str, default=None,
                        help="Docker container name with EnergyPlus")
    parser.add_argument("--scenario-name", type=str, default="scenario",
                        help="Scenario label used in output paths and CSV")
    parser.add_argument("--threshold", type=float, default=100.0,
                        help="Legacy occupied cooling unmet-hours threshold (kept for compatibility; not used for screened-metric triggering)")
    parser.add_argument("--retrofit-lag", type=int, default=0,
                        help="Years after break-year to use as sizing EPW (default: 0)")
    parser.add_argument("--max-gens", type=int, default=100,
                        help="Maximum re-freeze generations per pair (default: 100)")
    parser.add_argument("--epw-suffix", type=str, default="climatedelta",
                        help="EPW filename suffix (default: climatedelta)")
    parser.add_argument("--buildings", type=str, default="office,apartment,retail")
    parser.add_argument("--cities", type=str, default=None)
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel (building, city) workers (default: 3)")
    parser.add_argument("--force-restart", action="store_true",
                        help="Ignore saved state; restart from year 2025 for all pairs")
    args = parser.parse_args()

    if not args.local_eplus and not args.docker_name:
        print("ERROR: Must specify --local-eplus or --docker-name")
        return 1

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    executor: dict = (
        {"mode": "docker", "name": args.docker_name}
        if args.docker_name
        else {"mode": "local", "path": args.local_eplus}
    )

    building_list = [b.strip() for b in args.buildings.split(",")]
    buildings = {k: v for k, v in BUILDINGS.items() if k in building_list}
    cities = (
        {k: v for k, v in CITIES.items() if k in [c.strip() for c in args.cities.split(",")]}
        if args.cities else CITIES
    )

    epw_root          = (script_dir / args.epw_root).resolve()
    if args.output_root is not None:
        summary_root_arg = args.output_root
        sim_root_arg = args.output_root if args.sim_root is None else args.sim_root
    else:
        summary_root_arg = args.summary_root
        sim_root_arg = args.sim_root if args.sim_root is not None else args.summary_root

    summary_root      = (script_dir / summary_root_arg).resolve()
    sim_root          = (script_dir / sim_root_arg).resolve()
    metric_res_root   = (script_dir / args.metric_res_root).resolve()
    frozen_models_dir = (script_dir / args.frozen_models).resolve()
    source_models_dir = (script_dir / args.source_models).resolve()
    scenario_summary_out = summary_root / args.scenario_name
    scenario_sim_out     = sim_root / args.scenario_name
    screened_metrics  = load_screened_failure_metrics(metric_res_root, args.scenario_name)

    print("=" * 70)
    print("ADAPTIVE RE-FREEZE  (year-by-year sequential)")
    print("=" * 70)
    print(f"Scenario:     {args.scenario_name}")
    print(f"Metric res:   {metric_res_root / args.scenario_name / 'paper_metric_screening.csv'}")
    print(f"Legacy thresh:{args.threshold} h unmet cooling (ignored for break triggering)")
    print(f"Retrofit lag: {args.retrofit_lag} yr")
    print(f"Max gens:     {args.max_gens}")
    print(f"EPW root:     {epw_root}")
    print(f"Summary out:  {scenario_summary_out}")
    print(f"Sim out:      {scenario_sim_out / 'sim'}")
    print(f"Buildings:    {', '.join(buildings)}")
    print(f"Cities:       {', '.join(cities)}")
    print(f"Workers:      {args.workers}")
    print("Break metrics:" + " ".join(metric.key for metric in screened_metrics))
    print()

    pair_kwargs = [
        {
            "building":          bldg,
            "city":              city,
            "source_models_dir": source_models_dir,
            "frozen_models_dir": frozen_models_dir,
            "epw_root":          epw_root,
            "scenario_summary_dir": scenario_summary_out,
            "scenario_sim_dir": scenario_sim_out,
            "scenario_name":     args.scenario_name,
            "executor":          executor,
            "screened_metrics":  screened_metrics,
            "threshold":         args.threshold,
            "max_gens":          args.max_gens,
            "epw_suffix":        args.epw_suffix,
            "retrofit_lag":      args.retrofit_lag,
            "force_restart":     args.force_restart,
        }
        for bldg in buildings
        for city in cities
    ]

    all_sequences: list[list[dict]] = []

    if args.workers == 1:
        for kw in pair_kwargs:
            all_sequences.append(run_adaptive_refreeze_pair(**kw))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_pair_worker, kw): kw for kw in pair_kwargs}
            for fut in as_completed(futures):
                kw = futures[fut]
                label = f"{kw['building']}/{kw['city']}"
                try:
                    seq = fut.result()
                    all_sequences.append(seq)
                    print(f"  [DONE] {label}: {len(seq)} gen(s)")
                except Exception as exc:
                    print(f"  [ERROR] {label}: {exc}")
                    all_sequences.append([])

    csv_path = scenario_summary_out / "refreeze_sequence.csv"
    trigger_csv_path = scenario_summary_out / "annual_trigger_metrics.csv"
    write_sequence_csv(all_sequences, csv_path, args.scenario_name)
    aggregate_trigger_csv(pair_kwargs, scenario_summary_out, trigger_csv_path)

    total_events = sum(
        sum(1 for e in seq if e["generation"] > 0) for seq in all_sequences
    )
    never_fail = sum(
        1 for seq in all_sequences
        if any(e["break_year"] == "Never" for e in seq)
    )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Pairs:               {len(pair_kwargs)}")
    print(f"Re-freeze events:    {total_events}")
    print(f"Never-failing pairs: {never_fail}")
    print(f"CSV:                 {csv_path}")
    print(f"Trigger CSV:         {trigger_csv_path}")

    (scenario_summary_out / "summary.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scenario": args.scenario_name,
        "threshold": args.threshold,
        "retrofit_lag": args.retrofit_lag,
        "pairs": len(pair_kwargs),
        "refreeze_events": total_events,
        "never_failing": never_fail,
        "csv": str(csv_path),
        "annual_trigger_metrics_csv": str(trigger_csv_path),
        "summary_root": str(scenario_summary_out),
        "sim_root": str(scenario_sim_out / "sim"),
        "screened_failure_metrics": [metric.key for metric in screened_metrics],
    }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
