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
  <output-root>/<scenario>/
    refreeze_sequence.csv           — aggregated sequence for all pairs
    <building>/<city>/
      state.json                    — per-pair restart state (year-level granularity)
      frozen_gen<N>_<year>epw.epJSON
      sizing_snapshots.json
    sim/gen<N>/<building>/<city>/<year>/eplusout.sql   — simulation outputs

Usage:
    python run_adaptive_refreeze.py \\
        --epw-root ../epw_out/CORDEX_CMIP5_REMO2015_rcp85 \\
        --output-root refreeze_results/ \\
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
from datetime import datetime
from pathlib import Path

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
    "Toronto":     {"epw_city_name": "Toronto",     "city_key": "Toronto"},
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
# SQLite unmet-hours extraction
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


# ---------------------------------------------------------------------------
# Core adaptive loop
# ---------------------------------------------------------------------------

def run_adaptive_refreeze_pair(
    building: str,
    city: str,
    source_models_dir: Path,
    frozen_models_dir: Path,
    epw_root: Path,
    scenario_output_dir: Path,
    executor: dict,
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
    pair_dir  = scenario_output_dir / building / city
    pair_dir.mkdir(parents=True, exist_ok=True)
    state_path = pair_dir / "state.json"

    if force_restart and state_path.exists():
        state_path.unlink()

    state = _load_state(state_path)

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

    # Track per-gen simulation start year (for sequence entry)
    # On resume, gen N starts at (previous gen's break_year + 1)
    gen_start_year  = 2025 if gen == 0 else (
        sequence[-1]["break_year"] + 1 if sequence else 2025
    )

    print(f"\n  [{building}/{city}] Resuming: gen={gen}, current_model={current_frozen.name}")

    for year in YEARS:
        # Skip years before this generation's start year
        if year < gen_start_year:
            continue

        # Determine output dir for this (gen, year) simulation
        sim_out = scenario_output_dir / "sim" / f"gen{gen}" / building / city / str(year)

        # Skip years already completed for this gen
        if year in completed_years and (sim_out / "eplusout.sql").exists():
            unmet = _extract_unmet_hours(sim_out / "eplusout.sql")
        else:
            # Find EPW
            epw_file = epw_root / city / f"{epw_city}_{year}_{epw_suffix}.epw"
            if not epw_file.exists():
                print(f"    [{building}/{city}] [SKIP] EPW not found: {epw_file.name}")
                continue

            ok, err = _run_simulation(current_frozen, epw_file, sim_out, executor)
            if not ok:
                print(f"    [{building}/{city}] [FAIL] {year}: {err}")
                continue

            unmet = _extract_unmet_hours(sim_out / "eplusout.sql")
            completed_years.add(year)
            state["completed_years"] = sorted(completed_years)
            _save_state(state_path, state)

        print(f"    [{building}/{city}] gen{gen} {year}: {unmet:.1f}h unmet" if unmet is not None else
              f"    [{building}/{city}] gen{gen} {year}: unmet=N/A")

        # Check for failure
        if unmet is not None and unmet > threshold and gen < max_gens:
            # ── Record this generation's sequence entry ──────────────────────
            snap = _read_snapshot(pair_dir, gen)
            entry = _build_entry(building, city, gen, state["freeze_year"],
                                 gen_start_year, break_year=year,
                                 unmet_at_break=unmet, threshold=threshold,
                                 snap=snap)
            sequence.append(entry)
            print(f"  [{building}/{city}] *** BREAK gen{gen} at {year} ({unmet:.1f}h) ***")

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
                             snap=snap)
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
) -> dict:
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="ADAPTIVE RE-FREEZE: Year-by-year HVAC re-sizing on climate failure"
    )
    parser.add_argument("--epw-root", type=Path, required=True,
                        help="Root folder with future-climate EPW files")
    parser.add_argument("--output-root", type=Path, default=Path("refreeze_results"),
                        help="Root output directory")
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
                        help="Occupied cooling unmet hours threshold (default: 100)")
    parser.add_argument("--retrofit-lag", type=int, default=0,
                        help="Years after break-year to use as sizing EPW (default: 0)")
    parser.add_argument("--max-gens", type=int, default=6,
                        help="Maximum re-freeze generations per pair (default: 6)")
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
    output_root       = (script_dir / args.output_root).resolve()
    frozen_models_dir = (script_dir / args.frozen_models).resolve()
    source_models_dir = (script_dir / args.source_models).resolve()
    scenario_out      = output_root / args.scenario_name

    print("=" * 70)
    print("ADAPTIVE RE-FREEZE  (year-by-year sequential)")
    print("=" * 70)
    print(f"Scenario:     {args.scenario_name}")
    print(f"Threshold:    {args.threshold} h unmet cooling")
    print(f"Retrofit lag: {args.retrofit_lag} yr")
    print(f"Max gens:     {args.max_gens}")
    print(f"EPW root:     {epw_root}")
    print(f"Output:       {scenario_out}")
    print(f"Buildings:    {', '.join(buildings)}")
    print(f"Cities:       {', '.join(cities)}")
    print(f"Workers:      {args.workers}")
    print()

    pair_kwargs = [
        {
            "building":          bldg,
            "city":              city,
            "source_models_dir": source_models_dir,
            "frozen_models_dir": frozen_models_dir,
            "epw_root":          epw_root,
            "scenario_output_dir": scenario_out,
            "executor":          executor,
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

    csv_path = scenario_out / "refreeze_sequence.csv"
    write_sequence_csv(all_sequences, csv_path, args.scenario_name)

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

    (scenario_out / "summary.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scenario": args.scenario_name,
        "threshold": args.threshold,
        "retrofit_lag": args.retrofit_lag,
        "pairs": len(pair_kwargs),
        "refreeze_events": total_events,
        "never_failing": never_fail,
        "csv": str(csv_path),
    }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
