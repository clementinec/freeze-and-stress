#!/usr/bin/env python3
"""
FREEZE: Run TMY sizing and create frozen models with detailed outputs.

Two-step process in one script:
1. Run EnergyPlus with TMY weather to autosize HVAC systems
2. Extract sizing values from EIO and freeze into model + add output config

Supports 3 building types: office_medium, apartment_midrise, retail_standalone
Supports 4 cities: Los_Angeles, Miami, Montreal, Toronto

Usage:
    python freeze_model.py --docker-name thirsty_meitner
    python freeze_model.py --local-eplus /Applications/EnergyPlus-25-1-0/energyplus
    python freeze_model.py --local-eplus /path/to/energyplus --buildings office,retail
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from output_preset import apply_reporting_preset

# Building type configurations
BUILDINGS = {
    "office": {
        "name": "Office Medium",
        "short": "office",
        "source_pattern": "office_medium_{city}_patched.epJSON",
        "frozen_pattern": "office_{city}_frozen_detailed.epJSON",
    },
    "apartment": {
        "name": "Apartment MidRise",
        "short": "apartment",
        "source_pattern": "apartment_midrise_{city}_patched.epJSON",
        "frozen_pattern": "apartment_{city}_frozen_detailed.epJSON",
    },
    "retail": {
        "name": "Retail Standalone",
        "short": "retail",
        "source_pattern": "retail_standalone_{city}_patched.epJSON",
        "frozen_pattern": "retail_{city}_frozen_detailed.epJSON",
    },
}

# City configurations (TMY files are local, patched models come from --source-models)
CITIES = {
    "Los_Angeles": {
        "tmy": "LosAngeles_TMY.epw",
        "lat": 34.05,
        "city_key": "LosAngeles",  # Key used in patched model filenames
    },
    "Miami": {
        "tmy": "Miami_TMY.epw",
        "lat": 25.76,
        "city_key": "Miami",
    },
    "Montreal": {
        "tmy": "Montreal_TMY.epw",
        "lat": 45.50,
        "city_key": "Montreal",
    },
    "Toronto": {
        "tmy": "Toronto_TMY.epw",
        "lat": 43.65,
        "city_key": "Toronto",
    },
}


# ---------------------------------------------------------------------------
# EPW design-day derivation
# ---------------------------------------------------------------------------

def _wet_bulb_from_t_rh(dry_bulb_c: float, rh_pct: float) -> float:
    """
    Approximate wet-bulb temperature from dry-bulb and relative humidity.
    Uses the Stull (2011) formula, accurate to ±0.65°C for RH >= 5%.
    """
    t = dry_bulb_c
    rh = rh_pct
    wb = (
        t * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
        - 4.686035
    )
    return wb


def derive_design_day_from_epw(epw_path: Path) -> dict:
    """
    Parse EPW hourly data to derive summer design day conditions.

    Returns a dict with keys:
      max_dry_bulb_c         - 99.6th-percentile dry-bulb (ASHRAE cooling 0.4% DB)
      coincident_wet_bulb_c  - mean wet-bulb at hours within ±0.5°C of design DB
      daily_dry_bulb_range_c - mean diurnal range over the 10 hottest days
      mean_pressure_pa       - annual mean station pressure

    EPW hourly row layout (0-based, comma-separated after 8 header lines):
      col 0: Year, col 1: Month, col 2: Day, col 3: Hour (1-24)
      col 6: Dry Bulb (°C), col 7: Dew Point (°C), col 8: RH (%)
      col 9: Atmospheric Station Pressure (Pa)
    """
    rows: list[tuple[int, int, int, float, float, float, float]] = []  # month, day, hour, db, dp, rh, press

    with open(epw_path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh):
            if line_no < 8:
                continue
            parts = line.split(",")
            if len(parts) < 10:
                continue
            try:
                month = int(parts[1])
                day = int(parts[2])
                hour = int(parts[3])
                db = float(parts[6])
                dp = float(parts[7])
                rh = float(parts[8])
                press = float(parts[9])
                rows.append((month, day, hour, db, dp, rh, press))
            except (ValueError, IndexError):
                continue

    if not rows:
        raise ValueError(f"No parseable data rows found in EPW: {epw_path}")

    db_arr = np.array([r[3] for r in rows])
    rh_arr = np.array([r[5] for r in rows])
    press_arr = np.array([r[6] for r in rows])

    # 99.6th-percentile dry-bulb (ASHRAE 0.4% cooling design DB)
    design_db = float(np.percentile(db_arr, 99.6))

    # Mean coincident wet-bulb: average wet-bulb at hours within ±0.5°C of design_db
    wb_list: list[float] = []
    for r in rows:
        if abs(r[3] - design_db) <= 0.5:
            wb_list.append(_wet_bulb_from_t_rh(r[3], r[5]))
    coincident_wb = float(np.mean(wb_list)) if wb_list else _wet_bulb_from_t_rh(design_db, float(np.mean(rh_arr)))

    # Daily dry-bulb range: mean of (max-min) over the 10 hottest days
    daily: dict[tuple[int, int], list[float]] = {}
    for r in rows:
        key = (r[0], r[1])  # (month, day)
        daily.setdefault(key, []).append(r[3])
    daily_ranges = [(max(v) - min(v), max(v)) for v in daily.values() if len(v) >= 20]
    daily_ranges.sort(key=lambda x: -x[1])  # sort by daily max descending
    top10 = daily_ranges[:10]
    daily_range = float(np.mean([r[0] for r in top10])) if top10 else 8.0

    mean_pressure = float(np.mean(press_arr))

    return {
        "max_dry_bulb_c": round(design_db, 1),
        "coincident_wet_bulb_c": round(coincident_wb, 1),
        "daily_dry_bulb_range_c": round(daily_range, 1),
        "mean_pressure_pa": round(mean_pressure, 0),
    }


def patch_summer_design_day(source_model: dict, dd_stats: dict) -> dict:
    """
    Update the SummerDesignDay in SizingPeriod:DesignDay with derived future stats.
    Only updates: maximum_dry_bulb_temperature, wetbulb_or_dewpoint_at_maximum_dry_bulb,
    daily_dry_bulb_temperature_range, barometric_pressure.
    Optical depths (taub, taud), wind, and month/day are preserved.
    The WinterDesignDay is left unchanged.
    Returns the modified model dict (modified in-place).
    """
    dds = source_model.get("SizingPeriod:DesignDay", {})
    for name, dd in dds.items():
        if dd.get("day_type") == "SummerDesignDay":
            dd["maximum_dry_bulb_temperature"] = dd_stats["max_dry_bulb_c"]
            dd["wetbulb_or_dewpoint_at_maximum_dry_bulb"] = dd_stats["coincident_wet_bulb_c"]
            dd["daily_dry_bulb_temperature_range"] = dd_stats["daily_dry_bulb_range_c"]
            dd["barometric_pressure"] = dd_stats["mean_pressure_pa"]
    return source_model


# ---------------------------------------------------------------------------
# EnergyPlus execution wrappers
# ---------------------------------------------------------------------------

def run_tmy_sizing_docker(city: str, model_file: Path, tmy_file: Path, output_dir: Path, docker_name: str) -> bool:
    """Run TMY sizing simulation inside Docker container."""
    output_dir.mkdir(parents=True, exist_ok=True)
    docker_work_dir = "/tmp/tmy_sizing"

    subprocess.run(["docker", "exec", docker_name, "mkdir", "-p", docker_work_dir], check=True)
    subprocess.run(["docker", "cp", str(model_file), f"{docker_name}:{docker_work_dir}/model.epJSON"], check=True)
    subprocess.run(["docker", "cp", str(tmy_file), f"{docker_name}:{docker_work_dir}/weather.epw"], check=True)

    cmd = [
        "docker", "exec", "-w", docker_work_dir, docker_name,
        "energyplus", "-w", "weather.epw", "-d", "output", "model.epJSON"
    ]

    print(f"  Running EnergyPlus sizing simulation...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check success (message goes to stderr in Docker)
    if "EnergyPlus Completed Successfully" in result.stderr:
        subprocess.run(["docker", "cp", f"{docker_name}:{docker_work_dir}/output/.", str(output_dir)], check=True)
        return True

    print(f"  ERROR: Sizing simulation failed")
    print(result.stderr[-500:] if result.stderr else result.stdout[-500:])
    return False


def run_tmy_sizing_local(city: str, model_file: Path, tmy_file: Path, output_dir: Path, eplus_path: str) -> bool:
    """Run TMY sizing simulation using local EnergyPlus."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [eplus_path, "-w", str(tmy_file), "-d", str(output_dir), str(model_file)]

    print(f"  Running EnergyPlus sizing simulation...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check completion
    end_file = output_dir / "eplusout.end"
    if end_file.exists():
        with open(end_file, 'r') as f:
            if "EnergyPlus Completed Successfully" in f.read():
                return True

    print(f"  ERROR: Sizing simulation failed")
    return False


def run_sizing_simulation(
    model_file: Path,
    epw_file: Path,
    output_dir: Path,
    executor: dict,
) -> bool:
    """
    Run an EnergyPlus sizing simulation (unified wrapper).

    executor must be one of:
      {"mode": "local", "path": "/path/to/energyplus"}
      {"mode": "docker", "name": "container_name"}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if executor["mode"] == "docker":
        return run_tmy_sizing_docker("", model_file, epw_file, output_dir, executor["name"])
    else:
        return run_tmy_sizing_local("", model_file, epw_file, output_dir, executor["path"])


# ---------------------------------------------------------------------------
# EIO parsing and model freezing
# ---------------------------------------------------------------------------

def extract_design_values_from_eio(sizing_dir: Path) -> dict:
    """Extract design values from eio file (Component Sizing Information)."""
    design_values = {}
    eio_file = sizing_dir / "eplusout.eio"

    if not eio_file.exists():
        return design_values

    with open(eio_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('!'):
            continue

        if line.strip().startswith('Component Sizing Information'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5 and parts[0] == "Component Sizing Information":
                comp_type = parts[1]
                comp_name = parts[2]
                description = parts[3]

                # Strip units from description
                description = re.sub(r'\s*\[.*?\]\s*$', '', description).strip()

                try:
                    value = float(parts[4])

                    if comp_type not in design_values:
                        design_values[comp_type] = {}
                    if comp_name not in design_values[comp_type]:
                        design_values[comp_type][comp_name] = {}

                    design_values[comp_type][comp_name][description] = value
                except ValueError:
                    pass

    return design_values


def normalize_name(name: str) -> str:
    """Normalize object names for matching."""
    return re.sub(r'[_\s]+', ' ', name.upper().strip())


def field_name_to_description(field_name: str) -> str:
    """Map epJSON field names to EIO description strings."""
    mappings = {
        # Single-speed and generic DX coil fields
        "rated_total_cooling_capacity": "Design Size Nominal Total Capacity",
        "rated_sensible_heat_ratio": "Design Size Rated Sensible Heat Ratio",
        "rated_air_flow_rate": "Design Size Rated Air Flow Rate",
        # TwoSpeed DX coil fields (EIO uses lowercase epJSON field name as-is)
        "high_speed_gross_rated_total_cooling_capacity": "Design Size high_speed_gross_rated_total_cooling_capacity",
        "high_speed_rated_air_flow_rate": "Design Size high_speed_rated_air_flow_rate",
        "high_speed_rated_sensible_heat_ratio": "Design Size high_speed_rated_sensible_heat_ratio",
        "low_speed_gross_rated_total_cooling_capacity": "Design Size low_speed_gross_rated_total_cooling_capacity",
        "low_speed_rated_air_flow_rate": "Design Size low_speed_rated_air_flow_rate",
        "low_speed_gross_rated_sensible_heat_ratio": "Design Size low_speed_gross_rated_sensible_heat_ratio",
        # Water coil fields
        "design_water_flow_rate": "Design Size Design Water Flow Rate",
        "design_air_flow_rate": "Design Size Design Air Flow Rate",
        # Fan fields
        "maximum_flow_rate": "Design Size Maximum Flow Rate",
        "maximum_supply_air_flow_rate": "Design Size Maximum Supply Air Flow Rate",
        # Terminal unit fields
        "maximum_air_flow_rate": "Design Size Maximum Air Flow Rate",
        "maximum_hot_water_or_steam_flow_rate": "Design Size Maximum Reheat Water Flow Rate",
        "maximum_reheat_water_flow_rate": "Design Size Maximum Reheat Water Flow Rate",
        "maximum_hot_water_flow_rate": "Design Size Maximum Hot Water Flow Rate",
        "maximum_cold_water_flow_rate": "Design Size Maximum Cold Water Flow Rate",
        # Heating coil fields (EIO uses lowercase field name)
        "nominal_capacity": "Design Size nominal_capacity",
        "maximum_water_flow_rate": "Design Size Maximum Water Flow Rate",
    }
    return mappings.get(field_name, field_name)


def _extract_sizing_snapshot(design_values: dict, obj_type: str, obj_name: str | None = None) -> dict:
    """
    Extract primary sizing values from design_values for reporting.
    Returns a flat dict of capacity/airflow values found for the given component type.
    """
    snapshot: dict[str, float | None] = {}
    type_data = design_values.get(obj_type, {})
    if not type_data:
        return snapshot

    # Aggregate across all components of this type
    cooling_caps: list[float] = []
    airflows: list[float] = []
    heating_caps: list[float] = []

    for name, fields in type_data.items():
        if obj_name and normalize_name(name) != normalize_name(obj_name):
            continue
        for desc, val in fields.items():
            desc_lower = desc.lower()
            if "cooling capacity" in desc_lower or "total capacity" in desc_lower or "nominal total" in desc_lower:
                if "low speed" not in desc_lower:
                    cooling_caps.append(val)
            if "air flow" in desc_lower and "high speed" not in desc_lower and "low speed" not in desc_lower:
                airflows.append(val)
            if "heating" in desc_lower or ("nominal capacity" in desc_lower and "cooling" not in desc_lower):
                heating_caps.append(val)

    snapshot["total_cooling_capacity_w"] = sum(cooling_caps) if cooling_caps else None
    snapshot["total_airflow_m3s"] = sum(airflows) if airflows else None
    return snapshot


def apply_sizing_to_model(
    base_model_path: Path,
    sizing_dir: Path,
    output_path: Path,
    building_key: str,
) -> tuple[bool, dict]:
    """
    Apply EIO sizing values to a source model, add output preset, and save.

    This is a refactored version of the original freeze_and_add_outputs that also
    returns a sizing_snapshot dict with key capacity/airflow values for reporting.

    Returns (success, sizing_snapshot).
    sizing_snapshot keys: cooling_capacity_w, cooling_airflow_m3s, heating_capacity_w,
                          n_fields_replaced, n_fields_kept_autosize
    """
    # Load base model
    with open(base_model_path, 'r') as f:
        model = json.load(f)

    # Extract design values from EIO
    design_values = extract_design_values_from_eio(sizing_dir)

    if not design_values:
        print(f"  ERROR: No design values found in EIO file")
        return False, {}

    print(f"  Found {len(design_values)} component types in EIO")

    # Track changes
    fields_replaced = 0
    fields_kept_autosize = 0

    # EIO component type mapping (epJSON type → EIO type string)
    eio_type_mapping = {
        "Coil:Cooling:DX:TwoSpeed": "Coil:Cooling:DX:TwoSpeed",
        "Coil:Cooling:DX:SingleSpeed": "Coil:Cooling:DX:SingleSpeed",
        "Coil:Cooling:DX:MultiSpeed": "Coil:Cooling:DX:MultiSpeed",
        "Coil:Cooling:Water": "Coil:Cooling:Water",
        "Coil:Heating:Water": "Coil:Heating:Water",
        "Coil:Heating:Electric": "Coil:Heating:Electric",
        "Coil:Heating:Fuel": "Coil:Heating:Fuel",
        "Fan:VariableVolume": "Fan:VariableVolume",
        "Fan:ConstantVolume": "Fan:ConstantVolume",
        "Fan:OnOff": "Fan:OnOff",
        "Fan:SystemModel": "Fan:SystemModel",
        "AirTerminal:SingleDuct:VAV:Reheat": "AirTerminal:SingleDuct:VAV:Reheat",
        "ZoneHVAC:FourPipeFanCoil": "ZoneHVAC:FourPipeFanCoil",
        "AirLoopHVAC:UnitarySystem": "AirLoopHVAC:UnitarySystem",
    }

    # Replace Autosize fields with actual values
    for obj_type, objects in model.items():
        if not isinstance(objects, dict):
            continue

        eio_type = eio_type_mapping.get(obj_type)
        if not eio_type or eio_type not in design_values:
            continue

        for obj_name, obj_data in objects.items():
            # Try exact match first, then normalized
            component_data = design_values[eio_type].get(obj_name)

            if not component_data:
                normalized_obj_name = normalize_name(obj_name)
                for eio_name, eio_data in design_values.get(eio_type, {}).items():
                    if normalize_name(eio_name) == normalized_obj_name:
                        component_data = eio_data
                        break

            if not component_data:
                continue

            for field_name, field_value in obj_data.items():
                if field_value == "Autosize":
                    # Skip ratio/fraction fields
                    if "ratio" in field_name.lower() or "fraction" in field_name.lower():
                        fields_kept_autosize += 1
                        continue

                    eio_description = field_name_to_description(field_name)
                    if eio_description in component_data:
                        obj_data[field_name] = component_data[eio_description]
                        fields_replaced += 1

    preset_counts = apply_reporting_preset(model, building_key)

    # Save frozen model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(model, f, indent=2)

    size_kb = output_path.stat().st_size / 1024
    print(f"  Frozen: {output_path.name} ({size_kb:.1f} KB)")
    print(f"    Replaced {fields_replaced} autosize fields, kept {fields_kept_autosize} ratios")
    print(
        f"    Added {preset_counts['variables']} output variables, "
        f"{preset_counts['meters']} meters, {preset_counts['summary_reports']} summary reports, SQLite"
    )

    # Build sizing snapshot
    cooling_caps: list[float] = []
    airflows: list[float] = []
    heating_caps: list[float] = []

    for obj_type in ("Coil:Cooling:DX:TwoSpeed", "Coil:Cooling:DX:SingleSpeed",
                     "Coil:Cooling:DX:MultiSpeed", "Coil:Cooling:Water"):
        if obj_type not in design_values:
            continue
        for comp_fields in design_values[obj_type].values():
            for desc, val in comp_fields.items():
                dl = desc.lower()
                if any(k in dl for k in ("cooling capacity", "total capacity", "nominal total")):
                    if "low speed" not in dl:
                        cooling_caps.append(val)
                if "air flow" in dl and "high speed" not in dl and "low speed" not in dl:
                    airflows.append(val)

    for obj_type in ("Coil:Heating:Electric", "Coil:Heating:Fuel", "Coil:Heating:Water"):
        if obj_type not in design_values:
            continue
        for comp_fields in design_values[obj_type].values():
            for desc, val in comp_fields.items():
                dl = desc.lower()
                if "nominal capacity" in dl or "heating capacity" in dl:
                    heating_caps.append(val)

    sizing_snapshot = {
        "cooling_capacity_w": sum(cooling_caps) if cooling_caps else None,
        "cooling_airflow_m3s": sum(airflows) if airflows else None,
        "heating_capacity_w": sum(heating_caps) if heating_caps else None,
        "n_fields_replaced": fields_replaced,
        "n_fields_kept_autosize": fields_kept_autosize,
    }

    return True, sizing_snapshot


# Backward-compatible alias used by existing main()
def freeze_and_add_outputs(
    base_model_path: Path,
    sizing_dir: Path,
    output_path: Path,
    building_key: str,
) -> bool:
    """Backward-compatible wrapper around apply_sizing_to_model."""
    success, _ = apply_sizing_to_model(base_model_path, sizing_dir, output_path, building_key)
    return success


# ---------------------------------------------------------------------------
# re_freeze: full future-climate re-sizing pipeline
# ---------------------------------------------------------------------------

def re_freeze(
    source_model_path: Path,
    epw_path: Path,
    sizing_output_dir: Path,
    frozen_output_path: Path,
    building_key: str,
    executor: dict,
    update_design_days: bool = True,
) -> tuple[dict | None, dict | None]:
    """
    Re-freeze a building model for future climate conditions.

    Steps:
      1. Load the patched source model (still has Autosize fields).
      2. If update_design_days=True, derive design day statistics from epw_path
         (99.6th-percentile DB, coincident WB, daily range) and patch the
         SummerDesignDay in the model before running EnergyPlus.
      3. Run EnergyPlus sizing to produce eplusout.eio.
      4. Replace Autosize fields with sized values; add output preset; save.

    Args:
      source_model_path:  patched.epJSON with Autosize HVAC fields.
      epw_path:           future-climate EPW used to derive design day conditions.
      sizing_output_dir:  directory for EnergyPlus sizing outputs (eplusout.eio etc.).
      frozen_output_path: where to write the new frozen model (.epJSON).
      building_key:       "office", "apartment", or "retail".
      executor:           {"mode": "local", "path": ...} | {"mode": "docker", "name": ...}
      update_design_days: if True, update SummerDesignDay from EPW statistics.

    Returns:
      (frozen_model_dict, sizing_snapshot) on success, (None, None) on failure.
      sizing_snapshot contains capacity/airflow values and design day conditions.
    """
    print(f"  re_freeze: {source_model_path.name} + {epw_path.name}")

    # Load source model (always starts from Autosize state)
    with open(source_model_path, 'r') as f:
        model = json.load(f)

    # Optionally derive and patch design day conditions from future EPW
    dd_stats: dict = {}
    if update_design_days:
        print(f"  Deriving design day conditions from EPW...")
        try:
            dd_stats = derive_design_day_from_epw(epw_path)
            print(
                f"    Design DB: {dd_stats['max_dry_bulb_c']}°C  "
                f"WB: {dd_stats['coincident_wet_bulb_c']}°C  "
                f"Daily range: {dd_stats['daily_dry_bulb_range_c']}°C"
            )
            patch_summer_design_day(model, dd_stats)
        except Exception as exc:
            print(f"  WARNING: Could not derive design day from EPW: {exc}")
            dd_stats = {}

    # Write the (possibly patched) model to a temp file for EnergyPlus
    sizing_output_dir.mkdir(parents=True, exist_ok=True)
    temp_model_path = sizing_output_dir / "_sizing_model.epJSON"
    with open(temp_model_path, 'w') as f:
        json.dump(model, f, indent=2)

    # Run EnergyPlus sizing
    sizing_ok = run_sizing_simulation(temp_model_path, epw_path, sizing_output_dir, executor)
    if not sizing_ok:
        print(f"  ERROR: Sizing simulation failed for {source_model_path.name}")
        return None, None

    # Apply sized values to the original source model and save frozen model
    success, sizing_snapshot = apply_sizing_to_model(
        source_model_path, sizing_output_dir, frozen_output_path, building_key
    )
    if not success:
        return None, None

    # Augment snapshot with design day info
    sizing_snapshot["design_day_db_temp_c"] = dd_stats.get("max_dry_bulb_c")
    sizing_snapshot["design_day_wb_temp_c"] = dd_stats.get("coincident_wet_bulb_c")
    sizing_snapshot["design_day_range_c"] = dd_stats.get("daily_dry_bulb_range_c")

    # Load and return the frozen model dict
    with open(frozen_output_path, 'r') as f:
        frozen_model = json.load(f)

    return frozen_model, sizing_snapshot


# ---------------------------------------------------------------------------
# CLI entry point (unchanged behavior)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FREEZE: Run TMY sizing and create frozen models")
    parser.add_argument("--docker-name", type=str, default=None, help="Docker container name with EnergyPlus")
    parser.add_argument("--local-eplus", type=str, default=None, help="Path to local EnergyPlus executable")
    parser.add_argument("--output-dir", type=Path, default=Path("frozen_models"), help="Output directory for frozen models")
    parser.add_argument("--sizing-dir", type=Path, default=Path("tmy_sizing"), help="Directory for TMY sizing outputs")
    parser.add_argument("--source-models", type=Path, default=None,
                        help="Directory with climate-patched source models (default: ./source_models)")
    parser.add_argument("--buildings", type=str, default="office,apartment,retail",
                        help="Comma-separated list of building types: office,apartment,retail (default: all)")
    parser.add_argument("--cities", type=str, default=None,
                        help="Comma-separated list of cities (default: all)")
    args = parser.parse_args()

    if not args.docker_name and not args.local_eplus:
        print("ERROR: Must specify either --docker-name or --local-eplus")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    # Default source models location
    if args.source_models:
        source_models_dir = args.source_models
    else:
        source_models_dir = script_dir / "source_models"

    # Filter buildings
    building_list = [b.strip() for b in args.buildings.split(",")]
    buildings = {k: v for k, v in BUILDINGS.items() if k in building_list}

    # Filter cities
    if args.cities:
        city_list = [c.strip() for c in args.cities.split(",")]
        cities = {k: v for k, v in CITIES.items() if k in city_list}
    else:
        cities = CITIES

    total_models = len(buildings) * len(cities)

    executor: dict
    if args.docker_name:
        executor = {"mode": "docker", "name": args.docker_name}
    else:
        executor = {"mode": "local", "path": args.local_eplus}

    print("=" * 70)
    print("FREEZE: TMY SIZING + MODEL FREEZING + OUTPUT CONFIGURATION")
    print("=" * 70)
    print()
    print("This script will:")
    print("  1. Run EnergyPlus with TMY weather to autosize HVAC systems")
    print("  2. Extract sizing values from EIO files")
    print("  3. Replace 'Autosize' fields with actual values")
    print("  4. Add detailed output variables, meters, and SQLite output")
    print()
    print(f"Buildings: {len(buildings)} ({', '.join(buildings.keys())})")
    print(f"Cities: {len(cities)} ({', '.join(cities.keys())})")
    print(f"Total models: {total_models}")
    print(f"Source models: {source_models_dir}")
    print(f"Output: {args.output_dir}")
    print()

    if not source_models_dir.exists():
        print(f"ERROR: Source models directory not found: {source_models_dir}")
        sys.exit(1)

    results = {}

    for bldg_key, bldg_cfg in buildings.items():
        print(f"\n{'='*70}")
        print(f"BUILDING: {bldg_cfg['name'].upper()}")
        print(f"{'='*70}")

        results[bldg_key] = {}

        for city, city_cfg in cities.items():
            print(f"\n  {'='*50}")
            print(f"  {bldg_cfg['name']} - {city}")
            print(f"  {'='*50}")

            # Use city_key for patched model filenames (LosAngeles vs Los_Angeles)
            city_key = city_cfg["city_key"]
            model_file = source_models_dir / bldg_cfg["source_pattern"].format(city=city_key)
            tmy_file = script_dir / city_cfg["tmy"]
            sizing_output = args.sizing_dir / bldg_key / city
            frozen_output = args.output_dir / bldg_cfg["frozen_pattern"].format(city=city_key)

            if not model_file.exists():
                print(f"    ERROR: Model not found: {model_file}")
                results[bldg_key][city] = False
                continue

            if not tmy_file.exists():
                print(f"    ERROR: TMY file not found: {tmy_file}")
                results[bldg_key][city] = False
                continue

            # Step 1: Run TMY sizing
            print(f"\n    Step 1: Running TMY sizing...")
            sizing_ok = run_sizing_simulation(model_file, tmy_file, sizing_output, executor)

            if not sizing_ok:
                results[bldg_key][city] = False
                continue

            # Step 2: Freeze model and add outputs
            print(f"\n    Step 2: Freezing model and adding outputs...")
            freeze_ok = freeze_and_add_outputs(model_file, sizing_output, frozen_output, bldg_key)
            results[bldg_key][city] = freeze_ok

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    total_success = 0
    for bldg_key, bldg_cfg in buildings.items():
        if bldg_key in results:
            bldg_success = sum(results[bldg_key].values())
            bldg_total = len(results[bldg_key])
            total_success += bldg_success
            status = "OK" if bldg_success == bldg_total else "PARTIAL"
            print(f"[{status}] {bldg_cfg['name']}: {bldg_success}/{bldg_total}")
            for city, success in results[bldg_key].items():
                city_status = "OK" if success else "FAILED"
                print(f"      [{city_status}] {city}")
        print()

    print(f"Total: {total_success}/{total_models} frozen models created")

    if total_success == total_models:
        print(f"\nAll models frozen successfully!")
        print(f"Frozen models in: {args.output_dir}/")
        print(f"\nNext: Run STRESS test with run_future_simulations.py")
    else:
        print(f"\n{total_models - total_success} model(s) failed - check output above")

    return 0 if total_success == total_models else 1


if __name__ == "__main__":
    sys.exit(main())
