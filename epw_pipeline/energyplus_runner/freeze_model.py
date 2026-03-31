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
import os
import re
import subprocess
import sys
from pathlib import Path

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
        "rated_total_cooling_capacity": "Design Size Nominal Total Capacity",
        "rated_sensible_heat_ratio": "Design Size Rated Sensible Heat Ratio",
        "rated_air_flow_rate": "Design Size Rated Air Flow Rate",
        "design_water_flow_rate": "Design Size Design Water Flow Rate",
        "design_air_flow_rate": "Design Size Design Air Flow Rate",
        "maximum_flow_rate": "Design Size Maximum Flow Rate",
        "maximum_supply_air_flow_rate": "Design Size Maximum Supply Air Flow Rate",
        "maximum_hot_water_flow_rate": "Design Size Maximum Hot Water Flow Rate",
        "maximum_cold_water_flow_rate": "Design Size Maximum Cold Water Flow Rate",
        "maximum_air_flow_rate": "Design Size Maximum Air Flow Rate",
        "maximum_hot_water_or_steam_flow_rate": "Design Size Maximum Reheat Water Flow Rate",
        "maximum_reheat_water_flow_rate": "Design Size Maximum Reheat Water Flow Rate",
    }
    return mappings.get(field_name, field_name)


def freeze_and_add_outputs(
    base_model_path: Path,
    sizing_dir: Path,
    output_path: Path,
    building_key: str,
) -> bool:
    """Create frozen model with actual sizing values and detailed outputs."""

    # Load base model
    with open(base_model_path, 'r') as f:
        model = json.load(f)

    # Extract design values from EIO
    design_values = extract_design_values_from_eio(sizing_dir)

    if not design_values:
        print(f"  ERROR: No design values found in EIO file")
        return False

    print(f"  Found {len(design_values)} component types in EIO")

    # Track changes
    fields_replaced = 0
    fields_kept_autosize = 0

    # EIO type mapping
    eio_type_mapping = {
        "Coil:Cooling:DX:TwoSpeed": "Coil:Cooling:DX:TwoSpeed",
        "Coil:Cooling:Water": "Coil:Cooling:Water",
        "Coil:Heating:Water": "Coil:Heating:Water",
        "Fan:VariableVolume": "Fan:VariableVolume",
        "AirTerminal:SingleDuct:VAV:Reheat": "AirTerminal:SingleDuct:VAV:Reheat",
        "ZoneHVAC:FourPipeFanCoil": "ZoneHVAC:FourPipeFanCoil",
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

    return True


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
            if args.docker_name:
                sizing_ok = run_tmy_sizing_docker(city, model_file, tmy_file, sizing_output, args.docker_name)
            else:
                sizing_ok = run_tmy_sizing_local(city, model_file, tmy_file, sizing_output, args.local_eplus)

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
