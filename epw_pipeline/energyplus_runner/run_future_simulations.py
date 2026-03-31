#!/usr/bin/env python3
"""
STRESS: Run frozen models through future climate scenarios.

Tests current-climate-sized HVAC systems under future weather conditions.
Runs all years (2025-2100) for all cities and building types.

Supports 3 building types: office, apartment, retail
Supports 4 cities: Los_Angeles, Miami, Montreal, Toronto

Usage:
    python run_future_simulations.py --epw-root ../epw_out/CORDEX_CMIP5_REMO2015_rcp85 --docker-name thirsty_meitner
    python run_future_simulations.py --epw-root ../epw_out/CORDEX_CMIP5_REMO2015_rcp85 --local-eplus /Applications/EnergyPlus-25-1-0/energyplus
    python run_future_simulations.py --epw-root ../epw_out/CORDEX_CMIP5_REMO2015_rcp85 --local-eplus /path/to/eplus --start-year 2025 --end-year 2030
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Building type configurations
BUILDINGS = {
    "office": {
        "name": "Office Medium",
        "short": "office",
        "frozen_pattern": "office_{city}_frozen_detailed.epJSON",
    },
    "apartment": {
        "name": "Apartment MidRise",
        "short": "apartment",
        "frozen_pattern": "apartment_{city}_frozen_detailed.epJSON",
    },
    "retail": {
        "name": "Retail Standalone",
        "short": "retail",
        "frozen_pattern": "retail_{city}_frozen_detailed.epJSON",
    },
}

# City configurations
CITIES = {
    "Los_Angeles": {
        "epw_city_name": "Los_Angeles",  # Name in EPW filenames
        "city_key": "LosAngeles",  # Key used in frozen model filenames
    },
    "Miami": {
        "epw_city_name": "Miami",
        "city_key": "Miami",
    },
    "Montreal": {
        "epw_city_name": "Montreal",
        "city_key": "Montreal",
    },
    "Toronto": {
        "epw_city_name": "Toronto",
        "city_key": "Toronto",
    },
}

# All years from 2025 to 2100
YEARS = list(range(2025, 2101))  # 76 years


def run_simulation_docker(
    model_file: Path,
    epw_file: Path,
    output_dir: Path,
    docker_name: str,
) -> tuple[bool, str | None]:
    """Run a single EnergyPlus simulation inside Docker container."""
    output_dir.mkdir(parents=True, exist_ok=True)
    docker_work_dir = "/tmp/eplus_stress"

    try:
        subprocess.run(["docker", "exec", docker_name, "rm", "-rf", docker_work_dir],
                       capture_output=True, check=False)
        subprocess.run(["docker", "exec", docker_name, "mkdir", "-p", docker_work_dir], check=True)
        subprocess.run(["docker", "cp", str(model_file), f"{docker_name}:{docker_work_dir}/model.epJSON"], check=True)
        subprocess.run(["docker", "cp", str(epw_file), f"{docker_name}:{docker_work_dir}/weather.epw"], check=True)

        cmd = [
            "docker", "exec", "-w", docker_work_dir, docker_name,
            "energyplus", "-w", "weather.epw", "-d", "output", "model.epJSON"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Success message goes to stderr in Docker
        if "EnergyPlus Completed Successfully" in result.stderr:
            subprocess.run(["docker", "cp", f"{docker_name}:{docker_work_dir}/output/.", str(output_dir)], check=True)
            return True, None

        return False, result.stderr[-500:] if result.stderr else result.stdout[-500:]

    except subprocess.CalledProcessError as e:
        return False, str(e)


def run_simulation_local(
    model_file: Path,
    epw_file: Path,
    output_dir: Path,
    eplus_path: str,
) -> tuple[bool, str | None]:
    """Run a single EnergyPlus simulation using local executable."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [eplus_path, "-w", str(epw_file), "-d", str(output_dir), str(model_file)]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check completion
    end_file = output_dir / "eplusout.end"
    if end_file.exists():
        with open(end_file, 'r') as f:
            if "EnergyPlus Completed Successfully" in f.read():
                # Check for fatal errors
                err_file = output_dir / "eplusout.err"
                if err_file.exists():
                    with open(err_file, 'r') as f:
                        if "**  Fatal  **" in f.read():
                            return False, "Fatal error in simulation"
                return True, None

    return False, result.stderr[-500:] if result.stderr else "Simulation failed"


def main():
    parser = argparse.ArgumentParser(description="STRESS: Run frozen models through future climate")
    parser.add_argument("--epw-root", type=Path, required=True, help="Root folder with EPW files (e.g., epw_out_clean_all)")
    parser.add_argument("--frozen-models", type=Path, default=Path("frozen_models"), help="Directory with frozen models")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Output directory for simulation results")
    parser.add_argument("--docker-name", type=str, default=None, help="Docker container name with EnergyPlus")
    parser.add_argument("--local-eplus", type=str, default=None, help="Path to local EnergyPlus executable")
    parser.add_argument("--epw-suffix", type=str, default="climatedelta", help="EPW filename suffix (e.g., climatedelta, hybrid_blend)")
    parser.add_argument("--start-year", type=int, default=2025, help="Start year (default: 2025)")
    parser.add_argument("--end-year", type=int, default=2100, help="End year (default: 2100)")
    parser.add_argument("--cities", type=str, default=None, help="Comma-separated list of cities (default: all)")
    parser.add_argument("--buildings", type=str, default="office,apartment,retail",
                        help="Comma-separated list of building types: office,apartment,retail (default: all)")
    args = parser.parse_args()

    if not args.docker_name and not args.local_eplus:
        print("ERROR: Must specify either --docker-name or --local-eplus")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    # Filter years
    years = [y for y in YEARS if args.start_year <= y <= args.end_year]

    # Filter cities
    if args.cities:
        city_list = [c.strip() for c in args.cities.split(",")]
        cities = {k: v for k, v in CITIES.items() if k in city_list}
    else:
        cities = CITIES

    # Filter buildings
    building_list = [b.strip() for b in args.buildings.split(",")]
    buildings = {k: v for k, v in BUILDINGS.items() if k in building_list}

    total_sims = len(buildings) * len(cities) * len(years)

    print("=" * 70)
    print("STRESS: FUTURE CLIMATE SIMULATIONS WITH FROZEN MODELS")
    print("=" * 70)
    print()
    print(f"Buildings: {len(buildings)} ({', '.join(buildings.keys())})")
    print(f"Cities: {len(cities)} ({', '.join(cities.keys())})")
    print(f"Years: {len(years)} ({years[0]}-{years[-1]})")
    print(f"Total simulations: {total_sims}")
    print(f"EPW root: {args.epw_root}")
    print(f"EPW suffix: {args.epw_suffix}")
    print(f"Frozen models: {args.frozen_models}")
    print(f"Output: {args.output_dir}")
    print()

    if not HAS_TQDM:
        print("Note: Install tqdm for progress bar (pip install tqdm)")
        print()

    # Build task list
    tasks = []
    simulation_details = []  # Store detailed info per simulation

    for bldg_key, bldg_cfg in buildings.items():
        for city, cfg in cities.items():
            city_key = cfg["city_key"]
            frozen_model = args.frozen_models / bldg_cfg["frozen_pattern"].format(city=city_key)

            if not frozen_model.exists():
                print(f"WARNING: Frozen model not found: {frozen_model}")
                continue

            for year in years:
                epw_file = args.epw_root / city / f"{cfg['epw_city_name']}_{year}_{args.epw_suffix}.epw"
                output_dir = args.output_dir / bldg_key / city / str(year)
                tasks.append((bldg_key, city, year, frozen_model, epw_file, output_dir))

    if not tasks:
        print("ERROR: No valid simulation tasks found")
        sys.exit(1)

    # Run simulations
    results = {}  # results[bldg][city][year] = True/False
    failed_tasks = []
    start_time = datetime.now()

    if HAS_TQDM:
        pbar = tqdm(tasks, desc="Running", unit="sim",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for bldg_key, city, year, frozen_model, epw_file, output_dir in pbar:
            pbar.set_description(f"{bldg_key}/{city}/{year}")

            if bldg_key not in results:
                results[bldg_key] = {}
            if city not in results[bldg_key]:
                results[bldg_key][city] = {}

            sim_info = {
                "building": bldg_key,
                "city": city,
                "year": year,
                "frozen_model": str(frozen_model.name),
                "epw_file": str(epw_file.name) if epw_file.exists() else None,
                "output_dir": str(output_dir),
            }

            if not epw_file.exists():
                results[bldg_key][city][year] = False
                sim_info["success"] = False
                sim_info["error"] = f"EPW not found: {epw_file}"
                failed_tasks.append((bldg_key, city, year, sim_info["error"]))
            else:
                if args.docker_name:
                    success, error = run_simulation_docker(frozen_model, epw_file, output_dir, args.docker_name)
                else:
                    success, error = run_simulation_local(frozen_model, epw_file, output_dir, args.local_eplus)

                results[bldg_key][city][year] = success
                sim_info["success"] = success
                if not success:
                    sim_info["error"] = error
                    failed_tasks.append((bldg_key, city, year, error))
                else:
                    # Add output file info
                    sqlite_path = output_dir / "eplusout.sql"
                    sim_info["has_sqlite"] = sqlite_path.exists()

            simulation_details.append(sim_info)

            # Update postfix
            total_done = sum(sum(len(yrs) for yrs in c.values()) for c in results.values())
            total_success = sum(sum(sum(yrs.values()) for yrs in c.values()) for c in results.values())
            pbar.set_postfix(ok=f"{total_success}/{total_done}")
    else:
        count = 0
        for bldg_key, city, year, frozen_model, epw_file, output_dir in tasks:
            count += 1

            if bldg_key not in results:
                results[bldg_key] = {}
                print(f"\n{buildings[bldg_key]['name']}:")
            if city not in results[bldg_key]:
                results[bldg_key][city] = {}
                print(f"  {city}:")

            print(f"    [{count}/{len(tasks)}] {year}...", end=" ", flush=True)

            sim_info = {
                "building": bldg_key,
                "city": city,
                "year": year,
                "frozen_model": str(frozen_model.name),
                "epw_file": str(epw_file.name) if epw_file.exists() else None,
                "output_dir": str(output_dir),
            }

            if not epw_file.exists():
                print("EPW missing")
                results[bldg_key][city][year] = False
                sim_info["success"] = False
                sim_info["error"] = "EPW not found"
                failed_tasks.append((bldg_key, city, year, "EPW not found"))
            else:
                if args.docker_name:
                    success, error = run_simulation_docker(frozen_model, epw_file, output_dir, args.docker_name)
                else:
                    success, error = run_simulation_local(frozen_model, epw_file, output_dir, args.local_eplus)

                results[bldg_key][city][year] = success
                sim_info["success"] = success
                print("OK" if success else "FAILED")

                if not success:
                    sim_info["error"] = error
                    failed_tasks.append((bldg_key, city, year, error))
                else:
                    sqlite_path = output_dir / "eplusout.sql"
                    sim_info["has_sqlite"] = sqlite_path.exists()

            simulation_details.append(sim_info)

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    total_success = sum(sum(sum(yrs.values()) for yrs in c.values()) for c in results.values())
    total_done = sum(sum(len(yrs) for yrs in c.values()) for c in results.values())

    print(f"Completed: {total_success}/{total_done} simulations")
    print(f"Elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)\n")

    for bldg_key, bldg_cfg in buildings.items():
        if bldg_key in results:
            bldg_success = sum(sum(yrs.values()) for yrs in results[bldg_key].values())
            bldg_total = sum(len(yrs) for yrs in results[bldg_key].values())
            status = "OK" if bldg_success == bldg_total else "PARTIAL"
            print(f"[{status}] {bldg_cfg['name']}: {bldg_success}/{bldg_total}")
            for city in cities:
                if city in results[bldg_key]:
                    city_success = sum(results[bldg_key][city].values())
                    city_total = len(results[bldg_key][city])
                    city_status = "OK" if city_success == city_total else "PARTIAL"
                    print(f"      [{city_status}] {city}: {city_success}/{city_total}")

    # Save detailed summary JSON
    summary_path = args.output_dir / "simulation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_info": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "elapsed_seconds": elapsed,
            "epw_root": str(args.epw_root.resolve()),
            "epw_suffix": args.epw_suffix,
            "frozen_models_dir": str(args.frozen_models.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "year_range": [args.start_year, args.end_year],
            "executor": "docker" if args.docker_name else "local",
        },
        "totals": {
            "buildings": len(buildings),
            "cities": len(cities),
            "years": len(years),
            "total_simulations": total_done,
            "successful": total_success,
            "failed": total_done - total_success,
        },
        "by_building": {
            bldg_key: {
                "name": bldg_cfg["name"],
                "success": sum(sum(yrs.values()) for yrs in results.get(bldg_key, {}).values()),
                "total": sum(len(yrs) for yrs in results.get(bldg_key, {}).values()),
            }
            for bldg_key, bldg_cfg in buildings.items()
        },
        "by_city": {
            city: {
                "success": sum(sum(results.get(b, {}).get(city, {}).values()) for b in buildings),
                "total": sum(len(results.get(b, {}).get(city, {})) for b in buildings),
            }
            for city in cities
        },
        "simulations": simulation_details,  # Full per-simulation details
        "failed_tasks": [
            {"building": b, "city": c, "year": y, "error": e}
            for b, c, y, e in failed_tasks[:100]  # First 100 failures
        ],
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    if total_success == total_done:
        print(f"\nAll simulations completed successfully!")
        print(f"Results in: {args.output_dir}/")
    else:
        print(f"\n{total_done - total_success} simulation(s) failed")
        if failed_tasks:
            print("\nFirst few failures:")
            for bldg, city, year, error in failed_tasks[:5]:
                err_str = error[:80] if error else "Unknown error"
                print(f"  {bldg}/{city}/{year}: {err_str}")

    return 0 if total_success == total_done else 1


if __name__ == "__main__":
    sys.exit(main())
