#!/usr/bin/env python3
"""
Yearly Climate Simulation Pipeline using sMYM EPW files
Runs building simulations for each year (2001-2100) using pre-generated yearly EPW files
and frozen EPJSON files. Handles EnergyPlus crashes due to extreme weather conditions.

Uses: data/sMYM_epw/{city}/{city}_{year}.epw files
      data/epjson/{city}/ASHRAE901_OfficeMedium_STD2019_{city}_frozen.epJSON
"""

import pandas as pd
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import traceback
import argparse
import sys

# Import only what we need from existing modules
from eppy.modeleditor import IDF


def run_simulation_minimal(epjson_file, weather_file, output_dir, mode="annual"):
    """
    Run EnergyPlus simulation with minimal output (no -r flag) for yearly simulations.
    Custom version for run_smym.py to save disk space.
    mode: 'designday' | 'annual'
    """
    # Set IDD before loading epJSON
    IDF.setiddname("/Applications/EnergyPlus-25-1-0/Energy+.idd")

    os.makedirs(output_dir, exist_ok=True)

    # Base command without -r flag for minimal output
    cmd = [
        "energyplus",
        "-w", weather_file,   # Weather file
        "-d", output_dir,     # Output directory
    ]

    # Add mode flag
    if mode == "annual":
        cmd.insert(1, "-a")  # annual run
        print("Running full-year (annual) simulation with minimal output...")
    elif mode == "designday":
        cmd.insert(1, "-D")  # design-day only
        print("Running design-day (sizing only) simulation with minimal output...")
    else:
        raise ValueError("mode must be 'annual' or 'designday'")

    # Append epJSON file
    cmd.append(epjson_file)

    # Execute with error handling
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"{mode.capitalize()} simulation completed. Output in: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"EnergyPlus simulation failed with return code: {e.returncode}")
        print(f"Command: {' '.join(cmd)}")

        # Check for .err file in output directory
        err_file = os.path.join(output_dir, "eplusout.err")
        if os.path.exists(err_file):
            print("\n--- EnergyPlus Error File Content ---")
            with open(err_file, 'r') as f:
                err_content = f.read()
                print(err_content)

        if e.stderr:
            print(f"\nStderr: {e.stderr}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")

        raise


class YearlyClimateSimulationRunner:
    """Yearly climate simulation runner using sMYM EPW files"""

    def __init__(self, base_dir: str = None, cities: List[str] = None):
        """
        Initialize the yearly simulation pipeline

        Args:
            base_dir: Base directory path
            cities: List of cities to simulate (e.g., ['Los_Angeles', 'Miami'])
                   If None, defaults to all four cities
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent
        else:
            self.base_dir = Path(base_dir)

        # Define paths
        self.data_dir = self.base_dir / "data"
        self.epjson_dir = self.data_dir / "epjson"
        self.smym_epw_dir = self.data_dir / "sMYM_epw"

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_dir / "output" / f"smym_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create crash log file
        self.crash_log_path = self.output_dir / "crash_log.csv"
        self.initialize_crash_log()

        # Cities configuration - allow custom city selection
        all_available_cities = ["Los_Angeles", "Miami", "Montreal", "Toronto", "Phoenix", "Vancouver"]

        if cities is None:
            self.cities = all_available_cities
            print("Using all available cities")
        else:
            # Validate selected cities
            invalid_cities = [city for city in cities if city not in all_available_cities]
            if invalid_cities:
                raise ValueError(f"Invalid cities: {invalid_cities}. Available cities: {all_available_cities}")

            self.cities = cities
            print(f"Using selected cities: {self.cities}")

        print(f"Initialized Yearly Climate Simulation Runner")
        print(f"Base directory: {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"sMYM EPW directory: {self.smym_epw_dir}")
        print(f"Cities to simulate: {self.cities}")
        print(f"Crash log: {self.crash_log_path}")

    def analyze_crash_simulation_progress(self, err_file_path: str) -> Dict[str, str]:
        """
        Analyze EnergyPlus error file to determine crash location (date/time and progress).
        Returns simulation progress information when crash occurred.
        """
        try:
            if not os.path.exists(err_file_path):
                return {}

            with open(err_file_path, 'r') as f:
                err_content = f.read()

            progress_info = {
                'crash_month': '',
                'crash_day': '',
                'crash_hour': '',
                'crash_date_time': '',
                'progress_percentage': '',
                'warmup_completed': '',
                'last_completed_step': ''
            }

            # Look for simulation progress indicators
            lines = err_content.split('\n')

            # Find the last successfully completed simulation step
            for line in reversed(lines):
                line = line.strip()

                # Look for date/time patterns in EnergyPlus output
                # Pattern: "Beginning Zone Sizing Calculations"
                if "Beginning Zone Sizing" in line:
                    progress_info['last_completed_step'] = "Zone Sizing"
                    break
                elif "Beginning System Sizing Calculations" in line:
                    progress_info['last_completed_step'] = "System Sizing"
                    break
                elif "Beginning Plant Sizing Calculations" in line:
                    progress_info['last_completed_step'] = "Plant Sizing"
                    break
                elif "Initializing Simulation" in line:
                    progress_info['last_completed_step'] = "Initialization"
                    break
                elif "Warmup Convergence Information" in line:
                    progress_info['warmup_completed'] = "True"
                elif "Performing Zone Sizing Simulation" in line:
                    progress_info['last_completed_step'] = "Zone Sizing Simulation"
                elif "Performing System Sizing Simulation" in line:
                    progress_info['last_completed_step'] = "System Sizing Simulation"
                elif "Starting Simulation" in line:
                    progress_info['last_completed_step'] = "Annual Simulation Started"

                # Look for specific date/time when crash occurred
                # Pattern: updating or processing specific dates
                if "Update" in line and ("/" in line or ":" in line):
                    # Try to extract date information
                    import re
                    date_pattern = r'(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})'
                    match = re.search(date_pattern, line)
                    if match:
                        progress_info['crash_month'] = match.group(1)
                        progress_info['crash_day'] = match.group(2)
                        progress_info['crash_hour'] = match.group(3)
                        progress_info['crash_date_time'] = f"{match.group(1)}/{match.group(2)} {match.group(3)}:{match.group(4)}"

                        # Calculate approximate progress (assuming 12 months)
                        try:
                            month = int(match.group(1))
                            day = int(match.group(2))
                            hour = int(match.group(3))
                            # Rough calculation: (month-1)*30 + day-1 + hour/24) / 365 * 100
                            approx_day_of_year = (month-1) * 30 + day + hour/24
                            progress_pct = (approx_day_of_year / 365) * 100
                            progress_info['progress_percentage'] = f"{progress_pct:.1f}%"
                        except:
                            pass
                        break

            # If no specific date found, look for other progress indicators
            if not progress_info['crash_date_time']:
                # Look for environment periods or other indicators
                for line in reversed(lines):
                    if "Environment" in line and "started" in line.lower():
                        progress_info['last_completed_step'] = "Environment Started"
                        break
                    elif "Simulation" in line and ("complete" in line.lower() or "finished" in line.lower()):
                        progress_info['last_completed_step'] = "Simulation Completed"
                        progress_info['progress_percentage'] = "100%"
                        break

            return progress_info

        except Exception as e:
            print(f"Warning: Could not analyze crash progress from {err_file_path}: {e}")
            return {}

    def initialize_crash_log(self):
        """Initialize crash log CSV file"""
        crash_log_columns = [
            'timestamp', 'city', 'year', 'crash_type', 'error_message',
            'crash_month', 'crash_day', 'crash_hour', 'crash_date_time',
            'progress_percentage', 'last_completed_step', 'warmup_completed',
            'max_temp_c', 'min_temp_c', 'max_rh_pct', 'min_rh_pct',
            'max_pressure_pa', 'min_pressure_pa', 'max_wind_speed_mps',
            'epw_file_path', 'simulation_output_dir'
        ]

        # Create empty DataFrame and save as CSV
        crash_df = pd.DataFrame(columns=crash_log_columns)
        crash_df.to_csv(self.crash_log_path, index=False)
        print(f"Initialized crash log: {self.crash_log_path}")

    def get_frozen_epjson_path(self, city: str) -> str:
        """Get path to frozen EPJSON file for a city (using _wo_output_frozen suffix)"""
        frozen_epjson_path = self.epjson_dir / city / f"ASHRAE901_OfficeMedium_STD2019_{city}_wo_output_frozen.epJSON"

        if not frozen_epjson_path.exists():
            raise FileNotFoundError(f"Frozen EPJSON not found for {city}: {frozen_epjson_path}")

        return str(frozen_epjson_path)

    def get_yearly_epw_files(self, city: str) -> List[Tuple[int, str]]:
        """Get list of yearly EPW files for a city"""
        city_epw_dir = self.smym_epw_dir / city

        if not city_epw_dir.exists():
            raise FileNotFoundError(f"sMYM EPW directory not found for {city}: {city_epw_dir}")

        # Find all EPW files for this city
        epw_files = []
        for epw_file in city_epw_dir.glob(f"{city}_*.epw"):
            # Extract year from filename
            try:
                year_str = epw_file.stem.split('_')[-1]  # Get last part after underscore
                year = int(year_str)
                epw_files.append((year, str(epw_file)))
            except (ValueError, IndexError):
                print(f"Warning: Could not extract year from filename: {epw_file}")
                continue

        # Sort by year
        epw_files.sort(key=lambda x: x[0])
        print(f"Found {len(epw_files)} yearly EPW files for {city} ({min(epw_files)[0]}-{max(epw_files)[0]})")

        return epw_files

    def analyze_weather_extremes(self, epw_file_path: str) -> Dict[str, float]:
        """Analyze weather file for extreme conditions that might cause E+ crashes"""
        try:
            # Read EPW file and extract weather data
            with open(epw_file_path, 'r') as f:
                lines = f.readlines()

            # Skip header lines (first 8 lines typically)
            weather_data = []
            for line in lines[8:]:
                line = line.strip()
                if line and line[0].isdigit():
                    parts = line.split(',')
                    if len(parts) >= 35:  # Standard EPW format has 35 fields
                        try:
                            # Extract key weather variables
                            temp_c = float(parts[6])  # Dry bulb temperature
                            rh_pct = float(parts[8])  # Relative humidity
                            pressure_pa = float(parts[9])  # Atmospheric pressure
                            wind_speed = float(parts[21]) if parts[21] != '' else 0.0  # Wind speed

                            weather_data.append({
                                'temp_c': temp_c,
                                'rh_pct': rh_pct,
                                'pressure_pa': pressure_pa,
                                'wind_speed_mps': wind_speed
                            })
                        except (ValueError, IndexError):
                            continue

            if not weather_data:
                return {}

            # Convert to DataFrame for analysis
            df = pd.DataFrame(weather_data)

            # Calculate extremes
            extremes = {
                'max_temp_c': df['temp_c'].max(),
                'min_temp_c': df['temp_c'].min(),
                'max_rh_pct': df['rh_pct'].max(),
                'min_rh_pct': df['rh_pct'].min(),
                'max_pressure_pa': df['pressure_pa'].max(),
                'min_pressure_pa': df['pressure_pa'].min(),
                'max_wind_speed_mps': df['wind_speed_mps'].max()
            }

            return extremes

        except Exception as e:
            print(f"Warning: Could not analyze weather extremes for {epw_file_path}: {e}")
            return {}

    def log_crash(self, city: str, year: int, crash_type: str, error_message: str,
                  epw_file_path: str, simulation_output_dir: str, extremes: Dict[str, float]):
        """Log crash information to CSV file"""
        crash_record = {
            'timestamp': datetime.now().isoformat(),
            'city': city,
            'year': year,
            'crash_type': crash_type,
            'error_message': str(error_message)[:500],  # Truncate long error messages
            'crash_month': '',
            'crash_day': '',
            'crash_hour': '',
            'crash_date_time': '',
            'progress_percentage': '',
            'last_completed_step': '',
            'warmup_completed': '',
            'max_temp_c': extremes.get('max_temp_c', ''),
            'min_temp_c': extremes.get('min_temp_c', ''),
            'max_rh_pct': extremes.get('max_rh_pct', ''),
            'min_rh_pct': extremes.get('min_rh_pct', ''),
            'max_pressure_pa': extremes.get('max_pressure_pa', ''),
            'min_pressure_pa': extremes.get('min_pressure_pa', ''),
            'max_wind_speed_mps': extremes.get('max_wind_speed_mps', ''),
            'epw_file_path': epw_file_path,
            'simulation_output_dir': simulation_output_dir
        }

        # Analyze EnergyPlus error file to get crash simulation progress
        err_file_path = os.path.join(simulation_output_dir, "eplusout.err")
        progress_info = self.analyze_crash_simulation_progress(err_file_path)

        # Update crash record with progress information
        crash_record.update(progress_info)

        # Append to crash log CSV
        crash_df = pd.DataFrame([crash_record])
        crash_df.to_csv(self.crash_log_path, mode='a', header=False, index=False)

        print(f"✗ CRASH LOGGED: {city} {year} - {crash_type}")
        print(f"  Error: {str(error_message)[:100]}...")
        if extremes:
            print(f"  Temp range: {extremes.get('min_temp_c', 'N/A'):.1f}°C to {extremes.get('max_temp_c', 'N/A'):.1f}°C")
            print(f"  RH range: {extremes.get('min_rh_pct', 'N/A'):.1f}% to {extremes.get('max_rh_pct', 'N/A'):.1f}%")

    def run_yearly_simulation(self, city: str, year: int, epw_file_path: str,
                             frozen_epjson_path: str) -> bool:
        """Run simulation for one city-year combination"""
        print(f"\n{'='*60}")
        print(f"Running simulation: {city} {year}")
        print(f"{'='*60}")

        # Create year-specific output directory
        year_output_dir = self.output_dir / f"{city}_{year}"
        year_output_dir.mkdir(parents=True, exist_ok=True)

        # Analyze weather extremes before simulation for logging purposes
        extremes = self.analyze_weather_extremes(epw_file_path)

        try:
            # Run EnergyPlus simulation directly with the EPW file
            # (EPW files are pre-validated and cleaned in sMYM_gen.py)
            print(f"Running EnergyPlus simulation...")
            run_simulation_minimal(
                epjson_file=frozen_epjson_path,
                weather_file=epw_file_path,
                output_dir=str(year_output_dir),
                mode="annual"
            )

            # Check if simulation completed successfully
            err_file = year_output_dir / "eplusout.err"
            if err_file.exists():
                with open(err_file, 'r') as f:
                    err_content = f.read()

                # Check for fatal errors or multiple severe errors
                severe_count = err_content.count("** Severe **")
                fatal_count = err_content.count("** Fatal **")

                # More lenient error checking - only fail on fatal errors or many severe errors
                if fatal_count > 0 or severe_count > 20:
                    self.log_crash(city, year, "EnergyPlus_Error",
                                 f"Fatal errors: {fatal_count}, Severe errors: {severe_count}",
                                 epw_file_path, str(year_output_dir), extremes)
                    return False

                # Log warnings about remaining issues if severe errors exist but simulation completed
                if severe_count > 0:
                    print(f"⚠️  Warning: {severe_count} severe errors detected but simulation completed")

            print(f"✓ Successfully completed: {city} {year}")
            return True

        except Exception as e:
            # Log the crash with details
            error_type = type(e).__name__
            self.log_crash(city, year, error_type, str(e),
                         epw_file_path, str(year_output_dir), extremes)
            return False

    def run_city_yearly_simulations(self, city: str) -> Dict[str, int]:
        """Run yearly simulations for one city"""
        print(f"\n{'#'*80}")
        print(f"STARTING YEARLY SIMULATIONS FOR {city.upper()}")
        print(f"{'#'*80}")

        results = {"successful": 0, "failed": 0, "total": 0}

        try:
            # Get frozen EPJSON path
            frozen_epjson_path = self.get_frozen_epjson_path(city)
            print(f"Using frozen EPJSON: {frozen_epjson_path}")

            # Get yearly EPW files
            yearly_epw_files = self.get_yearly_epw_files(city)
            results["total"] = len(yearly_epw_files)

            if not yearly_epw_files:
                print(f"No yearly EPW files found for {city}")
                return results

            # Run simulation for each year
            for year, epw_file_path in yearly_epw_files:
                success = self.run_yearly_simulation(city, year, epw_file_path, frozen_epjson_path)

                if success:
                    results["successful"] += 1
                else:
                    results["failed"] += 1

                # Print progress
                completed = results["successful"] + results["failed"]
                print(f"Progress for {city}: {completed}/{results['total']} completed "
                      f"({results['successful']} successful, {results['failed']} failed)")

        except Exception as e:
            print(f"Error in city simulation for {city}: {e}")
            traceback.print_exc()

        print(f"\n{'#'*80}")
        print(f"COMPLETED YEARLY SIMULATIONS FOR {city.upper()}")
        print(f"Results: {results['successful']}/{results['total']} successful")
        print(f"{'#'*80}")

        return results

    def run_all_cities_yearly_simulations(self) -> Dict[str, Dict[str, int]]:
        """Run yearly simulations for all cities"""
        print(f"\n{'='*100}")
        print("STARTING MULTI-CITY YEARLY CLIMATE SIMULATIONS")
        print(f"{'='*100}")

        all_results = {}

        for city in self.cities:
            city_results = self.run_city_yearly_simulations(city)
            all_results[city] = city_results

        # Print final summary
        print(f"\n{'='*100}")
        print("YEARLY SIMULATION SUMMARY")
        print(f"{'='*100}")

        total_successful = 0
        total_simulations = 0

        for city, results in all_results.items():
            successful = results.get("successful", 0)
            total = results.get("total", 0)
            failed = results.get("failed", 0)

            total_successful += successful
            total_simulations += total

            print(f"{city:15}: {successful:3d}/{total:3d} successful ({failed:3d} failed)")

        print(f"{'':15}  {'='*20}")
        print(f"{'TOTAL':15}: {total_successful:3d}/{total_simulations:3d} successful "
              f"({total_simulations - total_successful:3d} failed)")

        print(f"\nCrash log saved to: {self.crash_log_path}")
        print(f"Output directory: {self.output_dir}")

        return all_results


def main():
    """Main function to run yearly climate simulations"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Yearly Climate Simulation Pipeline")
    parser.add_argument('--cities', type=str, nargs='+', help="List of cities to simulate (e.g., 'Los_Angeles Miami')")
    parser.add_argument('--base_dir', type=str, help="Base directory for the simulation")

    args = parser.parse_args()

    # Initialize yearly simulation runner
    runner = YearlyClimateSimulationRunner(base_dir=args.base_dir, cities=args.cities)

    # Run simulations for all cities
    results = runner.run_all_cities_yearly_simulations()

    return results


if __name__ == "__main__":
    results = main()
