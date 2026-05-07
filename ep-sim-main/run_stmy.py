#!/usr/bin/env python3
"""
Climate Change Simulation Pipeline
Runs building simulations under future climate scenarios for 4 cities.

Pipeline:
1. Use current TMY + DDY with standard epJSON to autosize components
2. Freeze the sized components in epJSON
3. Run simulations with frozen epJSON + future climate data (sTMY/AMY)

Cities: Los Angeles, Miami, Montreal, Toronto
"""

from pathlib import Path
from typing import Dict
from datetime import datetime

# Import existing modules
from city_epjson_gen import generate_city_epjson, generate_output_variables
from size_freeze import SizeFreeze
from run_pyEP import run_simulation
from fix_weather_file import fix_weather_file


class ClimateSimulationRunner:
    """Climate simulation runner for buildings"""

    def __init__(self, base_dir: str = None):
        """Initialize the simulation pipeline"""
        if base_dir is None:
            self.base_dir = Path(__file__).parent
        else:
            self.base_dir = Path(base_dir)

        # Define paths
        self.data_dir = self.base_dir / "data"
        self.epjson_dir = self.data_dir / "epjson"
        self.epw_dir = self.data_dir / "epw"
        self.stmy_dir = self.data_dir / "stmy_out_bias_corrected"
        self.annex80_dir = self.data_dir / "annex80"

        # Create timestamp-based output directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_dir / "output" / timestamp

        # Standard epJSON file
        self.std_epjson = self.epjson_dir / "ASHRAE901_OfficeMedium_STD2019_std.epJSON"

        # Cities to assess with their weather file mappings
        self.cities = {
            "Los_Angeles": {
                "state": "CA",
                "country": "USA",
                "epw_dir": "USA_CA_Los.Angeles.Intl.AP.722950_TMY3",
                "epw_file": "USA_CA_Los.Angeles.Intl.AP.722950_TMY3.epw",
                "ddy_file": "USA_CA_Los.Angeles.Intl.AP.722950_TMY3.ddy",
                "annex80_subdir": "WDTF_Annex80_build_losa_v1.0_1-13",
                "annex80_prefix": "3B_LosAngeles"
            },
            "Miami": {
                "state": "FL",
                "country": "USA",
                "epw_dir": "USA_FL_Miami.Intl.AP.722020_TMY3",
                "epw_file": "USA_FL_Miami.Intl.AP.722020_TMY3.epw",
                "ddy_file": "USA_FL_Miami.Intl.AP.722020_TMY3.ddy",
                "annex80_subdir": None,  # No annex80 data available
                "annex80_prefix": None
            },
            "Montreal": {
                "state": "QC",
                "country": "CAN",
                "epw_dir": "CAN_PQ_Montreal.Intl.AP.716270_CWEC",
                "epw_file": "CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw",
                "ddy_file": "CAN_PQ_Montreal.Intl.AP.716270_CWEC.ddy",
                "annex80_subdir": "WDTF_Annex80_build_mont_v1.0_1-11",
                "annex80_prefix": "6A_Montreal"
            },
            "Toronto": {
                "state": "ON",
                "country": "CAN",
                "epw_dir": "CAN_ON_Toronto.716240_CWEC",
                "epw_file": "CAN_ON_Toronto.716240_CWEC.epw",
                "ddy_file": "CAN_ON_Toronto.716240_CWEC.ddy",
                "annex80_subdir": "WDTF_Annex80_build_toro_v1.0_1-11",
                "annex80_prefix": "5A_Toronto"
            },
            "Phoenix": {
                "state": "AZ",
                "country": "USA",
                "epw_dir": "USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3",
                "epw_file": "USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.epw",
                "ddy_file": "USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.ddy",
                "annex80_subdir": None,
                "annex80_prefix": None
            },
            "Vancouver": {
                "state": "BC",
                "country": "CAN",
                "epw_dir": "CAN_BC_Vancouver.Intl.AP.718920_CWEC",
                "epw_file": "CAN_BC_Vancouver.718920_CWEC.epw",
                "ddy_file": "CAN_BC_Vancouver.718920_CWEC.ddy",
                "annex80_subdir": None,
                "annex80_prefix": None
            }
        }

        # Climate scenarios and years
        self.scenarios = ["2001", "2041", "2081"]  # Current, mid-century, end-century

        print(f"Initialized Climate Simulation Runner")
        print(f"Base directory: {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Standard epJSON: {self.std_epjson}")
        print(f"Cities: {list(self.cities.keys())}")
        print(f"Climate scenarios: {self.scenarios}")

    def get_weather_files(self, city: str) -> Dict[str, str]:
        """Get weather file paths for a city using direct specification"""
        if city not in self.cities:
            raise ValueError(f"City {city} not supported")

        city_info = self.cities[city]
        weather_files = {}

        # Current TMY file - now in subdirectory
        epw_subdir = self.epw_dir / city_info["epw_dir"]
        current_epw = epw_subdir / city_info["epw_file"]
        if current_epw.exists():
            weather_files["current"] = str(current_epw)
        else:
            print(f"Warning: Current EPW not found: {current_epw}")

        # DDY file - now in subdirectory
        ddy_file = epw_subdir / city_info["ddy_file"]
        if ddy_file.exists():
            weather_files["ddy"] = str(ddy_file)
        else:
            print(f"Warning: DDY file not found: {ddy_file}")

        # Future climate files (sTMY) - directly in stmy_dir, not in city subdirectories
        for scenario in self.scenarios:  # Include ALL scenarios including 2001
            # Look for files with scenario year ranges in name
            if scenario == "2001":
                stmy_pattern = f"{city}_stmy_2001_2020.epw"
            elif scenario == "2041":
                stmy_pattern = f"{city}_stmy_2041_2060.epw"
            elif scenario == "2081":
                stmy_pattern = f"{city}_stmy_2081_2100.epw"
            else:
                continue

            stmy_file = self.stmy_dir / stmy_pattern
            if stmy_file.exists():
                weather_files[f"stmy_{scenario}"] = str(stmy_file)
            else:
                # Try with "_fixed" version
                stmy_fixed_pattern = f"{city}_stmy_{scenario}*_fixed.epw"
                stmy_fixed_files = list(self.stmy_dir.glob(stmy_fixed_pattern))
                if stmy_fixed_files:
                    weather_files[f"stmy_{scenario}"] = str(stmy_fixed_files[0])
                else:
                    print(f"Warning: sTMY file for {scenario} not found: {stmy_file}")

        # Annex80 TMY files - if available for this city
        if city_info["annex80_subdir"] and city_info["annex80_prefix"]:
            annex80_city_dir = self.annex80_dir / city / city_info["annex80_subdir"]
            if annex80_city_dir.exists():
                for scenario in self.scenarios:
                    # Look for annex80 TMY files
                    if scenario == "2001":
                        annex80_pattern = f"{city_info['annex80_prefix']}_TMY_2001-2020.epw"
                    elif scenario == "2041":
                        annex80_pattern = f"{city_info['annex80_prefix']}_TMY_2041-2060.epw"
                    elif scenario == "2081":
                        annex80_pattern = f"{city_info['annex80_prefix']}_TMY_2081-2100.epw"
                    else:
                        continue

                    annex80_file = annex80_city_dir / annex80_pattern
                    if annex80_file.exists():
                        weather_files[f"annex80_{scenario}"] = str(annex80_file)
                    else:
                        print(f"Warning: Annex80 TMY file for {scenario} not found: {annex80_file}")
            else:
                print(f"Warning: Annex80 directory not found: {annex80_city_dir}")
        else:
            print(f"Info: No Annex80 data available for {city}")

        return weather_files

    def get_base_epjson(self) -> str:
        """Get the standard epJSON file path"""
        if not self.std_epjson.exists():
            raise FileNotFoundError(f"Standard epJSON file not found: {self.std_epjson}")
        return str(self.std_epjson)

    def create_city_epjson_with_ddy(self, std_epjson: str, city: str, weather_files: Dict[str, str]) -> str:
        """Create city-specific epJSON with TMY + DDY for sizing"""
        print(f"\n{'='*60}")
        print(f"Creating city epJSON with DDY for {city}")
        print(f"{'='*60}")

        if "ddy" not in weather_files:
            raise FileNotFoundError(f"DDY file not found for {city}")

        # Create city-specific directory in epjson folder
        city_epjson_dir = self.epjson_dir / city
        city_epjson_dir.mkdir(exist_ok=True)

        # Create output filename with new naming convention
        output_name = f"ASHRAE901_OfficeMedium_STD2019_{city}.epJSON"
        output_path = city_epjson_dir / output_name

        # Use city_epjson_gen to add location and design days
        city_epjson = generate_city_epjson(
            base_epjson_path=std_epjson,
            ddy_file_path=weather_files["ddy"],
            output_path=str(output_path),
            city_name=city
        )

        # Load the generated epJSON file to add Output:Variable entries
        import json
        with open(city_epjson, 'r', encoding='utf-8') as f:
            epjson_data = json.load(f)

        # Add Output:Variable entries using the generate_output_variables function
        epjson_data = generate_output_variables(epjson_data)

        # Save the modified epJSON file with Output:Variable entries
        with open(city_epjson, 'w', encoding='utf-8') as f:
            json.dump(epjson_data, f, indent=2, ensure_ascii=False)

        print(f"Added Output:Variable entries to city epJSON")
        print(f"Created city epJSON with DDY: {city_epjson}")
        return city_epjson

    def autosize_simulation(self, city_epjson: str, weather_file: str, city: str) -> str:
        """Run design day simulation for autosizing"""
        print(f"\n{'='*60}")
        print(f"Running autosizing simulation for {city}")
        print(f"{'='*60}")

        # Create output directory for sizing simulation with new naming (no sim_ prefix)
        sizing_output_dir = self.output_dir / f"sizing_{city}"
        sizing_output_dir.mkdir(parents=True, exist_ok=True)

        # Run design day simulation
        run_simulation(
            epjson_file=city_epjson,
            weather_file=weather_file,
            output_dir=str(sizing_output_dir),
            mode="designday"
        )

        # Find the eio file
        eio_file = sizing_output_dir / "eplusout.eio"
        if not eio_file.exists():
            raise FileNotFoundError(f"EIO file not found: {eio_file}")

        print(f"Sizing simulation completed. EIO file: {eio_file}")
        return str(eio_file)

    def freeze_sizes(self, eio_file: str, original_epjson: str, city: str) -> str:
        """Freeze component sizes in epJSON"""
        print(f"\n{'='*60}")
        print(f"Freezing component sizes for {city}")
        print(f"{'='*60}")

        # Create city-specific directory in epjson folder
        city_epjson_dir = self.epjson_dir / city
        city_epjson_dir.mkdir(exist_ok=True)

        # Create frozen epJSON filename with frozen suffix
        frozen_name = f"ASHRAE901_OfficeMedium_STD2019_{city}_frozen.epJSON"
        frozen_path = city_epjson_dir / frozen_name

        # Use SizeFreeze to extract sizes and create frozen epJSON
        size_freezer = SizeFreeze(eio_file, original_epjson)
        size_freezer.extract_component_sizing_from_eio()
        size_freezer.freeze_sizes_in_epjson(str(frozen_path))

        print(f"Created frozen epJSON: {frozen_path}")
        return str(frozen_path)

    def climate_scenarios_simulation(self, frozen_epjson: str, weather_files: Dict[str, str], city: str) -> Dict[str, str]:
        """Run simulations with frozen epJSON for all climate scenarios"""
        print(f"\n{'='*60}")
        print(f"Running climate scenario simulations for {city}")
        print(f"{'='*60}")

        simulation_results = {}

        # Run simulation for each available scenario
        for scenario_key, weather_file in weather_files.items():
            if scenario_key in ["ddy"]:  # Skip DDY file
                continue

            print(f"\nRunning simulation for {city} - {scenario_key}")

            # Create scenario-specific output directory with new naming (no sim_ prefix)
            scenario_output_dir = self.output_dir / f"{city}_{scenario_key}"
            scenario_output_dir.mkdir(parents=True, exist_ok=True)

            # Generate fixed weather file for future scenarios (not current TMY)
            weather_file_to_use = weather_file
            if scenario_key.startswith("stmy_"):
                # Create fixed version of the weather file
                weather_file_path = Path(weather_file)
                fixed_weather_file = weather_file_path.parent / f"{weather_file_path.stem}_fixed{weather_file_path.suffix}"

                try:
                    print(f"Fixing weather file: {weather_file}")
                    fix_weather_file(weather_file, str(fixed_weather_file))
                    weather_file_to_use = str(fixed_weather_file)
                    print(f"Using fixed weather file: {weather_file_to_use}")
                except Exception as e:
                    print(f"Warning: Could not fix weather file {weather_file}: {str(e)}")
                    print(f"Using original weather file: {weather_file}")

            try:
                # Run annual simulation with frozen epJSON
                run_simulation(
                    epjson_file=frozen_epjson,
                    weather_file=weather_file_to_use,
                    output_dir=str(scenario_output_dir),
                    mode="annual"
                )

                simulation_results[scenario_key] = str(scenario_output_dir)
                print(f"✓ Completed: {scenario_key}")

            except Exception as e:
                print(f"✗ Failed: {scenario_key} - {str(e)}")
                simulation_results[scenario_key] = f"FAILED: {str(e)}"

        return simulation_results

    def run_city_simulations(self, city: str) -> Dict[str, str]:
        """Run complete simulation pipeline for one city"""
        print(f"\n{'#'*80}")
        print(f"STARTING CLIMATE SIMULATIONS FOR {city.upper()}")
        print(f"{'#'*80}")

        try:
            # Get weather files using direct specification
            weather_files = self.get_weather_files(city)
            print(f"Found weather files for {city}: {list(weather_files.keys())}")

            if "current" not in weather_files:
                raise FileNotFoundError(f"Current TMY file not found for {city}")

            # Get base epJSON using direct specification - already std
            base_epjson = self.get_base_epjson()

            # Create city epJSON with DDY
            city_epjson = self.create_city_epjson_with_ddy(base_epjson, city, weather_files)

            # Run autosizing simulation
            eio_file = self.autosize_simulation(city_epjson, weather_files["current"], city)

            # Freeze sizes
            frozen_epjson = self.freeze_sizes(eio_file, city_epjson, city)

            # Run climate scenario simulations
            simulation_results = self.climate_scenarios_simulation(frozen_epjson, weather_files, city)

            print(f"\n{'#'*80}")
            print(f"COMPLETED SIMULATIONS FOR {city.upper()}")
            print(f"{'#'*80}")

            return simulation_results

        except Exception as e:
            print(f"\n{'!'*80}")
            print(f"ERROR IN SIMULATIONS FOR {city.upper()}: {str(e)}")
            print(f"{'!'*80}")
            return {
                "error": str(e)
            }

    def run_all_cities_simulations(self) -> Dict[str, Dict[str, str]]:
        """Run complete simulation pipeline for all cities"""
        print(f"\n{'='*100}")
        print("STARTING MULTI-CITY CLIMATE SIMULATIONS")
        print(f"{'='*100}")

        all_results = {}

        for city in self.cities.keys():
            simulation_results = self.run_city_simulations(city)
            all_results[city] = simulation_results

        print(f"\n{'='*100}")
        print("SIMULATION SUMMARY")
        print(f"{'='*100}")
        
        for city, results in all_results.items():
            if "error" in results:
                print(f"{city:15}: ERROR - {results['error']}")
            else:
                successful = len([r for r in results.values() if not r.startswith("FAILED:")])
                total = len(results)
                print(f"{city:15}: {successful}/{total} scenarios completed")

        return all_results


def main():
    """Main function to run the climate simulations"""
    # Initialize simulation runner
    runner = ClimateSimulationRunner()

    # Run simulations for all cities
    results = runner.run_all_cities_simulations()

    print(f"\nSimulations completed for {len(results)} cities.")
    return results


if __name__ == "__main__":
    main()
