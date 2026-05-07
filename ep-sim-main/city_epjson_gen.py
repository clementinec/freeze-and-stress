#!/usr/bin/env python3
"""
City-specific epJSON file generator.
Reads Site:Location and SizingPeriod:DesignDay information from a DDY file,
merges it with a cleaned base epJSON file, and generates a city-specific epJSON file.
"""

import json
import os
from typing import Dict, Any
from eppy.modeleditor import IDF


def generate_output_variables(epjson_data):
    """
    Generate Output:Variable entries for an EPJSON file.
    Includes:
      - Site weather variables
      - Zone comfort and occupancy
      - HVAC system node variables
      - Energy meters and CO2 emissions
    """

    output_variables = {}

    # Site weather variables
    site_vars = {
        "OutdoorAirDrybulbTemp": ("Site Outdoor Air Drybulb Temperature", "Outdoor air drybulb temperature"),
        "OutdoorAirRH": ("Site Outdoor Air Relative Humidity", "Outdoor relative humidity"),
        "WindSpeed": ("Site Wind Speed", "Outdoor wind speed"),
        "WindDirection": ("Site Wind Direction", "Outdoor wind direction"),
        "DiffuseSolar": ("Site Diffuse Solar Radiation Rate per Area", "Diffuse solar radiation"),
        "DirectSolar": ("Site Direct Solar Radiation Rate per Area", "Direct solar radiation"),
    }

    for i, (name, (var_name, comment)) in enumerate(site_vars.items()):
        key = f"Output:Variable_Site_{i}_{name}"
        output_variables[key] = {
            "variable_name": var_name,
            "key_value": "Environment",  # use Environment instead of *
            "reporting_frequency": "Timestep",
            "comment": comment
        }

    # Zone variables
    # Use Zone objects from EPJSON
    zone_names = list(epjson_data.get("Zone", {}).keys())
    zone_vars = {
        "OccupantCount": ("Zone People Occupant Count", "Number of people in the zone"),
        "AirTemp": ("Zone Air Temperature", "Zone air temperature"),
        "AirRH": ("Zone Air Relative Humidity", "Zone air relative humidity"),
        "HeatingSP": ("Zone Thermostat Heating Setpoint Temperature", "Heating setpoint temperature"),
        "CoolingSP": ("Zone Thermostat Cooling Setpoint Temperature", "Cooling setpoint temperature")
    }

    for zone in zone_names:
        for name, (var_name, comment) in zone_vars.items():
            key = f"Output:Variable_Zone_{zone}_{name}"
            output_variables[key] = {
                "variable_name": var_name,
                "key_value": zone,
                "reporting_frequency": "Timestep",
                "comment": comment
            }

    # Building energy and CO2
    building_vars = {
        "TotalHVACPower": ("Facility Total HVAC Electricity Demand Rate", "Total HVAC electricity demand"),
        "CO2Emissions": ("Environmental Impact Total CO2 Emissions Carbon Equivalent Mass", "Total CO2 emissions"),
        "ElectricityHVAC": ("Electricity:HVAC", "HVAC electricity consumption"),
        "CoolingElec": ("Cooling:Electricity", "Cooling electricity consumption"),
        "HeatingElec": ("Heating:Electricity", "Heating electricity consumption"),
        "LightsElec": ("InteriorLights:Electricity", "Interior lighting electricity consumption")
    }

    for i, (name, (var_name, comment)) in enumerate(building_vars.items()):
        key = f"Output:Variable_Building_{i}_{name}"
        output_variables[key] = {
            "variable_name": var_name,
            "key_value": "*",
            "reporting_frequency": "Timestep",
            "comment": comment
        }

    # Merge into EPJSON
    epjson_data["Output:Variable"] = output_variables
    return epjson_data


def parse_ddy_file(ddy_file_path: str, idd_file_path: str = "D:\\EnergyPlusV25-1-0\\Energy+.idd") -> Dict[str, Any]:
    """
    Parse DDY file to extract Site:Location and SizingPeriod:DesignDay information using eppy.

    Args:
        ddy_file_path: Path to the DDY file.
        idd_file_path: Path to the EnergyPlus IDD file.

    Returns:
        Dictionary containing location and design day information.
    """
    result = {
        "Site:Location": {},
        "SizingPeriod:DesignDay": {}
    }

    # Set IDD file
    IDF.setiddname(idd_file_path)

    # Read DDY file
    idf_ddy = IDF(ddy_file_path)

    # Extract Site:Location objects
    locations = idf_ddy.idfobjects["Site:Location"]
    for location in locations:
        location_name = location.Name
        result["Site:Location"][location_name] = {
            "latitude": float(location.Latitude),
            "longitude": float(location.Longitude),
            "time_zone": float(location.Time_Zone),
            "elevation": float(location.Elevation)
        }

    # Extract SizingPeriod:DesignDay objects
    design_days = idf_ddy.idfobjects["SizingPeriod:DesignDay"]
    for dd in design_days:
        design_day_name = dd.Name

        # Convert design day to epJSON format
        design_day_obj = {
            "month": int(dd.Month),
            "day_of_month": int(dd.Day_of_Month),
            "day_type": dd.Day_Type,
            "maximum_dry_bulb_temperature": float(dd.Maximum_DryBulb_Temperature),
            "daily_dry_bulb_temperature_range": float(dd.Daily_DryBulb_Temperature_Range),
            "dry_bulb_temperature_range_modifier_type": dd.DryBulb_Temperature_Range_Modifier_Type or "DefaultMultipliers",
            "humidity_condition_type": "WetBulb",  # Use correct EnergyPlus enum value
            "rain_indicator": "No",
            "snow_indicator": "No",
            "daylight_saving_time_indicator": "No"
        }

        # Handle optional fields with safer attribute access
        wetbulb_field = getattr(dd, 'Wetbulb_or_DewPoint_at_Maximum_DryBulb', None)
        if wetbulb_field:
            design_day_obj["wetbulb_or_dewpoint_at_maximum_dry_bulb"] = float(wetbulb_field)
        else:
            design_day_obj["wetbulb_or_dewpoint_at_maximum_dry_bulb"] = float(dd.Maximum_DryBulb_Temperature)

        barometric_field = getattr(dd, 'Barometric_Pressure', None)
        if barometric_field:
            design_day_obj["barometric_pressure"] = float(barometric_field)
        else:
            design_day_obj["barometric_pressure"] = 101325.0

        wind_speed_field = getattr(dd, 'Wind_Speed', None)
        if wind_speed_field:
            design_day_obj["wind_speed"] = float(wind_speed_field)
        else:
            design_day_obj["wind_speed"] = 0.0

        wind_dir_field = getattr(dd, 'Wind_Direction', None)
        if wind_dir_field:
            design_day_obj["wind_direction"] = int(float(wind_dir_field))
        else:
            design_day_obj["wind_direction"] = 0

        # Solar model specific fields - use correct EnergyPlus enum values
        solar_model = getattr(dd, 'Solar_Model_Indicator', None)
        if solar_model:
            design_day_obj["solar_model_indicator"] = solar_model

            if solar_model == "ASHRAEClearSky":
                sky_clearness = getattr(dd, 'Sky_Clearness', None)
                if sky_clearness:
                    design_day_obj["sky_clearness"] = float(sky_clearness)
                else:
                    design_day_obj["sky_clearness"] = 0.0

            elif solar_model == "ASHRAETau":
                taub = getattr(dd, 'ASHRAE_Clear_Sky_Optical_Depth_for_Beam_Irradiance_taub', None)
                if taub:
                    design_day_obj["ashrae_clear_sky_optical_depth_for_beam_irradiance_taub_"] = float(taub)
                else:
                    design_day_obj["ashrae_clear_sky_optical_depth_for_beam_irradiance_taub_"] = 0.344

                taud = getattr(dd, 'ASHRAE_Clear_Sky_Optical_Depth_for_Diffuse_Irradiance_taud', None)
                if taud:
                    design_day_obj["ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud_"] = float(taud)
                else:
                    design_day_obj["ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud_"] = 2.302
        else:
            # Default solar model based on design day type
            if "WinterDesignDay" in design_day_obj["day_type"]:
                design_day_obj["solar_model_indicator"] = "ASHRAEClearSky"
                design_day_obj["sky_clearness"] = 0.0
            else:  # SummerDesignDay
                design_day_obj["solar_model_indicator"] = "ASHRAETau"
                design_day_obj["ashrae_clear_sky_optical_depth_for_beam_irradiance_taub_"] = 0.344
                design_day_obj["ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud_"] = 2.302

        result["SizingPeriod:DesignDay"][design_day_name] = design_day_obj

    return result


def merge_epjson_with_ddy(base_epjson_path: str, ddy_data: Dict[str, Any], city_name: str = None) -> Dict[str, Any]:
    """
    Merge DDY data into the base epJSON file.

    Args:
        base_epjson_path: Path to the base epJSON file.
        ddy_data: Data parsed from the DDY file.
        city_name: City name for selecting appropriate ground temperatures.

    Returns:
        Merged epJSON data.
    """
    with open(base_epjson_path, 'r', encoding='utf-8') as f:
        epjson_data = json.load(f)

    # Merge Site:Location
    if "Site:Location" in ddy_data and ddy_data["Site:Location"]:
        epjson_data["Site:Location"] = ddy_data["Site:Location"]

    # Merge SizingPeriod:DesignDay
    if "SizingPeriod:DesignDay" in ddy_data and ddy_data["SizingPeriod:DesignDay"]:
        epjson_data["SizingPeriod:DesignDay"] = ddy_data["SizingPeriod:DesignDay"]

    # City-specific ground temperatures (January to December)
    ground_temperature_by_city = {
        "Los_Angeles": [15.0, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.0, 20.0, 19.0, 17.0, 15.5],
        "Miami":       [22.0, 22.5, 23.0, 24.0, 25.0, 26.0, 26.5, 27.0, 26.5, 26.0, 24.5, 23.5],
        "Toronto":     [3.0, 3.5, 5.0, 8.0, 12.0, 15.0, 17.0, 17.5, 15.0, 11.0, 7.0, 4.0],
        "Montreal":    [0.0, 0.5, 2.5, 6.0, 10.0, 13.5, 15.5, 16.0, 13.0, 9.0, 4.5, 1.5],
        "Phoenix":     [14.0, 15.5, 18.0, 22.0, 27.0, 31.0, 33.0, 33.0, 30.0, 25.0, 19.0, 15.0],
        "Vancouver":   [4.0, 4.5, 6.0, 8.5, 11.5, 14.0, 16.0, 16.0, 14.0, 10.5, 7.0, 4.5]
    }

    # Add city-specific ground temperatures if not present or empty
    if "Site:GroundTemperature:FCfactorMethod" not in epjson_data or not epjson_data["Site:GroundTemperature:FCfactorMethod"]:
        # Get city-specific ground temperatures, or use default if city not found
        if city_name and city_name in ground_temperature_by_city:
            temps = ground_temperature_by_city[city_name]
            print(f"Using city-specific ground temperatures for {city_name}")
        else:
            # Fallback to moderate climate default
            temps = [15.0, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.0, 20.0, 19.0, 17.0, 15.5]
            print(f"Using default ground temperatures (city: {city_name} not found in predefined list)")

        epjson_data["Site:GroundTemperature:FCfactorMethod"] = {
            f"{city_name or 'Default'} Ground Temperatures": {
                "january_ground_temperature": temps[0],
                "february_ground_temperature": temps[1],
                "march_ground_temperature": temps[2],
                "april_ground_temperature": temps[3],
                "may_ground_temperature": temps[4],
                "june_ground_temperature": temps[5],
                "july_ground_temperature": temps[6],
                "august_ground_temperature": temps[7],
                "september_ground_temperature": temps[8],
                "october_ground_temperature": temps[9],
                "november_ground_temperature": temps[10],
                "december_ground_temperature": temps[11]
            }
        }
        print(f"Added ground temperatures: Jan={temps[0]}°C, Jul={temps[6]}°C")

    return epjson_data


def generate_city_epjson(base_epjson_path: str, ddy_file_path: str, output_path: str, city_name: str = None):
    """
    Generate city-specific epJSON file.

    Args:
        base_epjson_path: Path to the base epJSON file (location info cleaned).
        ddy_file_path: Path to the DDY file.
        output_path: Output epJSON file path.
        city_name: City name (for file naming).
    """
    print(f"Processing DDY file: {ddy_file_path}")

    # Parse DDY file
    ddy_data = parse_ddy_file(ddy_file_path)

    print(f"Parsed {len(ddy_data['Site:Location'])} location(s)")
    print(f"Parsed {len(ddy_data['SizingPeriod:DesignDay'])} design day(s)")

    # Merge data
    merged_data = merge_epjson_with_ddy(base_epjson_path, ddy_data, city_name)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save result
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated city-specific epJSON file: {output_path}")

    # Print location summary
    for location_name, location_data in ddy_data["Site:Location"].items():
        print(f"\nLocation: {location_name}")
        print(f"  Latitude: {location_data['latitude']}°")
        print(f"  Longitude: {location_data['longitude']}°")
        print(f"  Time zone: GMT{location_data['time_zone']:+.1f}")
        print(f"  Elevation: {location_data['elevation']} m")

    # Print design day summary
    # for design_day_name, design_day_data in ddy_data["SizingPeriod:DesignDay"].items():
    #     print(f"\nDesign Day: {design_day_name}")
    #     print(f"  Type: {design_day_data['day_type']}")
    #     print(f"  Temperature: {design_day_data['maximum_dry_bulb_temperature']}°C")
    #     print(f"  Month: {design_day_data['month']}")

    # Return the output file path
    return output_path


if __name__ == "__main__":
    # Define cities and their corresponding data
    cities_config = {
        "Los_Angeles": {
            "ddy": "data/epw/USA_CA_Los.Angeles.Intl.AP.722950_TMY3/USA_CA_Los.Angeles.Intl.AP.722950_TMY3.ddy",
            "output_dir": "data/epjson/Los_Angeles",
            "display_name": "Los Angeles"
        },
        "Miami": {
            "ddy": "data/epw/USA_FL_Miami.Intl.AP.722020_TMY3/USA_FL_Miami.Intl.AP.722020_TMY3.ddy",
            "output_dir": "data/epjson/Miami",
            "display_name": "Miami"
        },
        "Toronto": {
            "ddy": "data/epw/CAN_ON_Toronto.716240_CWEC/CAN_ON_Toronto.716240_CWEC.ddy",
            "output_dir": "data/epjson/Toronto",
            "display_name": "Toronto"
        },
        "Montreal": {
            "ddy": "data/epw/CAN_PQ_Montreal.Intl.AP.716270_CWEC/CAN_PQ_Montreal.Intl.AP.716270_CWEC.ddy",
            "output_dir": "data/epjson/Montreal",
            "display_name": "Montreal"
        },
        "Phoenix": {
            "ddy": "data/epw/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.ddy",
            "output_dir": "data/epjson/Phoenix",
            "display_name": "Phoenix"
        },
        "Vancouver": {
            "ddy": "data/epw/CAN_BC_Vancouver.Intl.AP.718920_CWEC/CAN_BC_Vancouver.718920_CWEC.ddy",
            "output_dir": "data/epjson/Vancouver",
            "display_name": "Vancouver"
        }
    }

    # Base epJSON files
    base_epjson = "data/epjson/ASHRAE901_OfficeMedium_STD2019_std.epJSON"
    update_epjson = "data/epjson/ASHRAE901_OfficeMedium_STD2019_std_update.epJSON"

    print("=" * 80)
    print("Generating city-specific epJSON files for all cities")
    print("=" * 80)

    # First, create the update_epjson with Output:Variable entries if it doesn't exist
    if not os.path.exists(update_epjson):
        print("Creating base epJSON with Output:Variable entries...")
        with open(base_epjson, 'r', encoding='utf-8') as f:
            base_epjson_data = json.load(f)
        update_epjson_data = generate_output_variables(base_epjson_data)

        with open(update_epjson, "w") as f:
            json.dump(update_epjson_data, f, indent=2)
        print(f"Created: {update_epjson}")

    # Process each city
    total_files_generated = 0
    for city_key, config in cities_config.items():
        print("\n" + "=" * 60)
        print(f"Processing {config['display_name']}")
        print("=" * 60)

        # Define output file paths
        city_epjson_clean = f"{config['output_dir']}/ASHRAE901_OfficeMedium_STD2019_{city_key}_wo_output.epJSON"
        city_epjson_full = f"{config['output_dir']}/ASHRAE901_OfficeMedium_STD2019_{city_key}.epJSON"

        # Check if DDY file exists
        if not os.path.exists(config['ddy']):
            print(f"⚠️  DDY file not found: {config['ddy']}")
            print(f"   Skipping {config['display_name']}")
            continue

        try:
            # Generate clean version (without Output:Variable entries)
            print(f"\n1. Generating clean version for {config['display_name']}...")
            generate_city_epjson(
                base_epjson_path=base_epjson,
                ddy_file_path=config['ddy'],
                output_path=city_epjson_clean,
                city_name=config['display_name']
            )
            print(f"✅ Clean version saved: {city_epjson_clean}")
            total_files_generated += 1

            # Generate full version (with Output:Variable entries)
            print(f"\n2. Generating full version for {config['display_name']}...")
            generate_city_epjson(
                base_epjson_path=update_epjson,
                ddy_file_path=config['ddy'],
                output_path=city_epjson_full,
                city_name=config['display_name']
            )
            print(f"✅ Full version saved: {city_epjson_full}")
            total_files_generated += 1

            print(f"\n📁 {config['display_name']} files:")
            print(f"   - Clean: {city_epjson_clean}")
            print(f"   - Full:  {city_epjson_full}")

        except Exception as e:
            print(f"❌ Error processing {config['display_name']}: {str(e)}")
            continue

    # Final summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total files generated: {total_files_generated}")
    print("\nGenerated files by city:")

    for city_key, config in cities_config.items():
        city_epjson_clean = f"{config['output_dir']}/ASHRAE901_OfficeMedium_STD2019_{city_key}_wo_output.epJSON"
        city_epjson_full = f"{config['output_dir']}/ASHRAE901_OfficeMedium_STD2019_{city_key}.epJSON"

        print(f"\n{config['display_name']}:")
        if os.path.exists(city_epjson_clean):
            print(f"  ✅ Clean: {city_epjson_clean}")
        else:
            print(f"  ❌ Clean: {city_epjson_clean}")

        if os.path.exists(city_epjson_full):
            print(f"  ✅ Full:  {city_epjson_full}")
        else:
            print(f"  ❌ Full:  {city_epjson_full}")

    print("=" * 80)
