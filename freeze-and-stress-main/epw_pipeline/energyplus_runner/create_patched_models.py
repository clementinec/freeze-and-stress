#!/usr/bin/env python3
"""
Create patched source models for new cities (Phoenix, Vancouver).

Takes existing patched models as templates, parses DDY files for new cities,
and replaces site-specific sections (Site:Location, SizingPeriod:DesignDay,
Site:GroundTemperature, Site:WaterMainsTemperature).

Usage:
    python create_patched_models.py
"""

import json
import sys
from pathlib import Path

# ── City-specific data ──────────────────────────────────────────────────────
# Extracted from DDY / STAT files for each new city.

CITY_DATA = {
    "Phoenix": {
        "location": {
            "name": "Phoenix-Sky.Harbor.Intl.AP_AZ_USA WMO=722780",
            "latitude": 33.45,
            "longitude": -111.98,
            "time_zone": -7.0,
            "elevation": 337.0,
        },
        "ground_temps": [14.0, 15.5, 18.0, 22.0, 27.0, 31.0, 33.0, 33.0, 30.0, 25.0, 19.0, 15.0],
        "water_mains": {"annual_avg": 22.0, "max_diff": 12.0},
        "template_city": "Miami",  # hot climate template
    },
    "Vancouver": {
        "location": {
            "name": "Vancouver.Intl.AP_BC_CAN WMO=718920",
            "latitude": 49.18,
            "longitude": -123.17,
            "time_zone": -8.0,
            "elevation": 2.0,
        },
        "ground_temps": [4.0, 4.5, 6.0, 8.5, 11.5, 14.0, 16.0, 16.0, 14.0, 10.5, 7.0, 4.5],
        "water_mains": {"annual_avg": 9.5, "max_diff": 10.0},
        "template_city": "Montreal",  # cold climate template
    },
}

# Building types and their filename patterns
BUILDINGS = {
    "office": "office_medium",
    "apartment": "apartment_midrise",
    "retail": "retail_standalone",
}

MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


def parse_ddy_design_days(ddy_path: Path) -> dict:
    """Parse SizingPeriod:DesignDay objects from a DDY file (text IDF format)."""
    text = ddy_path.read_text(encoding="utf-8", errors="replace")

    design_days = {}
    # Split by object delimiter (semicolons end objects in IDF)
    objects = text.split(";")

    for obj in objects:
        obj = obj.strip()
        if not obj:
            continue

        # Check if this is a SizingPeriod:DesignDay object
        lines = [l.strip() for l in obj.split("\n") if l.strip() and not l.strip().startswith("!")]
        if not lines:
            continue

        # First line should contain the object type
        first_line = lines[0]
        if "SizingPeriod:DesignDay" not in first_line:
            continue

        # Parse comma-separated fields, stripping inline comments
        fields = []
        for line in lines:
            # Remove inline comments (everything after !)
            if "!" in line:
                line = line[:line.index("!")]
            # Split by comma and collect non-empty fields
            parts = [p.strip().rstrip(",") for p in line.split(",")]
            fields.extend([p for p in parts if p])

        if len(fields) < 15:
            continue

        # fields[0] = "SizingPeriod:DesignDay", fields[1] = name, etc.
        dd_name = fields[1]

        dd_obj = {
            "month": int(fields[2]),
            "day_of_month": int(fields[3]),
            "day_type": fields[4],
            "maximum_dry_bulb_temperature": float(fields[5]),
            "daily_dry_bulb_temperature_range": float(fields[6]),
            "dry_bulb_temperature_range_modifier_type": fields[7] if fields[7] else "DefaultMultipliers",
        }

        # PLACEHOLDER_DD_CONTINUE
        # fields[8] = dry_bulb_temp_range_modifier_schedule (usually empty)
        # fields[9] = humidity_condition_type
        humidity_type = fields[9] if len(fields) > 9 and fields[9] else "WetBulb"
        dd_obj["humidity_condition_type"] = humidity_type

        # fields[10] = wetbulb/dewpoint at max dry bulb
        if len(fields) > 10 and fields[10]:
            try:
                dd_obj["wetbulb_or_dewpoint_at_maximum_dry_bulb"] = float(fields[10])
            except ValueError:
                pass

        # fields[11] = humidity_condition_day_schedule (skip)
        # fields[12] = humidity_ratio_at_max_dry_bulb (skip if empty)
        # fields[13] = enthalpy_at_max_dry_bulb (skip if empty)
        # fields[14] = daily_wet_bulb_temperature_range (skip if empty)

        # fields[15] = barometric_pressure
        if len(fields) > 15 and fields[15]:
            try:
                dd_obj["barometric_pressure"] = float(fields[15])
            except ValueError:
                dd_obj["barometric_pressure"] = 101325.0
        else:
            dd_obj["barometric_pressure"] = 101325.0

        # fields[16] = wind_speed
        if len(fields) > 16 and fields[16]:
            try:
                dd_obj["wind_speed"] = float(fields[16])
            except ValueError:
                dd_obj["wind_speed"] = 0.0
        else:
            dd_obj["wind_speed"] = 0.0

        # fields[17] = wind_direction
        if len(fields) > 17 and fields[17]:
            try:
                dd_obj["wind_direction"] = int(float(fields[17]))
            except ValueError:
                dd_obj["wind_direction"] = 0
        else:
            dd_obj["wind_direction"] = 0

        dd_obj["rain_indicator"] = "No"
        dd_obj["snow_indicator"] = "No"
        dd_obj["daylight_saving_time_indicator"] = "No"

        # fields[21] = solar_model_indicator
        solar_model = fields[21] if len(fields) > 21 and fields[21] else None
        if solar_model:
            dd_obj["solar_model_indicator"] = solar_model

            if solar_model == "ASHRAEClearSky":
                if len(fields) > 26 and fields[26]:
                    try:
                        dd_obj["sky_clearness"] = float(fields[26])
                    except ValueError:
                        dd_obj["sky_clearness"] = 0.0
                else:
                    dd_obj["sky_clearness"] = 0.0

            elif solar_model == "ASHRAETau" or solar_model == "ASHRAETau2017":
                if len(fields) > 24 and fields[24]:
                    try:
                        dd_obj["ashrae_clear_sky_optical_depth_for_beam_irradiance_taub_"] = float(fields[24])
                    except ValueError:
                        pass
                if len(fields) > 25 and fields[25]:
                    try:
                        dd_obj["ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud_"] = float(fields[25])
                    except ValueError:
                        pass
        else:
            if "WinterDesignDay" in dd_obj.get("day_type", ""):
                dd_obj["solar_model_indicator"] = "ASHRAEClearSky"
                dd_obj["sky_clearness"] = 0.0
            else:
                dd_obj["solar_model_indicator"] = "ASHRAETau"

        design_days[dd_name] = dd_obj

    return design_days


def parse_ddy_location(ddy_path: Path) -> dict:
    """Parse Site:Location from a DDY file."""
    text = ddy_path.read_text(encoding="utf-8", errors="replace")
    objects = text.split(";")

    for obj in objects:
        obj = obj.strip()
        if "Site:Location" not in obj or "SizingPeriod" in obj:
            continue

        lines = [l.strip() for l in obj.split("\n") if l.strip() and not l.strip().startswith("!")]
        fields = []
        for line in lines:
            if "!" in line:
                line = line[:line.index("!")]
            parts = [p.strip().rstrip(",") for p in line.split(",")]
            fields.extend([p for p in parts if p])

        if len(fields) >= 5 and "Site:Location" in fields[0]:
            return {
                "name": fields[1],
                "latitude": float(fields[2]),
                "longitude": float(fields[3]),
                "time_zone": float(fields[4]),
                "elevation": float(fields[5]) if len(fields) > 5 else 0.0,
            }

    return None


def create_patched_model(
    template_path: Path,
    city_key: str,
    city_data: dict,
    design_days: dict,
    location_data: dict,
    output_path: Path,
):
    """Create a patched model by replacing site-specific data."""
    with open(template_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    # Replace Site:Location
    loc = location_data or city_data["location"]
    model["Site:Location"] = {
        loc["name"]: {
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "time_zone": loc["time_zone"],
            "elevation": loc["elevation"],
        }
    }

    # Replace SizingPeriod:DesignDay
    if design_days:
        model["SizingPeriod:DesignDay"] = design_days

    # Replace ground temperatures
    temps = city_data["ground_temps"]
    model["Site:GroundTemperature:FCfactorMethod"] = {
        "Site:GroundTemperature:FCfactorMethod 1": {
            f"{MONTH_NAMES[i]}_ground_temperature": temps[i]
            for i in range(12)
        }
    }

    # Replace water mains temperature
    wm = city_data["water_mains"]
    model["Site:WaterMainsTemperature"] = {
        "Site:WaterMainsTemperature 1": {
            "calculation_method": "Correlation",
            "annual_average_outdoor_air_temperature": wm["annual_avg"],
            "maximum_difference_in_monthly_average_outdoor_air_temperatures": wm["max_diff"],
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    print(f"  Created: {output_path.name}")


def main():
    script_dir = Path(__file__).resolve().parent
    source_dir = script_dir / "source_models"
    ddy_dir = Path("E:/00_research/03_freezethenstress/ep-sim-main/data/epw")

    ddy_files = {
        "Phoenix": ddy_dir / "USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3" / "USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.ddy",
        "Vancouver": ddy_dir / "CAN_BC_Vancouver.Intl.AP.718920_CWEC" / "CAN_BC_Vancouver.718920_CWEC.ddy",
    }

    print("=" * 60)
    print("Creating patched source models for Phoenix & Vancouver")
    print("=" * 60)

    for city_key, city_data in CITY_DATA.items():
        print(f"\n--- {city_key} ---")

        # Parse DDY file
        ddy_path = ddy_files[city_key]
        if not ddy_path.exists():
            print(f"  ERROR: DDY file not found: {ddy_path}")
            continue

        design_days = parse_ddy_design_days(ddy_path)
        location_data = parse_ddy_location(ddy_path)
        print(f"  Parsed {len(design_days)} design days from DDY")

        if location_data:
            print(f"  Location: {location_data['name']} ({location_data['latitude']}, {location_data['longitude']})")

        template_city = city_data["template_city"]

        for bldg_key, bldg_prefix in BUILDINGS.items():
            template_path = source_dir / f"{bldg_prefix}_{template_city}_patched.epJSON"
            output_path = source_dir / f"{bldg_prefix}_{city_key}_patched.epJSON"

            if not template_path.exists():
                print(f"  ERROR: Template not found: {template_path}")
                continue

            create_patched_model(
                template_path, city_key, city_data,
                design_days, location_data, output_path,
            )

    print(f"\nDone. Now run:")
    print(f"  python freeze_model.py --local-eplus <path-to-energyplus> --cities Phoenix,Vancouver")


if __name__ == "__main__":
    main()
