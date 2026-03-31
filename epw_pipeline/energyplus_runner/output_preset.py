#!/usr/bin/env python3
"""Shared reporting preset for frozen-model stress simulations."""

from __future__ import annotations

from pathlib import Path


REPRESENTATIVE_ZONES = {
    "office": "CORE_BOTTOM",
    "apartment": "G SW APARTMENT",
    "retail": "CORE_RETAIL",
}

SUMMARY_REPORTS = [
    "AnnualBuildingUtilityPerformanceSummary",
    "SystemSummary",
    "EnergyMeters",
]

COMMON_VARIABLES = [
    ("Environment", "Site Outdoor Air Drybulb Temperature"),
    ("Environment", "Site Outdoor Air Relative Humidity"),
    ("Environment", "Site Outdoor Air Wetbulb Temperature"),
    ("*", "Facility Total Purchased Electricity Rate"),
    ("*", "Facility Total Purchased Electricity Energy"),
    ("*", "Facility Total HVAC Electricity Demand Rate"),
    ("*", "Facility Total Building Electricity Demand Rate"),
    ("*", "Facility Total Electricity Demand Rate"),
    ("*", "Facility Heating Setpoint Not Met Time"),
    ("*", "Facility Cooling Setpoint Not Met Time"),
    ("*", "Facility Heating Setpoint Not Met While Occupied Time"),
    ("*", "Facility Cooling Setpoint Not Met While Occupied Time"),
]

ZONE_VARIABLES = [
    "Zone Mean Air Temperature",
    "Zone Air Relative Humidity",
    "Zone Air Humidity Ratio",
    "Zone Mean Radiant Temperature",
    "Zone Operative Temperature",
    "Zone Thermostat Air Temperature",
    "Zone Air System Sensible Heating Rate",
    "Zone Air System Sensible Cooling Rate",
    "Zone Air System Sensible Heating Energy",
    "Zone Air System Sensible Cooling Energy",
]

METER_NAMES = [
    "Electricity:Facility",
    "Electricity:HVAC",
    "Heating:Electricity",
    "Cooling:Electricity",
    "Fans:Electricity",
    "InteriorLights:Electricity",
    "InteriorEquipment:Electricity",
]


def infer_building_key(model_path: Path) -> str:
    """Infer building archetype from a frozen/source model filename."""
    name = model_path.name.lower()
    if name.startswith("office_") or name.startswith("office_medium_"):
        return "office"
    if name.startswith("apartment_") or name.startswith("apartment_midrise_"):
        return "apartment"
    if name.startswith("retail_") or name.startswith("retail_standalone_"):
        return "retail"
    raise ValueError(f"Cannot infer building archetype from filename: {model_path.name}")


def build_output_variables(building_key: str, reporting_frequency: str = "Timestep") -> dict[str, dict[str, str]]:
    """Build the default output-variable map for one archetype."""
    rep_zone = REPRESENTATIVE_ZONES[building_key]
    variables: dict[str, dict[str, str]] = {}

    for idx, (key_value, variable_name) in enumerate(COMMON_VARIABLES, start=1):
        variables[f"CommonOutputVar_{idx}"] = {
            "key_value": key_value,
            "reporting_frequency": reporting_frequency,
            "variable_name": variable_name,
        }

    start = len(variables) + 1
    for offset, variable_name in enumerate(ZONE_VARIABLES, start=start):
        variables[f"RepZoneOutputVar_{offset}"] = {
            "key_value": rep_zone,
            "reporting_frequency": reporting_frequency,
            "variable_name": variable_name,
        }

    return variables


def build_output_meters(reporting_frequency: str = "Timestep") -> dict[str, dict[str, str]]:
    """Build the default meter map."""
    return {
        f"OutputMeter_{idx}": {
            "key_name": meter_name,
            "reporting_frequency": reporting_frequency,
        }
        for idx, meter_name in enumerate(METER_NAMES, start=1)
    }


def apply_reporting_preset(model: dict, building_key: str) -> dict[str, int]:
    """Apply the stress-reporting preset to a model in-place."""
    for key in (
        "Output:Meter:MeterFileOnly",
        "Output:Meter:Cumulative",
        "Output:Meter:Cumulative:MeterFileOnly",
    ):
        model.pop(key, None)

    model["Output:Variable"] = build_output_variables(building_key)
    model["Output:Meter"] = build_output_meters()
    model["Output:SQLite"] = {
        "SQLite_Output": {
            "option_type": "SimpleAndTabular",
        }
    }
    model["Output:Table:SummaryReports"] = {
        "Output:Table:SummaryReports 1": {
            "reports": [{"report_name": report_name} for report_name in SUMMARY_REPORTS]
        }
    }
    model["OutputControl:ReportingTolerances"] = {
        "OutputControl:ReportingTolerances 1": {
            "tolerance_for_time_cooling_setpoint_not_met": 0.556,
            "tolerance_for_time_heating_setpoint_not_met": 0.556,
        }
    }

    return {
        "variables": len(model["Output:Variable"]),
        "meters": len(model["Output:Meter"]),
        "summary_reports": len(SUMMARY_REPORTS),
    }
