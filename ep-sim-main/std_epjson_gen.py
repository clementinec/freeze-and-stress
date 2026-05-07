#!/usr/bin/env python3
"""
Standard epJSON generator.
Removes all site-specific parameters from an ASHRAE standard epJSON file
to create a standardized template that can be used with city_epjson_gen.py.

This script removes the following site-specific parameters:
- Site:Location
- Site:GroundTemperature:FCfactorMethod
- Site:WaterMainsTemperature
- SizingPeriod:DesignDay
"""

import json
import os
from typing import Dict, Any, Tuple, List

def remove_site_specific_parameters(epjson_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    """
    Remove all site-specific parameters from the epJSON data.

    Args:
        epjson_data: The original epJSON data dictionary

    Returns:
        Tuple: (cleaned_data, removed_parameters) - Cleaned epJSON data and list of removed parameters
    """
    # Create a deep copy to avoid modifying the original data
    cleaned_data = json.loads(json.dumps(epjson_data))

    # List of site-specific parameter categories to clear
    site_specific_categories = [
        "Site:Location",
        "Site:GroundTemperature:FCfactorMethod",
        "Site:WaterMainsTemperature",
        "SizingPeriod:DesignDay"
    ]

    removed_parameters = []

    # Clear each category of site-specific parameters but keep the structure
    for category in site_specific_categories:
        if category in cleaned_data:
            removed_items = list(cleaned_data[category].keys())
            cleaned_data[category] = {}  # Set to empty dict instead of deleting
            removed_parameters.extend([(category, item) for item in removed_items])
            print(f"Cleared {category} with {len(removed_items)} items")

    return cleaned_data, removed_parameters

def generate_standard_epjson(input_file: str, output_file: str = None) -> str:
    """
    Generate a standardized epJSON file by removing site-specific parameters.

    Args:
        input_file: Path to the input ASHRAE epJSON file
        output_file: Path to the output standardized epJSON file (optional)

    Returns:
        str: Path to the generated output file
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Loading epJSON file: {input_file}")

    # Load the original epJSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            epjson_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in input file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading input file: {e}")

    print(f"Original epJSON contains {len(epjson_data)} top-level categories")

    # Remove site-specific parameters
    cleaned_data, removed_parameters = remove_site_specific_parameters(epjson_data)

    print(f"Standardized epJSON contains {len(cleaned_data)} top-level categories")
    print(f"Removed {len(removed_parameters)} site-specific parameters:")
    for category, item in removed_parameters:
        print(f"  - {category}: {item}")

    # Generate output filename if not provided
    if output_file is None:
        name, ext = os.path.splitext(input_file)
        if '_' in name:
            # Split by underscore and replace the last part (city) with "std"
            parts = name.split('_')
            if len(parts) > 1:
                parts[-1] = 'std'
                output_file = '_'.join(parts) + ext
            else:
                output_file = f"{name}_std{ext}"
        else:
            output_file = f"{name}_std{ext}"

    print(f"Output file: {output_file}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the standardized epJSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2)
        print(f"Standardized epJSON saved to: {output_file}")
    except Exception as e:
        raise RuntimeError(f"Error writing output file: {e}")

    return output_file

# Direct usage example
if __name__ == "__main__":
    input_file = "data/epjson/Miami/ASHRAE901_OfficeMedium_STD2019_Miami.epJSON"
    output_file = generate_standard_epjson(input_file)
    print(f"Generated standard epJSON: {output_file}")
