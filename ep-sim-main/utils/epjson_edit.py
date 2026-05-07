#!/usr/bin/env python3
"""
Test eppy editing epJSON file functionality
- Read original epJSON file
- Change one Autosize parameter to fixed value
- Compare before and after
"""

import json
import os
from eppy.modeleditor import IDF

def read_epjson(file_path):
    """Read epJSON file as dictionary"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_epjson(data, file_path):
    """Save dictionary as epJSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def compare_values(original_file, modified_file, object_type, object_name, field_name):
    """Compare specific field values between two epJSON files"""
    original_data = read_epjson(original_file)
    modified_data = read_epjson(modified_file)

    # Get values from both files
    original_value = original_data.get(object_type, {}).get(object_name, {}).get(field_name, "NOT_FOUND")
    modified_value = modified_data.get(object_type, {}).get(object_name, {}).get(field_name, "NOT_FOUND")

    print(f"\nComparison for {object_type} '{object_name}' field '{field_name}':")
    print(f"  Original: {original_value}")
    print(f"  Modified: {modified_value}")
    print(f"  Changed: {'Yes' if original_value != modified_value else 'No'}")

def epjson_editing(original_file, modified_file):
    """Test editing epJSON file directly with JSON"""
    print("Testing epJSON editing...")

    # Check if original file exists
    if not os.path.exists(original_file):
        print(f"Error: Original file not found: {original_file}")
        return

    try:
        # Read original epJSON file
        print("Reading original epJSON file...")
        epjson_data = read_epjson(original_file)

        # Modify AirLoopHVAC design supply air flow rate from Autosize to fixed value
        print("Modifying AirLoopHVAC design supply air flow rate...")

        if "AirLoopHVAC" in epjson_data and "VAV Sys 1" in epjson_data["AirLoopHVAC"]:
            air_loop = epjson_data["AirLoopHVAC"]["VAV Sys 1"]
            original_value = air_loop.get("design_supply_air_flow_rate", "NOT_FOUND")
            print(f"  System: VAV Sys 1")
            print(f"  Original value: {original_value}")

            # Change Autosize to fixed value (5.0 m3/s)
            air_loop["design_supply_air_flow_rate"] = 5.0
            print(f"  New value: {air_loop['design_supply_air_flow_rate']}")
        else:
            print("  Error: AirLoopHVAC 'VAV Sys 1' not found")
            return

        # Save modified file
        print(f"Saving modified file: {modified_file}")
        save_epjson(epjson_data, modified_file)

        # Check if file was saved successfully
        if os.path.exists(modified_file):
            file_size = os.path.getsize(modified_file)
            print(f"File saved successfully! Size: {file_size} bytes")

            # Compare the specific value that was changed
            compare_values(original_file, modified_file,
                         "AirLoopHVAC", "VAV Sys 1", "design_supply_air_flow_rate")

            print("\nTest completed successfully!")
        else:
            print("Error: File save failed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    epjson_editing(
        original_file="data/epjson/ASHRAE901_OfficeSmall_STD2019.epJSON",
        modified_file="data/epjson/ASHRAE901_OfficeSmall_STD2019_edited.epJSON"
    )
