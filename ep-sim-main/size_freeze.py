#!/usr/bin/env python3
"""
Size Freezing Script
Extract sizing data from Design Day simulation eio file
and replace Autosize parameters in epJSON file for annual simulation
"""

import json
import os
import re


class SizeFreeze:
    """Extract component sizes from EnergyPlus eio output and freeze them in epJSON"""

    # Mapping from eio parameter names to epJSON field names
    PARAMETER_MAPPING = {
        # Controller:OutdoorAir
        'Maximum Outdoor Air Flow Rate [m3/s]': 'maximum_outdoor_air_flow_rate',
        'Minimum Outdoor Air Flow Rate [m3/s]': 'minimum_outdoor_air_flow_rate',

        # AirLoopHVAC
        'Design Supply Air Flow Rate [m3/s]': 'design_supply_air_flow_rate',

        # Fan:OnOff, Fan:ConstantVolume, Fan:VariableVolume
        'Design Size maximum_flow_rate [m3/s]': 'maximum_flow_rate',

        # Coil:Cooling:DX:SingleSpeed
        'Design Size rated_air_flow_rate [m3/s]': 'rated_air_flow_rate',
        'Design Size gross_rated_total_cooling_capacity [W]': 'gross_rated_total_cooling_capacity',
        'Design Size gross_rated_sensible_heat_ratio': 'gross_rated_sensible_heat_ratio',

        # Coil:Heating:DX:SingleSpeed
        'Design Size gross_rated_heating_capacity [W]': 'gross_rated_heating_capacity',

        # Coil:Heating:Fuel
        'Design Size nominal_capacity [W]': 'nominal_capacity',

        # AirTerminal:SingleDuct:ConstantVolume:NoReheat
        'Design Size Maximum Air Flow Rate [m3/s]': 'maximum_air_flow_rate',

        # AirTerminal:SingleDuct:VAV:Reheat
        'Design Size maximum_air_flow_rate [m3/s]': 'maximum_air_flow_rate',

        # Coil:Cooling:DX:TwoSpeed
        'Design Size high_speed_rated_air_flow_rate [m3/s]': 'high_speed_rated_air_flow_rate',
        'Design Size high_speed_gross_rated_total_cooling_capacity [W]': 'high_speed_gross_rated_total_cooling_capacity',
        'Design Size low_speed_rated_air_flow_rate [m3/s]': 'low_speed_rated_air_flow_rate',
        'Design Size low_speed_gross_rated_total_cooling_capacity [W]': 'low_speed_gross_rated_total_cooling_capacity',

        # AirLoopHVAC:UnitaryHeatPump:AirToAir
        'Supply Air Flow Rate [m3/s]': 'supply_air_flow_rate',
        'Supply Air Flow Rate During Cooling Operation [m3/s]': 'cooling_supply_air_flow_rate',
        'Supply Air Flow Rate During Heating Operation [m3/s]': 'heating_supply_air_flow_rate',
        'Supply Air Flow Rate When No Cooling or Heating is Needed [m3/s]': 'no_load_supply_air_flow_rate',
        'Nominal Heating Capacity [W]': 'heating_coil_capacity',
        'Nominal Cooling Capacity [W]': 'cooling_coil_capacity',
        'Maximum Supply Air Temperature from Supplemental Heater [C]': 'maximum_supply_air_temperature_from_supplemental_heater',
        'Supplemental Heating Coil Nominal Capacity [W]': 'supplemental_heating_coil_capacity',
    }

    def __init__(self, eio_file_path, original_epjson_path):
        self.eio_file_path = eio_file_path
        self.original_epjson_path = original_epjson_path
        self.eio_sizing_data = {}

    def extract_component_sizing_from_eio(self):
        """Extract all Component Sizing Information from eio file"""
        print(f"Extracting Component Sizing Information from: {self.eio_file_path}")

        if not os.path.exists(self.eio_file_path):
            print(f"Error: eio file not found: {self.eio_file_path}")
            return

        with open(self.eio_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('Component Sizing Information'):
                    # Parse line format:
                    # Component Sizing Information, ComponentType, ComponentName, ParameterName, Value
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        component_type = parts[1]
                        component_name = parts[2]
                        parameter_name = parts[3]
                        value = parts[4]

                        # Store in nested dictionary
                        if component_type not in self.eio_sizing_data:
                            self.eio_sizing_data[component_type] = {}
                        if component_name not in self.eio_sizing_data[component_type]:
                            self.eio_sizing_data[component_type][component_name] = {}

                        self.eio_sizing_data[component_type][component_name][parameter_name] = value

        print(f"Extracted sizing data for {len(self.eio_sizing_data)} component types")
        for comp_type, components in self.eio_sizing_data.items():
            print(f"  {comp_type}: {len(components)} components")

    def freeze_sizes_in_epjson(self, output_epjson_path):
        """Replace Autosize with fixed values in epJSON file based on eio data"""
        print(f"\nFreezing sizes in epJSON...")

        # Read original epJSON
        with open(self.original_epjson_path, 'r', encoding='utf-8') as f:
            epjson_data = json.load(f)

        modified_count = 0

        # Process each component type from eio data
        for component_type in self.eio_sizing_data:
            if component_type not in epjson_data:
                print(f"\nSkipping {component_type} - not found in epJSON")
                continue

            # print(f"\nProcessing {component_type}...")
            modified_count += self._freeze_component_sizes(
                epjson_data, component_type
            )

        # Save modified epJSON
        with open(output_epjson_path, 'w', encoding='utf-8') as f:
            json.dump(epjson_data, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Total modified parameters: {modified_count}")
        print(f"Saved frozen epJSON to: {output_epjson_path}")
        print(f"{'='*80}")

    def _freeze_component_sizes(self, epjson_data, component_type):
        """Freeze sizes for a specific component type"""
        modified_count = 0

        for eio_component_name, sizing_params in self.eio_sizing_data[component_type].items():
            # Find matching component in epJSON
            epjson_name = self._find_matching_component(
                epjson_data[component_type].keys(),
                eio_component_name
            )

            if not epjson_name:
                print(f"  Warning: No match found for '{eio_component_name}'")
                continue

            # Freeze each parameter
            for eio_param_name, value in sizing_params.items():
                epjson_field_name = self._get_epjson_field_name(eio_param_name)

                if not epjson_field_name:
                    # Skip parameters we don't have mappings for
                    continue

                try:
                    # Try to convert to float
                    float_value = float(value)

                    # Get old value for logging
                    old_value = epjson_data[component_type][epjson_name].get(
                        epjson_field_name, 'NOT_FOUND'
                    )

                    # Set new value
                    epjson_data[component_type][epjson_name][epjson_field_name] = float_value

                    # print(f"  {epjson_name}.{epjson_field_name}: {old_value} → {float_value}")
                    modified_count += 1

                except (ValueError, TypeError) as e:
                    print(f"  Warning: Could not convert '{value}' for {epjson_name}.{epjson_field_name}: {e}")

        return modified_count

    def _find_matching_component(self, epjson_names, eio_name):
        """Find matching component name between epJSON and eio data"""
        # First try exact match
        if eio_name in epjson_names:
            return eio_name

        # Try case-insensitive match without spaces
        eio_normalized = eio_name.replace(' ', '').replace('_', '').upper()

        for epjson_name in epjson_names:
            epjson_normalized = epjson_name.replace(' ', '').replace('_', '').upper()
            if eio_normalized == epjson_normalized:
                return epjson_name

            # Try partial match (eio name contains epjson name or vice versa)
            if eio_normalized in epjson_normalized or epjson_normalized in eio_normalized:
                return epjson_name

        return None

    def _get_epjson_field_name(self, eio_param_name):
        """Get epJSON field name from eio parameter name"""
        # Direct mapping
        if eio_param_name in self.PARAMETER_MAPPING:
            return self.PARAMETER_MAPPING[eio_param_name]

        # Try to extract field name from eio parameter
        # Pattern: "Design Size field_name [units]" -> "field_name"
        match = re.match(r'Design Size ([\w_]+)', eio_param_name)
        if match:
            return match.group(1).lower()

        # Pattern: "Field Name [units]" -> "field_name"
        match = re.match(r'([\w\s]+)\s*\[', eio_param_name)
        if match:
            field_name = match.group(1).strip()
            # Convert to snake_case
            field_name = field_name.lower().replace(' ', '_')
            return field_name

        return None


def size_freeze(eio_file, original_epjson, frozen_epjson):
    """
    Size freezing workflow - extract sizing data from eio and freeze in epJSON

    Args:
        eio_file: Path to the design day simulation eio file
        original_epjson: Path to the original epJSON file with Autosize
        frozen_epjson: Path to save the frozen epJSON file
    """
    # Check if files exist
    if not os.path.exists(eio_file):
        print(f"Error: eio file not found: {eio_file}")
        return
    if not os.path.exists(original_epjson):
        print(f"Error: Original epJSON not found: {original_epjson}")
        return

    print("="*80)
    print("SIZE FREEZING WORKFLOW")
    print("="*80)

    # Create size freezer
    freezer = SizeFreeze(eio_file, original_epjson)

    # Step 1: Extract sizing data from eio file
    print("\nSTEP 1: Extracting sizing data from eio file")
    print("-"*80)
    freezer.extract_component_sizing_from_eio()

    # Step 2: Freeze all sizes in epJSON
    print("\nSTEP 2: Freezing sizes in epJSON")
    print("-"*80)
    freezer.freeze_sizes_in_epjson(frozen_epjson)

    print("\n" + "="*80)
    print("SIZE FREEZING COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    # Define four cities and their configurations
    cities_config = {
        'Los_Angeles': {
            'eio_file': "output/test/sizing_los_angeles/eplusout.eio",
            'epjsons': [
                {
                    'base': "data/epjson/Los_Angeles/ASHRAE901_OfficeMedium_STD2019_Los_Angeles.epJSON",
                    'frozen': "data/epjson/Los_Angeles/ASHRAE901_OfficeMedium_STD2019_Los_Angeles_frozen.epJSON"
                },
                {
                    'base': "data/epjson/Los_Angeles/ASHRAE901_OfficeMedium_STD2019_Los_Angeles_wo_output.epJSON",
                    'frozen': "data/epjson/Los_Angeles/ASHRAE901_OfficeMedium_STD2019_Los_Angeles_wo_output_frozen.epJSON"
                }
            ]
        },
        'Miami': {
            'eio_file': "output/test/sizing_miami/eplusout.eio",
            'epjsons': [
                {
                    'base': "data/epjson/Miami/ASHRAE901_OfficeMedium_STD2019_Miami.epJSON",
                    'frozen': "data/epjson/Miami/ASHRAE901_OfficeMedium_STD2019_Miami_frozen.epJSON"
                },
                {
                    'base': "data/epjson/Miami/ASHRAE901_OfficeMedium_STD2019_Miami_wo_output.epJSON",
                    'frozen': "data/epjson/Miami/ASHRAE901_OfficeMedium_STD2019_Miami_wo_output_frozen.epJSON"
                }
            ]
        },
        'Montreal': {
            'eio_file': "output/test/sizing_montreal/eplusout.eio",
            'epjsons': [
                {
                    'base': "data/epjson/Montreal/ASHRAE901_OfficeMedium_STD2019_Montreal.epJSON",
                    'frozen': "data/epjson/Montreal/ASHRAE901_OfficeMedium_STD2019_Montreal_frozen.epJSON"
                },
                {
                    'base': "data/epjson/Montreal/ASHRAE901_OfficeMedium_STD2019_Montreal_wo_output.epJSON",
                    'frozen': "data/epjson/Montreal/ASHRAE901_OfficeMedium_STD2019_Montreal_wo_output_frozen.epJSON"
                }
            ]
        },
        'Toronto': {
            'eio_file': "output/test/sizing_toronto/eplusout.eio",
            'epjsons': [
                {
                    'base': "data/epjson/Toronto/ASHRAE901_OfficeMedium_STD2019_Toronto.epJSON",
                    'frozen': "data/epjson/Toronto/ASHRAE901_OfficeMedium_STD2019_Toronto_frozen.epJSON"
                },
                {
                    'base': "data/epjson/Toronto/ASHRAE901_OfficeMedium_STD2019_Toronto_wo_output.epJSON",
                    'frozen': "data/epjson/Toronto/ASHRAE901_OfficeMedium_STD2019_Toronto_wo_output_frozen.epJSON"
                }
            ]
        }
    }

    print("="*100)
    print("Generate FROZEN EPJSON files for four cities")
    print("Two versions per city: base version and wo_output version")
    print("="*100)

    total_files = 0
    successful_files = 0

    for city, config in cities_config.items():
        print(f"\n{'='*60}")
        print(f"Processing city: {city}")
        print(f"{'='*60}")

        eio_file = config['eio_file']

        # Check if eio file exists
        if not os.path.exists(eio_file):
            print(f"❌ Error: eio file not found for {city}: {eio_file}")
            continue

        print(f"Using eio file: {eio_file}")

        for i, epjson_config in enumerate(config['epjsons'], 1):
            base_epjson = epjson_config['base']
            frozen_epjson = epjson_config['frozen']

            version_name = "Base version" if i == 1 else "wo_output version"
            print(f"\n--- {version_name} ---")
            print(f"Input:  {base_epjson}")
            print(f"Output:  {frozen_epjson}")

            total_files += 1

            # Check if base epJSON exists
            if not os.path.exists(base_epjson):
                print(f"❌ Warning: base epJSON not found: {base_epjson}")
                continue

            try:
                # Generate frozen epjson
                size_freeze(eio_file, base_epjson, frozen_epjson)
                successful_files += 1
                print(f"✅ Successfully generated: {frozen_epjson}")

            except Exception as e:
                print(f"❌ Error generating {frozen_epjson}: {str(e)}")

    print(f"\n{'='*100}")
    print("Summary")
    print(f"{'='*100}")
    print(f"Total files: {total_files}")
    print(f"Successfully generated: {successful_files}")
    print(f"Failed: {total_files - successful_files}")
    print(f"{'='*100}")
