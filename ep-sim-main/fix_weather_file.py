#!/usr/bin/env python3
"""
Fix weather file format by converting float dates to integers
"""

def fix_weather_file(input_file, output_file):
    """
    Fix EPW file by converting float date values to integers
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()

            # Skip header lines (they don't start with numbers)
            if not line or not line[0].isdigit():
                outfile.write(line + '\n')
                continue

            # Process data lines
            parts = line.split(',')
            if len(parts) >= 4:  # Should have at least year, month, day, hour
                try:
                    # Convert first 4 fields (year, month, day, hour) to integers
                    parts[0] = str(int(float(parts[0])))  # Year
                    parts[1] = str(int(float(parts[1])))  # Month
                    parts[2] = str(int(float(parts[2])))  # Day
                    parts[3] = str(int(float(parts[3])))  # Hour

                    outfile.write(','.join(parts) + '\n')
                except (ValueError, IndexError):
                    # If conversion fails, write original line
                    outfile.write(line + '\n')
            else:
                outfile.write(line + '\n')

if __name__ == "__main__":
    input_file = "data/stmy_out_bias_corrected/Los_Angeles_stmy_2081_2100.epw"
    output_file = "data/stmy_out_bias_corrected/Los_Angeles_stmy_2081_2100_fixed.epw"

    print(f"Fixing weather file: {input_file}")
    fix_weather_file(input_file, output_file)
    print(f"Fixed weather file saved as: {output_file}")

    # Verify the fix by checking first few data lines
    print("\nFirst few data lines after fix:")
    with open(output_file, 'r') as f:
        lines = f.readlines()
        data_lines = [line for line in lines if line.strip() and line.strip()[0].isdigit()]
        for i, line in enumerate(data_lines[:3]):
            print(f"Line {i+1}: {line.strip()}")
