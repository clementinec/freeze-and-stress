# Minimal Eppy + pyenergyplus-lbnl test with epJSON
# Function-based version for safe pytest and direct run

from eppy.modeleditor import IDF
import subprocess
import os

# --------------------------
# Function to run minimal EnergyPlus simulation
# --------------------------
def run_simulation(epjson_file, weather_file, output_dir, mode="annual"):
    """
    Run EnergyPlus simulation in either 'designday' or 'annual' mode.
    mode: 'designday' | 'annual'
    """

    # Set IDD before loading epJSON
    IDF.setiddname("/Applications/EnergyPlus-25-1-0/Energy+.idd")

    os.makedirs(output_dir, exist_ok=True)

    # Base command
    cmd = [
        "energyplus",
        "-r",                 # Run simulation (report)
        "-w", weather_file,   # Weather file
        "-d", output_dir,     # Output directory
    ]

    # Add mode flag
    if mode == "annual":
        cmd.insert(1, "-a")  # annual run
        print("Running full-year (annual) simulation...")
    elif mode == "designday":
        cmd.insert(1, "-D")  # design-day only
        print("Running design-day (sizing only) simulation...")
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
            print(f"\nStdout: {e.stdout}")

        raise


# --------------------------
# Entry point
# --------------------------
if __name__ == "__main__":
    base_epjson = "data/epjson/Los_Angeles/ASHRAE901_OfficeMedium_STD2019_Los_Angeles.epJSON"
    frozen_epjson = "data/epjson/Los_Angeles/ASHRAE901_OfficeMedium_STD2019_Los_Angeles_frozen.epJSON"
    weather_file = "data/annex80/Los_Angeles/WDTF_Annex80_build_losa_v1.0_1-13/3B_LosAngeles_HW_Longterm_MostSevere_Longest_2086.epw"  # Use fixed weather file

    # # Run DesignDay (sizing only)
    # run_simulation(
    #     epjson_file=base_epjson,
    #     weather_file=weather_file,
    #     output_dir="output/sim_office_la_designday",
    #     mode="designday"
    # )

    # # Run Annual (using auto sizing)
    # run_simulation(
    #     epjson_file=base_epjson,
    #     weather_file=weather_file,
    #     output_dir="output/sim_office_la_stmy_2041",
    #     mode="annual"
    # )

    # Run Annual (using frozen sizes)
    run_simulation(
        epjson_file=frozen_epjson,
        weather_file=weather_file,
        output_dir="output/test/OfficeMedium_la_annex80_hmy_2086",
        mode="annual"
    )
