# EnergyPlus Simulation Workflow Extract

## Overview
This package extracts the EnergyPlus simulation workflow described in `README.md` from `Part 2: EnergyPlus Simulation (WIP)` onward. It is intended as a compact handoff bundle for model preparation, weather-file handling, sizing, scenario simulation, and post-processing.

## Included Scripts

### Core workflow
- `std_epjson_gen.py`: remove site-specific objects from a base epJSON template.
- `city_epjson_gen.py`: merge DDY location and design-day objects into a city-specific epJSON.
- `fix_weather_file.py`: repair EPW rows when year/month/day/hour fields are stored as floats.
- `size_freeze.py`: extract autosized capacities from `eplusout.eio` and write a frozen epJSON.
- `run_pyEP.py`: minimal EnergyPlus launcher used by `run_stmy.py`.
- `run_stmy.py`: TMY-based scenario workflow using current TMY, sTMY, and Annex80 weather files.
- `sMYM_gen.py`: generate yearly EPW files for 2001-2100 from hourly climate CSVs.
- `run_smym.py`: annual batch simulation over yearly sMYM EPWs.
- `analysis.py`: parse `eplustbl.htm`, extract metrics, and generate summary plots/tables.

### Utility scripts
- `utils/create_epw_ddy.py`: convert TMY CSV files to EPW and generate DDY files.
- `utils/calculate_weather_var.py`: weather-variable helper functions used by `create_epw_ddy.py`.
- `utils/parse_annex80.py`: normalize Annex80 MY CSV time axes.
- `utils/epjson_edit.py`: quick epJSON editing example.
- `utils/diff_epjson.py`: compare two epJSON files.

## Expected Data Inputs
This extract does not duplicate the full simulation inputs under `data/`. The main scripts expect paths such as:
- `data/epjson/`
- `data/epw/`
- `data/stmy_out_bias_corrected/`
- `data/annex80/`
- `data/sMYM_epw/`

Hardcoded EnergyPlus paths in several scripts currently assume:
- `energyplus` available on `PATH`
- IDD at `/Applications/EnergyPlus-25-1-0/Energy+.idd`

## Demo Output
`output/simulation_demo/` is included as a worked example. It contains Los Angeles runs for:
- `OfficeMedium`
- `ApartmentMidRise`
- `RetailStandalone`

Each building has:
- `sizing/`: design-day autosizing outputs
- `annual_frozen/`: annual simulation outputs using frozen equipment sizes

## Suggested Execution Order
1. Prepare or standardize a base epJSON with `std_epjson_gen.py`.
2. Build city-specific models using DDY files via `city_epjson_gen.py`.
3. If needed, generate or repair EPW/DDY inputs with `utils/create_epw_ddy.py` and `fix_weather_file.py`.
4. Run sizing and freeze component sizes with `size_freeze.py` or through `run_stmy.py`.
5. Run scenario simulations with `run_stmy.py` or yearly runs with `run_smym.py`.
6. Analyze outputs with `analysis.py`.
