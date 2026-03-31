# Standalone Workspace Status

Last updated: 2026-03-31 (HKT)

## Purpose

This folder is the standalone V6-plus-EnergyPlus workspace prepared for the new manuscript workflow. It was built to separate the verified weather-generation path from older manuscript and test branches, and to give us one clean place to:

- generate future hourly weather from the verified V6 logic
- convert those outputs into annual EPWs
- run frozen EnergyPlus archetypes against those EPWs
- extract summary metrics and quick break-year style diagnostics

The workspace is intentionally centered on the verified `SRepw` V6 logic and the four cities with working downstream archetype assets:

- `Los_Angeles`
- `Miami`
- `Montreal`
- `Toronto`

Current climate-delta chains in this workspace:

- `CORDEX_CMIP5_REMO2015_rcp85`
- `CORDEX_CMIP5_RegCM_rcp85`
- `CMIP5_rcp85`

## What Was Set Up Here

The following work has already been done in this folder:

- copied the most up-to-date verified V6 weather workflow into `weather/`
- copied local weather inputs and scenario configuration into the standalone tree
- copied the downstream EPW generation and EnergyPlus runner stack into `epw_pipeline/`
- rewired the downstream pipeline so it points to the local standalone weather outputs
- kept the real-HVAC freeze-and-stress setup, not ideal loads
- added a shared EnergyPlus reporting preset in `epw_pipeline/energyplus_runner/output_preset.py`
- patched / retrofitted frozen models so future runs use the shared reporting preset
- added extraction and summarization scripts for SQLite metrics and unmet-hours diagnostics
- added an hourly monitor logger for long campaigns

## Important Compatibility Notes

- Python environment: tested here with the local Anaconda Python 3.10 environment
- Python packages: `numpy`, `pandas`, `statsmodels`, `PyYAML`, `openpyxl`, `tqdm`
- EnergyPlus: local runs were executed with `EnergyPlus 25.1.0`
- OS: current work was done on macOS

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

EnergyPlus local path used in this workspace:

```bash
/Applications/EnergyPlus-25-1-0/energyplus
```

## Strong Operational Warning

Do not treat OneDrive as a reliable live output location for large EnergyPlus campaigns.

Observed issues during this run:

- repeated `os error 60` (`ETIMEDOUT`)
- repeated `os error 89` (`ECANCELED`)
- inconsistent reads of original `eplusout.err` files
- some batch-marked failures reran successfully when moved to local `/tmp`

Interpretation:

- the weather files themselves do not currently look like the main problem
- the live output / sync layer very likely contaminated part of the failure map

Recommended practice for the next campaign:

1. Keep code in the repo.
2. Keep large input data outside git and provision it separately.
3. Run EnergyPlus outputs to a non-OneDrive local path.
4. Copy or archive results later if needed.

## Main Entry Points

### Weather generation wrapper

File: `run_weather_to_epw.py`

Purpose:

- top-level convenience wrapper for the weather side
- runs RS-VAR generation
- applies climate delta
- generates annual EPWs

Typical use:

```bash
python run_weather_to_epw.py
```

### V6 RS-VAR generation

File: `weather/run_v5_save_rsvar_batch.py`

Purpose:

- generates the baseline RS-VAR future weather time series from local city inputs

Typical use:

- normally driven by `run_weather_to_epw.py`

### Climate-delta application

File: `weather/apply_climate_delta_batch.py`

Purpose:

- applies the selected scenario chain to the RS-VAR outputs

Typical use:

- normally driven by `run_weather_to_epw.py`

### EPW generation

File: `epw_pipeline/generate_epws_from_v6.py`

Purpose:

- converts V6 future forecast CSVs into annual EPWs
- uses local TMY headers stored in `epw_pipeline/energyplus_runner/*.epw`

Typical use:

- normally driven by `run_weather_to_epw.py`

### Freeze source models

File: `epw_pipeline/energyplus_runner/freeze_model.py`

Purpose:

- runs TMY sizing
- replaces `Autosize` values with frozen capacities
- writes frozen epJSON models for future stress runs
- applies the shared reporting preset

Typical local use:

```bash
cd epw_pipeline/energyplus_runner
python freeze_model.py --local-eplus /Applications/EnergyPlus-25-1-0/energyplus
```

### Retrofit reporting preset onto existing frozen models

File: `epw_pipeline/energyplus_runner/retrofit_reporting_preset.py`

Purpose:

- reapplies the shared reporting block to an existing frozen-model set

Typical use:

```bash
cd epw_pipeline/energyplus_runner
python retrofit_reporting_preset.py
```

### Future EnergyPlus batch runner

File: `epw_pipeline/energyplus_runner/run_future_simulations.py`

Purpose:

- runs all selected archetypes, cities, and years against an EPW scenario tree
- supports local EnergyPlus or Docker
- default building set is `office, apartment, retail`
- default year range is `2025-2100`

Typical local use:

```bash
cd epw_pipeline/energyplus_runner
python run_future_simulations.py \
  --epw-root ../epw_out/CORDEX_CMIP5_REMO2015_rcp85 \
  --local-eplus /Applications/EnergyPlus-25-1-0/energyplus
```

Useful filters:

- `--start-year`
- `--end-year`
- `--cities`
- `--buildings`
- `--output-dir`

For future work, use a local non-OneDrive `--output-dir`.

### Monitoring

File: `monitor_campaign_status.py`

Purpose:

- writes hourly campaign snapshots to `run_status/hourly_status.log`

### Post-processing and diagnostics

Files:

- `epw_pipeline/energyplus_runner/extract_sqlite_metrics.py`
- `epw_pipeline/energyplus_runner/extract_unmet_hours_fast.py`
- `epw_pipeline/energyplus_runner/compute_quick_breakyears.py`
- `epw_pipeline/energyplus_runner/summarize_unmet_relative.py`

Purpose:

- extract annual metrics from EnergyPlus SQLite outputs
- build quick unmet-hours based break-year tables
- summarize unmet hours relative to the 2025 baseline

Status:

- these scripts are in place and were used during this campaign
- their generated outputs are local artifacts and are gitignored

## Current Reporting Preset

Shared reporting preset file:

- `epw_pipeline/energyplus_runner/output_preset.py`

Current payload includes:

- SQLite output
- summary reports:
  - `AnnualBuildingUtilityPerformanceSummary`
  - `SystemSummary`
  - `EnergyMeters`
- timestep facility electricity demand / energy outputs
- timestep facility setpoint-not-met outputs
- representative-zone thermal condition outputs
- selected electricity meters
- reporting tolerances for setpoint-not-met:
  - heating `0.556 C`
  - cooling `0.556 C`

## Current Campaign Status

The active REMO run was manually stopped on 2026-03-31 so that the workspace could be documented and prepared for git.

### Authoritative finished summaries

These two scenarios completed and wrote final `simulation_summary.json` files:

| Scenario | Success | Failed | Notes |
|---|---:|---:|---|
| `CMIP5_rcp85` | `845` | `67` | finished first pass |
| `CORDEX_CMIP5_RegCM_rcp85` | `829` | `83` | finished first pass |

Breakdown from the saved summary files:

| Scenario | Office | Apartment | Retail |
|---|---:|---:|---:|
| `CMIP5_rcp85` | `304/304` | `299/304` | `242/304` |
| `CORDEX_CMIP5_RegCM_rcp85` | `304/304` | `296/304` | `229/304` |

### REMO partial stop state

`CORDEX_CMIP5_REMO2015_rcp85` did not finish. It was interrupted manually during:

- `retail/Toronto/2036`

Observed stop-point details:

- latest live progress before interrupt: `847/912` attempted
- latest hourly monitor snapshot in `run_status/hourly_status.log`: `764/912`, latest `retail/Toronto/2036`
- no final `simulation_summary.json` was written for REMO

Practical reading:

- office was fully run
- apartment was fully attempted
- retail was only partially completed

### Latest hourly monitor entries

Recent log snapshot from `run_status/hourly_status.log`:

```text
[2026-03-31 09:29:01]
CORDEX_CMIP5_REMO2015_rcp85: sims=747/912, epws=304/304, latest=retail/Montreal/2095
CORDEX_CMIP5_RegCM_rcp85: sims=867/912, epws=304/304, latest=retail/Toronto/2100
CMIP5_rcp85: sims=857/912, epws=304/304, latest=retail/Toronto/2100
[2026-03-31 10:29:01]
CORDEX_CMIP5_REMO2015_rcp85: sims=764/912, epws=304/304, latest=retail/Toronto/2036
CORDEX_CMIP5_RegCM_rcp85: sims=867/912, epws=304/304, latest=retail/Toronto/2100
CMIP5_rcp85: sims=857/912, epws=304/304, latest=retail/Toronto/2100
```

Note:

- the hourly monitor is useful for progress tracking
- for finished scenarios, the authoritative success/failure counts are still the saved `simulation_summary.json` files

## Failure Interpretation So Far

Working conclusions from the current campaign:

- failures do not look like EPW corruption
- the same city/year EPWs often run for one archetype while another archetype fails
- at least two batch-marked failures reran successfully to local `/tmp`
- this means a nontrivial share of the failure map is likely batch / storage / sync contamination rather than deterministic building-physics failure

So the current failure inventory should be treated as:

- a rerun queue
- not a final scientific result

## Recommended Next Step

For the next clean pass:

1. keep this code/config workspace
2. provision local weather input data outside git
3. write EnergyPlus results to a non-OneDrive local path
4. rerun unfinished REMO retail years
5. rerun all currently failed cases into that local output root
6. only then freeze the clean success/failure map and downstream metrics

## Git Policy For This Workspace

This workspace is now configured so git should track:

- code
- configs
- lightweight documentation

Git should not track:

- weather input data
- generated weather outputs
- generated EPWs
- EnergyPlus results
- metric exports
- hourly monitoring logs

See `.gitignore` in this folder.
