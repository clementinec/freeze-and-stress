# Standalone V6 + EnergyPlus Workspace

This workspace is a clean standalone copy of the verified V6 weather-generation path plus the downstream EPW and EnergyPlus assets needed for freeze-and-stress experiments.

## Contents

- `weather/`
  - Verified V6 weather core copied from `SRepw`
  - Local `config.yaml` with the verified `CORDEX_CMIP5_REMO2015_rcp85` path plus additional copied CMIP5-based chains
  - Local copies of the 4-city ISD+solar inputs and multiple future projection chains
- `epw_pipeline/`
  - `generate_epws_from_v6.py` converts V6 hourly forecast CSVs into annual EPWs
  - `energyplus_runner/` contains local TMY files, source models, frozen models, and the future runner
- `run_weather_to_epw.py`
  - Wrapper that runs RS-VAR generation, climate-delta application, and EPW creation end-to-end

## Default Assumption

This standalone copy is intentionally centered on the verified V6 workflow and the 4 cities that already have downstream EnergyPlus building assets:

- `Los_Angeles`
- `Miami`
- `Montreal`
- `Toronto`

Current copied climate-delta scenarios:

- `CORDEX_CMIP5_REMO2015_rcp85`
- `CORDEX_CMIP5_RegCM_rcp85`
- `CMIP5_rcp85`

## Typical Flow

Generate weather and EPWs:

```bash
python run_weather_to_epw.py
```

Run future EnergyPlus simulations using the already-copied frozen models:

```bash
cd epw_pipeline/energyplus_runner
python run_future_simulations.py \
  --epw-root ../epw_out/CORDEX_CMIP5_REMO2015_rcp85 \
  --local-eplus /path/to/energyplus
```

If you want to re-freeze models locally instead of using the copied frozen set:

```bash
cd epw_pipeline/energyplus_runner
python freeze_model.py --local-eplus /path/to/energyplus
```

## Notes

- The weather side uses local relative paths only.
- The EPW headers are derived from the copied local TMY EPWs.
- The current standalone config is deliberately narrow: verified V6 logic, verified REMO2015 forcing, and the 4 cities with complete downstream building assets.
