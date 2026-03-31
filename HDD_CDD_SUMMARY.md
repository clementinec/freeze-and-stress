# HDD/CDD Summary

Last updated: 2026-03-31 (HKT)

## Method

- Source files: generated EPWs under `epw_pipeline/epw_out/`
- Metric base temperature: `18.3 C`
- Dry-bulb source: EPW hourly dry-bulb column
- Calculation:
  - `HDD18.3 = sum(max(18.3 - T, 0)) / 24`
  - `CDD18.3 = sum(max(T - 18.3, 0)) / 24`

Machine-readable output is stored in:

- `HDD_CDD_18p3C.csv`

## 2025 To 2100 HDD/CDD Results

| Scenario | City | HDD18.3 2025 | HDD18.3 2100 | Delta HDD | CDD18.3 2025 | CDD18.3 2100 | Delta CDD | Mean Drybulb 2025 | Mean Drybulb 2100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CMIP5_rcp85` | `Los_Angeles` | `799.8` | `166.7` | `-633.1` | `693.7` | `1598.6` | `+904.9` | `18.01` | `22.22` |
| `CMIP5_rcp85` | `Miami` | `24.0` | `3.0` | `-21.0` | `2703.5` | `3746.8` | `+1043.3` | `25.64` | `28.56` |
| `CMIP5_rcp85` | `Montreal` | `3935.3` | `2806.8` | `-1128.5` | `436.1` | `1176.5` | `+740.4` | `8.71` | `13.83` |
| `CMIP5_rcp85` | `Toronto` | `3638.8` | `2369.6` | `-1269.2` | `415.0` | `1312.2` | `+897.2` | `9.47` | `15.40` |
| `CORDEX_CMIP5_RegCM_rcp85` | `Los_Angeles` | `962.5` | `292.1` | `-670.4` | `604.5` | `1336.5` | `+732.0` | `17.32` | `21.16` |
| `CORDEX_CMIP5_RegCM_rcp85` | `Miami` | `30.8` | `7.6` | `-23.2` | `2751.4` | `3713.4` | `+962.0` | `25.75` | `28.45` |
| `CORDEX_CMIP5_RegCM_rcp85` | `Montreal` | `4074.5` | `2896.9` | `-1177.6` | `339.8` | `862.8` | `+523.0` | `8.07` | `12.73` |
| `CORDEX_CMIP5_RegCM_rcp85` | `Toronto` | `3688.5` | `2428.4` | `-1260.1` | `307.3` | `846.1` | `+538.8` | `9.04` | `13.96` |
| `CORDEX_CMIP5_REMO2015_rcp85` | `Los_Angeles` | `950.4` | `252.6` | `-697.8` | `647.7` | `1288.3` | `+640.6` | `17.47` | `21.14` |
| `CORDEX_CMIP5_REMO2015_rcp85` | `Miami` | `57.7` | `13.6` | `-44.1` | `2779.5` | `3723.7` | `+944.2` | `25.76` | `28.46` |
| `CORDEX_CMIP5_REMO2015_rcp85` | `Montreal` | `4022.3` | `3145.3` | `-877.0` | `258.7` | `702.2` | `+443.5` | `7.99` | `11.61` |
| `CORDEX_CMIP5_REMO2015_rcp85` | `Toronto` | `3665.1` | `2830.2` | `-834.9` | `299.8` | `770.3` | `+470.5` | `9.08` | `12.66` |

## Quick Read

- All pathways show the same directional signal:
  - `HDD` drops sharply by 2100
  - `CDD` rises sharply by 2100
- `Miami` is already cooling dominated in 2025 and becomes more extreme by 2100.
- `Montreal` and `Toronto` show the strongest combined shift from heating burden to cooling burden.
- `Los_Angeles` remains mixed in 2025 but becomes decisively more cooling dominated by 2100.

## Related Unmet-Hours Snapshot

The current unmet-hours snapshot used in quick discussion is stored in:

- `epw_pipeline/energyplus_runner/unmet_relative_summary_current/building_scenario_summary.csv`
- `epw_pipeline/energyplus_runner/unmet_relative_summary_current/city_snapshot.csv`

Important caveat:

- that unmet snapshot currently reflects office and apartment summaries
- retail unmet summaries were not refreshed after the later run progress
- REMO apartment is partial in the current snapshot
