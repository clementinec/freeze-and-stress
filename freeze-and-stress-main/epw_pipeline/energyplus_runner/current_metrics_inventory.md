# Current Metrics Inventory

Total metrics: 51

Available: 41

Missing: 10

## climate_drivers / thermal_shift

| Metric | Label | Unit | Status | Coverage | Source |
|---|---|---|---|---:|---|
| `annual_mean_drybulb_c` | Annual mean dry-bulb temperature | C | available | 100% | `outdoor_drybulb_c_mean` |
| `summer_mean_drybulb_c` | Summer mean dry-bulb temperature | C | available | 100% | `summer_mean_drybulb_c` |
| `annual_max_drybulb_c` | Annual maximum dry-bulb temperature | C | available | 100% | `outdoor_drybulb_c_max` |

## climate_drivers / demand_background

| Metric | Label | Unit | Status | Coverage | Source |
|---|---|---|---|---:|---|
| `cdd_18` | Cooling degree days | degree-days | available | 100% | `cdd_18` |
| `hdd_18` | Heating degree days | degree-days | available | 100% | `hdd_18` |
| `delta_cdd_18` | Change in cooling degree days | degree-days | available | 100% | `derived:cdd_18-baseline` |
| `delta_hdd_18` | Change in heating degree days | degree-days | available | 100% | `derived:hdd_18-baseline` |

## climate_drivers / humid_heat_stress

| Metric | Label | Unit | Status | Coverage | Source |
|---|---|---|---|---:|---|
| `maximum_wetbulb_c` | Maximum wet-bulb temperature | C | available | 100% | `outdoor_wetbulb_c_max` |
| `high_wetbulb_hours` | High wet-bulb hours | h | available | 100% | `high_wetbulb_hours` |
| `humidity_ratio_shift` | Humidity-ratio shift | kg/kg | available | 100% | `derived:outdoor_humidity_ratio-baseline` |
| `high_rh_hours` | High-RH hours | h | available | 100% | `high_rh_hours` |

## climate_drivers / solar_burden

| Metric | Label | Unit | Status | Coverage | Source |
|---|---|---|---|---:|---|
| `annual_ghi_kwh_m2` | Annual GHI | kWh/m2 | available | 100% | `annual_ghi_kwh_m2` |
| `summer_ghi_kwh_m2` | Summer GHI | kWh/m2 | available | 100% | `summer_ghi_kwh_m2` |
| `peak_ghi_w_m2` | Peak GHI | W/m2 | available | 100% | `peak_ghi_w_m2` |
| `extreme_solar_hours` | Extreme solar hours | h | available | 100% | `extreme_solar_hours` |

## climate_drivers / persistence_heatwave

| Metric | Label | Unit | Status | Coverage | Source |
|---|---|---|---|---:|---|
| `hot_days` | Hot days | days | available | 100% | `hot_days_35c` |
| `heatwave_days` | Heatwave days | days | available | 100% | `heatwave_days` |
| `maximum_consecutive_hot_days` | Maximum consecutive hot days | days | available | 100% | `max_consec_hot_days_35c` |
| `warm_spell_length` | Warm-spell length | days | available | 100% | `max_consec_hot_days_30c` |
| `hot_nights` | Hot nights | nights | available | 100% | `hot_nights` |

## building_responses / operational_resilience

| Metric | Label | Unit | Status | Coverage | Source |
|---|---|---|---|---:|---|
| `cooling_peak_demand_kw` | Cooling peak demand | kW | available | 99.7807% | `cooling_electricity_peak_kw` |
| `facility_peak_demand_kw` | Facility peak demand | kW | available | 100% | `facility_total_electricity_demand_rate_max_kw` |
| `heating_peak_demand_kw` | Heating peak demand | kW | available | 99.7807% | `heating_electricity_peak_kw` |
| `cooling_unmet_hours` | Cooling unmet hours | h | available | 99.7807% | `abups_occupied_cooling_not_met_hours` |
| `heating_unmet_hours` | Heating unmet hours | h | available | 99.7807% | `abups_occupied_heating_not_met_hours` |
| `cooling_setpoint_not_met_occupied_hours` | Cooling setpoint not met occupied hours | h | available | 100% | `facility_cooling_setpoint_not_met_occupied_time_total_hours` |

## building_responses / habitability_human_resilience

| Metric | Label | Unit | Status | Coverage | Source |
|---|---|---|---|---:|---|
| `peak_operative_temperature_c` | Peak operative temperature | C | available | 100% | `rep_zone_operative_temp_c_max` |
| `peak_indoor_air_temperature_c` | Peak indoor air temperature | C | available | 100% | `rep_zone_mean_air_temp_c_max` |
| `maximum_delta_t_k` | Maximum delta_t | K | available | 100% | `delta_t_max_k` |
| `annual_edh_c_h` | Annual EDH | C h | available | 100% | `annual_edh_c_h` |
| `maximum_daily_edh_k_h` | Maximum daily EDH | K h | available | 100% | `daily_edh_max_k_h` |
| `overheating_hours` | Overheating hours | h | available | 99.7807% | `abups_time_not_comfortable_simple_ashrae55_hours` |
| `daily_edh_exceedance_days` | Daily EDH exceedance days | days | available | 100% | `daily_edh_exceed_6kh_count` |
| `hours_of_safety` | Hours of safety | h | missing | 0% | `missing` |
| `set_exceedance_hours` | SET exceedance hours | h | missing | 0% | `missing` |

## building_responses / energy_burden

| Metric | Label | Unit | Status | Coverage | Source |
|---|---|---|---|---:|---|
| `annual_total_electricity_kwh` | Annual total electricity | kWh | available | 99.7807% | `electricity_facility_annual_kwh` |
| `annual_cooling_electricity_kwh` | Annual cooling electricity | kWh | available | 99.7807% | `cooling_electricity_annual_kwh` |
| `annual_heating_electricity_kwh` | Annual heating electricity | kWh | available | 99.7807% | `heating_electricity_annual_kwh` |
| `hvac_electricity_kwh` | HVAC electricity | kWh | available | 99.7807% | `electricity_hvac_annual_kwh` |
| `cooling_energy_intensity_kwh_m2` | Cooling energy intensity | kWh/m2 | available | 99.7807% | `derived:cooling_electricity_annual_kwh/area` |
| `heating_energy_intensity_kwh_m2` | Heating energy intensity | kWh/m2 | available | 99.7807% | `derived:heating_electricity_annual_kwh/area` |

## building_responses / durability_moisture_risk

| Metric | Label | Unit | Status | Coverage | Source |
|---|---|---|---|---:|---|
| `indoor_high_rh_hours` | Indoor high-RH hours | h | missing | 0% | `missing` |
| `indoor_rh_max_pct` | Indoor RH max | % | available | 100% | `rep_zone_rh_pct_max` |
| `indoor_humidity_ratio_exposure` | Indoor humidity-ratio exposure | kg/kg | available | 100% | `rep_zone_humidity_ratio_mean` |
| `condensation_risk_hours` | Condensation risk hours | h | missing | 0% | `missing` |
| `surface_condensation_risk` | Surface condensation risk | index | missing | 0% | `missing` |
| `mold_index` | Mold index | index | missing | 0% | `missing` |

## building_responses / valuation_secondary_overlay

| Metric | Label | Unit | Status | Coverage | Source |
|---|---|---|---|---:|---|
| `electricity_cost` | Electricity cost | currency | missing | 0% | `missing` |
| `carbon_emissions` | Carbon emissions | kgCO2e | missing | 0% | `missing` |
| `peak_demand_cost` | Peak demand cost | currency | missing | 0% | `missing` |
| `adaptation_regret` | Adaptation regret | index | missing | 0% | `missing` |
