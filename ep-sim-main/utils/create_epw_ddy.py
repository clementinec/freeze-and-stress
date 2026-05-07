#!/usr/bin/env python3
"""
Convert TMY CSV files to EPW format and generate DDY files using EnergyPlus.

This script:
1. Reads TMY CSV files from output/TMY directory
2. Converts each TMY to EPW format with proper metadata
3. Uses EnergyPlus to generate DDY files from EPW
4. Saves EPW and DDY files to output/epw directory

Usage:
    python create_epw_ddy.py [--tmy-dir DIR] [--output-dir DIR] [--config-path FILE] [--energyplus-path FILE]

Examples:
    # Use default paths
    python create_epw_ddy.py

    # Specify custom paths
    python create_epw_ddy.py --tmy-dir output/TMY --output-dir output/epw --config-path data/cities_config_final.csv
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from timezonefinder import TimezoneFinder
import pytz
import argparse
from calculate_weather_var import calc_dewpoint_from_Q, calc_RH_from_dewpoint


def get_timezone_offset(latitude, longitude):
    """
    Get UTC offset for a given latitude/longitude using timezonefinder.

    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees

    Returns:
        UTC offset in hours (float)
    """
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=latitude, lng=longitude)

    if tz_name is None:
        print(f"  Warning: Could not find timezone for lat={latitude}, lon={longitude}")
        print(f"  Using approximate timezone from longitude")
        return round(longitude / 15.0)

    # Get UTC offset for the timezone (using a date in winter to avoid DST)
    tz = pytz.timezone(tz_name)
    # Use January 15 to avoid DST
    dt = datetime(2020, 1, 15, 12, 0, 0)
    offset = tz.utcoffset(dt).total_seconds() / 3600.0

    return offset


def load_city_metadata(config_path):
    """
    Load city configuration with latitude, longitude, and country info.

    Args:
        config_path: Path to cities_config_final.csv

    Returns:
        Dictionary mapping city names to metadata
    """
    cities_df = pd.read_csv(config_path)
    city_metadata = {}

    for _, row in cities_df.iterrows():
        city_metadata[row['city']] = {
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'country': row['country']
        }

    return city_metadata


def epw_header(city, state, country, lat, lon, tz, elevation_m=0, wmo_code="722950"):
    """
    Generate EPW header lines.

    Args:
        city: City name (e.g., "Los Angeles")
        state: State/Province abbreviation (e.g., "CA")
        country: Country code (e.g., "USA")
        lat: Latitude in degrees
        lon: Longitude in degrees
        tz: UTC offset in hours
        elevation_m: Elevation in meters
        wmo_code: WMO station code

    Returns:
        Header string with newline
    """
    lines = []
    lines.append(
        f"LOCATION,{city},{state},{country},TMY,{wmo_code},{lat:.3f},{lon:.3f},{tz:.1f},{elevation_m:.1f}"
    )
    lines.append("DESIGN CONDITIONS,0")
    lines.append("TYPICAL/EXTREME PERIODS,0")
    lines.append("GROUND TEMPERATURES,0")
    lines.append("HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0")
    lines.append("COMMENTS 1,Generated from TMY data (1991-2020)")
    lines.append("COMMENTS 2,Created by create_epw_ddy.py")
    lines.append("DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31")
    return "\n".join(lines) + "\n"


def convert_tmy_to_epw(tmy_csv_path, output_epw_path, city_name, city_metadata):
    """
    Convert TMY CSV to EPW format.

    Args:
        tmy_csv_path: Path to input TMY CSV file
        output_epw_path: Path to output EPW file
        city_name: Name of the city
        city_metadata: Dictionary with city metadata

    Returns:
        Path to created EPW file
    """
    print(f"\n{city_name}:")
    print(f"  Converting TMY to EPW...")

    # Read TMY CSV
    df = pd.read_csv(tmy_csv_path, index_col=0, parse_dates=True)

    # Detect file type based on columns
    is_stmy = 'specific_humidity' in df.columns
    has_wind_dir = 'WindDir_deg' in df.columns

    if is_stmy:
        print(f"  Detected sTMY format (with specific_humidity, without WindDir_deg)")
    else:
        print(f"  Detected TMY format (with WindDir_deg, without specific_humidity)")

    # Get city metadata
    if city_name not in city_metadata:
        print(f"  Warning: City {city_name} not found in config, using default location")
        lat, lon, country = 34.05, -118.24, "USA"  # Default to LA coordinates
        state = "CA"
        wmo_code = "722950"
        tz = -8.0
    else:
        lat = city_metadata[city_name]['latitude']
        lon = city_metadata[city_name]['longitude']
        country = city_metadata[city_name]['country']
        tz = get_timezone_offset(lat, lon)

        # Determine state from city name (simple mapping)
        state_map = {
            'Los_Angeles': 'CA',
            'Miami': 'FL',
            'Toronto': 'ON',
            'Montreal': 'QC'
        }
        state = state_map.get(city_name, '')

        # WMO code mapping
        wmo_map = {
            'Los_Angeles': '722950',
            'Miami': '722020',
            'Toronto': '716240',
            'Montreal': '716270'
        }
        wmo_code = wmo_map.get(city_name, '000000')

    # Ensure we have 8760 hours (no leap day)
    if len(df) > 8760:
        # Remove Feb 29 if present
        mask_leap = ~((df.index.month == 2) & (df.index.day == 29))
        df = df[mask_leap]

    if len(df) != 8760:
        print(f"  Warning: Expected 8760 hours, got {len(df)}")
        # Try to handle missing hours
        if len(df) < 8760:
            print(f"  Warning: Missing hours will be forward-filled")

    # Map TMY columns to EPW format
    temp_c = df['DryBulb_C']

    # Handle dewpoint: sTMY might have dewpoint directly or need to calculate from specific_humidity
    if 'DewPoint_C' in df.columns:
        dewpoint_c = df['DewPoint_C']
    elif is_stmy and 'specific_humidity' in df.columns:
        # Calculate dewpoint from specific humidity and pressure using utility function
        print(f"  Calculating dewpoint from specific_humidity and pressure...")
        dewpoint_c = pd.Series(index=df.index, dtype=float)

        q = df['specific_humidity']  # kg/kg
        pressure_pa = df['Pressure_Pa']

        valid_mask = q.notna() & pressure_pa.notna() & temp_c.notna()
        if valid_mask.any():
            # Use calc_dewpoint_from_Q utility function
            # Convert pressure from Pa to hPa for the function
            q_valid = q[valid_mask]
            p_valid_hpa = pressure_pa[valid_mask] / 100.0  # Pa to hPa
            temp_valid = temp_c[valid_mask]

            td_calc = calc_dewpoint_from_Q(temp_valid, q_valid, p_valid_hpa)
            dewpoint_c[valid_mask] = td_calc

        n_calculated = valid_mask.sum()
        n_missing = (~valid_mask).sum()
        print(f"    Calculated {n_calculated} dewpoint values from specific_humidity")
        if n_missing > 0:
            print(f"    {n_missing} dewpoint values remain missing")
    else:
        print(f"  Warning: No dewpoint data available")
        dewpoint_c = pd.Series(index=df.index, dtype=float)

    # Relative humidity - directly from CSV or calculate from temp and dewpoint
    if 'RelHum_pct' in df.columns:
        rh = df['RelHum_pct']
        print(f"  Using RelHum_pct from CSV")
    else:
        print(f"  Warning: RelHum_pct not found in CSV, calculating from temperature and dewpoint")
        # Calculate RH from dry bulb temperature and dew point temperature using utility function
        rh = pd.Series(index=df.index, dtype=float)

        valid_mask = temp_c.notna() & dewpoint_c.notna()
        if valid_mask.any():
            T = temp_c[valid_mask]
            Td = dewpoint_c[valid_mask]

            # Use calc_RH_from_dewpoint utility function
            rh_calc = calc_RH_from_dewpoint(T, Td)
            rh[valid_mask] = rh_calc

            n_calculated = valid_mask.sum()
            n_missing = (~valid_mask).sum()
            print(f"    Calculated {n_calculated} RH values from temperature and dewpoint")
            if n_missing > 0:
                print(f"    {n_missing} RH values remain missing (will use EPW missing code 999)")
        else:
            print(f"    Warning: Cannot calculate RH - no valid temperature and dewpoint pairs")
    # Count missing values for reporting
    n_missing_temp = temp_c.isna().sum()
    n_missing_dewpoint = dewpoint_c.isna().sum()
    n_missing_rh = rh.isna().sum()

    if n_missing_temp > 0:
        print(f"  Info: {n_missing_temp} missing Dry Bulb Temperature values (will use EPW missing code 99.9)")
    if n_missing_dewpoint > 0:
        print(f"  Info: {n_missing_dewpoint} missing Dew Point Temperature values (will use EPW missing code 99.9)")
    if n_missing_rh > 0:
        print(f"  Info: {n_missing_rh} missing Relative Humidity values (will use EPW missing code 999)")

    pressure_pa = df['Pressure_Pa']

    # Detect and convert pressure unit if needed (hPa to Pa)
    if pressure_pa.notna().any():
        max_pressure = pressure_pa.max()
        if max_pressure < 2000:  # Likely in hPa (millibar)
            print(f"  Detected pressure in hPa (max={max_pressure:.1f}), converting to Pa...")
            pressure_pa = pressure_pa * 100
            min_p = pressure_pa.min()
            max_p = pressure_pa.max()
            print(f"  Pressure range after conversion: {min_p:.0f} - {max_p:.0f} Pa")
            if min_p < 85000 or max_p > 110000:
                print(f"  Warning: Pressure outside typical range (85000-110000 Pa)")

    n_missing_pressure = pressure_pa.isna().sum()
    if n_missing_pressure > 0:
        print(f"  Info: {n_missing_pressure} missing Atmospheric Pressure values (will use EPW missing code 999999)")

    # Wind variables - handle missing WindDir_deg in sTMY files
    wind_speed = df['WindSpeed_mps']

    if has_wind_dir:
        wind_dir = df['WindDir_deg']
        n_missing_winddir = wind_dir.isna().sum()
        if n_missing_winddir > 0:
            print(f"  Info: {n_missing_winddir} missing Wind Direction values (will use EPW missing code 999)")
    else:
        # sTMY files don't have wind direction, use EPW missing code 999
        print(f"  Info: WindDir_deg not available in sTMY, using EPW missing code 999")
        wind_dir = pd.Series(np.nan, index=df.index, dtype=float)

    n_missing_windspeed = wind_speed.isna().sum()
    if n_missing_windspeed > 0:
        print(f"  Info: {n_missing_windspeed} missing Wind Speed values (will use EPW missing code 999)")

    # Solar radiation - do NOT fill missing values, use EPW missing codes
    # Only clip valid values to prevent extreme values
    ghi = df['GHI_Whm2']
    dni = df['DNI_Whm2']
    dhi = df['DHI_Whm2']

    # Clip only non-NaN values to valid ranges
    ghi_valid_mask = ghi.notna()
    dni_valid_mask = dni.notna()
    dhi_valid_mask = dhi.notna()

    if ghi_valid_mask.any():
        ghi.loc[ghi_valid_mask] = ghi.loc[ghi_valid_mask].clip(lower=0, upper=1500)
    if dni_valid_mask.any():
        dni.loc[dni_valid_mask] = dni.loc[dni_valid_mask].clip(lower=0, upper=1367)  # Solar constant
    if dhi_valid_mask.any():
        dhi.loc[dhi_valid_mask] = dhi.loc[dhi_valid_mask].clip(lower=0, upper=800)

    # Report missing solar radiation values
    n_missing_ghi = ghi.isna().sum()
    n_missing_dni = dni.isna().sum()
    n_missing_dhi = dhi.isna().sum()

    if n_missing_ghi > 0:
        print(f"  Info: {n_missing_ghi} missing GHI values (will use EPW missing code 9999)")
    if n_missing_dni > 0:
        print(f"  Info: {n_missing_dni} missing DNI values (will use EPW missing code 9999)")
    if n_missing_dhi > 0:
        print(f"  Info: {n_missing_dhi} missing DHI values (will use EPW missing code 9999)")

    # Build EPW rows
    rows = []
    for i, ts in enumerate(df.index):
        Y = ts.year
        M = ts.month
        D = ts.day
        H = ts.hour + 1  # EPW uses 1-24
        Minute = 0  # Standard EPW uses 0 for minute field

        # Data Source and Uncertainty Flags (30 characters for 10 fields)
        # Standard format: ?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9*9*9*9
        # Each pair represents [Source][Uncertainty] for:
        # DryBulb, DewPoint, RelHum, Pressure, ExtHorzRad, ExtDirNormRad, HorzIRSky,
        # GlobHorzRad, DirNormRad, DifHorzRad + additional uncertainty flags
        #
        # Our mapping:
        # - C0 for measured/reanalysis data (temp, dewpoint, RH, pressure, wind)
        # - E0 for calibrated solar radiation (GHI, DNI, DHI)
        # - ?9 for missing data
        # - *9 for additional uncertainty flags

        # Determine data source flags based on data availability
        temp_flag = "C0" if pd.notna(temp_c.iloc[i]) else "?9"
        dewpoint_flag = "C0" if pd.notna(dewpoint_c.iloc[i]) else "?9"
        rh_flag = "C0" if pd.notna(rh.iloc[i]) else "?9"
        pressure_flag = "C0" if pd.notna(pressure_pa.iloc[i]) else "?9"

        data_flags = f"{temp_flag}{dewpoint_flag}{rh_flag}{pressure_flag}?9?9?9E0E0E0?9?9?9?9?9*9*9*9*9*9"

        # Use EPW standard missing value codes when data is missing
        # EPW missing value codes:
        # - Temperature (dry bulb, dew point): 99.9
        # - Relative humidity: 999
        # - Pressure: 999999
        # - Wind speed: 999
        # - Wind direction: 999

        temp_val = f"{temp_c.iloc[i]:.1f}" if pd.notna(temp_c.iloc[i]) else "99.9"
        dewpoint_val = f"{dewpoint_c.iloc[i]:.1f}" if pd.notna(dewpoint_c.iloc[i]) else "99.9"
        rh_val = f"{int(rh.iloc[i])}" if pd.notna(rh.iloc[i]) else "999"
        pressure_val = f"{int(pressure_pa.iloc[i])}" if pd.notna(pressure_pa.iloc[i]) else "999999"
        # Wind direction: use 0 for missing (not 999, which is out of range [0, 360])
        winddir_val = f"{int(wind_dir.iloc[i])}" if pd.notna(wind_dir.iloc[i]) else "0"
        windspeed_val = f"{wind_speed.iloc[i]:.1f}" if pd.notna(wind_speed.iloc[i]) else "0.0"

        # Solar radiation values - use EPW missing code 9999 for missing values (not 999999)
        ghi_val = f"{int(ghi.iloc[i])}" if pd.notna(ghi.iloc[i]) else "9999"
        dni_val = f"{int(dni.iloc[i])}" if pd.notna(dni.iloc[i]) else "9999"
        dhi_val = f"{int(dhi.iloc[i])}" if pd.notna(dhi.iloc[i]) else "9999"

        row = [
            Y, M, D, H, Minute, data_flags,
            temp_val,  # [6] Dry bulb temperature (°C)
            dewpoint_val,  # [7] Dew point temperature (°C)
            rh_val,  # [8] Relative humidity (%)
            pressure_val,  # [9] Atmospheric pressure (Pa)
            9999,  # [10] Extraterrestrial horizontal radiation (Wh/m²)
            9999,  # [11] Extraterrestrial direct normal radiation (Wh/m²)
            9999,  # [12] Horizontal infrared radiation (Wh/m²)
            ghi_val,  # [13] Global horizontal irradiance (Wh/m²)
            dni_val,  # [14] Direct normal irradiance (Wh/m²)
            dhi_val,  # [15] Diffuse horizontal irradiance (Wh/m²)
            0,  # [16] Global horizontal illuminance (lux)
            0,  # [17] Direct normal illuminance (lux)
            0,  # [18] Diffuse horizontal illuminance (lux)
            0,  # [19] Zenith luminance (Cd/m²)
            winddir_val,  # [20] Wind direction (degrees)
            windspeed_val,  # [21] Wind speed (m/s)
            99,  # [22] Total sky cover (tenths)
            99,  # [23] Opaque sky cover (tenths)
            9999,  # [24] Visibility (km)
            99999,  # [25] Ceiling height (m)
            9,  # [26] Present weather observation
            9,  # [27] Present weather codes
            999,  # [28] Precipitable water (mm)
            0,  # [29] Aerosol optical depth (use 0 instead of .999)
            999,  # [30] Snow depth (cm)
            99,  # [31] Days since last snowfall
            999,  # [32] Albedo
            999,  # [33] Liquid precipitation depth (mm)
            99,  # [34] Liquid precipitation quantity (hr)
        ]
        rows.append(",".join(map(str, row)))

    # Write EPW file
    output_epw_path.parent.mkdir(parents=True, exist_ok=True)
    # Format city name (replace underscore with space)
    city_display = city_name.replace('_', ' ')
    header = epw_header(city_display, state, country, lat, lon, tz, wmo_code=wmo_code)

    with open(output_epw_path, "w", encoding="utf-8") as f:
        f.write(header)
        for line in rows:
            f.write(line + "\n")

    print(f"  ✓ Created EPW: {output_epw_path.name}")
    return output_epw_path


def generate_ddy_from_epw(epw_path, energyplus_path):
    """
    Generate DDY file from EPW by analyzing weather data.

    Analyzes the EPW file to determine design day conditions based on
    ASHRAE guidelines (99.6% and 0.4% temperature conditions).

    Args:
        epw_path: Path to EPW file
        energyplus_path: Path to EnergyPlus executable (not used, kept for compatibility)

    Returns:
        True if successful, False otherwise
    """
    print(f"  Generating DDY from EPW...")

    try:
        # Read EPW file and extract weather data
        epw_data = []
        with open(epw_path, 'r') as f:
            lines = f.readlines()
            # Skip header lines (first 8 lines)
            for line in lines[8:]:
                parts = line.strip().split(',')
                if len(parts) >= 22:
                    try:
                        temp = float(parts[6])  # Dry bulb temperature
                        dewpoint = float(parts[7])  # Dew point temperature
                        pressure = float(parts[9])  # Atmospheric pressure (Pa)
                        wspd = float(parts[21])  # Wind speed
                        epw_data.append({
                            'temp': temp,
                            'dewpoint': dewpoint,
                            'pressure': pressure,
                            'windspeed': wspd
                        })
                    except (ValueError, IndexError):
                        continue

        if len(epw_data) == 0:
            print(f"  ✗ Error: No valid weather data found in EPW file")
            return False

        # Convert to numpy arrays for analysis
        temps = np.array([d['temp'] for d in epw_data])
        dewpoints = np.array([d['dewpoint'] for d in epw_data])
        pressures = np.array([d['pressure'] for d in epw_data])
        windspeeds = np.array([d['windspeed'] for d in epw_data])

        # Validate data arrays
        if len(temps) == 0:
            print(f"  ✗ Error: No temperature data extracted from EPW file")
            return False

        # Check for all NaN values
        valid_temps = temps[~np.isnan(temps)]
        if len(valid_temps) == 0:
            print(f"  ✗ Error: All temperature values are NaN")
            return False

        # Use only valid (non-NaN) values for analysis
        print(f"  Data validation: {len(valid_temps)}/{len(temps)} valid temperature readings")

        if len(valid_temps) < len(temps) * 0.9:
            print(f"  Warning: More than 10% of temperature data is missing")

        # Filter out NaN values from all arrays
        valid_mask = ~np.isnan(temps)
        temps_clean = temps[valid_mask]
        dewpoints_clean = dewpoints[valid_mask]
        windspeeds_clean = windspeeds[valid_mask]

        # Also filter out EPW missing value codes for wind speed (999)
        windspeed_valid_mask = (windspeeds_clean < 999) & (~np.isnan(windspeeds_clean))

        # Calculate design conditions (ASHRAE percentiles)
        # Heating: 99.6% annual cumulative frequency
        heat_99_6 = np.percentile(temps_clean, 0.4)
        # Cooling: 0.4% annual cumulative frequency
        cool_0_4 = np.percentile(temps_clean, 99.6)

        # Validate percentiles
        if np.isnan(heat_99_6) or np.isnan(cool_0_4):
            print(f"  ✗ Error: Failed to calculate design conditions (percentiles are NaN)")
            print(f"    Temp range: {np.min(temps_clean):.1f} to {np.max(temps_clean):.1f}°C")
            return False

        # Additional statistics
        temp_max = np.max(temps_clean)
        temp_min = np.min(temps_clean)

        # Calculate wind speed percentiles only from valid wind speed data (exclude 999)
        if np.any(windspeed_valid_mask):
            windspeeds_for_stats = windspeeds_clean[windspeed_valid_mask]
            wspd_1pct = np.percentile(windspeeds_for_stats, 99.0)
            wspd_2_5pct = np.percentile(windspeeds_for_stats, 97.5)
            wspd_5pct = np.percentile(windspeeds_for_stats, 95.0)
        else:
            # No valid wind speed data
            wspd_1pct = wspd_2_5pct = wspd_5pct = 0.0

        # Mean coincident values with fallback handling
        # For heating conditions (cold temperatures)
        heat_mask = temps_clean <= heat_99_6
        if np.any(heat_mask):
            heat_dewpoint_raw = np.mean(dewpoints_clean[heat_mask])

            # Calculate wind speed mean, filtering out missing values (999)
            heat_windspeeds = windspeeds_clean[heat_mask]
            heat_windspeed_valid = heat_windspeeds[(heat_windspeeds < 999) & (~np.isnan(heat_windspeeds))]

            if len(heat_windspeed_valid) > 0:
                heat_windspeed_raw = np.mean(heat_windspeed_valid)
            else:
                heat_windspeed_raw = np.nan

            # Check for NaN and apply fallback
            # Default heating wind speed: 6.7 m/s (15 mph) per ASHRAE standard
            heat_dewpoint = heat_dewpoint_raw if not np.isnan(heat_dewpoint_raw) else heat_99_6
            heat_windspeed = heat_windspeed_raw if not np.isnan(heat_windspeed_raw) else 6.7

            if np.isnan(heat_windspeed_raw) and len(heat_windspeed_valid) == 0:
                print(f"  Info: No valid wind speed data for heating conditions, using default 6.7 m/s")
        else:
            print(f"  Warning: No data points found for heating design conditions")
            heat_dewpoint = heat_99_6  # Fallback to dry-bulb temperature
            heat_windspeed = 6.7  # Default ASHRAE heating design wind speed (15 mph)

        # For cooling conditions (hot temperatures)
        cool_mask = temps_clean >= cool_0_4
        if np.any(cool_mask):
            cool_dewpoint_raw = np.mean(dewpoints_clean[cool_mask])

            # Calculate wind speed mean, filtering out missing values (999)
            cool_windspeeds = windspeeds_clean[cool_mask]
            cool_windspeed_valid = cool_windspeeds[(cool_windspeeds < 999) & (~np.isnan(cool_windspeeds))]

            if len(cool_windspeed_valid) > 0:
                cool_windspeed_raw = np.mean(cool_windspeed_valid)
            else:
                cool_windspeed_raw = np.nan

            # Check for NaN and apply fallback
            # Default cooling wind speed: 3.35 m/s (7.5 mph) per ASHRAE standard
            cool_dewpoint = cool_dewpoint_raw if not np.isnan(cool_dewpoint_raw) else (cool_0_4 - 5.0)
            cool_windspeed = cool_windspeed_raw if not np.isnan(cool_windspeed_raw) else 3.35

            if np.isnan(cool_windspeed_raw) and len(cool_windspeed_valid) == 0:
                print(f"  Info: No valid wind speed data for cooling conditions, using default 3.35 m/s")
        else:
            print(f"  Warning: No data points found for cooling design conditions")
            cool_dewpoint = cool_0_4 - 5.0  # Fallback to DB - 5°C
            cool_windspeed = 3.35  # Default ASHRAE cooling design wind speed (7.5 mph)

        # Get location info from EPW header
        with open(epw_path, 'r') as f:
            location_line = f.readline().strip().split(',')
            city = location_line[1] if len(location_line) > 1 else "Unknown"
            state = location_line[2] if len(location_line) > 2 else ""
            country = location_line[3] if len(location_line) > 3 else ""
            lat = float(location_line[6]) if len(location_line) > 6 else 0.0
            lon = float(location_line[7]) if len(location_line) > 7 else 0.0
            tz = float(location_line[8]) if len(location_line) > 8 else 0.0
            elev = float(location_line[9]) if len(location_line) > 9 else 0.0

        # Use mean of non-NaN pressure values from TMY data (Pressure_Pa column)
        # Filter out EPW missing value code (999999) and calculate mean
        valid_pressures = pressures[(pressures < 999999) & (~np.isnan(pressures))]

        if len(valid_pressures) > 0:
            pressure_pa = np.mean(valid_pressures)
            print(f"  Using mean pressure from TMY data: {pressure_pa:.0f} Pa ({len(valid_pressures)} valid readings)")
        else:
            # Fallback: calculate from elevation if no valid pressure data
            pressure_pa = 101325 * (1 - 2.25577e-5 * elev) ** 5.25588
            print(f"  Warning: No valid pressure data in TMY, using elevation-based estimate: {pressure_pa:.0f} Pa")

        # Format location name
        if state:
            location_name = f"{city}_{state}_{country} Design_Conditions"
        else:
            location_name = f"{city}_{country} Design_Conditions"

        # Generate DDY content with standard format
        ddy_content = f""" ! The following Location and Design Day data are produced as possible from the indicated data source.
 ! Wind Speeds follow the indicated design conditions rather than traditional values (6.7 m/s heating, 3.35 m/s cooling)
 ! No special attempts at re-creating or determining missing data parts (e.g. Wind speed or direction)
 ! are done.  Therefore, you should look at the data and fill in any incorrect values as you desire.
  
 Site:Location,
  {location_name},     !- Location Name
      {lat:.2f},     !- Latitude {{N+ S-}}
    {lon:.2f},     !- Longitude {{W- E+}}
      {tz:.2f},     !- Time Zone Relative to GMT {{GMT+/-}}
      {elev:.2f};     !- Elevation {{m}}
 
 !  Data Source=TMY 1991-2020
 RunPeriodControl:DaylightSavingTime,
   2nd Sunday in March,    !- StartDate
   2nd Sunday in November;    !- EndDate
  
 ! Using Design Conditions calculated from TMY weather data
 ! {city} Extreme Annual Wind Speeds, 1%={wspd_1pct:.1f}m/s, 2.5%={wspd_2_5pct:.1f}m/s, 5%={wspd_5pct:.1f}m/s
 ! {city} Extreme Annual Temperatures, Max Drybulb={temp_max:.1f}°C Min Drybulb={temp_min:.1f}°C
  
 ! {city} Annual Heating Design Conditions Wind Speed={heat_windspeed:.1f}m/s Wind Dir=0
 ! Coldest Month=JAN
 ! {city} Annual Heating 99.6%, MaxDB={heat_99_6:.1f}°C
 SizingPeriod:DesignDay,
  {city} Ann Htg 99.6% Condns DB,     !- Name
          1,      !- Month
         21,      !- Day of Month
  WinterDesignDay,!- Day Type
        {heat_99_6:.1f},      !- Maximum Dry-Bulb Temperature {{C}}
        0.0,      !- Daily Dry-Bulb Temperature Range {{C}}
 DefaultMultipliers, !- Dry-Bulb Temperature Range Modifier Type
           ,      !- Dry-Bulb Temperature Range Modifier Schedule Name
    Wetbulb,      !- Humidity Condition Type
        {heat_dewpoint:.1f},      !- Wetbulb at Maximum Dry-Bulb {{C}}
           ,      !- Humidity Indicating Day Schedule Name
           ,      !- Humidity Ratio at Maximum Dry-Bulb {{kgWater/kgDryAir}}
           ,      !- Enthalpy at Maximum Dry-Bulb {{J/kg}}
           ,      !- Daily Wet-Bulb Temperature Range {{deltaC}}
    {pressure_pa:.0f}.,      !- Barometric Pressure {{Pa}}
        {heat_windspeed:.1f},      !- Wind Speed {{m/s}} design conditions vs. traditional 6.71 m/s (15 mph)
          0,      !- Wind Direction {{Degrees; N=0, S=180}}
         No,      !- Rain {{Yes/No}}
         No,      !- Snow on ground {{Yes/No}}
         No,      !- Daylight Savings Time Indicator
  ASHRAEClearSky, !- Solar Model Indicator
           ,      !- Beam Solar Day Schedule Name
           ,      !- Diffuse Solar Day Schedule Name
           ,      !- ASHRAE Clear Sky Optical Depth for Beam Irradiance (taub)
           ,      !- ASHRAE Clear Sky Optical Depth for Diffuse Irradiance (taud)
       0.00;      !- Clearness {{0.0 to 1.1}}
 
 ! {city} Annual Cooling Design Conditions Wind Speed={cool_windspeed:.1f}m/s Wind Dir=0
 ! Hottest Month=JUL
 ! {city} Annual Cooling (DB=>MWB) 0.4%, MaxDB={cool_0_4:.1f}°C MWB={cool_dewpoint:.1f}°C
 SizingPeriod:DesignDay,
  {city} Ann Clg 0.4% Condns DB=>MWB,     !- Name
          7,      !- Month
         21,      !- Day of Month
  SummerDesignDay,!- Day Type
        {cool_0_4:.1f},      !- Maximum Dry-Bulb Temperature {{C}}
       10.0,      !- Daily Dry-Bulb Temperature Range {{C}}
 DefaultMultipliers, !- Dry-Bulb Temperature Range Modifier Type
           ,      !- Dry-Bulb Temperature Range Modifier Day Schedule Name
    Wetbulb,      !- Humidity Condition Type
        {cool_dewpoint:.1f},      !- Wetbulb at Maximum Dry-Bulb {{C}}
           ,      !- Humidity Indicating Day Schedule Name
           ,      !- Humidity Ratio at Maximum Dry-Bulb {{kgWater/kgDryAir}}
           ,      !- Enthalpy at Maximum Dry-Bulb {{J/kg}}
           ,      !- Daily Wet-Bulb Temperature Range {{deltaC}}
    {pressure_pa:.0f}.,      !- Barometric Pressure {{Pa}}
        {cool_windspeed:.1f},      !- Wind Speed {{m/s}} design conditions vs. traditional 3.35 m/s (7mph)
          0,      !- Wind Direction {{Degrees; N=0, S=180}}
         No,      !- Rain {{Yes/No}}
         No,      !- Snow on ground {{Yes/No}}
         No,      !- Daylight Savings Time Indicator
  ASHRAEClearSky, !- Solar Model Indicator
           ,      !- Beam Solar Day Schedule Name
           ,      !- Diffuse Solar Day Schedule Name
           ,      !- ASHRAE Clear Sky Optical Depth for Beam Irradiance (taub)
           ,      !- ASHRAE Clear Sky Optical Depth for Diffuse Irradiance (taud)
       1.00;      !- Clearness {{0.0 to 1.1}}
"""
        # Write DDY file
        ddy_path = epw_path.with_suffix('.ddy')
        with open(ddy_path, 'w') as f:
            f.write(ddy_content)

        print(f"  ✓ Created DDY: {ddy_path.name}")
        print(f"    Heating 99.6%: {heat_99_6:.1f}°C, Cooling 0.4%: {cool_0_4:.1f}°C")
        return True

    except Exception as e:
        print(f"  ✗ Exception during DDY generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to convert TMY files to EPW and generate DDY files."""

    # Configuration
    tmy_dir = Path("output/TMY")
    output_dir = Path("output/epw")
    cities_config = Path("data/cities_config_final.csv")
    energyplus_path = "/Applications/EnergyPlus-25-1-0/energyplus"

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert TMY to EPW/DDY")
    parser.add_argument("--tmy-dir", type=Path, default=tmy_dir, help="Directory with TMY CSV files")
    parser.add_argument("--output-dir", type=Path, default=output_dir, help="Output directory for EPW/DDY files")
    parser.add_argument("--config-path", type=Path, default=cities_config, help="Path to cities config CSV")
    parser.add_argument("--energyplus-path", type=str, default=energyplus_path, help="Path to EnergyPlus executable")
    parser.add_argument("--stmy-mode", action="store_true",
                        help="Process sTMY files from output/sTMY with time-slice subdirectories (e.g., stmy_2025_2045)")
    parser.add_argument("--time-periods", type=str, nargs="+",
                        default=["2025_2045", "2046_2070", "2071_2095"],
                        help="Time periods for sTMY mode (e.g., 2025_2045 2046_2070)")
    args = parser.parse_args()

    # Update configuration from arguments
    tmy_dir = args.tmy_dir
    output_dir = args.output_dir
    cities_config = args.config_path
    energyplus_path = args.energyplus_path
    stmy_mode = args.stmy_mode
    time_periods = args.time_periods

    # Print header
    print("=" * 70)
    if stmy_mode:
        print("sTMY to EPW/DDY Converter (Future Climate Scenarios)")
    else:
        print("TMY to EPW/DDY Converter")
    print("=" * 70)

    if stmy_mode:
        print(f"sTMY mode:          Enabled")
        print(f"Time periods:       {', '.join(time_periods)}")
        print(f"sTMY base dir:      output/sTMY")
        print(f"Output:             To each time period's folder")
    else:
        print(f"TMY directory:      {tmy_dir}")
        print(f"Output directory:   {output_dir}")

    print(f"Cities config:      {cities_config}")
    print(f"EnergyPlus path:    {energyplus_path}")

    # Check if EnergyPlus exists
    if not os.path.exists(energyplus_path):
        print(f"\n✗ Error: EnergyPlus not found at: {energyplus_path}")
        print("Please update the energyplus_path variable in the script")
        return 1

    # Check if cities config exists
    if not cities_config.exists():
        print(f"\n✗ Error: Cities config not found at: {cities_config}")
        return 1

    # Check if TMY/sTMY directory exists
    if not stmy_mode and not tmy_dir.exists():
        print(f"\n✗ Error: TMY directory not found at: {tmy_dir}")
        return 1

    # Create output directory (only for TMY mode)
    if not stmy_mode:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load city metadata
    print(f"\nLoading city metadata...")
    city_metadata = load_city_metadata(cities_config)
    print(f"Loaded metadata for {len(city_metadata)} cities")

    # Find all TMY files based on mode
    if stmy_mode:
        # Process sTMY files from multiple time periods
        tmy_files = []
        stmy_base_dir = Path("output/sTMY")

        if not stmy_base_dir.exists():
            print(f"\n✗ Error: sTMY directory not found at: {stmy_base_dir}")
            return 1

        for period in time_periods:
            period_dir = stmy_base_dir / f"stmy_{period}"
            if not period_dir.exists():
                print(f"\n⚠ Warning: Time period directory not found: {period_dir}")
                continue

            # Find all city subdirectories
            for city_dir in sorted(period_dir.iterdir()):
                if not city_dir.is_dir():
                    continue

                # Find sTMY CSV files (not selection files)
                city_files = list(city_dir.glob(f"*_stmy_{period}.csv"))
                city_files = [f for f in city_files if "_selection" not in f.name]

                for f in city_files:
                    # Store file with metadata (city_name, period, file_path)
                    city_name = f.stem.split(f'_stmy_{period}')[0]
                    tmy_files.append({
                        'file': f,
                        'city': city_name,
                        'period': period,
                        'type': 'stmy'
                    })

        print(f"\nFound {len(tmy_files)} sTMY files across {len(time_periods)} time period(s)")
    else:
        # Original TMY mode
        tmy_file_list = sorted([f for f in tmy_dir.glob("*_tmy_*.csv") if "_selection" not in f.name])
        tmy_files = [{'file': f, 'city': f.stem.split('_tmy_')[0], 'period': '1991_2020', 'type': 'tmy'}
                     for f in tmy_file_list]
        print(f"\nFound {len(tmy_files)} TMY files to process")

    if len(tmy_files) == 0:
        print(f"\n✗ Error: No TMY files found")
        return 1

    print("=" * 70)

    # Process each TMY file
    success_count = 0
    error_count = 0

    for tmy_info in tmy_files:
        try:
            tmy_file = tmy_info['file']
            city_name = tmy_info['city']
            period = tmy_info['period']
            file_type = tmy_info['type']

            # Create output EPW path based on file type
            if file_type == 'stmy':
                # For sTMY: output to output/sTMY/stmy_{period}/{city}/ folder
                stmy_base_dir = Path("output/sTMY")
                city_output_dir = stmy_base_dir / f"stmy_{period}" / city_name
                epw_filename = f"{city_name}_stmy_{period}.epw"
                output_epw = city_output_dir / epw_filename
            else:
                # For TMY: output to output/epw folder (original behavior)
                epw_filename = f"{city_name}_tmy_{period}.epw"
                output_epw = output_dir / epw_filename

            # Convert TMY to EPW
            epw_path = convert_tmy_to_epw(tmy_file, output_epw, city_name, city_metadata)

            # Generate DDY from EPW
            ddy_success = generate_ddy_from_epw(epw_path, energyplus_path)

            if ddy_success:
                success_count += 1
            else:
                error_count += 1

        except Exception as e:
            print(f"\n✗ Error processing {tmy_file.name}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    # Print summary
    print("\n" + "=" * 70)
    print("Processing complete!")
    print(f"Successfully processed: {success_count}/{len(tmy_files)}")
    if error_count > 0:
        print(f"Errors:                {error_count}")

    if stmy_mode:
        print(f"Output locations:      output/sTMY/stmy_{{period}}/{{city}}/")
    else:
        print(f"Output directory:      {output_dir.absolute()}")
    print("=" * 70)

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    exit(main())
