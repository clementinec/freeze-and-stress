#!/usr/bin/env python3
"""
Apply Climate Delta from Any GCM-RCM Variant

This script takes the saved RS-VAR output (baseline + anomaly) and applies
climate delta from a specified CORDEX_CMIP5 GCM-RCM variant.

INPUTS:
    1. RS-VAR output Excel/CSV (from run_v5_save_rsvar.py)
    2. CORDEX_CMIP5 CSV for any GCM-RCM variant

OUTPUT:
    - New Excel file with sheets:
        - "Output": Final forecast with climate delta applied
        - "Climate_Delta_{GCM-RCM}": The applied climate delta
        - "Metadata": Configuration and delta statistics

USAGE:
    python apply_climate_delta.py \
        --rsvar output/rsvar_output.xlsx \
        --cordex path/to/new_gcm_rcm.csv \
        --gcm-rcm "MPI-ESM-LR_REMO2015" \
        --output output/forecast_MPI-ESM-LR_REMO2015.xlsx

CLIMATE DELTA CALCULATION:
    For each date in the forecast period:
        delta(date) = CORDEX_CMIP5(date) - CORDEX_baseline(day_of_year)

    Where CORDEX_baseline is the 1991-2010 climatology (DOY means).
    This is consistent with the baseline computation in the main V5 pipeline.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any
import yaml

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR

# Load scenarios from config.yaml
CONFIG_PATH = REPO_ROOT / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    _config = yaml.safe_load(f)
    SCENARIOS = _config['scenarios']
    # Convert relative paths to absolute paths
    for scenario_name, scenario_config in SCENARIOS.items():
        for key in ['variant_root', 'cmip5_base_path', 'cmip6_base_path', 'cordex_base_path']:
            if key in scenario_config and scenario_config[key]:
                scenario_config[key] = str(REPO_ROOT / scenario_config[key])

# Variable transformation types (consistent with V5 pipeline)
ADDITIVE_VARS = {'temp', 'pressure', 'dewpoint'}
MULTIPLICATIVE_VARS = {'wind_speed', 'GHI', 'DNI', 'DHI', 'relative_humidity', 'specific_humidity'}

# Mapping from forecast vars to CORDEX_CMIP5 vars
VAR_TO_CORDEX = {
    'temp': 'tas_C',
    'pressure': 'ps_hPa',
    'wind_speed': 'sfcWind',
    'GHI': 'rsds',
    'relative_humidity': 'hurs',
    'specific_humidity': 'huss',
}

# Baseline period (must match RS-VAR training)
BASELINE_START = 1991
BASELINE_END = 2010

# Default paths
CITIES_CONFIG_PATH = REPO_ROOT / "data" / "cities_config.csv"
DEFAULT_RSVAR_DIR = REPO_ROOT / "output" / "RSVAR"
DEFAULT_OUT_DIR = REPO_ROOT / "output" / "RSVAR_cd"
DEFAULT_OUT_DIR_MONTHLY = REPO_ROOT / "output" / "RSVAR_cd_monthly"


def _load_cities_config(path: Path) -> pd.DataFrame:
    """Load cities_config.csv and return a DataFrame with at least city + latitude."""
    if not path.exists():
        raise SystemExit(f"ERROR: cities config not found: {path}")

    df = pd.read_csv(path)

    required = {"city", "latitude"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            "ERROR: cities_config.csv missing required columns: "
            f"{missing}. Found columns: {list(df.columns)}"
        )

    df = df.copy()
    df["city"] = df["city"].astype(str)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")

    bad = df[df["latitude"].isna()]["city"].tolist()
    if bad:
        raise SystemExit(f"ERROR: Invalid latitude for cities: {bad}")

    df = df.drop_duplicates(subset=["city"], keep="first")
    return df


def _normalize_city_args(requested: list[str] | None, available_cities: list[str], config_path: Path) -> list[str]:
    """Resolve requested city list (case-sensitive) and validate."""
    if not requested:
        return sorted(available_cities)

    avail_set = set(available_cities)
    missing = [c for c in requested if c not in avail_set]
    if missing:
        avail = ", ".join(sorted(avail_set))
        raise SystemExit(
            f"ERROR: City(ies) not found in {config_path}: {missing}.\n"
            f"Available cities: {avail}"
        )
    return requested


def _slug_city(city: str) -> str:
    return city.replace(' ', '_')


def _load_isd_baseline_mean(isd_path: Path, baseline_start: int, baseline_end: int) -> float:
    """
    Load ISD baseline mean temperature using same preprocessing as RS-VAR pipeline.

    This ensures consistency with run_v5_save_rsvar_batch.py and evalute_samy.py
    by using ForecastTo2100V5.load_isd_data() which includes:
    - Removing duplicate timestamps
    - Forward/backward fill (limit=6 hours)
    - dropna() to remove remaining missing data
    - Humidity variable calculations

    Returns:
        Mean temperature (°C) over baseline period, or np.nan if unavailable
    """
    if not isd_path or not isd_path.exists():
        print(f"[WARNING] ISD path not provided or does not exist: {isd_path}")
        return np.nan

    try:
        print(f"[DEBUG] Loading ISD baseline from: {isd_path}")
        # Use ForecastTo2100V5 to ensure consistent preprocessing
        from HGScripts_Dev.forecast_to_2100_v5_fixed_baseline import ForecastTo2100V5

        temp_pipeline = ForecastTo2100V5(
            isd_solar_path=str(isd_path),
            cordex_rcp85_path=os.devnull,  # Not used for baseline calculation
            lat=0.0,  # Not critical for baseline calculation
            baseline_start=baseline_start,
            baseline_end=baseline_end,
        )
        temp_pipeline.load_isd_data()

        if temp_pipeline.isd_baseline is not None and not temp_pipeline.isd_baseline.empty:
            if 'temp' in temp_pipeline.isd_baseline.columns:
                isd_baseline_mean = temp_pipeline.isd_baseline['temp'].mean()
                print(f"[DEBUG] Found {len(temp_pipeline.isd_baseline)} baseline records (after preprocessing)")
                return isd_baseline_mean
            else:
                print(f"[WARNING] 'temp' column not found in ISD baseline")
        else:
            print(f"[WARNING] No baseline data found in ISD file for {baseline_start}-{baseline_end}")
    except Exception as e:
        print(f"[WARNING] Could not load ISD baseline: {e}")

    return np.nan


def _print_delta_statistics(delta_stats: dict[str, list]):
    """Print climate delta statistics in a consistent format."""
    print(f"\nClimate delta statistics:")
    for var, values in delta_stats.items():
        if len(values) > 0:
            values = np.array(values)
            if var in ADDITIVE_VARS:
                print(f"  {var}: mean={np.mean(values):+.2f}, range=[{np.min(values):+.2f}, {np.max(values):+.2f}]")
            else:
                print(f"  {var}: mean={np.mean(values):.3f}x, range=[{np.min(values):.3f}x, {np.max(values):.3f}x]")


def _print_temperature_statistics_by_window(
    rsvar_df: pd.DataFrame,
    cordex_df: pd.DataFrame,
    baseline: dict,
    output_df: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
    isd_path: Path = None
):
    """Print temperature statistics for different time windows.

    Prints for each window (2026-2040, 2041-2060, 2061-2080, 2081-2100):
    - ISD baseline mean temp (from baseline period)
    - Weather projection warming signal (CORDEX - baseline)
    - RS-VAR warming signal (RSVAR output - baseline)
    - RS-VAR mean temp (final output)
    """
    time_windows = [
        (2026, 2040),
        (2041, 2060),
        (2061, 2080),
        (2081, 2100)
    ]

    print("\n" + "="*80)
    print("TEMPERATURE STATISTICS BY TIME WINDOW")
    print("="*80)

    # Load ISD baseline using same method as RS-VAR pipeline for consistency
    isd_baseline_mean = _load_isd_baseline_mean(isd_path, baseline_start, baseline_end)

    print(f"\nISD Baseline ({baseline_start}-{baseline_end}) Mean Temperature: {isd_baseline_mean:.2f}°C")

    # Check if CORDEX has temperature data
    cordex_daily = _cordex_daily_view(cordex_df)
    cordex_temp_col = 'tas_C' if 'tas_C' in cordex_daily.columns else None

    if cordex_temp_col is None:
        print("[WARNING] Temperature column not found in CORDEX data")

    # Get CORDEX baseline
    cordex_baseline_mean = np.nan
    if 'tas_C' in baseline and 'overall' in baseline['tas_C']:
        cordex_baseline_mean = baseline['tas_C']['overall']

    print(f"CORDEX Baseline ({baseline_start}-{baseline_end}) Mean Temperature: {cordex_baseline_mean:.2f}°C")

    for start_year, end_year in time_windows:
        print(f"\n{'-'*80}")
        print(f"Time Window: {start_year}-{end_year}")
        print(f"{'-'*80}")

        # RS-VAR future period
        rsvar_mask = (rsvar_df.index.year >= start_year) & (rsvar_df.index.year <= end_year)
        if rsvar_mask.sum() > 0:
            rsvar_period_mean = rsvar_df.loc[rsvar_mask, 'temp'].mean()
            rsvar_warming = rsvar_period_mean - isd_baseline_mean
        else:
            rsvar_period_mean = np.nan
            rsvar_warming = np.nan

        # CORDEX future period
        if cordex_temp_col:
            cordex_mask = (cordex_daily.index.year >= start_year) & (cordex_daily.index.year <= end_year)
            if cordex_mask.sum() > 0:
                cordex_period_mean = cordex_daily.loc[cordex_mask, cordex_temp_col].mean()
                cordex_warming = cordex_period_mean - cordex_baseline_mean
            else:
                cordex_period_mean = np.nan
                cordex_warming = np.nan
        else:
            cordex_period_mean = np.nan
            cordex_warming = np.nan

        # Final output (RS-VAR with climate delta applied)
        output_mask = (output_df.index.year >= start_year) & (output_df.index.year <= end_year)
        if output_mask.sum() > 0 and 'temp' in output_df.columns:
            output_period_mean = output_df.loc[output_mask, 'temp'].mean()
            output_warming = output_period_mean - isd_baseline_mean
        else:
            output_period_mean = np.nan
            output_warming = np.nan

        print(f"  ISD Baseline Mean:                {isd_baseline_mean:.2f}°C")
        print(f"  CORDEX Period Mean:               {cordex_period_mean:.2f}°C")
        print(f"  CORDEX Warming Signal:            {cordex_warming:+.2f}°C")
        print(f"  RS-VAR (before delta) Mean:       {rsvar_period_mean:.2f}°C")
        print(f"  RS-VAR (before delta) Warming:    {rsvar_warming:+.2f}°C")
        print(f"  RS-VAR (after delta) Mean:        {output_period_mean:.2f}°C")
        print(f"  RS-VAR (after delta) Warming:     {output_warming:+.2f}°C")


def _print_temperature_statistics_by_window_mixed(
    rsvar_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    baseline: dict,
    output_df: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
    isd_path: Path = None
):
    """Print temperature statistics for mixed resolution data.

    Similar to _print_temperature_statistics_by_window but for mixed daily/monthly data.
    """
    time_windows = [
        (2026, 2040),
        (2041, 2060),
        (2061, 2080),
        (2081, 2100)
    ]

    print("\n" + "="*80)
    print("TEMPERATURE STATISTICS BY TIME WINDOW")
    print("="*80)

    # Load ISD baseline using same method as RS-VAR pipeline for consistency
    isd_baseline_mean = _load_isd_baseline_mean(isd_path, baseline_start, baseline_end)

    print(f"\nISD Baseline ({baseline_start}-{baseline_end}) Mean Temperature: {isd_baseline_mean:.2f}°C")

    # Check if daily data has temperature
    daily_df = _apply_unit_conversions(daily_df)
    daily_view = _cordex_daily_view(daily_df)
    cordex_temp_col = 'tas_C' if 'tas_C' in daily_view.columns else None

    if cordex_temp_col is None:
        print("[WARNING] Temperature column not found in climate data")

    # Get climate baseline
    climate_baseline_mean = np.nan
    if 'tas_C' in baseline and 'overall' in baseline['tas_C']:
        climate_baseline_mean = baseline['tas_C']['overall']

    print(f"Climate Baseline ({baseline_start}-{baseline_end}) Mean Temperature: {climate_baseline_mean:.2f}°C")

    for start_year, end_year in time_windows:
        print(f"\n{'-'*80}")
        print(f"Time Window: {start_year}-{end_year}")
        print(f"{'-'*80}")

        # RS-VAR future period
        rsvar_mask = (rsvar_df.index.year >= start_year) & (rsvar_df.index.year <= end_year)
        if rsvar_mask.sum() > 0:
            rsvar_period_mean = rsvar_df.loc[rsvar_mask, 'temp'].mean()
            rsvar_warming = rsvar_period_mean - isd_baseline_mean
        else:
            rsvar_period_mean = np.nan
            rsvar_warming = np.nan

        # Climate future period
        if cordex_temp_col:
            climate_mask = (daily_view.index.year >= start_year) & (daily_view.index.year <= end_year)
            if climate_mask.sum() > 0:
                climate_period_mean = daily_view.loc[climate_mask, cordex_temp_col].mean()
                climate_warming = climate_period_mean - climate_baseline_mean
            else:
                climate_period_mean = np.nan
                climate_warming = np.nan
        else:
            climate_period_mean = np.nan
            climate_warming = np.nan

        # Final output (RS-VAR with climate delta applied)
        output_mask = (output_df.index.year >= start_year) & (output_df.index.year <= end_year)
        if output_mask.sum() > 0 and 'temp' in output_df.columns:
            output_period_mean = output_df.loc[output_mask, 'temp'].mean()
            output_warming = output_period_mean - isd_baseline_mean
        else:
            output_period_mean = np.nan
            output_warming = np.nan

        print(f"  ISD Baseline Mean:                {isd_baseline_mean:.2f}°C")
        print(f"  Climate Period Mean:              {climate_period_mean:.2f}°C")
        print(f"  Climate Warming Signal:           {climate_warming:+.2f}°C")
        print(f"  RS-VAR (before delta) Mean:       {rsvar_period_mean:.2f}°C")
        print(f"  RS-VAR (before delta) Warming:    {rsvar_warming:+.2f}°C")
        print(f"  RS-VAR (after delta) Mean:        {output_period_mean:.2f}°C")
        print(f"  RS-VAR (after delta) Warming:     {output_warming:+.2f}°C")


def enforce_solar_constraints(forecast_df, lat):
    """
    Enforce physical constraints on solar radiation.

    Consistent with ForecastTo2100V5._enforce_solar_constraints().
    """
    if not all(v in forecast_df.columns for v in ['GHI', 'DNI', 'DHI']):
        return forecast_df

    lat_rad = np.radians(lat)

    hours = forecast_df.index.hour.values
    doy = forecast_df.index.dayofyear.values

    # Solar geometry
    declination = np.radians(23.45) * np.sin(2 * np.pi * (doy - 81) / 365)
    hour_angle = (hours - 12) * 15
    hour_angle_rad = np.radians(hour_angle)

    cos_zenith = (np.sin(lat_rad) * np.sin(declination) +
                  np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle_rad))

    # Nighttime: zero all solar
    night_mask = cos_zenith <= 0.065
    forecast_df.loc[night_mask, 'GHI'] = 0.0
    forecast_df.loc[night_mask, 'DNI'] = 0.0
    forecast_df.loc[night_mask, 'DHI'] = 0.0

    # Non-negative
    forecast_df.loc[forecast_df['GHI'] < 0, 'GHI'] = 0.0
    forecast_df.loc[forecast_df['DNI'] < 0, 'DNI'] = 0.0
    forecast_df.loc[forecast_df['DHI'] < 0, 'DHI'] = 0.0

    # Daytime: reconcile with closure equation (preserve GHI)
    day_mask = ~night_mask
    if day_mask.sum() > 0:
        GHI = forecast_df.loc[day_mask, 'GHI'].values
        DNI = forecast_df.loc[day_mask, 'DNI'].values
        cos_z = cos_zenith[day_mask]

        # DHI = GHI - DNI * cos(zenith)
        DHI_reconciled = GHI - DNI * cos_z
        DHI_reconciled = np.maximum(DHI_reconciled, 0.0)
        forecast_df.loc[day_mask, 'DHI'] = DHI_reconciled

    return forecast_df


def load_rsvar_output(rsvar_path: Path):
    """Load RS-VAR output from Excel or CSV."""
    rsvar_path = Path(rsvar_path)

    print(f"[DEBUG] Loading RS-VAR from: {rsvar_path}")

    if rsvar_path.suffix == '.xlsx':
        # Try Excel first, fall back to CSV if openpyxl issues
        try:
            rsvar_df = pd.read_excel(rsvar_path, sheet_name='RSVAR_Output')
            try:
                metadata = pd.read_excel(rsvar_path, sheet_name='Metadata')
                metadata = dict(zip(metadata['parameter'], metadata['value']))
            except Exception:
                metadata = {}
        except Exception:
            # Any excel engine/import/read error: try CSV fallback
            csv_path = rsvar_path.with_suffix('.csv')
            if csv_path.exists():
                print(f"  (Using CSV fallback for RS-VAR)")
                rsvar_df = pd.read_csv(csv_path)
                metadata = {}
            else:
                raise
    else:
        rsvar_df = pd.read_csv(rsvar_path)
        metadata = {}

    # Parse datetime
    rsvar_df['datetime'] = pd.to_datetime(rsvar_df['datetime'])
    rsvar_df = rsvar_df.set_index('datetime')

    return rsvar_df, metadata


def load_and_merge_cordex_city_variant(variant_root: Path, city: str) -> tuple[pd.DataFrame, dict]:
    """Load and concat historical+rcp85 for a given city under a given variant_root.

    Expects:
      {variant_root}/{CitySlug}/historical.csv
      {variant_root}/{CitySlug}/rcp85.csv

    Returns merged daily dataframe indexed by 'time'.
    """
    city_dir = Path(variant_root) / _slug_city(city)
    hist_path = city_dir / "historical.csv"
    rcp_path = city_dir / "rcp85.csv"

    print(f"[DEBUG] Loading CORDEX historical from: {hist_path}")
    print(f"[DEBUG] Loading CORDEX rcp85 from: {rcp_path}")

    if not hist_path.exists():
        raise FileNotFoundError(str(hist_path))
    if not rcp_path.exists():
        raise FileNotFoundError(str(rcp_path))

    hist = pd.read_csv(hist_path, parse_dates=['time']).set_index('time')
    rcp = pd.read_csv(rcp_path, parse_dates=['time']).set_index('time')

    merged = pd.concat([hist, rcp], axis=0)
    merged = merged[~merged.index.duplicated(keep='first')]
    merged = merged.sort_index()

    # Apply unit conversions
    # IMPORTANT: this must happen on the full CORDEX dataframe, not only inside
    # compute_cordex_baseline(), because compute_daily_delta() reads from `cordex`.
    merged = _apply_unit_conversions(merged)

    meta = {
        'historical_path': str(hist_path),
        'rcp85_path': str(rcp_path),
        'variant_root': str(variant_root),
    }

    return merged, meta


def _apply_unit_conversions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard unit conversions to climate data."""
    for col in ['tas', 'tasmax', 'tasmin']:
        if col in df.columns and f"{col}_C" not in df.columns:
            df[f"{col}_C"] = pd.to_numeric(df[col], errors='coerce') - 273.15
    # ps (surface pressure) or psl (sea level pressure) -> hPa
    if 'ps' in df.columns and 'ps_hPa' not in df.columns:
        df['ps_hPa'] = pd.to_numeric(df['ps'], errors='coerce') / 100.0
    if 'psl' in df.columns and 'ps_hPa' not in df.columns:
        df['ps_hPa'] = pd.to_numeric(df['psl'], errors='coerce') / 100.0
    # surface_air_pressure (CNRM format) -> ps_hPa
    if 'surface_air_pressure' in df.columns and 'ps_hPa' not in df.columns:
        df['ps_hPa'] = pd.to_numeric(df['surface_air_pressure'], errors='coerce') / 100.0
    # near_surface_relative_humidity (CNRM format) -> hurs
    if 'near_surface_relative_humidity' in df.columns and 'hurs' not in df.columns:
        df['hurs'] = pd.to_numeric(df['near_surface_relative_humidity'], errors='coerce')
    return df


def _load_and_merge_climate_data(file_pairs: list[tuple[Path, Path]], required: bool = True) -> tuple[pd.DataFrame, str]:
    """Load and merge historical + future climate data from file pairs.

    Args:
        file_pairs: List of (hist_path, future_path) tuples to try
        required: If True, raise error if no files found

    Returns:
        (merged_df, model_name): Merged DataFrame indexed by time, and detected model name
    """
    for hist_path, future_path in file_pairs:
        if hist_path.exists() and future_path.exists():
            hist = pd.read_csv(hist_path, parse_dates=['time']).set_index('time')
            future = pd.read_csv(future_path, parse_dates=['time']).set_index('time')
            merged = pd.concat([hist, future], axis=0)
            merged = merged[~merged.index.duplicated(keep='first')].sort_index()

            # Detect model name from filename
            hist_name = hist_path.stem
            if 'cnrm_esm2_1' in hist_name:
                model_name = 'CNRM-ESM2-1'
            elif 'mpi_esm1_2_lr' in hist_name:
                model_name = 'MPI-ESM1-2-LR'
            elif 'mpi_esm_lr' in hist_name:
                model_name = 'MPI-ESM-LR (CMIP5)'
            elif '_day_' in hist_name:
                model_name = 'CORDEX-CMIP6'
            else:
                model_name = 'Unknown'

            return _apply_unit_conversions(merged), model_name

    if required:
        paths_tried = [f"{h} + {f}" for h, f in file_pairs]
        raise FileNotFoundError(f"No valid file pairs found. Tried: {paths_tried}")
    return pd.DataFrame(), "None"


def load_cmip6_or_cordex6(base_path: Path, city: str, ssp: str, scenario_id: str = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load CMIP5, CMIP6 or CORDEX-CMIP6 data with auto-detection of file format.

    Supports multiple formats:
    - CMIP5: mpi_esm_lr_historical.csv + mpi_esm_lr_rcp_8_5.csv
    - CMIP6 MPI: mpi_esm1_2_lr_historical.csv + mpi_esm1_2_lr_{ssp}.csv
    - CMIP6 CNRM: cnrm_esm2_1_historical.csv + cnrm_esm2_1_{ssp}.csv
    - CORDEX_CMIP6: {City}_historical_day_*.csv + {City}_{ssp}_day_*.csv

    Args:
        base_path: Base path to data directory
        city: City name (e.g., 'Los Angeles')
        ssp: SSP scenario (e.g., 'ssp245', 'ssp370', 'rcp85')
        scenario_id: Full scenario identifier (e.g., 'CMIP6_CNRM_ssp245') for model priority detection

    Returns:
        (daily_df, monthly_df, meta_dict)
    """
    city_slug = _slug_city(city)
    city_dir = Path(base_path) / city_slug

    # Determine future scenario file suffix based on SSP
    # For CMIP5: rcp85 -> rcp_8_5
    # For CMIP6: ssp245 -> ssp245
    if ssp.startswith('rcp'):
        # CMIP5 format: rcp85 -> rcp_8_5
        future_suffix = ssp.replace('rcp', 'rcp_').replace('85', '8_5').replace('45', '4_5').replace('26', '2_6')
    else:
        # CMIP6 format: keep as-is
        future_suffix = ssp

    # Try CMIP5 format: mpi_esm_lr_{experiment}.csv
    cmip5_daily_pairs = [(
        city_dir / "mpi_esm_lr_historical.csv",
        city_dir / f"mpi_esm_lr_{future_suffix}.csv"
    )]
    cmip5_monthly_pairs = [(
        city_dir / "mpi_esm_lr_historical_monthly.csv",
        city_dir / f"mpi_esm_lr_{future_suffix}_monthly.csv"
    )]

    # Try CMIP6 format: mpi_esm1_2_lr_{experiment}.csv
    cmip6_daily_pairs = [(
        city_dir / "mpi_esm1_2_lr_historical.csv",
        city_dir / f"mpi_esm1_2_lr_{ssp}.csv"
    )]
    cmip6_monthly_pairs = [(
        city_dir / "mpi_esm1_2_lr_historical_monthly.csv",
        city_dir / f"mpi_esm1_2_lr_{ssp}_monthly.csv"
    )]

    # Try CMIP6 CNRM format: cnrm_esm2_1_{experiment}.csv
    cnrm_daily_pairs = [(
        city_dir / "cnrm_esm2_1_historical.csv",
        city_dir / f"cnrm_esm2_1_{ssp}.csv"
    )]
    cnrm_monthly_pairs = [(
        city_dir / "cnrm_esm2_1_historical_monthly.csv",
        city_dir / f"cnrm_esm2_1_{ssp}_monthly.csv"
    )]

    # Try CORDEX_CMIP6 format: {City}_{experiment}_day_*.csv
    cordex6_daily_pairs = [(
        city_dir / f"{city_slug}_historical_day_19500101-20141231.csv",
        city_dir / f"{city_slug}_{ssp}_day_20150101-21001231.csv"
    )]
    cordex6_monthly_pairs = [(
        city_dir / f"{city_slug}_historical_mon_195001-201412.csv",
        city_dir / f"{city_slug}_{ssp}_mon_201501-210012.csv"
    )]

    # Determine loading priority based on scenario_id
    # If scenario contains "CNRM", prioritize CNRM model files
    if scenario_id and "CNRM" in scenario_id:
        # Priority: CNRM -> CMIP6(MPI) -> CMIP5 -> CORDEX_CMIP6
        daily_pairs = cnrm_daily_pairs + cmip6_daily_pairs + cmip5_daily_pairs + cordex6_daily_pairs
        monthly_pairs = cnrm_monthly_pairs + cmip6_monthly_pairs + cmip5_monthly_pairs + cordex6_monthly_pairs
        print(f"  [Model Priority] Using CNRM model priority for scenario: {scenario_id}")
    else:
        # Default priority: CMIP5 -> CMIP6(MPI) -> CNRM -> CORDEX_CMIP6
        daily_pairs = cmip5_daily_pairs + cmip6_daily_pairs + cnrm_daily_pairs + cordex6_daily_pairs
        monthly_pairs = cmip5_monthly_pairs + cmip6_monthly_pairs + cnrm_monthly_pairs + cordex6_monthly_pairs

    # Try loading daily data (required)
    daily_df, model_name = _load_and_merge_climate_data(daily_pairs, required=True)
    print(f"  [Model Detected] Loaded {model_name} for {city}/{scenario_id or ssp}")

    # Try loading monthly data (optional)
    monthly_df, _ = _load_and_merge_climate_data(monthly_pairs, required=False)

    meta = {
        'base_path': str(base_path),
        'city': city,
        'ssp': ssp,
        'model_name': model_name,
    }

    return daily_df, monthly_df, meta


def _cordex_daily_view(cordex: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row-per-day CORDEX dataframe.

    CORDEX files are often daily but timestamped at 12:00 (or may contain
    duplicates). We avoid brittle `index.date == ...` + iloc[0] selection by
    collapsing to daily means.
    """
    if cordex.empty:
        return cordex

    c = cordex.copy()

    # Ensure a monotonically increasing index for resample safety.
    if not c.index.is_monotonic_increasing:
        c = c.sort_index()

    # If there are multiple rows per day, take the mean (numeric only).
    # This is safe for daily-mean variables and will also handle 12:00 stamps.
    daily = c.resample('D').mean(numeric_only=True)
    return daily


def _cordex_monthly_view(cordex: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row-per-month CORDEX dataframe (month-start index)."""
    if cordex.empty:
        return cordex

    c = cordex.copy()

    if not c.index.is_monotonic_increasing:
        c = c.sort_index()

    monthly = c.resample('MS').mean(numeric_only=True)
    return monthly


def build_effective_monthly_df(daily_df: pd.DataFrame, monthly_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a monthly dataframe for monthly-delta mode.

    Preference order:
    1) Start from daily data resampled to monthly (ensures tas exists).
    2) Overlay/replace with actual monthly data where available.

    Returns:
        (combined_monthly_df, missing_in_monthly, monthly_only_cols)
    """
    # Ensure conversions are applied (tas_C, ps_hPa, etc.)
    daily_df = _apply_unit_conversions(daily_df.copy())
    monthly_df = _apply_unit_conversions(monthly_df.copy())

    monthly_from_daily = _cordex_monthly_view(daily_df)

    if monthly_df.empty:
        missing_in_monthly = sorted(set(monthly_from_daily.columns))
        return monthly_from_daily, missing_in_monthly, []

    monthly_df_norm = _cordex_monthly_view(monthly_df)

    # Prefer true monthly values; fill gaps with resampled daily data
    combined = monthly_df_norm.combine_first(monthly_from_daily)

    missing_in_monthly = sorted(set(monthly_from_daily.columns) - set(monthly_df_norm.columns))
    monthly_only_cols = sorted(set(monthly_df_norm.columns) - set(monthly_from_daily.columns))
    return combined, missing_in_monthly, monthly_only_cols


def compute_cordex_baseline(cordex: pd.DataFrame, baseline_start: int, baseline_end: int, mode: str = "daily") -> dict[str, dict[str, Any]]:
    """Compute baseline climatology from merged CORDEX data.

    Args:
        cordex: merged daily CORDEX dataframe
        baseline_start: start year
        baseline_end: end year
        mode: "daily" (day-of-year) or "monthly" (month-of-year)
    """
    cordex = cordex.copy()

    if mode not in {"daily", "monthly"}:
        raise ValueError(f"Invalid baseline mode: {mode}. Expected 'daily' or 'monthly'.")

    # Unit conversions (consistent with V5 pipeline)
    cordex = _apply_unit_conversions(cordex)

    baseline_mask = (
        (cordex.index.year >= baseline_start) &
        (cordex.index.year <= baseline_end)
    )
    cordex_baseline_data = cordex.loc[baseline_mask]

    # Strict: baseline must come from the mandated period.
    if cordex_baseline_data.empty:
        raise ValueError(
            f"CORDEX baseline slice is empty for {baseline_start}-{baseline_end}. "
            f"CORDEX period is {cordex.index.min()} to {cordex.index.max()} (n={len(cordex):,}). "
            "Cannot compute baseline-period climatology, so climate delta for additive vars (tas/ps) will be missing."
        )

    baseline = {}
    baseline_vars = {
        'tas_C': 'additive',
        'tasmax_C': 'additive',
        'tasmin_C': 'additive',
        'ps_hPa': 'additive',
        'sfcWind': 'multiplicative',
        'rsds': 'multiplicative',
        'hurs': 'multiplicative',
        'huss': 'multiplicative',
    }

    for var, transform_type in baseline_vars.items():
        if var not in cordex_baseline_data.columns:
            continue

        if mode == "daily":
            # Day-of-year climatology; for daily data this is well-defined.
            doy_clim = cordex_baseline_data.groupby(cordex_baseline_data.index.dayofyear)[var].mean()
            baseline[var] = {
                'doy': doy_clim,
                'overall': cordex_baseline_data[var].mean(),
                'transform': transform_type
            }
        else:
            # Month-of-year climatology
            month_clim = cordex_baseline_data.groupby(cordex_baseline_data.index.month)[var].mean()
            baseline[var] = {
                'month': month_clim,
                'overall': cordex_baseline_data[var].mean(),
                'transform': transform_type
            }

    # Extra guardrails for the common failure mode you hit: temp/pressure missing.
    required = ['tas_C', 'ps_hPa']
    missing_required = [v for v in required if v not in baseline]
    if missing_required:
        # Provide actionable info: did we fail to create derived columns or are they absent in the baseline slice?
        available_cols = sorted(map(str, cordex_baseline_data.columns))
        raise ValueError(
            "Baseline computed, but required additive baseline variables are missing: "
            f"{missing_required}.\n"
            "This usually means the CORDEX files don't contain expected raw columns (tas/ps) "
            "or they are not numeric in the baseline period.\n"
            f"Baseline slice years: {cordex_baseline_data.index.min()} -> {cordex_baseline_data.index.max()} (n={len(cordex_baseline_data):,}).\n"
            f"Available columns in baseline slice: {available_cols}"
        )

    return baseline


def compute_mixed_resolution_baseline(
    daily_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
    var_resolution_config: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Compute baseline climatology for mixed daily/monthly resolution data.

    Args:
        daily_df: Daily resolution CORDEX data
        monthly_df: Monthly resolution CORDEX data
        baseline_start: Start year for baseline period
        baseline_end: End year for baseline period
        var_resolution_config: Dict mapping variable names to 'daily' or 'monthly'

    Returns:
        Dict with baseline info for each variable
    """
    baseline = {}

    # Make copies to avoid modifying original data
    daily_df = daily_df.copy()
    monthly_df = monthly_df.copy()

    # Unit conversions
    daily_df = _apply_unit_conversions(daily_df)
    monthly_df = _apply_unit_conversions(monthly_df)

    # Default variable configuration
    # NOTE: ps_hPa resolution depends on the dataset:
    #   - CMIP6: daily (converted from psl)
    #   - CORDEX_CMIP6: monthly (from ps)
    baseline_vars = {
        # Daily variables (temperature, wind)
        'tas_C': ('daily', 'additive'),
        'tasmax_C': ('daily', 'additive'),
        'tasmin_C': ('daily', 'additive'),
        'sfcWind': ('daily', 'multiplicative'),
        # Monthly variables (solar, humidity, pressure)
        'ps_hPa': ('monthly', 'additive'),  # Default to monthly, will auto-detect below
        'rsds': ('monthly', 'multiplicative'),
        'hurs': ('monthly', 'multiplicative'),
        'huss': ('monthly', 'multiplicative'),
    }

    # Auto-detect ps_hPa resolution based on actual data availability
    # (CMIP6 has it in daily, CORDEX_CMIP6 may have it in monthly)
    if 'ps_hPa' in daily_df.columns and 'ps_hPa' not in var_resolution_config:
        baseline_vars['ps_hPa'] = ('daily', 'additive')
    elif 'ps_hPa' in monthly_df.columns and 'ps_hPa' not in var_resolution_config:
        baseline_vars['ps_hPa'] = ('monthly', 'additive')

    # Override with user config if provided
    for var in var_resolution_config:
        if var in baseline_vars:
            resolution = var_resolution_config[var]
            transform = baseline_vars[var][1]  # Keep transform type
            baseline_vars[var] = (resolution, transform)

    # Compute daily baselines
    daily_mask = (
        (daily_df.index.year >= baseline_start) &
        (daily_df.index.year <= baseline_end)
    )
    daily_baseline_data = daily_df.loc[daily_mask]

    if daily_baseline_data.empty:
        raise ValueError(
            f"Daily baseline slice is empty for {baseline_start}-{baseline_end}. "
            f"Daily data period: {daily_df.index.min()} to {daily_df.index.max()}"
        )

    # Compute monthly baselines
    monthly_mask = (
        (monthly_df.index.year >= baseline_start) &
        (monthly_df.index.year <= baseline_end)
    )
    monthly_baseline_data = monthly_df.loc[monthly_mask]

    if monthly_baseline_data.empty:
        raise ValueError(
            f"Monthly baseline slice is empty for {baseline_start}-{baseline_end}. "
            f"Monthly data period: {monthly_df.index.min()} to {monthly_df.index.max()}"
        )

    # Process each variable
    for var, (resolution, transform_type) in baseline_vars.items():
        if resolution == 'daily':
            if var not in daily_baseline_data.columns:
                continue

            # Day-of-year climatology for daily data
            doy_clim = daily_baseline_data.groupby(daily_baseline_data.index.dayofyear)[var].mean()
            baseline[var] = {
                'resolution': 'daily',
                'doy': doy_clim,
                'overall': daily_baseline_data[var].mean(),
                'transform': transform_type
            }

        elif resolution == 'monthly':
            if var not in monthly_baseline_data.columns:
                continue

            # Month-of-year climatology for monthly data
            month_clim = monthly_baseline_data.groupby(monthly_baseline_data.index.month)[var].mean()
            baseline[var] = {
                'resolution': 'monthly',
                'month_clim': month_clim,
                'overall': monthly_baseline_data[var].mean(),
                'transform': transform_type
            }

    # Check required variables
    required = ['tas_C', 'ps_hPa']
    missing_required = [v for v in required if v not in baseline]
    if missing_required:
        available_daily = sorted(map(str, daily_baseline_data.columns))
        available_monthly = sorted(map(str, monthly_baseline_data.columns))
        raise ValueError(
            f"Required baseline variables are missing: {missing_required}.\n"
            f"Available daily columns: {available_daily}\n"
            f"Available monthly columns: {available_monthly}"
        )

    return baseline


def compute_daily_delta(cordex, baseline, target_date):
    """Compute climate delta for a specific date.

    For additive variables: delta = CORDEX_CMIP5(date) - CORDEX_baseline(doy)
    For multiplicative variables: ratio = CORDEX_CMIP5(date) / CORDEX_baseline(doy)

    This mirrors compute_climate_delta_daily() in ForecastTo2100V5.
    """
    deltas = {}

    # Use a robust daily view (1 row per day) instead of relying on exact
    # timestamps (CORDEX daily data is often at 12:00).
    cordex_daily = _cordex_daily_view(cordex)

    # Get CORDEX value for target date
    target_day = pd.Timestamp(target_date).normalize()
    if target_day not in cordex_daily.index:
        return deltas

    cordex_target = cordex_daily.loc[target_day]

    doy = target_day.timetuple().tm_yday

    for var, baseline_info in baseline.items():
        if var not in cordex_daily.columns:
            continue

        future_val = cordex_target[var]
        baseline_val = baseline_info['doy'].get(doy, baseline_info['overall'])

        if baseline_info['transform'] == 'additive':
            deltas[var] = future_val - baseline_val
        else:  # multiplicative
            if baseline_val > 0:
                deltas[var] = future_val / baseline_val
            else:
                deltas[var] = 1.0

    return deltas


def compute_monthly_delta(cordex_monthly, baseline, target_date):
    """Compute climate delta using monthly mean values.

    For additive variables: delta = CORDEX_monthly(target_month) - CORDEX_baseline(month)
    For multiplicative variables: ratio = CORDEX_monthly(target_month) / CORDEX_baseline(month)
    """
    deltas = {}

    if cordex_monthly.empty:
        return deltas

    target_day = pd.Timestamp(target_date).normalize()
    target_month_start = target_day.replace(day=1)

    if target_month_start not in cordex_monthly.index:
        return deltas

    cordex_target = cordex_monthly.loc[target_month_start]
    month = target_month_start.month

    for var, baseline_info in baseline.items():
        if var not in cordex_monthly.columns:
            continue

        future_val = cordex_target[var]
        baseline_val = baseline_info['month'].get(month, baseline_info['overall'])

        if baseline_info['transform'] == 'additive':
            deltas[var] = future_val - baseline_val
        else:  # multiplicative
            if baseline_val > 0:
                deltas[var] = future_val / baseline_val
            else:
                deltas[var] = 1.0

    return deltas


def compute_mixed_resolution_delta(daily_df, monthly_df, baseline, target_date):
    """Compute climate delta for mixed daily/monthly resolution data.

    For daily variables: delta = future(date) - baseline(doy)
    For monthly variables: delta = future(month) - baseline(month)

    Args:
        daily_df: Daily resolution CORDEX data
        monthly_df: Monthly resolution CORDEX data
        baseline: Baseline climatology computed by compute_mixed_resolution_baseline
        target_date: Target date for delta computation

    Returns:
        Dict of deltas for each variable
    """
    deltas = {}
    target_day = pd.Timestamp(target_date).normalize()
    doy = target_day.timetuple().tm_yday
    target_month = target_day.month

    for var, baseline_info in baseline.items():
        resolution = baseline_info.get('resolution', 'daily')
        transform = baseline_info['transform']

        if resolution == 'daily':
            # Daily variable: use daily data
            cordex_daily = _cordex_daily_view(daily_df)

            if target_day not in cordex_daily.index:
                continue
            if var not in cordex_daily.columns:
                continue

            future_val = cordex_daily.loc[target_day, var]
            baseline_val = baseline_info['doy'].get(doy, baseline_info['overall'])

        elif resolution == 'monthly':
            # Monthly variable: find the month containing target_date
            target_month_start = target_day.replace(day=1)

            # Find closest month in monthly_df
            if target_month_start not in monthly_df.index:
                # Try to find the month
                month_mask = (monthly_df.index.year == target_day.year) & (monthly_df.index.month == target_month)
                if month_mask.sum() == 0:
                    continue
                future_val = monthly_df.loc[month_mask, var].iloc[0]
            else:
                if var not in monthly_df.columns:
                    continue
                future_val = monthly_df.loc[target_month_start, var]

            baseline_val = baseline_info['month_clim'].get(target_month, baseline_info['overall'])

        else:
            continue

        # Apply transform
        if transform == 'additive':
            deltas[var] = future_val - baseline_val
        else:  # multiplicative
            if baseline_val > 0:
                deltas[var] = future_val / baseline_val
            else:
                deltas[var] = 1.0

    return deltas


def apply_climate_delta(rsvar_df, cordex, baseline, gcm: str, rcm: str | None, lat: float, baseline_start: int = BASELINE_START, baseline_end: int = BASELINE_END, isd_path: Path = None, debug: bool = False, delta_mode: str = "daily"):
    """
    Apply climate delta to RS-VAR output.

    For each date:
        output(date) = rsvar(date) + delta(date)  [additive vars]
        output(date) = rsvar(date) * ratio(date)  [multiplicative vars]

    Also enforces solar constraints after applying delta (consistent with V5 pipeline).
    """
    model_name = f"{gcm}_{rcm}" if rcm else gcm
    print("\n" + "="*80)
    print(f"APPLYING CLIMATE DELTA: {model_name}")
    print("="*80)

    output_df = rsvar_df.copy()

    if debug:
        print("\n[DEBUG] RS-VAR columns:")
        print("  ", sorted(map(str, output_df.columns.tolist())))
        print("[DEBUG] Baseline keys:")
        print("  ", sorted(map(str, baseline.keys())))
        needed = [VAR_TO_CORDEX['temp'], VAR_TO_CORDEX['pressure']]
        for k in needed:
            print(f"[DEBUG] baseline has {k}? {k in baseline}")

    if delta_mode not in {"daily", "monthly"}:
        raise ValueError(f"Invalid delta_mode: {delta_mode}. Expected 'daily' or 'monthly'.")

    # Get unique dates in forecast
    dates = pd.to_datetime(output_df.index.date).unique()
    mode_label = "daily" if delta_mode == "daily" else "monthly"
    print(f"\nComputing {mode_label} deltas for {len(dates):,} days...")

    # Precompute monthly view if needed
    cordex_monthly = None
    if delta_mode == "monthly":
        cordex_monthly = _cordex_monthly_view(cordex)

    # Precompute all daily deltas
    daily_deltas = {}
    missing_days = 0

    # Debug counters: how often each CORDEX var shows up in computed deltas
    delta_key_hits: dict[str, int] = {}

    for date in dates:
        if delta_mode == "daily":
            d = compute_daily_delta(cordex, baseline, date)
        else:
            d = compute_monthly_delta(cordex_monthly, baseline, date)
        if not d:
            missing_days += 1
        else:
            if debug:
                for k in d.keys():
                    delta_key_hits[str(k)] = delta_key_hits.get(str(k), 0) + 1
        daily_deltas[date.date()] = d

    if missing_days:
        print(f"  WARNING: {missing_days:,}/{len(dates):,} forecast days not found in CORDEX; leaving unchanged for those days")

    if debug:
        print("\n[DEBUG] Daily delta key hit counts (top 20):")
        items = sorted(delta_key_hits.items(), key=lambda kv: kv[1], reverse=True)
        for k, v in items[:20]:
            print(f"  {k}: {v:,} days")
        for k in [VAR_TO_CORDEX['temp'], VAR_TO_CORDEX['pressure']]:
            print(f"[DEBUG] daily deltas include {k}? hits={delta_key_hits.get(k, 0):,} days")

    output_df['_date'] = output_df.index.date

    # Track delta statistics
    delta_stats = {var: [] for var in VAR_TO_CORDEX.keys()}

    for date, group_idx in output_df.groupby('_date').groups.items():
        deltas = daily_deltas.get(date, {})

        for var, cordex_var in VAR_TO_CORDEX.items():
            if var not in output_df.columns:
                continue
            if cordex_var not in deltas:
                continue

            delta_or_ratio = deltas[cordex_var]
            delta_stats[var].append(delta_or_ratio)

            if var in ADDITIVE_VARS:
                output_df.loc[group_idx, var] += delta_or_ratio
            elif var in MULTIPLICATIVE_VARS:
                output_df.loc[group_idx, var] *= delta_or_ratio

    # Clean up
    output_df.drop('_date', axis=1, inplace=True)

    if debug:
        print("\n[DEBUG] delta_stats counts:")
        for var in VAR_TO_CORDEX.keys():
            print(f"  {var}: n={len(delta_stats.get(var, [])):,}")
        # Explain the most common reasons for missing temp/pressure
        for var in ['temp', 'pressure']:
            cordex_var = VAR_TO_CORDEX[var]
            reasons = []
            if var not in rsvar_df.columns:
                reasons.append(f"RS-VAR missing column '{var}'")
            if cordex_var not in baseline:
                reasons.append(f"baseline missing '{cordex_var}'")
            # if baseline has it but we still have no hits, likely target-date lookup mismatch
            if cordex_var in baseline and delta_key_hits.get(cordex_var, 0) == 0:
                reasons.append(f"computed daily deltas never contained '{cordex_var}' (target date lookup mismatch?)")
            if reasons:
                print(f"[DEBUG] why {var} might be missing: " + "; ".join(reasons))

    # Enforce solar constraints (consistent with V5 pipeline)
    print(f"\nEnforcing solar constraints...")
    output_df = enforce_solar_constraints(output_df, lat)

    # Print delta statistics
    _print_delta_statistics(delta_stats)

    # Print temperature statistics by time window
    _print_temperature_statistics_by_window(
        rsvar_df, cordex, baseline, output_df,
        baseline_start, baseline_end, isd_path
    )

    return output_df, delta_stats


def apply_mixed_resolution_climate_delta(
    rsvar_df,
    daily_df,
    monthly_df,
    baseline,
    gcm: str,
    rcm: str | None,
    lat: float,
    baseline_start: int = BASELINE_START,
    baseline_end: int = BASELINE_END,
    isd_path: Path = None,
    debug: bool = False
):
    """Apply climate delta from mixed daily/monthly resolution data to RS-VAR output.

    Uses vectorized operations for performance (handles 76+ years in seconds).

    Args:
        rsvar_df: RS-VAR hourly output
        daily_df: Daily resolution CORDEX data
        monthly_df: Monthly resolution CORDEX data
        baseline: Baseline climatology from compute_mixed_resolution_baseline
        gcm: GCM model name
        rcm: RCM model name (or None for GCM-only)
        lat: Latitude for solar constraints
        baseline_start: Baseline start year
        baseline_end: Baseline end year
        debug: Enable debug output

    Returns:
        (output_df, delta_stats)
    """
    model_name = f"{gcm}_{rcm}" if rcm else gcm
    print("\n" + "="*80)
    print(f"APPLYING MIXED RESOLUTION CLIMATE DELTA: {model_name}")
    print("="*80)

    output_df = rsvar_df.copy()

    # Make copies and apply unit conversions
    daily_df = daily_df.copy()
    monthly_df = monthly_df.copy()

    daily_df = _apply_unit_conversions(daily_df)
    monthly_df = _apply_unit_conversions(monthly_df)


    if debug:
        print("\n[DEBUG] RS-VAR columns:")
        print("  ", sorted(map(str, output_df.columns.tolist())))
        print("[DEBUG] Baseline keys:")
        print("  ", sorted(map(str, baseline.keys())))

    # Prepare daily climate data (one row per day)
    daily_view = _cordex_daily_view(daily_df)

    # Create date column for output
    output_df['_date'] = output_df.index.normalize()
    output_df['_doy'] = output_df.index.dayofyear
    output_df['_month'] = output_df.index.month

    print(f"\nApplying climate deltas (vectorized)...")

    # Track delta statistics
    delta_stats = {var: [] for var in VAR_TO_CORDEX.keys()}

    for var, cordex_var in VAR_TO_CORDEX.items():
        if var not in output_df.columns:
            continue
        if cordex_var not in baseline:
            continue

        baseline_info = baseline[cordex_var]
        resolution = baseline_info.get('resolution', 'daily')
        transform = baseline_info['transform']

        if resolution == 'daily':
            if cordex_var not in daily_view.columns:
                continue

            # Get future values aligned by date
            future_vals = daily_view[cordex_var].reindex(output_df['_date'])

            # Get baseline values by DOY
            baseline_doy = baseline_info['doy']
            baseline_vals = output_df['_doy'].map(lambda d: baseline_doy.get(d, baseline_info['overall']))

        elif resolution == 'monthly':
            if monthly_df.empty or cordex_var not in monthly_df.columns:
                continue

            # For monthly data, create a lookup by (year, month)
            monthly_df_copy = monthly_df.copy()
            monthly_df_copy['_ym'] = monthly_df_copy.index.to_period('M')
            monthly_lookup = monthly_df_copy.groupby('_ym')[cordex_var].first()

            # Get future values by year-month
            output_ym = output_df.index.to_period('M')
            future_vals = output_ym.map(lambda ym: monthly_lookup.get(ym, np.nan))

            # Get baseline values by month
            month_clim = baseline_info['month_clim']
            baseline_vals = output_df['_month'].map(lambda m: month_clim.get(m, baseline_info['overall']))

        else:
            continue

        # Convert to numpy arrays
        future_vals = pd.to_numeric(future_vals, errors='coerce').values
        baseline_vals = pd.to_numeric(baseline_vals, errors='coerce').values

        # Compute delta/ratio
        if transform == 'additive':
            delta = future_vals - baseline_vals
            # Apply delta
            valid_mask = ~np.isnan(delta)
            output_df.loc[valid_mask, var] = output_df.loc[valid_mask, var].values + delta[valid_mask]
            delta_stats[var] = delta[valid_mask].tolist()
        else:  # multiplicative
            ratio = np.where(baseline_vals > 0, future_vals / baseline_vals, 1.0)
            # Apply ratio
            valid_mask = ~np.isnan(ratio)
            output_df.loc[valid_mask, var] = output_df.loc[valid_mask, var].values * ratio[valid_mask]
            delta_stats[var] = ratio[valid_mask].tolist()

    # Clean up temp columns
    output_df.drop(['_date', '_doy', '_month'], axis=1, inplace=True)

    # Enforce solar constraints
    print(f"Enforcing solar constraints...")
    output_df = enforce_solar_constraints(output_df, lat)

    # Print delta statistics
    _print_delta_statistics(delta_stats)

    # Print temperature statistics by time window
    _print_temperature_statistics_by_window_mixed(
        rsvar_df, daily_df, monthly_df, baseline, output_df,
        baseline_start, baseline_end, isd_path
    )

    return output_df, delta_stats


def save_output(
    output_df: pd.DataFrame,
    metadata_rows: list[list],
    delta_stats: dict[str, list],
    output_path: Path,
    preview_first_year: bool = False,
):
    """Save output to Excel (if possible) + CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare output DataFrame
    output_out = output_df.copy()
    output_out.insert(0, 'datetime', output_out.index)
    output_out = output_out.reset_index(drop=True)

    # Prepare delta statistics DataFrame
    delta_data = []
    for var, values in delta_stats.items():
        if len(values) > 0:
            values = np.array(values)
            delta_data.append({
                'variable': var,
                'cordex_var': VAR_TO_CORDEX.get(var, ''),
                'transform': 'additive' if var in ADDITIVE_VARS else 'multiplicative',
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            })
    delta_df = pd.DataFrame(delta_data)

    # Prepare metadata
    metadata = pd.DataFrame(metadata_rows, columns=['parameter', 'value'])

    # Add a compact per-variable delta summary to Metadata as well.
    # (Users often check only Metadata and miss the Climate_Delta sheet.)
    if not delta_df.empty:
        delta_meta_rows: list[list] = []
        for _, row in delta_df.sort_values('variable').iterrows():
            var = str(row['variable'])
            transform = str(row['transform'])
            if transform == 'additive':
                summary = f"mean={row['mean']:+.4f}, std={row['std']:.4f}, min={row['min']:+.4f}, max={row['max']:+.4f}"
            else:
                summary = f"mean={row['mean']:.6f}x, std={row['std']:.6f}x, min={row['min']:.6f}x, max={row['max']:.6f}x"

            delta_meta_rows.append([f"delta::{var}::{transform}", summary])

        metadata = pd.concat(
            [metadata, pd.DataFrame(delta_meta_rows, columns=['parameter', 'value'])],
            axis=0,
            ignore_index=True,
        )

    # Sanitize sheet name (Excel has 31 char limit)
    delta_sheet_name = f"Climate_Delta"

    # Save to Excel (with fallback for openpyxl issues)
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            output_out.to_excel(writer, sheet_name='Output', index=False)
            delta_df.to_excel(writer, sheet_name=delta_sheet_name, index=False)
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
        print(f"  ✓ Saved Excel: {output_path}")
    except Exception:
        print(f"  (Skipping Excel write; saving CSV only)")

    # Also save CSV
    csv_path = output_path.with_suffix('.csv')
    output_out.to_csv(csv_path, index=False)
    print(f"  ✓ Saved CSV:   {csv_path}")

    # Optional: save first-year preview
    if preview_first_year:
        output_out['datetime'] = pd.to_datetime(output_out['datetime'])
        first_year = int(output_out['datetime'].dt.year.min())
        first_year_df = output_out[output_out['datetime'].dt.year == first_year]
        preview_path = output_path.with_name(f"{output_path.stem}_{first_year}.csv")
        first_year_df.to_csv(preview_path, index=False)
        print(f"  ✓ Saved first-year preview: {preview_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply Climate Delta (Batch: multi-city, multi-scenario)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--config", type=str, default=str(CITIES_CONFIG_PATH),
        help="Path to cities_config.csv (must include columns: city, latitude)"
    )
    parser.add_argument(
        "--city", type=str, nargs='*', default=None,
        help="City name(s) to process (default: all cities in cities_config.csv)"
    )

    parser.add_argument(
        "--rsvar-dir", type=str, default=str(DEFAULT_RSVAR_DIR),
        help="Directory containing RS-VAR outputs (default: output/RSVAR)"
    )

    parser.add_argument(
        "--out-dir", type=str, default=None,
        help=(
            "Output directory root. If omitted, writes to output/RSVAR_cd/"
        )
    )
    parser.add_argument(
        "--delta-mode", type=str, default="daily", choices=["daily", "monthly"],
        help="Delta application mode: daily (default) or monthly (month-mean delta)"
    )

    parser.add_argument(
        "--baseline-start", type=int, default=BASELINE_START,
        help=f"Baseline start year (default: {BASELINE_START})"
    )
    parser.add_argument(
        "--baseline-end", type=int, default=BASELINE_END,
        help=f"Baseline end year (default: {BASELINE_END})"
    )

    parser.add_argument(
        "--overwrite", action='store_true',
        help="Overwrite outputs if they already exist"
    )

    parser.add_argument(
        "--debug", action='store_true',
        help="Print debug information about baseline/deltas to diagnose missing variables"
    )
    parser.add_argument(
        "--preview-first-year", action='store_true',
        help="Save a CSV containing only the first forecast year (for quick inspection)"
    )

    parser.add_argument(
        "--scenario", type=str, nargs='*', default=None,
        help=(
            "Scenario(s) to run (default: all). "
            f"Available: {', '.join(SCENARIOS.keys())}"
        )
    )

    parser.add_argument(
        "--list-scenarios", action='store_true',
        help="List available scenarios and exit"
    )

    args = parser.parse_args()

    # Handle --list-scenarios
    if args.list_scenarios:
        print("Available scenarios:")
        for name, conf in SCENARIOS.items():
            gcm = conf.get('gcm', 'UNKNOWN')
            rcm = conf.get('rcm', None)
            model_str = f"{gcm}_{rcm}" if rcm else gcm
            print(f"  {name} ({model_str})")
        return

    cities_config_path = Path(args.config)
    cities_df = _load_cities_config(cities_config_path)

    # Avoid itertuples attribute access (IDE typing can get confused)
    cities_by_name = dict(zip(cities_df["city"], cities_df["latitude"].astype(float)))

    requested_cities = _normalize_city_args(args.city, list(cities_by_name.keys()), cities_config_path)

    rsvar_dir = Path(args.rsvar_dir)
    if not rsvar_dir.exists():
        raise SystemExit(f"ERROR: RS-VAR directory not found: {rsvar_dir}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = DEFAULT_OUT_DIR_MONTHLY if args.delta_mode == "monthly" else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter scenarios based on --scenario argument
    if args.scenario:
        invalid = [s for s in args.scenario if s not in SCENARIOS]
        if invalid:
            avail = ", ".join(SCENARIOS.keys())
            raise SystemExit(
                f"ERROR: Unknown scenario(s): {invalid}\n"
                f"Available: {avail}\n"
                "Use --list-scenarios to see all available scenarios."
            )
        scenarios = {k: SCENARIOS[k] for k in args.scenario}
    else:
        scenarios = SCENARIOS

    print("="*80)
    print("APPLY CLIMATE DELTA (BATCH)")
    print("="*80)
    print(f"Cities: {requested_cities}")
    print(f"Scenarios: {list(scenarios.keys())}")
    print(f"RSVAR dir: {rsvar_dir}")
    print(f"Out dir:   {out_dir}")
    print(f"Baseline:  {args.baseline_start}-{args.baseline_end}")
    print(f"Delta mode: {args.delta_mode}")

    # ISD data directory for baseline temperature
    isd_dir = REPO_ROOT / "data" / "ISD_complete_solar"

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for city in requested_cities:
        lat = cities_by_name[city]
        city_slug = _slug_city(city)

        # find rsvar file: prefer xlsx then csv
        rsvar_xlsx = rsvar_dir / f"rsvar_output_{city_slug}.xlsx"
        rsvar_csv = rsvar_dir / f"rsvar_output_{city_slug}.csv"

        # Find ISD file for baseline temperature
        # Try both lowercase slug and capitalized city name formats
        isd_path = isd_dir / f"{city_slug}_with_solar.csv"
        if not isd_path.exists():
            # Try capitalized format (e.g., Montreal_with_solar.csv, Los_Angeles_with_solar.csv)
            isd_path = isd_dir / f"{city.replace(' ', '_')}_with_solar.csv"

        if not isd_path.exists():
            print(f"[WARNING] ISD file not found: {isd_path}")
            isd_path = None
        else:
            print(f"[DEBUG] Found ISD file: {isd_path}")
        if rsvar_xlsx.exists():
            rsvar_path = rsvar_xlsx
        elif rsvar_csv.exists():
            rsvar_path = rsvar_csv
        else:
            print(f"\n[SKIP] {city}: RS-VAR output not found (expected {rsvar_xlsx} or {rsvar_csv})")
            n_skip += 1
            continue

        print("\n" + "-"*80)
        print(f"City: {city} (lat={lat})")
        print(f"RSVAR: {rsvar_path}")

        try:
            rsvar_df, rsvar_metadata = load_rsvar_output(rsvar_path)
        except Exception as e:
            print(f"  [FAIL] Could not load RS-VAR for {city}: {e}")
            n_fail += 1
            continue

        for scenario_id, sconf in scenarios.items():
            gcm = sconf.get('gcm', 'UNKNOWN')
            rcm = sconf.get('rcm', None)

            city_out_dir = out_dir / scenario_id / city_slug
            city_out_dir.mkdir(parents=True, exist_ok=True)
            mode_suffix = "_monthly" if args.delta_mode == "monthly" else ""
            out_xlsx = city_out_dir / f"forecast_{city_slug}_{scenario_id}{mode_suffix}.xlsx"

            if out_xlsx.exists() and not args.overwrite:
                print(f"  [SKIP] {scenario_id}: output exists: {out_xlsx}")
                n_skip += 1
                continue

            print(f"  Scenario: {scenario_id}")

            # Determine scenario type:
            # 1. CMIP5/CMIP6/CORDEX_CMIP6: has cmip5_base_path, cmip6_base_path or cordex_base_path
            # 2. CORDEX_CMIP5: has variant_root
            is_cmip_type = 'cmip5_base_path' in sconf or 'cmip6_base_path' in sconf or 'cordex_base_path' in sconf

            if is_cmip_type:
                # CMIP5, CMIP6 or CORDEX_CMIP6 workflow (unified loader)
                base_path = Path(sconf.get('cordex_base_path') or sconf.get('cmip6_base_path') or sconf.get('cmip5_base_path'))

                # Extract SSP/RCP from scenario_id
                # Examples: "CMIP6_ssp245" -> "ssp245", "CMIP5_rcp85" -> "rcp85", "CORDEX_CMIP6_CRCM5_ssp245" -> "ssp245"
                import re
                scenario_match = re.search(r'(ssp\d+|rcp\d+)', scenario_id)
                ssp = scenario_match.group(1) if scenario_match else scenario_id

                try:
                    daily_df, monthly_df, meta = load_cmip6_or_cordex6(base_path, city, ssp, scenario_id=scenario_id)
                except Exception as e:
                    print(f"  [FAIL] Could not load data for {city}/{scenario_id}: {e}")
                    n_fail += 1
                    continue

                try:
                    monthly_df_effective = monthly_df
                    var_resolution_config = {}

                    if args.delta_mode == "monthly":
                        monthly_df_effective, missing_in_monthly, monthly_only_cols = build_effective_monthly_df(
                            daily_df, monthly_df
                        )
                        if missing_in_monthly:
                            print(f"  [Monthly Mode] Filled missing monthly vars from daily resample: {missing_in_monthly}")
                        if monthly_only_cols:
                            print(f"  [Monthly Mode] Monthly-only vars retained: {monthly_only_cols}")

                        # Force monthly deltas for all variables when requested
                        var_resolution_config = {
                            'tas_C': 'monthly',
                            'tasmax_C': 'monthly',
                            'tasmin_C': 'monthly',
                            'ps_hPa': 'monthly',
                            'sfcWind': 'monthly',
                            'rsds': 'monthly',
                            'hurs': 'monthly',
                            'huss': 'monthly',
                        }

                    baseline = compute_mixed_resolution_baseline(
                        daily_df, monthly_df_effective,
                        args.baseline_start, args.baseline_end,
                        var_resolution_config
                    )
                except Exception as e:
                    print(f"  [FAIL] Could not compute baseline for {city}/{scenario_id}: {e}")
                    n_fail += 1
                    continue

                try:
                    output_df, delta_stats = apply_mixed_resolution_climate_delta(
                        rsvar_df, daily_df, monthly_df_effective, baseline,
                        gcm=gcm, rcm=rcm, lat=lat,
                        baseline_start=args.baseline_start,
                        baseline_end=args.baseline_end,
                        isd_path=isd_path,
                        debug=args.debug
                    )
                except Exception as e:
                    print(f"  [FAIL] Delta apply failed for {city}/{scenario_id}: {e}")
                    n_fail += 1
                    continue

                metadata_rows = [
                    ['city', city],
                    ['city_slug', city_slug],
                    ['lat', lat],
                    ['scenario_id', scenario_id],
                    ['gcm', gcm],
                    ['rcm', rcm if rcm else 'N/A'],
                    ['base_path', str(base_path)],
                    ['ssp', ssp],
                    ['baseline_start', args.baseline_start],
                    ['baseline_end', args.baseline_end],
                    ['delta_mode', args.delta_mode],
                    ['n_hours', len(output_df)],
                    ['rsvar_source', rsvar_metadata.get('isd_path', str(rsvar_path))],
                ]

            else:
                # Original CORDEX_CMIP5 workflow
                variant_root = Path(sconf['variant_root'])

                try:
                    cordex, cordex_meta = load_and_merge_cordex_city_variant(variant_root, city)
                except Exception as e:
                    print(f"  [FAIL] Could not load CORDEX for {city}/{scenario_id}: {e}")
                    n_fail += 1
                    continue

                baseline = compute_cordex_baseline(
                    cordex,
                    args.baseline_start,
                    args.baseline_end,
                    mode=args.delta_mode
                )

                try:
                    output_df, delta_stats = apply_climate_delta(
                        rsvar_df,
                        cordex,
                        baseline,
                        gcm=gcm,
                        rcm=rcm,
                        lat=lat,
                        baseline_start=args.baseline_start,
                        baseline_end=args.baseline_end,
                        isd_path=isd_path,
                        debug=args.debug,
                        delta_mode=args.delta_mode,
                    )
                except Exception as e:
                    print(f"  [FAIL] Delta apply failed for {city}/{scenario_id}: {e}")
                    n_fail += 1
                    continue

                # Metadata rows for CMIP5
                metadata_rows = [
                    ['city', city],
                    ['city_slug', city_slug],
                    ['lat', lat],
                    ['scenario_id', scenario_id],
                    ['gcm', gcm],
                    ['rcm', rcm if rcm else 'N/A'],
                    ['variant_root', str(variant_root)],
                    ['cordex_historical_path', cordex_meta.get('historical_path', '')],
                    ['cordex_rcp85_path', cordex_meta.get('rcp85_path', '')],
                    ['baseline_start', args.baseline_start],
                    ['baseline_end', args.baseline_end],
                    ['delta_mode', args.delta_mode],
                    ['n_hours', len(output_df)],
                    ['rsvar_source', rsvar_metadata.get('isd_path', str(rsvar_path))],
                ]

            save_output(
                output_df,
                metadata_rows,
                delta_stats,
                out_xlsx,
                preview_first_year=args.preview_first_year,
            )
            n_ok += 1

    print("\n" + "="*80)
    print("BATCH SUMMARY")
    print("="*80)
    print(f"OK:   {n_ok}")
    print(f"SKIP: {n_skip}")
    print(f"FAIL: {n_fail}")


if __name__ == "__main__":
    main()
