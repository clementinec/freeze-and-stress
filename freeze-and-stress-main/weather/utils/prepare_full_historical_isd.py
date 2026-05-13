#!/usr/bin/env python3
"""
Prepare full historical ISD data (1943-2024) for all cities, merged with NSRDB solar.
This replaces the artificially limited 1991-2020 files.
"""

import pandas as pd
from pathlib import Path
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WEATHER_DATA = Path(__file__).parent.parent / "data"
OLD_ISD_DIR  = WEATHER_DATA / "ISD_complete"
NSRDB_SOURCE = WEATHER_DATA / "NSRDB"
OUTPUT_DIR   = WEATHER_DATA / "ISD_complete_solar"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Load config for baseline and validation periods
CONFIG_FILE = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

baseline_start = config['baseline']['start_year']
baseline_end = config['baseline']['end_year']
validation_start = pd.to_datetime(config['validation']['start_date'])
validation_end = pd.to_datetime(config['validation']['end_date'])

CITY_TIMEZONES = config.get('city_timezones', {})

logger.info(f"Configuration loaded:")
logger.info(f"  Baseline period: {baseline_start}-{baseline_end}")
logger.info(f"  Validation period: {validation_start.date()} to {validation_end.date()}")
logger.info(f"  Output: {OUTPUT_DIR}")
logger.info(f"  City timezones loaded: {len(CITY_TIMEZONES)} cities")


def convert_utc_to_local(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Convert UTC timestamps to local time based on city timezone.

    Args:
        df: DataFrame with DatetimeIndex in UTC
        city: City name (used to lookup timezone)

    Returns:
        DataFrame with timezone-naive DatetimeIndex in local time
    """
    # Look up timezone for the city
    timezone = CITY_TIMEZONES.get(city)

    if not timezone:
        logger.warning(f"    ⚠ No timezone mapping found for '{city}', keeping UTC")
        return df

    try:
        # Convert index to UTC-aware
        if df.index.tz is None:
            df_converted = df.copy()
            df_converted.index = df_converted.index.tz_localize('UTC')
        else:
            df_converted = df.copy()

        # Convert to local timezone
        df_converted.index = df_converted.index.tz_convert(timezone)

        # Remove timezone info to make it timezone-naive (for compatibility)
        df_converted.index = df_converted.index.tz_localize(None)

        logger.info(f"    ✓ Converted from UTC to {timezone}")
        logger.info(f"    ✓ Time range: {df_converted.index.min()} to {df_converted.index.max()}")

        return df_converted

    except Exception as e:
        logger.error(f"    ✗ Error converting timezone: {e}")
        logger.info(f"    Keeping original timestamps")
        return df

# city name -> (nsrdb_name, nsrdb_filename_suffix)
CITIES = {
    "Los_Angeles": ("los_angeles", "1998-2024"),
    "Miami":       ("miami",       "1998-2024"),
    "Montreal":    ("montreal",    "1998-2024"),
    "Toronto":     ("toronto",     "1998-2024"),
    "Phoenix":     ("phoenix",     "1998-2024"),
    "Vancouver":   ("vancouver",   "1998-2024"),
}

for isd_name, (nsrdb_name, nsrdb_suffix) in CITIES.items():
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing {isd_name}")
    logger.info(f"{'='*70}")

    # Load FULL historical ISD data
    isd_file = OLD_ISD_DIR / f"ISD_complete_{isd_name}.csv"
    if not isd_file.exists():
        logger.warning(f"  ⚠ ISD file not found: {isd_file}")
        continue

    logger.info(f"  Loading historical ISD: {isd_file.name}")
    isd_df = pd.read_csv(isd_file, index_col=0, parse_dates=True, low_memory=False)

    # Rename columns to standard format
    column_mapping = {
        'air_temperature': 'temp',
        'dew_point_temperature': 'dewpoint',
        'sea_level_pressure': 'pressure',
        'wind_direction_angle': 'wind_dir',
        'wind_speed_rate': 'wind_speed'
    }

    # Keep only the core columns we need
    columns_to_keep = []
    for old_name, new_name in column_mapping.items():
        if old_name in isd_df.columns:
            columns_to_keep.append(old_name)

    isd_df = isd_df[columns_to_keep].copy()
    isd_df = isd_df.rename(columns=column_mapping)
    isd_df.index.name = 'DATE'

    logger.info(f"    ✓ ISD raw: {len(isd_df):,} records (may include non-hourly)")
    logger.info(f"    Period: {isd_df.index.min()} to {isd_df.index.max()}")

    # Resample to hourly by taking mean of non-NaN values within each hour
    logger.info(f"  Resampling to hourly (mean of non-NaN values)...")
    isd_df = isd_df.resample('h').mean()

    logger.info(f"    ✓ ISD hourly: {len(isd_df):,} records")
    logger.info(f"    Temperature coverage: {isd_df['temp'].notna().sum() / len(isd_df) * 100:.1f}%")

    # Convert from UTC to local standard time (LST)
    logger.info(f"  Converting from UTC to local time...")
    isd_df = convert_utc_to_local(isd_df, isd_name)

    # Load NSRDB
    nsrdb_file = NSRDB_SOURCE / f"nsrdb_{nsrdb_name}_{nsrdb_suffix}.csv"
    if not nsrdb_file.exists():
        logger.warning(f"  ⚠ NSRDB file not found: {nsrdb_file}")
        logger.info(f"  Saving ISD without solar data...")
        output_file = OUTPUT_DIR / f"{isd_name}_with_solar.csv"
        isd_df.to_csv(output_file)
        logger.info(f"  ✓ SAVED: {output_file.name} (no solar data)")
        continue

    logger.info(f"  Loading NSRDB: {nsrdb_file.name}")
    nsrdb_df = pd.read_csv(nsrdb_file)

    # Create datetime from NSRDB columns
    nsrdb_df['datetime'] = pd.to_datetime(nsrdb_df[['Year', 'Month', 'Day', 'Hour', 'Minute']].rename(
        columns={'Year': 'year', 'Month': 'month', 'Day': 'day', 'Hour': 'hour', 'Minute': 'minute'}))

    logger.info(f"    ✓ NSRDB: {len(nsrdb_df):,} records")
    logger.info(f"    Period: {nsrdb_df['datetime'].min()} to {nsrdb_df['datetime'].max()}")

    # NSRDB is at :30 past hour (e.g., 12:30), representing the hour ending at that time
    # ISD is now resampled to exact hours (e.g., 12:00, 13:00)
    # We need to align NSRDB's 12:30 with ISD's 12:00
    # So we round NSRDB's datetime to the hour (12:30 -> 13:00), then shift back 1 hour to get 12:00
    nsrdb_df['datetime_aligned'] = nsrdb_df['datetime'].dt.floor('h')

    # Select solar columns
    solar_cols = ['datetime_aligned', 'GHI', 'DHI', 'DNI']
    nsrdb_solar = nsrdb_df[solar_cols].copy()

    # Merge on aligned datetime
    logger.info(f"  Merging ISD (hourly) with NSRDB (half-hour aligned to hour)...")
    isd_df_reset = isd_df.reset_index()
    merged = pd.merge(isd_df_reset, nsrdb_solar, left_on='DATE', right_on='datetime_aligned', how='left')
    merged = merged.drop(['datetime_aligned'], axis=1)
    merged = merged.set_index('DATE')

    # Keep missing solar data as NaN (do not fill with 0)

    # Reorder columns
    column_order = ['temp', 'dewpoint', 'pressure', 'wind_dir', 'wind_speed', 'GHI', 'DHI', 'DNI']
    available_cols = [c for c in column_order if c in merged.columns]
    merged = merged[available_cols]

    # Statistics
    ghi_available = merged['GHI'].notna().sum()
    ghi_nonzero = (merged['GHI'] > 0).sum()
    temp_coverage = merged['temp'].notna().sum() / len(merged) * 100
    ghi_coverage = ghi_available / len(merged) * 100

    logger.info(f"\n  ✓ MERGED DATA:")
    logger.info(f"    Total records: {len(merged):,}")
    logger.info(f"    Period: {merged.index.min()} to {merged.index.max()}")
    logger.info(f"    Years: {merged.index.year.max() - merged.index.year.min() + 1}")
    logger.info(f"    Temperature coverage: {temp_coverage:.1f}%")
    logger.info(f"    GHI coverage: {ghi_coverage:.1f}% ({ghi_available:,} records)")
    logger.info(f"    GHI non-zero: {ghi_nonzero:,} ({ghi_nonzero/len(merged)*100:.1f}%)")
    logger.info(f"    GHI mean (valid): {merged['GHI'].mean():.2f} W/m²")

    # Save
    output_file = OUTPUT_DIR / f"{isd_name}_with_solar.csv"
    merged.to_csv(output_file)

    file_size_mb = output_file.stat().st_size / 1024**2
    logger.info(f"\n  ✓ SAVED: {output_file.name}")
    logger.info(f"    Size: {file_size_mb:.1f} MB")

logger.info(f"\n{'='*70}")
logger.info("✓ ALL CITIES COMPLETE")
logger.info(f"{'='*70}")
