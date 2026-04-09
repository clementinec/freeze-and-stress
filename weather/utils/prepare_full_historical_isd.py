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

PROJECT_ROOT = Path(__file__).parent.parent
OLD_ISD_DIR = PROJECT_ROOT / "data" / "ISD_complete"
# NSRDB_SOURCE = Path("/Users/hongshanguo/Library/CloudStorage/OneDrive-TheUniversityOfHongKong/Misc/EPWs/tests/NSRDB")
# OUTPUT_DIR = PROJECT_ROOT / "data" / "ISD_full_history"

NSRDB_SOURCE = PROJECT_ROOT / "data" / "NSRDB"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ISD_complete_solar"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Load config for baseline and validation periods
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

baseline_start = config['baseline']['start_year']
baseline_end = config['baseline']['end_year']
validation_start = pd.to_datetime(config['validation']['start_date'])
validation_end = pd.to_datetime(config['validation']['end_date'])

# Load city timezone mappings
CITY_TIMEZONES = config.get('city_timezones', {})

# Create output directory for baseline + validation period
BASELINE_OUTPUT_DIR = PROJECT_ROOT / "data" / "ISD_baseline_solar"
BASELINE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

logger.info(f"Configuration loaded:")
logger.info(f"  Baseline period: {baseline_start}-{baseline_end}")
logger.info(f"  Validation period: {validation_start.date()} to {validation_end.date()}")
logger.info(f"  Combined period: {baseline_start} to {validation_end.date()}")
logger.info(f"  Full history output: {OUTPUT_DIR}")
logger.info(f"  Baseline+validation output: {BASELINE_OUTPUT_DIR}")
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


CITIES = {
    # "Los_Angeles": "los_angeles",
    # "Miami": "miami",
    # "Toronto": "toronto",
    # "Montreal": "montreal",
    "New_York": "new_york",
}

for isd_name, nsrdb_name in CITIES.items():
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
    nsrdb_file = NSRDB_SOURCE / f"nsrdb_{nsrdb_name}_1998-2024.csv"
    if not nsrdb_file.exists():
        logger.warning(f"  ⚠ NSRDB file not found: {nsrdb_file}")
        logger.info(f"  Saving ISD without solar data...")
        output_file = OUTPUT_DIR / f"{isd_name}_with_solar.csv"
        isd_df.to_csv(output_file)
        logger.info(f"  ✓ SAVED: {output_file.name} (no solar data)")

        # Also save baseline + validation period
        logger.info(f"\n  Extracting baseline + validation period...")
        baseline_start_date = pd.to_datetime(f"{baseline_start}-01-01")
        baseline_isd = isd_df[(isd_df.index >= baseline_start_date) & (isd_df.index <= validation_end)]

        if len(baseline_isd) > 0:
            baseline_output_file = BASELINE_OUTPUT_DIR / f"{isd_name}_with_solar.csv"
            baseline_isd.to_csv(baseline_output_file)

            baseline_file_size_mb = baseline_output_file.stat().st_size / 1024**2
            baseline_years = baseline_isd.index.year.max() - baseline_isd.index.year.min() + 1
            baseline_temp_coverage = baseline_isd['temp'].notna().sum() / len(baseline_isd) * 100

            logger.info(f"  ✓ BASELINE+VALIDATION SAVED: {baseline_output_file.name}")
            logger.info(f"    Period: {baseline_isd.index.min()} to {baseline_isd.index.max()}")
            logger.info(f"    Years: {baseline_years}")
            logger.info(f"    Records: {len(baseline_isd):,}")
            logger.info(f"    Temperature coverage: {baseline_temp_coverage:.1f}%")
            logger.info(f"    Size: {baseline_file_size_mb:.1f} MB")
        else:
            logger.warning(f"  ⚠ No data in baseline+validation period for {isd_name}")

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

    # Extract and save baseline + validation period
    logger.info(f"\n  Extracting baseline + validation period...")
    baseline_start_date = pd.to_datetime(f"{baseline_start}-01-01")
    baseline_merged = merged[(merged.index >= baseline_start_date) & (merged.index <= validation_end)]

    if len(baseline_merged) > 0:
        baseline_output_file = BASELINE_OUTPUT_DIR / f"{isd_name}_with_solar.csv"
        baseline_merged.to_csv(baseline_output_file)

        baseline_file_size_mb = baseline_output_file.stat().st_size / 1024**2
        baseline_years = baseline_merged.index.year.max() - baseline_merged.index.year.min() + 1
        baseline_temp_coverage = baseline_merged['temp'].notna().sum() / len(baseline_merged) * 100

        logger.info(f"  ✓ BASELINE+VALIDATION SAVED: {baseline_output_file.name}")
        logger.info(f"    Period: {baseline_merged.index.min()} to {baseline_merged.index.max()}")
        logger.info(f"    Years: {baseline_years}")
        logger.info(f"    Records: {len(baseline_merged):,}")
        logger.info(f"    Temperature coverage: {baseline_temp_coverage:.1f}%")
        if 'GHI' in baseline_merged.columns:
            baseline_ghi_coverage = baseline_merged['GHI'].notna().sum() / len(baseline_merged) * 100
            logger.info(f"    GHI coverage: {baseline_ghi_coverage:.1f}%")
        logger.info(f"    Size: {baseline_file_size_mb:.1f} MB")
    else:
        logger.warning(f"  ⚠ No data in baseline+validation period for {isd_name}")

logger.info(f"\n{'='*70}")
logger.info("✓ ALL CITIES COMPLETE")
logger.info(f"{'='*70}")
