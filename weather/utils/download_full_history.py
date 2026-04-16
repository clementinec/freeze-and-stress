#!/usr/bin/env python3
"""
Download FULL historical ISD data (1943-2024) for all 4 cities and stitch into complete CSVs.
Creates comprehensive data quality report with station metadata.
"""

import pandas as pd
import gzip
from pathlib import Path
from ftplib import FTP
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

STATIONS = {
    # "Los_Angeles": {
    #     "code": "722950-23174",
    #     "name": "LOS ANGELES INTERNATIONAL AIRPORT",
    #     "lat": 34.0522,
    #     "lon": -118.2437,
    #     "distance_km": 18.3
    # },
    # "Miami": {
    #     "code": "722020-12839",
    #     "name": "MIAMI INTERNATIONAL AIRPORT",
    #     "lat": 25.7617,
    #     "lon": -80.1918,
    #     "distance_km": 12.9
    # },
    # "Montreal": {
    #     "code": "716270-94792",
    #     "name": "TRUDEAU INTL",
    #     "lat": 45.5017,
    #     "lon": -73.5673,
    #     "distance_km": 11.9
    # },
    # "Toronto": {
    #     "code": "716240-99999",
    #     "name": "LESTER B PEARSON INTL",
    #     "lat": 43.6532,
    #     "lon": -79.3832,
    #     "distance_km": 2.6
    # },
    "New_York": {
        "code": "725030-14732",
        "name": "LA GUARDIA AIRPORT",
        "lat": 40.779,
        "lon": -73.880,
        "distance_km": 12.9
    }
}

PROJECT_ROOT = Path(__file__).parent.parent
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "ISD_full_history"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ISD_complete"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Year range - try full history
YEARS = list(range(1943, 2025))  # 1943-2024

# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_isd_file(gz_file_path):
    """Parse ISD fixed-width format file"""
    records = []
    with gzip.open(gz_file_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if len(line) < 105:
                continue
            try:
                year = int(line[15:19])
                month = int(line[19:21])
                day = int(line[21:23])
                hour = int(line[23:25])
                minute = int(line[25:27])

                temp_str = line[87:92].strip()
                temp = float(temp_str) / 10 if temp_str not in ['+9999', '9999', ''] else None
                temp_quality = line[92:93]

                dew_str = line[93:98].strip()
                dew = float(dew_str) / 10 if dew_str not in ['+9999', '9999', ''] else None
                dew_quality = line[98:99]

                pressure_str = line[99:104].strip()
                pressure = float(pressure_str) / 10 if pressure_str not in ['99999', ''] else None
                pressure_quality = line[104:105] if len(line) > 104 else None

                wind_dir_str = line[60:63].strip()
                wind_dir = float(wind_dir_str) if wind_dir_str not in ['999', ''] else None
                wind_dir_quality = line[63:64]

                wind_speed_str = line[65:69].strip()
                wind_speed = float(wind_speed_str) / 10 if wind_speed_str not in ['9999', ''] else None
                wind_speed_quality = line[69:70]

                records.append({
                    'datetime': pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute),
                    'air_temperature': temp,
                    'air_temperature_quality_code': temp_quality,
                    'dew_point_temperature': dew,
                    'dew_point_quality_code': dew_quality,
                    'sea_level_pressure': pressure,
                    'sea_level_pressure_quality_code': pressure_quality,
                    'wind_direction_angle': wind_dir,
                    'wind_direction_quality_code': wind_dir_quality,
                    'wind_speed_rate': wind_speed,
                    'wind_speed_quality_code': wind_speed_quality
                })
            except:
                continue
    return pd.DataFrame(records) if records else pd.DataFrame()


# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_city_full_history(city_name, station_code):
    """Download full available history for a city"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Downloading {city_name} - Full History")
    logger.info(f"Station: {STATIONS[city_name]['name']} ({station_code})")
    logger.info(f"{'='*80}")

    city_dir = DOWNLOAD_DIR / city_name
    city_dir.mkdir(parents=True, exist_ok=True)

    ftp = FTP("ftp.ncei.noaa.gov")
    ftp.login()
    logger.info("✓ Connected to NOAA FTP")

    downloaded_years = []
    year_stats = []

    for year in YEARS:
        filename = f"{station_code}-{year}.gz"
        local_path = city_dir / filename

        # Skip if already downloaded
        if local_path.exists() and local_path.stat().st_size > 0:
            logger.info(f"  {year}: Already exists, parsing...")
            df = parse_isd_file(local_path)
            if not df.empty:
                downloaded_years.append(year)
                temp_count = df['air_temperature'].notna().sum()
                year_stats.append({
                    'year': year,
                    'records': len(df),
                    'temp_coverage': temp_count / len(df) * 100 if len(df) > 0 else 0
                })
                logger.info(f"    ✓ {len(df)} records, {temp_count} with temp ({temp_count/len(df)*100:.1f}%)")
            continue

        # Download from FTP
        try:
            ftp.cwd(f"/pub/data/noaa/{year}/")
            with open(local_path, 'wb') as f:
                ftp.retrbinary(f'RETR {filename}', f.write)

            # Parse immediately to check
            df = parse_isd_file(local_path)
            if not df.empty:
                downloaded_years.append(year)
                temp_count = df['air_temperature'].notna().sum()
                year_stats.append({
                    'year': year,
                    'records': len(df),
                    'temp_coverage': temp_count / len(df) * 100 if len(df) > 0 else 0
                })
                logger.info(f"  {year}: ✓ Downloaded ({len(df)} records, {temp_count/len(df)*100:.1f}% temp)")
            else:
                logger.warning(f"  {year}: Downloaded but no parseable data")
                local_path.unlink()  # Remove empty file

        except Exception as e:
            if "550" in str(e):  # File not found
                logger.debug(f"  {year}: Not available")
            else:
                logger.warning(f"  {year}: ✗ Failed - {e}")
            if local_path.exists():
                local_path.unlink()

    ftp.quit()

    logger.info(f"\n✓ {city_name}: Downloaded {len(downloaded_years)} years ({min(downloaded_years)}-{max(downloaded_years)})")
    return downloaded_years, year_stats


# ============================================================================
# STITCHING FUNCTIONS
# ============================================================================

def stitch_city_data(city_name):
    """Stitch all annual files for a city into one complete CSV"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Stitching {city_name} - Complete Dataset")
    logger.info(f"{'='*80}")

    city_dir = DOWNLOAD_DIR / city_name
    gz_files = sorted(city_dir.glob("*.gz"))

    if not gz_files:
        logger.error(f"No data files found for {city_name}")
        return None

    logger.info(f"Found {len(gz_files)} annual files, parsing...")

    all_data = []
    for gz_file in gz_files:
        df = parse_isd_file(gz_file)
        if not df.empty:
            all_data.append(df)
            year = gz_file.stem.split('-')[-1]
            logger.info(f"  {year}: {len(df)} records")

    if not all_data:
        logger.error(f"No parseable data for {city_name}")
        return None

    # Combine all years
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('datetime').reset_index(drop=True)

    # Save to CSV
    output_file = OUTPUT_DIR / f"ISD_complete_{city_name}.csv"
    combined.to_csv(output_file, index=False)

    logger.info(f"\n✓ SAVED: {output_file}")
    logger.info(f"  Total records: {len(combined):,}")
    logger.info(f"  Date range: {combined['datetime'].min()} to {combined['datetime'].max()}")
    logger.info(f"  File size: {output_file.stat().st_size / 1024**2:.1f} MB")

    return combined


# ============================================================================
# DATA QUALITY REPORT
# ============================================================================

def generate_quality_report():
    """Generate comprehensive data quality markdown report"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Generating Data Quality Report")
    logger.info(f"{'='*80}")

    report_lines = []
    report_lines.append("# ISD Complete Historical Data - Quality Report")
    report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Source:** NOAA Integrated Surface Database (ISD)")
    report_lines.append(f"**Downloaded:** Full available history (1943-2024)")
    report_lines.append("\n---\n")

    for city_name, info in STATIONS.items():
        report_lines.append(f"## {city_name.replace('_', ' ')}")
        report_lines.append(f"\n### Station Information")
        report_lines.append(f"- **Station Code:** {info['code']}")
        report_lines.append(f"- **Station Name:** {info['name']}")
        report_lines.append(f"- **Coordinates:** {info['lat']}°N, {info['lon']}°W")
        report_lines.append(f"- **Distance from city center:** {info['distance_km']} km")

        # Load complete data
        csv_file = OUTPUT_DIR / f"ISD_complete_{city_name}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file, parse_dates=['datetime'])

            report_lines.append(f"\n### Data Coverage")
            report_lines.append(f"- **Total records:** {len(df):,}")
            report_lines.append(f"- **Date range:** {df['datetime'].min()} to {df['datetime'].max()}")
            report_lines.append(f"- **Years covered:** {(df['datetime'].max() - df['datetime'].min()).days / 365.25:.1f} years")

            # Variable coverage
            report_lines.append(f"\n### Variable Coverage (% non-null)")
            temp_pct = df['air_temperature'].notna().sum() / len(df) * 100
            dew_pct = df['dew_point_temperature'].notna().sum() / len(df) * 100
            pressure_pct = df['sea_level_pressure'].notna().sum() / len(df) * 100
            wind_dir_pct = df['wind_direction_angle'].notna().sum() / len(df) * 100
            wind_speed_pct = df['wind_speed_rate'].notna().sum() / len(df) * 100

            report_lines.append(f"- **Temperature:** {temp_pct:.1f}%")
            report_lines.append(f"- **Dew Point:** {dew_pct:.1f}%")
            report_lines.append(f"- **Pressure:** {pressure_pct:.1f}%")
            report_lines.append(f"- **Wind Direction:** {wind_dir_pct:.1f}%")
            report_lines.append(f"- **Wind Speed:** {wind_speed_pct:.1f}%")

            # Temporal resolution
            df['year'] = df['datetime'].dt.year
            yearly_counts = df.groupby('year').size()
            avg_obs_per_day = yearly_counts.mean() / 365.25

            report_lines.append(f"\n### Temporal Resolution")
            report_lines.append(f"- **Average observations per day:** {avg_obs_per_day:.1f}")
            if avg_obs_per_day >= 20:
                report_lines.append(f"- **Resolution:** Sub-hourly")
            elif avg_obs_per_day >= 18:
                report_lines.append(f"- **Resolution:** Hourly")
            elif avg_obs_per_day >= 6:
                report_lines.append(f"- **Resolution:** 3-6 hourly (synoptic)")
            else:
                report_lines.append(f"- **Resolution:** Sparse (< 6/day)")

            # Validation period stats (2011-2020)
            val_data = df[(df['datetime'] >= '2011-01-01') & (df['datetime'] <= '2020-12-31')]
            if len(val_data) > 0:
                val_temp_pct = val_data['air_temperature'].notna().sum() / len(val_data) * 100
                report_lines.append(f"\n### Validation Period (2011-2020)")
                report_lines.append(f"- **Records:** {len(val_data):,}")
                report_lines.append(f"- **Temperature coverage:** {val_temp_pct:.1f}%")

            # Baseline period stats (1991-2010)
            base_data = df[(df['datetime'] >= '1991-01-01') & (df['datetime'] <= '2010-12-31')]
            if len(base_data) > 0:
                base_temp_pct = base_data['air_temperature'].notna().sum() / len(base_data) * 100
                report_lines.append(f"\n### Baseline Period (1991-2010)")
                report_lines.append(f"- **Records:** {len(base_data):,}")
                report_lines.append(f"- **Temperature coverage:** {base_temp_pct:.1f}%")

            report_lines.append("\n---\n")
        else:
            report_lines.append(f"\n**ERROR:** File not found: {csv_file}\n")
            report_lines.append("---\n")

    # Write report
    report_file = OUTPUT_DIR / "DATA_QUALITY_REPORT.md"
    report_file.write_text('\n'.join(report_lines))
    logger.info(f"✓ Quality report saved: {report_file}")

    return report_file


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("ISD FULL HISTORY DOWNLOAD & STITCHING")
    logger.info("="*80)
    logger.info(f"Cities: {len(STATIONS)}")
    logger.info(f"Year range: {min(YEARS)}-{max(YEARS)}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("")

    # Download all cities
    for city_name, info in STATIONS.items():
        download_city_full_history(city_name, info["code"])

    # Stitch all cities
    logger.info("\n" + "="*80)
    logger.info("STITCHING COMPLETE DATASETS")
    logger.info("="*80)

    for city_name in STATIONS.keys():
        stitch_city_data(city_name)

    # Generate quality report
    generate_quality_report()

    logger.info("\n" + "="*80)
    logger.info("✓ COMPLETE")
    logger.info("="*80)
    logger.info(f"\nOutput files:")
    logger.info(f"  - {OUTPUT_DIR}/ISD_complete_*.csv (4 cities)")
    logger.info(f"  - {OUTPUT_DIR}/DATA_QUALITY_REPORT.md")
