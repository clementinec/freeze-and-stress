#!/usr/bin/env python3
"""
NSRDB Weather Data Crawler
Based on NREL NSRDB GOES Aggregated PSM v4 API
Download weather data for specified cities using coordinates from a dictionary
"""

import requests
import time
import os
import pandas as pd
from typing import Dict, List, Optional, Union
import logging
from io import StringIO

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'nsrdb_download.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NSRDBDownloader:
    """NSRDB Weather Data Downloader"""

    def __init__(self, api_key: str, email: str):
        """
        Initialize downloader

        Args:
            api_key: NREL Developer API key
            email: Email address for receiving download links
        """
        self.api_key = api_key
        self.email = email

        # Different API endpoints for different datasets
        self.api_endpoints = {
            'goes_aggregated': "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download",
            'goes_full_disc': "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-full-disc-v4-0-0-download",
            'meteosat': "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-msg-v1-0-0-download"
        }

        self.session = requests.Session()

        # City coordinates dictionary with API endpoint configuration
        # Format: city_name: [longitude, latitude, api_endpoint, available_years]
        self.cities = {
            # "Los Angeles": [-118.2437, 34.0522, 'goes_aggregated', (1998, 2024)],
            # "Miami": [-80.1918, 25.7617, 'goes_aggregated', (1998, 2024)],
            # "Toronto": [-79.3832, 43.6532, 'goes_aggregated', (1998, 2024)],
            # "Montreal": [-73.5673, 45.5017, 'goes_aggregated', (1998, 2024)],
            "New York": [-74.0060, 40.7128, 'goes_aggregated', (1998, 2024)],
            # "Sao Paulo": [-46.6333, -23.5505, 'goes_full_disc', (2018, 2024)],
            # "Stockholm": [18.0686, 59.3293, 'meteosat', (2005, 2022)]
        }

        # Default weather attributes - ALL available attributes
        self.default_attributes = [
            'air_temperature',
            'alpha', 'aod', 'asymmetry',
            'clearsky_dhi', 'clearsky_dni', 'clearsky_ghi',
            'cloud_fill_flag', 'cloud_type',
            'dew_point', 'dhi', 'dni',
            'fill_flag','ghi', 'ozone',
            'relative_humidity', 'solar_zenith_angle',
            'ssa', 'surface_albedo','surface_pressure',
            'total_precipitable_water',
            'wind_direction', 'wind_speed'
        ]


    def add_city(self, city_name: str, longitude: float, latitude: float, api_endpoint: str = 'goes_aggregated', available_years: tuple = (1998, 2024)):
        """Add new city to cities dictionary with API endpoint configuration"""
        self.cities[city_name] = [longitude, latitude, api_endpoint, available_years]
        logger.info(f"Added city: {city_name} ({longitude}, {latitude}) using {api_endpoint} API for years {available_years[0]}-{available_years[1]}")

    def get_city_coordinates(self, city_name: str) -> Optional[List]:
        """Get city configuration (coordinates, API endpoint, available years)"""
        return self.cities.get(city_name)

    def get_city_available_years(self, city_name: str) -> Optional[tuple]:
        """Get available years for a specific city"""
        city_config = self.cities.get(city_name)
        if city_config and len(city_config) >= 4:
            return city_config[3]
        return None

    def filter_years_by_availability(self, city_name: str, requested_years: List[int]) -> List[int]:
        """Filter requested years by what's available for the city"""
        available_years = self.get_city_available_years(city_name)
        if not available_years:
            return requested_years

        min_year, max_year = available_years
        filtered_years = [year for year in requested_years if min_year <= year <= max_year]

        if len(filtered_years) < len(requested_years):
            skipped_years = [year for year in requested_years if year not in filtered_years]
            logger.warning(f"{city_name}: Skipping years {skipped_years} (available: {min_year}-{max_year})")

        return filtered_years

    def create_wkt_point(self, longitude: float, latitude: float) -> str:
        """Create WKT point format string"""
        return f"POINT({longitude} {latitude})"

    def download_single_csv(self,
                           city_name: str,
                           years: Union[int, List[int]],  # Modified to accept single year or list of years
                           attributes: Optional[List[str]] = None,
                           interval: int = 60,
                           utc: bool = False,
                           leap_day: bool = False) -> Optional[pd.DataFrame]:
        """
        Download single city data for one or multiple years

        Args:
            city_name: City name
            years: Year (int) or list of years (List[int])
            attributes: List of attributes to download
            interval: Time interval (30 or 60 minutes)
            utc: Whether to use UTC time
            leap_day: Whether to include leap day

        Returns:
            pandas DataFrame or None
        """
        coordinates = self.get_city_coordinates(city_name)
        if not coordinates:
            logger.error(f"City {city_name} coordinates not found")
            return None

        longitude, latitude, api_endpoint, available_years = coordinates
        wkt = self.create_wkt_point(longitude, latitude)

        if attributes is None:
            attributes = self.default_attributes

        # Handle both single year and multiple years
        if isinstance(years, int):
            years_str = str(years)
        else:
            years_str = ','.join(map(str, years))

        params = {
            'api_key': self.api_key,
            'wkt': wkt,
            'attributes': ','.join(attributes),
            'names': years_str,
            'utc': str(utc).lower(),
            'leap_day': str(leap_day).lower(),
            'interval': interval,
            'email': self.email
        }

        url = f"{self.api_endpoints[api_endpoint]}.csv"

        try:
            logger.info(f"Downloading {city_name} {years_str} data from {api_endpoint}...")
            response = self.session.get(url, params=params, timeout=300)
            response.raise_for_status()

            # Parse CSV data
            lines = response.text.strip().split('\n')

            # Find data start line (usually line 3)
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith('Year,Month,Day'):
                    data_start = i
                    break

            if data_start == 0:
                logger.error(f"Cannot find data header, response: {response.text[:500]}")
                return None

            # Parse data section
            data_text = '\n'.join(lines[data_start:])
            df = pd.read_csv(StringIO(data_text))

            # Add city information
            df['City'] = city_name
            df['Longitude'] = longitude
            df['Latitude'] = latitude

            logger.info(f"Successfully downloaded {city_name} {years_str} data, {len(df)} records")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {city_name} {years_str} data: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing {city_name} {years_str} data: {e}")
            return None

    def download_multiple_years(self,
                               city_name: str,
                               years: List[int],
                               attributes: Optional[List[str]] = None,
                               interval: int = 60,
                               utc: bool = False,
                               leap_day: bool = False,
                               delay: float = 2.0) -> Optional[pd.DataFrame]:
        """
        Download single city data for multiple years (year by year to respect API limits)

        Args:
            city_name: City name
            years: List of years
            attributes: List of attributes to download
            interval: Time interval (30 or 60 minutes)
            utc: Whether to use UTC time
            leap_day: Whether to include leap day
            delay: Request delay time (seconds) to respect API limits

        Returns:
            Combined pandas DataFrame or None
        """
        all_data = []

        # Filter years by availability for the city
        years = self.filter_years_by_availability(city_name, years)

        for i, year in enumerate(years):
            if i > 0:  # No delay for first request
                logger.info(f"Waiting {delay} seconds...")
                time.sleep(delay)

            df = self.download_single_csv(
                city_name=city_name,
                years=year,  # Single year at a time
                attributes=attributes,
                interval=interval,
                utc=utc,
                leap_day=leap_day
            )

            if df is not None:
                all_data.append(df)
            else:
                logger.warning(f"Skipping year {year} for city {city_name}")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Successfully combined {len(all_data)} years data for {city_name}, total {len(combined_df)} records")
            return combined_df
        else:
            logger.error(f"No data downloaded successfully for {city_name}")
            return None

    def download_multiple_cities_csv(self,
                                   cities: List[str],
                                   years: Union[int, List[int]],  # Modified to accept single year or list of years
                                   attributes: Optional[List[str]] = None,
                                   interval: int = 60,
                                   utc: bool = False,
                                   leap_day: bool = False,
                                   delay: float = 2.0) -> Optional[pd.DataFrame]:
        """
        Download multiple cities data for one or multiple years

        Args:
            cities: List of city names
            years: Year (int) or list of years (List[int])
            attributes: List of attributes to download
            interval: Time interval (30 or 60 minutes)
            utc: Whether to use UTC time
            leap_day: Whether to include leap day
            delay: Request delay time (seconds) to respect API limits

        Returns:
            Combined pandas DataFrame or None
        """
        all_data = []

        for i, city in enumerate(cities):
            if i > 0:  # No delay for first request
                logger.info(f"Waiting {delay} seconds...")
                time.sleep(delay)

            # For multiple years, download year by year to respect API limits
            if isinstance(years, list) and len(years) > 1:
                df = self.download_multiple_years(
                    city_name=city,
                    years=years,
                    attributes=attributes,
                    interval=interval,
                    utc=utc,
                    leap_day=leap_day,
                    delay=delay
                )
            else:
                df = self.download_single_csv(
                    city_name=city,
                    years=years,
                    attributes=attributes,
                    interval=interval,
                    utc=utc,
                    leap_day=leap_day
                )

            if df is not None:
                all_data.append(df)
            else:
                logger.warning(f"Skipping city {city}")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Successfully combined {len(all_data)} cities data, total {len(combined_df)} records")
            return combined_df
        else:
            logger.error("No data downloaded successfully")
            return None

    def save_data(self, df: pd.DataFrame, filename: str = None, output_dir: str = "./data/NSRDB"):
        """
        Save data to file with automatic naming based on city and year data

        Args:
            df: DataFrame to save
            filename: File name (optional, will auto-generate if not provided)
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        # Auto-generate filename if not provided
        if filename is None:
            if 'City' in df.columns and 'Year' in df.columns:
                cities = df['City'].unique()
                years = sorted(df['Year'].unique())

                if len(cities) == 1:
                    # Single city
                    city_name = cities[0].replace(' ', '_').lower()
                    if len(years) == 1:
                        # Single year
                        filename = f"nsrdb_{city_name}_{years[0]}.csv"
                    else:
                        # Multiple years
                        year_range = f"{years[0]}-{years[-1]}"
                        filename = f"nsrdb_{city_name}_{year_range}.csv"
                else:
                    # Multiple cities
                    if len(years) == 1:
                        # Single year
                        filename = f"nsrdb_{years[0]}.csv"
                    else:
                        # Multiple years
                        year_range = f"{years[0]}-{years[-1]}"
                        filename = f"nsrdb_multi_cities_{year_range}.csv"
            else:
                filename = "nsrdb_data.csv"

        filepath = os.path.join(output_dir, filename)

        try:
            if filename.endswith('.csv'):
                df.to_csv(filepath, index=False)
            elif filename.endswith('.xlsx'):
                df.to_excel(filepath, index=False)
            else:
                # Default save as CSV
                filepath += '.csv'
                df.to_csv(filepath, index=False)

            logger.info(f"Data saved to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            return None

    def list_available_cities(self) -> Dict[str, List[float]]:
        """List all available cities and their coordinates"""
        return self.cities.copy()


def main():
    """Main function example"""

    # API configuration (replace with actual API key and email)
    API_KEY = "IDOcTRUqCMxGhuw8jOO93jjTd6vr0us6cmYhqUFu"  # Get from https://developer.nrel.gov/signup/
    EMAIL = "kanxuan.he42@gmail.com"

    if API_KEY == "YOUR_API_KEY_HERE":
        print("Please set correct API key and email address")
        print("Get API key from: https://developer.nrel.gov/signup/")
        return

    # Create downloader instance
    downloader = NSRDBDownloader(api_key=API_KEY, email=EMAIL)

    # Show available cities with their API endpoints and available years
    print("Available cities and their configurations:")
    for city, config in downloader.list_available_cities().items():
        longitude, latitude, api_endpoint, available_years = config
        print(f"  {city}:")
        print(f"    Coordinates: [{longitude:.4f}, {latitude:.4f}]")
        print(f"    API Endpoint: {api_endpoint}")
        print(f"    Available Years: {available_years[0]}-{available_years[1]}")
        print()

    # Download each city individually with data from their available year ranges
    cities_to_download = [
        # "Los Angeles",
        # "Miami",
        # "Toronto",
        # "Montreal",
        "New York",
        # "Sao Paulo",
        # "Stockholm"
    ]

    # Use full range - each city will be filtered to its available years
    years_range = list(range(1998, 2025))  # 1998 to 2024 inclusive

    print(f"Downloading data for {len(cities_to_download)} cities...")
    print("Note: Each city will be filtered to its available year range\n")

    for i, city in enumerate(cities_to_download):
        print(f"\n{'='*60}")
        print(f"Processing city {i+1}/{len(cities_to_download)}: {city}")

        # Show what years will actually be downloaded for this city
        available_years = downloader.get_city_available_years(city)
        if available_years:
            actual_years = downloader.filter_years_by_availability(city, years_range)
            print(f"Available years for {city}: {available_years[0]}-{available_years[1]} ({len(actual_years)} years)")

        print(f"{'='*60}")

        # Download data for this city using year-by-year method to respect API limits
        city_data = downloader.download_multiple_years(
            city_name=city,
            years=years_range,  # Will be filtered automatically
            delay=2.0  # Respect API limits
        )

        if city_data is not None:
            # Auto-naming: will generate "nsrdb_<city>_<year_range>.csv"
            filepath = downloader.save_data(city_data)
            print(f"✓ {city} data saved successfully")
            print(f"  Shape: {city_data.shape}")
            print(f"  Years: {min(city_data['Year'])}-{max(city_data['Year'])} ({len(city_data['Year'].unique())} years)")
            print(f"  File: {filepath}")
        else:
            print(f"✗ Failed to download data for {city}")

        # Add extra delay between cities to be safe with API limits
        if i < len(cities_to_download) - 1:  # Don't wait after the last city
            print(f"\nWaiting 5 seconds before processing next city...")
            time.sleep(5)

    print(f"\n{'='*60}")
    print("Download process completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
