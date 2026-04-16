import pandas as pd
import numpy as np
import xarray as xr
import os
import glob
import zipfile
import io
import argparse
from tqdm import tqdm

def wrap_lon(lon):
    """Convert longitude from [-180,180] to [0,360] if needed."""
    return lon % 360

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract CMIP5 climate data time series for cities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract daily CMIP5 data with default settings
  python extract_CMIP5.py -i data/CMIP5/CMIP5_mpi_esm_lr_rcp85_r1i1p1_daily -o data/CMIP5
  
  # Extract monthly CMIP5 data
  python extract_CMIP5.py -i data/CMIP5/CMIP5_mpi_esm_lr_historical_r1i1p1_monthly -o data/CMIP5 -e historical
  
  # Extract specific years for rcp85 daily data
  python extract_CMIP5.py -i data/CMIP5/CMIP5_mpi_esm_lr_rcp85_r1i1p1_daily -o data/CMIP5 -e rcp85 --year-start 2020 --year-end 2050
  
  # Extract from base directory containing multiple subdirectories
  python extract_CMIP5.py -i data/CMIP5 -o data/CMIP5 -e historical rcp85
        """
    )

    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=False,
        default='data/CMIP5',
        help='Input directory containing CMIP5 data subdirectories (default: data/CMIP5)'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=False,
        default='data/CMIP5',
        help='Output directory for extracted CSV files (default: data/CMIP5)'
    )

    parser.add_argument(
        '-c', '--city-config',
        type=str,
        default='data/cities_config_final.csv',
        help='Path to cities configuration CSV file (default: data/cities_config_final.csv)'
    )

    parser.add_argument(
        '-e', '--experiment',
        type=str,
        nargs='+',
        default=['historical', 'rcp85'],
        choices=['historical', 'rcp85'],
        help='Experiment(s) to process (default: historical rcp85)'
    )

    parser.add_argument(
        '--year-start',
        type=int,
        default=None,
        help='Start year for data extraction (default: all available years)'
    )

    parser.add_argument(
        '--year-end',
        type=int,
        default=None,
        help='End year for data extraction (default: all available years)'
    )

    return parser.parse_args()

def load_cities(config_path):
    """Load cities from config file."""
    df = pd.read_csv(config_path)
    cities = {}

    for _, row in df.iterrows():
        cities[row['city']] = {
            'lat': row['latitude'],
            'lon': row['longitude'],
            'country': row['country']
        }

    return cities

def get_year_from_filename(filename):
    """Extract year range from CORDEX filename."""
    # Examples:
    # NAM-22_gerics_remo2015_historical_r1i1p1_1970.nc -> (1970, 1970)
    # NAM-22_gerics_remo2015_historical_r1i1p1_1971_1975.nc -> (1971, 1975)
    # NAM-22_gerics_remo2015_rcp_8_5_r1i1p1_2006_2010.nc -> (2006, 2010)
    parts = filename.replace('.nc', '').split('_')

    # Get the last parts which are years
    try:
        if parts[-1].isdigit() and len(parts[-1]) == 4:
            year_end = int(parts[-1])
            if parts[-2].isdigit() and len(parts[-2]) == 4:
                year_start = int(parts[-2])
                return (year_start, year_end)
            else:
                return (year_end, year_end)
    except:
        pass

    return (None, None)

def filter_files_by_year(nc_files, year_start, year_end):
    """Filter NetCDF files based on year range."""
    if year_start is None and year_end is None:
        return nc_files

    filtered_files = []
    for nc_file in nc_files:
        filename = os.path.basename(nc_file)
        file_year_start, file_year_end = get_year_from_filename(filename)

        if file_year_start is None or file_year_end is None:
            continue

        # Check if file overlaps with requested year range
        if year_start is not None and file_year_end < year_start:
            continue
        if year_end is not None and file_year_start > year_end:
            continue

        filtered_files.append(nc_file)

    return filtered_files

def process_cmip5_file(nc_file, cities, city_records, year_start=None, year_end=None):
    """
    Process a single CMIP5 file (zip or nc) containing multiple NetCDF files (one per variable).

    Args:
        nc_file: Path to zip or nc file
        cities: Dict of {city_name: {'lat': lat, 'lon': lon, ...}}
        city_records: Dict to accumulate data {city: {var_name: [df1, df2, ...]}}
        year_start: Optional start year filter
        year_end: Optional end year filter
    """
    try:
        with zipfile.ZipFile(nc_file, 'r') as z:
            nc_list = [name for name in z.namelist() if name.endswith('.nc')]
            print(f"    Found {len(nc_list)} NetCDF files in zip")

            for nc_name in nc_list:
                var_name = os.path.basename(nc_name).split('_')[0]  # variable name
                try:
                    with z.open(nc_name) as f:
                        nc_bytes = io.BytesIO(f.read())
                        ds = xr.open_dataset(nc_bytes)
                except Exception as e:
                    print(f"      Failed to open {nc_name}: {e}")
                    continue

                try:
                    da_var = ds[var_name]

                    # CMIP5 data has 1D lat/lon arrays
                    if 'lat' in ds.coords and 'lon' in ds.coords:
                        lat_arr = ds['lat'].values  # 1D array
                        lon_arr = ds['lon'].values  # 1D array
                    else:
                        print(f"      Skipping variable {var_name}: lat/lon coordinates not found")
                        ds.close()
                        continue

                    # Debug: print coordinate info for first variable only
                    if var_name == os.path.basename(nc_list[0]).split('_')[0]:
                        print(f"      Coordinate system: lat shape={lat_arr.shape}, lon shape={lon_arr.shape}")

                    for city, info in cities.items():
                        lat = info['lat']
                        lon = info['lon']

                        # Wrap city longitude to [0,360] to match CMIP5 grid
                        # CMIP5 uses 0-360 format, city config uses -180 to 180
                        lon_target = wrap_lon(lon)

                        # Find nearest grid point in 1D arrays
                        lat_idx = np.argmin(np.abs(lat_arr - lat))
                        lon_idx = np.argmin(np.abs(lon_arr - lon_target))

                        # Debug: print matched coordinates for first variable only
                        if var_name == os.path.basename(nc_list[0]).split('_')[0]:
                            grid_lat = float(lat_arr[lat_idx])
                            grid_lon = float(lon_arr[lon_idx])
                            print(f"      {city:15s} target=({lat:7.3f}, {lon:8.3f}) -> ({lat:7.3f}, {lon_target:7.3f})  "
                                  f"grid=({grid_lat:7.3f}, {grid_lon:7.3f})  idx=({lat_idx},{lon_idx})")

                        # Select time series: data is (time, lat, lon)
                        da_city = da_var[:, lat_idx, lon_idx]

                        times = da_city['time'].values
                        values = da_city.values

                        # Filter by year if specified
                        if year_start is not None or year_end is not None:
                            df_temp = pd.DataFrame({'time': times, var_name: values})
                            df_temp['time'] = pd.to_datetime(df_temp['time'])

                            if year_start is not None:
                                df_temp = df_temp[df_temp['time'].dt.year >= year_start]
                            if year_end is not None:
                                df_temp = df_temp[df_temp['time'].dt.year <= year_end]

                            if df_temp.empty:
                                continue

                            times = df_temp['time'].values
                            values = df_temp[var_name].values

                        df_city = pd.DataFrame({
                            'time': times,
                            var_name: values
                        })

                        # Append to city_records
                        if var_name not in city_records[city]:
                            city_records[city][var_name] = []
                        city_records[city][var_name].append(df_city)

                except Exception as e:
                    print(f"      Error processing variable {var_name}: {e}")
                finally:
                    ds.close()

    except Exception as e:
        print(f"    Failed to open zip file {nc_file}: {e}")
        return

def process_experiment(base_dir, output_dir, experiment, cities, year_start=None, year_end=None):
    """
    Process all NetCDF files for a given experiment from all subdirectories.

    Args:
        base_dir: Base directory containing CMIP5 subdirectories
        output_dir: Output directory for CSV files
        experiment: 'historical' or 'rcp85'
        cities: Dict of {city_name: {'lat': lat, 'lon': lon, ...}}
        year_start: Optional start year filter
        year_end: Optional end year filter
    """
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    # Find all subdirectories in base_dir
    all_subdirs = [d for d in os.listdir(base_dir)
                   if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]

    if not all_subdirs:
        print(f"No subdirectories found in {base_dir}")
        return

    # Filter subdirectories that match the experiment (optional, process all if no match)
    matching_subdirs = []
    for d in all_subdirs:
        d_lower = d.lower()
        if experiment == 'historical' and 'historical' in d_lower:
            matching_subdirs.append(d)
        elif experiment == 'rcp85' and ('rcp85' in d_lower or 'rcp_8_5' in d_lower or 'rcp_85' in d_lower):
            matching_subdirs.append(d)

    # If no matching subdirs, process all (backward compatibility)
    subdirs = matching_subdirs if matching_subdirs else all_subdirs

    print(f"\nProcessing {experiment}")
    print(f"Found {len(subdirs)} relevant subdirectories in {base_dir}: {', '.join(subdirs)}")

    # Process each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)

        # Detect temporal resolution from subdirectory name
        temporal_resolution = "unknown"
        if "_daily" in subdir.lower():
            temporal_resolution = "daily"
        elif "_monthly" in subdir.lower():
            temporal_resolution = "monthly"

        # Find all NetCDF files in this subdirectory
        all_nc_files = sorted(glob.glob(os.path.join(subdir_path, "*.nc")))

        # Filter files by experiment name in filename
        nc_files = []
        for nc_file in all_nc_files:
            filename = os.path.basename(nc_file).lower()
            # Check if file matches the experiment
            if experiment == 'historical' and 'historical' in filename:
                nc_files.append(nc_file)
            elif experiment == 'rcp85' and ('rcp85' in filename or 'rcp_8_5' in filename or 'rcp_85' in filename):
                nc_files.append(nc_file)

        # Filter by year
        nc_files = filter_files_by_year(nc_files, year_start, year_end)

        if len(nc_files) == 0:
            print(f"  No {experiment} NetCDF files found in {subdir}")
            continue

        print(f"\n  Subdirectory: {subdir}")
        print(f"  Temporal resolution: {temporal_resolution}")
        print(f"  Found {len(nc_files)} {experiment} NetCDF files")
        if year_start or year_end:
            print(f"  Year filter: {year_start or 'start'} to {year_end or 'end'}")

        # Process each NetCDF file separately and save for each city
        for nc_file in tqdm(nc_files, desc=f"  {subdir}"):
            print(f"    Processing: {os.path.basename(nc_file)}")

            # Initialize city records for this file
            city_records = {city: {} for city in cities}

            # Process the file
            process_cmip5_file(nc_file, cities, city_records, year_start, year_end)

            # Get output filename from original nc filename (remove .nc extension)
            output_filename = os.path.basename(nc_file).replace('.nc', '.csv')

            # Add temporal resolution suffix for monthly data
            is_monthly = (
                temporal_resolution == "monthly"
                or "_monthly" in os.path.basename(nc_file).lower()
            )
            if is_monthly:
                output_filename = output_filename.replace('.csv', '_monthly.csv')

            # Save data for each city
            for city, var_dict in city_records.items():
                if not var_dict:
                    continue

                df_all = None
                for var_name, df_list in var_dict.items():
                    if not df_list:
                        continue

                    # Concatenate all time periods for this variable
                    df_var = pd.concat(df_list, ignore_index=True)
                    df_var = df_var.sort_values('time').reset_index(drop=True)

                    if df_all is None:
                        df_all = df_var
                    else:
                        df_all = pd.merge(df_all, df_var, on='time', how='outer')

                if df_all is None or df_all.empty:
                    continue

                # Save CSV with original filename
                city_folder = os.path.join(output_dir, city.replace(' ', '_'))
                os.makedirs(city_folder, exist_ok=True)
                output_file = os.path.join(city_folder, output_filename)

                df_all = df_all.sort_values('time').reset_index(drop=True)
                df_all.to_csv(output_file, index=False)
                print(f"      Saved: {city.replace(' ', '_')}/{output_filename} ({len(df_all)} time steps, {len(df_all.columns)-1} variables)")

def extract_model_info_from_filename(filename):
    """
    Extract model information from CMIP5 filename.

    Example: mpi_esm_lr_rcp_8_5_r1i1p1_2006_2009.nc
    Returns: (model_name, experiment, ensemble, year_start, year_end)
    """
    basename = os.path.basename(filename).replace('.nc', '')
    parts = basename.split('_')

    # Find year information (last 1 or 2 parts should be years)
    year_start = None
    year_end = None

    if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 4:
        year_end = parts[-1]
        if parts[-2].isdigit() and len(parts[-2]) == 4:
            year_start = parts[-2]
            year_info_len = 2
        else:
            year_start = year_end
            year_info_len = 1
    else:
        year_info_len = 0

    # Remove year parts
    name_parts = parts[:-year_info_len] if year_info_len > 0 else parts

    # Extract model name (usually first few parts before experiment)
    # Try to identify experiment type
    experiment = None
    model_parts = []
    ensemble = None

    for i, part in enumerate(name_parts):
        if 'historical' in part.lower():
            experiment = 'historical'
            model_parts = name_parts[:i]
            if i + 1 < len(name_parts):
                ensemble = name_parts[i + 1]
            break
        elif 'rcp' in part.lower():
            # Handle rcp_8_5, rcp85, rcp_85, etc.
            if i + 2 < len(name_parts) and name_parts[i+1] in ['8', '85'] and name_parts[i+2] in ['5', '']:
                experiment = 'rcp85'
                model_parts = name_parts[:i]
                if i + 3 < len(name_parts):
                    ensemble = name_parts[i + 3]
            elif i + 1 < len(name_parts) and name_parts[i+1] in ['8', '85']:
                experiment = 'rcp85'
                model_parts = name_parts[:i]
                if i + 2 < len(name_parts):
                    ensemble = name_parts[i + 2]
            else:
                experiment = 'rcp85'
                model_parts = name_parts[:i]
                if i + 1 < len(name_parts):
                    ensemble = name_parts[i + 1]
            break

    if not model_parts:
        model_parts = name_parts

    model_name = '_'.join(model_parts)

    return model_name, experiment, ensemble, year_start, year_end

def main():
    """Main function to process experiments."""
    args = parse_args()

    # Print configuration
    print("="*80)
    print("CMIP5 Data Extraction")
    print("="*80)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"City config:      {args.city_config}")
    print(f"Experiments:      {', '.join(args.experiment)}")
    if args.year_start or args.year_end:
        print(f"Year range:       {args.year_start or 'start'} to {args.year_end or 'end'}")
    else:
        print(f"Year range:       All available years")
    print("="*80)

    # Load cities
    if not os.path.exists(args.city_config):
        print(f"\nError: City config file not found: {args.city_config}")
        return

    cities = load_cities(args.city_config)

    print(f"\nLoaded cities from {args.city_config}")
    print(f"  Total: {len(cities)} cities")

    # Process each experiment
    for experiment in args.experiment:
        process_experiment(
            args.input_dir,
            args.output_dir,
            experiment,
            cities,
            args.year_start,
            args.year_end
        )

    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"Output saved to: {args.output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
