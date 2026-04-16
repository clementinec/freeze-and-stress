import pandas as pd
import numpy as np
import xarray as xr
import os
import glob
import zipfile
import io
import argparse
from tqdm import tqdm
import re

def wrap_lon(lon):
    """Convert longitude from [-180,180] to [0,360] if needed."""
    return lon % 360

def _align_city_lon_to_grid(lon, lon_arr):
    """Align city longitude to the grid's longitude convention."""
    lon_min = float(np.nanmin(lon_arr))
    lon_max = float(np.nanmax(lon_arr))
    # If grid is 0..360 (common for some CORDEX datasets), wrap city lon.
    if lon_min >= 0.0 and lon_max > 180.0:
        return wrap_lon(lon)
    return lon

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract CORDEX climate data time series for cities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract NAM-22 data with default settings
  python extract_CORDEX.py -i /Volumes/T7/0weather_data/CORDEX_CORE -o data/CORDEX -d NAM-22
  
  # Extract multiple domains for historical only
  python extract_CORDEX.py -i /path/to/data -o output -d NAM-22 EUR-11 -e historical
  
  # Extract specific years for rcp85
  python extract_CORDEX.py -i /path/to/data -o output -d NAM-22 -e rcp85 --year-start 2020 --year-end 2050
        """
    )

    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=False,
        default='/Volumes/T7/0weather_data/CORDEX_CORE',
        help='Input directory containing CORDEX data (default: /Volumes/T7/0weather_data/CORDEX_CORE)'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=False,
        default='data/CORDEX_CMIP5',
        help='Output directory for extracted CSV files (default: data/CORDEX_CMIP5)'
    )

    parser.add_argument(
        '-c', '--city-config',
        type=str,
        default='data/cities_config_final.csv',
        help='Path to cities configuration CSV file (default: data/cities_config_final.csv)'
    )

    parser.add_argument(
        '-d', '--domain',
        type=str,
        nargs='+',
        required=True,
        help='CORDEX domain(s) to process (e.g., NAM-22 EUR-11 EAS-22 SEA-22)'
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        default=None,
        help='Model name filter (case-insensitive; e.g., RegCM, REMO2015). If omitted, process all models.'
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

def load_cities_by_domain(config_path):
    """Load cities from config file, grouped by CORDEX domain."""
    df = pd.read_csv(config_path)
    cities_by_domain = {}

    for _, row in df.iterrows():
        domain = row['cordex_domain']
        if domain not in cities_by_domain:
            cities_by_domain[domain] = {}
        cities_by_domain[domain][row['city']] = {
            'lat': row['latitude'],
            'lon': row['longitude'],
            'country': row['country']
        }

    return cities_by_domain

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

def process_cordex_zip(zip_file, cities, city_records, year_start=None, year_end=None):
    """
    Process a single CORDEX zip file containing multiple NetCDF files (one per variable).

    Args:
        zip_file: Path to zip file
        cities: Dict of {city_name: {'lat': lat, 'lon': lon, ...}}
        city_records: Dict to accumulate data {city: {var_name: [df1, df2, ...]}}
        year_start: Optional start year filter
        year_end: Optional end year filter
    """
    try:
        with zipfile.ZipFile(zip_file, 'r') as z:
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

                    # Get lat/lon arrays from dataset (not from variable)
                    # CORDEX data has 2D lat/lon arrays regardless of dimension names
                    if 'lat' in ds.coords and 'lon' in ds.coords:
                        lat_arr = ds['lat'].values
                        lon_arr = ds['lon'].values
                    else:
                        print(f"      Skipping variable {var_name}: lat/lon coordinates not found")
                        ds.close()
                        continue

                    for city, info in cities.items():
                        lat = info['lat']
                        lon = info['lon']

                        # Align city longitude to grid convention (0..360 or -180..180)
                        lon_target = _align_city_lon_to_grid(lon, lon_arr)

                        # Find nearest grid point
                        dist2 = (lat_arr - lat)**2 + (lon_arr - lon_target)**2
                        idx_flat = dist2.argmin()
                        y_idx, x_idx = np.unravel_index(idx_flat, lat_arr.shape)

                        # Debug: print matched coordinates for first variable only
                        if var_name == os.path.basename(nc_list[0]).split('_')[0]:
                            grid_lat = float(lat_arr[y_idx, x_idx])
                            grid_lon = float(lon_arr[y_idx, x_idx])
                            print(f"      {city:15s} target=({lat:7.3f}, {lon:8.3f}) -> ({lat:7.3f}, {lon_target:7.3f})  "
                                  f"grid=({grid_lat:7.3f}, {grid_lon:7.3f})  idx=({y_idx},{x_idx})")

                        # Select time series based on dimension names
                        if 'rlat' in da_var.dims and 'rlon' in da_var.dims:
                            da_city = da_var[:, y_idx, x_idx]
                        elif 'lat' in da_var.dims and 'lon' in da_var.dims:
                            da_city = da_var[:, y_idx, x_idx]
                        else:
                            # Try generic indexing
                            da_city = da_var[:, y_idx, x_idx]

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
        print(f"    Failed to open zip file {zip_file}: {e}")
        return

def _dataset_folder(domain: str, model: str = "REMO2015", ensemble: str = "r1i1p1") -> str:
    """Return output folder name, independent of experiment."""
    return f"{domain}_{model}_{ensemble}"


def _experiment_tokens(experiment: str) -> list[str]:
    """Return acceptable folder-name tokens for a given experiment."""
    if experiment == "historical":
        return ["historical"]
    if experiment == "rcp85":
        # Common CORDEX naming variants across providers
        return ["rcp85", "rcp8.5", "rcp_8_5", "rcp_8.5", "rcp-8.5"]
    return [experiment]


def _iter_candidate_experiment_dirs(base_dir: str, domain: str, experiment: str, model: str | None = None) -> list[str]:
    """Discover experiment directories under base_dir by matching domain + experiment tokens.

    We intentionally *don't* require ensemble like r1i1p1 because it varies by dataset.
    """
    if not os.path.isdir(base_dir):
        return []

    domain_l = domain.lower()
    exp_tokens = [t.lower() for t in _experiment_tokens(experiment)]

    candidates: list[str] = []
    model_l = model.lower() if model else None

    for entry in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, entry)
        if not os.path.isdir(full):
            continue

        name_l = entry.lower()
        if domain_l not in name_l:
            continue
        if not any(tok in name_l for tok in exp_tokens):
            continue
        if model_l and model_l not in name_l:
            continue

        candidates.append(full)

    return candidates


def _infer_model_ensemble_from_dirname(dirname: str, default_model: str = "REMO2015") -> tuple[str, str]:
    """Best-effort parse of model + ensemble from an experiment directory name.

    Expected common patterns:
      <DOMAIN>_<something>_<model>_<experiment>_<ensemble>
      <DOMAIN>_<model>_<experiment>_<ensemble>

    If unsure, we fall back to defaults that keep outputs stable.
    """
    base = os.path.basename(dirname)

    # Ensemble like r1i1p1, r3i1p1f1, etc.
    m_ens = re.search(r"(r\d+i\d+p\d+(?:f\d+)?)", base)
    ensemble = m_ens.group(1) if m_ens else "unknown-ens"

    # Model: try to locate token right before experiment token
    parts = base.split('_')
    parts_l = [p.lower() for p in parts]

    exp_idx = None
    for tok in ("historical", "rcp85", "rcp8.5"):
        if tok in parts_l:
            exp_idx = parts_l.index(tok)
            break

    model = default_model
    if exp_idx is not None and exp_idx - 1 >= 0:
        cand = parts[exp_idx - 1]
        # Avoid capturing domain itself or empty tokens
        if cand and cand.upper() != parts[0].upper():
            model = cand.upper() if cand.isupper() else cand

    return model, ensemble


def process_experiment(base_dir, output_dir, domain, experiment, cities, year_start=None, year_end=None, model_filter=None):
    """
    Process all NetCDF/zip files for a given domain and experiment.

    This version discovers experiment folders by matching `domain` only (no fixed ensemble).

    Args:
        base_dir: Base directory containing CORDEX data
        output_dir: Output directory for CSV files
        domain: CORDEX domain (e.g., 'NAM-22')
        experiment: 'historical' or 'rcp85'
        cities: Dict of {city_name: {'lat': lat, 'lon': lon, ...}}
        year_start: Optional start year filter
        year_end: Optional end year filter
    """
    exp_dirs = _iter_candidate_experiment_dirs(
        base_dir=base_dir,
        domain=domain,
        experiment=experiment,
        model=model_filter,
    )

    if not exp_dirs:
        print(f"No directories found for domain={domain} experiment={experiment} under {base_dir}")
        if model_filter:
            print(f"  Model filter: {model_filter}")
        print(f"  Searched for experiment tokens: {', '.join(_experiment_tokens(experiment))}")
        return

    # Group by output dataset folder (domain_model_ensemble); process each separately.
    dirs_by_dataset: dict[str, list[str]] = {}
    for d in exp_dirs:
        model, ensemble = _infer_model_ensemble_from_dirname(d)
        dataset_folder = _dataset_folder(domain=domain, model=model, ensemble=ensemble)
        dirs_by_dataset.setdefault(dataset_folder, []).append(d)

    for dataset_folder, exp_dirs_for_dataset in dirs_by_dataset.items():
        # Initialize city records per dataset (so different ensembles don't get mixed)
        city_records = {city: {} for city in cities}

        # Collect container files from all matching dirs for this dataset
        container_files: list[str] = []
        for exp_dir in exp_dirs_for_dataset:
            # Some distributions use .nc (which in this repo is treated as a zip container of per-var nc's)
            # Others use .zip explicitly.
            container_files.extend(sorted(glob.glob(os.path.join(exp_dir, "*.nc"))))
            container_files.extend(sorted(glob.glob(os.path.join(exp_dir, "*.zip"))))

        # Dedupe while preserving order
        seen = set()
        container_files = [p for p in container_files if not (p in seen or seen.add(p))]

        # Filter by year based on filename patterns
        container_files = filter_files_by_year(container_files, year_start, year_end)

        print(f"\nProcessing {domain} - {experiment}")
        print(f"Dataset: {dataset_folder}")
        print(f"Matched dirs ({len(exp_dirs_for_dataset)}):")
        for d in exp_dirs_for_dataset:
            print(f"  - {d}")
        print(f"Found {len(container_files)} container files")
        if year_start or year_end:
            print(f"  Year filter: {year_start or 'start'} to {year_end or 'end'}")

        if len(container_files) == 0:
            print("No files found matching criteria")
            continue

        for path in tqdm(container_files, desc=f"{domain}-{experiment}"):
            print(f"  Processing: {os.path.basename(path)}")
            # Existing logic expects a zipfile-like container. Many datasets use .nc as a zip.
            process_cordex_zip(path, cities, city_records, year_start, year_end)

        # Merge variables and save for each city
        for city, var_dict in city_records.items():
            if not var_dict:
                print(f"  No data extracted for {city}")
                continue

            df_all = None
            for var_name, df_list in var_dict.items():
                if not df_list:
                    continue

                df_var = pd.concat(df_list, ignore_index=True)
                df_var = df_var.sort_values('time').reset_index(drop=True)

                if df_all is None:
                    df_all = df_var
                else:
                    df_all = pd.merge(df_all, df_var, on='time', how='outer')

            if df_all is None or df_all.empty:
                print(f"  No data to save for {city}")
                continue

            city_folder = os.path.join(output_dir, dataset_folder, city.replace(' ', '_'))
            os.makedirs(city_folder, exist_ok=True)
            output_file = str(os.path.join(city_folder, f"{experiment}.csv"))
            df_all = df_all.sort_values('time').reset_index(drop=True)
            df_all.to_csv(output_file, index=False)
            print(f"  Saved: {output_file} ({len(df_all)} time steps, {len(df_all.columns)-1} variables)")

def main():
    """Main function to process all domains and experiments."""
    args = parse_args()

    # Print configuration
    print("="*80)
    print("CORDEX Data Extraction")
    print("="*80)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"City config:      {args.city_config}")
    print(f"Domains:          {', '.join(args.domain)}")
    print(f"Experiments:      {', '.join(args.experiment)}")
    if args.model:
        print(f"Model filter:     {args.model}")
    if args.year_start or args.year_end:
        print(f"Year range:       {args.year_start or 'start'} to {args.year_end or 'end'}")
    else:
        print(f"Year range:       All available years")
    print("="*80)

    # Load cities grouped by domain
    if not os.path.exists(args.city_config):
        print(f"\nError: City config file not found: {args.city_config}")
        return

    cities_by_domain = load_cities_by_domain(args.city_config)

    print(f"\nLoaded cities from {args.city_config}")
    for domain, cities in cities_by_domain.items():
        print(f"  {domain}: {len(cities)} cities - {', '.join(cities.keys())}")

    # Process each requested domain
    for domain in args.domain:
        if domain not in cities_by_domain:
            print(f"\nWarning: No cities found for domain {domain}, skipping...")
            continue

        cities = cities_by_domain[domain]
        if not cities:
            print(f"\nWarning: No cities to process for {domain}, skipping...")
            continue

        print(f"\n{'='*80}")
        print(f"Processing {domain} domain with {len(cities)} cities")
        print(f"{'='*80}")

        # Process each experiment
        for experiment in args.experiment:
            process_experiment(
                args.input_dir,
                args.output_dir,
                domain,
                experiment,
                cities,
                args.year_start,
                args.year_end,
                model_filter=args.model
            )

    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"Output saved to: {args.output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
