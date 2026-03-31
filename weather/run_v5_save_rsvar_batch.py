#!/usr/bin/env python3
"""
V5 Climate-Delta Pipeline: Save RS-VAR Output for Reuse

This script runs the RS-VAR model and saves the output (baseline + anomaly)
WITHOUT applying climate delta. This allows reusing the same RS-VAR output
with different GCM-RCM climate projections.

INPUT DATA FORMAT:
    - ISD data files are assumed to be in LST (Local Standard Time), no timezone conversion is performed
    - Training uses ALL available data with year <= 2010 (following val_vs_isd_v2.py logic)
    - Seasonal profiles are computed from baseline period only (e.g., 1991-2010)

TRAINING DATA:
    - RS-VAR models are trained on ALL historical data up to 2010
    - This maximizes the amount of training data available
    - Baseline period (1991-2010) is used ONLY for:
        * Computing seasonal profiles (month-hour climatology)
        * Computing baseline statistics for climate delta reference

OUTPUT:
    - Excel file with sheets:
        - "RSVAR_Output": Hourly baseline + zero-mean anomaly (no climate delta)
        - "Metadata": Pipeline configuration and baseline statistics
        - "Seasonal_Profile": Month-hour climatology from ISD baseline

    - The RSVAR_Output can then be combined with ANY climate delta using
      the companion script: apply_climate_delta.py

USAGE:
    python run_v5_save_rsvar.py
    python run_v5_save_rsvar.py --output my_rsvar_output.xlsx

BATCH USAGE (this script):
    # Run all cities found in data/ISD_complete_solar/*_with_solar.csv
    python run_v5_save_rsvar_batch.py

    # Run one or more cities
    python run_v5_save_rsvar_batch.py --city Los_Angeles
    python run_v5_save_rsvar_batch.py --city Los_Angeles Miami

NOTE:
    This script generates an RSVAR-only baseline+anomaly output and does NOT
    require any CORDEX input. Apply deltas later with apply_climate_delta.py.
"""

import sys
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR

from HGScripts_Dev.forecast_to_2100_v5_fixed_baseline import ForecastTo2100V5


# ==========================================================================
# CONFIGURATION
# ==========================================================================
CITIES_CONFIG_PATH = REPO_ROOT / "data" / "cities_config.csv"

BASELINE_START = 1991
BASELINE_END = 2010
FORECAST_START = "2025-01-01"
FORECAST_END = "2100-12-31"

# New defaults for batch mode
DEFAULT_ISD_MERGED_DIR = SCRIPT_DIR / "data" / "ISD_complete_solar"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "RSVAR"


# ==========================================================================
# HELPER FUNCTIONS
# ==========================================================================

def _load_cities_config(path: Path) -> pd.DataFrame:
    """Load cities_config.csv and return a DataFrame with at least city + latitude.

    Expected header includes:
    continent,kg_label,ashrae_label,city,country,latitude,longitude,cordex_domain
    """
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

    # Normalize types
    df = df.copy()
    df["city"] = df["city"].astype(str)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")

    bad = df[df["latitude"].isna()]["city"].tolist()
    if bad:
        raise SystemExit(f"ERROR: Invalid latitude for cities: {bad}")

    # Drop duplicate city definitions (keep first)
    df = df.drop_duplicates(subset=["city"], keep="first")

    return df


def _normalize_city_args(requested: list[str] | None, available_cities: list[str]) -> list[str]:
    """Resolve requested city list (case-sensitive) and validate."""
    if not requested:
        return sorted(available_cities)

    avail_set = set(available_cities)
    missing = [c for c in requested if c not in avail_set]
    if missing:
        avail = ", ".join(sorted(avail_set))
        raise SystemExit(
            f"ERROR: City(ies) not found in {CITIES_CONFIG_PATH}: {missing}.\n"
            f"Available cities: {avail}"
        )
    return requested


class RSVAROnlyPipeline(ForecastTo2100V5):
    """Modified pipeline that outputs RS-VAR results WITHOUT climate delta."""

    def __init__(self, isd_solar_path, lat=34.0, baseline_start=1991, baseline_end=2010, seed=42):
        # ForecastTo2100V5 requires a CORDEX path, but RSVAR-only runs never load or use CORDEX.
        # Provide a safe dummy value to satisfy the base class constructor.
        dummy_cordex = os.devnull
        super().__init__(
            isd_solar_path=isd_solar_path,
            cordex_rcp85_path=dummy_cordex,
            lat=lat,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            seed=seed,
        )

    def forecast_rsvar_only(self, target_vars, start_date='2025-01-01',
                            end_date='2100-12-31', chunk_years=5):
        """Forecast using RS-VAR only (baseline + zero-mean anomaly)."""
        print("\n" + "="*80)
        print("GENERATING RS-VAR OUTPUT (NO CLIMATE DELTA)")
        print("="*80)
        print(f"*** Using baseline period: {self.baseline_start}-{self.baseline_end} ***")

        forecast_index = pd.date_range(start=start_date, end=end_date, freq='h')
        n_steps = len(forecast_index)
        chunk_size = chunk_years * 8760
        n_chunks = int(np.ceil(n_steps / chunk_size))

        print(f"Forecast period: {start_date} to {end_date}")
        print(f"Total steps: {n_steps:,} hours")

        if len(self.regime_models) == 0:
            print(f"ERROR: No regime models available")
            return None

        # Forecast regime probabilities
        print(f"\n1. Forecasting regime probabilities...")
        daily_regime_probs = self._forecast_regime_probabilities(start_date, end_date)

        regime_prob_cols = sorted(
            [col for col in daily_regime_probs.columns if col.endswith('_prob')],
            key=lambda name: int(name.split('_')[1]) if name.split('_')[1].isdigit() else name
        )

        # Get initial conditions
        recent_data = self.isd_baseline[target_vars].iloc[-1000:].copy()
        recent_data['month'] = recent_data.index.month
        recent_data['hour'] = recent_data.index.hour

        deviation_vars = []
        for var in target_vars:
            if var not in self.seasonal_profiles:
                continue
            seasonal_vals = recent_data.apply(
                lambda row: self.seasonal_profiles[var][(row['month'], row['hour'])],
                axis=1
            )
            recent_data[f'{var}_deviation'] = recent_data[var] - seasonal_vals
            deviation_vars.append(f'{var}_deviation')

        hourly_regime_probs = daily_regime_probs[regime_prob_cols].reindex(forecast_index, method='ffill')
        hourly_regime_probs = hourly_regime_probs.div(hourly_regime_probs.sum(1), axis=0)

        # Forecast in chunks (NO climate delta)
        print(f"\n2. Forecasting with RS-VAR (no climate delta)...")
        all_forecasts = []

        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, n_steps)
            chunk_len = chunk_end - chunk_start
            chunk_index = forecast_index[chunk_start:chunk_end]

            print(f"\n  Chunk {chunk_idx + 1}/{n_chunks}: {chunk_len:,} hours")

            # 1. Get baseline seasonal profile
            chunk_df = pd.DataFrame(index=chunk_index, columns=target_vars, dtype=float)
            chunk_df['month'] = chunk_index.month
            chunk_df['hour'] = chunk_index.hour
            chunk_df['year'] = chunk_index.year

            for var in target_vars:
                if var not in self.seasonal_profiles:
                    continue
                seasonal_values = chunk_df.apply(
                    lambda row: self.seasonal_profiles[var][(row['month'], row['hour'])],
                    axis=1
                )
                chunk_df[var] = seasonal_values

            # 2. Probability-weighted RS-VAR deviations
            chunk_probs = hourly_regime_probs.loc[chunk_index]

            blended_deviations = pd.DataFrame(
                np.zeros((chunk_len, len(deviation_vars))),
                index=chunk_index,
                columns=deviation_vars
            )

            regime_weights = pd.DataFrame(
                0.0, index=chunk_index, columns=regime_prob_cols, dtype=float
            )

            for regime_i, var_model in sorted(self.regime_models.items()):
                regime_prob_col = f'regime_{regime_i}_prob'
                if regime_prob_col not in chunk_probs:
                    continue

                try:
                    last_deviation = recent_data[deviation_vars].iloc[-var_model.k_ar:].values
                    regime_deviation_forecast = var_model.forecast(y=last_deviation, steps=chunk_len)

                    regime_probs_vals = chunk_probs[regime_prob_col].values.reshape(-1, 1)
                    weighted_deviations = regime_deviation_forecast * regime_probs_vals

                    blended_deviations += weighted_deviations
                    regime_weights.loc[chunk_index, regime_prob_col] = chunk_probs[regime_prob_col].values

                except Exception:
                    continue

            total_weights = regime_weights.sum(axis=1).clip(lower=1e-10)
            blended_deviations = blended_deviations.div(total_weights, axis=0)

            # 3. CENTER deviations to zero-mean
            for dev_var in blended_deviations.columns:
                dev_mean = blended_deviations[dev_var].mean()
                blended_deviations[dev_var] = blended_deviations[dev_var] - dev_mean

            # 4. Add centered RS-VAR deviations to baseline profile
            for var in target_vars:
                dev_var = f'{var}_deviation'
                if dev_var in blended_deviations.columns:
                    chunk_df[var] = chunk_df[var] + blended_deviations[dev_var]

            # NOTE: NO climate delta applied here!

            # Drop helper columns
            chunk_df = chunk_df[target_vars]

            # Update recent data for next chunk
            recent_deviation_update = chunk_df.copy()
            recent_deviation_update['month'] = chunk_index.month
            recent_deviation_update['hour'] = chunk_index.hour
            for var in target_vars:
                if var not in self.seasonal_profiles:
                    continue
                profile = self.seasonal_profiles[var]
                seasonal_vals = pd.Series(
                    [float(profile.get((m, h), np.nan))
                     for m, h in zip(recent_deviation_update['month'], recent_deviation_update['hour'])],
                    index=recent_deviation_update.index,
                )
                recent_deviation_update[f'{var}_deviation'] = chunk_df[var] - seasonal_vals

            recent_data = pd.concat([recent_data, recent_deviation_update]).iloc[-1000:]

            all_forecasts.append(chunk_df)

        forecast_df = pd.concat(all_forecasts)

        # Enforce solar constraints (physical, not climate-related)
        forecast_df = self._enforce_solar_constraints(forecast_df)

        print(f"\n✓ RS-VAR forecast complete: {n_steps:,} hours")
        print(f"  Output = baseline_profile({self.baseline_start}-{self.baseline_end}) + RS_VAR_anomaly")
        print(f"  NOTE: Climate delta NOT applied - use apply_climate_delta.py")

        return forecast_df


def run_and_save_rsvar(isd_path, lat, output_path, city):
    """Run RS-VAR pipeline and save output to Excel.

    Args:
        isd_path: Path to ISD data file (assumed to be in LST - Local Standard Time)
        lat: Latitude
        output_path: Output file path
        city: City name
    """
    print("=" * 80)
    print("V5 RS-VAR PIPELINE (SAVE FOR REUSE)")
    print("=" * 80)
    print(f"\nCity: {city}")
    print(f"Inputs:")
    print(f"  ISD data:    {isd_path}")
    print(f"  Latitude:    {lat}°N")
    print(f"  Note:        ISD data assumed to be in LST (Local Standard Time)")
    print(f"\nBaseline period: {BASELINE_START}-{BASELINE_END}")
    print(f"Forecast period: {FORECAST_START} to {FORECAST_END}")

    # Initialize pipeline
    pipeline = RSVAROnlyPipeline(
        isd_solar_path=str(isd_path),
        lat=lat,
        baseline_start=BASELINE_START,
        baseline_end=BASELINE_END,
    )

    # Load data and train models
    pipeline.load_isd_data()

    # Define target variables
    target_vars = ['temp', 'pressure', 'wind_speed', 'GHI', 'DHI', 'DNI',
                   'relative_humidity', 'specific_humidity']
    target_vars = [v for v in target_vars if v in pipeline.isd_baseline.columns]

    pipeline.extract_seasonal_profiles(target_vars)
    pipeline.engineer_features()
    pipeline.standardize_features(target_vars)
    pipeline.train_regime_switching(k_regimes=3)
    pipeline.train_regime_conditional_vars(target_vars)

    # Generate RS-VAR output (no climate delta)
    rsvar_output = pipeline.forecast_rsvar_only(
        target_vars,
        start_date=FORECAST_START,
        end_date=FORECAST_END,
        chunk_years=10
    )

    # Prepare output DataFrames
    print("\n" + "="*80)
    print("SAVING TO EXCEL")
    print("="*80)

    # 1. RS-VAR Output sheet
    rsvar_df = rsvar_output.copy()
    rsvar_df.insert(0, 'datetime', rsvar_df.index)
    rsvar_df = rsvar_df.reset_index(drop=True)

    # 2. Metadata sheet
    metadata = pd.DataFrame([
        ['isd_path', str(isd_path)],
        ['lat', lat],
        ['baseline_start', BASELINE_START],
        ['baseline_end', BASELINE_END],
        ['forecast_start', FORECAST_START],
        ['forecast_end', FORECAST_END],
        ['isd_baseline_temp_mean', pipeline.isd_baseline['temp'].mean()],
        ['isd_baseline_temp_std', pipeline.isd_baseline['temp'].std()],
        ['n_hours', len(rsvar_output)],
        ['target_vars', ', '.join(target_vars)],
        ['note', 'RS-VAR output without climate delta. Use apply_climate_delta.py to add climate signal.'],
    ], columns=['parameter', 'value'])

    # 3. Seasonal Profile sheet
    profile_data = []
    for var in target_vars:
        if var in pipeline.seasonal_profiles:
            profile = pipeline.seasonal_profiles[var]
            for (month, hour), value in profile.items():
                profile_data.append({
                    'variable': var,
                    'month': month,
                    'hour': hour,
                    'value': value
                })
    profile_df = pd.DataFrame(profile_data)

    # Save to Excel
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        rsvar_df.to_excel(writer, sheet_name='RSVAR_Output', index=False)
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
        profile_df.to_excel(writer, sheet_name='Seasonal_Profile', index=False)

    print(f"\n✓ Saved to: {output_path}")
    print(f"  Sheets:")
    print(f"    - RSVAR_Output: {len(rsvar_df):,} rows")
    print(f"    - Metadata: Pipeline configuration")
    print(f"    - Seasonal_Profile: Month-hour climatology")

    # Also save as CSV for convenience
    csv_path = output_path.with_suffix('.csv')
    rsvar_df.to_csv(csv_path, index=False)
    print(f"\n✓ Also saved CSV: {csv_path}")

    return rsvar_output, pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run V5 RS-VAR Pipeline and Save Output for Reuse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--isd", type=str, default=None,
        help=(
            "Path to ISD + solar merged CSV. "
            "If omitted, batch mode will use data/ISD_hguo/merged/<city>_with_solar.csv "
            "for cities listed in data/cities_config.csv"
        )
    )
    parser.add_argument(
        "--config", type=str, default=str(CITIES_CONFIG_PATH),
        help="Path to cities_config.csv (must include columns: city, latitude)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "Output Excel file path (single-file mode) or output directory (batch mode). "
            "If omitted, writes to output/RSVAR/rsvar_output_<city>.xlsx"
        )
    )

    # Batch processing arguments
    parser.add_argument(
        "--city", type=str, nargs='*', default=None,
        help="City name(s) to process (default: all cities in data/cities_config.csv)"
    )

    args = parser.parse_args()

    cities_config_path = Path(args.config)
    cities_df = _load_cities_config(cities_config_path)
    cities_by_name = {row.city: float(row.latitude) for row in cities_df.itertuples(index=False)}

    # Determine processing mode
    # - If --isd is provided: single-file mode (requires --city with exactly 1 entry to select latitude)
    # - Else: batch mode over cities in cities_config.csv
    if args.isd:
        isd_path = Path(args.isd)
        if not isd_path.exists():
            print(f"ERROR: ISD file not found: {isd_path}")
            sys.exit(1)

        if not args.city or len(args.city) != 1:
            print("ERROR: In --isd single-file mode, you must provide exactly one --city to select latitude")
            sys.exit(1)

        city = args.city[0]
        if city not in cities_by_name:
            print(f"ERROR: City '{city}' not found in {cities_config_path}")
            sys.exit(1)

        lat = cities_by_name[city]
        output_path = Path(args.output) if args.output else (DEFAULT_OUTPUT_DIR / f"rsvar_output_{city}.xlsx")

        run_and_save_rsvar(isd_path, lat, output_path, city)

    else:
        isd_dir = DEFAULT_ISD_MERGED_DIR
        if not isd_dir.exists():
            print(f"ERROR: ISD merged directory not found: {isd_dir}")
            sys.exit(1)

        requested_cities = _normalize_city_args(args.city, list(cities_by_name.keys()))

        # In batch mode, --output must be a directory if provided
        output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
        if output_dir.suffix.lower() in {".xlsx", ".xls"}:
            print("ERROR: In batch mode, --output should be a directory (not a .xlsx file)")
            sys.exit(1)

        all_rsvar_outputs = {}
        for city in requested_cities:
            print(f"\nProcessing city: {city}")

            lat = cities_by_name[city]
            # Input file is derived from city name
            city_isd_path = isd_dir / f"{city.replace(' ', '_')}_with_solar.csv"
            if not city_isd_path.exists():
                print(f"ERROR: ISD file not found for city '{city}': {city_isd_path}")
                sys.exit(1)

            city_output_path = output_dir / f"rsvar_output_{city.replace(' ', '_')}.xlsx"

            rsvar_output, _pipeline = run_and_save_rsvar(city_isd_path, lat, city_output_path, city)
            all_rsvar_outputs[city] = rsvar_output

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print("\nNext step: Use apply_climate_delta.py to add climate signal from any GCM-RCM.")


if __name__ == "__main__":
    main()
