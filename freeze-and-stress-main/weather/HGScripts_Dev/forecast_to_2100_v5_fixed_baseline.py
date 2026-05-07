"""
Complete Pipeline V5: Fixed Baseline Climate-Delta Approach

Key improvements over V4:
1. Fixed baseline period (default 1991-2010) for BOTH ISD and CORDEX_CMIP5
2. Explicit DAILY climate delta: delta = CORDEX_CMIP5(date) - CORDEX_baseline(doy)
3. Clean three-term decomposition: output = baseline + anomaly + delta
4. Variable-specific transformations (additive vs multiplicative)
5. Optional stationarity test on baseline anomalies
6. Configurable baseline mode (fixed vs rolling)

Formula:
    T_output = mu_T_isd(doy,hour) + temp_anom + delta_T(date)

Where:
    - mu_T_isd: ISD baseline climatology (baseline period only)
    - temp_anom: RS-VAR anomaly (zero-mean, trained on baseline period)
    - delta_T: CORDEX_CMIP5 DAILY warming signal = CORDEX_CMIP5(date) - CORDEX_baseline(doy)
              Preserves full day-to-day variability from CORDEX_CMIP5
"""

import argparse
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')


class ForecastTo2100V5:
    """
    Forecast using fixed baseline climate-delta approach.

    Key parameters:
        baseline_start: Start year for baseline period (default: 1991)
        baseline_end: End year for baseline period (default: 2010)
        baseline_mode: 'fixed' or 'rolling' (default: 'fixed')
        validation_start: Start year for validation period (default: 2011)
        validation_end: End year for validation period (default: 2025)
    """

    # Variable transformation types
    ADDITIVE_VARS = {'temp', 'pressure', 'dewpoint'}
    MULTIPLICATIVE_VARS = {'wind_speed', 'GHI', 'DNI', 'DHI', 'relative_humidity', 'specific_humidity'}

    def __init__(self, isd_solar_path, cordex_rcp85_path, lat=34.0,
                 baseline_start=1991, baseline_end=2010,
                 validation_start=2011, validation_end=2025,
                 baseline_mode='fixed', seed=42):
        """
        Initialize the forecaster.

        Args:
            isd_solar_path: Path to merged ISD + solar CSV
            cordex_rcp85_path: Path to CORDEX_CMIP5 RCP8.5 CSV
            lat: Latitude for solar calculations
            baseline_start: Start year of baseline period
            baseline_end: End year of baseline period
            validation_start: Start year of validation period
            validation_end: End year of validation period
            baseline_mode: 'fixed' or 'rolling'
            seed: Random seed for regime probability forecasting
        """
        self.isd_path = isd_solar_path
        self.cordex_path = cordex_rcp85_path
        self.lat = lat

        # Baseline configuration
        self.baseline_start = baseline_start
        self.baseline_end = baseline_end
        self.validation_start = validation_start
        self.validation_end = validation_end
        self.baseline_mode = baseline_mode
        self.seed = seed

        # Data containers
        self.isd_data = None
        self.isd_baseline = None  # ISD data filtered to baseline period
        self.cordex_data = None
        self.cordex_baseline = None  # CORDEX_CMIP5 climatology for baseline period

        # Model components
        self.seasonal_profiles = {}  # Computed from baseline period only
        self.regime_probs = None
        self.regime_models = {}
        self.ms_result = None

        # Outputs
        self.forecast_hourly = None
        self.forecast_corrected = None
        self.stationarity_results = {}

    def load_isd_data(self):
        """Load ISD + solar data and extract baseline period."""
        print("="*80)
        print("STEP 1: LOADING ISD + SOLAR DATA")
        print("="*80)

        self.isd_data = pd.read_csv(self.isd_path, parse_dates=['DATE'])
        self.isd_data.set_index('DATE', inplace=True)

        if self.isd_data.index.duplicated().any():
            n_dups = self.isd_data.index.duplicated().sum()
            print(f"  Removing {n_dups:,} duplicate timestamps")
            self.isd_data = self.isd_data[~self.isd_data.index.duplicated(keep='first')]

        print(f"Loaded {len(self.isd_data):,} hourly records")
        print(f"Full period: {self.isd_data.index.min()} to {self.isd_data.index.max()}")

        # Handle missing values
        solar_vars = ['GHI', 'DNI', 'DHI']
        if all(v in self.isd_data.columns for v in solar_vars):
            print(f"\n✓ Solar radiation data found")

        for col in ['precip_depth', 'precip_duration']:
            if col in self.isd_data.columns:
                self.isd_data[col].fillna(0, inplace=True)

        for col in solar_vars:
            if col in self.isd_data.columns:
                self.isd_data.loc[self.isd_data[col] < 0, col] = 0

        # Limit forward/backward fill to 6 hours to avoid propagating bad data across long gaps
        self.isd_data.ffill(limit=6, inplace=True)
        self.isd_data.bfill(limit=6, inplace=True)
        self.isd_data.dropna(inplace=True)

        # Calculate humidity variables
        print(f"\nCalculating humidity variables...")
        self._calculate_humidity_variables()

        # Extract baseline period (for climate delta reference)
        baseline_mask = (
            (self.isd_data.index.year >= self.baseline_start) &
            (self.isd_data.index.year <= self.baseline_end)
        )
        self.isd_baseline = self.isd_data.loc[baseline_mask].copy()

        print(f"\n*** BASELINE PERIOD: {self.baseline_start}-{self.baseline_end} ***")
        print(f"    ISD baseline records: {len(self.isd_baseline):,} hours")
        print(f"    ISD baseline coverage: {self.isd_baseline.index.min()} to {self.isd_baseline.index.max()}")

        if len(self.isd_baseline) < 8760 * 5:  # Less than 5 years
            print(f"  ⚠️  WARNING: Baseline period has less than 5 years of data!")

        # Create training dataset: ALL ISD EXCEPT validation period (2011-2020)
        # This prevents data leakage when validating against 2011-2020
        training_mask = self.isd_data.index.year <= 2010
        self.isd_training = self.isd_data.loc[training_mask].copy()

        print(f"\n*** TRAINING PERIOD (RS-VAR): ALL DATA UP TO 2010 ***")
        print(f"    ISD training records: {len(self.isd_training):,} hours")
        print(f"    ISD training coverage: {self.isd_training.index.min()} to {self.isd_training.index.max()}")
        print(f"    Training years: {self.isd_training.index.year.max() - self.isd_training.index.year.min() + 1}")

        return self

    def _calculate_humidity_variables(self):
        """Calculate RH and specific humidity from T, Td, P using Magnus formula."""
        T = self.isd_data['temp'].values
        Td = self.isd_data['dewpoint'].values
        P = self.isd_data['pressure'].values

        def e_sat(T_celsius):
            return 6.112 * np.exp(17.67 * T_celsius / (T_celsius + 243.5))

        e_actual = e_sat(Td)
        e_sat_T = e_sat(T)

        RH = 100.0 * e_actual / e_sat_T
        RH = np.clip(RH, 0, 100)

        q = 0.622 * e_actual / (P - 0.378 * e_actual)
        q = np.clip(q, 0, 0.06)

        self.isd_data['relative_humidity'] = RH
        self.isd_data['specific_humidity'] = q

        print(f"  ✓ Relative humidity: {RH.mean():.1f}% (range: {RH.min():.1f}-{RH.max():.1f}%)")
        print(f"  ✓ Specific humidity: {q.mean()*1000:.2f} g/kg")

    def load_cordex_rcp85(self):
        """Load CORDEX_CMIP5 RCP8.5 projections and compute baseline climatology."""
        print("\n" + "="*80)
        print("STEP 2: LOADING CORDEX_CMIP5 PROJECTIONS + COMPUTING BASELINE")
        print("="*80)

        self.cordex_data = pd.read_csv(self.cordex_path, parse_dates=['time'])
        self.cordex_data.set_index('time', inplace=True)

        print(f"Loaded {len(self.cordex_data):,} daily records")
        print(f"Full period: {self.cordex_data.index.min()} to {self.cordex_data.index.max()}")

        # Unit conversions
        for col in ['tas', 'tasmax', 'tasmin']:
            if col in self.cordex_data.columns:
                self.cordex_data[f'{col}_C'] = self.cordex_data[col] - 273.15

        if 'ps' in self.cordex_data.columns:
            self.cordex_data['ps_hPa'] = self.cordex_data['ps'] / 100

        # Compute CORDEX_CMIP5 baseline climatology (same period as ISD)
        self._compute_cordex_baseline()

        return self

    def _compute_cordex_baseline(self):
        """
        Compute CORDEX_CMIP5 baseline climatology for the baseline period.

        This computes DAY-OF-YEAR means for each variable during baseline_start to baseline_end.
        Daily deltas preserve the full temporal resolution of CORDEX_CMIP5.
        """
        print(f"\n*** COMPUTING CORDEX_CMIP5 BASELINE ({self.baseline_start}-{self.baseline_end}) ***")

        baseline_mask = (
            (self.cordex_data.index.year >= self.baseline_start) &
            (self.cordex_data.index.year <= self.baseline_end)
        )
        cordex_baseline_data = self.cordex_data.loc[baseline_mask]

        print(f"    CORDEX_CMIP5 baseline records: {len(cordex_baseline_data):,} days")

        if len(cordex_baseline_data) < 365 * 5:
            print(f"  ⚠️  WARNING: CORDEX_CMIP5 baseline has less than 5 years!")

        # Compute DAY-OF-YEAR climatology for baseline period
        self.cordex_baseline = {}

        # Variables to compute baseline for
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

            # Day-of-year climatology (mean by DOY) - DAILY resolution
            doy_clim = cordex_baseline_data.groupby(cordex_baseline_data.index.dayofyear)[var].mean()
            self.cordex_baseline[var] = {
                'doy': doy_clim,
                'overall': cordex_baseline_data[var].mean(),
                'transform': transform_type
            }

            print(f"    {var}: baseline mean = {cordex_baseline_data[var].mean():.2f}")

        print(f"\n✓ CORDEX_CMIP5 baseline climatology computed for {len(self.cordex_baseline)} variables (DAILY resolution)")

    def compute_climate_delta_daily(self, target_date):
        """
        Compute climate delta for a specific DATE (not month).

        For additive variables: delta = CORDEX_CMIP5(date) - CORDEX_baseline(doy)
        For multiplicative variables: ratio = CORDEX_CMIP5(date) / CORDEX_baseline(doy)

        Returns dict of {variable: delta_or_ratio}
        """
        deltas = {}

        # Get CORDEX_CMIP5 value for target date
        target_mask = (self.cordex_data.index.date == target_date.date())
        cordex_target = self.cordex_data.loc[target_mask]

        if len(cordex_target) == 0:
            return deltas

        doy = target_date.timetuple().tm_yday

        for var, baseline_info in self.cordex_baseline.items():
            if var not in cordex_target.columns:
                continue

            future_val = cordex_target[var].iloc[0]  # Single daily value
            baseline_val = baseline_info['doy'].get(doy, baseline_info['overall'])

            if baseline_info['transform'] == 'additive':
                deltas[var] = future_val - baseline_val
            else:  # multiplicative
                if baseline_val > 0:
                    deltas[var] = future_val / baseline_val
                else:
                    deltas[var] = 1.0  # No change if baseline is zero

        return deltas

    def extract_seasonal_profiles(self, target_vars):
        """
        Extract seasonal/diurnal profiles FROM BASELINE PERIOD ONLY.

        This is the key fix from V4: profiles are computed only from the
        baseline period (e.g., 1991-2010), not the entire history.
        """
        print("\n" + "="*80)
        print(f"STEP 3: EXTRACTING SEASONAL PROFILES (BASELINE {self.baseline_start}-{self.baseline_end} ONLY)")
        print("="*80)

        # Use baseline data only
        data_with_time = self.isd_baseline.copy()
        data_with_time['month'] = data_with_time.index.month
        data_with_time['hour'] = data_with_time.index.hour

        for var in target_vars:
            if var not in data_with_time.columns:
                continue

            # Group by month and hour, take mean (baseline period only)
            profile = data_with_time.groupby(['month', 'hour'])[var].mean()
            self.seasonal_profiles[var] = profile

            var_mean = data_with_time[var].mean()
            seasonal_range = profile.max() - profile.min()

            print(f"\n  {var}:")
            print(f"    Baseline mean: {var_mean:.2f}")
            print(f"    Seasonal/diurnal range: {seasonal_range:.2f}")
            print(f"    Profile min: {profile.min():.2f} (month={profile.idxmin()[0]}, hour={profile.idxmin()[1]})")
            print(f"    Profile max: {profile.max():.2f} (month={profile.idxmax()[0]}, hour={profile.idxmax()[1]})")

        print(f"\n✓ Extracted seasonal profiles for {len(self.seasonal_profiles)} variables")
        print(f"  *** Profiles computed from BASELINE PERIOD ONLY ({self.baseline_start}-{self.baseline_end}) ***")

        return self

    def test_stationarity(self, target_vars):
        """
        Test stationarity of anomalies (ISD - baseline profile) during baseline period.

        Uses ADF and KPSS tests. For a valid VAR model, anomalies should be stationary.
        """
        print("\n" + "="*80)
        print("STEP 3b: STATIONARITY TEST ON BASELINE ANOMALIES")
        print("="*80)

        data_with_time = self.isd_baseline.copy()
        data_with_time['month'] = data_with_time.index.month
        data_with_time['hour'] = data_with_time.index.hour

        self.stationarity_results = {}

        for var in target_vars:
            if var not in self.seasonal_profiles:
                continue

            # Compute anomalies: observed - seasonal profile
            profile = self.seasonal_profiles[var]
            seasonal_values = data_with_time.apply(
                lambda row: profile.get((row['month'], row['hour']), np.nan),
                axis=1
            )
            anomalies = data_with_time[var] - seasonal_values
            anomalies = anomalies.dropna()

            if len(anomalies) < 1000:
                print(f"  {var}: insufficient data for stationarity test")
                continue

            # Subsample for faster testing (daily means)
            daily_anom = anomalies.resample('D').mean().dropna()

            # ADF test (null: non-stationary)
            try:
                adf_stat, adf_pval, _, _, _, _ = adfuller(daily_anom, maxlag=30)
                adf_stationary = adf_pval < 0.05
            except Exception as e:
                adf_stat, adf_pval, adf_stationary = np.nan, np.nan, None

            # KPSS test (null: stationary)
            try:
                kpss_stat, kpss_pval, _, _ = kpss(daily_anom, regression='c', nlags='auto')
                kpss_stationary = kpss_pval > 0.05
            except Exception as e:
                kpss_stat, kpss_pval, kpss_stationary = np.nan, np.nan, None

            self.stationarity_results[var] = {
                'adf_stat': adf_stat,
                'adf_pval': adf_pval,
                'adf_stationary': adf_stationary,
                'kpss_stat': kpss_stat,
                'kpss_pval': kpss_pval,
                'kpss_stationary': kpss_stationary,
                'anomaly_mean': anomalies.mean(),
                'anomaly_std': anomalies.std(),
            }

            status = "✓" if (adf_stationary and kpss_stationary) else "⚠️"
            print(f"  {var}: ADF p={adf_pval:.4f} ({'stationary' if adf_stationary else 'non-stationary'}), "
                  f"KPSS p={kpss_pval:.4f} ({'stationary' if kpss_stationary else 'non-stationary'}) {status}")
            print(f"         anomaly mean={anomalies.mean():.4f}, std={anomalies.std():.2f}")

        # Summary
        n_stationary = sum(1 for v in self.stationarity_results.values()
                          if v['adf_stationary'] and v['kpss_stationary'])
        n_tested = len(self.stationarity_results)
        print(f"\n✓ Stationarity summary: {n_stationary}/{n_tested} variables pass both tests")

        return self.stationarity_results

    def engineer_features(self):
        """Engineer temperature anomalies for regime identification using training data (pre-2011)."""
        print("\n" + "="*80)
        print("STEP 4: FEATURE ENGINEERING (TRAINING DATA ONLY)")
        print("="*80)

        # Use training data (excludes validation period to prevent leakage)
        for window in [168, 720]:
            rolling_mean = self.isd_training['temp'].rolling(window=window, min_periods=window//2).mean()
            self.isd_training[f'temp_anomaly_{window}h'] = self.isd_training['temp'] - rolling_mean

        p10 = self.isd_training['temp'].quantile(0.10)
        p90 = self.isd_training['temp'].quantile(0.90)

        self.isd_training['temp_extreme_cold'] = (self.isd_training['temp'] < p10).astype(int)
        self.isd_training['temp_extreme_hot'] = (self.isd_training['temp'] > p90).astype(int)

        # Heatwave detection
        from itertools import groupby
        daily_max = self.isd_training['temp'].resample('D').max()
        threshold = daily_max.quantile(0.90)
        exceeds = (daily_max > threshold).astype(int)

        hw_day_indices = []
        for key, group in groupby(enumerate(exceeds), lambda x: x[1]):
            group_list = list(group)
            if key == 1 and len(group_list) >= 3:
                hw_day_indices.extend([idx for idx, _ in group_list])

        heatwave_days = pd.Series(0, index=daily_max.index)
        if hw_day_indices:
            heatwave_days.iloc[hw_day_indices] = 1

        self.isd_training['heatwave_flag'] = 0
        for date in heatwave_days[heatwave_days == 1].index:
            self.isd_training.loc[self.isd_training.index.date == date.date(), 'heatwave_flag'] = 1

        print(f"✓ Engineered features on training data ({self.isd_training.index.min().year}-{self.isd_training.index.max().year})")

        return self

    def standardize_features(self, target_vars):
        """Standardize variables using baseline statistics, apply to training data."""
        print("\n" + "="*80)
        print("STEP 5: STANDARDIZATION (BASELINE STATISTICS, APPLIED TO TRAINING DATA)")
        print("="*80)

        self.means_ = {}
        self.stds_ = {}

        # Compute statistics from baseline period (for reference)
        for col in target_vars:
            if col in self.isd_baseline.columns:
                self.means_[col] = self.isd_baseline[col].mean()
                self.stds_[col] = self.isd_baseline[col].std()

        # Apply standardization to training data (excludes validation period)
        for col in target_vars:
            if col in self.isd_training.columns and col in self.means_:
                self.isd_training[f'{col}_std'] = (
                    (self.isd_training[col] - self.means_[col]) / self.stds_[col]
                )

        print(f"Standardized {len(target_vars)} variables using baseline statistics, applied to training data")
        return self

    def train_regime_switching(self, k_regimes=3, exog_vars=None):
        """Train Markov-switching model on training data (excludes validation period)."""
        print("\n" + "="*80)
        print(f"STEP 6: REGIME IDENTIFICATION ({k_regimes} regimes, TRAINING DATA ONLY)")
        print("="*80)

        # Use training data only (excludes validation period to prevent leakage)
        daily_data = self.isd_training.resample('D').agg({
            col: 'mean' for col in self.isd_training.columns
            if self.isd_training[col].dtype in ['float64', 'int64'] and col.endswith('_std')
        })

        for col in ['heatwave_flag', 'temp_extreme_hot', 'temp_extreme_cold']:
            if col in self.isd_training.columns:
                daily_data[col] = self.isd_training[col].resample('D').max()

        if exog_vars:
            for ex in exog_vars:
                if ex in daily_data.columns:
                    continue
                if ex in self.isd_training.columns:
                    if ex.endswith('_flag') or 'extreme' in ex:
                        daily_data[ex] = self.isd_training[ex].resample('D').max()
                    else:
                        daily_data[ex] = self.isd_training[ex].resample('D').mean()

        daily_data.dropna(inplace=True)

        endog = daily_data['temp_std']
        print(f"Training on {len(endog):,} daily observations (training data, excludes validation)")

        exog = None
        if exog_vars:
            available = [v for v in exog_vars if v in daily_data.columns]
            if available:
                exog = daily_data.loc[endog.index, available]
                print(f"Exogenous vars: {available}")

        try:
            model = MarkovAutoregression(
                endog,
                k_regimes=k_regimes,
                order=1,
                exog=exog,
                switching_variance=True
            )

            result = model.fit(method='nm', maxiter=1000, disp=False)
            print(f"✓ Markov-switching fitted - AIC: {result.aic:.2f}")

            regime_probs_daily = result.smoothed_marginal_probabilities
            regime_probs_hourly = np.zeros((k_regimes, len(self.isd_training)))

            for i in range(k_regimes):
                daily_probs = pd.Series(regime_probs_daily[i], index=endog.index)
                hourly_probs = daily_probs.reindex(self.isd_training.index, method='ffill')
                regime_probs_hourly[i] = hourly_probs.values

            self.regime_probs = regime_probs_hourly
            self.ms_result = result

            for i in range(k_regimes):
                mask = self.regime_probs[i] > 0.5
                regime_data = self.isd_training.iloc[mask]
                print(f"\n  Regime {i}: {mask.sum():,} hours ({mask.sum()/len(self.isd_training)*100:.1f}%)")
                print(f"    Temp: {regime_data['temp'].mean():.1f}°C")

            return result

        except Exception as e:
            print(f"✗ Failed: {e}")
            return None

    def train_regime_conditional_vars(self, target_vars, k_regimes=3):
        """Train VAR models on deviations from baseline seasonal profile using training data."""
        print("\n" + "="*80)
        print("STEP 7: REGIME-CONDITIONAL VAR FOR VARIATIONS (TRAINING DATA ONLY)")
        print("="*80)

        print("Training VAR on deviations from baseline seasonal profile...")

        for regime_i in range(k_regimes):
            print(f"\n--- Regime {regime_i} ---")

            mask = self.regime_probs[regime_i] > 0.5
            regime_indices = self.isd_training.index[mask]

            regime_data = self.isd_training.loc[regime_indices, target_vars].copy()

            # Calculate deviations from baseline seasonal profile
            regime_data_with_time = regime_data.copy()
            regime_data_with_time['month'] = regime_data.index.month
            regime_data_with_time['hour'] = regime_data.index.hour

            for var in target_vars:
                if var not in self.seasonal_profiles:
                    continue

                profile = self.seasonal_profiles[var]
                seasonal_values = pd.Series(
                    [float(profile.get((m, h), np.nan))
                     for m, h in zip(regime_data_with_time['month'], regime_data_with_time['hour'])],
                    index=regime_data_with_time.index,
                )
                regime_data[f'{var}_deviation'] = regime_data[var] - seasonal_values

            deviation_vars = [f'{v}_deviation' for v in target_vars if f'{v}_deviation' in regime_data.columns]

            if len(deviation_vars) == 0 or len(regime_data) < 100:
                print(f"Skipping (insufficient data)")
                continue

            try:
                model = VAR(regime_data[deviation_vars])
                result = model.fit(maxlags=1, ic='aic')
                print(f"✓ VAR(1) fitted on deviations - {len(regime_data):,} obs, AIC: {result.aic:.2f}")

                self.regime_models[regime_i] = result

            except Exception as e:
                print(f"✗ Failed: {e}")

        return self.regime_models

    def _forecast_regime_probabilities(self, start_date, end_date):
        """Forecast regime probabilities using Markov transition matrix."""
        transition_matrix = self.ms_result.regime_transition
        k_regimes = transition_matrix.shape[0]

        last_regime_probs = self.regime_probs[:, -1]
        last_regime_probs = last_regime_probs / last_regime_probs.sum()

        daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(daily_dates)

        regime_columns = [f'regime_{i}_prob' for i in range(k_regimes)]

        # Build climatology from training data
        hist_probs = pd.DataFrame(
            self.regime_probs.T,
            index=self.isd_training.index,
            columns=regime_columns
        )
        daily_hist = hist_probs.resample('D').mean()
        climatology = daily_hist.groupby(daily_hist.index.dayofyear)[regime_columns].mean()

        climatology_weight = 0.2
        volatility = 0.12
        rng = np.random.default_rng(self.seed)

        regime_probs_forecast = np.zeros((k_regimes, n_days))
        current_probs = last_regime_probs.copy().reshape(-1, 1)

        for day_idx in range(n_days):
            current_probs = transition_matrix.T @ current_probs
            blended = current_probs.flatten()

            if climatology is not None and len(climatology) > 0:
                doy = int(daily_dates[day_idx].timetuple().tm_yday)
                if doy in climatology.index:
                    clim = climatology.loc[doy].to_numpy(dtype=float)
                else:
                    nearest_idx = climatology.index.get_indexer([doy], method='nearest')[0]
                    clim = climatology.iloc[nearest_idx].to_numpy(dtype=float)
                clim = np.maximum(clim, 1e-8)
                clim = clim / clim.sum()
                blended = (1.0 - climatology_weight) * blended + climatology_weight * clim

            if volatility > 0:
                noise = rng.normal(0.0, volatility, size=blended.shape)
                noise -= noise.mean()
                blended = blended * np.exp(noise)

            blended = np.maximum(blended, 1e-8)
            blended = blended / blended.sum()

            regime_probs_forecast[:, day_idx] = blended
            current_probs = blended.reshape(-1, 1)

        probs_df = pd.DataFrame(
            regime_probs_forecast.T,
            index=daily_dates,
            columns=regime_columns
        )

        self.regime_transition_matrix = transition_matrix
        self.daily_regime_probs_forecast = probs_df
        return probs_df

    def forecast_with_climate_delta(self, target_vars, start_date='2025-01-01',
                                    end_date='2100-12-31', chunk_years=5):
        """
        Forecast using explicit climate-delta approach.

        Formula: output = baseline_profile + RS-VAR_anomaly + climate_delta

        Where:
            - baseline_profile: ISD seasonal profile from baseline period
            - RS-VAR_anomaly: Zero-mean variation from regime-switching VAR
            - climate_delta: CORDEX_CMIP5(year,month) - CORDEX_baseline(month)
        """
        print("\n" + "="*80)
        print("STEP 8: FORECASTING WITH EXPLICIT CLIMATE-DELTA")
        print("="*80)
        print(f"*** Using baseline period: {self.baseline_start}-{self.baseline_end} ***")

        forecast_index = pd.date_range(start=start_date, end=end_date, freq='h')
        n_steps = len(forecast_index)
        chunk_size = chunk_years * 8760
        n_chunks = int(np.ceil(n_steps / chunk_size))

        print(f"Forecast period: {start_date} to {end_date}")
        print(f"Total steps: {n_steps:,} hours")

        if len(self.regime_models) == 0:
            print(f"✗ No regime models available")
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

        # Precompute DAILY climate deltas for all dates in forecast period
        print(f"\n2. Precomputing DAILY climate deltas...")
        daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        climate_deltas = {}
        for date in daily_dates:
            climate_deltas[date.date()] = self.compute_climate_delta_daily(date)

        print(f"   Computed deltas for {len(climate_deltas):,} days")

        # Show sample deltas
        sample_date = daily_dates[len(daily_dates) // 2]  # Middle of forecast
        sample_delta = climate_deltas.get(sample_date.date(), {})
        print(f"   Sample climate delta for {sample_date.date()}:")
        for var, delta in list(sample_delta.items())[:3]:
            print(f"     {var}: {delta:.3f}")

        # Forecast in chunks
        print(f"\n3. Forecasting with RS-VAR + climate delta...")
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

            # 3. CENTER deviations to zero-mean (removes regime probability mismatch bias)
            # This is critical: the RS-VAR regime probabilities in forecast may differ from
            # training frequencies, causing systematic bias. Centering removes this.
            for dev_var in blended_deviations.columns:
                dev_mean = blended_deviations[dev_var].mean()
                blended_deviations[dev_var] = blended_deviations[dev_var] - dev_mean

            # 4. Add centered RS-VAR deviations to baseline profile
            for var in target_vars:
                dev_var = f'{var}_deviation'
                if dev_var in blended_deviations.columns:
                    chunk_df[var] = chunk_df[var] + blended_deviations[dev_var]

            # 5. Apply climate delta (CORDEX_future - CORDEX_baseline)
            self._apply_climate_delta(chunk_df, climate_deltas, target_vars)

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

        # Enforce solar constraints
        forecast_df = self._enforce_solar_constraints(forecast_df)

        self.forecast_hourly = forecast_df

        print(f"\n✓ Forecast complete: {n_steps:,} hours")
        print(f"  Formula: baseline_profile({self.baseline_start}-{self.baseline_end}) + RS_VAR_anomaly + climate_delta")

        return forecast_df

    def forecast_without_climate_delta(self, target_vars, start_date='2025-01-01',
                                       end_date='2100-12-31', chunk_years=5):
        """
        Forecast using baseline + RS-VAR anomaly only (NO climate delta).

        Formula: output = baseline_profile + RS-VAR_anomaly

        Where:
            - baseline_profile: ISD seasonal profile from baseline period
            - RS-VAR_anomaly: Zero-mean variation from regime-switching VAR

        This method is used for generating reusable RSVAR anomalies that can be
        combined with different climate deltas from multiple scenarios.
        """
        print("\n" + "="*80)
        print("STEP 8: FORECASTING WITH RSVAR ANOMALY (NO CLIMATE DELTA)")
        print("="*80)
        print(f"*** Using baseline period: {self.baseline_start}-{self.baseline_end} ***")
        print(f"*** Climate delta will be applied separately in downstream processing ***")

        forecast_index = pd.date_range(start=start_date, end=end_date, freq='h')
        n_steps = len(forecast_index)
        chunk_size = chunk_years * 8760
        n_chunks = int(np.ceil(n_steps / chunk_size))

        print(f"Forecast period: {start_date} to {end_date}")
        print(f"Total steps: {n_steps:,} hours")

        if len(self.regime_models) == 0:
            print(f"✗ No regime models available")
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

        # Forecast in chunks (WITHOUT climate delta)
        print(f"\n2. Forecasting with RS-VAR (baseline + anomaly only)...")
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

            # 3. CENTER deviations to zero-mean (removes regime probability mismatch bias)
            # This is critical: the RS-VAR regime probabilities in forecast may differ from
            # training frequencies, causing systematic bias. Centering removes this.
            for dev_var in blended_deviations.columns:
                dev_mean = blended_deviations[dev_var].mean()
                blended_deviations[dev_var] = blended_deviations[dev_var] - dev_mean

            # 4. Add centered RS-VAR deviations to baseline profile
            for var in target_vars:
                dev_var = f'{var}_deviation'
                if dev_var in blended_deviations.columns:
                    chunk_df[var] = chunk_df[var] + blended_deviations[dev_var]

            # NOTE: Climate delta is NOT applied here - will be applied separately
            # by apply_climate_delta() or apply_mixed_resolution_climate_delta()

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

        # Enforce solar constraints
        forecast_df = self._enforce_solar_constraints(forecast_df)

        self.forecast_hourly = forecast_df

        print(f"\n✓ Forecast complete: {n_steps:,} hours")
        print(f"  Formula: baseline_profile({self.baseline_start}-{self.baseline_end}) + RS_VAR_anomaly")
        print(f"  Note: Climate delta NOT included - apply separately for multi-scenario analysis")

        return forecast_df

    def _apply_climate_delta(self, chunk_df, climate_deltas, target_vars):
        """
        Apply DAILY climate deltas to chunk with variable-specific transformations.

        Additive for: temp, pressure, dewpoint
        Multiplicative for: wind, humidity, solar

        Each day gets its own delta (CORDEX_date - CORDEX_baseline_doy).
        """
        # Mapping from target vars to CORDEX_CMIP5 vars
        var_to_cordex = {
            'temp': 'tas_C',
            'pressure': 'ps_hPa',
            'wind_speed': 'sfcWind',
            'GHI': 'rsds',
            'relative_humidity': 'hurs',
            'specific_humidity': 'huss',
        }

        # Group by date for efficiency (same delta for all hours in a day)
        chunk_df['_date'] = chunk_df.index.date

        for date, group_idx in chunk_df.groupby('_date').groups.items():
            deltas = climate_deltas.get(date, {})

            for var in target_vars:
                cordex_var = var_to_cordex.get(var)
                if cordex_var is None or cordex_var not in deltas:
                    continue

                delta_or_ratio = deltas[cordex_var]

                if var in self.ADDITIVE_VARS:
                    # Additive: add delta
                    chunk_df.loc[group_idx, var] += delta_or_ratio
                elif var in self.MULTIPLICATIVE_VARS:
                    # Multiplicative: multiply by ratio
                    chunk_df.loc[group_idx, var] *= delta_or_ratio

        # Clean up helper column
        chunk_df.drop('_date', axis=1, inplace=True)

    def _enforce_solar_constraints(self, forecast_df, preserve='blend'):
        """Enforce physical constraints on solar radiation."""
        print(f"\n  Enforcing solar radiation constraints...")

        # Check if at least GHI exists
        if 'GHI' not in forecast_df.columns:
            print(f"    ⚠ No GHI column found, skipping solar constraints")
            return forecast_df

        has_dni_dhi = all(v in forecast_df.columns for v in ['DNI', 'DHI'])
        if not has_dni_dhi:
            print(f"    ⚠ DNI/DHI not found, applying simplified constraints (GHI only)")

        lat_deg = self.lat
        lat = np.radians(lat_deg)

        hours = forecast_df.index.hour.values
        doy = forecast_df.index.dayofyear.values

        declination = np.radians(23.45) * np.sin(2 * np.pi * (doy - 81) / 365)
        hour_angle = (hours - 12) * 15
        hour_angle_rad = np.radians(hour_angle)

        cos_zenith = (np.sin(lat) * np.sin(declination) +
                      np.cos(lat) * np.cos(declination) * np.cos(hour_angle_rad))

        night_mask = cos_zenith <= 0.0  # Only zero when sun below horizon

        # Force nighttime radiation to 0
        forecast_df.loc[night_mask, 'GHI'] = 0.0
        if has_dni_dhi:
            forecast_df.loc[night_mask, 'DNI'] = 0.0
            forecast_df.loc[night_mask, 'DHI'] = 0.0

        # Force negative values to 0
        forecast_df.loc[forecast_df['GHI'] < 0, 'GHI'] = 0.0
        if has_dni_dhi:
            forecast_df.loc[forecast_df['DNI'] < 0, 'DNI'] = 0.0
            forecast_df.loc[forecast_df['DHI'] < 0, 'DHI'] = 0.0

        # Apply closure only if DNI/DHI exist
        day_mask = ~night_mask
        if has_dni_dhi and day_mask.sum() > 0:
            if preserve == 'blend':
                computed_GHI = forecast_df.loc[day_mask, 'DHI'] + forecast_df.loc[day_mask, 'DNI'] * cos_zenith[day_mask]
                forecast_df.loc[day_mask, 'GHI'] = 0.5 * forecast_df.loc[day_mask, 'GHI'] + 0.5 * computed_GHI
                forecast_df.loc[forecast_df['GHI'] < 0, 'GHI'] = 0.0
            elif preserve == 'GHI':
                GHI_corrected = forecast_df.loc[day_mask, 'GHI'].values
                DNI_current = forecast_df.loc[day_mask, 'DNI'].values
                cos_z = cos_zenith[day_mask]
                DHI_reconciled = GHI_corrected - DNI_current * cos_z
                DHI_reconciled = np.maximum(DHI_reconciled, 0.0)
                forecast_df.loc[day_mask, 'DHI'] = DHI_reconciled

        if has_dni_dhi:
            print(f"    ✓ Zeroed {night_mask.sum():,} night hours, applied closure to {day_mask.sum():,} daytime hours")
        else:
            print(f"    ✓ Zeroed {night_mask.sum():,} night hours, forced non-negative GHI")

        return forecast_df

    def _calculate_dewpoint_from_RH(self, forecast_df):
        """Back-calculate dewpoint from RH and T."""
        if 'relative_humidity' not in forecast_df.columns or 'temp' not in forecast_df.columns:
            return forecast_df

        T = forecast_df['temp'].values
        RH = np.clip(forecast_df['relative_humidity'].values, 1.0, 100.0)

        def e_sat(T_celsius):
            return 6.112 * np.exp(17.67 * T_celsius / (T_celsius + 243.5))

        e_sat_T = e_sat(T)
        e_actual = (RH / 100.0) * e_sat_T
        e_actual = np.maximum(e_actual, 0.1)

        ln_ratio = np.log(e_actual / 6.112)
        denominator = 17.67 - ln_ratio
        denominator = np.where(np.abs(denominator) < 0.01, 0.01, denominator)

        Td = 243.5 * ln_ratio / denominator
        Td = np.clip(Td, -50, 50)
        Td = np.minimum(Td, T)

        forecast_df['dewpoint'] = Td

        print(f"  ✓ Dewpoint: {np.nanmean(Td):.1f}°C (range: {np.nanmin(Td):.1f}-{np.nanmax(Td):.1f}°C)")

        return forecast_df

    def apply_physics_qa(self, forecast_df):
        """Apply physics QA and back-calculate dewpoint."""
        print("\n" + "="*80)
        print("STEP 9: PHYSICS QA")
        print("="*80)

        # Clip humidity
        if 'relative_humidity' in forecast_df.columns:
            forecast_df['relative_humidity'] = np.clip(forecast_df['relative_humidity'], 0, 100)
        if 'specific_humidity' in forecast_df.columns:
            forecast_df['specific_humidity'] = np.clip(forecast_df['specific_humidity'], 0, 0.06)

        # Back-calculate dewpoint
        forecast_df = self._calculate_dewpoint_from_RH(forecast_df)

        # Final solar closure
        forecast_df = self._enforce_solar_constraints(forecast_df, preserve='GHI')

        self.forecast_corrected = forecast_df
        print(f"✓ Physics QA complete")

        return forecast_df

    def run_full_pipeline(self, chunk_years=5, test_stationarity=True,
                          forecast_start='2025-01-01', forecast_end='2100-12-31'):
        """Execute complete pipeline."""
        print("\n" + "="*80)
        print(f"CLIMATE-DELTA PIPELINE V5 (FIXED BASELINE: {self.baseline_start}-{self.baseline_end})")
        print(f"FORECAST PERIOD: {forecast_start} to {forecast_end}")
        print("="*80)

        self.load_isd_data()
        self.load_cordex_rcp85()

        # Define target variables
        exclude = ['heatwave_flag', 'precip_indicator', 'wind_dir', 'wind_u', 'wind_v',
                   'temp_extreme_hot', 'temp_extreme_cold', 'temp_anomaly_168h', 'temp_anomaly_720h',
                   'dewpoint']

        target_vars = [col for col in self.isd_baseline.columns
                      if col not in exclude and
                      self.isd_baseline[col].dtype in ['float64', 'int64'] and
                      not col.endswith('_std') and
                      not col.endswith('_flag') and
                      not col.endswith('_indicator')]

        print(f"\nTarget variables: {target_vars}")

        # Extract seasonal profiles (baseline only)
        self.extract_seasonal_profiles(target_vars)

        # Stationarity test
        if test_stationarity:
            self.test_stationarity(target_vars)

        # Engineer features and train models
        self.engineer_features()
        self.standardize_features(target_vars)
        self.train_regime_switching(k_regimes=3, exog_vars=['temp_anomaly_168h', 'heatwave_flag'])
        self.train_regime_conditional_vars(target_vars, k_regimes=3)

        # Forecast with explicit climate delta
        self.forecast_with_climate_delta(target_vars, start_date=forecast_start,
                                         end_date=forecast_end, chunk_years=chunk_years)

        # Physics QA
        self.apply_physics_qa(self.forecast_hourly)

        return {
            'isd_data': self.isd_data,
            'isd_baseline': self.isd_baseline,
            'cordex_data': self.cordex_data,
            'cordex_baseline': self.cordex_baseline,
            'seasonal_profiles': self.seasonal_profiles,
            'stationarity_results': self.stationarity_results,
            'regime_models': self.regime_models,
            'target_vars': target_vars,
            'forecast_hourly': self.forecast_hourly,
            'forecast_corrected': self.forecast_corrected,
            'baseline_period': (self.baseline_start, self.baseline_end),
        }


class ForecastTo2100V5Rolling(ForecastTo2100V5):
    """
    Variant with rolling baseline for comparison.

    For each forecast year Y, the baseline is computed over [Y-window, Y-1].
    """

    def __init__(self, isd_solar_path, cordex_rcp85_path,
                 rolling_window=20, **kwargs):
        kwargs['baseline_mode'] = 'rolling'
        super().__init__(isd_solar_path, cordex_rcp85_path, **kwargs)
        self.rolling_window = rolling_window

    def _get_rolling_baseline(self, target_year):
        """Get baseline period for a rolling window ending before target_year."""
        end_year = target_year - 1
        start_year = end_year - self.rolling_window + 1
        return start_year, end_year

    # Note: Full rolling implementation would require recomputing baselines
    # for each forecast period. This is a placeholder for comparison.


if __name__ == "__main__":
    CITY_LATITUDES = {
        'los_angeles': 34.05,
        'miami': 25.76,
        'toronto': 43.65,
        'montreal': 45.50,
    }

    parser = argparse.ArgumentParser(description="Run RS-VAR forecast pipeline V5 (fixed baseline, 2025-2100).")
    parser.add_argument("--isd", type=str,
                        default="data/ISD_complete_solar/Los_Angeles_with_solar.csv",
                        help="Path to merged ISD + solar CSV")
    parser.add_argument("--cordex", type=str,
                        default="data/CORDEX_CMIP5/NAM-22_REMO2015_r1i1p1/Los_Angeles/rcp85.csv",
                        help="Path to CORDEX_CMIP5 RCP8.5 CSV")
    parser.add_argument("--lat", type=float, default=None,
                        help="Latitude for solar calculations")
    parser.add_argument("--city", type=str, default="los_angeles",
                        choices=list(CITY_LATITUDES.keys()),
                        help="City name to infer latitude")
    parser.add_argument("--baseline-start", type=int, default=1991,
                        help="Start year of baseline period (default: 1991)")
    parser.add_argument("--baseline-end", type=int, default=2010,
                        help="End year of baseline period (default: 2010)")
    parser.add_argument("--forecast-start", type=str, default="2025-01-01",
                        help="Forecast start date (default: 2025-01-01)")
    parser.add_argument("--forecast-end", type=str, default="2100-12-31",
                        help="Forecast end date (default: 2100-12-31)")
    parser.add_argument("--chunk-years", type=int, default=5,
                        help="Forecast chunk size in years")
    parser.add_argument("--skip-stationarity", action="store_true",
                        help="Skip stationarity testing")
    args = parser.parse_args()

    lat = args.lat if args.lat is not None else CITY_LATITUDES.get(args.city.lower(), 34.0)

    print(f"Running V5 pipeline for {args.city} (lat={lat}°N)")
    print(f"Baseline period: {args.baseline_start}-{args.baseline_end}")

    pipeline = ForecastTo2100V5(
        isd_solar_path=args.isd,
        cordex_rcp85_path=args.cordex,
        lat=lat,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
    )

    results = pipeline.run_full_pipeline(
        chunk_years=args.chunk_years,
        test_stationarity=not args.skip_stationarity,
        forecast_start=args.forecast_start,
        forecast_end=args.forecast_end
    )

    print("\n" + "="*80)
    print("PIPELINE V5 COMPLETE")
    print("="*80)
    print(f"Baseline period: {results['baseline_period']}")
    print(f"Stationarity results: {len(results['stationarity_results'])} variables tested")
