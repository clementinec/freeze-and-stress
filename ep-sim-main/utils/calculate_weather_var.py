#!/usr/bin/env python3
"""
Weather variable calculation utilities.

This module provides functions for calculating derived weather variables
such as dewpoint temperature and relative humidity.
"""

import numpy as np
import pandas as pd


def calc_dewpoint_from_RH(T, RH):
    """
    Calculate dewpoint from temperature and relative humidity using Magnus formula.

    Args:
        T: Temperature in °C
        RH: Relative humidity in %

    Returns:
        Dewpoint temperature in °C
    """
    RH = np.clip(RH, 1.0, 100.0)
    e_sat = 6.112 * np.exp(17.67 * T / (T + 243.5))
    e_actual = (RH / 100.0) * e_sat
    e_actual = np.maximum(e_actual, 0.1)
    ln_ratio = np.log(e_actual / 6.112)
    denominator = np.where(np.abs(17.67 - ln_ratio) < 0.01, 0.01, 17.67 - ln_ratio)
    Td = 243.5 * ln_ratio / denominator
    Td = np.clip(Td, -50, 50)
    Td = np.minimum(Td, T)
    return Td


def calc_dewpoint_from_Q(T, Q, P):
    """
    Calculate dewpoint from specific humidity, temperature, and pressure.

    Args:
        T: Temperature in °C
        Q: Specific humidity in kg/kg
        P: Pressure in hPa

    Returns:
        Dewpoint temperature in °C
    """
    # Calculate vapor pressure from specific humidity
    # q = 0.622 * e / (P - 0.378 * e)
    # Solving for e: e = q * P / (0.622 + 0.378 * q)
    e_actual = Q * P / (0.622 + 0.378 * Q)
    e_actual = np.maximum(e_actual, 0.1)

    # Use Magnus formula to get dewpoint from vapor pressure
    ln_ratio = np.log(e_actual / 6.112)
    denominator = np.where(np.abs(17.67 - ln_ratio) < 0.01, 0.01, 17.67 - ln_ratio)
    Td = 243.5 * ln_ratio / denominator
    Td = np.clip(Td, -50, 50)
    Td = np.minimum(Td, T)
    return Td


def calc_RH_from_dewpoint(T, Td):
    """
    Calculate relative humidity from temperature and dewpoint.

    Args:
        T: Temperature in °C
        Td: Dewpoint temperature in °C

    Returns:
        Relative humidity in %
    """
    def e_sat(T_celsius):
        return 6.112 * np.exp(17.67 * T_celsius / (T_celsius + 243.5))

    e_actual = e_sat(Td)
    e_sat_T = e_sat(T)
    RH = 100.0 * e_actual / e_sat_T
    RH = np.clip(RH, 0, 100)
    return RH


def calc_Q_from_dewpoint(Td, P):
    """
    Calculate specific humidity from dewpoint and pressure.

    Args:
        Td: Dewpoint temperature in °C
        P: Pressure in hPa

    Returns:
        Specific humidity in kg/kg
    """
    # Calculate vapor pressure from dewpoint using Magnus formula
    e_actual = 6.112 * np.exp(17.67 * Td / (Td + 243.5))

    # Calculate specific humidity: q = 0.622 * e / (P - 0.378 * e)
    Q = 0.622 * e_actual / (P - 0.378 * e_actual)

    # Ensure positive and reasonable values (typical range: 0 to 0.05 kg/kg)
    Q = np.clip(Q, 0, 0.05)
    return Q


def enforce_solar_constraints(forecast_df, latitude, longitude, solar_zenith_cos_func, preserve='blend'):
    """
    Enforce physical constraints on solar radiation:
    1. Zero at night (cos(zenith) <= 0.065)
    2. Clip negatives to 0
    3. Enforce closure: GHI = DHI + DNI*cos(zenith)

    Parameters:
    - forecast_df: DataFrame with GHI, DNI, DHI columns and DatetimeIndex
    - latitude: Latitude in degrees (required)
    - longitude: Longitude in degrees (required)
    - solar_zenith_cos_func: Function to calculate cos(zenith angle)
    - preserve: 'blend' (default, before bias correction) - blend GHI toward closure
               'GHI' (after bias correction) - preserve corrected GHI, decompose to DNI/DHI

    Returns:
    - forecast_df: DataFrame with enforced solar constraints

    Raises:
    - ValueError: If latitude or longitude is None
    """
    print(f"\n  Enforcing solar radiation physical constraints (preserve={preserve})...")

    if not all(v in forecast_df.columns for v in ['GHI', 'DNI', 'DHI']):
        return forecast_df

    # Require latitude and longitude
    if latitude is None or longitude is None:
        raise ValueError(
            "Latitude and longitude are required for solar angle calculation. "
            "Please provide valid coordinates."
        )

    print(f"  Using location: {latitude:.4f}°N, {longitude:.4f}°E for solar angle calculation")

    # Calculate cosine of solar zenith angle with proper longitude correction
    cos_zenith = solar_zenith_cos_func(latitude, longitude, forecast_df.index)

    # 1. Zero radiation when sun below horizon (cos_zenith <= 0.065)
    night_mask = cos_zenith <= 0.065
    forecast_df.loc[night_mask, 'GHI'] = 0.0
    forecast_df.loc[night_mask, 'DNI'] = 0.0
    forecast_df.loc[night_mask, 'DHI'] = 0.0

    # 2. Clip all negatives to 0
    forecast_df.loc[forecast_df['GHI'] < 0, 'GHI'] = 0.0
    forecast_df.loc[forecast_df['DNI'] < 0, 'DNI'] = 0.0
    forecast_df.loc[forecast_df['DHI'] < 0, 'DHI'] = 0.0

    # 3. Enforce closure: GHI = DHI + DNI*cos(zenith)
    # Only during daytime
    day_mask = ~night_mask
    if day_mask.sum() > 0:
        if preserve == 'blend':
            # Before bias correction: blend GHI toward closure
            computed_GHI = forecast_df.loc[day_mask, 'DHI'] + forecast_df.loc[day_mask, 'DNI'] * cos_zenith[day_mask]
            forecast_df.loc[day_mask, 'GHI'] = 0.5 * forecast_df.loc[day_mask, 'GHI'] + 0.5 * computed_GHI
            forecast_df.loc[forecast_df['GHI'] < 0, 'GHI'] = 0.0

        elif preserve == 'GHI':
            # After bias correction: preserve corrected GHI, decompose to DNI/DHI
            # Use Erbs et al. (1982) correlation based on clearness index

            GHI_corrected = forecast_df.loc[day_mask, 'GHI'].values
            cos_z = cos_zenith[day_mask]

            # Extraterrestrial radiation (simplified)
            solar_constant = 1367.0  # W/m²
            I0 = solar_constant * cos_z
            I0 = np.maximum(I0, 1.0)  # Avoid division by zero

            # Clearness index (kt = GHI / I0)
            kt = np.clip(GHI_corrected / I0, 0.0, 1.0)

            # Erbs correlation for diffuse fraction (DHI/GHI)
            # kt <= 0.22: overcast
            # 0.22 < kt <= 0.80: partly cloudy
            # kt > 0.80: clear sky
            diffuse_fraction = np.zeros_like(kt)

            mask1 = kt <= 0.22
            mask2 = (kt > 0.22) & (kt <= 0.80)
            mask3 = kt > 0.80

            diffuse_fraction[mask1] = 1.0 - 0.09 * kt[mask1]
            diffuse_fraction[mask2] = 0.9511 - 0.1604*kt[mask2] + 4.388*kt[mask2]**2 - 16.638*kt[mask2]**3 + 12.336*kt[mask2]**4
            diffuse_fraction[mask3] = 0.165

            diffuse_fraction = np.clip(diffuse_fraction, 0.0, 1.0)

            # Calculate DHI and DNI
            DHI_new = GHI_corrected * diffuse_fraction
            DNI_new = (GHI_corrected - DHI_new) / np.maximum(cos_z, 0.01)

            # Physical bounds
            DHI_new = np.clip(DHI_new, 0.0, GHI_corrected)
            DNI_new = np.clip(DNI_new, 0.0, 1200.0)  # Max DNI ~1200 W/m²

            # Re-enforce closure (may have small errors due to clipping)
            DHI_final = GHI_corrected - DNI_new * cos_z
            DHI_final = np.maximum(DHI_final, 0.0)

            forecast_df.loc[day_mask, 'DHI'] = DHI_final
            forecast_df.loc[day_mask, 'DNI'] = DNI_new

            print(f"    ✓ Decomposed GHI using Erbs model (kt range: {kt.min():.3f}-{kt.max():.3f})")
            print(f"      Diffuse fraction range: {diffuse_fraction.min():.3f}-{diffuse_fraction.max():.3f}")

    n_night = night_mask.sum()
    n_day = day_mask.sum()
    print(f"    ✓ Zeroed {n_night:,} night hours ({n_night/len(forecast_df)*100:.1f}%)")
    print(f"    ✓ Applied closure constraint to {n_day:,} daytime hours ({n_day/len(forecast_df)*100:.1f}%)")

    return forecast_df

