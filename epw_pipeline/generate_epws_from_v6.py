#!/usr/bin/env python3
"""
Convert standalone V6 hourly forecast CSVs into year-by-year EPW files.

Expected weather input layout:
  ../weather/output/RSVAR_cd/<scenario>/<city>/forecast_<city>_<scenario>.csv

Output layout:
  epw_out/<scenario>/<city>/<city>_<year>_climatedelta.epw
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd


CITY_CONFIG = {
    "Los_Angeles": {"tmy": "LosAngeles_TMY.epw"},
    "Miami": {"tmy": "Miami_TMY.epw"},
    "Montreal": {"tmy": "Montreal_TMY.epw"},
    "Toronto": {"tmy": "Toronto_TMY.epw"},
}


def _infer_pressure_pa(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dropna().mean() < 2000:
        s = s * 100.0
    return s


def _dewpoint_from_t_rh(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    rh = rh_pct.clip(lower=0.1, upper=100.0) / 100.0
    a = 17.27
    b = 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + np.log(rh)
    return (b * alpha) / (a - alpha)


def _read_header_from_tmy(tmy_path: Path, source_name: str) -> str:
    if not tmy_path.exists():
        raise FileNotFoundError(f"TMY EPW not found: {tmy_path}")

    with open(tmy_path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = [handle.readline().rstrip("\n") for _ in range(8)]

    if len(lines) < 8 or not lines[0].startswith("LOCATION,"):
        raise ValueError(f"TMY EPW header is incomplete: {tmy_path}")

    lines[5] = "COMMENTS 1,Generated from standalone V6 climate-delta pipeline"
    lines[6] = f"COMMENTS 2,Source hourly CSV: {source_name}"
    return "\n".join(lines) + "\n"


def _load_forecast(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "datetime" in df.columns:
        dt_col = "datetime"
    elif "DATE" in df.columns:
        dt_col = "DATE"
    else:
        raise ValueError(f"No datetime column found in {csv_path}")

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.set_index(dt_col).sort_index()

    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]

    return df


def _prepare_epw_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "temp" not in out.columns:
        raise ValueError("Forecast output missing required 'temp' column.")

    if "relative_humidity" not in out.columns:
        out["relative_humidity"] = 50.0

    out["relative_humidity"] = out["relative_humidity"].clip(0, 100)

    if "dewpoint" not in out.columns:
        out["dewpoint"] = _dewpoint_from_t_rh(out["temp"], out["relative_humidity"])

    if "pressure" in out.columns:
        out["pressure"] = _infer_pressure_pa(out["pressure"])
    else:
        out["pressure"] = 101325.0

    for col, default in {
        "wind_speed": 0.0,
        "wind_dir": 0.0,
        "GHI": 0.0,
        "DNI": 0.0,
        "DHI": 0.0,
        "precip_depth": 0.0,
        "precip_duration": 0.0,
    }.items():
        if col not in out.columns:
            out[col] = default

    return out


def _write_epw_for_year(df: pd.DataFrame, year: int, out_path: Path, header: str) -> None:
    year_df = df.loc[str(year)].copy()
    year_df = year_df[~((year_df.index.month == 2) & (year_df.index.day == 29))]

    desired_index = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")
    if calendar.isleap(year):
        desired_index = desired_index[~((desired_index.month == 2) & (desired_index.day == 29))]

    year_df = year_df.reindex(desired_index)
    year_df.ffill(inplace=True)
    year_df.bfill(inplace=True)

    rows = []
    for ts, row in year_df.iterrows():
        rows.append(
            ",".join(
                map(
                    str,
                    [
                        ts.year,
                        ts.month,
                        ts.day,
                        ts.hour + 1,
                        60,
                        0,
                        f"{float(row['temp']):.1f}",
                        f"{float(row['dewpoint']):.1f}",
                        f"{float(row['relative_humidity']):.0f}",
                        f"{float(row['pressure']):.0f}",
                        0,
                        0,
                        0,
                        f"{float(max(row['GHI'], 0.0)):.1f}",
                        f"{float(max(row['DNI'], 0.0)):.1f}",
                        f"{float(max(row['DHI'], 0.0)):.1f}",
                        0,
                        0,
                        0,
                        0,
                        int(0 if pd.isna(row["wind_dir"]) else row["wind_dir"]),
                        f"{float(max(row['wind_speed'], 0.0)):.1f}",
                        0,
                        0,
                        9999,
                        99999,
                        0,
                        0,
                        999,
                        0,
                        0,
                        0,
                        0,
                        f"{float(max(row['precip_depth'], 0.0)):.1f}",
                        f"{float(np.clip(row['precip_duration'], 0, 6)):.1f}",
                    ],
                )
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(header)
        for row in rows:
            handle.write(row + "\n")


def generate_city_epws(
    weather_csv: Path,
    city_slug: str,
    tmy_path: Path,
    out_dir: Path,
    epw_suffix: str,
    start_year: int,
    end_year: int,
) -> list[Path]:
    forecast = _prepare_epw_columns(_load_forecast(weather_csv))
    header = _read_header_from_tmy(tmy_path, weather_csv.name)

    years = sorted(set(forecast.index.year))
    years = [year for year in years if start_year <= year <= end_year]

    written = []
    for year in years:
        out_path = out_dir / city_slug / f"{city_slug}_{year}_{epw_suffix}.epw"
        _write_epw_for_year(forecast, year, out_path, header)
        written.append(out_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EPWs from standalone V6 hourly forecast CSVs.")
    parser.add_argument(
        "--weather-root",
        type=Path,
        default=Path("../weather/output/RSVAR_cd"),
        help="Root directory containing V6 climate-delta forecast outputs.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="CORDEX_CMIP5_REMO2015_rcp85",
        help="Scenario directory name under --weather-root.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("epw_out"),
        help="Root directory to write EPWs into.",
    )
    parser.add_argument(
        "--cities",
        type=str,
        default=None,
        help="Comma-separated city slugs (default: all supported cities).",
    )
    parser.add_argument("--epw-suffix", type=str, default="climatedelta")
    parser.add_argument("--start-year", type=int, default=2025)
    parser.add_argument("--end-year", type=int, default=2100)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    weather_root = (script_dir / args.weather_root).resolve() if not args.weather_root.is_absolute() else args.weather_root
    out_root = (script_dir / args.out_root).resolve() if not args.out_root.is_absolute() else args.out_root
    scenario_dir = weather_root / args.scenario

    if not scenario_dir.exists():
        raise SystemExit(f"Weather scenario directory not found: {scenario_dir}")

    if args.cities:
        cities = [city.strip() for city in args.cities.split(",") if city.strip()]
    else:
        cities = sorted(CITY_CONFIG.keys())

    missing = [city for city in cities if city not in CITY_CONFIG]
    if missing:
        raise SystemExit(f"Unsupported cities: {missing}. Supported: {sorted(CITY_CONFIG)}")

    total_written = 0
    for city_slug in cities:
        weather_csv = scenario_dir / city_slug / f"forecast_{city_slug}_{args.scenario}.csv"
        tmy_path = script_dir / "energyplus_runner" / CITY_CONFIG[city_slug]["tmy"]

        if not weather_csv.exists():
            raise SystemExit(f"Forecast CSV not found for {city_slug}: {weather_csv}")

        written = generate_city_epws(
            weather_csv=weather_csv,
            city_slug=city_slug,
            tmy_path=tmy_path,
            out_dir=out_root / args.scenario,
            epw_suffix=args.epw_suffix,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        print(f"{city_slug}: wrote {len(written)} EPWs to {out_root / args.scenario / city_slug}")
        total_written += len(written)

    print(f"Complete: wrote {total_written} EPWs.")


if __name__ == "__main__":
    main()
