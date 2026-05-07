"""Utilities for parsing Annex80 CSV outputs.

Annex80 provides "MY" CSVs that look like they don't contain a year in the
`time_lst` column (e.g. `1/1/01 0:00`). In practice, some of these files contain
multiple years of hourly data concatenated together, *reusing the same*
month/day/hour textual timestamp for each year.

This module normalizes those files into a real, year-aware, strictly increasing
hourly time series so downstream validation metrics can align with ISD/V5.

Contract
--------
- Input: a DataFrame with a `time_lst` column and weather variables.
- Output: a DataFrame whose `time_lst` is normalized (overwritten) to the
  format `DD/MM/YYYY H:MM` (e.g. `01/01/2001 0:00`) and sorted by real time.
- If `time_lst` already matches that desired format, normalization is a strict
  no-op (no parsing and no extra columns).

"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

_WINDOW_RE = re.compile(r"(?P<start>\d{4})-(?P<end>\d{4})")
_MY_FILE_RE = re.compile(r"_MY_(?P<window>\d{4}-\d{4})", re.IGNORECASE)


def parse_window_label(window_label: str) -> Tuple[int, int]:
    """Parse a window label like '2001-2020' into (start_year, end_year)."""
    m = _WINDOW_RE.search(window_label)
    if not m:
        raise ValueError(f"Invalid window_label (expected 'YYYY-YYYY'): {window_label!r}")
    return int(m.group("start")), int(m.group("end"))


def _is_strict_hourly_series(dt: pd.Series) -> bool:
    if dt.isna().any():
        return False
    if not dt.is_monotonic_increasing:
        return False
    # Allow duplicate timestamps? No.
    if dt.duplicated().any():
        return False
    diffs = dt.diff().dropna().unique()
    return len(diffs) == 1 and diffs[0] == pd.Timedelta(hours=1)


def _has_any_4digit_year_strings(s: pd.Series) -> bool:
    # Match patterns like 1/1/2041 or 01/01/2041 anywhere in the string.
    return s.astype(str).str.contains(r"\b\d{1,2}/\d{1,2}/\d{4}\b", regex=True).any()


def _is_time_lst_in_desired_format(s: pd.Series) -> bool:
    """Return True if all rows match `DD/MM/YYYY H:MM` (hour may be 1-2 digits)."""
    pat = r"^\s*\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2}\s*$"
    serie = s.astype(str)
    return bool(serie.str.match(pat, na=False).all())


def _format_time_lst_ddmmyyyy_hmm(dt: pd.Series) -> pd.Series:
    """Format timestamps as `DD/MM/YYYY H:MM`.

    Notes
    -----
    - Day/month are zero-padded to 2 digits.
    - Hour is NOT zero-padded (matches example `... 0:00`).
    - Minute is always 2 digits.
    """
    # dt.strftime can't do "non-padded hour" portably, so we build it.
    return (
        dt.dt.strftime("%d/%m/%Y")
        + " "
        + dt.dt.hour.astype(int).astype(str)
        + ":"
        + dt.dt.minute.astype(int).astype(str).str.zfill(2)
    )


def _parse_time_lst_with_explicit_2digit_year_policy(time_lst: pd.Series) -> pd.Series:
    """Parse `time_lst` of the form m/d/YY H:MM into timestamps.

    Repo convention for Annex80 MY in this project:
    - All 2-digit years `YY` represent years in the 2000s, i.e. `20YY`.
      Example: '1/1/41 0:00' -> 2041-01-01 00:00

    We implement this explicitly instead of relying on pandas' %y pivot.
    """
    # Extract components. Allow single-digit m/d and H.
    parts = time_lst.astype(str).str.extract(
        r"^\s*(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<yy>\d{2})\s+(?P<h>\d{1,2}):(?P<min>\d{2})\s*$"
    )
    if parts.isna().any(axis=1).any():
        bad = parts.isna().any(axis=1)
        n_bad = int(bad.sum())
        raise ValueError(
            "Failed to parse some Annex80 `time_lst` values with expected format m/d/YY H:MM "
            f"(n={n_bad}). Example bad value: {time_lst.loc[bad].iloc[0]!r}"
        )

    yy = parts["yy"].astype(int)

    # Explicit rule: YY always means 20YY.
    year = 2000 + yy

    dt = pd.to_datetime(
        {
            "year": year,
            "month": parts["m"].astype(int),
            "day": parts["d"].astype(int),
            "hour": parts["h"].astype(int),
            "minute": parts["min"].astype(int),
        },
        errors="coerce",
    )

    if dt.isna().any():
        n_bad = int(dt.isna().sum())
        raise ValueError(
            f"Parsed {n_bad} invalid datetimes from Annex80 `time_lst` (likely invalid calendar dates like Feb 29)."
        )

    return dt


def normalize_annex80_my_df(
    df: pd.DataFrame,
    *,
    window_label: Optional[str] = None,
    start_year: Optional[int] = None,
    tz: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize Annex80 MY DataFrame into a real year-aware hourly time series.

    Returns
    -------
    pd.DataFrame
        Copy of df with `time_lst` overwritten to a normalized `DD/MM/YYYY H:MM`
        string format, sorted by real time.

    Important
    ---------
    If `time_lst` is already in the desired `DD/MM/YYYY H:MM` format, this
    function returns the dataframe unchanged **without** parsing datetimes and
    **without** adding any `datetime` column.

    Notes
    -----
    `tz` is accepted for backward compatibility, but this function never returns
    tz-aware timestamps because it does not return a `datetime` column.
    """
    if "time_lst" not in df.columns:
        raise ValueError("Annex80 MY is missing required column 'time_lst'")

    # True no-op fast-path: if already in desired format, we do nothing at all.
    # This is intentional: user may have pre-normalized the CSV externally.
    if _is_time_lst_in_desired_format(df["time_lst"]):
        out = df.copy()
        # Ensure no stale datetime column is propagated.
        if "datetime" in out.columns:
            out = out.drop(columns=["datetime"])  # type: ignore[arg-type]
        return out

    out: pd.DataFrame = df.copy()

    time_col = out["time_lst"]
    if not isinstance(time_col, pd.Series):
        time_col = pd.Series(time_col)

    # If the input already encodes a real year (e.g. 1/1/2041 0:00), parse it and sort.
    # - 4-digit year: parse with %Y.
    # - 2-digit year: project convention is YY -> 20YY.
    if _has_any_4digit_year_strings(time_col):
        t: pd.Series = pd.to_datetime(time_col, format="%m/%d/%Y %H:%M", errors="coerce")
        if t.isna().any():
            t = pd.to_datetime(time_col, errors="coerce")
        if t.isna().any():
            raise ValueError(f"Failed to parse some Annex80 `time_lst` with 4-digit years (n={int(t.isna().sum())})")

        out["datetime"] = t
        out = out.sort_values("datetime").reset_index(drop=True)
        out["time_lst"] = _format_time_lst_ddmmyyyy_hmm(pd.to_datetime(out["datetime"]))
        out = out.drop(columns=["datetime"])  # type: ignore[arg-type]
        return out

    # Try strict 2-digit-year parsing (m/d/YY H:MM). This covers inputs like '1/1/41 0:00'.
    # In this project, YY is always interpreted as 20YY.
    # If this fails, fall back to pandas' general parser.
    try:
        t = _parse_time_lst_with_explicit_2digit_year_policy(time_col)
    except ValueError:
        t = pd.to_datetime(time_col, errors="coerce")
        if t.isna().any():
            raise ValueError(f"Failed to parse some Annex80 `time_lst` values (n={int(t.isna().sum())})")

    # If parsed datetimes contain >1 year or no duplicates, treat as explicit-year series and sort.
    n_years_in_input = int(pd.Series(t.dt.year).nunique())
    if n_years_in_input > 1 or not t.duplicated().any():
        out["datetime"] = t
        out = out.sort_values("datetime").reset_index(drop=True)
        out["time_lst"] = _format_time_lst_ddmmyyyy_hmm(pd.to_datetime(out["datetime"]))
        out = out.drop(columns=["datetime"])  # type: ignore[arg-type]
        return out

    # If timestamps repeat, we need to expand with real years via window_label/start_year.
    if start_year is None:
        if window_label is None:
            raise ValueError("Need window_label or start_year to expand Annex80 MY into real years")
        start_year, end_year = parse_window_label(window_label)
    else:
        end_year = None

    month = t.dt.month.astype(int)
    day = t.dt.day.astype(int)
    hour = t.dt.hour.astype(int)
    minute = t.dt.minute.astype(int)

    key = (((month * 100 + day) * 100 + hour) * 100 + minute).astype("int64")
    within = pd.Series(key).groupby(key, sort=False).cumcount().to_numpy()

    if end_year is not None:
        n_years_expected = end_year - start_year + 1
    else:
        n_years_expected = int(within.max()) + 1

    counts = pd.Series(key).value_counts()
    max_count = int(counts.max())
    if max_count != n_years_expected:
        raise ValueError(
            f"Annex80 MY year expansion mismatch for window {window_label or start_year}: "
            f"expected each (month/day/hour) key to repeat {n_years_expected} times, but max repetition is {max_count}. "
            "This file may be a single-year template or use a different layout."
        )

    if within.max() >= n_years_expected:
        raise ValueError(
            f"Annex80 MY has more than {n_years_expected} entries for at least one time-of-year key; cannot assign years."
        )

    years = (start_year + within).astype(int)

    dt = pd.to_datetime(
        {
            "year": years,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
        },
        errors="coerce",
    )

    bad = dt.isna()
    if bad.any():
        n_bad = int(bad.sum())
        out = out.loc[~bad].copy()
        dt = dt.loc[~bad]
        print(
            f"[normalize_annex80_my_df] Dropped {n_bad} rows with invalid calendar dates "
            f"while expanding years for window {window_label or start_year}."
        )

    out["datetime"] = dt
    out = out.sort_values("datetime").reset_index(drop=True)

    if out["datetime"].duplicated().any():
        raise ValueError("Expanded Annex80 datetime has duplicates; cannot build a clean hourly series")

    out["time_lst"] = _format_time_lst_ddmmyyyy_hmm(pd.to_datetime(out["datetime"]))
    out = out.drop(columns=["datetime"])  # type: ignore[arg-type]

    return out


def _discover_city_dirs(annex80_dir: Path) -> list[Path]:
    """Return subdirectories that contain at least one Annex80 MY CSV."""
    city_dirs: list[Path] = []
    for p in sorted(annex80_dir.iterdir()):
        if not p.is_dir():
            continue
        if any(_MY_FILE_RE.search(f.name) for f in p.glob("*.csv")):
            city_dirs.append(p)
    return city_dirs


def _extract_window_label_from_name(path: Path) -> str:
    m = _MY_FILE_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot infer window label from filename: {path.name!r}")
    return m.group("window")


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint: normalize all Annex80 MY CSVs into year-aware hourly series."""
    parser = argparse.ArgumentParser(description="Normalize Annex80 MY CSVs and write to data/annex80_MY")
    parser.add_argument(
        "--annex80-dir",
        type=Path,
        default=Path("data/annex80"),
        help="Input root directory containing city folders (default: data/annex80)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/annex80_MY"),
        help="Output root directory (default: data/annex80_MY)",
    )
    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional list of city folder names to process (e.g. Los_Angeles Toronto)",
    )
    parser.add_argument(
        "--tz",
        default=None,
        help="Optional timezone name to localize timestamps (default: tz-naive)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if present",
    )

    args = parser.parse_args(argv)

    annex80_dir: Path = args.annex80_dir
    out_dir: Path = args.out_dir

    if not annex80_dir.exists():
        raise SystemExit(f"Input directory does not exist: {annex80_dir}")

    city_dirs = _discover_city_dirs(annex80_dir)
    if args.cities is not None:
        wanted = set(args.cities)
        city_dirs = [p for p in city_dirs if p.name in wanted]

    n_in = 0
    n_out = 0

    for city_dir in city_dirs:
        my_files = sorted([p for p in city_dir.glob("*.csv") if _MY_FILE_RE.search(p.name)])
        if not my_files:
            continue

        city_out_dir = out_dir / city_dir.name
        city_out_dir.mkdir(parents=True, exist_ok=True)

        for csv_path in my_files:
            n_in += 1
            window_label = _extract_window_label_from_name(csv_path)

            out_path = city_out_dir / csv_path.name
            if out_path.exists() and not args.overwrite:
                continue

            df = pd.read_csv(csv_path)
            norm = normalize_annex80_my_df(df, window_label=window_label, tz=args.tz)
            norm.to_csv(out_path, index=False)
            n_out += 1

    print(f"Annex80 MY normalized: inputs={n_in}, written={n_out}, out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
