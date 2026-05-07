#!/usr/bin/env python3
"""Compute EDH (Exceedance Degree Hours) metrics from EnergyPlus SQLite outputs
and merge them into an existing annual_metrics.csv.

Metrics computed (base temperature default = 26 °C):
  - annual_edh_c_h      : Σ max(0, T_op − T_base) × Δt   [°C·h]
  - daily_edh_max_k_h   : max daily EDH across the year   [K·h]
  - delta_t_max_k        : max single-timestep exceedance  [K]
"""

import argparse
import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

from output_preset import REPRESENTATIVE_ZONES

DEFAULT_BASE_TEMP = 26.0


# ---------------------------------------------------------------------------
# SQLite helpers (same patterns as extract_sqlite_metrics.py)
# ---------------------------------------------------------------------------

def get_active_env_period(conn):
    row = conn.execute(
        """
        SELECT EnvironmentPeriodIndex, COUNT(*) AS n
        FROM Time WHERE WarmupFlag = 0
        GROUP BY EnvironmentPeriodIndex
        ORDER BY n DESC, EnvironmentPeriodIndex ASC
        LIMIT 1
        """
    ).fetchone()
    return row[0] if row else None


def find_operative_temp_index(conn, zone_name):
    row = conn.execute(
        """
        SELECT ReportDataDictionaryIndex
        FROM ReportDataDictionary
        WHERE Name = 'Zone Operative Temperature' AND KeyValue = ?
        LIMIT 1
        """,
        (zone_name,),
    ).fetchone()
    return row[0] if row else None


def fetch_hourly_with_day(conn, dict_index, env_period):
    """Return list of (value, month, day, interval_minutes) tuples."""
    rows = conn.execute(
        """
        SELECT rd.Value, t.Month, t.Day, t.Interval
        FROM ReportData AS rd
        JOIN Time AS t ON rd.TimeIndex = t.TimeIndex
        WHERE rd.ReportDataDictionaryIndex = ?
          AND t.WarmupFlag = 0
          AND t.EnvironmentPeriodIndex = ?
        ORDER BY rd.TimeIndex
        """,
        (dict_index, env_period),
    ).fetchall()
    return [(float(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in rows]


# ---------------------------------------------------------------------------
# EDH computation
# ---------------------------------------------------------------------------

def compute_edh(hourly_data, base_temp):
    """Return (annual_edh, daily_edh_max, delta_t_max) from hourly data."""
    if not hourly_data:
        return None, None, None

    daily_edh = defaultdict(float)
    annual_edh = 0.0
    delta_t_max = 0.0

    for temp, month, day, interval_min in hourly_data:
        exceedance = temp - base_temp
        if exceedance > 0:
            dt_hours = interval_min / 60.0
            edh_step = exceedance * dt_hours
            annual_edh += edh_step
            daily_edh[(month, day)] += edh_step
            if exceedance > delta_t_max:
                delta_t_max = exceedance

    daily_max = max(daily_edh.values()) if daily_edh else 0.0
    exceed_days = sum(1 for v in daily_edh.values() if v > 6.0)
    return round(annual_edh, 4), round(daily_max, 4), round(delta_t_max, 4), exceed_days


def process_one(sql_path, building, base_temp):
    """Compute EDH for one simulation SQLite file."""
    zone = REPRESENTATIVE_ZONES.get(building)
    if not zone:
        print(f"  WARNING: unknown building type '{building}', skipping")
        return None, None, None, None

    conn = sqlite3.connect(sql_path)
    try:
        env_period = get_active_env_period(conn)
        if env_period is None:
            print(f"  WARNING: no active env period in {sql_path}")
            return None, None, None, None

        dict_index = find_operative_temp_index(conn, zone)
        if dict_index is None:
            print(f"  WARNING: Zone Operative Temperature not found for '{zone}' in {sql_path}")
            return None, None, None, None

        hourly = fetch_hourly_with_day(conn, dict_index, env_period)
        if not hourly:
            print(f"  WARNING: no hourly data in {sql_path}")
            return None, None, None, None

        return compute_edh(hourly, base_temp)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute EDH metrics from EnergyPlus SQLite and merge into annual_metrics.csv")
    parser.add_argument("--file", required=True, help="Path to annual_metrics.csv")
    parser.add_argument("--base-temp", type=float, default=DEFAULT_BASE_TEMP, help=f"Comfort base temperature (default {DEFAULT_BASE_TEMP} °C)")
    args = parser.parse_args()

    csv_path = Path(args.file)
    print(f"Reading: {csv_path}")
    print(f"Base temperature: {args.base_temp} °C")

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_fields = list(reader.fieldnames)
        rows = list(reader)

    print(f"Rows: {len(rows)}")

    edh_cols = ["annual_edh_c_h", "daily_edh_max_k_h", "delta_t_max_k", "daily_edh_exceed_6kh_count"]
    success = 0

    for i, row in enumerate(rows):
        sql_path = row.get("sql_path", "")
        building = row.get("building", "")
        city = row.get("city", "")
        year = row.get("year", "")

        if not sql_path or not Path(sql_path).exists():
            print(f"  [{i+1}/{len(rows)}] {building}/{city}/{year} — SQLite not found: {sql_path}")
            for col in edh_cols:
                row[col] = ""
            continue

        print(f"  [{i+1}/{len(rows)}] {building}/{city}/{year} ...", end=" ")
        annual, daily_max, delta_t, exceed_days = process_one(sql_path, building, args.base_temp)

        row["annual_edh_c_h"] = annual if annual is not None else ""
        row["daily_edh_max_k_h"] = daily_max if daily_max is not None else ""
        row["delta_t_max_k"] = delta_t if delta_t is not None else ""
        row["daily_edh_exceed_6kh_count"] = exceed_days if exceed_days is not None else ""

        if annual is not None:
            success += 1
            print(f"EDH={annual:.1f}  daily_max={daily_max:.2f}  ΔT_max={delta_t:.2f}  exceed_days={exceed_days}")
        else:
            print("SKIPPED")

    # Merge new columns into fieldnames
    out_fields = original_fields[:]
    for col in edh_cols:
        if col not in out_fields:
            out_fields.append(col)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {success}/{len(rows)} rows updated. Saved to {csv_path}")


if __name__ == "__main__":
    main()
