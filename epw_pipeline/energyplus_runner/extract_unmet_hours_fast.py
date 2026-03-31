#!/usr/bin/env python3
"""Fast extractor for occupied cooling unmet hours from EnergyPlus SQLite files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path


TARGET_QUERY = """
SELECT Value
FROM TabularDataWithStrings
WHERE ReportName = 'AnnualBuildingUtilityPerformanceSummary'
  AND TableName = 'Comfort and Setpoint Not Met Summary'
  AND RowName = 'Time Setpoint Not Met During Occupied Cooling'
LIMIT 1
"""


def safe_float(value: object) -> float | None:
    """Convert a tabular value to float when possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def discover_sqlite_files(results_root: Path, min_age_seconds: int) -> list[tuple[str, str, str, int, Path]]:
    """Discover result SQLite files under results/<scenario>/<building>/<city>/<year>/."""
    discovered: list[tuple[str, str, str, int, Path]] = []
    now = time.time()
    for sql_path in sorted(results_root.rglob("eplusout.sql")):
        try:
            age_seconds = now - sql_path.stat().st_mtime
        except FileNotFoundError:
            continue
        if age_seconds < min_age_seconds:
            continue

        parts = sql_path.relative_to(results_root).parts
        if len(parts) != 5:
            continue
        scenario, building, city, year_text, _ = parts
        if not year_text.isdigit():
            continue
        discovered.append((scenario, building, city, int(year_text), sql_path))
    return discovered


def extract_unmet_hours(sql_path: Path) -> float | None:
    """Extract occupied cooling unmet hours from one SQLite database."""
    conn = sqlite3.connect(f"file:{sql_path}?mode=ro", uri=True, timeout=0.1)
    try:
        row = conn.execute(TARGET_QUERY).fetchone()
        return safe_float(row[0]) if row else None
    finally:
        conn.close()


def extract_one(entry: tuple[str, str, str, int, Path]) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Extract one row, returning either a result row or a failure row."""
    scenario, building, city, year, sql_path = entry
    try:
        return (
            {
                "scenario": scenario,
                "building": building,
                "city": city,
                "year": year,
                "sql_path": str(sql_path),
                "abups_occupied_cooling_not_met_hours": extract_unmet_hours(sql_path),
            },
            None,
        )
    except Exception as exc:
        return (
            None,
            {
                "scenario": scenario,
                "building": building,
                "city": city,
                "year": year,
                "sql_path": str(sql_path),
                "error": str(exc),
            },
        )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast extraction of occupied cooling unmet hours from current SQLite outputs")
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="Scenario result root")
    parser.add_argument("--output-dir", type=Path, default=Path("metric_exports_unmet_fast"), help="Output directory")
    parser.add_argument("--scenarios", type=str, default=None, help="Optional comma-separated scenario filter")
    parser.add_argument(
        "--min-age-seconds",
        type=int,
        default=600,
        help="Skip SQLite files modified more recently than this age to avoid in-flight outputs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, (cpu_count() or 1)),
        help="Number of parallel workers to use for independent SQLite reads",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    results_root = (script_dir / args.results_root).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    discovered = discover_sqlite_files(results_root, args.min_age_seconds)
    if args.scenarios:
        allowed = {item.strip() for item in args.scenarios.split(",") if item.strip()}
        discovered = [row for row in discovered if row[0] in allowed]

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    workers = max(1, int(args.workers))
    if workers == 1:
        for entry in discovered:
            row, failure = extract_one(entry)
            if row is not None:
                rows.append(row)
            if failure is not None:
                failures.append(failure)
    else:
        with Pool(processes=workers) as pool:
            for row, failure in pool.imap_unordered(extract_one, discovered, chunksize=max(1, len(discovered) // (workers * 8) or 1)):
                if row is not None:
                    rows.append(row)
                if failure is not None:
                    failures.append(failure)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "annual_metrics.csv", rows)
    write_csv(output_dir / "extraction_failures.csv", failures)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "sqlite_files_processed": len(discovered),
        "rows_written": len(rows),
        "failures": len(failures),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
