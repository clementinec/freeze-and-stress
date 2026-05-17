#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT=""
EPW_ROOT="${SCRIPT_DIR%/energyplus_runner}/epw_out"
METRIC_EXPORTS_ROOT="${SCRIPT_DIR}/metric_exports"
METRIC_RES_ROOT="${SCRIPT_DIR}/metric_res"
SCENARIOS=""
PYTHON_BIN="${PYTHON:-python3}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --results-root /path/to/results [options]

Options:
  --results-root PATH       Root results directory.
  --epw-root PATH           Root EPW directory. Default: ${EPW_ROOT}
  --metric-exports-root PATH
                            Root directory for annual_metrics outputs. Default: ${METRIC_EXPORTS_ROOT}
  --metric-res-root PATH    Root directory for metric_res outputs. Default: ${METRIC_RES_ROOT}
  --scenarios A,B,C         Optional comma-separated scenario filter
  --python BIN              Python executable. Default: ${PYTHON_BIN}
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --epw-root)
      EPW_ROOT="$2"
      shift 2
      ;;
    --metric-exports-root)
      METRIC_EXPORTS_ROOT="$2"
      shift 2
      ;;
    --metric-res-root)
      METRIC_RES_ROOT="$2"
      shift 2
      ;;
    --scenarios)
      SCENARIOS="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${RESULTS_ROOT}" ]]; then
  echo "--results-root is required" >&2
  usage >&2
  exit 1
fi

mkdir -p "${METRIC_EXPORTS_ROOT}" "${METRIC_RES_ROOT}"

extract_cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/extract_sqlite_metrics.py"
  --results-root "${RESULTS_ROOT}"
  --output-dir "${METRIC_EXPORTS_ROOT}"
)
if [[ -n "${SCENARIOS}" ]]; then
  extract_cmd+=(--scenarios "${SCENARIOS}")
fi

echo "[1/3] Extracting annual metrics from simulation results"
"${extract_cmd[@]}"

scenario_dirs=()
manifest_path="${METRIC_EXPORTS_ROOT}/extraction_summary_all.json"
while IFS= read -r scenario_dir; do
  [[ -n "${scenario_dir}" ]] && scenario_dirs+=("${scenario_dir}")
done < <(
  MANIFEST_PATH="${manifest_path}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

manifest = Path(os.environ["MANIFEST_PATH"])
if not manifest.exists():
    raise SystemExit(0)

data = json.loads(manifest.read_text(encoding="utf-8"))
for item in data.get("scenarios", []):
    output_dir = item.get("output_dir")
    rows_written = int(item.get("rows_written", 0) or 0)
    sqlite_files_processed = int(item.get("sqlite_files_processed", 0) or 0)
    if output_dir and (rows_written > 0 or sqlite_files_processed > 0):
        print(output_dir)
PY
)

if [[ ${#scenario_dirs[@]} -eq 0 ]]; then
  echo "No scenario outputs were produced under ${METRIC_EXPORTS_ROOT}" >&2
  exit 1
fi

scenario_count=${#scenario_dirs[@]}
scenario_index=0
for scenario_dir in "${scenario_dirs[@]}"; do
  scenario_index=$((scenario_index + 1))
  scenario="$(basename "${scenario_dir}")"
  annual_csv="${scenario_dir}/annual_metrics.csv"
  extended_csv="${scenario_dir}/annual_metrics_extended.csv"
  scenario_metric_res="${METRIC_RES_ROOT}/${scenario}"

  if [[ ! -f "${annual_csv}" ]]; then
    echo "Skipping ${scenario}: missing ${annual_csv}" >&2
    continue
  fi

  annual_row_count="$("${PYTHON_BIN}" - "${annual_csv}" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, encoding="utf-8", newline="") as handle:
    reader = csv.reader(handle)
    try:
        next(reader)
    except StopIteration:
        print(0)
        raise SystemExit(0)
    print(sum(1 for _ in reader))
PY
)"
  if [[ "${annual_row_count}" == "0" ]]; then
    echo "Skipping ${scenario}: ${annual_csv} has 0 data rows" >&2
    continue
  fi

  echo "[2/4] (${scenario_index}/${scenario_count}) Computing EDH metrics for ${scenario}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/compute_edh_from_sqlite.py" \
    --file "${annual_csv}"

  echo "[3/4] (${scenario_index}/${scenario_count}) Computing derived climate drivers for ${scenario}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/metric_exports/compute_derived_drivers.py" \
    --input "${annual_csv}" \
    --output "${extended_csv}" \
    --epw-root "${EPW_ROOT}"

  echo "[4/4] (${scenario_index}/${scenario_count}) Computing break-year and screening outputs for ${scenario}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/compute_break_years_multi_metric.py" \
    --file "${extended_csv}" \
    --output-dir "${scenario_metric_res}"
done

echo "Saved metric exports under ${METRIC_EXPORTS_ROOT}/<scenario>"
echo "Saved metric results under ${METRIC_RES_ROOT}/<scenario>"
