#!/bin/bash
####################################################################################
###                                                                              ###
### run_office_metric_pipeline.sbatch                                            ###
###                                                                              ###
### Submit one scenario end-to-end pipeline on HKU HPC:                          ###
### 1. Run office EnergyPlus simulations for all cities / years                  ###
### 2. Run metric extraction + derived drivers + break-year analysis             ###
###                                                                              ###
### Usage:                                                                       ###
###   sbatch run_office_metric_pipeline.sbatch <EPW_ROOT> <SCENARIO> [RESULTS_ROOT] ###
###                                                                              ###
### Arguments:                                                                   ###
###   EPW_ROOT   Root folder that contains scenario subfolders                   ###
###              Example: /path/to/epw_out                                       ###
###   SCENARIO   Scenario folder name under EPW_ROOT and results/                ###
###              Example: CORDEX_CMIP5_REMO2015_rcp85                            ###
###                                                                              ###
### Optional argument:                                                           ###
###   RESULTS_ROOT  Root folder for simulation outputs only.                     ###
###                 If omitted, defaults to the project path                     ###
###                 freeze-and-stress-main/epw_pipeline/energyplus_runner/results ###
###                                                                              ###
### Notes:                                                                       ###
### - metric_exports/<scenario>/... and metric_res/<scenario>/... remain under   ###
###   the project default paths.                                                 ###
### - It requires EnergyPlus to be available on compute nodes.                   ###
###   Set ENERGYPLUS_BIN or ensure `energyplus` is on PATH.                      ###
### - It assumes the Python environment already has pandas / numpy (and tqdm     ###
###   optionally).                                                               ###
###                                                                              ###
####################################################################################

#SBATCH --job-name=office_metric_pipeline
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your@email
##SBATCH --partition=intel
##SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=%x.out.%j
#SBATCH --error=%x.err.%j

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: sbatch $(basename "$0") <EPW_ROOT> <SCENARIO> [RESULTS_ROOT]" >&2
  exit 1
fi

EPW_ROOT_INPUT="$1"
SCENARIO="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER_DIR="${REPO_ROOT}/freeze-and-stress-main/epw_pipeline/energyplus_runner"
RUN_FUTURE_SCRIPT="${RUNNER_DIR}/run_future_simulations.py"
RUN_METRIC_SCRIPT="${RUNNER_DIR}/run_metric_pipeline.sh"

EPW_SCENARIO_DIR="${EPW_ROOT_INPUT}/${SCENARIO}"
DEFAULT_RESULTS_ROOT="${RUNNER_DIR}/results"
RESULTS_ROOT="${3:-${DEFAULT_RESULTS_ROOT}}"
SCENARIO_RESULTS_DIR="${RESULTS_ROOT}/${SCENARIO}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ENERGYPLUS_BIN="${ENERGYPLUS_BIN:-$(command -v energyplus || true)}"

if [[ ! -d "${EPW_SCENARIO_DIR}" ]]; then
  echo "Scenario EPW directory not found: ${EPW_SCENARIO_DIR}" >&2
  exit 1
fi

if [[ ! -f "${RUN_FUTURE_SCRIPT}" ]]; then
  echo "Missing script: ${RUN_FUTURE_SCRIPT}" >&2
  exit 1
fi

if [[ ! -x "${RUN_METRIC_SCRIPT}" ]]; then
  echo "Metric pipeline script is not executable: ${RUN_METRIC_SCRIPT}" >&2
  exit 1
fi

if [[ -z "${ENERGYPLUS_BIN}" ]]; then
  echo "EnergyPlus executable not found on PATH. Export ENERGYPLUS_BIN before sbatch, or load the appropriate module." >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "==========================================================="
echo "Job Start Time : $(date '+%Y/%m/%d -- %H:%M:%S')"
echo "Repo Root      : ${REPO_ROOT}"
echo "Runner Dir     : ${RUNNER_DIR}"
echo "EPW Root Input : ${EPW_ROOT_INPUT}"
echo "Scenario       : ${SCENARIO}"
echo "Scenario EPWs  : ${EPW_SCENARIO_DIR}"
echo "Results Root   : ${RESULTS_ROOT}"
echo "Results Dir    : ${SCENARIO_RESULTS_DIR}"
echo "Python         : ${PYTHON_BIN}"
echo "EnergyPlus     : ${ENERGYPLUS_BIN}"
echo "==========================================================="

mkdir -p "${SCENARIO_RESULTS_DIR}"

echo "[A/2] Running future simulations for scenario ${SCENARIO}"
"${PYTHON_BIN}" "${RUN_FUTURE_SCRIPT}" \
  --epw-root "${EPW_SCENARIO_DIR}" \
  --output-dir "${SCENARIO_RESULTS_DIR}" \
  --local-eplus "${ENERGYPLUS_BIN}" \
  --buildings office

echo "[B/2] Running metric pipeline for scenario ${SCENARIO}"
"${RUN_METRIC_SCRIPT}" \
  --results-root "${RESULTS_ROOT}" \
  --epw-root "${EPW_ROOT_INPUT}" \
  --scenarios "${SCENARIO}" \
  --python "${PYTHON_BIN}"

echo "==========================================================="
echo "Job Finish Time: $(date '+%Y/%m/%d -- %H:%M:%S')"
echo "Results stored under:"
echo "  ${RESULTS_ROOT}/${SCENARIO}"
echo "  ${RUNNER_DIR}/metric_exports/${SCENARIO}"
echo "  ${RUNNER_DIR}/metric_res/${SCENARIO}"
echo "==========================================================="

exit 0
