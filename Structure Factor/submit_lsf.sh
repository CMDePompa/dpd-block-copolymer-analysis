#!/usr/bin/env bash
# =============================================================================
# LSF submission script for structure-factor pipeline version 4
# =============================================================================
# Edit the BSUB directives and paths below, then submit with: bsub < submit_lsf.sh
#
# The pipeline is single-node, single-process Python (no MPI). NumPy uses
# OpenMP internally — set OMP_NUM_THREADS to match your allocated core count.

#BSUB -J sk_analysis
#BSUB -n 8                            # number of cores
#BSUB -R "rusage[mem=8GB]"            # memory (adjust for large trajectories)
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00                        # wall-time limit
#BSUB -o out.sk_analysis
#BSUB -e err.sk_analysis
#BSUB -q single_chassis               # change to your queue

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup — adjust to your HPC module system
# ---------------------------------------------------------------------------
# module load python/3.11               # uncomment and adjust as needed

# LSF starts a non-interactive shell, so initialize Conda explicitly.
CONDA_ENV="${CONDA_ENV:-/usr/local/usrapps/pasquinell/md_env}"
if [[ -z "${CONDA_BASE:-}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda is not on PATH; load your HPC Conda module first." >&2
        exit 1
    fi
    CONDA_BASE="$(conda info --base)"
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG=/path/to/your_config.yaml      # path to your edited config YAML
PIPELINE_DIR="/path/to/Structure Factor"  # directory containing run_pipeline.py

export OMP_NUM_THREADS=${LSB_DJOB_NUMPROC:-8}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "${PIPELINE_DIR}" || { echo "ERROR: Cannot cd to ${PIPELINE_DIR}"; exit 1; }

echo "=== structure-factor pipeline v4 ==="
echo "Config:  ${CONFIG}"
echo "Host:    $(hostname)"
echo "Cores:   ${OMP_NUM_THREADS}"
echo "Started: $(date)"

"${PYTHON_BIN}" run_pipeline.py --config "${CONFIG}" --version
"${PYTHON_BIN}" run_pipeline.py --config "${CONFIG}"

echo "Finished: $(date)"
