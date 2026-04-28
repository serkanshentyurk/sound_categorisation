#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Environment setup — sourced by all SLURM jobs.
#
# Usage (in other .sh files):
#   source "$(dirname "$0")/env_setup.sh"
# ─────────────────────────────────────────────────────────────────────────────

# Load conda (SWC cluster)
module load miniconda

# Activate environment
# NOTE: verify this matches your actual conda env name.
# If the env doesn't exist, the job will fail here rather than
# silently running with the base environment.
CONDA_ENV="sound_cat"
conda activate "${CONDA_ENV}" || {
    echo "ERROR: conda env '${CONDA_ENV}' not found."
    echo "Available envs: $(conda env list --json | python3 -c 'import sys,json; print([e.split(\"/\")[-1] for e in json.load(sys.stdin)[\"envs\"]])')"
    exit 1
}

# Headless matplotlib (no X display on compute nodes)
export MPLBACKEND=Agg

# Repo root
export REPO_DIR="/nfs/nhome/live/sshentyurk/repos/sound_categorisation"

# Ensure results directories exist
python3 -c "import sys; sys.path.insert(0, '${REPO_DIR}'); from scripts.config import ensure_dirs; ensure_dirs()"

echo "Environment: $(which python3)"
echo "Repo:        ${REPO_DIR}"
echo "Node:        $(hostname)"
echo "Date:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
export PYTHONUNBUFFERED=1
