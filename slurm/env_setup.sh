#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Environment setup — sourced by all SLURM jobs.
#
# Usage (in other .sh files):
#   source "$(dirname "$0")/env_setup.sh"
# ─────────────────────────────────────────────────────────────────────────────

# Load conda (SWC cluster)
module load miniconda/23.10.0

# Activate environment
conda activate sound_categorisation

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
