#!/bin/bash
#SBATCH -p cpu,cpu_backfill
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --time=6:00:00
# One (animal, model, seed) per array task on the full 8,000-point grid: ~0.35 s per grid point
# on a laptop core, so 1-2 h here. submit.sh throttles the array (GS_ARRAY_THROTTLE, default 150
# concurrent tasks); cpu_backfill takes tasks when cpu is full.
# Submit via:  bash slurm/submit.sh grid-search --cohort C --distribution D --fit-target update_matrix --run-id ID
set -euo pipefail

export PYTHONNOUSERSITE=1     # never import from ~/.local (it shadows the env's numpy/scipy/torch)
export PYTHONUNBUFFERED=1     # progress lines reach the .out file as they happen
module load miniconda
conda activate sound_cat
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "=== grid_search task ${SLURM_ARRAY_TASK_ID} on $(hostname) $(date) ==="
sc-grid-search --task-id "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done $(date) ==="
