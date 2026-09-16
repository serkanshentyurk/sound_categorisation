#!/bin/bash
#SBATCH --job-name=run_gs
#SBATCH --output=results/logs/run_gs_%A_%a.out
#SBATCH --error=results/logs/run_gs_%A_%a.err
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
# Submit via:  bash slurm/submit.sh {train|condition|gs} [args]   (derives --array from the task grid)
set -euo pipefail

module load miniconda
conda activate sound_cat
cd "${SLURM_SUBMIT_DIR}"

echo "=== run_gs task ${SLURM_ARRAY_TASK_ID} on $(hostname) $(date) ==="
python -m scripts.run_gs --task-id "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done $(date) ==="
