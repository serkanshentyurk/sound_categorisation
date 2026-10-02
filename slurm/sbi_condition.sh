#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=1-00:00
# Submit via:  bash slurm/submit.sh {sbi-train|sbi-condition|grid-search} [args]
# submit.sh sets --array, --job-name and the log paths (inside the run directory) and passes --run-id.
set -euo pipefail

module load miniconda
conda activate sound_cat
cd "${SLURM_SUBMIT_DIR}"

echo "=== sbi_condition task ${SLURM_ARRAY_TASK_ID} on $(hostname) $(date) ==="
sc-sbi-condition --task-id "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done $(date) ==="
