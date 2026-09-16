#!/bin/bash
#SBATCH --job-name=train_sbi
#SBATCH --output=results/logs/train_sbi_%A_%a.out
#SBATCH --error=results/logs/train_sbi_%A_%a.err
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=3-00:00
# Submit via:  bash slurm/submit.sh {train|condition|gs} [args]   (derives --array from the task grid)
set -euo pipefail

module load miniconda
conda activate sound_cat
cd "${SLURM_SUBMIT_DIR}"

echo "=== train_sbi task ${SLURM_ARRAY_TASK_ID} on $(hostname) $(date) ==="
python -m scripts.train_sbi --task-id "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done $(date) ==="
