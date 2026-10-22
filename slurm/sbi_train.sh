#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --mem=48G
#SBATCH --time=2-00:00
# One network per array task (TRAIN_GRID: rep x model x distribution). Simulations run on every
# core of the task (joblib); training is torch on the same cores. Upper estimate per task: 2-4 h
# simulation + up to 6 h training (moments, 100k sims); the limit leaves 4x headroom.
# Submit via:  bash slurm/submit.sh sbi-train [sc-sbi-train args]
set -euo pipefail

module load miniconda
conda activate sound_cat
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "=== sbi_train task ${SLURM_ARRAY_TASK_ID} on $(hostname) ($(nproc) cores) $(date) ==="
sc-sbi-train --task-id "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done $(date) ==="
