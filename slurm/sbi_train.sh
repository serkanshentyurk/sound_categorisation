#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --mem=48G
#SBATCH --time=2-00:00
# One network per array task (TRAIN_GRID: rep x model x distribution). Simulations run on every
# core of the task (joblib); training is torch on the same cores. Upper estimate per task:
# pooled (100k sims) ~16 h, moments (150k sims) ~30 h, simulation and training together.
# Submit via:  bash slurm/submit.sh sbi-train [sc-sbi-train args]
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "=== sbi_train task ${SLURM_ARRAY_TASK_ID} on $(hostname) ($(nproc) cores) $(date) ==="
sc-sbi-train --task-id "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done $(date) ==="
