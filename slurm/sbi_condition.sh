#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=1-00:00
# One (rep, model) network per array task, conditioned on every animal of the cohort: 64 CV
# repeats + the 1,000-sample saved posterior per animal. Upper estimate: ~2 h for 20 animals.
# Submit via:  bash slurm/submit.sh sbi-condition --cohort C --distribution D --run-id ID
set -euo pipefail

export PYTHONNOUSERSITE=1     # never import from ~/.local (it shadows the env's numpy/scipy/torch)
export PYTHONUNBUFFERED=1     # progress lines reach the .out file as they happen
module load miniconda
conda activate sound_cat
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "=== sbi_condition task ${SLURM_ARRAY_TASK_ID} on $(hostname) $(date) ==="
sc-sbi-condition --task-id "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done $(date) ==="
