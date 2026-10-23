#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
# One (rep, model) network per array task: SBC_N_DRAWS prior draws, each simulated and
# conditioned like an animal (1,000 posterior samples). Upper estimate: ~1 h, more if the
# rejection sampler's acceptance is low.
# Submit via:  bash slurm/submit.sh sbi-sbc --cohort C --distribution D --run-id ID
set -euo pipefail

export PYTHONNOUSERSITE=1     # never import from ~/.local (it shadows the env's numpy/scipy/torch)
export PYTHONUNBUFFERED=1     # progress lines reach the .out file as they happen
module load miniconda
conda activate sound_cat
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "=== sbi_sbc task ${SLURM_ARRAY_TASK_ID} on $(hostname) $(date) ==="
sc-sbi-sbc --task-id "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done $(date) ==="
