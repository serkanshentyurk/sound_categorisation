#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
# One (rep, model, session count) per array task (SBC_GRID, 12 tasks): SBC_DRAWS_BY_SESSIONS prior
# draws (1,000 at 5 sessions, 500 at 3 and at 2), each simulated and conditioned like an animal
# (1,000 posterior samples). Upper estimate: ~4 h for a 5-session task, ~1.5 h for the others,
# more if the rejection sampler's acceptance is low.
# Submit via:  bash slurm/submit.sh sbi-sbc --cohort C --distribution D --run-id ID
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "=== sbi_sbc task ${SLURM_ARRAY_TASK_ID} on $(hostname) $(date) ==="
sc-sbi-sbc --task-id "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done $(date) ==="
