#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# One job: every network in a directory against the current code and settings (sc-sbi-check);
# the fingerprint simulates 5,000 prior draws per network on all 20 cores. With --real-run it
# also loads the snapshot and re-selects that run's sessions. Upper estimate: ~2 h for four
# networks.
# Submit via:  bash slurm/submit.sh sbi-check --cohort C --run-id ID [--networks-dir DIR] [--real-run C/ID]
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "=== sbi_check on $(hostname) $(date) ==="
sc-sbi-check --n-jobs "${SLURM_CPUS_PER_TASK}" "$@"
echo "=== done $(date) ==="
