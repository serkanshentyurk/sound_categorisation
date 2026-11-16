#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --mem=32G
#SBATCH --time=0-03:00
#SBATCH --job-name=sc-smoke
#SBATCH --output=sc-smoke_%j.out
#SBATCH --error=sc-smoke_%j.out
# The whole model-identification chain, small, in a scratch folder (sc-smoke): four tiny networks, a
# synthetic cohort, conditioning, grid search, calibration, the network check, consensus and the
# validation tables, with a check after each stage. Run it after a change to the fitting code and
# before submitting the real training array. It reads and writes nothing real: everything goes
# under <data root>/smoke/<time> (or --dir). The log is sc-smoke_<job id>.out in the folder you
# submit from; its last line is SMOKE PASSED or SMOKE FAILED with the check that failed.
# Upper estimate: 1 h (most of it the pooled BE network).
#
#   sbatch slurm/smoke.sh                        # from the repo folder
#   sbatch slurm/smoke.sh --n-simulations 1000   # sc-smoke arguments pass through
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh

echo "=== sc-smoke on $(hostname) ($(nproc) cores) $(date) ==="
sc-smoke "$@"
echo "=== done $(date) ==="
