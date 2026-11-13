#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=2-00:00
# The report runs on a capture's snapshot, into the capture's own results root (migration/README.md):
# the opto battery on opto1-cohort and switch adaptation on behaviour1-cohort, full settings.
# Upper estimate: overnight. Needs capture.sh's snapshot first.
# Submit from the repository root:  sbatch -o "$OUT/reports_%j.out" migration/reports.sh "$OUT"
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
out="${1:?usage: sbatch migration/reports.sh OUTPUT_FOLDER}"

echo "=== reports on $(hostname) $(date) ==="
sc-reports battery --snapshot "$out/snapshot" --root "$out/results" --jobs "${SLURM_CPUS_PER_TASK}"
sc-reports switch-adaptation --cohort behaviour1-cohort --snapshot "$out/snapshot" --root "$out/results"
echo "=== done $(date) ==="
