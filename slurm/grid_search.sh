#!/bin/bash
#SBATCH -p cpu,cpu_backfill
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
# One (animal, model, seed) per array task on the full 8,000-point grid. On the learning-prior grid
# (November 2026) a task took a median 1.4 h for BE and 2.4 h for SC, 90 % of them under 4 and 5 h;
# a few SC fits ran past 6 h, the earlier limit, hence 12 h. submit.sh throttles the array
# (GS_ARRAY_THROTTLE, default 150 concurrent tasks); cpu_backfill takes tasks when cpu is full. Tasks
# that left no partial are resubmitted with submit.sh ... --run-id ID --missing.
# Submit via:  bash slurm/submit.sh grid-search --cohort C --distribution D --fit-target update_matrix --run-id ID
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh

echo "=== grid_search task ${SLURM_ARRAY_TASK_ID} on $(hostname) $(date) ==="
sc-grid-search --task-id "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done $(date) ==="
