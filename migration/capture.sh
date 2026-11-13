#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
# Export a snapshot with the installed behav_utils and capture it (migration/README.md).
# Upper estimate: 1.5 h (tests ~3 min, export ~30 min from ceph, capture ~45 min on 16 cores).
# Submit from the repository root:  sbatch -o "$OUT/capture_%j.out" migration/capture.sh "$OUT"
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
out="${1:?usage: sbatch migration/capture.sh OUTPUT_FOLDER}"
mkdir -p "$out"

echo "=== capture on $(hostname) $(date) ==="
python -c "import behav_utils, sound_categorisation as s; print('behav_utils', behav_utils.__version__, '| sound_categorisation', s.__version__)"
# the capture's own tests on this environment first: they stop the job if the project code and the
# installed behav_utils do not fit together (about 3 min)
python -m pytest migration -q -p no:cacheprovider
sc-export-snapshot --output "$out/snapshot"
python -m migration.capture --snapshot "$out/snapshot" --out "$out/capture" --n-jobs "${SLURM_CPUS_PER_TASK}"
echo "=== done $(date) ==="
