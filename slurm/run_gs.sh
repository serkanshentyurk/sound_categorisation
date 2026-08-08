#!/bin/bash
#SBATCH --job-name=run_gs
#SBATCH --output=results/logs/run_gs_%A_%a.out
#SBATCH --error=results/logs/run_gs_%A_%a.err
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-0
#
# Grid-search model identification. Unlike run_sbi, the array size is
# DATA-DEPENDENT: n_animals x n_models x n_seeds. Compute it per phase with
# --count and pass --array on the sbatch line (it overrides the header above);
# the phase + run options go AFTER the script name (they reach run_gs via "$@").
# Fit ONE phase per launch, then a single gather job for that phase.
#
#   mkdir -p results/logs
#   # --- uniform ---
#   N=$(python -m scripts.run_gs --source real --distribution uniform \
#          --run full --fit-target update_matrix --count)
#   sbatch --array=0-$((N-1)) slurm/run_gs.sh --source real --distribution uniform \
#          --run full --fit-target update_matrix
#   # after it finishes, gather that phase's partials -> finals:
#   python -m scripts.run_gs --source real --distribution uniform \
#          --run full --fit-target update_matrix --gather
#   # --- then repeat the three commands for --distribution hard_a and hard_b ---
#
# Smoke first (one task; --smoke-test uses a sparse 9-point grid + 2 seeds, seconds):
#   sbatch --array=0 slurm/run_gs.sh --source real --distribution uniform \
#          --run quick --fit-target update_matrix --smoke-test
#
# GS is torch-free (numpy grid sweep + UM), so it also runs locally without the
# cluster -- no --task-id runs all seeds serially:
#   python -m scripts.run_gs --source real --distribution uniform \
#          --run full --fit-target update_matrix
# ---------------------------------------------------------------------------
set -euo pipefail

module load miniconda
conda activate sound_cat
cd "${SLURM_SUBMIT_DIR}"

echo "=== run_gs task ${SLURM_ARRAY_TASK_ID} on $(hostname) $(date) ==="
python -m scripts.run_gs --task-id "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done $(date) ==="
