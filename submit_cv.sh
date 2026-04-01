#!/bin/bash
#SBATCH --job-name=cv_grid
#SBATCH --output=logs/cv_%A_%a.out
#SBATCH --error=logs/cv_%A_%a.err
#
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH --time=0-2:00
#
#SBATCH --array=1-64

source ~/.bashrc
conda activate sound_cat

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

REPO_DIR="/nfs/nhome/live/sshentyurk/repos/sound_categorisation"
CONFIG="${REPO_DIR}/config_slurm.yaml"
OUTPUT_DIR="${REPO_DIR}/cv_results"

# Animals to fit (space-separated)
ANIMALS="SS01 SS04 SS05 SS06 SS07 SS08 SS09"

# Grid: 'full' or 'coarse'
GRID="full"

# CV settings
FIT_WITH="update"
N_FOLDS=2

# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

mkdir -p "${REPO_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"

SEED=${SLURM_ARRAY_TASK_ID}

echo "=== Job ${SLURM_JOB_ID}, Task ${SLURM_ARRAY_TASK_ID} ==="
echo "Seed: ${SEED}, Grid: ${GRID}"
echo "Animals: ${ANIMALS}"
echo ""

for ANIMAL in ${ANIMALS}; do
    echo "──── ${ANIMAL}, seed ${SEED} ────"
    python "${REPO_DIR}/run_cv_single.py" \
        --config "${CONFIG}" \
        --animal "${ANIMAL}" \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_DIR}" \
        --grid "${GRID}" \
        --fit-with "${FIT_WITH}" \
        --n-folds "${N_FOLDS}"
    echo ""
done

echo "=== Done ==="