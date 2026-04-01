#!/bin/bash
#SBATCH --job-name=cv_grid
#SBATCH --output=logs/cv_%A_%a.out
#SBATCH --error=logs/cv_%A_%a.err
#SBATCH --array=1-64            # one task per seed; adjust to N_SEEDS
#SBATCH --cpus-per-task=8       # joblib uses these for parallel grid search
#SBATCH --mem=8G
#SBATCH --time=02:00:00         # ~30 min per seed (coarse), ~2h (full)
#SBATCH --partition=cpu         # adjust to your cluster

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these
# ═══════════════════════════════════════════════════════════════════════════════

# Conda/venv activation
module load miniconda/23.10.0
conda activate sound_cat

# Paths
CONFIG="/nfs/nhome/live/sshentyurk/repos/sound_categorisation/config_slurm.yaml"
SCRIPT_DIR="/nfs/nhome/live/sshentyurk/repos/sound_categorisation"
OUTPUT_DIR="${SCRIPT_DIR}/cv_results"

# Animals to fit (space-separated)
ANIMALS="SS01 SS04 SS05 SS06 SS07 SS08 SS09"

# Grid resolution: 'full' or 'coarse'
GRID="full"

# CV settings
FIT_WITH="update"   # or 'update' for UM MSE
N_FOLDS=2

# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

SEED=${SLURM_ARRAY_TASK_ID}

echo "=== Job ${SLURM_JOB_ID}, Task ${SLURM_ARRAY_TASK_ID} ==="
echo "Seed: ${SEED}, Grid: ${GRID}"
echo "Animals: ${ANIMALS}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo ""

for ANIMAL in ${ANIMALS}; do
    echo "──── ${ANIMAL}, seed ${SEED} ────"
    python "${SCRIPT_DIR}/run_cv_single.py" \
        --config "${CONFIG}" \
        --animal "${ANIMAL}" \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_DIR}" \
        --grid "${GRID}" \
        --old-code-dir "${OLD_CODE_DIR}" \
        --old-models-dir "${OLD_MODELS_DIR}" \
        --fit-with "${FIT_WITH}" \
        --n-folds "${N_FOLDS}"
    echo ""
done

echo "=== Done ==="
