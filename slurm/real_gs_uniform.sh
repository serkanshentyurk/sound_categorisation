#!/bin/bash
#SBATCH --job-name=gs_uniform
#SBATCH --output=results/logs/gs_uniform_%A_%a.out
#SBATCH --error=results/logs/gs_uniform_%A_%a.err
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH --time=1-00:00
#
# Grid-search CV on real animals, uniform distribution.
# Array job: one task per (animal × model × fit_target).
# Each task runs all 64 seeds internally.
#
# Before submitting:
#   1. Update ANIMALS array below with your current animal list
#   2. Update --array to match: 0-(N_ANIMALS*4 - 1)
#      e.g. 12 animals → --array=0-47
#
# Usage:
#   sbatch --array=0-47 slurm/real_gs_uniform.sh
# ─────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/env_setup.sh"
cd "${REPO_DIR}"

# ── Configuration ────────────────────────────────────────────────────────────
# UPDATE THIS LIST with your actual animal IDs
ANIMALS=(SS01 SS04 SS05 SS06 SS07 SS08 SS09 SS10 SS11 SS12 SS13 SS14)
MODELS=(BE SC)
FIT_TARGETS=(update_matrix conditional_psych)
DISTRIBUTION="uniform"

# ── Map SLURM_ARRAY_TASK_ID to (animal, model, fit_target) ──────────────────
N_ANIMALS=${#ANIMALS[@]}
N_MODELS=${#MODELS[@]}
N_TARGETS=${#FIT_TARGETS[@]}
N_TOTAL=$((N_ANIMALS * N_MODELS * N_TARGETS))

TASK_ID=${SLURM_ARRAY_TASK_ID}

if [ "${TASK_ID}" -ge "${N_TOTAL}" ]; then
    echo "Task ID ${TASK_ID} exceeds total ${N_TOTAL}. Exiting."
    exit 0
fi

# Decompose: task_id = animal_idx * (N_MODELS * N_TARGETS) + model_idx * N_TARGETS + target_idx
ANIMAL_IDX=$((TASK_ID / (N_MODELS * N_TARGETS)))
REMAINDER=$((TASK_ID % (N_MODELS * N_TARGETS)))
MODEL_IDX=$((REMAINDER / N_TARGETS))
TARGET_IDX=$((REMAINDER % N_TARGETS))

ANIMAL="${ANIMALS[$ANIMAL_IDX]}"
MODEL="${MODELS[$MODEL_IDX]}"
FIT_TARGET="${FIT_TARGETS[$TARGET_IDX]}"

echo "=== Task ${TASK_ID}/${N_TOTAL}: ${ANIMAL} / ${MODEL} / ${FIT_TARGET} ==="
echo ""

python3 scripts/run_gs_single.py \
    --animal "${ANIMAL}" \
    --model "${MODEL}" \
    --fit-target "${FIT_TARGET}" \
    --distribution "${DISTRIBUTION}" \
    --config config_slurm.yaml

echo ""
echo "=== Finished ==="
