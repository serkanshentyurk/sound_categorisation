#!/bin/bash
#SBATCH --job-name=sbi_dyn
#SBATCH --output=results/logs/sbi_dyn_%A_%a.out
#SBATCH --error=results/logs/sbi_dyn_%A_%a.err
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH --time=0-3:00
#
# Per-animal dynamic SBI with RandomWalk-linked parameters.
# Array job: one task per (animal × model × fit_target).
#
# Priority D (nice-to-have). Run after A-C results are in.
#
# Before submitting:
#   1. Update ANIMALS array below
#   2. Update --array to match: 0-(N_ANIMALS*4 - 1)
#      e.g. 12 animals → --array=0-47
#
# Usage:
#   sbatch --array=0-47 slurm/sbi_dynamic.sh
# ─────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/env_setup.sh"
cd "${REPO_DIR}"

# ── Configuration ────────────────────────────────────────────────────────────
ANIMALS=(SS01 SS04 SS05 SS06 SS07 SS08 SS09 SS10 SS11 SS12 SS13 SS14)
MODELS=(be sc)
FIT_TARGETS=(update_matrix conditional_psych)
DISTRIBUTION="uniform"

# ── Map TASK_ID ──────────────────────────────────────────────────────────────
N_ANIMALS=${#ANIMALS[@]}
N_MODELS=${#MODELS[@]}
N_TARGETS=${#FIT_TARGETS[@]}
N_TOTAL=$((N_ANIMALS * N_MODELS * N_TARGETS))

TASK_ID=${SLURM_ARRAY_TASK_ID}

if [ "${TASK_ID}" -ge "${N_TOTAL}" ]; then
    echo "Task ID ${TASK_ID} exceeds total ${N_TOTAL}. Exiting."
    exit 0
fi

ANIMAL_IDX=$((TASK_ID / (N_MODELS * N_TARGETS)))
REMAINDER=$((TASK_ID % (N_MODELS * N_TARGETS)))
MODEL_IDX=$((REMAINDER / N_TARGETS))
TARGET_IDX=$((REMAINDER % N_TARGETS))

ANIMAL="${ANIMALS[$ANIMAL_IDX]}"
MODEL="${MODELS[$MODEL_IDX]}"
FIT_TARGET="${FIT_TARGETS[$TARGET_IDX]}"

echo "=== Task ${TASK_ID}/${N_TOTAL}: ${ANIMAL} / ${MODEL} / ${FIT_TARGET} ==="
echo ""

python3 scripts/run_sbi_dynamic_randomwalk.py \
    --animal "${ANIMAL}" \
    --model "${MODEL}" \
    --fit-target "${FIT_TARGET}" \
    --distribution "${DISTRIBUTION}"

echo ""
echo "=== Finished ==="
