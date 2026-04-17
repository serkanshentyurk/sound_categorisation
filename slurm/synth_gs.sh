#!/bin/bash
#SBATCH --job-name=synth_gs
#SBATCH --output=results/logs/synth_gs_%A_%a.out
#SBATCH --error=results/logs/synth_gs_%A_%a.err
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH --time=0-4:00
#
# Grid-search on synthetic cohort.
# Array job: one task per (animal_index × model × fit_target).
#
# Requires: synthetic cohorts already generated (run synthetic_generate.sh first).
#
# Before submitting:
#   Update --array to match: 0-(N_ANIMALS*4 - 1)
#   Default: 20 BE + 20 SC = 40 animals × 2 models × 2 targets = 160 tasks
#            → --array=0-159
#
# Usage:
#   sbatch --array=0-159 slurm/synth_gs.sh static_uniform
#   sbatch --array=0-159 slurm/synth_gs.sh learning_uniform
# ─────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/env_setup.sh"
cd "${REPO_DIR}"

# ── Configuration ────────────────────────────────────────────────────────────
COHORT="${1:?Usage: sbatch synth_gs.sh <static_uniform|learning_uniform>}"
N_ANIMALS=40          # 20 BE + 20 SC
MODELS=(BE SC)
FIT_TARGETS=(update_matrix conditional_psych)

# For learning cohort, use expert sessions only (last 8)
if [ "${COHORT}" == "learning_uniform" ]; then
    SESSIONS_KEY="expert_sessions"
else
    SESSIONS_KEY="sessions"
fi

# ── Map TASK_ID ──────────────────────────────────────────────────────────────
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

MODEL="${MODELS[$MODEL_IDX]}"
FIT_TARGET="${FIT_TARGETS[$TARGET_IDX]}"

echo "=== Task ${TASK_ID}/${N_TOTAL}: cohort=${COHORT}, animal=${ANIMAL_IDX}, ${MODEL}, ${FIT_TARGET} ==="
echo ""

python3 scripts/validation/run_synth_gs.py \
    --cohort "${COHORT}" \
    --animal-index "${ANIMAL_IDX}" \
    --model "${MODEL}" \
    --fit-target "${FIT_TARGET}" \
    --sessions-key "${SESSIONS_KEY}"

echo ""
echo "=== Finished ==="
