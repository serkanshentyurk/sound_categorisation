#!/bin/bash
#SBATCH --job-name=synth_sbi
#SBATCH --output=results/logs/synth_sbi_%A_%a.out
#SBATCH --error=results/logs/synth_sbi_%A_%a.err
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=1-00:00
#
# SBI model comparison on synthetic cohort.
# Array job: one task per (animal_index × fit_target).
# Requires: trained SNPE networks (run train_snpe.sh first).
#
# Before submitting:
#   Update --array to match: 0-(N_ANIMALS*2 - 1)
#   Default: 40 animals × 2 targets = 80 tasks → --array=0-79
#
# Usage:
#   JOB_SNPE=<jobid from train_snpe>
#   sbatch --dependency=afterok:${JOB_SNPE} --array=0-79 slurm/synth_sbi.sh static_uniform
# ─────────────────────────────────────────────────────────────────────────────

source "/nfs/nhome/live/sshentyurk/repos/sound_categorisation/slurm/env_setup.sh"
cd "${REPO_DIR}"

# ── Configuration ────────────────────────────────────────────────────────────
COHORT="${1:?Usage: sbatch synth_sbi.sh <static_uniform|learning_uniform>}"
N_ANIMALS=40
FIT_TARGETS=(update_matrix conditional_psych)

# ── Map TASK_ID ──────────────────────────────────────────────────────────────
N_TARGETS=${#FIT_TARGETS[@]}
N_TOTAL=$((N_ANIMALS * N_TARGETS))

TASK_ID=${SLURM_ARRAY_TASK_ID}

if [ "${TASK_ID}" -ge "${N_TOTAL}" ]; then
    echo "Task ID ${TASK_ID} exceeds total ${N_TOTAL}. Exiting."
    exit 0
fi

ANIMAL_IDX=$((TASK_ID / N_TARGETS))
TARGET_IDX=$((TASK_ID % N_TARGETS))
FIT_TARGET="${FIT_TARGETS[$TARGET_IDX]}"

echo "=== Task ${TASK_ID}/${N_TOTAL}: cohort=${COHORT}, animal=${ANIMAL_IDX}, ${FIT_TARGET} ==="
echo ""

python3 scripts/validation/run_synth_sbi.py \
    --cohort "${COHORT}" \
    --animal-index "${ANIMAL_IDX}" \
    --fit-target "${FIT_TARGET}"

echo ""
echo "=== Finished ==="
