#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Submit a SLURM array job with dynamically determined eligible animals.
#
# Instead of hardcoding animal lists in each SLURM script, this wrapper:
#   1. Queries which animals qualify via list_eligible_animals.py
#   2. Writes the list to a temp file the job reads at runtime
#   3. Computes the correct --array range
#   4. Submits the job
#
# Usage:
#   # GS on all animals with expert uniform sessions:
#   bash slurm/submit.sh gs_uniform --preset expert_uniform
#
#   # Dynamic SBI on animals with ≥5 expert sessions:
#   bash slurm/submit.sh sbi_dynamic --preset expert_uniform --min-sessions 5
#
#   # Synthetic validation (no animal query needed):
#   bash slurm/submit.sh synth_gs static_uniform
#   bash slurm/submit.sh synth_sbi static_uniform
#
#   # SNPE training (no array, no animal query):
#   bash slurm/submit.sh train_snpe be uniform
#
#   # Dry run (show what would be submitted):
#   bash slurm/submit.sh gs_uniform --preset expert_uniform --dry-run
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_DIR="${REPO_DIR}/slurm"
ANIMALS_DIR="${REPO_DIR}/results/logs/animal_lists"
mkdir -p "${ANIMALS_DIR}"

JOB_TYPE="${1:?Usage: bash slurm/submit.sh <job_type> [options]}"
shift

# ── Parse common options ────────────────────────────────────────────────────
PRESET=""
MIN_SESSIONS=3
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)       PRESET="$2"; shift 2 ;;
        --min-sessions) MIN_SESSIONS="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Job-type dispatch ───────────────────────────────────────────────────────

case "${JOB_TYPE}" in

    gs_uniform|sbi_dynamic)
        # These need a dynamic animal list
        if [ -z "${PRESET}" ]; then
            PRESET="expert_uniform"
            echo "No --preset given, defaulting to '${PRESET}'"
        fi

        echo "Querying eligible animals (preset=${PRESET}, min_sessions=${MIN_SESSIONS})..."
        ANIMALS=$(cd "${REPO_DIR}" && python3 scripts/list_eligible_animals.py \
            --preset "${PRESET}" --min-sessions "${MIN_SESSIONS}" --format flat)

        if [ -z "${ANIMALS}" ]; then
            echo "ERROR: No eligible animals found."
            exit 1
        fi

        # Write to file for the SLURM job to read
        ANIMALS_ARRAY=(${ANIMALS})
        N_ANIMALS=${#ANIMALS_ARRAY[@]}
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        ANIMALS_FILE="${ANIMALS_DIR}/${JOB_TYPE}_${TIMESTAMP}.txt"
        printf '%s\n' "${ANIMALS_ARRAY[@]}" > "${ANIMALS_FILE}"

        echo "  ${N_ANIMALS} animals: ${ANIMALS}"
        echo "  Saved to: ${ANIMALS_FILE}"

        # Compute array dimensions
        N_MODELS=2      # BE, SC
        N_TARGETS=2     # update_matrix, conditional_psych
        N_TOTAL=$((N_ANIMALS * N_MODELS * N_TARGETS))
        ARRAY_SPEC="0-$((N_TOTAL - 1))"

        echo "  Array: ${ARRAY_SPEC} (${N_TOTAL} tasks)"

        SLURM_SCRIPT="${SLURM_DIR}/real_${JOB_TYPE#gs_}.sh"
        if [ "${JOB_TYPE}" == "sbi_dynamic" ]; then
            SLURM_SCRIPT="${SLURM_DIR}/sbi_dynamic.sh"
        elif [ "${JOB_TYPE}" == "gs_uniform" ]; then
            SLURM_SCRIPT="${SLURM_DIR}/real_gs_uniform.sh"
        fi

        if [ ! -f "${SLURM_SCRIPT}" ]; then
            echo "ERROR: SLURM script not found: ${SLURM_SCRIPT}"
            exit 1
        fi

        CMD="sbatch --array=${ARRAY_SPEC} --export=ALL,ANIMALS_FILE=${ANIMALS_FILE} ${SLURM_SCRIPT}"
        echo ""
        echo "  ${CMD}"

        if [ "${DRY_RUN}" = true ]; then
            echo "(dry run — not submitted)"
        else
            ${CMD}
        fi
        ;;

    synth_gs|synth_sbi|synthetic_generate|train_snpe)
        # These don't need dynamic animal lists
        SLURM_SCRIPT="${SLURM_DIR}/${JOB_TYPE}.sh"
        if [ ! -f "${SLURM_SCRIPT}" ]; then
            echo "ERROR: SLURM script not found: ${SLURM_SCRIPT}"
            exit 1
        fi

        CMD="sbatch ${SLURM_SCRIPT} ${EXTRA_ARGS[*]:-}"
        echo "${CMD}"

        if [ "${DRY_RUN}" = true ]; then
            echo "(dry run — not submitted)"
        else
            ${CMD}
        fi
        ;;

    *)
        echo "Unknown job type: ${JOB_TYPE}"
        echo ""
        echo "Available:"
        echo "  gs_uniform      Grid-search CV on real data"
        echo "  sbi_dynamic     Dynamic SBI on real data"
        echo "  synth_gs        Grid-search on synthetic cohort"
        echo "  synth_sbi       SBI on synthetic cohort"
        echo "  synthetic_generate  Generate synthetic cohorts"
        echo "  train_snpe      Train SNPE network"
        exit 1
        ;;
esac
