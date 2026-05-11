#!/bin/bash
#SBATCH --job-name=synth_dyn
#SBATCH --output=results/logs/synth_dyn_%A_%a.out
#SBATCH --error=results/logs/synth_dyn_%A_%a.err
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH --time=2-00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#
# Synthetic dynamic SBI validation.
# Each task generates one synthetic animal (true model = BE or SC),
# then runs BOTH BE and SC dynamic SBI with RandomWalk linking.
#
# Array: 10 BE + 10 SC = 20 animals → --array=0-19
# Each task ~30-60 min (two model fits per animal).
#
# Usage:
#   sbatch --array=0-19 slurm/synth_sbi_dynamic.sh
#
#   # Smoke test (faster, fewer sims)
#   sbatch --array=0-3 --export=SMOKE=1 slurm/synth_sbi_dynamic.sh
# ─────────────────────────────────────────────────────────────────────────────

source "/nfs/nhome/live/sshentyurk/repos/sound_categorisation/slurm/env_setup.sh"
cd "${REPO_DIR}"

mkdir -p results/logs

# ── Configuration ────────────────────────────────────────────────────────────
N_PER_MODEL=10
MODELS=(be sc)

SMOKE_FLAG=""
if [ "${SMOKE:-0}" = "1" ]; then
    SMOKE_FLAG="--smoke-test"
    echo "** SMOKE TEST MODE **"
fi

# ── Map TASK_ID ──────────────────────────────────────────────────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID}
N_MODELS=${#MODELS[@]}
N_TOTAL=$((N_PER_MODEL * N_MODELS))

if [ "${TASK_ID}" -ge "${N_TOTAL}" ]; then
    echo "Task ID ${TASK_ID} exceeds total ${N_TOTAL}. Exiting."
    exit 0
fi

MODEL_IDX=$((TASK_ID / N_PER_MODEL))
ANIMAL_IDX=$((TASK_ID % N_PER_MODEL))

MODEL="${MODELS[$MODEL_IDX]}"

echo "=== Task ${TASK_ID}/${N_TOTAL}: ${MODEL} #${ANIMAL_IDX} ==="
echo ""

python3 scripts/validation/run_synth_sbi_dynamic.py \
    --model "${MODEL}" \
    --animal-index "${ANIMAL_IDX}" \
    ${SMOKE_FLAG}

echo ""
echo "=== Finished ==="
