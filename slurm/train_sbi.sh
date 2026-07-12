#!/bin/bash
#SBATCH --job-name=train_sbi
#SBATCH --output=results/logs/train_sbi_%A_%a.out
#SBATCH --error=results/logs/train_sbi_%A_%a.err
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=3-00:00
#SBATCH --array=0-17
#
# Train the 18 amortised SBI networks = 3 reps x 2 models x 3 distributions.
# One network per array task; index -> (rep, model, distribution) via decode_task
# (rep-major, model-mid, distribution-minor). No GPU: the training is CPU-only.
#
# Submit from the repo root:
#   mkdir -p results/logs            # Slurm needs the log dir to exist first
#   sbatch slurm/train_sbi.sh
#
# Smoke first (one quick task, 500 sims) before the real 18:
#   sbatch --array=0 slurm/train_sbi.sh --smoke-test
#
# Re-run a subset (e.g. only the moments nets, tasks 6-11):
#   sbatch --array=6-11 slurm/train_sbi.sh
#
# RESOURCES: --time=3-00:00 and --mem=32G are deliberately generous, NOT tight
# estimates — the moments rep runs 150k simulations (tasks 6-11) and is the slow
# one (pooled/single use 50k), and SBI training time is hard to predict up front,
# so this errs high to avoid a mid-run timeout. Trade-off: a long requested wall
# can sit longer in the queue (less backfill). After the first real run, read the
# actual runtimes from the logs and tighten if you want faster scheduling.
# -c 8 assumes the simulator parallelises; check CPU efficiency in the logs.
# ---------------------------------------------------------------------------
set -euo pipefail

module load miniconda
conda activate sound_cat
cd "${SLURM_SUBMIT_DIR}"

echo "=== train_sbi task ${SLURM_ARRAY_TASK_ID}/17 on $(hostname) $(date) ==="
python -m scripts.train_sbi --task-id "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done $(date) ==="
