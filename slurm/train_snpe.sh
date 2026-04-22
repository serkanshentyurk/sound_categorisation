#!/bin/bash
#SBATCH --job-name=train_snpe
#SBATCH --output=results/logs/train_snpe_%j.out
#SBATCH --error=results/logs/train_snpe_%j.err
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=2-00:00
#
# Train one amortised SNPE network.
#
# Usage:
#   sbatch slurm/train_snpe.sh be uniform
#   sbatch slurm/train_snpe.sh sc uniform
# ─────────────────────────────────────────────────────────────────────────────

source "/nfs/nhome/live/sshentyurk/repos/sound_categorisation/slurm/env_setup.sh"
cd "${REPO_DIR}"

MODEL="${1:?Usage: sbatch train_snpe.sh <be|sc> <uniform|hard_a|hard_b>}"
DISTRIBUTION="${2:-uniform}"

echo "=== Training SNPE: ${MODEL} / ${DISTRIBUTION} ==="
echo ""

python3 scripts/train_snpe.py \
    --model "${MODEL}" \
    --distribution "${DISTRIBUTION}"

echo ""
echo "=== Finished ==="
