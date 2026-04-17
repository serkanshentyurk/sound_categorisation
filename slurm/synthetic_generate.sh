#!/bin/bash
#SBATCH --job-name=synth_gen
#SBATCH --output=results/logs/synth_gen_%j.out
#SBATCH --error=results/logs/synth_gen_%j.err
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=0-2:00
#
# Generate synthetic cohorts (static + learning, 20 BE + 20 SC each).
# Single job, run once before synth_gs and synth_sbi.
#
# Usage:
#   sbatch slurm/synthetic_generate.sh
# ─────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/env_setup.sh"
cd "${REPO_DIR}"

echo "=== Generating Synthetic Cohorts ==="
echo ""

python3 scripts/generate_synthetic_cohort.py

echo ""
echo "=== Finished ==="
