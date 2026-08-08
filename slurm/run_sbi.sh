#!/bin/bash
#SBATCH --job-name=run_sbi
#SBATCH --output=results/logs/run_sbi_%A_%a.out
#SBATCH --error=results/logs/run_sbi_%A_%a.err
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --array=0-5
#
# Condition the trained SBI networks on a cohort = 3 reps x 2 models = 6
# (rep, model) tasks PER PHASE. --distribution picks the phase (its phase-matched
# networks and, for real data, its sessions), so submit once per phase and pass
# the phase + run options AFTER the script name (they reach run_sbi via "$@"):
#
#   mkdir -p results/logs            # Slurm needs the log dir to exist first
#   sbatch --array=0-5 slurm/run_sbi.sh --source real --distribution uniform --run expert
#   sbatch --array=0-5 slurm/run_sbi.sh --source real --distribution hard_a  --run expert
#   sbatch --array=0-5 slurm/run_sbi.sh --source real --distribution hard_b  --run expert
#
# Synthetic (the --cohort name should encode the phase):
#   sbatch --array=0-5 slurm/run_sbi.sh --source synthetic --cohort static_uniform \
#          --distribution uniform --run full
#
# Smoke first (one task, 2 CV repeats) before the full 6:
#   sbatch --array=0 slurm/run_sbi.sh --source real --distribution uniform --run smoke --smoke-test
#
# Requires the 18 networks already trained (slurm/train_sbi.sh). Conditioning is
# much lighter than training (forward passes + held-out simulation, no SNPE fit),
# so mem/time are modest; read the logs and tighten if you want.
# ---------------------------------------------------------------------------
set -euo pipefail

module load miniconda
conda activate sound_cat
cd "${SLURM_SUBMIT_DIR}"

echo "=== run_sbi task ${SLURM_ARRAY_TASK_ID}/5 on $(hostname) $(date) ==="
python -m scripts.run_sbi --task-id "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done $(date) ==="
