#!/usr/bin/env bash
# Submit one stage of the model-identification chain with the array size derived from the code,
# so the job count can never disagree with the task grid.
#
#   bash slurm/submit.sh train  [train_sbi args]        # networks:  sound_categorisation.inference.tasks.TRAIN_GRID
#   bash slurm/submit.sh condition --source real --distribution uniform ...   # CONDITION_GRID
#   bash slurm/submit.sh gs --source real --fit-target update_matrix ...      # gs_grid(animals, n_seeds)
#
# Afterwards:  sc-run-gs --gather ...   and   sc-consensus --cohort real
# Entry points come from `pip install -e .` (pyproject [project.scripts]); the env must be installed.
set -euo pipefail
cd "$(dirname "$0")/.."
stage="${1:?train|condition|gs}"; shift
case "$stage" in
  train)     cmd=sc-train-sbi; sh=slurm/train_sbi.sh ;;
  condition) cmd=sc-run-sbi;   sh=slurm/run_sbi.sh ;;
  gs)        cmd=sc-run-gs;    sh=slurm/run_gs.sh ;;
  *) echo "unknown stage $stage"; exit 1 ;;
esac
range="$("$cmd" --print-array "$@")"
mkdir -p results/logs
echo "sbatch --array=$range $sh $*"
sbatch --array="$range" "$sh" "$@"
