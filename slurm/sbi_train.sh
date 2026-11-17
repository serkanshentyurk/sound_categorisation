#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --mem=48G
#SBATCH --time=4-00:00
# One network per array task (TRAIN_GRID: rep x model x distribution). Prior draws are simulated on
# every core of the task (joblib, one thread per worker) until the network's training set is full;
# the density estimator then trains with torch on the same cores. A network with the accuracy rule
# (pooled) keeps only draws whose every session passes it and gives a draw up at its first failing
# session; BE passes about one draw in seven, SC about six in ten.
# Upper estimates per task, from the November 2026 jobs (3 to 4 core-seconds per full simulation,
# no saving from giving failing draws up early): pooled BE (100k kept, about 700k drawn) 30 h;
# pooled SC (100k kept, about 170k drawn) 10 h; single BE / SC (200k, one session each) 12 h.
# The log gives draws, passed, kept and the simulation and training times separately.
# The time limit is 4 days (the partition allows 10); a task that finishes early gives its node back.
# Each task saves its simulations beside the network (snpe_<rep>_<model>_<dist>_training.npz) before
# the density estimator trains: a task that stops after that point is resubmitted as it was and reads
# them back. One that stops while simulating loses them.
# Check the chain first:  sbatch slurm/smoke.sh
# Submit via:  bash slurm/submit.sh sbi-train [sc-sbi-train args]
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "=== sbi_train task ${SLURM_ARRAY_TASK_ID} on $(hostname) ($(nproc) cores) $(date) ==="
sc-sbi-train --task-id "${SLURM_ARRAY_TASK_ID}" "$@"
echo "=== done $(date) ==="
