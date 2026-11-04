# The cluster environment for this repo. Source it, don't run it:
#
#   source slurm/env.sh
#
# The Slurm jobs source it, and so does the load-env alias (START_HERE.md, "On the cluster").
module load miniconda
conda activate sound_cat
export PYTHONNOUSERSITE=1      # ignore packages in ~/.local: they would be imported instead of the env's
export PYTHONUNBUFFERED=1      # progress lines reach Slurm's .out files as they happen
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ -z "${BEHAV_DATA_DIR:-}" ]; then
  echo "BEHAV_DATA_DIR is not set. Add to ~/.bashrc: export BEHAV_DATA_DIR=/ceph/akrami/<Name>/Head_Fixed_Behavior/Data" >&2
fi
