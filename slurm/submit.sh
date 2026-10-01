#!/usr/bin/env bash
# Submit one stage of the model-identification chain. The array size is derived from the code (so
# the job count can never disagree with the task grid) and the run id is created once here, so every
# array task writes into the same run directory and its logs land in that run's logs/ folder.
#
#   bash slurm/submit.sh train     [sc-train-sbi args]                 # networks → <data root>/snpe_networks
#   bash slurm/submit.sh gs        --source real --distribution uniform --fit-target update_matrix [--run-id ID]
#   bash slurm/submit.sh condition --source real --distribution uniform [--run-id ID]
#
# gs / condition: without --run-id a new run is created under model_identification/<cohort> and its id
# is printed; pass that id to the other stages and to --gather / sc-consensus so they share the run:
#
#   RUN=$(sc-new-run --report model_identification --cohort real)
#   bash slurm/submit.sh gs        --source real --distribution uniform --fit-target update_matrix --run-id $RUN
#   bash slurm/submit.sh condition --source real --distribution uniform --run-id $RUN
#   sc-run-gs --source real --distribution uniform --fit-target update_matrix --run-id $RUN --gather
#   sc-consensus --cohort real --distribution uniform --run-id $RUN
#
# Requires the env to be installed (`pip install -e .` provides the sc-* commands).
set -euo pipefail
cd "$(dirname "$0")/.."
stage="${1:?train|condition|gs}"; shift
case "$stage" in
  train)     cmd=sc-train-sbi; sh=slurm/train_sbi.sh ;;
  condition) cmd=sc-run-sbi;   sh=slurm/run_sbi.sh ;;
  gs)        cmd=sc-run-gs;    sh=slurm/run_gs.sh ;;
  *) echo "unknown stage $stage"; exit 1 ;;
esac

# --- where do the logs go?
if [ "$stage" = train ]; then
  logdir="$(python -c 'from sound_categorisation.data.paths import data_root; print(data_root() / "snpe_networks" / "logs")')"
  mkdir -p "$logdir"
  extra=()
else
  # cohort label: --cohort for synthetic, --label (default real) for real
  cohort="$(python - "$@" <<'PY'
import sys
a = sys.argv[1:]
def get(flag, default=None):
    return a[a.index(flag) + 1] if flag in a else default
src = get('--source', 'synthetic')
print(get('--cohort') if src == 'synthetic' else get('--label', 'real'))
PY
)"
  if printf '%s\n' "$@" | grep -qx -- '--run-id'; then
    run_id="$(python - "$@" <<'PY'
import sys; a = sys.argv[1:]; print(a[a.index('--run-id') + 1])
PY
)"
    extra=()
  else
    run_id="$(sc-new-run --report model_identification --cohort "$cohort")"
    echo "new run: model_identification/$cohort/$run_id"
    extra=(--run-id "$run_id")
  fi
  logdir="$(python -c "from sound_categorisation.data.paths import run_dir; print(run_dir('model_identification', '$cohort', '$run_id') / 'logs')")"
  mkdir -p "$logdir"
fi

range="$("$cmd" --print-array "$@" ${extra[@]+"${extra[@]}"})"
echo "sbatch --array=$range --output=$logdir/${stage}_%A_%a.out --error=$logdir/${stage}_%A_%a.err $sh $* ${extra[*]+"${extra[*]}"}"
sbatch --array="$range" --job-name="$stage" --output="$logdir/${stage}_%A_%a.out" --error="$logdir/${stage}_%A_%a.err" \
       "$sh" "$@" ${extra[@]+"${extra[@]}"}
