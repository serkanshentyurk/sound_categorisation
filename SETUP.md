# Setup

## 1. Environment

Python ≥ 3.10. One conda env for everything; on the cluster it is called `sound_cat`, locally whatever
you like.

```bash
conda create -n sound_cat python=3.11 && conda activate sound_cat
cd <repo root>                      # the folder with pyproject.toml
pip install -e behav_utils/         # the library, from src/ layout
pip install -e ".[dev]"             # the project + pytest + ruff
pip install -e ".[fit]"             # torch, sbi, ssm, hmmlearn — only needed for model fitting
```

Both packages must be installed; `pip install -e .` also installs the `sc-*` commands (`sc-reports`, `sc-export-snapshot`, `sc-train-sbi`, `sc-run-sbi`, `sc-run-gs`, `sc-consensus`). Check:

```bash
python -c "import behav_utils, sound_categorisation; print(behav_utils.__version__, behav_utils.__file__)"
```

The path must end in `behav_utils/src/behav_utils/__init__.py`.

## 2. Data

Raw sessions are Bonsai CSVs, one folder per animal, per session, under a root the config points to.
Two locations are known to the code (`sound_categorisation/paths.py`):

| where   | data root                                                                   |
| ------- | --------------------------------------------------------------------------- |
| laptop  | `<repo>/../../data/` — i.e. a `data/` folder two levels above the repo |
| cluster | `/ceph/akrami/Serkan/Head_Fixed_Behavior/Data/Processed`                  |

`config.yaml` (repo root) maps the CSV columns, names the cohorts, defines session presets and
per-animal session-type overrides (`session_types`). On the cluster a `config_slurm.yaml` next to it,
if present, overrides paths.

## 3. Snapshot

Everything analysis-side loads a pickled snapshot of the experiment rather than the CSVs:

```bash
sc-export-snapshot            # writes <data root>/behaviour/snapshots/sound_cat_snapshot.pkl
sc-export-snapshot --check-only
```

Re-export when sessions are added or when column mappings in `config.yaml` change. Session types and
presets are re-applied from the config at load time, so editing those does **not** require a re-export.
The loader warns when the snapshot is old or the config hash differs.

Animal genotypes come from `animal_metadata.json` next to the data; an animal without one is reported
as `unknown` and excluded from genotype tests.

## 4. Verify

```bash
pytest behav_utils/tests -q           # library, no data needed
pytest tests/unit -q                  # project, fast
pytest tests/e2e -q                   # the report pipeline on synthetic data + pinned numbers, minutes
pytest tests/fit -q                   # torch + sbi; skips itself where they are missing
ruff check .
```

Tests never touch a results root; everything goes to pytest's `tmp_path`. To keep the e2e outputs for a
look: `pytest tests/e2e -q --basetemp=/tmp/sc_e2e`. If a deliberate change moves the pinned numbers:
`pytest tests/e2e -q --regen-reference`, then commit `tests/e2e/reference/e2e_contrasts.csv` and say so.

## 5. Runs and results

Every analysis run writes under one root as `<report>/<cohort>/<run_id>/` and points `latest` at
itself. The root is `<repo>/results` locally and `<data root>/results` on the cluster; `SC_RESULTS_ROOT`
overrides it (and `SC_DATA_ROOT` the data root). Nothing under it is versioned — provenance is the
`meta.json` in every folder (run id, command line, snapshot, settings, versions, git sha + dirty flag).

```bash
sc-reports battery                                         # fast check → full opto battery → summary
sc-reports opto --distribution Hard-A --toi opto           # one condition, a new run
sc-reports opto --distribution Hard-A --run-id <id>        # into an existing run
sc-reports switches --cohort behaviour1-cohort             # the switch-adaptation report
sc-reports summary [--run <id>]                            # pages from the latest (or named) opto run
```

`docs/runs.md` lists every command, what it computes, inputs, outputs and rough duration;
`docs/results_guide.md` explains how to read an opto run.

## 6. Cluster (SWC HPC)

```bash
ssh <user>@ssh.swc.ucl.ac.uk
module load miniconda && conda activate sound_cat
cd <repo> && pip install -e behav_utils/ && pip install -e .     # once per checkout: provides the sc-* commands
bash slurm/submit.sh train                                                     # 18 SBI networks → data root
RUN=$(sc-new-run --report model_identification --cohort real)                  # one run id for the chain
bash slurm/submit.sh gs        --source real --distribution uniform --fit-target update_matrix --run-id $RUN
bash slurm/submit.sh condition --source real --distribution uniform --run-id $RUN
sc-run-gs --source real --distribution uniform --fit-target update_matrix --run-id $RUN --gather
sc-consensus --cohort real --distribution uniform --run-id $RUN
```

`submit.sh` asks each command for its array range (`--print-array`) so the job count always matches the
task grid in `sound_categorisation/inference/tasks.py`, and sends the Slurm logs to `<run>/logs/`. Check
the pipeline first with `--fast` (tiny grids, few repeats; the run id gets a `_fast` suffix).

## 7. Notebooks

`notebooks/shared_setup.py` gives `load_data()` (snapshot or CSV), paths and cohorts. Analysis imports go
in the cell that uses them. The notebooks are being rewritten to read a run's tables
(`paths.resolve_run`) rather than recompute (see ARCHITECTURE.md, "Notebooks").

## Troubleshooting

| symptom                               | cause / fix                                                                                                                                                    |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: behav_utils`  | not installed in this env →`pip install -e behav_utils/`                                                                                                    |
| `KeyError: preset 'expert_uniform'` | presets come from the config; load an experiment/snapshot first, or call`sound_categorisation.data.cohort.ensure_presets()`                                       |
| snapshot "config has changed" warning | column mappings changed → re-export; session-type/preset edits alone are fine                                                                                 |
| `compare_groups: need two groups`   | only one genotype in the selection (e.g.`--limit 1`); rows are still written, tests skipped                                                                  |
| CI passes locally but not on GitHub   | a file under`sound_categorisation/reports/` not committed (check `git status`), or ruff run only on part of the tree — run `ruff check .` from the root |
