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

Both packages must be installed; nothing in the repo puts folders on `sys.path`. Check:

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
python -m scripts.export_snapshot            # writes <data root>/behaviour/snapshots/sound_cat_snapshot.pkl
python -m scripts.export_snapshot --check-only
```

Re-export when sessions are added or when column mappings in `config.yaml` change. Session types and
presets are re-applied from the config at load time, so editing those does **not** require a re-export.
The loader warns when the snapshot is old or the config hash differs.

Animal genotypes come from `animal_metadata.json` next to the data; an animal without one is reported
as `unknown` and excluded from genotype tests.

## 4. Verify

```bash
pytest behav_utils/tests -q           # library, no data needed
pytest tests -q                       # project (torch-only files skip without torch)
ruff check .
python -m sound_categorisation.reports selftest     # synthetic end-to-end, ~30 s
```

## 5. Reports

```bash
bash run_reports.sh                   # selftest → fast structure check on real data → full battery → summary
python -m sound_categorisation.reports group --with-animals --distribution Hard-A --toi opto   # one job
python -m sound_categorisation.reports summary
```

Outputs: `results/reports/<cohort>/…` (git-ignored). See `docs/results_guide.md`.

## 6. Cluster (SWC HPC)

```bash
ssh <user>@ssh.swc.ucl.ac.uk
module load miniconda && conda activate sound_cat
cd <repo>
bash slurm/submit.sh train                                         # 18 SBI networks
bash slurm/submit.sh condition --source real --distribution uniform --run expert
bash slurm/submit.sh gs --source real --fit-target update_matrix --distribution uniform
python -m scripts.run_gs --gather --source real --distribution uniform --fit-target update_matrix
python -m scripts.consensus --run expert --cohort real
```

`submit.sh` asks each script for its array range (`--print-array`) so the job count always matches the
task grid in `sound_categorisation/tasks.py`. Logs go to `results/logs/`. Smoke first:
`sbatch --array=0 slurm/train_sbi.sh --smoke-test`.

## 7. Notebooks

`notebooks/shared_setup.py` gives `load_data()` (snapshot or CSV), paths and cohorts. Analysis imports go
in the cell that uses them. The notebooks are being rewritten to read `results/reports/` tables rather
than recompute (see ARCHITECTURE.md, "Notebooks").

## Troubleshooting

| symptom                               | cause / fix                                                                                                                                                    |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: behav_utils`  | not installed in this env →`pip install -e behav_utils/`                                                                                                    |
| `KeyError: preset 'expert_uniform'` | presets come from the config; load an experiment/snapshot first, or call`sound_categorisation.cohort.ensure_presets()`                                       |
| snapshot "config has changed" warning | column mappings changed → re-export; session-type/preset edits alone are fine                                                                                 |
| `compare_groups: need two groups`   | only one genotype in the selection (e.g.`--limit 1`); rows are still written, tests skipped                                                                  |
| CI passes locally but not on GitHub   | a file under`sound_categorisation/reports/` not committed (check `git status`), or ruff run only on part of the tree — run `ruff check .` from the root |
