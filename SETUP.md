# Setup

Install on the cluster and on your laptop; the routines that use them are in
[START_HERE.md](START_HERE.md).

## 1. Install

You need conda and Python 3.11 or newer (behav_utils 1.0 needs it). The conda environment is called `sound_cat`.

On the cluster (SWC HPC: `ssh <SWC username>@ssh.swc.ucl.ac.uk`, then `ssh hpc-gw2`):

```bash
module load miniconda
conda create -n sound_cat python=3.11
conda activate sound_cat
mkdir -p ~/repos
cd ~/repos
git clone https://github.com/serkanshentyurk/sound_categorisation.git
cd sound_categorisation
pip install -e ".[dev,fit]"
```

On the laptop, clone into a `repos` folder too: the local data then sits next to it, in `<X>/data`.

```bash
conda create -n sound_cat python=3.11
conda activate sound_cat
mkdir -p <X>/repos
cd <X>/repos
git clone https://github.com/serkanshentyurk/sound_categorisation.git
cd sound_categorisation
pip install -e ".[dev]"
```

`[dev]` adds pytest and ruff; `[fit]` adds torch and sbi, which only the cluster needs (model
identification). `pip install -e .` installs the `sc-*` commands and `behav_utils` at the tag pinned in
`pyproject.toml`, with its `store` extra (pyarrow), which the snapshot needs. Keep the `-e`: the code finds
`config.yaml`, the data and the results from the repository.

## 2. One-time settings

Add the lines in [START_HERE.md, "Your own data folder"](START_HERE.md#your-own-data-folder) to
`~/.bashrc` on the cluster and `~/.zshrc` on the laptop, then open a new terminal. On the cluster that gives
you `load-env`, which goes to the repository and loads the environment (`slurm/env.sh`).

## 3. Check

```bash
python -c "import behav_utils, sound_categorisation; print(behav_utils.__version__, sound_categorisation.__version__)"
echo $BEHAV_DATA_DIR
```

## 4. Tests

```bash
pytest tests/unit -q
pytest tests/e2e -q
ruff check .
```

`tests/unit` takes seconds and `tests/e2e` minutes (the report pipeline on synthetic data, compared with
pinned numbers). `pytest tests/fit -q` needs torch and sbi and skips itself without them. Tests only
write to temporary folders.

If a change moves the pinned numbers on purpose, regenerate them and say so in the commit:

```bash
pytest tests/e2e -q --regen-reference
```

## 5. Working on behav_utils at the same time

Install your local copy of the library on top, with its `store` extra; an editable install wins over the
pinned one:

```bash
pip install -e "<path to your behav_utils checkout>[store]"
```

Then update the project with `sc-update --keep-behav-utils`, so your copy stays. When the project needs a library
change, tag a release of `behav_utils` and move the tag in `pyproject.toml`.

## 6. Notebooks

[notebooks/README.md](notebooks/README.md). They read the newest run of each report. To try them without
data:

```bash
sc-make-synthetic-run --results /tmp/sc_results --data /tmp/sc_data
SC_NB_SYNTHETIC=1 SC_RESULTS_ROOT=/tmp/sc_results SC_DATA_ROOT=/tmp/sc_data jupyter lab notebooks/
```

## 7. Which environment made a result

Every run folder holds `environment.txt`, the installed packages and their versions at the time the run was
created; nothing needs doing for that. `pyproject.toml` does not pin versions, so for results that have to be
reproducible later (a thesis, a paper), also commit the cluster environment once the runs are final:

```bash
conda env export -n sound_cat --no-builds > environment.lock.yml
git add environment.lock.yml
git commit -m "Environment of the final runs"
```

`conda env create -f environment.lock.yml` rebuilds it.

## Troubleshooting

| symptom | fix |
|---|---|
| `BEHAV_DATA_DIR is not set` | add it to `~/.bashrc` / `~/.zshrc` (START_HERE.md) and open a new terminal |
| `SWC_USER is not set` (sc-fetch) | add `export SWC_USER=<your SWC username>` to `~/.zshrc` |
| `ModuleNotFoundError: behav_utils` | `pip install -e .` in this environment |
| `sc-update` stops at `git pull` | your checkout has commits that are not on GitHub; sort it out with git by hand |
| `KeyError: preset 'expert_uniform'` | presets come from `config.yaml`: load the snapshot first, or call `sound_categorisation.data.cohort.ensure_presets()` |
| warning "config has changed" on loading the snapshot | column mappings, `session_types` or `sessions_to_ignore` changed: run `sc-export-snapshot` again (they are applied when the data are loaded); editing presets alone is fine |
| "Snapshot format N, this code reads 4", or "is a pickle snapshot" | run `sc-export-snapshot` again on the cluster, then `sc-fetch snapshot` |
| `ImportError: cannot import name … from 'behav_utils.data'` | this environment has behav_utils older than 1.0: `pip install -e .`, or `python -m pip install --upgrade "behav_utils[store] @ git+https://github.com/serkanshentyurk/behav_utils.git@v1.0.0"` |
| "the store needs pyarrow" | `pip install -e .` (it brings behav_utils' `store` extra), or `pip install pyarrow` |
| `which python` is the shared miniconda, not `~/.conda/envs/sound_cat/bin/python` (pip then fails with "Permission denied") | `source slurm/env.sh`, which puts the environment first; if `~/.bashrc` activates it too, wrap those lines in `if [[ $- == *i* ]]; then … fi` |
| export stops: "the animal_id column says 'X' but the reader says 'Y'" | a session folder filed under the wrong animal: move it to the right animal's folder, or list its date under `sessions_to_ignore` |
| Slurm job pending with `ReqNodeNotAvail` | the `cpu` partition's nodes are down (`sinfo -p cpu`): `scontrol update JobId=<id> Partition=cpu_backfill`, or submit with `-p cpu_backfill` |
| `sbatch --wrap` job fails with "source: not found" | `--wrap` runs `sh`: wrap the command in `bash -lc '…'` |
| `compare_groups: need two groups` | only one genotype selected (e.g. `--limit 1`); the rows are still written |
