# Setup

Install on the cluster and on your laptop; the routines that use them are in
[START_HERE.md](START_HERE.md).

## 1. Install

You need conda and Python 3.10 or newer. The conda environment is called `sound_cat`.

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
`pyproject.toml`. Keep the `-e`: the code finds `config.yaml`, the data and the results from the repository.

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

Install your local copy of the library on top; an editable install wins over the pinned one:

```bash
pip install -e <path to your behav_utils checkout>
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

## Troubleshooting

| symptom | fix |
|---|---|
| `BEHAV_DATA_DIR is not set` | add it to `~/.bashrc` / `~/.zshrc` (START_HERE.md) and open a new terminal |
| `SWC_USER is not set` (sc-fetch) | add `export SWC_USER=<your SWC username>` to `~/.zshrc` |
| `ModuleNotFoundError: behav_utils` | `pip install -e .` in this environment |
| `sc-update` stops at `git pull` | your checkout has commits that are not on GitHub; sort it out with git by hand |
| `KeyError: preset 'expert_uniform'` | presets come from `config.yaml`: load the snapshot first, or call `sound_categorisation.data.cohort.ensure_presets()` |
| warning "config has changed" on loading the snapshot | column mappings changed: run `sc-export-snapshot` again; editing session types or presets alone is fine |
| warning that the snapshot was written by another `behav_utils` version | run `sc-export-snapshot` again |
| `compare_groups: need two groups` | only one genotype selected (e.g. `--limit 1`); the rows are still written |
