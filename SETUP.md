# Setup Guide

## Folder Structure

The project expects this layout — the repo sits inside a parent folder alongside a shared `data/` directory:

```
.../
├── data/
│   └── behaviour/
│       └── snapshots/              ← processed snapshots
└── repos/
    └── sound_categorisation/       ← this repo
```

The lab's raw data lives on a shared drive, mounted differently per machine (e.g. `/Volumes/akrami/` on macOS, `/ceph/akrami/` on the SWC cluster).

## Installation

```bash
cd .../repos
git clone https://github.com/serkanshentyurk/sound_categorisation.git
cd sound_categorisation

conda create -n sound_categorisation python=3.11 -y
conda activate sound_categorisationegorisation

pip install -e behav_utils/
pip install numpy scipy pandas matplotlib seaborn joblib pyyaml ipywidgets

# SBI (required for inference notebooks)
pip install torch sbi

# SSM (required for SLDS notebook)
cd .../repos
git clone https://github.com/lindermanlab/ssm
cd ssm && pip install numpy cython && pip install -e . --no-build-isolation
```

## Data Configuration

The project uses a single `config.yaml` with an environment variable for the data path, so it works on any machine without editing.

### Set the environment variable

Find where the lab drive is mounted on your machine, then add one line to your shell profile.

**macOS** (`~/.zshrc`):
```bash
export BEHAV_DATA_DIR="/Volumes/akrami/Serkan/Head_Fixed_Behavior/Data"
```

**Linux / SWC cluster** (`~/.bashrc`):
```bash
export BEHAV_DATA_DIR="/ceph/akrami/Serkan/Head_Fixed_Behavior/Data"
```

**Windows** (System Settings → Environment Variables):
```
BEHAV_DATA_DIR = Z:\akrami\Serkan\Head_Fixed_Behavior\Data
```

Then reload: `source ~/.zshrc` (macOS) or `source ~/.bashrc` (Linux).

Verify:
```bash
echo $BEHAV_DATA_DIR
ls $BEHAV_DATA_DIR/Raw    # should show animal folders
```

### Cluster: ensure SSH sessions load the variable

SSH login shells sometimes skip `~/.bashrc`. Add to `~/.bash_profile`:
```bash
source ~/.bashrc
```

## Data Loading

Notebooks load data via **snapshots** — preprocessed pickles that are fast to load and don't require the lab drive to be mounted.

### Export a snapshot

On any machine with access to the raw data:
```bash
cd .../repos/sound_categorisation
python scripts/export_snapshot.py
```

This reads CSVs from `$BEHAV_DATA_DIR/Raw`, processes them, and saves the snapshot. On the SWC cluster it saves to the lab drive's `Processed/behaviour/snapshots/` directory; locally it saves to `.../data/behaviour/snapshots/`.

### Copy a snapshot (if you can't access raw data)

Ask a colleague for their `sound_cat_snapshot.pkl` and place it at:
```
.../data/behaviour/snapshots/sound_cat_snapshot.pkl
```

### Use in notebooks

```python
from shared_setup import *
experiment, info = load_data()
```

This tries, in order: snapshot → CSV from config → synthetic fallback. No code changes needed per machine.

### Check if a snapshot is stale

```bash
python scripts/export_snapshot.py --check-only
```

This compares the snapshot's session counts against current raw data and reports any new sessions.

## Syncing Data from the Cluster

After running cluster jobs or exporting a snapshot on the cluster:

**If the lab drive is mounted locally** (macOS):
```bash
# Copy snapshot
cp /Volumes/akrami/.../Processed/behaviour/snapshots/sound_cat_snapshot.pkl \
   .../data/behaviour/snapshots/sound_cat_snapshot.pkl

# Copy cluster results (GS, SBI outputs)
scp -r user@ssh.swc.ucl.ac.uk:~/repos/sound_categorisation/results/ \
    .../repos/sound_categorisation/results/
```

**One-command sync** (optional): place `scripts/sync_snapshot.sh` and run:
```bash
./scripts/sync_snapshot.sh
```

This SSHes into the cluster, exports the snapshot, then copies from the mounted drive.

## Running Notebooks

```bash
cd .../repos/sound_categorisation/notebooks
jupyter notebook
```

Each notebook has a `MODE` toggle:
- `'load'` — reads pre-computed results from `results/` (default)
- `'run'` — quick local execution with small settings

For `'load'` mode, cluster results must be in `results/`. See the cluster pipeline in `README.md`.

## Cluster Setup

```bash
ssh ssh.swc.ucl.ac.uk
cd ~/repos/sound_categorisation

# Load environment
module load miniconda
conda activate sound_categorisation

# Submit jobs (see slurm/README.md for details)
sbatch slurm/train_snpe.sh

# After jobs complete, gather results
python scripts/gather_cv_results.py --all

# Export snapshot
python scripts/export_snapshot.py
```

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Data directory not found: ${BEHAV_DATA_DIR}/Raw` | Env var not set | Add to shell profile, then `source` it |
| `Snapshot is Xh old` | Stale data | Re-export: `python scripts/export_snapshot.py` |
| `Failed to unpickle snapshot` | Code changed since export | Re-export from raw data |
| `Config has changed since snapshot was exported` | Column mappings changed | Re-export |
| Notebooks show synthetic data | No snapshot found | Check `.../data/behaviour/snapshots/` exists |
| `ModuleNotFoundError` | Wrong conda env | `conda activate sound_categorisation` |
| Capitalised + lowercase folders on cluster | macOS case insensitivity | `git config core.ignorecase false`, see README |
