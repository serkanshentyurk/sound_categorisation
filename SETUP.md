# Sound Categorisation — Setup Guide

## For You (Serkan)

### One-time setup (already done)

**Mac `~/.zshrc`:**
```bash
export BEHAV_DATA_DIR="/Volumes/akrami/Serkan/Head_Fixed_Behavior/Data"
```

**Cluster `~/.bashrc`:**
```bash
export BEHAV_DATA_DIR="/ceph/akrami/Serkan/Head_Fixed_Behavior/Data"
```

**Cluster `~/.bash_profile`** (so SSH sessions load it):
```bash
source ~/.bashrc
```

**Local folder structure:**
```
~/Desktop/pro/PhD/main/
├── data/
│   └── behaviour/
│       └── snapshots/        ← snapshot goes here
└── repos/
    └── sound_categorisation/ ← the repo
```

### Daily workflow

**When new sessions are collected:**
```bash
# 1. SSH into cluster
ssh ssh.swc.ucl.ac.uk

# 2. Export snapshot
cd ~/repos/sound_categorisation
module load miniconda
conda activate sound_cat
python scripts/export_snapshot.py

# 3. Exit
exit

# 4. On Mac — copy from mounted lab drive
cp /Volumes/akrami/Serkan/Head_Fixed_Behavior/Data/Processed/behaviour/snapshots/sound_cat_snapshot.pkl \
   ~/Desktop/pro/PhD/main/data/behaviour/snapshots/sound_cat_snapshot.pkl
```

**Running notebooks (Mac):**
```bash
cd ~/Desktop/pro/PhD/main/repos/sound_categorisation/notebooks
jupyter notebook
```
```python
# First cell of any notebook:
from shared_setup import *
experiment, info = load_data()
```

**Running cluster jobs:**
```bash
ssh ssh.swc.ucl.ac.uk
cd ~/repos/sound_categorisation
module load miniconda
conda activate sound_cat

# Submit jobs as usual
sbatch slurm/train_snpe.sh

# Copy results to Mac when done
# (from Mac terminal:)
scp -r ssh.swc.ucl.ac.uk:~/repos/sound_categorisation/results/ \
    ~/Desktop/pro/PhD/main/repos/sound_categorisation/results/
```

**Check if snapshot is stale:**
```bash
ssh ssh.swc.ucl.ac.uk
cd ~/repos/sound_categorisation
python scripts/export_snapshot.py --check-only
```

---

## For a New Student

### Prerequisites

- Python 3.10+
- Git
- Access to the lab drive (ask IT for mount)
- SSH access to SWC cluster (ask IT)

### Step 1: Find your lab drive mount point

The lab data lives on a shared drive. Find where it's mounted on your machine:

| OS | Typical mount point |
|---|---|
| macOS | `/Volumes/akrami/` |
| Windows | `Z:\akrami\` or similar mapped drive |
| Linux (cluster) | `/ceph/akrami/` |

The data directory is at: `<mount_point>/Serkan/Head_Fixed_Behavior/Data`

Confirm it exists:
```bash
ls <mount_point>/Serkan/Head_Fixed_Behavior/Data/Raw
# Should show animal folders: SS01, SS04, SS05, ...
```

### Step 2: Set the environment variable

**macOS** — add to `~/.zshrc`:
```bash
export BEHAV_DATA_DIR="<mount_point>/Serkan/Head_Fixed_Behavior/Data"
```

**Linux** — add to `~/.bashrc`:
```bash
export BEHAV_DATA_DIR="<mount_point>/Serkan/Head_Fixed_Behavior/Data"
```

**Windows** — System Settings → Environment Variables → New:
```
BEHAV_DATA_DIR = Z:\akrami\Serkan\Head_Fixed_Behavior\Data
```

Then reload:
```bash
source ~/.zshrc    # macOS
source ~/.bashrc   # Linux
# Windows: restart terminal
```

Verify:
```bash
echo $BEHAV_DATA_DIR
# Should print your path
```

### Step 3: Create folder structure

Your working folder can be anywhere. The required structure is:
```
your_folder/
├── data/
│   └── behaviour/
│       └── snapshots/        ← processed data goes here
└── repos/
    └── sound_categorisation/ ← the repo goes here
```

```bash
mkdir -p your_folder/data/behaviour/snapshots
mkdir -p your_folder/repos
```

### Step 4: Clone the repo

```bash
cd your_folder/repos
git clone https://github.com/serkanshentyurk/sound_categorisation.git
```

### Step 5: Install dependencies

```bash
cd sound_categorisation

# Create conda environment
conda create -n sound_cat python=3.11 -y
conda activate sound_cat

# Install behav_utils (editable)
pip install -e behav_utils/

# Install project dependencies
pip install -r requirements.txt
```

### Step 6: Get the data

**Option A — Export from raw data yourself** (if lab drive is mounted):
```bash
python scripts/export_snapshot.py
```

**Option B — Copy an existing snapshot** from a colleague:
```bash
cp /path/to/their/snapshot.pkl your_folder/data/behaviour/snapshots/sound_cat_snapshot.pkl
```

### Step 7: Verify everything works

```bash
cd notebooks
python -c "
from shared_setup import *
experiment, info = load_data()
print(f'Mode: {info[\"mode\"]}')
for aid, animal in experiment.animals.items():
    print(f'  {aid}: {animal.n_sessions} sessions')
"
```

Expected output:
```
Loaded snapshot: 12 animals, 340 sessions (exported 2026-04-25)
Mode: snapshot
  SS01: 42 sessions
  SS04: 31 sessions
  ...
```

If no snapshot exists and the lab drive isn't mounted, it falls back
to synthetic data automatically — you can still run notebooks.

### Step 8: Cluster setup (if running jobs)

```bash
ssh ssh.swc.ucl.ac.uk

# Add env var (one time)
echo 'export BEHAV_DATA_DIR="/ceph/akrami/Serkan/Head_Fixed_Behavior/Data"' >> ~/.bashrc
echo 'source ~/.bashrc' >> ~/.bash_profile
source ~/.bashrc

# Clone repo
cd ~/repos
git clone https://github.com/serkanshentyurk/sound_categorisation.git
cd sound_categorisation

# Create conda env
module load miniconda
conda create -n sound_cat python=3.11 -y
conda activate sound_cat
pip install -e behav_utils/
pip install -r requirements.txt

# Test
python scripts/export_snapshot.py
```

---

## Troubleshooting

**`FileNotFoundError: Data directory not found: ${BEHAV_DATA_DIR}/Raw`**
→ Environment variable not set. Run `echo $BEHAV_DATA_DIR`. If empty, add it to your shell profile and `source` it.

**`Snapshot is Xh old`**
→ Re-export: `python scripts/export_snapshot.py`

**`Failed to unpickle snapshot`**
→ `behav_utils` data classes changed since the snapshot was made. Re-export from raw data.

**`Config has changed since snapshot was exported`**
→ Column mappings may have changed. Re-export to be safe.

**Notebooks show synthetic data instead of real data**
→ No snapshot found. Check `ls your_folder/data/behaviour/snapshots/`.

**Import errors (ModuleNotFoundError)**
→ Make sure you're in the `sound_cat` conda environment: `conda activate sound_cat`.
