# Sound Categorisation

Computational modelling and analysis pipeline for investigating the dynamic necessity of posterior parietal cortex (PPC) during statistical model updating in mice.

## The Experiment

Mice perform a two-alternative forced choice (2-AFC) sound amplitude categorisation task. Structured noise stimuli are presented; mice report "low" or "high" via left/right lick. After reaching expert performance on a uniform stimulus distribution, the distribution shifts (Uniform → Hard-A → Hard-B → Hard-A), creating three qualitatively different transitions: first novel, second with meta-experience, and familiar return.

## The Question

Is PPC causally necessary when an animal's internal statistical model of the stimulus distribution is inadequate for current demands, and does it become dispensable once that model is sufficient?

## Approach

Two computational models describe how mice update their internal model:

- **BE (Boundary Estimation)**: tracks the optimal decision boundary via exponential moving average, parametrised by learning rate η
- **SC (Stimulus Category)**: tracks category distributions via kernel density estimation, parametrised by category weight γ

The pipeline assigns each animal as BE or SC, tracks parameter dynamics across learning and distribution shifts, and uses these assignments to predict optogenetic inactivation effects.

## Repository Structure

```
sound_categorisation/
├── behav_utils/          # Cross-lab behavioural data library (see its own README)
├── models/               # BE and SC computational models
├── analysis/             # Project-specific analysis (grid search, adaptation, validation)
├── inference/            # Simulation-based inference (SBI/SNPE, diagnostics)
├── notebooks/            # Top-level analysis notebooks (00–50, load results)
│   └── dev/              # Development/validation notebooks (run locally)
├── scripts/              # Cluster and local entry points
│   └── validation/       # Synthetic validation scripts
├── slurm/                # SLURM submission scripts for the SWC cluster
├── results/              # All outputs (git-ignored)
├── legacy/               # Old codebase (reference only)
├── config.yaml           # Data loading configuration
└── shared_setup.py       # Notebook import boilerplate
```

## Installation

```bash
git clone https://github.com/serkanshentyurk/sound_categorisation.git
cd sound_categorisation
# Create environment
conda create -n sound_categorisation python=3.11
conda activate sound_categorisation

# Core dependencies
pip install numpy scipy pandas matplotlib seaborn scikit-learn joblib pyyaml ipywidgets

# SBI (optional, required for notebooks 21, 31, and SBI scripts)
pip install torch sbi

# SSM (required for notebook 40)
cd ~/repos
git clone https://github.com/lindermanlab/ssm
cd ssm
pip install numpy cython
pip install -e . --no-build-isolation

# Install behav_utils in development mode
cd ~/repos/sound_categorisation
pip install -e behav_utils/
```

## Data Setup

The config uses an environment variable for the data path, so one config file works on any machine.

**Set once per machine** — add to your shell profile (`~/.zshrc` on Mac, `~/.bashrc` on cluster):
```bash
export BEHAV_DATA_DIR="/your/mount/point/Head_Fixed_Behavior/Data"
```

**Export a snapshot** (converts raw CSVs to a fast-loading pickle):
```bash
python scripts/export_snapshot.py
```

**Sync snapshot to local machine** (Mac, with lab drive mounted):
```bash
./scripts/sync_snapshot.sh
```

Notebooks load the snapshot automatically via `shared_setup.py`. See [SETUP.md](SETUP.md) for detailed instructions and new student onboarding.

## Notebooks

Numbered for reading order. Each has a `MODE` toggle at the top: `'load'` reads cluster results, `'run'` does a quick local analysis.

| # | Notebook | Purpose |
|---|---|---|
| 00 | Data Exploration | Raw data overview, trial counts, accuracy trajectories |
| 01 | Model Explorer | Interactive BE/SC parameter widgets |
| 02 | Feature Selection | PCA, feature correlations, stat selection |
| 03 | Parameter Sensitivity | How summary stats respond to model parameters |
| 04 | Model vs Real | Model predictions overlaid on real data |
| 10 | **Validation Summary** | Synthetic validation accuracy across scenarios and fit targets |
| 20 | **GS Model Selection** | Per-animal BE/SC assignment via grid-search CV |
| 21 | **SBI Model Selection** | Per-animal BE/SC assignment via amortised SNPE |
| 22 | **Consensus** | 4-method agreement matrix and final assignments |
| 30 | Adaptation Analysis | Post-shift behavioural adaptation |
| 31 | Parameter Dynamics | Dynamic SBI parameter trajectories (RandomWalk) |
| 40 | SLDS | Latent behavioural state inference |
| 50 | Opto Predictions | Optogenetic effect predictions from model + SLDS |

Development notebooks in `notebooks/dev/` (2a–2h) contain the detailed synthetic validation work.

## Cluster Pipeline

All compute-heavy work runs on the SWC cluster via SLURM. Scripts are in `scripts/`, SLURM wrappers in `slurm/`.

### Before submitting

```bash
# Verify everything works locally
bash scripts/test_all.sh
```

### Submission order

```bash
# Stage 1: no dependencies
JOB_GEN=$(sbatch --parsable slurm/synthetic_generate.sh)
JOB_BE=$(sbatch --parsable slurm/train_snpe.sh be uniform)
JOB_SC=$(sbatch --parsable slurm/train_snpe.sh sc uniform)

# Stage 2: after synthetic cohorts generated
sbatch --dependency=afterok:${JOB_GEN} --array=0-159 slurm/synth_gs.sh static_uniform
sbatch --dependency=afterok:${JOB_GEN} --array=0-159 slurm/synth_gs.sh learning_uniform

# Stage 3: after SNPE trained + cohorts generated
sbatch --dependency=afterok:${JOB_BE}:${JOB_SC},afterok:${JOB_GEN} \
       --array=0-79 slurm/synth_sbi.sh static_uniform

# Stage 4: real data (no dependency)
sbatch --array=0-47 slurm/real_gs_uniform.sh

# Local: condition SBI on real animals
python scripts/condition_sbi_local.py --distribution uniform

# Stage 5 (nice-to-have): dynamic SBI
sbatch --array=0-47 slurm/sbi_dynamic.sh
```

### After cluster jobs complete

```bash
# Aggregate results
python scripts/gather_cv_results.py --all --include-validation

# Open notebooks 10, 20, 21, 22 with MODE = 'load'
```

## Model Selection Methods

Four methods are used for BE vs SC assignment. Each combines a fitting method with a scoring target:

| Method | Fitting | Scoring target |
|---|---|---|
| GS × UM | Grid-search CV (64 seeds) | Update matrix MSE |
| GS × CP | Grid-search CV (64 seeds) | Conditional psychometric MSE |
| SBI × UM | Amortised SNPE (heuristic stats) | Update matrix MSE |
| SBI × CP | Amortised SNPE (heuristic stats) | Conditional psychometric MSE |

An animal is confidently assigned when all four methods agree. The consensus notebook (22) reports per-animal agreement.

## Key Design Decisions

- **Stimulus range normalised to [-1, 1]** throughout
- **Session-level analysis**: all summary stats and model fitting operate at session granularity
- **Block-level CV splits**: CV folds split by session block, not by trial, preserving sequential structure
- **SBI trained on heuristics only**: no update matrix or conditional psychometric in the SNPE training stats; these are used only for scoring
- **Dynamic SBI uses RandomWalk**: per-animal, not amortised; learning parameters vary, perceptual parameters are constant
- **Every result includes metadata**: git SHA, timestamp, hostname, library versions, config constants

## Dependencies

- Python 3.11+
- numpy, scipy, pandas, matplotlib, seaborn, scikit-learn, joblib
- PyTorch + sbi (for SBI analysis)
- ssm (for SLDS, Linderman lab)
- ipywidgets (for interactive notebooks)

## References

- Akrami et al. (2018). Posterior parietal cortex represents sensory history and mediates its effects on behaviour. *Nature*.
- Cranmer, Brehmer, Louppe (2020). The frontier of simulation-based inference. *PNAS*.
- Linderman et al. (2017). Bayesian learning and inference in recurrent switching linear dynamical systems. *AISTATS*.
