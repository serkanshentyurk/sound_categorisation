# behav_utils

Config-driven Python library for loading, analysing, and plotting trial-based behavioural data from head-fixed rodent experiments.

## Overview

`behav_utils` provides a clean data class hierarchy and a registry of summary statistics for 2-AFC behavioural tasks. It is designed to be reusable across projects in the lab — the library knows nothing about specific computational models (BE, SC, etc.) and contains no project-specific analysis code.

## Installation

```bash
# Development mode (recommended during active use)
pip install -e .

# Or standard install
pip install .
```

Requires Python 3.10+.

## Quick Start

```python
from behav_utils import load_experiment

# Load from a YAML config that maps CSV columns to internal names
experiment = load_experiment('config.yaml')
animal = experiment.get_animal('SS05')
session = animal.sessions[0]

# Per-session summary statistics
session.stats(['accuracy', 'recency', 'win_stay'])
# → {'accuracy': 0.82, 'recency': 0.15, 'win_stay': 0.71}

# Stat trajectory across sessions
animal.stat_trajectory('accuracy')
# → array([0.54, 0.61, 0.68, 0.75, 0.82, ...])

# Plotting
session.plot_psychometric()
animal.plot_trajectory('accuracy')
experiment.plot_trajectory('accuracy', combine='mean_sem')
```

## Data Classes

```
ExperimentData
  └── AnimalData (one per animal)
        └── SessionData (one per session)
              ├── SessionMetadata (fields dict: animal_id, stage, protocol, ...)
              └── TrialData (arrays: stimulus, choice, correct, outcome, ...)

FittingData    (flat arrays pooled from sessions, for model fitting)
```

**`TrialData`** holds per-trial numpy arrays. Required: `stimulus`, `choice`, `correct`, `outcome`, `trial_number`. Optional: `reaction_time`, `abort`, `opto_on`, `category`.

**`SessionData`** wraps `TrialData` with metadata and provides `.stats()` for computing summary statistics. Also exposes `.plot_psychometric()` and `.plot_trials()`.

**`AnimalData`** holds a list of sessions and provides `.get_sessions(stage=...)` for filtering, `.stat_trajectory(stat_name)` for tracking a statistic across sessions, and plotting methods.

**`SessionMetadata`** is a flexible dict wrapper. Fields are populated from the YAML config's column mappings. Common properties (`animal_id`, `stage`, `date`) are exposed as attributes.

## Config-Driven Loading

A YAML config file defines how CSV columns map to internal field names:

```yaml
project:
  name: "My Experiment"
  data_dir: "/path/to/data"

session_metadata:
  animal_id:
    csv_name: "Subject"
    dtype: str
  stage:
    csv_name: "Training_Stage"
    dtype: str

trial_data:
  stimulus:
    csv_name: "Stimulus_Value"
    dtype: float
  choice:
    csv_name: "Response_Side"
    dtype: float
    mapping: {"Left": 0, "Right": 1}
```

This means `behav_utils` works with any lab's CSV format without code changes — just update the config.

## Summary Statistics

Over 20 registered statistics, computed via `session.stats()` or `compute_summary_stats()`:

| Category | Stats |
|---|---|
| Performance | `accuracy`, `hard_accuracy`, `easy_accuracy`, `hard_easy_ratio`, `binned_accuracy` |
| Psychometric | `psychometric` (PSE, slope, lapse), `conditional_psychometric`, `psychometric_gof` |
| History effects | `recency`, `stimulus_recency`, `recency_divergence`, `history_interaction_r2` |
| Choice patterns | `win_stay`, `lose_shift`, `choice_autocorr`, `perseveration`, `choice_entropy` |
| Bias | `side_bias` |
| Sensitivity | `stimulus_sensitivity`, `sd_profile`, `binned_choice_prob` |
| Sequential | `update_matrix` (8×8 conditional P(choice) matrix) |
| History model | `logistic_history` (GLM weights for recent stimuli/choices) |

### Adding a new statistic

```python
from behav_utils.analysis.summary_stats import register_stat

@register_stat('my_stat')
def compute_my_stat(choices, stimuli, categories, **kwargs):
    """Compute something from trial arrays."""
    return {'my_value': float(np.mean(choices))}
```

The function receives filtered trial arrays and returns a dict of named values. It is then available via `session.stats(['my_stat'])`.

## Update Matrix

The update matrix quantifies how the previous stimulus influences the current choice, conditioned on stimulus bins:

```python
from behav_utils.analysis.update_matrix import compute_update_matrix

um, conditional_matrix, info = compute_update_matrix(
    stimuli, choices, categories,
    n_bins=8,
    trial_filter='post_correct',
)
# um: (8, 8) — shift in P(Right) relative to marginal
# conditional_matrix: (8, 8) — absolute P(Right) per condition
```

Pooling across sessions:

```python
from behav_utils.analysis.update_matrix import compute_update_matrix_from_sessions

um = compute_update_matrix_from_sessions(sessions, method='pool')
```

## Session Selection

Preset filters for common session subsets:

```python
from behav_utils.data.selection import select_sessions

expert_sessions = select_sessions(animal, 'expert_uniform')
all_task_sessions = select_sessions(animal, 'all_task')
```

Custom filters via `SessionFilter` frozen dataclass.

## Synthetic Data

Generate synthetic animals and sessions for validation:

```python
from behav_utils import generate_synthetic_animal, sample_stimuli

# With a custom simulator
def my_simulator(stimuli, categories, rng):
    # Your model here
    return choices

animal, sessions = generate_synthetic_animal(
    animal_id='SYN01',
    n_sessions=20,
    trials_per_session=350,
    simulator=my_simulator,
)
```

## Plotting

```python
# Session-level
session.plot_psychometric(ax=ax)

# Animal-level
animal.plot_trajectory('accuracy')
animal.plot_psychometric()  # pooled across sessions

# Experiment-level
experiment.plot_trajectory('accuracy', combine='mean_sem')

# Update matrices
from behav_utils.plotting.update_matrix import plot_phase_update_matrices
plot_phase_update_matrices({'expert': um_expert, 'post_shift': um_post})

# Direct function access
from behav_utils.plotting.psychometric import plot_psychometric_curve
from behav_utils.plotting.trajectory import plot_stat_trajectory
```

## Package Structure

```
behav_utils/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── schema.py           # YAML config loading and validation
├── data/
│   ├── __init__.py
│   ├── structures.py        # TrialData, SessionData, AnimalData, FittingData
│   ├── loading.py           # CSV → data classes
│   ├── selection.py         # SessionFilter, select_sessions
│   ├── synthetic.py         # Synthetic data generation
│   └── neural.py            # NeuralData, Epoch (stub for imaging data)
├── analysis/
│   ├── __init__.py
│   ├── summary_stats.py     # Stat registry + 20+ registered stats
│   ├── update_matrix.py     # Update matrix computation
│   ├── psychometry.py       # Psychometric curve fitting
│   ├── session_features.py  # Feature matrix construction
│   └── utils.py             # Shared helpers
└── plotting/
    ├── __init__.py
    ├── psychometric.py      # Psychometric curve plots
    ├── update_matrix.py     # Update matrix heatmaps
    ├── session.py           # Per-session visualisations
    ├── trajectory.py        # Stat trajectory plots
    └── styles.py            # Colours, style constants
```

## Scope

`behav_utils` deliberately excludes:

- Computational models (BE, SC, or any other model)
- Simulation-based inference (SBI, SNPE, posterior estimation)
- Adaptation/learning analysis
- SLDS or HMM state inference
- Optogenetic analysis
- Any project-specific code

These belong in the project repository that imports `behav_utils`.

## Dependencies

- numpy
- scipy
- pandas
- matplotlib
- scikit-learn
- pyyaml

## Roadmap

- [ ] PyPI release (`pip install behav_utils`)
- [ ] Full test suite with pytest
- [ ] Formal documentation (Sphinx or MkDocs)
- [ ] Type annotations throughout
- [ ] Neural data integration (calcium imaging, spike data)
- [ ] Multi-animal batch analysis utilities

## Licence

TBD

## Citation

If you use `behav_utils` in published work, please cite this repository.
