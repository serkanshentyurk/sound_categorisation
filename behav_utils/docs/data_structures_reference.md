# Data Structures Reference

## Overview

behav_utils organises behavioural data in a hierarchy that mirrors experimental structure:

```
ExperimentData                    All animals in a project
  └── AnimalData                  One animal, all its sessions
        └── SessionData           One behavioural session
              ├── SessionMetadata  Task parameters (constant within session)
              └── TrialData        Trial-by-trial arrays
```

Separately, `FittingData` provides a flat per-session array format for SBI inference.

---

## TrialData

The lowest level — per-trial arrays for a single session.

### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `trial_number` | int array | Original trial numbers from CSV |
| `stimulus` | float array | Stimulus values |
| `choice` | float array | Category-space choice: 0=A, 1=B, NaN=no response |
| `choice_raw` | float array | Raw choice from CSV (before conversion) |
| `outcome` | str array | Trial outcome ('Correct', 'Incorrect', 'Abort') |
| `correct` | bool array | Whether choice matched category |
| `category` | int array | Derived from stimulus: 0=A, 1=B |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `reaction_time` | float array | Response latency (NaN for aborts) |
| `abort` | bool array | Whether animal broke fixation |
| `opto_on` | bool array | Whether optogenetics was active |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `.n_trials` | int | Total trial count |
| `.no_response` | bool array | True where `choice` is NaN |
| `.valid_mask` | bool array | `~abort & ~no_response` |

### Key Methods

All are thin wrappers to `behav_utils.data.filtering`.

#### `get_arrays()`

Extract analysis-ready arrays. **No kwargs** — always excludes aborts (invalid data), returns everything else. All scientific filtering (opto, no-response, custom) happens upstream via `filter_trials()`.

```python
arrays = session.trials.get_arrays()
```

Returns dict:

| Key | Description |
|-----|-------------|
| `'stimuli'` | Stimulus values |
| `'categories'` | True categories 0/1 |
| `'choices'` | Choices 0/1/NaN |
| `'no_response'` | Boolean mask for NaN choices |
| `'reaction_times'` | RT values |
| `'trial_indices'` | Original indices for back-mapping |

#### `build_mask(exclude_abort=True, exclude_opto=True)`

Build a boolean trial mask. Wrapper to `filtering.build_mask()`.

#### `opto_mask(delta=0)`

Boolean mask for trials relative to opto events. Wrapper to `filtering.opto_mask()`.

- `delta=0` — opto trials themselves
- `delta=1` — first trial after each opto trial
- `delta=-1` — trial before each opto trial
- `delta='control'` — non-opto trials (not adjacent to opto)

#### `filter(mask, label='custom')`

Return a new TrialData with only the trials where `mask` is True. Wrapper to `filtering.filter_trial_data()`.

#### `stats(stat_names)`

Compute summary statistics on these trials. **No filtering kwargs** — filter first, then call stats.

```python
# Correct pattern:
clean_session = session.filter()
clean_session.stats(['accuracy', 'pse', 'recency'])
```

---

## SessionData

One behavioural session — metadata + trial data.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | str | Unique identifier |
| `session_idx` | int | Ordinal position within animal (0-based) |
| `date` | datetime.date | Session date |
| `metadata` | SessionMetadata | Task parameters |
| `trials` | TrialData | Trial-by-trial arrays |
| `masking` | bool | Whether this is a masking (sham) session |
| `filter_info` | dict or None | Metadata about filtering applied |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `.n_trials` | int | Total trials (reflects filtering) |
| `.stage` | str | Training stage (from metadata) |
| `.distribution` | str | Stimulus distribution |
| `.is_filtered` | bool | Whether filtering has been applied |

### Key Methods

```python
# Filter trials
clean = session.filter()                              # standard: exclude abort + opto
opto = session.filter(session.trials.opto_mask(0))    # opto trials only

# Get arrays from (pre-filtered) session
arrays = session.get_arrays()

# Stats (filter first!)
clean.stats(['accuracy', 'pse', 'slope'])
```

---

## SessionMetadata

Task parameters constant within a session. Access by attribute or `.get()`:

```python
session.metadata.stage
session.metadata.get('sound_contingency')
session.metadata.fields  # raw dict
```

---

## AnimalData

One animal — all sessions in chronological order.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `animal_id` | str | Animal identifier |
| `sessions` | list[SessionData] | Chronologically ordered sessions |
| `metadata` | dict | Animal-level metadata (genotype, etc.) |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `.n_sessions` | int | Session count |
| `.session_ids` | list[str] | All session IDs |

### Convenience Plot Methods

These are thin wrappers that call `compute_` then `plot_` internally. For full control, call the functions directly.

```python
# Thin wrappers (quick exploration)
animal.plot_psychometric(ax=ax, mode='pooled')
animal.plot_trajectory('accuracy', ax=ax)
animal.plot_um(ax=ax)

# Full control (recommended)
from behav_utils import (
    select_sessions, filter_trials,
    compute_psychometric, plot_psychometric, PALETTE,
)
sessions = select_sessions(animal, preset='expert_uniform')
clean = filter_trials(sessions)
result = compute_psychometric(clean, mode='pooled', n_bootstrap=200)
fig, ax = plt.subplots()
plot_psychometric(result, ax=ax, color=PALETTE[0])
```

---

## ExperimentData

All animals in one project.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `animals` | dict[str, AnimalData] | animal_id → AnimalData |
| `config` | ProjectConfig or None | Loaded config |

### Methods

```python
experiment = load_experiment('config.yaml')
animal = experiment.get_animal('SS05')
all_animals = experiment.get_animals(min_sessions=10)
```

---

## FittingData

Flat per-session arrays for SBI inference. Built from pre-filtered sessions:

```python
from behav_utils import select_sessions, filter_trials, fitting_data_from_sessions

sessions = select_sessions(animal, preset='expert_uniform')
clean = filter_trials(sessions)
fd = fitting_data_from_sessions(clean, animal.animal_id)

# fd.stimuli       — list of arrays, one per session
# fd.choices       — list of arrays
# fd.n_sessions    — int
# fd.animal_id     — str
```

---

## Pipeline Pattern

Every notebook should follow this pattern:

```python
from behav_utils import (
    load_experiment, select_sessions, filter_trials,
    compute_psychometric, compute_um, compute_trajectory, compute_comparison,
    plot_psychometric, plot_um, plot_trajectory, plot_comparison,
    PALETTE, apply_style,
)
apply_style()

# 1. Load
experiment = load_experiment('config.yaml')
animal = experiment.get_animal('SS05')

# 2. Select sessions (session-level)
sessions = select_sessions(animal, preset='expert_uniform')

# 3. Filter trials (trial-level) — ALWAYS EXPLICIT
clean = filter_trials(sessions)

# 4. Analyse — returns result dicts
psych = compute_psychometric(clean, mode='pooled', n_bootstrap=200)
um = compute_um(clean)
traj = compute_trajectory(clean, ['accuracy', 'pse'])

# 5. Plot — draws result dicts
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_psychometric(psych, ax=axes[0], color=PALETTE[0])
plot_um(um, ax=axes[1])
plot_trajectory(traj, 'accuracy', ax=axes[2])
```

### Comparing two conditions

```python
ctrl = filter_trials(sessions)
opto = filter_trials(sessions, lambda s: s.trials.opto_mask(0))

# Option A: full statistical comparison
comp = compute_comparison(ctrl, opto, label_a='Control', label_b='Opto')
plot_comparison(comp, metric='psychometric')

# Option B: overlay individually
ctrl_psych = compute_psychometric(ctrl)
opto_psych = compute_psychometric(opto)
fig, ax = plt.subplots()
plot_psychometric(ctrl_psych, ax=ax, color=PALETTE[0], label='Control')
plot_psychometric(opto_psych, ax=ax, color=PALETTE[1], label='Opto')
ax.legend()
```

---

## Synthetic Data

Generate test data without real experiments:

```python
from behav_utils import generate_synthetic_animal
from behav_utils.data.synthetic import noisy_psychometric_simulator

# Built-in simulator: (stimuli, categories, rng, sigma, lapse) -> choices
animal, info = generate_synthetic_animal(
    animal_id='SYN01',
    n_sessions=20,
    trials_per_session=200,
    simulator=noisy_psychometric_simulator,
    simulator_kwargs={'sigma': 0.3, 'lapse': 0.05},
)

# Learning trajectory: parameters change across sessions
animal, info = generate_synthetic_animal(
    animal_id='LEARN01',
    n_sessions=20,
    simulator=noisy_psychometric_simulator,
    per_session_simulator_kwargs=[
        {'sigma': 0.8 - i * 0.03, 'lapse': max(0.01, 0.15 - i * 0.006)}
        for i in range(20)
    ],
)

# Custom simulator: any (stimuli, categories, rng, **kwargs) -> choices callable
def my_simulator(stimuli, categories, rng, **kwargs):
    return choices

animal, info = generate_synthetic_animal(
    simulator=my_simulator,
    simulator_kwargs={'my_param': 0.5},
)
```
