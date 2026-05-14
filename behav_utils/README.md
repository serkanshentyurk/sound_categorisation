# behav_utils

Config-driven Python library for loading, analysing, and plotting trial-based behavioural data from head-fixed rodent experiments.

## Quick Start

```python
from behav_utils import load_experiment, select_sessions

# Load from YAML config
experiment = load_experiment('config.yaml')
animal = experiment.get_animal('SS05')

# Filter sessions
expert = select_sessions(animal, preset='expert_uniform')

# Summary stats
expert[0].stats(['accuracy', 'pse', 'slope', 'recency'])
# → {'accuracy': 0.82, 'pse': 0.03, 'slope': 0.15, 'recency': 0.12}

# Trajectory across sessions
animal.stat_trajectory('accuracy')

# Compare groups
from behav_utils.plotting.psychometric import plot_psychometric_overlay
plot_psychometric_overlay({
    'Early': animal.sessions[:5],
    'Late':  animal.sessions[-5:],
})
```

## Data Classes

```
ExperimentData                   # All animals
  └── AnimalData                 # One animal, list of sessions
        └── SessionData          # One session
              ├── SessionMetadata # Dict-like: animal_id, stage, date, ...
              └── TrialData      # Arrays: stimulus, choice, correct, ...

FittingData                      # Flat arrays pooled from sessions
```

### TrialData arrays

| Required | Optional |
|:---------|:---------|
| `stimulus`, `choice`, `correct`, `outcome`, `trial_number` | `reaction_time`, `abort`, `opto_on`, `category`, `distribution` |

### Key methods

| Object | Method | Returns |
|:-------|:-------|:--------|
| `SessionData` | `.stats(stat_names, exclude_opto=True)` | Dict of stat values |
| `SessionData` | `.plot_psychometric(ax=)` | Figure |
| `SessionData` | `.plot_trials()` | Figure |
| `AnimalData` | `.get_sessions(stage=, distribution=)` | Filtered session list |
| `AnimalData` | `.stat_trajectory(stat_name)` | Array of values across sessions |
| `AnimalData` | `.plot_trajectory(stat_name)` | Figure |
| `AnimalData` | `.plot_psychometric()` | Pooled psychometric |
| `ExperimentData` | `.get_animal(animal_id)` | AnimalData |
| `ExperimentData` | `.plot_trajectory(stat_name, combine='mean_sem')` | Figure |

---

## Module Reference

### `behav_utils.config.schema`

Config loading and validation.

| Function | Purpose |
|:---------|:--------|
| `load_config(path)` | Parse YAML → `ProjectConfig` |
| `validate_csv_against_config(df, config)` | Check CSV matches config |

### `behav_utils.data.loading`

CSV loading pipeline.

| Function | Purpose |
|:---------|:--------|
| `load_experiment(config_or_path)` | Load all animals → `ExperimentData` |
| `load_animal(animal_dir, config)` | Load one animal → `AnimalData` |
| `load_session_csv(path, config)` | Load one CSV → `SessionData` |
| `load_from_directory(data_dir, config)` | Auto-discover and load all |
| `convert_choice_to_category(...)` | Apply spatial→category mapping |
| `parse_date_from_path(path, config)` | Extract session date |

### `behav_utils.data.selection`

Session filtering with presets.

| Function | Purpose |
|:---------|:--------|
| `select_sessions(animal, preset=, **overrides)` | Filter sessions by preset or custom criteria |
| `fitting_data_from_sessions(sessions)` | Pool sessions → `FittingData` |
| `register_preset(name, **kwargs)` | Register a new preset |
| `register_presets_from_config(config)` | Load presets from YAML |
| `list_presets()` | Show available presets |
| `get_preset(name)` | Get preset's `SessionFilter` |

`SessionFilter` fields: `stage`, `distribution`, `min_accuracy`, `last_fraction`, `first_n`, `after_session_idx`.

### `behav_utils.data.synthetic`

Synthetic data generation for validation.

| Function | Purpose |
|:---------|:--------|
| `generate_synthetic_animal(animal_id, n_sessions, simulator)` | Full synthetic animal |
| `generate_synthetic_session(n_trials, simulator)` | Single session |
| `sample_stimuli(n, distribution='uniform')` | Draw stimulus arrays |
| `noisy_psychometric_simulator(mu, sigma, lapse_low, lapse_high)` | Returns a simulator function |

### `behav_utils.analysis.psychometry`

Psychometric curve fitting (cumulative Gaussian, 4-param: μ, σ, lapse_low, lapse_high).

| Function | Purpose |
|:---------|:--------|
| `fit_psychometric(stimuli, choices)` | Fit → dict with `mu`, `sigma`, `lapse_low`, `lapse_high`, `success` |
| `compute_psychometric_gof(stimuli, choices)` | Goodness-of-fit (deviance) |
| `compute_psych_error(stimuli, choices, n_bins)` | Binned residuals |

### `behav_utils.analysis.summary_stats`

Registry of 25+ summary statistics. All available via `session.stats()`.

| Stat name | Returns |
|:----------|:--------|
| `accuracy` | Overall fraction correct |
| `psychometric` | `pse`, `slope`, `lapse_low`, `lapse_high` |
| `conditional_psychometric` | Per-previous-stimulus psychometric |
| `psychometric_gof` | Deviance goodness-of-fit |
| `recency` | Recency index (bias toward recent stimuli) |
| `stimulus_recency` | Stimulus-specific recency |
| `recency_divergence` | KL divergence of recency profiles |
| `history_interaction_r2` | R² of stimulus × history interaction |
| `win_stay` | P(same choice \| previous correct) |
| `lose_shift` | P(different choice \| previous wrong) |
| `choice_autocorr` | Choice autocorrelation (1-back) |
| `perseveration` | Perseveration index |
| `choice_entropy` | Shannon entropy of choice sequence |
| `side_bias` | Proportion choosing one side |
| `stimulus_sensitivity` | Logistic regression weight on stimulus |
| `hard_accuracy`, `easy_accuracy` | Accuracy for hard/easy trials |
| `hard_easy_ratio` | hard_accuracy / easy_accuracy |
| `update_matrix` | 8×8 conditional update matrix |
| `logistic_history` | GLM weights for recent stimuli/choices |
| `sd_profile` | Stimulus-dependent profile features |
| `binned_accuracy` | Per-bin accuracy vector |
| `binned_choice_prob` | Per-bin P(choose B) vector |

**Adding a custom stat:**
```python
from behav_utils.analysis.summary_stats import register_stat

@register_stat('my_metric')
def compute_my_metric(choices, stimuli, categories, **kwargs):
    return {'my_value': float(np.mean(choices))}
```

| Utility | Purpose |
|:--------|:--------|
| `compute_summary_stats(session, stat_names)` | Compute stats for a session |
| `compute_summary_stats_per_session(sessions, stat_names)` | Batch compute |
| `compute_stats_for_sbi(sessions, stat_names)` | Flat array for SBI |
| `list_available_stats()` | Show registered stat names |
| `describe_stats()` | Show stats with descriptions |

### `behav_utils.analysis.update_matrix`

Update matrix: how previous stimulus influences current choice.

| Function | Purpose |
|:---------|:--------|
| `compute_update_matrix(stimuli, choices, categories, n_bins=8)` | → `(um, conditional, info)` |
| `compute_update_matrix_from_sessions(sessions, method='pool')` | Pool sessions → single UM |
| `matrix_error(um_a, um_b)` | Element-wise RMSE between two UMs |

### `behav_utils.analysis.comparison`

General-purpose two-condition comparison (permutation tests, bootstrap CIs, Fisher's exact).

| Function | Purpose |
|:---------|:--------|
| `compare_conditions(stim_a, ch_a, cat_a, stim_b, ch_b, cat_b)` | Full comparison → dict with diffs, p-values, CIs, UM metrics |
| `permutation_test_params(stim_a, ch_a, stim_b, ch_b, n_perm=1000)` | Permutation p-values for psychometric params |
| `bootstrap_param_diff(stim_a, ch_a, stim_b, ch_b, n_boot=1000)` | Bootstrap CIs on param differences |

### `behav_utils.analysis.session_features`

Feature matrix construction for dimensionality reduction.

| Function | Purpose |
|:---------|:--------|
| `compute_session_features(session, stat_names)` | Feature vector for one session |
| `build_feature_matrix(sessions, stat_names)` | DataFrame: sessions × features |
| `build_feature_matrix_multi(animals, stat_names)` | Multi-animal feature matrix |
| `zscore_features(df)` | Z-score normalisation |

### `behav_utils.analysis.utils`

Shared helpers.

| Function | Purpose |
|:---------|:--------|
| `cumulative_gaussian(x, mu, sigma, lapse_low, lapse_high)` | Psychometric function |
| `generate_stimuli(n, distribution)` | Stimulus array |

---

## Plotting Reference

### `behav_utils.plotting.psychometric`

| Function | Purpose |
|:---------|:--------|
| `plot_psychometric(session, ax=)` | Single session psychometric |
| `plot_session_psychometrics(sessions, mode='overlay'\|'grid')` | Multiple sessions |
| `plot_psychometric_compare(session_groups, mode='session_mean')` | Side-by-side subplots per group |
| `plot_psychometric_overlay(session_groups, mode='pooled')` | All groups on one axes |

**`session_groups`** is always `Dict[str, List[SessionData]]` — labels map to session lists.

### `behav_utils.plotting.trajectory`

| Function | Purpose |
|:---------|:--------|
| `plot_stat_trajectory(animal, stat_name)` | One animal, one stat across sessions |
| `plot_multi_animal_trajectory(animals, stat_name, combine='mean_sem')` | Cohort trajectory |
| `plot_stat_grid(animals, stat_names)` | Grid: animals × stats |

### `behav_utils.plotting.update_matrix`

| Function | Purpose |
|:---------|:--------|
| `plot_update_matrix(um, ax=)` | Single UM heatmap |
| `plot_update_matrix_comparison(um_a, um_b, diff=True)` | Side-by-side + difference |
| `plot_phase_update_matrices(phase_dict)` | Named phases side-by-side |
| `plot_sd_profile(session)` | Stimulus-dependent profile |
| `plot_conditional_psychometrics(session)` | Per-previous-stimulus curves |

### `behav_utils.plotting.session`

| Function | Purpose |
|:---------|:--------|
| `plot_session_trials(session)` | Trial-by-trial raster |
| `plot_session_comparison(sess_a, sess_b)` | Two sessions side-by-side |

### `behav_utils.plotting.styles`

| Function / Constant | Purpose |
|:---------------------|:--------|
| `apply_style()` | Apply default matplotlib style |
| `COLOURS` | Dict of standard colours |
| `UM_CMAP` | Diverging colourmap for update matrices |
| `get_animal_colours(animal_ids)` | Consistent per-animal colours |

---

## Session Selection Presets

Presets are registered from `config.yaml` via `register_presets_from_config()`.

| Preset | Filters |
|:-------|:--------|
| `expert_uniform` | stage=Full_Task_Cont, distribution=Uniform, min_accuracy=0.70, last 50% |
| `all_uniform` | stage=Full_Task_Cont, distribution=Uniform |
| `naive_uniform` | stage=Full_Task_Cont, distribution=Uniform, first 5 |
| `all_hard_a` | stage=Full_Task_Cont, distribution=Hard-A |
| `expert_hard_a` | stage=Full_Task_Cont, distribution=Hard-A, min_accuracy=0.60, last 50% |
| `all_full_task` | stage=Full_Task_Cont |
| `all_stages` | No filter |

Custom:
```python
from behav_utils.data.selection import SessionFilter
my_filter = SessionFilter(stage='Full_Task_Cont', min_accuracy=0.65, first_n=10)
sessions = my_filter.apply(animal)
```

---

## Package Structure

```
behav_utils/
├── config/
│   └── schema.py            # YAML config loading, ProjectConfig dataclass
├── data/
│   ├── structures.py        # TrialData, SessionData, AnimalData, ExperimentData
│   ├── loading.py           # CSV → data classes
│   ├── selection.py         # SessionFilter, presets
│   ├── synthetic.py         # Synthetic data generation
│   └── neural.py            # NeuralData, Epoch (imaging stub)
├── analysis/
│   ├── summary_stats.py     # 25+ registered stats
│   ├── update_matrix.py     # Update matrix computation
│   ├── psychometry.py       # Psychometric curve fitting
│   ├── comparison.py        # Two-condition comparison + permutation tests
│   ├── session_features.py  # Feature matrix construction
│   └── utils.py             # Shared helpers
└── plotting/
    ├── psychometric.py      # Psychometric curves (single, compare, overlay)
    ├── update_matrix.py     # UM heatmaps, comparisons, profiles
    ├── trajectory.py        # Stat trajectories (single, multi-animal, grid)
    ├── session.py           # Per-session trial rasters
    └── styles.py            # Colours, style constants
```