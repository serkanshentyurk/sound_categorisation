# Sound Categorisation

PhD project — Akrami Lab, UCL. Investigating the dynamic causal necessity of posterior parietal cortex (PPC) during statistical model updating in mice performing a 2-AFC auditory categorisation task.

## The hypothesis

PPC is causally necessary when an animal's internal statistical model of the stimulus distribution is inadequate, and becomes dispensable once that model is sufficient. Dispensability is a natural consequence of model adequacy, not a discrete mode switch.

---

## Repository layout

```
sound_categorisation/
├── behav_utils/        # General-purpose 2AFC analysis library
│
├── models/             # BE and SC computational models
│
├── inference/          # SBI (amortised + dynamic per-animal)
│   └── constants.py    # SBI_STATS and similar config
│
├── analysis/           # Real-data analyses
├── validation/         # Synthetic-data testing of the pipeline
├── utils/              # Math primitives + CV helpers
│
├── plotting/           # Project-specific visualisations
│
├── scripts/            # CLI entry points
├── slurm/              # Cluster job templates
├── notebooks/          # Interactive analysis
├── tests/              # pytest suite
└── config.yaml         # Column mapping, presets, masking sessions
```

## Two repos in one folder

- **`behav_utils/`** is a standalone library (general 2AFC analysis). Eventually pip-installable; currently used via direct import. Nothing project-specific lives here.
- **All other folders** are project code that builds on `behav_utils`.

## Three project-code folders

- **`analysis/`** — what to compute on real data (consensus, grid_search, opto, adaptation)
- **`validation/`** — synthetic-data testing of the pipeline (cohorts, model_id, sbi-validation)
- **`utils/`** — shared math + CV bookkeeping (stimulus_distribution, cv_utils, fold_utils)

## Workflow philosophy

Every analysis follows the same four steps:

```python
data       = load_experiment(config)                       # LOAD
filtered   = select_sessions(animal, preset='...')         # FILTER
result     = compute_X(filtered, ...)                      # COMPUTE
fig        = plot_X(result, ...)                           # PLOT
```

Each step is visible at the call site. No function does two pipeline steps in one call. Filtering, grouping, looping → notebook. Math → modules.

---

## Modules

### `behav_utils/`

| Module | Contents |
|---|---|
| `data/` | Loading, filtering, data classes (`ExperimentData → AnimalData → SessionData → TrialData`) |
| `analysis/` | psychometry, update matrix, summary stats (24-stat registry), comparison (incl. group-level), trajectory, session features |
| `plotting/` | psychometric, update matrix, trajectory, comparison |
| `config/` | YAML schema + loader |

### `models/`

| Module | Class / Function | Purpose |
|---|---|---|
| `BE_core.py` | `BEModel`, `BEParams`, `BEState`, `ModelTrace` | Boundary Estimation. Params: `sigma_percep`, `A_repulsion`, `eta_learning`, `eta_relax` |
| `SC_core.py` | `SCModel`, `SCParams`, `SCState` | Stimulus Category. Params: `sigma_percep`, `A_repulsion`, `gamma`, `sigma_update` |
| `perception.py` | `perceive_stimulus` | Shared perceptual noise model |

### `inference/`

| Module | Class / Function | Purpose |
|---|---|---|
| `amortised.py` | `AmortisedSBI` | Train once on curriculum, condition on many animals. Static BE-vs-SC selection. |
| `fitting.py` | `SBIFitter` | Per-animal training. Time-varying parameters via `ConstantSpec`, `GPSpec`, `RandomWalkSpec`. |
| `comparison.py` | `compute_cv_comparison`, `compute_model_comparison` | CV-based model selection |
| `types.py` | `ConstantSpec`, `GPSpec`, `RandomWalkSpec`, `ThetaLayout` | Specs for parameter linking |
| `priors.py` | `MultiSessionPrior`, link classes | Prior construction |
| `simulator.py` | `create_be_simulator`, `create_sc_simulator` | Wrap models for SBI |
| `constants.py` | `SBI_STATS` etc. | Shared constants (no CLI dependency) |

### `analysis/`

| Module | Function | Purpose |
|---|---|---|
| `consensus.py` | `compute_consensus_summary`, `load_all_assignments` | 4-method model selection vote (GS-UM, GS-CP, SBI-UM, SBI-CP) |
| `grid_search.py` | `compute_grid_search_cv`, `compute_sessions_blocked`, `compute_static_vs_dynamic`, `simulate_model_matrices`, `ParameterGrid`, `DEFAULT_GRID`, `COARSE_GRID` | GS-CV pipeline |
| `opto.py` | `assign_opto_phases` | Partition opto-experiment sessions into named phases |
| `adaptation.py` | `detect_shifts` | Find distribution-shift boundaries |

### `validation/`

| Module | Function | Purpose |
|---|---|---|
| `cohorts.py` | `make_synthetic_cohort`, `make_learning_cohort`, `generate_session_with_distribution` | Synthetic cohort generators |
| `model_id.py` | `run_gs_model_id` | GS-CV identification on synthetic data |
| `sbi.py` | `compute_sbc_ranks`, `compute_parameter_recovery`, `compute_param_stat_correlations` | SBI calibration + recovery |

### `utils/`

| Module | Function | Purpose |
|---|---|---|
| `stimulus_distribution.py` | `sample_distribution`, `compute_distribution_density`, `compute_normative_pse` | Hard-A/B density + ideal observer |
| `cv_utils.py` | `compute_empirical_um`, `simulate_model_um`, `compute_gs_seed_errors`, `compute_cv_dataframes` | UM CV helpers |
| `fold_utils.py` | `split_folds_by_block`, `merge_smallest_adjacent` | Block-aware CV fold construction |

### `plotting/`

| Module | Function |
|---|---|
| `assignment.py` | `plot_assignment_strip` |
| `cv.py` | `plot_cv_comparison`, `plot_winner_summary`, `plot_update_matrix`, `plot_um_comparison`, `plot_param_distributions` |
| `sbi_posterior.py` | `plot_marginal_posteriors`, `plot_pairplot`, `plot_posterior_psychometric` |
| `sbi_trajectories.py` | `plot_parameter_trajectories`, `plot_performance_trajectory`, `plot_learning_trajectory` |
| `sbi_validation.py` | `plot_sbc_ranks`, `plot_sbc_ecdf`, `plot_recovery_scatter`, `plot_recovery_bias`, `plot_param_stat_correlations` |

---

## Quick start

```python
from behav_utils.config import load_config
from behav_utils.data.loading import load_experiment
from behav_utils.data.selection import select_sessions
from behav_utils.analysis.psychometry import compute_psychometric
from behav_utils.plotting.psychometric import plot_psychometric

config = load_config('config.yaml')
experiment = load_experiment(config)
animal = experiment.animals['SS01']

expert_sessions = select_sessions(animal, preset='expert_uniform')
result = compute_psychometric(expert_sessions, mode='pooled')
plot_psychometric(result)
```

---

## Example workflows

### Opto: within-session opto-on vs opto-off (one animal)

```python
from behav_utils.data.filtering import filter_session, opto_mask
from behav_utils.analysis.comparison import compute_comparison
from behav_utils.plotting.comparison import plot_comparison
from analysis.opto import assign_opto_phases

phases = assign_opto_phases(animal.sessions)
sessions = phases['shift_with_opto']

on  = [filter_session(s, opto_mask(s.trials, 0))         for s in sessions]
off = [filter_session(s, opto_mask(s.trials, 'control')) for s in sessions]

result = compute_comparison(on, off, label_a='opto_on', label_b='opto_off',
                              n_bootstrap=1000, n_permutations=1000)
plot_comparison(result)
```

### Adaptation: pre-shift vs post-shift comparison

```python
from analysis.adaptation import detect_shifts
from behav_utils.analysis.comparison import compute_comparison
from behav_utils.plotting.comparison import plot_comparison

shifts = detect_shifts(animal)
for shift in shifts:
    pre  = [s for s in animal.sessions
            if s.session_idx <  shift['session_idx']
            and s.distribution == shift['from_distribution']]
    post = [s for s in animal.sessions
            if s.session_idx >= shift['session_idx']
            and s.distribution == shift['to_distribution']]
    result = compute_comparison(pre, post, label_a='pre', label_b='post',
                                  n_bootstrap=1000)
    plot_comparison(result)
```

### Group-level: HET vs WT (unpaired, across animals)

```python
from behav_utils.analysis.comparison import (
    compute_per_animal_stats, compute_group_comparison,
)
from analysis.opto import assign_opto_phases

het = [a for a in animals if a.genotype == 'HET']
wt  = [a for a in animals if a.genotype == 'WT']

het_shift_sessions = {a.animal_id: assign_opto_phases(a.sessions)['shift_with_opto']
                       for a in het}
wt_shift_sessions  = {a.animal_id: assign_opto_phases(a.sessions)['shift_with_opto']
                       for a in wt}

df_het = compute_per_animal_stats(het, het_shift_sessions)
df_wt  = compute_per_animal_stats(wt,  wt_shift_sessions)

result = compute_group_comparison(df_het, df_wt,
                                    label_a='HET', label_b='WT',
                                    paired=False)
# result['p_values']['mu'] = Mann-Whitney p for HET vs WT PSE
```

### Group-level: opto-on vs opto-off within HET (paired, same animals)

```python
sessions_on  = {a.animal_id: [filter_session(s, opto_mask(s.trials, 0))
                              for s in assign_opto_phases(a.sessions)['shift_with_opto']]
                for a in het}
sessions_off = {a.animal_id: [filter_session(s, opto_mask(s.trials, 'control'))
                              for s in assign_opto_phases(a.sessions)['shift_with_opto']]
                for a in het}

df_on  = compute_per_animal_stats(het, sessions_on)
df_off = compute_per_animal_stats(het, sessions_off)

result = compute_group_comparison(df_on, df_off,
                                    label_a='opto_on', label_b='opto_off',
                                    paired=True)
# result['p_values']['mu'] = Wilcoxon p for within-animal opto effect
```

### Static SBI on expert data

```python
from inference import AmortisedSBI
from behav_utils.data.selection import select_sessions

expert = select_sessions(animal, preset='expert_uniform')

sbi = AmortisedSBI(model_type='be', curriculum=[('uniform', 2500)])
sbi.train(n_simulations=50_000)
posterior = sbi.fit(expert)
```

### Per-session parameter trajectory (naïve → expert)

```python
import pandas as pd

trajectory = []
for sess in animal.sessions:
    posterior = sbi.fit([sess])
    samples = posterior.sample((1000,))
    trajectory.append({
        'session_idx': sess.session_idx,
        **{name: float(samples[:, i].median())
            for i, name in enumerate(sbi.param_names)}
    })
trajectory_df = pd.DataFrame(trajectory)
```

### Dynamic SBI with parameter linking

```python
from inference import SBIFitter, RandomWalkSpec, ConstantSpec

fitter = SBIFitter(
    fitting_data=animal.fitting_data(),
    model_type='be',
    param_links={
        'eta_learning': RandomWalkSpec(bounds=(0.0, 1.0)),
        'sigma_percep': ConstantSpec(bounds=(0.05, 0.5)),
        'A_repulsion':  ConstantSpec(bounds=(0.0, 1.0)),
        'eta_relax':    RandomWalkSpec(bounds=(0.0, 1.0)),
    },
)
fitter.train(n_simulations=50_000)
result = fitter.fit()
trajectory = fitter.extract_trajectories(result)
```

### SBI validation

```python
from validation.sbi import compute_sbc_ranks, compute_parameter_recovery
from plotting.sbi_validation import plot_sbc_ranks, plot_recovery_scatter

sbc = compute_sbc_ranks(posterior, simulator, prior,
                         n_sbc_runs=1000, n_posterior_samples=1000)
plot_sbc_ranks(sbc)

recovery = compute_parameter_recovery(posterior, simulator, prior,
                                        n_recoveries=200)
plot_recovery_scatter(recovery)
```

---

## Configuration

`config.yaml` defines:
- Column mappings (your CSV columns ↔ internal field names)
- Distribution name mappings (e.g. `Asym_Right → Hard-A`)
- Session-filter presets (`naive`, `expert_uniform`, etc.)
- Masking sessions (per-animal list of dates)

See `behav_utils/configs/config_full_reference.yaml` for the full schema.

---

## Models

Two trial-to-trial choice updating models:

- **BE (Boundary-Estimation)** — maintains belief over the decision boundary; updates after each trial based on perceived stimulus and feedback. Four parameters.
- **SC (Stimulus-Category)** — maintains beliefs over the full stimulus distribution per category; updates the chosen category's belief after each trial. Four parameters.

Likelihood intractable for both → SBI (no MLE).

---

## Naming conventions

- `compute_X(data, ...)` → returns result dict
- `plot_X(result, ax=...)` → consumes dict, returns `Axes`
- `fit_X(arrays, ...)` → low-level fit, returns param dict
- Psychometric fit params: `mu`, `sigma`, `lapse_low`, `lapse_high`, `accuracy` everywhere. Plot labels translate to literature terms (PSE, slope).

For across-animal claims: use `compute_per_animal_stats` + `compute_group_comparison`, not pooled `compute_comparison`. Pooled trials assume trial-level independence — wrong for group-level claims.

---

## Cluster

SLURM templates at `slurm/`. CLI entry points in `scripts/`. Being restructured separately — expect manual editing of cluster paths until that's done.
