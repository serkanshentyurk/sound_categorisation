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
├── inference/          # SBI (amortised; retrained per window as needed)
│   ├── amortised.py    #   - AmortisedSBI (pooled, manuscript path)
│   ├── simulator.py    #   - model→simulator wrappers + prior
│   ├── representation.py  # - sessions → summary-stat vector
│   ├── selection.py    #   - condition_sbi: held-out UM/CP per (animal, model)
│   ├── types.py        #   - ModelType, ParamConfig
│   └── constants.py    #   - SBI_STATS
│
├── analysis/           # Real-data analyses
│   ├── consensus.py    #   - cross-method BE/SC consensus vote
│   ├── grid_search.py  #   - GS-CV pipeline (manuscript protocol)
│   ├── phase.py        #   - phase / opto-phase assignment
│   └── adaptation.py   #   - distribution-shift detection
│
├── validation/         # Synthetic-data diagnostics
│   └── feature_diagnostics.py  # - which stats separate BE vs SC
│
├── utils/              # Math primitives + CV helpers
│   ├── stimulus_distributions.py  #   - Hard-A/B density + normative observer
│   ├── cv_utils.py                #   - neutral CV schema + BE-vs-SC comparator
│   └── fold_utils.py              #   - block-aware CV fold construction
│
├── plotting/           # Project-specific visualisations
│   ├── assignment.py              #   - animal × method assignment strip
│   ├── overview.py                #   - experiment timeline
│   ├── cv.py                      #   - CV summaries + confusion/recovery
│   ├── sbi_posterior.py           #   - posterior plots
│   └── sbi_validation.py          #   - SBC + recovery diagnostics
│
├── scripts/            # CLI entry points (being rebuilt; see notes below)
├── notebooks/          # Interactive analyses
├── tests/              # pytest suite (~100 tests)
├── config.yaml         # Column mapping, presets, masking sessions
└── smoke_test.py       # 10-test functional smoke check
```

## Two repos in one folder

- **`behav_utils/`** is a standalone library (general 2AFC analysis), pip-installable (`pip install -e behav_utils/`). Nothing project-specific lives here.
- **All other folders** are project code that builds on `behav_utils`.

## Three project-code folders

- **`analysis/`** — what to compute on real data
- **`validation/`** — synthetic-data testing of the pipeline
- **`utils/`** — shared math + CV bookkeeping

Each has a clear single job. Cross-folder dependencies are shallow and one-directional (project → behav_utils, never reverse).

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

| Submodule | Contents |
|---|---|
| `data/` | Loading, filtering, data classes (`ExperimentData → AnimalData → SessionData → TrialData`) |
| `analysis/` | psychometry, update matrix, summary stats (24-stat registry), comparison (incl. group-level), trajectory, session features |
| `plotting/` | psychometric, update matrix, trajectory, comparison |
| `config/` | YAML schema + loader |

### `models/`

| File | Contents |
|---|---|
| `BE_core.py` | `BEModel`, `BEParams`, `BEState` |
| `SC_core.py` | `SCModel`, `SCParams`, `SCState` |
| `perception.py` | Shared perceptual noise + boundary repulsion |
| `trace.py` | `ModelTrace` (per-trial state record) |

### `inference/`

Amortised SBI. A network is trained on simulated data (per representation and
model type) and conditioned on real or synthetic animals; networks are cheap to
retrain for any window length. There is no RandomWalk / GP / per-animal linking
machinery. Model identification is held-out UM/CP MSE: `condition_sbi` produces
the per-(animal, model) errors, and `utils.cv_utils.compare_models` makes the
BE-vs-SC call.

| File | Public API |
|---|---|
| `amortised.py` | `AmortisedSBI` — `train` / `save` / `load` / `condition` (pooled SBI, matches GS-CV) |
| `simulator.py` | `build_simulator`, `theta_to_params`, `get_param_names`, `get_bounds_arrays`, `wrap_for_sbi` |
| `representation.py` | `to_stat_vector` (sessions → summary-stat vector; pooled or moments mode) |
| `selection.py` | `condition_sbi` (held-out UM/CP per (animal, model); BE-vs-SC is `utils.cv_utils.compare_models`) |
| `types.py` | `ModelType`, `ParamConfig`, `get_default_param_configs` |
| `constants.py` | `SBI_STATS` (the ten heuristic summary stats) |

`from inference import AmortisedSBI, condition_sbi, ...` — see `inference/__init__.py` for the full export list.

### `analysis/`

| File | Public API |
|---|---|
| `consensus.py` | `load_all_assignments`, `compute_consensus_summary` (cross-method BE/SC consensus over the neutral CV schema) |
| `grid_search.py` | `compute_grid_search_cv`, `simulate_model_matrices`, `sessions_to_arrays`, `ParameterGrid`, `DEFAULT_GRID`, `COARSE_GRID` |
| `phase.py` | `compute_phase`, `filter_phase`, `is_opto_cohort`, `PANELS`, `PHASE_ORDER` |
| `adaptation.py` | `detect_shifts` |

### `validation/`

| File | Public API |
|---|---|
| `feature_diagnostics.py` | `compute_param_stat_correlations` (which summary stats separate BE vs SC) |

Synthetic-cohort generation lives in `scripts/validation/generate_synthetic_cohort.py` (`generate_cohort`, plus a CLI).

### `utils/`

| File | Public API |
|---|---|
| `stimulus_distributions.py` | `sample_distribution`, `compute_distribution_density`, `compute_normative_pse` |
| `cv_utils.py` | `compare_models`, `compute_seed_errors`, `save_cv_result`, `load_cv_results`, `params_to_str` (neutral CV schema + method-agnostic BE-vs-SC comparator) |
| `fold_utils.py` | `split_folds_by_block`, `merge_smallest_adjacent` |

### `plotting/`

| File | Public API |
|---|---|
| `assignment.py` | `plot_assignment_strip` |
| `overview.py` | `plot_timeline` |
| `cv.py` | `plot_cv_comparison`, `plot_winner_summary`, `plot_update_matrix`, `plot_um_comparison`, `plot_param_distributions`, `plot_confusion`, `plot_recovery` |
| `sbi_posterior.py` | `plot_marginal_posteriors`, `plot_pairplot`, `plot_posterior_psychometric` |
| `sbi_validation.py` | `plot_sbc_ranks`, `plot_sbc_ecdf`, `plot_recovery_scatter`, `plot_recovery_bias`, `plot_param_stat_correlations` |

For pair-comparison plots (opto, pre/post shift, HET vs WT), use `behav_utils.plotting.comparison.plot_comparison`.

---

## Notebooks

Organised by epistemic role, not topic: describe → validate → select → manipulate → characterise.

| Decade | Role | Notebooks |
|---|---|---|
| `0x` | Foundations & inspection (real data, descriptive) | `00` behaviour QC · `01` animal overview · `02` BE vs SC comparison |
| `1x` | Methods & validation (synthetic ground truth) | `10` feature extraction · `11` GS validation · `12` SBI validation · `13` HMM validation |
| `2x` | Static model selection (real data, single-state) | `20` GS results · `21` SBI conditioning · `22` SBI results · `23` cross-method consensus → BE/SC labels |
| `3x` | Optogenetics (Aim 2, causal) | `30` uniform · `31` hard · `32` session-by-session |
| `4x` | State-space models (HMM / LDS / SLDS) | `40` HMM · `41` LDS · `42` SLDS · `43` rSLDS · `44` compare SSM results |

`05` is the only notebook that is a standalone tool rather than a story step — it
sits in `0x` because its job is inspecting real data. `12` validates the SLDS +
per-state machinery that everything in `3x`/`4x` depends on. The consensus in
`20` produces the BE/SC labels every downstream notebook cites.

`notebooks/shared_setup.py` provides common paths and data loading used across
the story notebooks.

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

shifts = detect_shifts(animal)
for shift in shifts:
    pre  = [s for s in animal.sessions
            if s.session_idx <  shift['session_idx']
            and s.distribution == shift['from_distribution']]
    post = [s for s in animal.sessions
            if s.session_idx >= shift['session_idx']
            and s.distribution == shift['to_distribution']]
    result = compute_comparison(pre, post, label_a='pre', label_b='post')
```

### Group-level: HET vs WT (unpaired)

```python
from behav_utils.analysis.comparison import compute_per_animal_stats, compute_group_comparison

df_het = compute_per_animal_stats(het_animals)
df_wt  = compute_per_animal_stats(wt_animals)

result = compute_group_comparison(df_het, df_wt, label_a='HET', label_b='WT', paired=False)
# result['p_values']['mu'] = Mann-Whitney p for HET vs WT PSE
```

### Group-level: opto-on vs opto-off within HET (paired)

```python
df_on  = compute_per_animal_stats(het_animals, sessions_per_animal=sessions_on)
df_off = compute_per_animal_stats(het_animals, sessions_per_animal=sessions_off)
result = compute_group_comparison(df_on, df_off, label_a='opto_on', label_b='opto_off', paired=True)
# result['p_values']['mu'] = Wilcoxon p for within-animal opto effect
```

### Static SBI on expert data (manuscript path)

```python
from inference import AmortisedSBI, condition_sbi

# Train once on simulated data (per representation + model), then condition.
net = AmortisedSBI('be', N=15, T=350, mode='pooled')
net.train(n_simulations=50_000)
results = condition_sbi(expert_sessions, net, 'BE')   # held-out UM MSE per rep
# BE-vs-SC: utils.cv_utils.compare_models, or load_cv_results over a results dir
```

---

## Configuration

`config.yaml` defines:
- Column mappings (your CSV columns ↔ internal field names)
- Distribution name mappings (`Asym_Right → Hard-A`, `Asym_Left → Hard-B`)
- Session-filter presets (`naive`, `expert_uniform`, etc.)
- Masking sessions (per-animal list of dates)

See `behav_utils/configs/config_full_reference.yaml` for the full schema.

---

## Models

Two trial-to-trial choice updating models:

- **BE (Boundary-Estimation)** — maintains belief over the decision boundary; updates after each trial based on perceived stimulus and feedback. Parameters: `sigma_percep`, `A_repulsion`, `eta_learning`, `eta_relax`.
- **SC (Stimulus-Category)** — maintains beliefs over the full stimulus distribution per category; updates the chosen category's belief after each trial. Parameters: `sigma_percep`, `A_repulsion`, `gamma`, `sigma_update`.

Likelihood intractable for both → SBI (no MLE).

---

## Naming conventions

- `compute_X(data, ...)` → returns result dict
- `plot_X(result, ax=...)` → consumes result dict, returns `Axes`
- `fit_X(arrays, ...)` → low-level fit, returns param dict
- Psychometric fit params use math names everywhere: `mu`, `sigma`, `lapse_low`, `lapse_high`, `accuracy`. Plot labels translate to literature terms (PSE, slope) at display time.

For across-animal claims: use `compute_per_animal_stats` + `compute_group_comparison`, NOT pooled `compute_comparison`. Pooled trials assume trial-level independence — wrong for group-level claims.

---

## Tests and smoke check

```bash
pytest tests/                     # full pytest suite (~100 tests)
python3 smoke_test.py             # 10-test functional check
```

The smoke test exercises a representative subset of the pipeline end-to-end in seconds, useful for sanity-checking after any change.

---

## What's working, what's coming

**Production-ready now:**
- Behavioural analyses (psychometry, update matrix, summary stats, trajectory)
- Group comparisons (pair-level and animal-level)
- Static SBI for BE vs SC selection (manuscript path)
- Adaptation analysis (shift detection)
- Opto phase assignment + the `05` opto QC visualiser

**On the horizon:**
- Results notebooks `20`–`61` (model selection, SLDS, per-state trajectories, adaptation, opto) — see the Notebooks table.
- SLDS + per-state validation (`12`) — gates `3x`/`4x`.
- Scripts / SLURM rebuild (CLI entry points for cluster jobs).

---

## Cluster

`scripts/` is being rebuilt. Currently only `config.py`, `snapshot.py`, `export_snapshot.py` remain as live infrastructure. CLI entry points for SBI training, GS-CV, etc. will be rebuilt when cluster runs resume.

`slurm/` was removed during cleanup and will be rebuilt alongside the new scripts.
