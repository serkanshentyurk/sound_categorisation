# Sound Categorisation

Computational modelling and analysis pipeline for investigating the dynamic necessity of posterior parietal cortex (PPC) during statistical model updating in mice.

## The Question

Is PPC causally necessary when an animal's internal statistical model of the stimulus distribution is inadequate, and does it become dispensable once that model is sufficient?

## Repository Structure

```
sound_categorisation/
├── behav_utils/          # Cross-lab library (see behav_utils/README.md)
├── models/               # BE and SC computational models
├── analysis/             # Project-specific analysis
├── inference/            # Simulation-based inference (SBI/SNPE)
├── plotting/             # Project-specific plotting
├── legacy/               # Old codebase (still imported by analysis/cv_utils.py)
├── notebooks/            # Analysis notebooks (00–99)
│   ├── shared_setup.py   # Notebook import boilerplate
│   └── dev/              # Development/validation notebooks
├── scripts/              # Cluster and local entry points
│   └── validation/       # Synthetic validation scripts
├── slurm/                # SLURM submission scripts
├── tests/                # pytest tests
├── config.yaml           # Data loading configuration
└── shared_setup.py       # Notebook import boilerplate
```

---

## Module Reference

### `models/` — Computational models

Two models of how mice update their internal representation:

| Module | Class / Function | Purpose |
|:-------|:-----------------|:--------|
| `BE_core.py` | `BEModel`, `BEParams`, `BEState` | Boundary Estimation: tracks decision boundary via exponential moving average. Params: η (learning rate), σ_n (noise) |
| `BE_model.py` | `BoundaryEstimationModel` | High-level wrapper for fitting |
| `SC_core.py` | `SCModel`, `SCParams`, `SCState` | Stimulus Category: tracks category distributions via KDE. Params: γ (category weight), σ_n (noise) |
| `SC_model.py` | `StimulusCategoryModel` | High-level wrapper for fitting |
| `perception.py` | `perceive_stimulus()` | Shared perceptual noise model |

### `analysis/` — Project-specific analysis

| Module | Key functions | Purpose |
|:-------|:-------------|:--------|
| `grid_search.py` | `grid_search_cv()`, `run_cv_both_models()`, `parameter_sweep()` | GS-CV model selection (2-fold × 64 seeds, 8×8 UM MSE) |
| `consensus.py` | `load_all_assignments()`, `consensus_summary()` | 4-method consensus (GS-UM, GS-CP, SBI-UM, SBI-CP) |
| `adaptation.py` | `detect_all_manipulations()`, `adaptation_trajectory()`, `fit_recovery_curve()`, `compare_phases()`, `build_phase_blocks()` | Post-shift behavioural analysis (15 functions) |
| `opto.py` | `opto_relative_mask(session, delta=)`, `within_session_effect()`, `phase_stability()`, `expert_null_test()`, `expert_um_test()`, `animal_opto_report()` | Optogenetic inactivation analysis |
| `cv_utils.py` | `load_cv_pickles()`, `build_long_df()`, `build_summary_table()` | Cross-validation result loading/formatting |
| `validation.py` | `make_synthetic_cohort()`, `make_learning_cohort()`, `make_shift_cohort()` | Synthetic data for pipeline validation |
| `stimulus_distribution.py` | `sample_distribution()`, `sample_hard_a()`, `sample_hard_b()` | Hard-A/B asymmetric distribution sampling |
| `slds.py` | `predict_state()`, `compute_bic()` | SLDS/HMM state inference utilities |
| `fold_utils.py` | `split_folds_by_block()` | Block-aware CV fold splitting |

### `inference/` — Simulation-based inference

| Module | Key functions / classes | Purpose |
|:-------|:----------------------|:--------|
| `fitting.py` | `SBIFitter`, `train_sbi()`, `quick_fit()`, `build_prior()`, `build_simulator()` | Main SBI interface: train SNPE, extract posteriors/trajectories |
| `priors.py` | `MultiSessionPrior`, `RandomWalkLink`, `GPLink`, `create_multisession_prior()` | Multi-session priors with parameter linking |
| `simulator.py` | `Simulator`, `create_be_simulator()`, `create_sc_simulator()`, `wrap_for_sbi()` | Model simulators compatible with `sbi` package |
| `types.py` | `ThetaLayout`, `ConstantSpec`, `RandomWalkSpec`, `GPSpec` | Parameter specification types |
| `comparison.py` | `run_animal_pipeline()`, `cv_comparison()`, `compare_models()` | SBI-based model comparison pipeline |
| `diagnostics.py` | `run_sbc()`, `parameter_recovery()`, `plot_recovery_scatter()` | Simulation-based calibration and recovery |

### `plotting/` — Project-specific plotting

| Module | Key functions | Purpose |
|:-------|:-------------|:--------|
| `cv.py` | `plot_cv_comparison()`, `gs_seed_errors()`, `plot_winner_summary()` | Grid-search CV result visualisation |
| `assignment.py` | `plot_assignment_strip()` | Cohort-level BE/SC assignment strip |
| `sbi.py` | `plot_parameter_trajectories()`, `plot_marginal_posteriors()`, `plot_pairplot()` | SBI posterior and trajectory plots |
| `adaptation.py` | `plot_animal_trajectory()`, `plot_shift_psychometric()`, `plot_group_trajectories()` | Post-shift adaptation plots |
| `opto.py` | `plot_opto_psychometric()`, `plot_phase_trajectory()`, `plot_opto_um_comparison()`, `plot_equivalence_test()`, `plot_animal_opto_report()` | Opto inactivation analysis plots |
| `animal_report.py` | `plot_animal_summary()`, `plot_cv_results()`, `plot_model_fits()` | Per-animal multi-panel report figures |
| `validation_report.py` | `plot_synth_summary()`, `plot_recovery_overlay()`, `build_confusion_matrix()` | Synthetic validation report figures |

### `legacy/` — Old codebase

Still imported by `analysis/cv_utils.py` for `post_correct_update_matrix`, `BE_model`, `SC_model`, `matrix_error`. ~4000 lines. Intended for eventual removal once these functions are ported.

---

## Notebooks

Numbered for reading order. Each has a `MODE` toggle: `'load'` reads cluster results, `'run'` does local analysis.

| File | Title | Purpose | Depends on |
|:-----|:------|:--------|:-----------|
| `00` | Data Exploration | Raw data overview, trial counts, accuracy trajectories | — |
| `01` | Model Explorer | Interactive BE/SC parameter widgets | — |
| `02` | Feature Selection | PCA, feature correlations, stat selection for SBI | — |
| `03` | Parameter Sensitivity | How summary stats respond to model parameters | — |
| `04` | Model vs Real | Model predictions overlaid on real data | — |
| `10` | **Validation Summary** | Synthetic validation: accuracy, recovery, calibration | Cluster: synth GS + SBI |
| `15` | Example Animal | Deep dive on one real animal | Cluster: real GS + SBI |
| `20` | **GS Model Selection** | Per-animal BE/SC assignment via grid-search CV | Cluster: real GS |
| `21` | **SBI Model Selection** | Per-animal BE/SC assignment via amortised SNPE | Cluster: real SBI |
| `22` | **Consensus** | 4-method agreement matrix and final assignments | NB 20, 21 |
| `30` | Adaptation Analysis | Post-shift behavioural adaptation | NB 22 |
| `31` | Parameter Dynamics | Dynamic SBI parameter trajectories (RandomWalk) | Cluster: dynamic SBI |
| `40` | SLDS | Latent behavioural state inference (HMM + SLDS) | — |
| `50` | Opto Predictions | Optogenetic effect predictions from model + SLDS | NB 22, 40 |
| `51` | **Expert Opto Analysis** | Expert-phase opto inactivation results + TOST | NB 22, data |
| `99` | Presentation Figures | Publication-ready figures | All above |

`dev/` notebooks (2a–2h) contain detailed synthetic validation development work.

---

## Pipeline Architecture

```
SLDS (WHERE: learning phase)
  → per-state static model selection (HOW: BE vs SC)
    → consistent? → dynamic SBI on winning model (trajectories)
    → inconsistent? → per-state assignments (strategy switching)
```

## Model Selection Methods

| Method | Fitting | Scoring | Status |
|:-------|:--------|:--------|:-------|
| GS × UM | Grid-search CV (2-fold × 64 seeds) | Update matrix MSE | ✓ Secondary |
| GS × CP | Grid-search CV | Conditional psychometric MSE | ✓ Secondary |
| SBI × UM | Amortised SNPE | Update matrix MSE | ✓ Primary |
| SBI × CP | Amortised SNPE | Conditional psychometric MSE | ✓ Secondary |

Current results: 11 BE / 2 SC / 9 unclear from 23 animals.

## Key Design Decisions

- Stimulus range normalised to [-1, 1] throughout
- Session-level granularity for all claims
- Block-level CV splits (not trial-level) to preserve sequential structure
- SBI for within-model parameter estimation only (not model comparison)
- Dynamic SBI uses RandomWalk linking (smooth parameter evolution)
- `behav_utils` is general; project-specific code stays in this repo
