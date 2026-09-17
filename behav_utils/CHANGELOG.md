# Changelog

## 0.4.0 — 2026-09
- Library is task-agnostic: `session_types` and `session_presets` come from the config; hardcoded
  presets, `masking`/`washout`/ALM vocabulary and the Hard-A/B adaptation module removed.
  `SessionFilter.exclude_types` replaces the three exclusion flags. `SessionData.masking/.washout` removed.
- `find_switches` (generic distribution-switch detection).
- `pse_dynamics` fit producer (exponential vs step PSE trajectory), optional pinned shape.
- `UpdateMatrix.from_matrix`.
- `filter_trials(trial_type='all')` drops abort trials (bug fix; docstring and code disagreed).
- src/ layout; own test suite in `tests/`.

## 0.3.0 — 2026-09
- Typed stat registry: `TrialArrays` + `compute_stats -> pd.Series`; `@stat`/`@fit`; no family names.
- Readout dataclasses: `PsychometricCurve`, `UpdateMatrix`, `ConditionalPsychometric`, `BinnedCurve`,
  `SerialDependenceProfile`, each with one draw-only plotter.
- `compute_stat -> PhaseStats`, `compute_delta_stat -> DeltaStats`, `compute_interaction -> Interaction`,
  `compute_rolling_stats -> RollingStats`; resampling engines return DataFrames of draws.
- Removed: `summary_stats`, `stats_table`, `trajectory`, `compute_psychometric`, `compute_um`,
  `average_um`, `plot_um`, `plot_psychometric`, `compare_phases`.
- Bit-exact with 0.2 for every scalar statistic and every bootstrap draw (verified before deletion).

## 0.2.0 — 2026-07
- Baseline before the refactor.
