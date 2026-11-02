
# Changelog — sound_categorisation

## Unreleased

Handover review, batch 3: report numbers change; marked ▲. Needs `behav_utils` 0.9.0.

- ▲ Hard-A / Hard-B density (`data/stimuli.py`): the hard side is the rig's truncated exponential, rate
  2.15329 (`Distribution_Exp_Rate` in `Sound_Cat_V2.bonsai`, recorded in every Hard session), mean distance
  from the boundary 1/3 — the snapshot gives 0.328-0.334 after the stable start. It used the legacy density
  λ·exp(-λd) + exp(-λ), λ ≈ 1.841 (mean 0.378), which this rig never played. Every `normative_pse` grows by
  about a third (Hard-A at sigma 0.2: +0.059 → +0.078), so the switch report's and the trajectory's
  convergence values shrink by about a quarter. Synthetic Hard sessions use the same density. `rate` is a
  parameter of every function; sampling is exact (inverse CDF).
- ▲ `post_opto` contrasts: the reference was every laser-off trial, post-opto ones included, so about a third
  of the reference was the test set and the effect was diluted (and the permutation gave those trials two
  labels; `behav_utils` 0.9.0 refuses that). The reference is now the other laser-off trials, in `within`
  and `within_masking` of both designs, hence `dod`. The key stays `post_opto_vs_non_opto`.
- ▲ `compensation`: laser-off trials of the laser sessions vs the masking sessions' unflagged trials (was all
  masking trials, whose flagged ones carry the masking effect). Both designs.
- ▲ Cohort QC (`adaptation.BiasRule`): an animal is biased only when its flagged Hard sessions lean the same
  way on both Hard-A and Hard-B; flags confined to one distribution used to count as the same lean. The rule
  (thresholds, majority, `both_distributions`) is a `BiasRule`, `Settings.bias_rule` in the opto report and
  `run_switches(bias_rule=…)` in the switch report, written to meta.json.
- ▲ Through `behav_utils` 0.9.0 (exclusions before positional selection): `expert_uniform` now takes the last
  half of the Uniform sessions that are not masking / washout / ALM, so the trajectory baseline can differ where
  the old order left fewer than 5 sessions. The baseline itself is unchanged in definition and is now stated in
  the report README, including that it departs from the masking-only rule; `Settings.baseline_preset`,
  `baseline_last_n`, `baseline_trials` change it.
- Pin `behav_utils` v0.9.0.
- `tests/e2e/reference/e2e_contrasts.csv` regenerated: only the four `compensation` rows moved (reference
  n 540 → 363, the unflagged masking trials); every other pinned number is unchanged.

Handover review, batch 1 (no report number changes).

- `compare_models` pairs BE and SC by rep (grid-search seed / SBI repeat). It used to drop NaN errors from
  each model's list separately and pair the rest by position, so one failed seed (a fit that raised returns
  NaN) misaligned every later pair and could flip the call or its p. Winner, means and p now come from the
  same pairs; `comparison` gains `n_pairs`. **Calls can change** where a run had NaN or missing reps:
  re-run `sc-consensus` on existing runs (it only reads results). `errors_by_rep` is new; `compute_seed_errors`
  is unchanged (its docstring now says not to pair its lists), and the usage example in `plotting/cv.py`,
  which paired them by position, now uses `errors_by_rep`.
- `sc-consensus` with ground truth: `consensus_correct` is NaN for Unclear / Split (was False), and the summary
  reports how many animals were called and the accuracy among those, instead of counting Unclear as wrong.
- The vote's provenance is written down (module docstrings, ARCHITECTURE.md, docs/runs.md): it is the
  manuscript's model selection, with a paired Wilcoxon over seeds / SBI repeats in place of its ANOVA.
- `snapshot_dir()` is `<data root>/behaviour/snapshots` (`paths.data_root`). It used to pick ceph whenever the
  machine ran Linux, so on any non-cluster Linux machine it ignored `SC_DATA_ROOT`. Same paths as before on
  a laptop and on the cluster.
- `load_experiment_any` without a snapshot now removes the trials whose sound was never sent, as the export
  does, so both load paths give the same trials (it used to return them).
- `remove_silent_trials` lines the CSV up against `trials.stimulus_raw` (CSV units) rather than the normalised
  `stimulus`; identical for this config, robust to a `stimulus_range` change.
- `GS_N_SEEDS` 32 → 16 (what the Oct 2026 real run used); `--n-seeds` already overrides it.
- Packaging: `ssm` dropped from `.[fit]` (nothing imports it; SLDS is not written). The version is set once,
  in `sound_categorisation/__init__.py` (it said 0.6.1 while pyproject said 0.7.0); pyproject reads it.
- Docs: SETUP's path to `data/paths.py`, the editable-install requirement, the full `sc-*` list; README
  wording on where `behav_utils` lives; 4 networks by default (18 available) in SETUP / ARCHITECTURE;
  `docs/runs.md` says ≥ 250 valid trials (the preset) and 16 seeds.

- `behav_utils` 0.8.0 (pin moved from 0.7.1). Call sites changed: `SessionData.summary()` is a
  `SessionSummary` (`.accuracy`, not `['perf']`) in `data/cohort.py`; `compare_groups` returns a
  `GroupComparison` whose `.table()` feeds `group_tests.csv` in `reports/compute.py` and
  `reports/light_artefact.py` (same `p` / `min_p` columns; the unread `test` / `paired` columns are gone);
  `reports/figures.py` uses `plot_stat_comparison` / `plot_interaction` (the single-panel plotters,
  formerly `*_single`). Numbers unchanged: `tests/e2e/reference/e2e_contrasts.csv` matches to 1e-9.
  Behaviour to know about: `load_experiment` now raises on a session it used to skip with a warning
  (`strict=False` restores the skip); `ExperimentData.load` warns on a snapshot written by another
  library version — re-export the snapshot rather than carry it forward; `animal.session_table` is
  counts + accuracy only (per-session statistics come from `compute_phase_stats(per_session=True)`);
  `TrialArrays.trial_key` is available for aligning per-trial neural data.
- Tests: the two grid-search unit tests were vacuous (a `hasattr` guard that never held; a smoke
  test that passed the dict of both grids, caught the resulting exception and skipped). They now
  check `FAST < COARSE < DEFAULT` per model and run `compute_grid_search_cv` on `FAST_GRID['BE']`
  with no try/except. `test_run_id_shape` accepts `nogit` so it passes from a tarball.


- `data/stimulus_check.py`: `check_stimulus_delivery(trial_summary_csv)` reads a run's Trial_Epochs and
  Long_Form_Timestamps logs and says, per trial, whether the sound was sent (True / False / NA when the logs cannot
  tell). numpy and pandas only, so the file can be shared on its own.
- `export_snapshot` removes trials whose sound was never sent (the rig's Asym sampler fault: about 6 % of trials in
  Asym_Left / Asym_Right sessions run before 5 Oct 2026, all on the hard side, each carrying the previous trial's
  Stim_Relative). The trial after a removed one loses its lag-1 history; trials that cannot be checked (the
  cut-off end of runs that ended with Bonsai being killed) are kept, with the result in
  `trials.extra['stimulus_check']`. Counts are in the snapshot metadata under `stimulus_check`. Sessions whose
  files do not line up with the loaded trials are left as loaded, with a warning.
- Snapshot format version 2: version-1 snapshots still contain those trials, so `load_snapshot` refuses them.
  Re-export (`sc-export-snapshot`) and re-run the reports; Hard-session numbers move slightly, Uniform does not,
  and the synthetic e2e reference is unchanged.

## 0.7.0 — 2026-10

- Model identification for the talk run. `sc-feature-selection` (tables, plots, forward-selection
  curve, saved cohort). Training/conditioning grids come from `settings.SBI_TRAIN_REPS` /
  `SBI_TRAIN_DISTRIBUTIONS` (4 networks now). Real animals: the snapshot, the `model_id_<dist>`
  presets (`last_n_qualifying`, `behav_utils` 0.7.1), `MODEL_ID_MIN_SESSIONS`, a `sessions/<dist>.csv`
  per run. `sc-sbi-condition` saves 1,000 posterior samples per animal (`posterior`); its CV
  `best_params` is that posterior's median. New `sc-sbi-sbc` + `inference/calibration.py`
  (`compute_sbc_ranks`, coverage). Consensus voters: GS-UM, SBI-pooled-UM, SBI-moments-UM.
  Local `sc-grid-search` writes partials as the cluster does; seeds by cohort kind. Synthetic
  cohort default 10 + 10; parameter truths drawn from one stream per model. `simulate_sessions`
  factored out of the simulator (bit-exact). Stale-network warning when a net's stat vector
  differs from `SBI_STATS`. `SBI_TRAIN_T` is now a per-session range `(250, 500)` matching real
  Uniform sessions (`simulate_sessions`/`draw_T`; the synthetic cohort and SBC draw from it too);
  `model_id_*` presets use `min_trials: 250`. A skipped conditioning now warns with the reason.
  Training simulations run in parallel (`wrap_for_sbi`, joblib, per-row seeds so n_jobs does not
  change results); the local grid-search runner takes `--n-jobs`. Slurm: `sbi_sbc.sh`, resources
  sized for the `cpu` partition, grid-search arrays throttled (`GS_ARRAY_THROTTLE`), thread caps.
  `AmortisedSBI.condition` samples the flow with a proposal budget instead of sbi's unbudgeted
  rejection loop (an observation the net puts ~0 prior-box mass on -- typically the other
  model's data -- used to run for hours); results carry `acceptance_rate`. Jobs run unbuffered.
  **Numbers change:** `split_folds_by_block` now makes contiguous folds balanced by session count
  (4 -> 2/2, 5 -> 3/2) instead of the trial-balanced greedy merge, which could put a single
  session in a fold and starve the moments net. Grid search and SBI share the same folds as before.

## 0.6.1 — 2026-10

- `behav_utils` v0.7.0 (its `analysis/` reorganised; public names unchanged): the four submodule imports
  the project made follow the new paths. CI tolerates pytest's "nothing collected" on `tests/fit`.

## 0.6.0 — 2026-10 (library externalised)

- `behav_utils` is no longer vendored: the project depends on
  `behav_utils @ git+https://github.com/serkanshentyurk/behav_utils.git@v0.6.0`. Bump that tag when the
  project needs a newer library; for parallel development install a local checkout editable on top.

## 0.5.1 — 2026-10

- Run ids carry the time (`YYYY-MM-DD_HHMM_<sha7>`): two runs of the same kind on the same day at the same
  commit no longer collide. Older, date-only ids are still recognised.
- `sc-reports battery` runs `check → opto-contrasts --all → light-artefact → summary`, one run per stage,
  with `--fast` (pipe-clean), `--skip` and `--only` (resume a crashed battery).

## 0.5.0 — 2026-10 (light-artefact report)

- `sc-reports light-artefact`: light-on − light-off on every light-carrying session set (`ppc_opto`,
  `ppc_sham`, `alm_uni`, `alm_bi`) with `side_bias accuracy mu sigma dprime criterion`, P(B) per stimulus
  bin on vs off, the `ppc_sham − alm` site interaction, and the WT-vs-HET fold; `behaviour/light_artefact.py`
  + `reports/light_artefact.py`; results under `light_artefact/<cohort>/<run_id>/<distribution>/`.
    Notebook `20` §3 reads it. `behav_utils` 0.6.0 adds the `sdt` fitter (`dprime`, `criterion`).
- CLIs echo the options on a wrong input: unknown cohort → the cohorts in `config.yaml` (and synthetic
  cohorts on disk), unknown animal → the animals in the experiment, unknown run id → the runs under that
  cohort, missing report → the cohorts that have runs.
- `behav_utils` packaging metadata (classifiers, URLs, keywords) and, inside `behav_utils/`, its own
  `.github/workflows/{ci,release}.yml` and `.gitignore` so the folder is a complete repository when copied
  out; a `v*` tag there builds, checks and tests the wheel and attaches it to a GitHub release.

## 0.4.1 — 2026-10 (naming pass + levels)

- Names say what they are: `sc-reports opto-contrasts | switch-adaptation | summary | battery` (subcommands =
  report folders); `--trial-class opto|post_opto` (was `--toi`); one `--site ppc|alm_uni|alm_bi` (was
  `--design` + `--site`) and one `site` column in every table (`reports.tables.normalise_site` reads older
  runs); `sc-grid-search`, `sc-sbi-condition`, `sc-sbi-train` (were `sc-run-gs`, `sc-run-sbi`,
  `sc-train-sbi`); `slurm/submit.sh grid-search|sbi-condition|sbi-train`; the model-identification runners
  take `--cohort` only (a synthetic cohort or a config cohort, detected from the name; `--source`/`--label`
  gone) and restrict real runs to the cohort's animals; the synthetic cohort is `synthetic` (was `selftest`).
- `levels.csv` per animal in opto-contrasts runs: the observed value of every stat in every condition the
  contrasts were built from.
- Notebooks: `10` reads levels; `11` classifies each switch as toward / none / reverse and lists the odd
  adapters.

## 0.4.0 — 2026-10 (the cleanup pass)

- Package laid out by aims: `settings.py`, `data/`, `behaviour/`, `models/`, `inference/`, `features/`,
  `reports/`, `plotting/`, `cli/`. `paths.py` split into locations (`data/paths.py`) and constants
  (`settings.py`). `BE_core`/`SC_core` → `be_core`/`sc_core`.
- Entry points replace `scripts/`: `sc-reports`, `sc-export-snapshot`, `sc-make-synthetic-cohort`,
  `sc-new-run`, `sc-sbi-train`, `sc-sbi-condition`, `sc-grid-search`, `sc-consensus`.
- Results: one root (`paths.results_root()`), one layout `<report>/<cohort>/<run_id>/`, run id =
  date + git sha (+ `_fast`), `latest` symlink + `latest.txt`, `meta.json` stamped with run id, argv and
  git state. `run_reports.sh` replaced by `sc-reports battery`. `slurm/submit.sh` creates and shares the
  run id across array tasks and sends logs to `<run>/logs/`.
- `sc-reports` subcommands are report types (`opto`, `switches`, `summary`, `battery`); `--level`
  replaces `animal`/`group`; `synthetic` removed in favour of `pytest tests/e2e`.
- Model identification writes `grid_search/`, `sbi/` and `consensus/` into one run; fixes the
  consensus reading a path the runners never wrote (missing distribution level). `--run quick|full`
  and `--smoke-test` → `--fast` / `--coarse`; `SMOKE_*` → `FAST_*`.
- Tests split into `tests/{unit,e2e,fit}`; pinned reference at `tests/e2e/reference/e2e_contrasts.csv`
  (`--regen-reference`); `test_simulator` back in CI; new `test_paths`.
- Removed dead code: `plotting/overview.py`, `plot_delta_paired`, `plot_stat_trajectory`,
  `apply_smoke_test_overrides`, `load_animal_data`, `list_animal_ids`; `reports/selftest.py` (generator
  now `data/synthetic.py`).
- Docs: `docs/runs.md` (every command, inputs, outputs), README/ARCHITECTURE/SETUP/LLM_CONTEXT
  rewritten for the layout; `behav_utils` 0.5.0 (see its CHANGELOG).
- Notebooks rewritten: `00` data/task, `10` expert behaviour, `11` switch adaptation, `20` opto experts,
  `21` opto Hard, `31` model identification; each reads a run's tables (`notebooks/nb_setup.py`) and
  documents the command that produced them. `sc-make-synthetic-run` builds synthetic runs so CI executes
  the notebooks (`nbmake`, `SC_NB_SYNTHETIC=1`). `behav_utils/notebooks/example_workflow.ipynb` shows the
  library end to end on synthetic data. `dev/` notebooks kept, not maintained.

## 0.3.0 — 2026-09

- Report pipeline, typed contrasts, switch-adaptation report, task grids, CI.
