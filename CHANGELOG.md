
# Changelog — sound_categorisation

## Unreleased

- **behav_utils 1.0** (pyproject: `behav_utils[store] @ …@v1.0.0`, Python ≥ 3.11; CI on 3.11 and 3.12).
  - `config.yaml`: `reader:` (csv_folders) replaces `file_structure:` (`animal_pattern`, `session_pattern`,
    `date_format` gone; `merge_files` and `min_trials_per_file` now written out); `task.b_side: positive` replaces
    `category_rule`, and `inputs`, `outputs`, `n_categories` are gone; the tracker's settings moved to
    `custom.daily_tracker` (1.0 refuses unknown top-level keys).
  - Every import from a behav_utils module path now comes from the public sub-package (`behav_utils.data`,
    `.stats`, `.readouts`, …). Names 1.0 removed or made private are replaced: `contrast_key` (a local helper, same
    keys), `parse_date_from_path` (`raw_mirror.folder_date`), `sample_stimuli` (`data.stimuli.sample_distribution`),
    `list_producers` (`features.feature_diagnostics.list_producers`, built from the public tuples; the same 26
    producers), `get_preset` (`preset_from_spec` on the config's preset).
  - Update matrices take the task's score: `fit_update_matrix(..., correct=)` gets `pool_arrays`' `correct` for
    data and `grid_search.score_choices` for simulated choices; grid search and model selection pass the task on.
  - Daily tracker: no-response and early-lick counts read the rig's labels from `outcome_recorded` (1.0's
    `outcome` holds codes, and the old reading counted zero); `correct` is the task's score; the session folder is
    `source` (1.0's name for it; `csv_path` was a file inside it); the file time is the folder's newest CSV.
    Sessions whose only file has fewer than `min_trials_per_file` rows are no longer loaded (1.0), so the tracker
    no longer lists them.
  - Snapshot (still a pickle): format 3. A snapshot exported with behav_utils 0.10 is refused (its objects
    unpickle under 1.0 without the new fields): re-export. The stimulus check reads the session folder from
    `source` and follows 1.0's file rules; loading no longer re-applies `session_types` (1.0 applies them when
    the data are loaded, and keeps that step private): edit them, then re-export.
  - Feature diagnostics draw their fixed stimulus sequence with the project's sampler (notebook 90's numbers can
    move at Monte-Carlo level; `SBI_STATS` does not change).

- `slurm/submit.sh` keeps only the last line of `--print-array` and stops if it is not an array range. Grid
  search on a real cohort prints the snapshot and cohort lines first, and all of it went into `--array`
  (sbatch: invalid job array specification).
- SBI networks see how much data they are given: `to_stat_vector` appends the number of sessions and valid trials
  (`SBI_DESIGN_FEATURES`) and the networks train on 2–6 sessions (was 4–6), so the CV halves (2–3 sessions) the model
  comparison conditions on are inside the training design (100k pooled / 150k moments simulations). Networks saved
  before this have no design features and still condition as they were trained. **The October 2026 networks are
  superseded; retrain (move them to `snpe_networks/2026-10/` first) and redo the all-mice run.**
- `AmortisedSBI` saves its design features and trained prior bounds; `load` raises when the saved parameter order
  differs from the model's; conditioning warns when the design features differ from settings and reports
  `budget_hit`. `sc-sbi-train` refuses to replace a network trained with a different representation (`--overwrite`).
- `sc-sbi-check` (new, `inference/network_check.py`, `slurm/sbi_check.sh`, `submit.sh sbi-check`): saved networks
  against the current settings and code, and a past run's sessions against the current selection.
- SBC: one task per (rep, model, session count) (`SBC_GRID`: 5, 3 and 2 sessions; 1,000 / 500 / 500 draws); each draw
  keeps the posterior mean, SD, quantiles and acceptance; output under `sbc/<D>/n<N>/`; seeds from `VALIDATION_SEED`
  (SBC draws used to share numpy seeds, and so stimulus streams, with training simulations).
- `sc-make-synthetic-cohort --shape-from <cohort>/<run> --accuracy-rule`: synthetic animals copy real session counts
  and lengths and pass the model-ID accuracy threshold; seeds from `VALIDATION_SEED` + 1,000,000; the cohort records
  how many draws were rejected.
- Grid search uses 16 seeds for synthetic cohorts too (`SYNTH_GS_N_SEEDS` removed). Grid search and SBI keep each
  repeat's per-fold errors (`fold_errors`).
- Consensus: `n_sig_BE`, `n_sig_SC` and `contested` (a call made with a significant vote against it).
- `sc-validation-tables` (new, `inference/validation.py`): the thesis tables of a validation run. Notebook `91`
  (appendix) draws them; `11` points to it and shows both consensus rules. SBC and recovery plotters are single-panel
  (`plot_sbc_ecdf_diff`, `plot_coverage`, `plot_recovery_scatter`, `plot_contraction`; the multi-panel ones and the
  recovery plotters without a producer are gone); `plot_confusion` shows a no-call column.
- `sc-daily-tracker`: a daily HTML report of every animal with a session in the last 14 days, read from the
  session files on ceph without the snapshot or the cluster. Per day: trials, responded, correct,
  no-responses, early-lick aborts, accuracy (overall, hard, easy), right choices − 0.5, accuracy right − left,
  PSE and lapses (trials without light), preset, stage, distribution, rig. Per condition (preset), the last
  five sessions pooled and as mean ± SD with the individual sessions, light trials separately. Flags: side
  bias (BiasRule per session, free-fit PSE), low trials, accuracy drop, missing day. Move-on marks from
  behaviour_rig/TRAINING.md. Every number is in the new `config.yaml` block `daily_tracker`.
  The page opens on an overview sorted by urgency (with 14-day trends and session numbers); each
  animal has its own view (#<animal>, arrow keys to step through); expected animals with no session on
  ceph are listed (`daily_tracker.expected_cohorts`); `daily_tracker_latest.html` for a bookmark.
- `data/raw_mirror.py`: the tracker's local copy, from the local folder (rig PC, cluster) or over ssh
  (`find` + `tar`, one connection); only `Trial_Summary` and `Session_Parameters`, never videos.
- `adaptation.session_bias_checks`: the per-session part of `BiasRule`, naming the test that failed;
  `flag_biased_sessions` uses it (no number changes).
- `behav_utils` pinned to v0.10.0 (adds `accuracy_b_minus_a`).
- `sc-daily-tracker --no-copy`: rebuild the page from the local copy, without contacting ceph; the page
  says so at the top. A `--cohort` run no longer deletes the other animals from the local copy.
- `DAILY_TRACKER.md`: setting up and reading the daily tracker without knowing the code (whose data,
  who logs in, where the copy and the pages go, what to do when it fails); linked from README and
  START_HERE.
- `SWC_NODE`: the login node `sc-fetch` and `sc-daily-tracker` go through (default `hpc-gw2`), e.g.
  `hpc-gw1` when `hpc-gw2` does not answer. A failed ssh listing now suggests it.

- Notebooks follow the argument: `10` expert behaviour without light (computes point estimates from the
  snapshot, the same sessions model identification fits), `11` model identification (was `31`; the all-mice
  run first, descriptive), `12` switches (was `11`), `20`, `21`, `90` SBI feature selection (was `30`).
- `12`: switches without a fit (a PSE missing or beyond ±1) are left out instead of counted as reverse; the
  expected direction is the switch's; animals the bias check drops are not tallied; §6 compares the Hard-A −
  Hard-B criterion difference with the ideal observer's.
- `20`, §4b: equivalence — per animal and statistic, is the 95 % interval inside a margin set in advance
  (`mu`, `sigma` ±0.10; the rest ±0.05).
- `21`: neither design produced adaptation, so the post-shift prediction is untested.
- START_HERE "Status" and LLM_CONTEXT "What the data say" carry the November 2026 numbers.
- With the real-data results of §4b (equivalence) and §6 (blocks), and the size of the silent-trial residue
  (about 1 % of Hard trials) in the caveats.

- `sc-reports opto-contrasts` / `battery --jobs N`: N animals computed at once (separate processes). Same
  numbers as one at a time, since every resampling is seeded; writing and plotting stay in the main process.
  `joblib>=1.3`.
- Reports: a cohort member with no data (SS03) is left out with a warning instead of stopping the run; an
  animal named with `--animals` that has no data still stops it.
- START_HERE.md: "New animals" (rig row, cohort, genotype in `animal_metadata.json`, session types, re-export).

- Pin `behav_utils` v0.9.1: empty `Trial_Summary` files (a run stopped before writing anything) are skipped
  as if absent; the export used to stop on them (`SS01/SOUND_CAT_SS01_2026_1_22`). The silent-trial check
  ignores them the same way. A session whose real data sat next to an empty file was dropped whole by
  `behav_utils` < 0.8 and so missing from older snapshots; it is in now.

Docs. No code changes.

- `START_HERE.md` is the way in: the three repositories, your data folder, the everyday routines, where
  things are, who to ask, the caveats in the data.
- README, SETUP, `docs/runs.md`, ARCHITECTURE, the notebooks README and LLM_CONTEXT rewritten to match it,
  in plain language; command blocks carry no inline comments.

Everyday commands. No report numbers change.

- On the cluster every path now hangs off `BEHAV_DATA_DIR` (your folder on ceph): raw data in `Raw`,
  snapshot and networks in `Processed`, results in `Processed/results`. It stops with a message if
  `BEHAV_DATA_DIR` is not set, instead of falling back to `/ceph/akrami/Serkan/...`.
- One `config.yaml` on every machine: a `config_slurm.yaml` next to it on the cluster is no longer read
  (its only job was the raw-data path, which `BEHAV_DATA_DIR` now gives).
- `slurm/env.sh`: the cluster environment in one file (module, conda env, thread settings). The Slurm
  jobs source it; the `load-env` alias in START_HERE.md does too.
- `sc-update`: `git pull` + `pip install -e .`; `--keep-behav-utils` installs every dependency except
  `behav_utils`, so an editable copy stays (plain `pip --no-deps` would also skip new dependencies).
- `sc-fetch snapshot | networks | synthetic | results <report>/<cohort>`: copies from ceph to the same
  place on a laptop (rsync over ssh, as `$SWC_USER`).

Handover review, batch 3. Report numbers change (▲). Needs `behav_utils` 0.9.0.

- ▲ Hard-A/B stimulus density now matches the rig: a truncated exponential with rate 2.15329 (mean distance
  from the boundary 1/3). The old density was not what the rig played. Normative PSEs are about a third
  larger, so convergence values drop by about a quarter.
- ▲ `post_opto` is compared with the other laser-off trials. Before, the post-opto trials were also in the
  reference, which diluted the effect.
- ▲ `compensation` compares laser-off trials with the unflagged masking trials (was: all masking trials).
- ▲ Bias QC: an animal counts as biased only if its flagged sessions lean the same way on both Hard-A and
  Hard-B. The rule's settings are in `adaptation.BiasRule`.
- ▲ The trajectory baseline can shift slightly: `behav_utils` 0.9.0 drops excluded sessions before taking
  the last N. Its settings are `Settings.baseline_*`; the report README says how it is chosen.
- Pin `behav_utils` v0.9.0. The pinned e2e reference was regenerated; only the four `compensation` rows moved.

Handover review, batch 1. No report numbers change.

- `compare_models` pairs BE and SC by seed. It used to pair them by position after dropping NaNs, which could
  misalign the pairs. On the October all-mice run no call changed.
- `sc-consensus` with ground truth reports how many animals were called and how many of those were right;
  Unclear and Split no longer count as wrong.
- `snapshot_dir()` follows `SC_DATA_ROOT` on every machine (it ignored it on any Linux machine).
- Loading without a snapshot now also removes the trials whose sound was never sent.
- `remove_silent_trials` compares against the raw stimulus column (same result for this config).
- Grid search defaults to 16 seeds (was 32); `--n-seeds` changes it.
- `ssm` removed from `.[fit]`. The version is set in one place, `sound_categorisation/__init__.py`.
- Docs: corrected paths and network counts; the model-ID call is documented as the manuscript's recipe.

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
