
# Changelog — sound_categorisation

## Unreleased

- **Which fitted model predicts the Hard sessions?** (thesis analysis 3, agreed 20 November 2026).
  `sc-hard-predictions --fits all-mice/<run id> --cohort C` (`inference/hard_predictions.py`,
  `plotting/hard_predictions.py`): each animal runs from its fitted Uniform sessions (the fit run's
  `sessions/uniform.csv`) straight into its Hard sessions, from its first switch into Hard to its last Hard session;
  what lies between (SS08's and SS13's toggles, the opto mice's Uniform masking and laser sessions) is left out and
  listed in `sessions.csv`. For each model and draw of its posterior (default 200), a copy of the animal plays that
  span in one continuous run, carrying what it learnt from day to day; on laser trials it learns as on any other or
  not at all (`--laser-learning`, both by default where there are laser trials). The animal and every copy go
  through the switch report's own computation; which sessions enter is fixed to the animal's, and the Uniform
  reference is one fit on the fitted sessions for both, so observed values can differ a little from the switch
  report's. Scored by the observed value's percentile among the predictions and its CRPS, summed per model over the
  animals, the animals the bias check keeps and flags apart. Writes `hard_predictions/<cohort>/<run_id>/`;
  notebook 11 §6 draws it (the former §6 is now §7). The synthetic runs add a Hard-prediction run with prior draws
  in place of fits. `reports.switches.animal_switch_result` is one animal's part of `compute_switches_cohort`, with
  the bias check optional; the switch report's tables are unchanged.
- **Grid search: a 12-hour limit, and one command to resubmit unfinished tasks.** On the learning-prior grid a task
  took a median 1.4 h (BE) and 2.4 h (SC), 90 % of them under 4 and 5 h. The 6-hour limit stopped 9 of the 608
  all-mice tasks (SC fits of SS20, SS21 and SS23) and 56 of the first thousand synthetic ones, so
  `slurm/grid_search.sh` now asks for 12 h. `bash slurm/submit.sh grid-search ... --run-id ID --missing` resubmits
  the tasks of a run that left no partial (a time limit, a node failure). `sc-grid-search --print-missing` now prints
  them as SLURM ranges ('0-3,7', `tasks.slurm_array`), or 'none', so thousands of tasks stay a short `--array`.
- **Is a silenced trial remembered on the next one?** (thesis analysis 4, agreed 18 November 2026).
  `sc-blocked-update --fits all-mice/<run id>` (`inference/blocked_update.py`, `plotting/blocked_update.py`) runs
  each animal's fitted BE and SC models (the first 50 samples of the pooled network's saved posterior) on its own
  Uniform opto sessions twice, learning from every trial and not learning on laser trials
  (`laser_blocks_update`), with the same random numbers; blocked − normal on the post-laser contrast is the effect
  complete blocking would have on recency, win-stay and lose-shift. It is read against the animal's observed
  post-laser delta-of-deltas from the battery (opto − masking sessions) and the equivalence margins: too weak to
  test, not fully blocked, consistent with blocking, beyond blocking, unclear. Masking sessions are not simulated:
  nothing is silenced there, so the two runs would be the same and that term cancels. Writes
  `blocked_update/<cohort>/<run_id>/{predicted, summary, sessions}.csv`; notebook 20 §7 draws it. Both models for
  every mouse, since model identification is descriptive. The synthetic runs add the post-laser condition on Uniform
  (every statistic, with intervals) and a blocked-update run with prior draws in place of fits.
- `cv_utils.read_cv_result` reads a saved fit on a machine without torch (moved from `prior_range`, which uses it).
- Summary pages: the overall page's panel titles ran into each other (now two lines each, and the page title too);
  the session-order panel on the Uniform page said grey = masking, but grey is every Uniform session without the
  laser (regular, masking, washout) and coloured is the laser at PPC or ALM; the Hard pages' PSE axis said
  "all trials", which stopped being true when laser days moved to their laser-off trials. Labels only:
  `sc-reports summary` redraws them from the tables.
- Notebooks: `open_run` gave a `KeyError` instead of its message for a report it had no hint for, so notebook 11 §5
  stopped when the validation cohort had no noise-floor run; every report now has a hint.
- **`no_learning` renamed `no_history`** in `noise_floor.csv`, `noise_floor_sessions.csv`, `floor_ratios` and the plot: the
  reference's curve is fitted on choices the animal made while updating, so it carries the average of that updating;
  it lacks only the dependence on the previous trial, which is what the update matrix measures. Rerun
  `sc-noise-floor`. First real run (all-mice, 6fd4a93): the no-history reference scores a median 1.20 times the floor
  on the session split, above 1.35 in 6 of 19 mice, and 1.11 within sessions (above 1.35 in 4).
- **Laser days in the trajectory on laser-off trials.** `compute_trajectory` measured every session on all trials,
  so a laser day's criterion and the shape of the laser days included the laser-on trials, while the switch
  report measures a laser day on its laser-off trials, as the docs said both did. On the opto cohort's battery
  (6fd4a93) the two reports agreed exactly on masking days and differed on laser days by a median 0.26 in a day's
  convergence. One rule now (`adaptation.measured_trials`: laser-off trials on a laser day, all trials otherwise),
  the trajectory's default (`trials='auto'`) and the switch report's; the shapes of a laser phase are fitted on
  the same trials. The battery's trajectory pages need a rerun; its contrasts are untouched.
- **Noise floor for the held-out error** (thesis analysis 1, agreed 18 November 2026). `sc-noise-floor --cohort C`
  (`inference/noise_floor.py`, `plotting/noise_floor.py`) writes `noise_floor/<cohort>/<run_id>/`: per animal, the
  update-matrix error between the two halves the fits are scored on (sessions 1–3 against 4–5; each session's first
  half against its second for the single network), a no-history reference (the animal's psychometric fitted on
  one half and scored on the other like a fit, with no dependence on the previous trial) and, on a synthetic cohort, the true
  model scored the same way. A fit is scored on one simulation of itself, as noisy as the held-out half, so a
  correct model scores about the floor, not half of it as the plan had said: on 44 simulated mice (five sessions
  of 250–500 trials) the true model averaged 1.03 times the floor, 0.76–1.35 for 80 % of them; with its matrix
  averaged over simulations it would be 0.56. Within a session the no-history reference sits barely above the
  floor (about 1.05 times on four simulated mice), so half-sessions hardly separate any model from one without history.
  Notebook 11 §5 draws each model's held-out error over the floor. `inference/selection.py` now names the fits'
  splits and scoring for reuse (`fold_sessions`, `session_halves`, `empirical_target`, `target_from_choices`,
  `simulated_target`; conditioning unchanged), and `sc-make-synthetic-run` adds the synthetic floor run.

- **Each phase's own shape for the criterion; blocks judged by their whole-block fit** (agreed 18 November 2026).
  - The criterion is fitted with the slope and lapses of its phase, not the expert-Uniform ones: one 4-parameter
    fit on the phase's pooled trials (`phase_shape`), a phase being a block of days or, in the daily alternation,
    every day of one distribution and session type (`phase_shapes`; laser days on laser-off trials); the Uniform
    phase keeps the expert-Uniform fit, and the criterion before a switch takes the previous phase's shape. Some
    blocked mice change their lapses between phases (SS05: lower lapse 0.20 in Hard-A, upper 0.20 in Hard-B), and
    one shape for every phase read that as a criterion shift: −0.09 on SS05's Hard-A − Hard-B, −0.08 on SS09's first
    switch, −0.04 on SS07; 0.02 or less in the opto mice. In simulation with SS05's shapes the expert shape reads
    Hard-A − Hard-B as −0.235 against a true −0.162; each phase's shape reads −0.148 (six seeds). The cost is
    precision: Hard stimuli sit near the boundary and pin the lapses down loosely, so a phase's shape carries its
    own error. In simulation of opto-like mice (lapses the same in every phase, 400 trials a day, 40 mice per
    case) neither rule is biased, and the per-mouse spread of `convergence_100_300` (robust SD) goes from 0.16 to
    0.19 on masking days at half-following, 0.09 to 0.13 at full following, 0.30 to 0.45 on laser days with no
    following. For the opto mice the old shape's bias was 0.02 or less, so for days the two rules are close; for
    blocks the bias was the larger error. The shapes are
    in `trajectory.csv` (`shape_*`) and `switches.csv` (`pre_shape_*`, `shape_*`). The bias check keeps the
    expert-Uniform shape (a coarse |criterion| > 0.4 screen; the excluded mice do not move with the method).
  - Blocks: the result per mouse is `block_convergence` (`pre_post.csv`, `animals.csv`): the whole block's pooled
    PSE − the previous phase's, over the normative shift, from 4-parameter fits on thousands of trials; empty when
    either PSE lies outside ±1. The curve's `convergence_100_300` and `convergence_final` are no longer given per
    mouse for blocks (one switch per kind, spans of 0.01–0.06 against window noise of 0.3–0.6 put them between
    −3.6 and +7.8); the curve stays as a description (the mean over mice, no interval). The blocked speed page shows
    `block_convergence` by direction, Hard-A → Hard-B being the switch the slow upward drift works against. `SwitchResult.normative_from`;
    `switches.block_table`; `meta.json` records `criterion_shape: phase`.
  - `compute_switch(shape=…)` is now `expert_shape=…` (the Uniform phase's shape); the switch report's
    `sessions.csv` drops `pse_fixed` (the bias check's column lives in `bias.csv`).
  - The blocked before/after page counts only the blocks whose PSEs lie inside ±1, as `block_convergence` does
    (the others drawn dotted and named), and puts each number on its own title line (they ran into each other).
    First real runs on this measure (b194525): days, masking 0.41 [−0.04, 0.85], laser 0.24 [−0.15, 0.62] over
    six mice (0.39 and 0.24 with the expert shape); blocks unchanged (whole-block fits do not use the shape).
  - Notebook 13 (Fig 3.6) shows days by `convergence_100_300` and blocks by `block_convergence`; its curve panel
    shows the days alone (on the real tables the blocked curve swung between −1 and 2 and hid them); notebook 21,
    START_HERE and the notebooks README reword the claim: the opto mice followed the daily switch partly at most,
    and in the blocks one clean mouse followed, two partly, one until one-sided, one the wrong way.
  - `reports/summary.py` skips header-only CSVs when gathering tables (pandas' FutureWarning on empty frames).

- **One adaptation measure for both Hard designs; the timescale analysis removed** (agreed 18 November 2026).
  `behaviour/adaptation.py` measures how fast the criterion follows the distribution one way for the switch
  report and the battery: the criterion is the 50 % point with slope and lapses fixed at the expert-Uniform fit
  (`pinned_pse`; a fit that ends at the ±1 edge of the stimulus range is missing), its curve in running 50-trial
  windows every 10 trials (`criterion_curve`), convergence from the criterion over the last 250 trials before the
  switch (0) to the normative one at the expert-Uniform sigma (1), unclipped, and one number per switch,
  `convergence_100_300` (windows centred on trials 100-300, inside the first day). A mouse's switches of one kind
  are combined weighted by |normative − before| (`pool_convergence`); in simulation the plain mean of the ratios
  read a half shift as 0.42 and no shift as −0.13, this gives 0.49 and 0.01.
  - Fixed: the switch report's shape-fixed methods (`pinned`, `pinned_running`) set the fitted location parameter
    of each window against a 50 % point before the switch; with unequal lapses no change then read anywhere from
    −0.97 to +0.75 on the behaviour1 switches. Both sides are now 50 % points with one shape per animal (the expert-Uniform fit,
    no longer a 250-trial pre-switch fit, whose upper lapse reached its 0.5 bound in some mice).
  - Fixed: `toward_pre` (overnight) was signed by the end-of-session value it is measured from, so noise alone
    made it positive (+0.06 to +0.20 in simulation at these mice's noise; the clean behaviour1 mice read +0.12).
    It is now signed by the direction of the switch, and pairs at the edge are missing. Notebook 12's overnight
    panel counted the sum of the values as a number of pairs; it counts pairs.
  - The switch report handles the daily alternation: `hard_design` per animal (median run of days on one
    distribution ≥ 3 → blocked); in the daily design each day of ≥ 100 trials is a switch, masking days kept, a
    laser day measured on laser-off trials (its own and the day before's); `switch_kind` = first / novel / return
    for blocks, first day / laser day / masking day / mixed day for days. New tables `animals.csv` (one number per
    animal and kind) and `bias.csv`; a speed page per design (the curve with 95 % t intervals over mice, the number
    per mouse, and for blocks where the Hard-A → Hard-B block ends, the switch the slow upward drift works
    against). The manuscript recipe stays for comparison, with the normative criterion at the expert-Uniform sigma.
  - One bias check, `bias_sessions` (every Hard session on laser-off trials, `BiasRule`), used by the switch
    report and the battery (`compute_trajectory(bias=True)`, `Trajectory.bias`); each report used to apply the rule
    to its own sessions, and mice near its 50 % line came out differently.
  - Battery trajectory on the same measure: curves in 50-trial windows with the shape fixed (they were 100-trial
    free fits measured from the previous session's whole-session PSE); `baseline_pse` is the criterion over the
    previous session's last 250 trials; new columns `pse_pinned`, `pse_last250`, `convergence_100_300`, `span`;
    `convergence_final` is over the session's last 250 trials. The per-animal trajectory page shows
    `convergence_100_300`, summary row 6 the 50 % point.
  - Removed: `behaviour/timescale.py`, `plotting/timescale.py`, `reports/timescale.py`, `sc-reports timescale` and
    its tests (the session-level memory model could not pin the memory down: 2 of 16 mice). Notebook `13_timescale`
    is replaced by `13_adaptation_speed` (Fig 3.6 from one switch run per cohort). The synthetic switch run holds
    the blocked and the daily animals.
- **Training sets saved beside the networks** (agreed 17 November 2026). `sc-sbi-train` writes each network's
  simulations to `snpe_<rep>_<model>_<dist>_training.npz` (`inference/training_sets.py`: plain numpy, `theta` in
  the parameters' own units, `x` as simulated, and a JSON `meta` with the configuration, draws and passes) as soon
  as they are done, before the density estimator trains. Run again with the same settings, a task reads them back
  instead of simulating (`AmortisedSBI.train(training_set=...)`; the network records `training_set_reused`): the
  checkpoint training lacked. A set made with other settings (model, prior and its bounds, accuracy rule,
  schedule, N, T, burn-in, representation, statistics, design features, number kept, seed) is refused before
  anything is simulated; `--overwrite` simulates afresh and replaces it. A set that cannot be written is a
  warning, not a failure. The sets also give the model classifier the networks' own training distribution
  without simulating again. `sc-smoke` checks every set and retrains one network from its set. Archive a network
  with its set (`snpe_*`).
- **Training structure for the retrain** (agreed 16 November 2026). What changes for the networks:
  - *Prior:* every model fit uses one prior variant, `settings.SBI_PRIOR = 'learning'`: the networks, the grid
    search grid, synthetic cohorts, calibration draws. A network records its variant (`prior` in the file; a file
    without one is a `manuscript` network) and is conditioned, calibrated and checked on its own ranges.
  - *Flat space:* BE's two rates are log-uniform under `learning`. The density estimator is trained on their
    logarithms (`models/priors.py`: `to_flat` / `from_flat` / `flat_bounds`); samples and estimates come out in
    the parameters' own units. Calibration results and the recovery tables are in flat space (new `scale` column).
  - *Accuracy rule:* the pooled network trains only on simulations whose every session reaches the session
    filter's minimum accuracy (`accuracy_rule` in `SBI_REPRESENTATIONS`), as every selected real session does.
    Prior draws are simulated until `n_simulations` are kept (BE: about one in seven passes), a draw is given up
    at its first failing session, and calibration draws are filtered the same way. Why: under `manuscript` only
    about 4 % of BE's training simulations were where the animals are.
  - *Representations:* `SBI_TRAIN_REPS = ('pooled', 'single')`. Pooled: 2–5 sessions (was 2–6; no animal has 6)
    of 250–600 trials (was 250–500; two animals average more than 500 a session), 100,000 kept. Single: one
    session of 100–600 trials (its held-out test conditions on half-sessions), 200,000, no accuracy rule (for one
    session the rule is readable from the accuracy statistic, and half-sessions are not selected by it).
    Moments is no longer trained: its between-session variances rest on 2–5 sessions, and under `learning` it
    covered 10 (SC) and 3 (BE) of 19 animals against 15 and 13 for pooled.
  - *Consensus:* grid search and pooled SBI vote; the single-session network is listed beside them
    (`SBI-single-UM_be` / `_sc` / `_lower` / `_lower_pvalue`) without a vote, and the summary prints every
    method's mean held-out error. `sc-sbi-sbc` covers pooled at 5 / 3 / 2 sessions and single at 1 (8 tasks).
  - *Grid search:* the full grid spans the prior in use (`grid_from_prior`: same 8,000 points per model,
    log-spaced for BE's rates, `sigma_percep` to 0.5); `--grid manuscript` keeps the old grid.
  - *Training loop:* `AmortisedSBI.simulate_training_set` draws and simulates in rounds with one thread per
    worker (the job's thread count used to reach every worker), shuts the worker pool down before the density
    estimator trains, and the log gives draws, passed, kept and the simulation and training times separately.
    The training set does not depend on the number of workers.
- `slurm/sbi_train.sh`: time limit 4 days (was 2; the partition allows 10). The pooled BE network's upper
  estimate is 30 h.
- **`sc-smoke` / `sbatch slurm/smoke.sh`** (`cli/smoke.py`): the whole chain, small, in a scratch folder (its own
  data and results roots), with a check after each stage; ends with `SMOKE PASSED` or the failed check. For use
  after a change to the fitting code and before submitting real jobs.
- `sc-make-synthetic-cohort`: truths come from the prior in use (`--prior` names another), recorded in
  `generation`; the candidate limit per animal under the accuracy rule went from 50 to 1,000 (BE passes rarely).
- `sc-sbi-range`: the default prior is the one in use; acceptance rates are read only from posteriors of networks
  trained on the prior being shown.
- `preset_min_accuracy` moved to `data/cohort.py`; `accuracy_rule(rep, distribution)` is in
  `inference/simulator.py`.
- **Prior variants** (`models/priors.py`, `docs/prior_variants.md`): `manuscript` (the ranges the first networks
  were trained on) and `learning` (the same four parameters with wider learning ranges: BE `eta_learning` 0.005
  to 0.9 and `eta_relax` 0.001 to 0.4, drawn uniformly in the logarithm; SC `gamma` 0.30 to 0.999). The wider
  ranges come from a scan of the simulator, recorded in the doc.
  Why: `sc-sbi-range` put 9 of 19 expert mice outside both models under the trained ranges (Uniform, pooled).
- **`sc-sbi-range --prior {manuscript,learning,all}`**: the range tables under a prior variant, written to
  `<run>/range/<D>/<prior>/` (a table made before the variants is in `<run>/range/<D>/`: move it into
  `manuscript/`). New per prior: `recognised.csv`, the share of each model's draws that a classifier trained on
  other draws of both models calls that model (every draw; draws passing the accuracy rule). `comparison.csv`
  and `comparison_animals.csv` beside the variants put them side by side: how many animals each prior covers
  per model, how well each model's draws are recognised, and which models cover each animal.
  Result when added (19 expert mice, Uniform, pooled; `learning` from 1,000 draws per model, `manuscript` from
  2,000): animals covered by BE / SC / neither were 4 / 6 / 9 under `manuscript` and 13 / 15 / 3 under
  `learning`; BE's passing draws were recognised 0.92 and 0.59. The wider ranges cover the animals, and where
  the animals are, a slowly learning BE and SC give the same statistics.
- **Tried and removed: a side bias and a lapse** shared by both models
  (`P(B) = (1 - lapse) * P_model(B at s_hat - bias) + lapse / 2`), with the prior variants `nuisance` and `wide`.
  They did not bring the animals inside the models (BE / SC / neither: 1 / 5 / 14 on the trained ranges, 4 / 14 /
  5 on the wider ones) and the lapse cost BE accuracy it cannot spare. `docs/prior_variants.md` keeps the
  table. The models are unchanged from before; `tests/unit/test_models.py` now pins their choices.
- **`sc-sbi-range`** (`inference/prior_range.py`, `cli/sbi_range.py`): where each animal's statistics fall in
  what BE and SC simulate at the training configuration. Per representation and model it draws parameters from
  the prior (2,000 by default), simulates them exactly as the training simulator does (a test keeps the two
  equal), and writes `<run>/range/<D>/<prior>/`: `simulated.csv` (quantiles per feature), `animals.csv` (each animal's
  value, its percentile, outside the central 99 % or not) and `summary.csv` (features outside, the worst one,
  `nearest_percentile`, and the saved posterior's acceptance rate when the run has it). Two reference sets: every
  prior draw, and the draws whose every session reaches the session filter's minimum accuracy. No torch and no
  network needed: results saved on the cluster are read on a machine without torch (their torch objects are
  skipped; the acceptance rate is a plain number).
  Why: on Uniform, about 4 % of BE prior draws reach a pooled accuracy above 0.70 and none above 0.80 (150
  draws), while the animals are selected at accuracy above 0.70; the table shows, per animal, whether the
  networks were trained near it.
- **Equivalence in the pipeline**: `sc-reports opto-contrasts` writes `equivalence.csv` per animal: the `within`
  contrast (trials unit) against margins set in advance (`settings.EQUIVALENCE_MARGINS`, recorded in `meta.json`
  under `settings.margins`), with the verdict within margin / effect / unclear
  (`behaviour.contrasts.equivalence_verdict`). Notebook 20 §4b reads the table (`nb_setup.load_equivalence`)
  where it used to compute the verdicts itself. The margins and the rule are unchanged; a run made before this
  has no table, and the notebook says so.
- **Every run records its environment**: `paths.start_run` writes `<run>/environment.txt`, each installed package
  with its version, when it creates a run. SETUP §7 says how to commit the cluster environment for final runs.
- **CI runs the SBI tests**: a second job installs CPU-only torch and sbi and runs `tests/fit`, which the first
  job could only skip. `features/feature_diagnostics.py` gets a small test (`tests/unit/test_feature_diagnostics.py`).
- **Session intervals described as rough**: the generated results README and `behaviour/contrasts.py` no longer
  call the session-unit interval the honest one. It is the right unit for anything that varies by session, but
  with three to six sessions per set an interval meant to cover 95 % covers roughly 80 to 90 % (simulated with
  no true difference: the interval excluded zero in 18 % of runs at three sessions per set, 10 % at six, 8 % at
  ten). No number changes.

- **behav_utils 1.1.0** (pyproject: `behav_utils[store] @ …@v1.1.0`). One change moves numbers: the session
  bootstrap now draws the sessions two phases share once for both, so a contrast within the same sessions is
  paired.
  - `within` and `within_masking` (laser-on − laser-off in the same sessions) and the light-artefact contrasts:
    the session-unit interval is narrower, because session-to-session variation common to both trial sets now
    cancels. `dod` and the site interaction are differences of those, so their session-unit intervals narrow too.
  - `between`, `compensation` and `vs_ppc` compare different sessions: still drawn separately, from a different
    random stream, so their session-unit intervals move by resampling noise only.
  - Trial-unit intervals, permutation p-values and every observed value are unchanged.
  - From fewer than about ten sessions a session-unit interval is too narrow (behav_utils docs, *Resampling
    units*); the session sets here are small (the Hard phase has three sessions per distribution and session
    type), so read those intervals as rough.
  - `tests/e2e/reference/e2e_contrasts.csv` regenerated: its ten session-unit rows changed, the ten trial-unit
    rows did not. Runs made before this have the old session-unit intervals; rerun `sc-reports battery` for
    current ones.
  - behav_utils 1.1.0 needs pandas 2.1 or later and numpy 1.24.1 or later.

- **Every contrast resamples with its own seed** (`behaviour/contrasts.py`: `SEED` + `SEED_OFFSET`; the
  light-artefact sets: `CONTRAST_SEED` + the set's position). Until now `within` and `within_masking` used the
  same seed, so when the laser and masking sets had the same number of sessions (three and three on each Hard
  distribution) every draw took the same session positions in both, and `dod` paired the k-th laser session with
  the k-th masking session. The light-artefact site interaction (`ppc_sham − alm`) had the same coupling.
  - `within` and the first light-artefact set (`ppc_opto`) keep their seed and their numbers. `within_masking`,
    `between`, `compensation`, `vs_ppc` and the other light-artefact sets get new seeds: their intervals and p-values
    move by resampling noise, and `dod` and the site interaction move with them. Observed values do not change.
  - `ppc_contrasts` and `alm_contrasts` take `seed=`.
  - The e2e reference is regenerated for this too.

- **Docs for behav_utils 1.0**: SETUP (the `store` extra, and troubleshooting for what the move ran into:
  an environment without 1.0, `python` resolving to the shared miniconda, a session filed under the wrong
  animal, a drained `cpu` partition, `sbatch --wrap` running `sh`), START_HERE (strict loading; the rig's
  `Correct` in the early stages), LLM_CONTEXT (1.0's traps: public imports, session ids and trial keys, the
  task's score, the store, the BE/SC models' draws), DAILY_TRACKER, README, runs.md.

- **BE and SC as behav_utils models** (`models/simulate.py`): `BoundaryEstimation` and
  `StatisticalCategorisation` subclass `behav_utils.models.Model`, so behav_utils' `simulate`,
  `simulate_responses` and synthetic generators run them; `make_model('be' | 'sc' | ModelType, params)` builds
  one. They stay in this project (behav_utils is model-agnostic) and wrap the unchanged cores. A run draws one
  seed from its rng and is then `simulate_choices` with it, so every simulation is what it was: the same SBI
  training sets, synthetic cohorts and calibration runs, and the trained networks stay valid (checked value
  for value; tests pin it).
  `laser_blocks_update=True` stops learning on laser trials (off by default: the laser is ignored, as before).
  - `inference.simulator.simulate_sessions` runs each session through behav_utils' `simulate` on a `Design`.
  - `data.stimuli.RigDistribution('Uniform' | 'Hard-A' | 'Hard-B')`: the rig's distributions as a behav_utils
    `StimulusDistribution`, for `make_design` and the synthetic generators.
  - The cores' `make_simulator`, written for 0.10's `generate_synthetic_animal(simulator=...)`, is gone.
- **The snapshot is a behav_utils store** (format 4): `<data root>/behaviour/snapshots/sound_cat_store/`, a folder
  of Parquet tables and JSON, written by `sc-export-snapshot` and copied by `sc-fetch snapshot`. Pickles
  (formats 2 and 3) are refused: export again on the cluster.
  - The store keeps every loaded trial with its stimulus check (`trials.extra['stimulus_check']`), and
    `load_snapshot` removes the trials whose sound was never sent. Removal waits until loading because a store
    rebuilds the lag-1 view from the trials it holds: removed at export, the trial after a silent one would get
    the trial before it as its predecessor instead of no history. A loaded store gives the same trials and history
    as the CSV path (`load_experiment_any` without a snapshot); a test checks this.
  - `snapshot.attach_stimulus_check` / `apply_stimulus_check` split what `remove_silent_trials` does (it still does
    both, for the CSV path); `read_snapshot_metadata` reads what the export recorded without loading trials.
  - The export prints, and records as `changes_since_0_10`, where behav_utils 1.0 can give another number than
    0.10 did on these data: sessions found but not loaded, trials on the boundary or with a NaN stimulus, trials
    after one without a score, and trials whose recorded Correct disagrees with the task's score.
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
