# LLM_CONTEXT — sound_categorisation

Read this first if you are an AI assistant working on this repository; a new person should start with
START_HERE.md. This is the orientation the code cannot give you: what the science is, how the code is
layered, the rules that hold, and where the traps are. The library lives in its own repository
(github.com/serkanshentyurk/behav_utils) and has its own LLM_CONTEXT.md there.

## The science in six lines
Head-fixed mice categorise sounds (2-AFC). The stimulus distribution is switched (Uniform → Hard-A ↔
Hard-B). PPC is silenced optogenetically on 30 % of trials (VGAT-ChR2-EYFP HET; WT littermates get the
same light with no opsin). Hypothesis: PPC is causally necessary while the animal's statistical model of
the task is being updated, dispensable once it is adequate. Two behavioural strategies are modelled: BE
(boundary estimation) and SC (statistical categorisation). Aims: characterise behaviour and identify the
model (grid search + simulation-based inference); silence PPC in expert and post-shift phases; image PPC.

## What the data say — do not contradict these without new tables

Each group names the runs it comes from; the numbers live in those runs' tables.

Opto battery of 4–5 November 2026. Its conclusions stand in the rerun of 19 November
(`opto_contrasts/opto1-cohort/2026-11-19_0041_ac1c40a`, behav_utils 1.1.0), whose tables hold the final
numbers. Medians are WT vs HET (4 vs 5 animals on Uniform, 3 vs 4 on Hard); `min_p` is the smallest p those
sizes allow.

- Expert Uniform, HET: no large effect of silencing; median within-session effects near zero (`mu` +0.006,
  accuracy +0.003). Equivalence (notebook 20, §4b; 95 % intervals inside ±0.10 for `mu`/`sigma`, ±0.05 for the
  rest, set in advance): within margin in 2 of 5 HET animals for `mu`, accuracy, easy accuracy and the lapses;
  unclear in all 5 for hard accuracy and the history statistics (intervals wider than the margin). Side bias
  shifts on laser trials in 4 of 5 HET animals (median −0.05), the same direction as the WT light effect
  (−0.12, 4 of 4), so likely the light. Prediction 1 is consistent with the data, not shown as a null.
- Expert Uniform, WT: light alone shifts the criterion toward A, `mu` +0.19 vs +0.01 (p 0.016 = `min_p`: all
  four WT above all five HET); not at ALM (WT criterion about 0.04); masking does not reproduce it (`mu` −0.16
  vs −0.04, p 0.29). The WT animals are the light control, not the masking sessions.
- Compensation (laser-off trials of laser sessions vs the unflagged trials of masking sessions): WT `mu` −0.24
  vs HET +0.06, p 0.11 — the same direction as before the reference was corrected, smaller, not separated.
- ALM in HET: large criterion medians on laser trials (0.94 unilateral, 1.16 bilateral, against 0.04; p
  0.19–0.29), and the laser-off trials of ALM sessions shift too (`mu` +0.42 / +0.33; side bias −0.20
  unilateral, p 0.016). ALM silencing seems to act on the whole session.
- Carry-over: none; every post-opto median near zero, no p below 0.06.
- Hard phase (opto1), the light: the WT light effect has the same direction as on Uniform (`mu` +0.36 Hard-A,
  +0.15 Hard-B, vs about +0.06; p 0.114, `min_p` 0.057).

Adaptation: the switch runs at b194525 (18 November 2026), both cohorts; notebook 13.

- Daily alternation (opto1, the six mice the bias check keeps): the criterion followed partly at most.
  Convergence over trials 100–300 (0 = the criterion before the switch, 1 = the normative one): masking days
  0.41 [−0.04, 0.85], laser days on their laser-off trials 0.24 [−0.15, 0.62].
- Blocks (behaviour1, the five mice the bias check keeps, whole-block fits): one mouse followed every switch by
  about the normative amount, two partly, one until its last block went one-sided, one the wrong way. The whole
  block's share of the normative shift, mean over the mice with a measurable block: Uniform → Hard-B 0.91
  [−0.91, 2.72], Hard-B → Hard-A 0.65 [−0.68, 1.98], Hard-A → Hard-B 0.06 [−1.30, 1.42]. The criterion drifts
  upwards across days, which works against Hard-A → Hard-B, so that switch is the one that tells following
  from drift.
- Which animals the bias check flags: each switch run's `switches/bias.csv` (`biased_animal`). Read it there;
  lists copied into notes have gone stale before.
- Prediction 2 (post-shift opto) is untested: the opto cohort followed its daily alternation partly at most,
  so silencing had little updating to disrupt, and the cohort that followed in blocks had no laser. Untested,
  not refuted.

Model identification, November 2026.

- Noise floor (`sc-noise-floor`, all-mice, 6fd4a93): a model with no trial history scores a median 1.20 times
  the floor between sessions (above 1.35 in 6 of 19 mice) and 1.11 within sessions. At these data sizes the
  held-out update matrix tells a model with history from one without only weakly, so for most mice a BE–SC
  difference in held-out error says little about strategy.
- Silenced-trial memory (`sc-blocked-update`, on the October fits `all-mice/2026-10-23_0951_64a0f89`): no mouse
  is consistent with blocking, and the test rarely decides. Complete blocking would move recency and win-stay
  by about −0.08 under BE and −0.03 under SC (HET medians), against observed 95 % intervals of about ±0.18 to
  ±0.21.

## Repository shape
Two packages, imports one way: `sound_categorisation` (project) → `behav_utils` (library, 1.0). Both are
pip installed (`pip install -e .`, which pulls `behav_utils[store]` from its tagged release and installs the
`sc-*` commands); nothing in the packages manipulates `sys.path`. Import library names from their public
homes (`behav_utils.data`, `.stats`, `.readouts`, `.models`, …), never from module paths below them. Layout
and module roles: `README.md`, `ARCHITECTURE.md`. Installing: `SETUP.md`. Data folders and routines:
`START_HERE.md`.

Paths: every path comes from `data/paths.py`. On the cluster all of them hang off `BEHAV_DATA_DIR` (raw
data in `Raw`, snapshot and networks in `Processed`, results in `Processed/results`); the code stops if it
is not set. `slurm/env.sh` is the cluster environment; `sc-update` pulls and reinstalls; `sc-fetch` copies
cluster folders to the same place on a laptop.

## Rules that hold
1. The library is task-agnostic. Distribution names, session types, genotypes, normative observers, cohort
   logic and report layouts are project code. If a library change needs the word `masking`, stop.
2. Pipeline: `load → select_sessions → filter_trials → TrialArrays → compute_x → plot_x`. Computes take
   pre-filtered sessions; plots compute nothing.
3. Every compute returns a typed result (Series/DataFrame or a frozen dataclass with `to_rows()`).
   Contrasts are `DeltaStats` / `Interaction`; the project wraps them in `OptoContrasts`.
4. Reports are a pipeline: scripts compute from the snapshot and write tidy tables + `meta.json`; figures
   and summary pages read results/tables; notebooks read a run's tables (`paths.resolve_run`). Do not put a bootstrap in a
   notebook, and do not draw a page from anything but a result or a table.
5. Numbers are pinned: `tests/e2e/reference/e2e_contrasts.csv`. A change that moves a number must
   regenerate it deliberately (`pytest tests/e2e --regen-reference`) and say so in the commit.
6. Statistics honesty: permutation p only for within-session contrasts (label randomised per trial);
   session-level contrasts get session-bootstrap intervals and no p; genotype rank tests have a floor
   (`min_achievable_p`) — per-animal consistency is the evidence.
7. Working style: agree the design before code for anything above a bug fix; deliver complete files;
   run `ruff check .`, `pytest tests/unit tests/e2e -q`; British
   English; say what a change costs and what could go wrong.

## Where to look for
- which sessions an animal has of each type → `data/cohort.py: collect_sessions_ppc / _alm`; `config.yaml: session_types`
- what a contrast is → `behaviour/contrasts.py` docstring; `docs/results_guide.md` in words
- the numbers behind any page → `results/opto_contrasts/<cohort>/latest/<dist>/<design>_<trial_class>/<animal>/contrasts.csv`
- which command writes what, and where → `docs/runs.md`; where the data live → `START_HERE.md`
- per-session adaptation → `behaviour/adaptation.py` (`Trajectory`); columns explained in the generated results README
- SLURM job counts → `inference/tasks.py`; never hard-code an `--array`
- run provenance → `meta.json` in every results folder (snapshot, config, settings, versions, git SHA)
- BE/SC as behav_utils models (for `simulate`, `simulate_responses`, the synthetic generators) →
  `models/simulate.py` (`make_model`); the rig's distributions as a `StimulusDistribution` →
  `data/stimuli.RigDistribution`
- what behav_utils 1.0 could change on our data → `sc-export-snapshot`'s table, stored in the snapshot's
  metadata as `changes_since_0_10` (`snapshot.read_snapshot_metadata`)

## Traps
- Uniform ran masking block *then* laser block; Hard ran laser sessions *then* masking sessions. Any
  between-session contrast contains time. The trajectory rows on the Hard pages show the drift.
- Each Hard session is one switch; there is no multi-session block after a switch. Do not analyse
  "trials since switch across sessions".
- `expert_uniform` and other presets exist only after a config/snapshot is loaded; for in-memory data
  call `sound_categorisation.data.cohort.ensure_presets()`.
- `filter_trials(trial_type='all')` keeps laser trials (and drops aborts, like every trial type).
- Loading is strict (behav_utils 1.0): a session that fails to load raises, and so does an `Animal_ID`
  column naming another animal than the folder (SS21's 4 July 2026 run is in `sessions_to_ignore` for that).
  `strict=False` warns instead, but look at the files first: a skipped session shifts every later
  `session_idx`. A session whose only file has fewer than `min_trials_per_file` (10) rows is not loaded.
- Session ids are `<animal>_<YYYYMMDD>` (the date from the folder name); the folder's name is
  `SessionData.source_id`, its path `SessionData.source` (a folder, not a CSV). A trial key is a hash of
  experimenter, animal and session id with the trial number in its low bits: `project.experimenter` is
  empty; set it before aligning neural data on `trial_key`, never after.
- The task scores: `TrialData.correct` / `TrialArrays.correct` are its score, `TrialData.outcome` holds
  outcome codes, and the rig's labels and `Correct` column are kept as `outcome_recorded` /
  `correct_recorded` (the daily tracker counts no-responses and early licks from those labels). In the early
  training stages the rig's `Correct` means something else; never analyse `correct_recorded`.
  `prev_correct` is NaN after a trial without a score. `fit_update_matrix` takes the score as `correct=`
  (simulated choices: `grid_search.score_choices`).
- The snapshot is a behav_utils store (Parquet), format 4; pickles are refused. It keeps every trial with its
  stimulus check, and loading removes the silent ones, because a store rebuilds the lag-1 view from the
  trials it holds. Session types and `sessions_to_ignore` are applied when the data are loaded, so after
  editing them export again.
- A `BoundaryEstimation` / `StatisticalCategorisation` run equals `simulate_choices` with one seed drawn from
  the run's rng; keep it so (tests pin it), or the trained SBI networks no longer match their simulator.
- `animal.session_table` is counts + accuracy, for picking sessions; per-session statistics come from
  `compute_phase_stats(..., per_session=True)`. `compare_groups` returns a `GroupComparison`
  (`.test(stat)`, `.table()`), not a dict.
- Per-session PSE at this cohort's sigma has SE ≈ 0.1; `pse_fixed` (only the criterion free) is steadier.
  `pse_dynamics` τ is not identifiable at ~500 trials; it is in the tables and per-animal PDFs, not on the
  summary pages, for that reason.
- SS12 has no data; SS17 and SS22 have no Hard sessions; SS20 has no Uniform laser sessions.
- The Hard-A/B hard side is the rig's truncated exponential (rate 2.15329, `data/stimuli.py`), not the
  legacy density; normative PSEs depend on it.
- `post_opto` is compared with the other laser-off trials; post-opto trials are laser-off trials, and
  `behav_utils` refuses to permute phases that share trials.
- The model-ID p-values come from seeds that vary simulation noise on a fixed data split (the manuscript's
  recipe); they say the call is stable under that noise, not that the data favour it.
- Data before 5 October 2026 carry rig faults (START_HERE.md, caveats); they are kept and stated. About 1 %
  of Hard trials may still be silent (the unchecked end of each run, and 13 Hard sessions with no check).
- `*.egg-info`, `build/`, `__pycache__` are build artefacts; they must not be tracked.

## Open work (19 November 2026)
Aim 1: the October 2026 all-mice run (12 BE, 3 SC, 4 unclear; descriptive, notebook 11) is superseded: its
networks were trained on 4–6 sessions under the manuscript's prior and conditioned on 2–3-session CV halves.
The networks are being retrained, Uniform only, under the wider `learning` prior (`settings.SBI_PRIOR`) with the
number of sessions and valid trials as inputs (`SBI_DESIGN_FEATURES`): pooled (2–5 sessions of 250–600 trials,
accuracy rule), which votes with the grid search, and single-session, reported beside it as exploratory
per-session fits (`SBI_TRAIN_REPS`). On them the all-mice run (`all-mice/2026-11-17_2256_195866b`) and an SBI
validation run on 100 BE + 100 SC synthetic mice shaped like the real ones (`synthetic_uniform_learning_k100`:
real-run seeds and repeats, SBC, network check; notebook 91) are redone. Primary consensus rule: more
significant votes (≥ 1), contested calls flagged; the two-vote rule reported alongside. The Hard-block and
daily-alternation predictions from the Uniform fits are built (`sc-hard-predictions`), not yet run on real fits.
Still to do: the per-animal model classifier, disRNN.
Optional: the learning phase (state-space models). Not planned: Hard-distribution and moments networks (both
still train by name, e.g. `sc-sbi-train --rep moments`).
Next-cohort design (blocked distributions, interleaved session types, opsin-negative light control at
matched power).
Neural data (Aim 3): align on `TrialArrays.trial_key`; the alignment layer is project code, not library.
