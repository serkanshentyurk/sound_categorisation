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

## What the data say (runs of 4–5 November 2026) — do not contradict these without new tables

Medians are WT vs HET (4 vs 5 animals on Uniform, 3 vs 4 on Hard); `min_p` is the smallest p those sizes allow.

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
- Hard phase (opto1): no tracking of the daily A/B alternation, slow drift; the WT light effect has the same
  direction (`mu` +0.36 Hard-A, +0.15 Hard-B, vs about +0.06; p 0.114, `min_p` 0.057).
- Blocked switches (behaviour1, without the animals the bias check drops: SS01, SS06, SS11): block medians of
  `mu` Hard-A 0.226, Hard-B 0.137, Uniform 0.141, so Hard-A − Hard-B +0.09 against the ideal +0.19 — a partial
  shift, all of it in the Hard-A blocks. The Hard-B blocks came first and the criterion drifts upwards over
  time (seen in opto1), so part of this may be drift. Switch by switch, 10 of 16 switches with a fit move the
  right way, 4 the wrong way (3 of them SS05), 2 not at all. The bias check also drops SS14, SS16 (opto1).
- Prediction 2 (post-shift opto) is untested: the opto cohort did not adapt, and the cohort that adapted
  partially had no laser. Untested, not refuted.

## Repository shape
Two packages, imports one way: `sound_categorisation` (project) → `behav_utils` (library). Both are pip
installed (`pip install -e .`, which pulls `behav_utils` from its tagged release and installs the `sc-*`
commands); nothing in the packages manipulates `sys.path`. Layout and module roles: `README.md`,
`ARCHITECTURE.md`. Installing: `SETUP.md`. Data folders and routines: `START_HERE.md`.

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

## Traps
- Uniform ran masking block *then* laser block; Hard ran laser sessions *then* masking sessions. Any
  between-session contrast contains time. The trajectory rows on the Hard pages show the drift.
- Each Hard session is one switch; there is no multi-session block after a switch. Do not analyse
  "trials since switch across sessions".
- `expert_uniform` and other presets exist only after a config/snapshot is loaded; for in-memory data
  call `sound_categorisation.data.cohort.ensure_presets()`.
- `filter_trials(trial_type='all')` keeps laser trials (and drops aborts, like every trial type).
- `load_experiment` (behav_utils ≥ 0.8) raises on a session that fails to load instead of skipping it
  with a warning; `strict=False` restores the skip, but look at the file first — a skipped session
  shifts every later `session_idx`. `ExperimentData.load` warns when a snapshot was written by another
  library version: re-export (`sc-export-snapshot`) rather than carry it forward.
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
  `behav_utils` ≥ 0.9 refuses to permute phases that share trials.
- The model-ID p-values come from seeds that vary simulation noise on a fixed data split (the manuscript's
  recipe); they say the call is stable under that noise, not that the data favour it.
- Data before 5 October 2026 carry rig faults (START_HERE.md, caveats); they are kept and stated. About 1 %
  of Hard trials may still be silent (the unchecked end of each run, and 13 Hard sessions with no check).
- `*.egg-info`, `build/`, `__pycache__` are build artefacts; they must not be tracked.

## Open work
Aim 1: the all-mice consensus run exists (October 2026: 12 BE, 3 SC, 4 Unclear; descriptive, notebook 11), but
is superseded: its networks were trained on 4–6 sessions and conditioned on 2–3-session CV halves. Nov 2026: networks
retrained with the number of sessions and valid trials as inputs (2–6 sessions), all-mice run redone, and an SBI
validation run (synthetic cohort of 50 BE + 50 SC shaped like the real animals, real-run seeds and repeats; SBC at 5, 3
and 2 sessions; network check; notebook 91). Primary consensus rule: more significant votes (≥ 1), contested calls
flagged; the two-vote rule reported alongside. The learning phase (state-space models) and disRNN are not written;
Hard-distribution and single-session networks are not planned (no question the data can answer needs them).
Next-cohort design (blocked distributions, interleaved session types, opsin-negative light control at
matched power).
Neural data (Aim 3): align on `TrialArrays.trial_key`; the alignment layer is project code, not library.
