# LLM_CONTEXT — sound_categorisation

Read this first if you are an AI assistant (or a new person) working on this repository. It is the
orientation the code cannot give you: what the science is, how the code is layered, the rules that hold,
and where the traps are. The library has its own file: `behav_utils/LLM_CONTEXT.md`.

## The science in six lines
Head-fixed mice categorise sounds (2-AFC). The stimulus distribution is switched (Uniform → Hard-A ↔
Hard-B). PPC is silenced optogenetically on 30 % of trials (VGAT-ChR2-EYFP HET; WT littermates get the
same light with no opsin). Hypothesis: PPC is causally necessary while the animal's statistical model of
the task is being updated, dispensable once it is adequate. Two behavioural strategies are modelled: BE
(boundary estimation) and SC (statistical categorisation). Aims: characterise behaviour and identify the
model (grid search + simulation-based inference); silence PPC in expert and post-shift phases; image PPC.

## What the data say so far (opto1 cohort, Sept 2026) — do not contradict these without new tables
- Expert Uniform, HET: silencing changes nothing on the silenced trial (all stats within ±0.1, bounded CIs).
- Expert Uniform, WT: light alone shifts the criterion toward A in 4/4 animals; not at ALM; no carry-over.
  Silencing removes that effect. Masking sessions (laser at zero) do not reproduce the laser, so the WT
  animals are the light control, not the masking sessions.
- WT compensate: laser-off trials inside laser sessions sit below masking sessions (contrast `compensation`).
- Hard phase: the daily A/B alternation produced no measurable criterion tracking; slow drift dominates;
  the "laser sessions vs masking sessions" contrast is order-confounded. Prediction 2 is untested, not refuted.

## Repository shape
Two packages, imports one way: `sound_categorisation` (project) → `behav_utils` (library). Both are pip
installed (`pip install -e behav_utils/ && pip install -e .`); nothing manipulates `sys.path`. Layout and
module roles: `README.md`, `ARCHITECTURE.md`. Environments, data, cluster: `SETUP.md`.

## Rules that hold
1. The library is task-agnostic. Distribution names, session types, genotypes, normative observers, cohort
   logic and report layouts are project code. If a library change needs the word `masking`, stop.
2. Pipeline: `load → select_sessions → filter_trials → TrialArrays → compute_x → plot_x`. Computes take
   pre-filtered sessions; plots compute nothing.
3. Every compute returns a typed result (Series/DataFrame or a frozen dataclass with `to_rows()`).
   Contrasts are `DeltaStats` / `Interaction`; the project wraps them in `OptoContrasts`.
4. Reports are a pipeline: scripts compute from the snapshot and write tidy tables + `meta.json`; figures
   and summary pages read results/tables; notebooks read `results/reports/`. Do not put a bootstrap in a
   notebook, and do not draw a page from anything but a result or a table.
5. Numbers are pinned: `tests/reference/selftest_contrasts.csv`. A change that moves a number must
   regenerate it deliberately and say so in the commit.
6. Statistics honesty: permutation p only for within-session contrasts (label randomised per trial);
   session-level contrasts get session-bootstrap intervals and no p; genotype rank tests have a floor
   (`min_achievable_p`) — per-animal consistency is the evidence.
7. Working style: agree the design before code for anything above a bug fix; deliver complete files;
   run `ruff check .`, `pytest behav_utils/tests -q`, `pytest tests -q`, and the report selftest; British
   English; say what a change costs and what could go wrong.

## Where to look for
- which sessions an animal has of each type → `cohort.collect_sessions_ppc / _alm`; `config.yaml: session_types`
- what a contrast is → `contrasts.py` docstring; `docs/results_guide.md` in words
- the numbers behind any page → `results/reports/<cohort>/<dist>/<design>_<toi>/<animal>/contrasts.csv`
- per-session adaptation → `adaptation.py` (`Trajectory`); columns explained in the generated results README
- SLURM job counts → `tasks.py`; never hard-code an `--array`
- run provenance → `meta.json` in every results folder (snapshot, config, settings, versions, git SHA)

## Traps
- Uniform ran masking block *then* laser block; Hard ran laser sessions *then* masking sessions. Any
  between-session contrast contains time. The trajectory rows on the Hard pages show the drift.
- Each Hard session is one switch; there is no multi-session block after a switch. Do not analyse
  "trials since switch across sessions".
- `expert_uniform` and other presets exist only after a config/snapshot is loaded; for in-memory data
  call `sound_categorisation.cohort.ensure_presets()`.
- `filter_trials(trial_type='all')` keeps laser trials (and drops aborts, like every trial type).
- Per-session PSE at this cohort's sigma has SE ≈ 0.1; `pse_fixed` (only the criterion free) is steadier.
  `pse_dynamics` τ is not identifiable at ~500 trials; it is in the tables and per-animal PDFs, not on the
  summary pages, for that reason.
- SS12 has no data; SS17 and SS22 have no Hard sessions; SS20 has no Uniform laser sessions.
- The `sound_categorisation.egg-info` / `behav_utils/src/behav_utils.egg-info` folders are build
  artefacts; they must not be tracked.

## Open work
Notebooks (stage 3) await the agreed chapter story; Aim 1 real-data consensus run; next-cohort design
(blocked distributions, interleaved session types, opsin-negative light control at matched power).
