# Architecture — sound_categorisation

## Two packages, one direction

```
sound_categorisation  (project)  ──imports──▶  behav_utils  (library)
```
The library never imports the project. Everything that names a distribution, a session type, a genotype,
a normative observer or a cohort lives in the project. Everything that could apply to any 2-AFC task lives
in the library. When in doubt: if a new function needs the word `masking` or `Hard-A`, it is project code.

## The project layer

```
settings.py          constants: DISTRIBUTIONS, MODEL_TYPES, FIT_TARGETS, GS_*/SBI_*/SYNTH_*/SMOKE_* sizes,
                     expert-selection thresholds, BASE_SEED
data/
  cohort.py          load_experiment_any (snapshot or CSV), gather_genotypes, ensure_presets,
                     collect_sessions_ppc / collect_sessions_alm (the per-animal session sets a design compares),
                     AnimalRecord + load_animals (real or synthetic animals for model identification)
  snapshot.py        export/load the pickled experiment; re-applies session types and presets from config
  stimuli.py         Hard-A / Hard-B densities, normative PSE at a given sigma, stimulus sampling
  paths.py           data/results roots (laptop vs cluster), result-dir naming, build_metadata, project config
behaviour/
  contrasts.py       OptoContrasts = the contrasts of the opto design, as typed library results:
                       within         opto sessions:    laser-on − laser-off trials     (permutation p valid)
                       within_masking masking sessions: flagged − unflagged trials      (null control)
                       between        opto sessions − masking sessions, all trials      (bootstrap only)
                       compensation   laser-off trials of opto sessions − masking sessions (bootstrap only)
                       dod            within − within_masking (Interaction)
                       vs_ppc         ALM sessions − PPC opto sessions (ALM design)
                     STATS / STATS_RT / SENSITIVITY / BIAS: which statistics each design reports
  adaptation.py      compute_trajectory(animal, distributions) -> Trajectory: one row per session in order;
                     compute_switches / SwitchResult: block-level adaptation after a distribution switch
models/              BE (boundary-estimation) and SC (statistical-categorisation) generative models
inference/           amortised SBI (types, simulator, representation, selection, amortised),
                     grid_search (CV over model parameters, held-out UM / conditional-psychometric MSE),
                     cv_utils (CV result schema), fold_utils (block-aware folds), consensus (GS + SBI votes),
                     tasks (TaskGrid: TRAIN_GRID, CONDITION_GRID, gs_grid)
features/            feature_diagnostics: which summary statistics carry BE/SC identity and parameters
reports/             compute.py  AnimalResult / GroupResult / Settings (no matplotlib)
                     tables.py   to_tables, group_tables, readout_arrays, write_result / read_result
                     figures.py  one function per page; pdf.py  per-animal and group PDFs
                     summary.py  the four-page summary; switches.py  the switch-adaptation report
                     readme.py   the generated results README; selftest.py synthetic ExperimentData
                     cli.py      `python -m sound_categorisation.reports` / `sc-reports`
plotting/            project plotters (CV, opto swarms, assignment, SBI diagnostics)
cli/                 entry points (pyproject [project.scripts]): sc-export-snapshot, sc-make-smoke-cohort,
                     sc-train-sbi, sc-run-sbi, sc-run-gs, sc-consensus
```

## The report pipeline

```
snapshot ──▶ compute_animal ──▶ AnimalResult ──▶ to_tables ──▶ contrasts.csv, trajectory.csv, readouts.npz, meta.json
                    │                                              │
                    └──▶ animal_pdf                                └──▶ summary (reads tables only)
         compute_group  ──▶ GroupResult   ──▶ group_tables ──▶ group_rows.csv, group_tests.csv
```
Scripts compute deterministically from the snapshot and write tidy tables with metadata (snapshot,
config, settings, package versions, git SHA). Figures are drawn from results or tables, never computed
in place. Notebooks are meant to read `results/reports/` and narrate; a notebook that runs a bootstrap is
a smell. `tests/reference/selftest_contrasts.csv` pins every number the fast selftest produces; a change
that moves a number fails `tests/test_reports.py` until the reference is regenerated on purpose.

Resampling units: within-session contrasts use the trial bootstrap and a permutation p (the laser was
randomised per trial); between-session contrasts use the session bootstrap and no p (session type was
assigned per session, and masking sessions were recorded after laser sessions — an order confound the
pages state). Genotype comparisons are rank tests on per-animal effects; with 4 vs 5 animals the floor is
p = 0.016, so per-animal consistency carries the argument.

## The model-identification chain

```
sc-train-sbi    TRAIN_GRID (18 tasks)   simulate BE/SC → train one amortised network per (rep, model, distribution)
sc-run-sbi      CONDITION_GRID (6)      condition each animal's expert data → posterior + held-out MSE
sc-run-gs       gs_grid(animals, seeds) grid search CV per (animal, model, seed) → partials → --gather
sc-consensus                            GS + SBI calls → assignments.csv, summary.txt
slurm/submit.sh {train|condition|gs}        derives --array from the grids
```
`load_animals(source)` in `data/cohort.py` gives the same `AnimalRecord` shape for real and synthetic animals,
so the runners are identical downstream; synthetic cohorts carry ground truth for validation.

## Design as executed (opto1 cohort)
Expert Uniform: a masking block then a laser block per animal, plus ALM (uni/bi) sessions. Hard phase:
six laser sessions alternating Hard-A/Hard-B daily, then six masking sessions with the same alternation.
Each Hard session is therefore one switch, and the trajectory is per session, not per block.

## Notebooks (planned numbering)
`00` data & task · `10` expert behaviour · `11` adaptation · `20` opto Uniform (PPC + masking + ALM) ·
`21` opto Hard (+ session trajectory) · `30` model-identification methods · `31` model-identification
results · `40` opto × model. Each reads the report tables; one single-animal walk-through cell per
notebook shows the library pipeline explicitly.

## Conventions
- Verbs: `load_`, `select_`, `filter_`, `compute_`, `plot_`; results are typed; plotters draw only.
- CSV/JSON on disk for anything a human or notebook reads; pickle only for the snapshot and networks.
- British English; line length 110; `ruff` clean; CI runs both suites and the report selftest.
