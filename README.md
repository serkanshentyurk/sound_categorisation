# Sound categorisation — PPC and statistical model updating

PhD project, Akrami Lab (Sainsbury Wellcome Centre, UCL). Head-fixed mice categorise sounds (2-AFC);
the stimulus distribution is switched between Uniform, Hard-A and Hard-B; posterior parietal cortex
(PPC) is silenced optogenetically (VGAT-ChR2-EYFP, 30 % of trials, bilateral). The question is whether
PPC is causally necessary while the animal's statistical model of the task is being updated, and
dispensable once that model is adequate.

This repository holds two things:

- **`behav_utils/`** — a task-agnostic 2-AFC analysis library, pip-installable on its own
  ([its README](behav_utils/README.md)).
- **`sound_categorisation/`** — the project package: cohorts, the opto contrasts, per-session adaptation,
  BE/SC models and inference, the report pipeline.

## Layout

```
sound_categorisation/                 ← repo root (pyproject.toml, config.yaml)
├── behav_utils/                      library: src/behav_utils, tests, docs, own pyproject
├── sound_categorisation/             project package
│   ├── cohort.py                     load the experiment, genotypes, per-animal session sets, AnimalRecord
│   ├── contrasts.py                  the opto contrasts (OptoContrasts, ppc_contrasts, alm_contrasts)
│   ├── adaptation.py                 per-session trajectory + PSE dynamics after a switch
│   ├── stimuli.py                    Hard-A/B densities, normative PSE
│   ├── models/                       BE and SC generative models, simulation, traces
│   ├── inference/                    amortised SBI (types, simulator, representation, selection, amortised)
│   ├── grid_search.py, consensus.py  grid-search CV, GS+SBI consensus
│   ├── cv_utils.py, fold_utils.py    CV result schema, block-aware folds
│   ├── validation/                   SBI feature selection diagnostics
│   ├── tasks.py                      TaskGrid: one task-index convention for every SLURM array
│   ├── paths.py, snapshot.py         data/result locations, run metadata, snapshot export/load
│   ├── plotting/                     project plotters (CV, opto swarms, assignment, SBI diagnostics)
│   └── reports/                      compute → tables → figures → PDF → summary (+ README, selftest)
├── scripts/                          CLIs: export_snapshot, make_smoke_cohort, train_sbi, run_sbi, run_gs, consensus
├── slurm/                            job scripts + submit.sh (array size derived from tasks.py)
├── notebooks/                        exploration; shared_setup.py (paths, load_data)
├── tests/                            project tests (+ tests/reference: pinned report numbers)
├── config.yaml                       cohorts, column mappings, session_presets, session_types
└── run_reports.sh                    the overnight report battery
```

## Quick start

```bash
pip install -e behav_utils/ && pip install -e ".[dev]"     # both packages (see SETUP.md for the data)
pytest behav_utils/tests -q && pytest tests -q
python -m sound_categorisation.reports selftest              # synthetic end-to-end, seconds
bash run_reports.sh                                          # the real battery (hours) → results/reports/
```

Results land in `results/reports/<cohort>/`: tidy CSV tables + `meta.json` per animal and group, PDFs,
a four-page `summary.pdf`, and a generated `README.md` describing every column and page.

## Read next

- [SETUP.md](SETUP.md) — environments, data, snapshot, cluster.
- [ARCHITECTURE.md](ARCHITECTURE.md) — how the project layer is built on the library; the analyses and
  what each contrast means; the model-identification chain.
- [LLM_CONTEXT.md](LLM_CONTEXT.md) — orientation for an AI assistant working on this repo.
- [docs/results_guide.md](docs/results_guide.md) — how to navigate and read a results folder.
- `behav_utils/` — [README](behav_utils/README.md), [ARCHITECTURE](behav_utils/ARCHITECTURE.md),
  [docs/](behav_utils/docs).

## Status (September 2026)

Aim 2, expert phase: analysed (opto1 cohort, 5 HET / 4 WT). Aim 2, post-shift: the daily A/B alternation
did not produce measurable adaptation; a blocked design is needed. Aim 1 (BE/SC identification via grid
search + SBI) has its pipeline in place; the real-data consensus run is pending. Aim 3 (imaging) not started.
