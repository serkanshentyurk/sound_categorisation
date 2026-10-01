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
│   ├── settings.py                   constants: distributions, model types, fit sizes, thresholds, seeds
│   ├── data/                         cohort (experiment, genotypes, session sets, AnimalRecord),
│   │                                 snapshot (export/load), stimuli (Hard-A/B densities, normative PSE), paths
│   ├── behaviour/                    contrasts (OptoContrasts, ppc/alm_contrasts), adaptation (switches, trajectory)
│   ├── models/                       BE and SC generative models, perception, simulation, traces
│   ├── inference/                    amortised SBI, simulator, representation, selection, grid_search,
│   │                                 cv_utils, fold_utils, consensus, tasks (TaskGrid for every SLURM array)
│   ├── features/                     SBI feature-selection diagnostics
│   ├── reports/                      compute → tables → figures → PDF → summary (+ README, selftest, CLI)
│   ├── plotting/                     project plotters (CV, opto swarms, assignment, SBI diagnostics)
│   └── cli/                          entry points, installed as sc-* commands: reports, export_snapshot,
│                                     make_synthetic_cohort, new_run, train_sbi, run_sbi, run_gs, consensus
├── slurm/                            job scripts + submit.sh (array size derived from tasks.py)
├── notebooks/                        exploration; shared_setup.py (paths, load_data)
├── tests/                            project tests (+ tests/reference: pinned report numbers)
├── docs/                             runs.md (every command and where it writes), results_guide.md
└── config.yaml                       cohorts, column mappings, session_presets, session_types
```

## Quick start

```bash
pip install -e behav_utils/ && pip install -e ".[dev]"     # both packages (see SETUP.md for the data); installs the sc-* commands
pytest behav_utils/tests -q && pytest tests -q
sc-reports selftest                                          # synthetic end-to-end (opto + switches + summary)
sc-reports battery                                           # the real opto battery (hours)
```

Every run writes under `results/<report>/<cohort>/<run_id>/` (`opto_contrasts`, `switch_adaptation`,
`model_identification`; run id = date + git SHA; `latest` points at the newest run): tidy CSV tables +
`meta.json` next to every PDF, a four-page `summary.pdf`, and a generated `README.md` describing every
column and page. `docs/runs.md` lists every command, what it computes and where it writes.

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
