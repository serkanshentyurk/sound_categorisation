# Sound categorisation: PPC and statistical model updating

PhD project, Akrami Lab (Sainsbury Wellcome Centre, UCL). Head-fixed mice categorise sounds (2-AFC); the
stimulus distribution is switched between Uniform, Hard-A and Hard-B; posterior parietal cortex (PPC) is
silenced optogenetically (VGAT-ChR2-EYFP, 30 % of trials, bilateral). The question is whether PPC is needed
while the animal updates its statistical model of the sounds, and not once that model fits.

**New here? Read [START_HERE.md](START_HERE.md) first.** It covers the three repositories, your data
folder, the everyday routines and who to ask.

This repository is the analysis. It builds on [`behav_utils`](https://github.com/serkanshentyurk/behav_utils),
a general 2-AFC library installed from a tagged release; the experiments run from the `behaviour_rig`
repository.

## Layout

```
sound_categorisation/                 ← repo root (pyproject.toml, config.yaml)
├── sound_categorisation/             project package
│   ├── settings.py                   constants: distributions, model types, fit sizes, thresholds, seeds
│   ├── data/                         cohort (experiment, genotypes, session sets, AnimalRecord),
│   │                                 snapshot (export/load), stimuli (Hard-A/B densities, normative PSE), paths
│   ├── behaviour/                    contrasts (OptoContrasts, ppc/alm_contrasts), adaptation (switches, trajectory)
│   ├── models/                       BE and SC generative models, perception, simulation, traces
│   ├── inference/                    amortised SBI, simulator, representation, selection, grid_search,
│   │                                 cv_utils, fold_utils, consensus, tasks (TaskGrid for every SLURM array)
│   ├── features/                     SBI feature-selection diagnostics
│   ├── reports/                      compute → tables → figures → PDF → summary (+ README, CLI)
│   ├── plotting/                     project plotters (CV, opto swarms, assignment, SBI diagnostics)
│   └── cli/                          the sc-* commands (export_snapshot, update, fetch, new_run, sbi_train,
│                                     sbi_condition, sbi_sbc, grid_search, consensus, feature_selection, ...)
├── slurm/                            env.sh (the cluster environment), job scripts, submit.sh
├── notebooks/                        the story, one notebook per chapter (notebooks/README.md); nb_setup.py
├── tests/                            unit/ (fast) · e2e/ (report pipeline on synthetic data, pinned reference) · fit/ (torch)
├── docs/                             runs.md (every command and where it writes), results_guide.md
└── config.yaml                       cohorts, column mappings, session_presets, session_types
```

## Quick start

Install: [SETUP.md](SETUP.md). Then, on the laptop:

```bash
sc-fetch snapshot
sc-reports switch-adaptation --cohort behaviour1-cohort
sc-reports battery
```

Every run writes to `results/<report>/<cohort>/<run_id>/`: tidy CSV tables, `meta.json`, PDFs and a
`README.md` that explains every column and page. `latest` points at the newest run.

## Read next

- [START_HERE.md](START_HERE.md): the whole project, from the rig to the results.
- [SETUP.md](SETUP.md): installing, settings, tests.
- [docs/runs.md](docs/runs.md): every command, what it computes, where it writes.
- [docs/results_guide.md](docs/results_guide.md): how to read a results folder.
- [ARCHITECTURE.md](ARCHITECTURE.md): how the code is built.
- [CHANGELOG.md](CHANGELOG.md): what changed when.
- [LLM_CONTEXT.md](LLM_CONTEXT.md): orientation for an AI assistant.
