# Changelog — sound_categorisation

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
