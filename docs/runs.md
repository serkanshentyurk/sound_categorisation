# Runs — every command, what it computes, where it writes

All results go under one root (`paths.results_root()`: `<repo>/results` locally, `<data root>/results`
on the cluster, `SC_RESULTS_ROOT` to override) as

```
<results root>/<report>/<cohort>/<run_id>/...
<results root>/<report>/<cohort>/latest          → the newest run (symlink; latest.txt holds the id)
```

`run_id` = `YYYY-MM-DD_HHMM_<git sha7>`, plus `_fast` for reduced runs and an optional label. Every folder a
command writes carries a `meta.json` (run id, argv, snapshot, config, settings, package versions, git sha
and dirty flag). Inputs that are not results — the snapshot, synthetic cohorts, trained SBI networks — live
under `paths.data_root()` (`<repo>/../../data` locally, ceph on the cluster, `SC_DATA_ROOT` to override).

Three kinds of run, three homes:

| kind | purpose | lives in | writes to |
|---|---|---|---|
| tests | prove the code works | `tests/{unit,e2e,fit}`, `behav_utils/tests/` | pytest `tmp_path` only |
| validation | prove the method works on synthetic ground truth | the same commands, on a `synthetic_*` cohort | `model_identification/synthetic_*/<run_id>/` |
| analysis | real cohorts | `sc-*` commands, `slurm/` | `<report>/<cohort>/<run_id>/` |

## Behaviour reports (laptop; `sc-reports`)

| command | computes | inputs | writes | time |
|---|---|---|---|---|
| `sc-reports opto-contrasts --distribution D [--trial-class opto\|post_opto] [--site ppc\|alm_uni\|alm_bi] [--level animal\|group\|both]` | the opto contrasts of one condition: per-animal `contrasts.csv`, `levels.csv` (every stat's value per condition), `trajectory*.csv`, `readouts.npz`, PDFs; the WT-vs-HET fold (`group_rows.csv`, `group_tests.csv`) | snapshot, `config.yaml` | `opto_contrasts/<cohort>/<run_id>/<D>/<design>[_<site>]_<trial_class>/{<animal>/, group/, pdf/}` | minutes per animal (full), seconds (`--fast`) |
| `sc-reports opto-contrasts --all` | every site × distribution × trial class, into one run | as above | as above, every condition | hours |
| `sc-reports summary [--run latest\|<id>]` | the four summary pages + generated `README.md`/`GUIDE.md` from a run's tables (no recomputation) | an opto run | `<run>/summary.pdf`, `summary_*.png`, `README.md` | seconds |
| `sc-reports battery [--limit N] [--fast] [--skip S…] [--only S]` | stages `check` (one-animal fast structure check, its own `_fast` run) → `opto-contrasts --all` → `light-artefact` (Uniform) → `summary`; `--fast` makes it a pipe-clean of the whole sequence; `--only` resumes one stage | snapshot | one run per stage; `latest` ends on the full ones | hours |
| `sc-reports light-artefact [--distribution Uniform] [--fast]` | light-on − light-off on every light-carrying session set (`ppc_opto`, `ppc_sham`, `alm_uni`, `alm_bi`), with `criterion`/`dprime`; P(B) per stimulus bin on vs off; `ppc_sham − alm` site test; WT-vs-HET fold | snapshot | `light_artefact/<cohort>/<run_id>/<distribution>/{<animal>/, group/, pdf/}` + `README.md` | minutes |
| `sc-reports switch-adaptation --cohort C` | block-level adaptation after each distribution switch: switches, pre/post, convergence (manuscript Fig. 5C recipe), session dynamics, psychometrics; per-animal and summary PDFs | snapshot | `switch_adaptation/<cohort>/<run_id>/switches/` | minutes |

Common options: `--cohort` (default `opto1-cohort`), `--snapshot`, `--config`, `--root`, `--run-id`
(write into an existing run), `--fast` (few draws, scalar stats, no readouts), `--limit N`, `--animals`.

## Data

| command | computes | inputs | writes | time |
|---|---|---|---|---|
| `sc-export-snapshot [--check-only]` | the pickled experiment every analysis loads | Bonsai CSVs via `config.yaml` | `<data root>/snapshots/sound_cat_snapshot.pkl` (`behaviour/snapshots/` on ceph) | minutes |
| `sc-make-synthetic-cohort --distribution D [--name N] [--n-per-model K]` | a synthetic cohort with known BE/SC ground truth | — | `<data root>/synthetic_cohorts/<name>.pkl` | seconds |
| `sc-new-run --report R --cohort C [--fast] [--label L]` | an empty run directory (+ `logs/`), prints the id | — | `<R>/<C>/<run_id>/` | instant |
| `sc-make-synthetic-run --results DIR --data DIR [--no-model-id]` | synthetic opto, switch and model-identification runs in the real layout, for the notebooks and CI | — | `DIR/opto_contrasts/synthetic/…`, `switch_adaptation/synthetic/…`, `light_artefact/synthetic/…`, `model_identification/synthetic_uniform/…` | minutes |

## Model identification (cluster; one phase per launch, one run id per chain)

| command | computes | inputs | writes | time |
|---|---|---|---|---|
| `sc-sbi-train` / `slurm/submit.sh sbi-train` | one amortised network per (rep, model, distribution): `TRAIN_GRID`, 18 tasks | the models | `<data root>/snpe_networks/snpe_<rep>_<model>_<dist>.pkl` (+ `logs/`) | hours per task (GPU-free) |
| `sc-grid-search --cohort C --distribution D --fit-target T --run-id ID [--task-id i]` / `submit.sh grid-search` | grid-search CV per (animal, model, seed) → partials | snapshot or cohort | `model_identification/<cohort>/<ID>/grid_search/<T>/<D>/partials/` | minutes–hours per task |
| `sc-grid-search ... --run-id ID --gather` | partials → one final per (animal, model) | the partials | `.../grid_search/<T>/<D>/<animal>_<model>.pkl` | seconds |
| `sc-sbi-condition --cohort C --distribution D --run-id ID [--task-id i]` / `submit.sh sbi-condition` | condition the phase-matched networks on every animal → posterior + held-out MSE, `CONDITION_GRID` (6 tasks) | networks, snapshot or cohort | `model_identification/<cohort>/<ID>/sbi/<T>/<D>/<rep>/<animal>_<model>.pkl` | minutes per task |
| `sc-consensus --cohort C --distribution D [--run-id ID\|latest]` | GS + SBI calls → one BE/SC call per animal | the finals of one run | `model_identification/<cohort>/<ID>/consensus/<D>/{assignments.csv, summary.txt, meta.json}` | seconds |

`--fast` on `sc-grid-search` / `sc-sbi-condition` / `sc-sbi-train` uses the tiny grid / few repeats / few simulations and
suffixes the run id with `_fast`. `--coarse` on `sc-grid-search` selects the coarse grid. `submit.sh` creates the
run id when none is given and prints it; Slurm logs go to `<run>/logs/`. `C` is a synthetic cohort
(`sc-make-synthetic-cohort`) or a cohort name from `config.yaml`; the kind is detected from the name.

## What reads what

| consumer | reads |
|---|---|
| `sc-reports summary` | the opto run it is given (`latest` by default) |
| `sc-consensus` | one `model_identification` run, one distribution |
| notebooks | a run's tables via `nb_setup.open_run` → `paths.resolve_run(report, cohort, 'latest')`; `SC_NB_SYNTHETIC=1` reads the synthetic runs |
| `tests/e2e/` | nothing on disk; it builds the synthetic experiment, runs the pipeline into `tmp_path` and compares against `tests/e2e/reference/` |

## Not yet here

SLDS state assignment — planned under `sound_categorisation/slds/`.
