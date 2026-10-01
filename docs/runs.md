# Runs — every command, what it computes, where it writes

All results go under one root (`paths.results_root()`: `<repo>/results` locally, `<data root>/results`
on the cluster, `SC_RESULTS_ROOT` to override) as

```
<results root>/<report>/<cohort>/<run_id>/...
<results root>/<report>/<cohort>/latest          → the newest run (symlink; latest.txt holds the id)
```

`run_id` = `YYYY-MM-DD_<git sha7>`, plus `_fast` for reduced runs and an optional label. Every folder a
command writes carries a `meta.json` (run id, argv, snapshot, config, settings, package versions, git sha
and dirty flag). Inputs that are not results — the snapshot, synthetic cohorts, trained SBI networks — live
under `paths.data_root()` (`<repo>/../../data` locally, ceph on the cluster, `SC_DATA_ROOT` to override).

Three kinds of run, three homes:

| kind | purpose | lives in | writes to |
|---|---|---|---|
| tests | prove the code works | `tests/`, `behav_utils/tests/` | pytest `tmp_path` only |
| validation | prove the method works on synthetic ground truth | the same commands, on a `synthetic_*` cohort | `model_identification/synthetic_*/<run_id>/` |
| analysis | real cohorts | `sc-*` commands, `slurm/` | `<report>/<cohort>/<run_id>/` |

## Behaviour reports (laptop; `sc-reports`)

| command | computes | inputs | writes | time |
|---|---|---|---|---|
| `sc-reports opto --distribution D [--toi opto\|post_opto] [--design ppc\|alm --site uni\|bi] [--level animal\|group\|both]` | the opto contrasts of one condition: per-animal `contrasts.csv`, `trajectory*.csv`, `readouts.npz`, PDFs; the WT-vs-HET fold (`group_rows.csv`, `group_tests.csv`) | snapshot, `config.yaml` | `opto_contrasts/<cohort>/<run_id>/<D>/<design>[_<site>]_<toi>/{<animal>/, group/, pdf/}` | minutes per animal (full), seconds (`--fast`) |
| `sc-reports opto --all` | every design × distribution × toi, into one run | as above | as above, every condition | hours |
| `sc-reports summary [--run latest\|<id>]` | the four summary pages + generated `README.md`/`GUIDE.md` from a run's tables (no recomputation) | an opto run | `<run>/summary.pdf`, `summary_*.png`, `README.md` | seconds |
| `sc-reports battery [--limit N]` | fast one-animal structure check (its own `_fast` run) → `opto --all` → `summary` | snapshot | two runs; `latest` ends on the full one | hours |
| `sc-reports switches --cohort C` | block-level adaptation after each distribution switch: switches, pre/post, convergence (manuscript Fig. 5C recipe), session dynamics, psychometrics; per-animal and summary PDFs | snapshot | `switch_adaptation/<cohort>/<run_id>/switches/` | minutes |
| `sc-reports selftest [--out DIR]` | the whole report pipeline on synthetic data (opto fast + one full condition, summary, switches) | nothing | `DIR` (default a temp dir), same layout as real runs | minutes |

Common options: `--cohort` (default `opto1-cohort`), `--snapshot`, `--config`, `--root`, `--run-id`
(write into an existing run), `--fast` (few draws, scalar stats, no readouts), `--limit N`, `--animals`.

## Data

| command | computes | inputs | writes | time |
|---|---|---|---|---|
| `sc-export-snapshot [--check-only]` | the pickled experiment every analysis loads | Bonsai CSVs via `config.yaml` | `<data root>/snapshots/sound_cat_snapshot.pkl` (`behaviour/snapshots/` on ceph) | minutes |
| `sc-make-synthetic-cohort --distribution D [--name N] [--n-per-model K]` | a synthetic cohort with known BE/SC ground truth | — | `<data root>/synthetic_cohorts/<name>.pkl` | seconds |
| `sc-new-run --report R --cohort C [--fast] [--label L]` | an empty run directory (+ `logs/`), prints the id | — | `<R>/<C>/<run_id>/` | instant |

## Model identification (cluster; one phase per launch, one run id per chain)

| command | computes | inputs | writes | time |
|---|---|---|---|---|
| `sc-train-sbi` / `slurm/submit.sh train` | one amortised network per (rep, model, distribution): `TRAIN_GRID`, 18 tasks | the models | `<data root>/snpe_networks/snpe_<rep>_<model>_<dist>.pkl` (+ `logs/`) | hours per task (GPU-free) |
| `sc-run-gs --source real\|synthetic --distribution D --fit-target T --run-id ID [--task-id i]` / `submit.sh gs` | grid-search CV per (animal, model, seed) → partials | snapshot or cohort | `model_identification/<cohort>/<ID>/grid_search/<T>/<D>/partials/` | minutes–hours per task |
| `sc-run-gs ... --run-id ID --gather` | partials → one final per (animal, model) | the partials | `.../grid_search/<T>/<D>/<animal>_<model>.pkl` | seconds |
| `sc-run-sbi --source ... --distribution D --run-id ID [--task-id i]` / `submit.sh condition` | condition the phase-matched networks on every animal → posterior + held-out MSE, `CONDITION_GRID` (6 tasks) | networks, snapshot or cohort | `model_identification/<cohort>/<ID>/sbi/<T>/<D>/<rep>/<animal>_<model>.pkl` | minutes per task |
| `sc-consensus --cohort C --distribution D [--run-id ID\|latest]` | GS + SBI calls → one BE/SC call per animal | the finals of one run | `model_identification/<cohort>/<ID>/consensus/<D>/{assignments.csv, summary.txt, meta.json}` | seconds |

`--fast` on `run_gs` / `run_sbi` / `train_sbi` uses the tiny grid / few repeats / few simulations and
suffixes the run id with `_fast`. `--coarse` on `run_gs` selects the coarse grid. `submit.sh` creates the
run id when none is given and prints it; Slurm logs go to `<run>/logs/`. `<cohort>` is the synthetic
cohort name, or `--label` (default `real`) for real data.

## What reads what

| consumer | reads |
|---|---|
| `sc-reports summary` | the opto run it is given (`latest` by default) |
| `sc-consensus` | one `model_identification` run, one distribution |
| notebooks (being rewritten) | a run's tables via `paths.resolve_run(report, cohort, 'latest')` |
| `tests/test_reports.py` | nothing on disk; it runs the selftest into `tmp_path` and compares against `tests/reference/` |

## Not yet here

Light-artefact report (choice-by-stimulus on/off, side bias, SDT) — planned as `sc-reports artefact`.
SLDS state assignment — planned under `sound_categorisation/slds/`.
