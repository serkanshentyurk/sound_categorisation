# Runs — every command, what it computes, where it writes

Where to run each command, and the one-time settings, are in [START_HERE.md](../START_HERE.md). All results go
under one root (`paths.results_root()`: `<repo>/results` on a laptop, `$BEHAV_DATA_DIR/Processed/results` on
the cluster, `SC_RESULTS_ROOT` to override) as

```
<results root>/<report>/<cohort>/<run_id>/...
<results root>/<report>/<cohort>/latest          → the newest run (symlink; latest.txt holds the id)
```

`run_id` = `YYYY-MM-DD_HHMM_<git sha7>`, plus `_fast` for reduced runs and an optional label. Every folder a
command writes carries a `meta.json` (run id, argv, snapshot, config, settings, package versions, git sha
and dirty flag). Inputs that are not results — the snapshot, synthetic cohorts, trained SBI networks — live
under `paths.data_root()` (`<repo>/../../data` on a laptop, `$BEHAV_DATA_DIR/Processed` on the cluster,
`SC_DATA_ROOT` to override).

Three kinds of run, three homes:

| kind | purpose | lives in | writes to |
|---|---|---|---|
| tests | prove the code works | `tests/{unit,e2e,fit}` (the library's suite lives in its own repo) | pytest `tmp_path` only |
| validation | prove the method works on synthetic ground truth | the same commands, on a `synthetic_*` cohort | `model_identification/synthetic_*/<run_id>/` |
| analysis | real cohorts | `sc-*` commands, `slurm/` | `<report>/<cohort>/<run_id>/` |

## Everyday

| command | does | where |
|---|---|---|
| `load-env` | goes to the repository and loads the cluster environment (`slurm/env.sh`); a `~/.bashrc` alias, see START_HERE.md | cluster |
| `sc-update [--keep-behav-utils]` | `git pull`, then `pip install -e .`, then prints the versions; `--keep-behav-utils` installs every dependency except `behav_utils`, so an editable copy stays | both |
| `sc-fetch snapshot \| networks \| synthetic` | copies that folder from `$BEHAV_DATA_DIR/Processed` on ceph to the same place under `data_root()` | laptop |
| `sc-fetch results <report>/<cohort>[/<run_id>]` | copies results from ceph to the same place under `results_root()` | laptop |
| `sc-daily-tracker [--days N] [--cohort C] [--no-copy] [--no-open] [--dry-run]` | how each animal is doing, from the session files on ceph (no snapshot): copies the shown animals' `Trial_Summary` and `Session_Parameters` to `<data root>/raw_mirror/`, writes `<results root>/daily_tracker/daily_tracker_<date>.html` and opens it; numbers in `config.yaml`, block `daily_tracker` | laptop or rig PC |

`sc-fetch` uses rsync over ssh as `$SWC_USER`, through `ssh.swc.ucl.ac.uk` to `hpc-gw2` (or the login node
in `$SWC_NODE`, e.g. `hpc-gw1` when `hpc-gw2` does not answer); `--dry-run` lists
what it would copy.

## Behaviour reports (laptop; `sc-reports`)

| command | computes | inputs | writes | time |
|---|---|---|---|---|
| `sc-reports opto-contrasts --distribution D [--trial-class opto\|post_opto] [--site ppc\|alm_uni\|alm_bi] [--level animal\|group\|both] [--jobs N]` | the opto contrasts of one condition: per-animal `contrasts.csv`, `levels.csv` (every stat's value per condition), `trajectory*.csv`, `readouts.npz`, PDFs; the WT-vs-HET fold (`group_rows.csv`, `group_tests.csv`) | snapshot, `config.yaml` | `opto_contrasts/<cohort>/<run_id>/<D>/<design>[_<site>]_<trial_class>/{<animal>/, group/, pdf/}` | minutes per animal (full), seconds (`--fast`) |
| `sc-reports opto-contrasts --all` | every site × distribution × trial class, into one run | as above | as above, every condition | hours |
| `sc-reports summary [--run latest\|<id>]` | the four summary pages + generated `README.md`/`GUIDE.md` from a run's tables (no recomputation) | an opto run | `<run>/summary.pdf`, `summary_*.png`, `README.md` | seconds |
| `sc-reports battery [--jobs N] [--limit N] [--fast] [--skip S…] [--only S]` | stages `check` (one-animal fast structure check, its own `_fast` run) → `opto-contrasts --all` → `light-artefact` (Uniform) → `summary`; `--jobs N` computes N animals at once in `opto-contrasts` (same numbers); `--fast` makes it a pipe-clean of the whole sequence; `--only` resumes one stage | snapshot | one run per stage; `latest` ends on the full ones | about 6 h with `--jobs 1`, roughly that divided by N |
| `sc-reports light-artefact [--distribution Uniform] [--fast]` | light-on − light-off on every light-carrying session set (`ppc_opto`, `ppc_sham`, `alm_uni`, `alm_bi`), with `criterion`/`dprime`; P(B) per stimulus bin on vs off; `ppc_sham − alm` site test; WT-vs-HET fold | snapshot | `light_artefact/<cohort>/<run_id>/<distribution>/{<animal>/, group/, pdf/}` + `README.md` | minutes |
| `sc-reports switch-adaptation --cohort C` | block-level adaptation after each distribution switch: switches, pre/post, convergence (manuscript Fig. 5C recipe), session dynamics, psychometrics; per-animal and summary PDFs | snapshot | `switch_adaptation/<cohort>/<run_id>/switches/` | minutes |

Common options: `--cohort` (default `opto1-cohort`), `--snapshot`, `--config`, `--root`, `--run-id`
(write into an existing run), `--fast` (few draws, scalar stats, no readouts), `--limit N`, `--animals`.

## Data

| command | computes | inputs | writes | time |
|---|---|---|---|---|
| `sc-export-snapshot [--check-only]` | the pickled experiment every analysis loads (run it on the cluster) | the session CSVs in `$BEHAV_DATA_DIR/Raw`, via `config.yaml` | `<data root>/behaviour/snapshots/sound_cat_snapshot.pkl` | minutes |
| `sc-make-synthetic-cohort --distribution D [--name N] [--n-per-model K]` | a synthetic cohort with known BE/SC ground truth | — | `<data root>/synthetic_cohorts/<name>.pkl` | seconds |
| `sc-new-run --report R --cohort C [--fast] [--label L]` | an empty run directory (+ `logs/`), prints the id | — | `<R>/<C>/<run_id>/` | instant |
| `sc-make-synthetic-run --results DIR --data DIR [--no-model-id]` | synthetic opto, switch and model-identification runs in the real layout, for the notebooks and CI | — | `DIR/opto_contrasts/synthetic/…`, `switch_adaptation/synthetic/…`, `light_artefact/synthetic/…`, `model_identification/synthetic_uniform/…` | minutes |

## Feature selection (laptop; `sc-feature-selection`)

| command | computes | inputs | writes | time |
|---|---|---|---|---|
| `sc-feature-selection [--distribution uniform] [--stats …] [--n-sims N] [--n-trials T] [--scorer hgb] [--fast]` | which summary stats the SBI networks should see: per-stat BE-vs-SC CV AUC, per-stat per-parameter CV R², stat–stat correlations, a forward-selection curve and a proposed set; the simulated cohort is saved so every table and plot can be remade | the models (prior draws, one long session per simulated animal) | `feature_selection/<distribution>/<run_id>/{identity_auc, recovery_r2*, stat_correlation_*, forward_selection}.{csv,png}`, `proposed_stats.txt`, `cohort.pkl`, `meta.json` | ~15–30 min (full), ~2 min (`--fast`) |

`SBI_STATS` (`inference/constants.py`) is set by hand from these outputs; the networks are trained on that vector.

## Model identification (cluster; one phase per launch, one run id per chain)

| command | computes | inputs | writes | time |
|---|---|---|---|---|
| `sc-sbi-train` / `slurm/submit.sh sbi-train` | one amortised network per (rep, model, distribution): `TRAIN_GRID` = `SBI_TRAIN_REPS` × models × `SBI_TRAIN_DISTRIBUTIONS` (now pooled/moments × BE/SC × uniform = 4 tasks; widen the settings tuples for more), trained on 2–6 sessions with the number of sessions and valid trials as inputs (`SBI_DESIGN_FEATURES`); refuses to replace a network trained with a different representation (`--overwrite`) | the models, `SBI_STATS` | `<data root>/snpe_networks/snpe_<rep>_<model>_<dist>.pkl` (+ `logs/`) | ≤ 16 h pooled, ≤ 30 h moments per task |
| `sc-grid-search --cohort C --distribution D --fit-target T --run-id ID [--task-id i]` / `submit.sh grid-search` | grid-search CV per (animal, model, seed) → partials | snapshot or cohort | `model_identification/<cohort>/<ID>/grid_search/<T>/<D>/partials/` | minutes–hours per task |
| `sc-grid-search ... --run-id ID --gather` | partials → one final per (animal, model) | the partials | `.../grid_search/<T>/<D>/<animal>_<model>.pkl` | seconds |
| `sc-sbi-condition --cohort C --distribution D --run-id ID [--task-id i]` / `submit.sh sbi-condition` | condition the phase-matched networks on every animal → held-out MSE per CV repeat plus the saved all-session posterior (`SBI_N_POSTERIOR_SAMPLES_SAVED` samples, key `posterior`), `CONDITION_GRID` (4 tasks) | networks, snapshot or cohort | `model_identification/<cohort>/<ID>/sbi/<T>/<D>/<rep>/<animal>_<model>.pkl` | minutes per task |
| `sc-sbi-sbc --cohort C --distribution D --run-id ID [--task-id i] [--n-draws N]` | calibration of each network at 5 sessions (an animal's data) and at 3 and 2 (the CV halves): `SBC_DRAWS_BY_SESSIONS` fresh prior draws, conditioned like animals → SBC ranks, KS p-values, coverage at 50/80/90/95 %, and per draw the posterior mean, SD, quantiles and acceptance (`SBC_GRID`, 12 tasks; seeds from `VALIDATION_SEED`) | networks | `<run>/sbc/<D>/n<N>/<rep>/<model>.pkl`, `<model>_coverage.csv` | ≤ 4 h per task |
| `sc-sbi-check --cohort C --run-id ID [--networks-dir DIR] [--real-run C/ID]` / `submit.sh sbi-check` | every network in a directory against the current settings and code: stored config, parameter order, trained prior, and a fingerprint (feature medians of 5,000 fresh simulations against those stored at training); with `--real-run`, that run's sessions against what the current code selects | networks (and the snapshot for `--real-run`) | `<run>/network_check/<label>.csv`, `sessions_<cohort>_<id>_<D>.csv` | ≤ 2 h (one 20-core job) |
| `sc-validation-tables --cohort C --run-id ID [--real C/ID] [--october C/ID]` | every thesis table of a validation run: SBC summary, coverage, ECDF bands, parameter recovery, model recovery per voter and consensus under both rules with intervals, contested calls, grid search by true σ_percep, the real animals' σ_percep and calls | the run's SBC and model-identification results | `<run>/tables/*.csv` | minutes (laptop or login node) |
| `sc-consensus --cohort C --distribution D [--run-id ID\|latest]` | GS-UM + SBI-pooled-UM + SBI-moments-UM calls → one BE/SC call per animal | the finals of one run | `model_identification/<cohort>/<ID>/consensus/<D>/{assignments.csv, summary.txt, meta.json}` | seconds |

Real cohorts: sessions come from the `model_id_<D>` preset (`config.yaml`: regular sessions only, the last 5 that pass
accuracy > 0.7 and ≥ 250 valid trials), read from the snapshot; an animal with fewer than `MODEL_ID_MIN_SESSIONS` (4) is
left out. Every runner writes `<run>/sessions/<D>.csv` listing each animal's sessions (kept and dropped). Grid-search
seeds: 16 for every cohort, so a synthetic validation run repeats the real procedure; `--n-seeds N` changes it (pass it
through `submit.sh`). Each repeat and seed keeps its per-fold errors (`fold_errors`). The BE/SC call follows the
manuscript (see ARCHITECTURE.md): the consensus calls the model with more significant votes (at least one) and flags a
call made with a significant vote against it (`contested`); `--min-votes 2` is the stricter rule. With ground truth,
`sc-consensus` reports how many animals were called and how many of those were right.

## SBI validation (cluster, then laptop)

Thesis-standard checks of the SBI model identification, all in one run of a synthetic cohort shaped like the real
animals (`sc-make-synthetic-cohort --shape-from all-mice/<id> --accuracy-rule`: each synthetic animal copies a real
animal's session count and lengths, and a draw is kept only if every session passes the `model_id_<D>` accuracy
threshold). The order: archive old networks, train, make the cohort, then the grid search (the long part; it does not
need the networks), SBI conditioning, `sc-consensus`, `sc-sbi-sbc`, `sc-sbi-check` (current and archived networks), and
finally `sc-validation-tables`. Notebook 91 reads `<run>/tables/` and `<run>/network_check/`. The real cohort gets a
fresh run with the same commands, so every thesis number comes from one code version.

`--fast` on `sc-grid-search` / `sc-sbi-condition` / `sc-sbi-train` / `sc-sbi-sbc` uses the tiny grid / few repeats / few simulations / few draws and
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
