# Scripts

Command-line entry points for cluster jobs and local tasks.

## Overview

These scripts are called by SLURM jobs (in `../slurm/`) or run locally.
Each script accepts CLI arguments, loads config from `config.py`, and
writes results to `../results/`.

## Key scripts

- **`config.py`** — shared constants (paths, simulation params, stat lists).
  All scripts import from here; no duplication.

### Cluster scripts

- `train_snpe.py` — train one amortised SNPE (one model × one distribution)
- `run_gs_single.py` — run grid-search CV for one animal × seed × model × target
- `run_sbi_dynamic_randomwalk.py` — per-animal dynamic SBI with RandomWalk link
- `generate_synthetic_cohort.py` — generate synthetic BE/SC animals
- `validation/run_synth_gs.py` — GS on synthetic cohort
- `validation/run_synth_sbi.py` — SBI on synthetic cohort

### Local scripts

- `condition_sbi_local.py` — condition trained SNPE on real animals
- `gather_cv_results.py` — aggregate cluster GS outputs
- `test_all.sh` — run every script with `--smoke-test` to verify the pipeline

## Smoke testing

Every script accepts `--smoke-test`, which uses small values (2 seeds, 500 sims,
2 animals) and completes in seconds. Use before submitting SLURM jobs:

```bash
bash scripts/test_all.sh
```

## Conventions

- Every result is saved alongside a metadata dict (see `config.build_metadata`)
  capturing: timestamp, host, git SHA, library versions, and all config constants.
- Scripts write to paths defined in `config.py` (`RESULTS_DIR`, `SNPE_DIR`, etc.).
- Failed cluster tasks write an error log to `results/logs/` and exit non-zero,
  so SLURM can detect them. The rest of the array job continues.

## Adding a new script

1. Import from `scripts.config` for paths and constants.
2. Add `--smoke-test` flag and honour it via `apply_smoke_test_overrides()`.
3. Save results with `build_metadata()` attached.
4. Add to `test_all.sh`.
