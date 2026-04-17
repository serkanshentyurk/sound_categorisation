# SLURM Jobs

Cluster submission scripts for the SWC cluster (`ssh.swc.ucl.ac.uk`).

## Environment

All jobs source `env_setup.sh` which:
- Loads miniconda
- Activates the `sound_categorisation` conda environment
- Sets `MPLBACKEND=Agg` (no X display on compute nodes)

## Jobs

### Infrastructure (one-off)

- `train_snpe.sh` — trains one amortised SNPE. GPU partition. Submit per
  (model × distribution) combination: 2 jobs for uniform-only lab meeting.

- `synthetic_generate.sh` — generates synthetic cohorts once. CPU, short.

### Real data

- `real_gs_uniform.sh` — array job, one task per (animal × model × fit_target).
  Each task runs all 64 seeds internally (so tasks take ~1-2 hours).

### Synthetic validation

- `synth_gs.sh` — array job, GS on synthetic cohorts (40 animals × 2 models × 2 targets).

### Dynamic SBI

- `sbi_dynamic.sh` — array job, per-animal RandomWalk SBI. Runs after main pipeline.

## Conventions

- Every job first verifies the script works with `--smoke-test` before running
  at full scale. Run that check locally:
  ```bash
  bash scripts/test_all.sh
  ```
- Outputs go to `results/` (paths in `scripts/config.py`).
- Logs go to `results/logs/slurm_{jobname}_{jobid}.log`.
- Jobs use `--dependency=afterok:<jobid>` where order matters (e.g. SNPE
  training must finish before SBI conditioning).

## Resource notes

- GPU jobs: request 1 GPU via `--gres=gpu:1`. SWC cluster partition is
  `gpu` (confirm with IT).
- CPU-bound array tasks: `-c 8` typical, `--time=0-3:00` usually enough.
- Memory: ~4-8 GB per task for grid search; more for SBI training.

## Submitting

```bash
# Pipeline:
sbatch slurm/synthetic_generate.sh            # 1 job
sbatch slurm/train_snpe.sh BE uniform          # 1 job
sbatch slurm/train_snpe.sh SC uniform          # 1 job
sbatch slurm/real_gs_uniform.sh                # array job
sbatch slurm/synth_gs.sh                       # array job
sbatch slurm/sbi_dynamic.sh                    # array job (after main pipeline)
```
