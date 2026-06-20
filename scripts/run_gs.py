#!/usr/bin/env python3
"""
Grid-search model identification — synthetic or real data, one file.

Three entry points share the per-seed unit (_gs_seed):
  - run_gs_cohort(): serial, all seeds in-process, writes FINAL pickles.
                     Used by the notebook QUICK run.
  - main() --task-id: one (animal, model, seed) per SLURM array task, writes a
                      PARTIAL. The full cluster run.
  - main() --gather:  concatenate partials into the FINAL neutral pickle.

Output dir: grid_search/{run}/{label}_{fit_target}/   (label = cohort or experiment).
  finals:   {animal}_{model}.pkl
  partials: partials/{animal}_{model}_seed{seed}.pkl
All finals are written via save_cv_result (neutral cross-method schema), so
load_cv_results reads quick, full, synthetic and real identically.

Cluster usage (synthetic full run):
    # array upper bound:
    N=$(python scripts/run_gs.py --source synthetic --cohort static_uniform \
            --run full --fit-target update_matrix --count)
    # one task per (animal, model, seed):
    sbatch --array=0-$((N-1)) run_gs.sh   # each task runs:
    python scripts/run_gs.py --source synthetic --cohort static_uniform \
        --run full --fit-target update_matrix --task-id $SLURM_ARRAY_TASK_ID
    # then a single gather job:
    python scripts/run_gs.py --source synthetic --cohort static_uniform \
        --run full --fit-target update_matrix --gather

Real data: same, with --source real --label <name> [--config <path>].
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.config import (
    SYNTH_GS_N_SEEDS, SMOKE_GS_N_SEEDS, GS_BURN_IN, GS_N_BINS, GS_N_FOLDS,
    BASE_SEED, FIT_TARGETS, results_dir, build_metadata,
)
from scripts.providers import load_animals
from analysis.grid_search import compute_grid_search_cv, DEFAULT_GRID, COARSE_GRID
from utils.cv_utils import save_cv_result

MODELS = ('BE', 'SC')


def _gs_seed(record, model, seed, grid, fit_target,
             burn_in=GS_BURN_IN, n_folds=GS_N_FOLDS, n_bins=GS_N_BINS):
    """Run one (animal, model, seed) GS-CV; return a neutral result dict.

    Maps the grid-search compute keys (avg_test_error / best_params_single)
    into the neutral schema (test_error / best_params).
    """
    try:
        r = compute_grid_search_cv(
            record.sessions, model, grid=grid, n_folds=n_folds, seed=seed,
            burn_in=burn_in, n_bins=n_bins, fit_target=fit_target,
        )
        return {'rep': seed, 'test_error': r['avg_test_error'],
                'best_params': r['best_params_single']}
    except Exception as e:
        return {'rep': seed, 'test_error': np.nan,
                'best_params': None, 'error_msg': str(e)}


def _save(out_path, animal_id, model, results, fit_target, true_model, true_params):
    save_cv_result(
        out_path, animal_id, model, results, fit_target,
        true_model=true_model, true_params=true_params,
        metadata=build_metadata(
            'run_gs.py',
            {'model': model, 'n_results': len(results), 'fit_target': fit_target},
        ),
    )


def run_gs_partial(record, model, seed, out_dir, grid, fit_target, **kw):
    """Cluster array task: one seed -> a partial pickle under partials/."""
    result = _gs_seed(record, model, seed, grid, fit_target, **kw)
    out_path = Path(out_dir) / 'partials' / f'{record.animal_id}_{model}_seed{seed}.pkl'
    _save(out_path, record.animal_id, model, [result], fit_target,
          record.true_model, record.true_params)
    return out_path


def run_gs_cohort(records, out_dir, n_seeds, fit_target, coarse=True,
                  models=MODELS, base_seed=BASE_SEED, **kw):
    """Notebook QUICK run: all seeds in-process -> FINAL pickle per (animal, model)."""
    grid_set = COARSE_GRID if coarse else DEFAULT_GRID
    out_paths = []
    for record in records:
        for model in models:
            results = [
                _gs_seed(record, model, base_seed + s, grid_set[model], fit_target, **kw)
                for s in range(1, n_seeds + 1)
            ]
            out_path = Path(out_dir) / f'{record.animal_id}_{model}.pkl'
            _save(out_path, record.animal_id, model, results, fit_target,
                  record.true_model, record.true_params)
            errs = [r['test_error'] for r in results if not np.isnan(r['test_error'])]
            mean = np.mean(errs) if errs else np.nan
            print(f'  {record.animal_id} / {model}: mean_error={mean:.5f} '
                  f'({len(errs)}/{n_seeds}) -> {out_path.name}')
            out_paths.append(out_path)
    return out_paths


def gather_results(out_dir):
    """Concatenate partials/ into FINAL {animal}_{model}.pkl per (animal, model)."""
    out_dir = Path(out_dir)
    pdir = out_dir / 'partials'
    if not pdir.exists():
        print(f'No partials directory at {pdir}')
        return []

    groups = {}
    n_partials = 0
    for pkl in sorted(pdir.glob('*.pkl')):
        with open(pkl, 'rb') as f:
            d = pickle.load(f)
        n_partials += 1
        key = (d['animal_id'], d['model'])
        g = groups.setdefault(key, {
            'results': [], 'true_model': d.get('true_model'),
            'true_params': d.get('true_params'), 'fit_target': d.get('fit_target'),
        })
        g['results'].extend(d['results'])

    out_paths = []
    for (aid, model), g in groups.items():
        g['results'].sort(key=lambda r: r['rep'])
        out_path = out_dir / f'{aid}_{model}.pkl'
        _save(out_path, aid, model, g['results'], g['fit_target'],
              g['true_model'], g['true_params'])
        out_paths.append(out_path)

    print(f'Gathered {n_partials} partials -> {len(out_paths)} finals in {out_dir}')
    return out_paths


def _decode_task(task_id, n_animals, n_seeds):
    """Flat SLURM array index -> (animal_idx, model, seed_idx)."""
    seed_idx = task_id % n_seeds
    model_idx = (task_id // n_seeds) % len(MODELS)
    animal_idx = task_id // (n_seeds * len(MODELS))
    return animal_idx, MODELS[model_idx], seed_idx


def main():
    p = argparse.ArgumentParser(description='GS model identification (synthetic or real)')
    p.add_argument('--source', required=True, choices=['synthetic', 'real'])
    p.add_argument('--cohort', default=None, help='synthetic: cohort name')
    p.add_argument('--label', default=None, help='real: dataset label for the output dir')
    p.add_argument('--config', default=None, help='real: config.yaml path')
    p.add_argument('--run', choices=['quick', 'full'], default='full')
    p.add_argument('--fit-target', required=True, choices=list(FIT_TARGETS))
    p.add_argument('--task-id', type=int, default=None, help='SLURM array task id')
    p.add_argument('--gather', action='store_true', help='combine partials into finals')
    p.add_argument('--count', action='store_true',
                   help='print the array size (n_animals*n_models*n_seeds) and exit')
    p.add_argument('--n-seeds', type=int, default=None)
    p.add_argument('--smoke-test', action='store_true')
    args = p.parse_args()

    label = args.cohort if args.source == 'synthetic' else (args.label or 'real')
    out_dir = results_dir('grid_search', args.run, label, args.fit_target)
    coarse = args.run == 'quick'
    n_seeds = args.n_seeds or (SMOKE_GS_N_SEEDS if args.smoke_test else SYNTH_GS_N_SEEDS)

    if args.gather:
        gather_results(out_dir)
        return

    records = load_animals(args.source, cohort=args.cohort, config_path=args.config)

    if args.count:
        print(len(records) * len(MODELS) * n_seeds)
        return

    grid_set = COARSE_GRID if coarse else DEFAULT_GRID
    print(f'=== GS [{args.run}] {args.source}/{label} / {args.fit_target} ===')
    print(f'  {len(records)} animals x {len(MODELS)} models x {n_seeds} seeds, '
          f'grid={"coarse" if coarse else "full"}')
    print(f'  out={out_dir}')

    t0 = time.time()
    if args.task_id is not None:
        total = len(records) * len(MODELS) * n_seeds
        if args.task_id >= total:
            print(f'task-id {args.task_id} out of range (total {total})')
            sys.exit(1)
        animal_idx, model, seed_idx = _decode_task(args.task_id, len(records), n_seeds)
        record = records[animal_idx]
        seed = BASE_SEED + seed_idx + 1
        path = run_gs_partial(record, model, seed, out_dir, grid_set[model], args.fit_target)
        print(f'  task {args.task_id} -> {record.animal_id}/{model}/seed{seed} -> {path.name}')
    else:
        run_gs_cohort(records, out_dir, n_seeds, args.fit_target, coarse=coarse)
    print(f'  Done in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
