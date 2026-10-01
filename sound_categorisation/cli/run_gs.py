#!/usr/bin/env python3
"""
Grid-search model identification — synthetic or real data, one file.

Three entry points share the per-seed unit (_gs_seed):
  - run_gs_cohort(): serial, all seeds in-process, writes FINAL pickles.
                     Used by the notebook QUICK run.
  - main() --task-id: one (animal, model, seed) per SLURM array task, writes a
                      PARTIAL. The full cluster run.
  - main() --gather:  concatenate partials into the FINAL neutral pickle.

Output: <results root>/model_identification/<cohort>/<run_id>/grid_search/<fit_target>/<distribution>/
(``paths.model_id_dir``; run_sbi writes ``sbi/...`` into the same run, consensus reads both).
  finals:   {animal}_{model}.pkl
  partials: partials/{animal}_{model}_seed{seed}.pkl
<cohort> is the synthetic cohort name, or --label for real data (default 'real').
All finals are written via save_cv_result (neutral cross-method schema), so
load_cv_results reads quick, full, synthetic and real identically. Each result's
metadata stamps model, distribution and fit_target.

--distribution is required: fit ONE phase per launch (its sessions + its output
subdir). For real data the sessions default to expert_<distribution> unless
--preset is given; for synthetic the --cohort name should encode the phase.

Cluster usage (real, one phase; repeat for hard_a and hard_b). The run id is created once and
shared by every array task — ``slurm/submit.sh gs ...`` does all of this:
    RUN=$(sc-new-run --report model_identification --cohort real)
    bash slurm/submit.sh gs --source real --distribution uniform --fit-target update_matrix --run-id $RUN
    sc-run-gs --source real --distribution uniform --fit-target update_matrix --run-id $RUN --gather

Locally: omit --run-id (a new run is created) and --task-id (all seeds run in-process).
--fast: the tiny grid and FAST_GS_N_SEEDS, run id suffixed _fast. --coarse: the coarse grid.
Synthetic: --source synthetic --cohort <name> (name should encode the phase).
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from sound_categorisation.data.cohort import load_animals
from sound_categorisation.data.paths import build_metadata, model_id_dir, start_run
from sound_categorisation.inference.cv_utils import save_cv_result
from sound_categorisation.inference.grid_search import (
    COARSE_GRID,
    DEFAULT_GRID,
    FAST_GRID,
    compute_grid_search_cv,
)
from sound_categorisation.settings import (
    BASE_SEED,
    DISTRIBUTIONS,
    FAST_GS_N_SEEDS,
    FIT_TARGETS,
    GS_BURN_IN,
    GS_N_BINS,
    GS_N_FOLDS,
    SYNTH_GS_N_SEEDS,
)

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


def _save(out_path, animal_id, model, results, fit_target, true_model,
          true_params, distribution):
    save_cv_result(
        out_path, animal_id, model, results, fit_target,
        true_model=true_model, true_params=true_params,
        metadata=build_metadata(
            'run_gs.py',
            {'model': model, 'distribution': distribution,
             'n_results': len(results), 'fit_target': fit_target},
        ),
    )


def run_gs_partial(record, model, seed, out_dir, grid, fit_target, distribution, **kw):
    """Cluster array task: one seed -> a partial pickle under partials/."""
    result = _gs_seed(record, model, seed, grid, fit_target, **kw)
    out_path = Path(out_dir) / 'partials' / f'{record.animal_id}_{model}_seed{seed}.pkl'
    _save(out_path, record.animal_id, model, [result], fit_target,
          record.true_model, record.true_params, distribution)
    return out_path


def run_gs_cohort(records, out_dir, n_seeds, fit_target, distribution, coarse=True,
                  grid=None, models=MODELS, base_seed=BASE_SEED, **kw):
    """Notebook QUICK run: all seeds in-process -> FINAL pickle per (animal, model).

    ``distribution`` (uniform / hard_a / hard_b) is stamped into each result's
    metadata; it must match the phase the sessions were selected for. ``grid``
    overrides the grid explicitly (e.g. FAST_GRID); otherwise COARSE/DEFAULT by
    ``coarse``.
    """
    grid_set = grid or (COARSE_GRID if coarse else DEFAULT_GRID)
    out_paths = []
    for record in records:
        for model in models:
            results = [
                _gs_seed(record, model, base_seed + s, grid_set[model], fit_target, **kw)
                for s in range(1, n_seeds + 1)
            ]
            out_path = Path(out_dir) / f'{record.animal_id}_{model}.pkl'
            _save(out_path, record.animal_id, model, results, fit_target,
                  record.true_model, record.true_params, distribution)
            errs = [r['test_error'] for r in results if not np.isnan(r['test_error'])]
            mean = np.mean(errs) if errs else np.nan
            print(f'  {record.animal_id} / {model}: mean_error={mean:.5f} '
                  f'({len(errs)}/{n_seeds}) -> {out_path.name}')
            out_paths.append(out_path)
    return out_paths


def gather_results(out_dir, distribution):
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
              g['true_model'], g['true_params'], distribution)
        out_paths.append(out_path)

    print(f'Gathered {n_partials} partials -> {len(out_paths)} finals in {out_dir}')
    return out_paths


def _decode_task(task_id, n_animals, n_seeds):
    """Flat SLURM array index -> (animal_idx, model, seed_idx); see sound_categorisation.inference.tasks.gs_grid."""
    from sound_categorisation.inference.tasks import gs_grid
    d = gs_grid(list(range(n_animals)), n_seeds, MODELS).decode(task_id)
    return d['animal'], d['model'], d['seed']


def main():
    p = argparse.ArgumentParser(description='GS model identification (synthetic or real)')
    p.add_argument('--source', required=True, choices=['synthetic', 'real'])
    p.add_argument('--cohort', default=None, help='synthetic: cohort name')
    p.add_argument('--label', default=None, help='real: dataset label for the output dir')
    p.add_argument('--distribution', default=None, choices=list(DISTRIBUTIONS),
                   help='Phase to fit (required). Selects the sessions (real: '
                        'expert_<distribution> unless --preset) and the output '
                        'subdir, matching run_sbi. Fit one phase per launch.')
    p.add_argument('--config', default=None, help='real: config.yaml path')
    p.add_argument('--preset', default=None,
                   help='real: session-selection preset '
                        '(default expert_<distribution>).')
    p.add_argument('--run-id', default=None, help='existing run id to write into (required with --task-id)')
    p.add_argument('--fast', action='store_true', help='tiny grid + FAST_GS_N_SEEDS; run id gets _fast')
    p.add_argument('--coarse', action='store_true', help='coarse grid instead of the full grid')
    p.add_argument('--fit-target', required=True, choices=list(FIT_TARGETS))
    p.add_argument('--task-id', type=int, default=None, help='SLURM array task id')
    p.add_argument('--print-array', action='store_true', help='print the SLURM --array range and exit')
    p.add_argument('--gather', action='store_true', help='combine partials into finals')
    p.add_argument('--count', action='store_true',
                   help='print the array size (n_animals*n_models*n_seeds) and exit')
    p.add_argument('--n-seeds', type=int, default=None)
    args = p.parse_args()

    if not args.distribution:
        p.error('--distribution is required (uniform / hard_a / hard_b) — '
                'fit one phase per launch.')
    preset = args.preset or f'expert_{args.distribution}'

    label = args.cohort if args.source == 'synthetic' else (args.label or 'real')
    if args.task_id is not None and not args.run_id:
        p.error('--task-id needs --run-id (every array task must write into the same run)')
    n_seeds = args.n_seeds or (FAST_GS_N_SEEDS if args.fast else SYNTH_GS_N_SEEDS)
    coarse = args.coarse

    if args.gather:
        if not args.run_id:
            p.error('--gather needs --run-id')
        run = start_run('model_identification', label, args.run_id, fast=args.fast)
        gather_results(model_id_dir(run, 'grid_search', args.fit_target, args.distribution), args.distribution)
        return

    records = load_animals(args.source, cohort=args.cohort,
                           config_path=args.config, preset=preset)

    if args.count:
        print(len(records) * len(MODELS) * n_seeds)
        return
    if args.print_array:
        from sound_categorisation.inference.tasks import gs_grid
        print(gs_grid([r.animal_id for r in records], n_seeds, MODELS).slurm_range())
        return

    run = start_run('model_identification', label, args.run_id, fast=args.fast)
    out_dir = model_id_dir(run, 'grid_search', args.fit_target, args.distribution)
    grid_set = (FAST_GRID if args.fast
                else (COARSE_GRID if coarse else DEFAULT_GRID))
    ng = (len(grid_set['BE'].sigma_percep_values) *
          len(grid_set['BE'].A_repulsion_values) *
          len(grid_set['BE'].model_param1_values) *
          len(grid_set['BE'].model_param2_values))
    print(f'=== GS [{run.name}] {args.source}/{label} phase={args.distribution} preset={preset} '
          f'/ {args.fit_target} | grid={ng} pts x {n_seeds} seeds ===')
    print(f'  {len(records)} animals x {len(MODELS)} models x {n_seeds} seeds, '
          f'grid={"fast" if args.fast else ("coarse" if coarse else "full")}')
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
        path = run_gs_partial(record, model, seed, out_dir, grid_set[model],
                              args.fit_target, args.distribution)
        print(f'  task {args.task_id} -> {record.animal_id}/{model}/seed{seed} -> {path.name}')
    else:
        run_gs_cohort(records, out_dir, n_seeds, args.fit_target,
                      args.distribution, coarse=coarse, grid=grid_set)
    print(f'  Done in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
