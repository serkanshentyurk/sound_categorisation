#!/usr/bin/env python3
"""
Run grid-search CV for one (animal, model_type, fit_target) combination.
Bundles all seeds internally.

Usage:
    python scripts/run_gs_single.py --animal SS01 --model BE --fit-target update_matrix
    python scripts/run_gs_single.py --animal SS01 --model BE --fit-target update_matrix --smoke-test

Output:
    results/cv/{distribution}_{fit_target}/cv_{animal}_{model}.pkl
"""

import argparse
import pickle
import sys
import time
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.config import (
    GS_N_SEEDS, GS_BURN_IN, GS_N_BINS, GS_N_FOLDS,
    CV_DIR, BASE_SEED, STAGE, FIT_TARGETS,
    SMOKE_GS_N_SEEDS,
    build_metadata, ensure_dirs,
    load_animal_data,
)


def main():
    parser = argparse.ArgumentParser(description='Grid-search CV for one animal')
    parser.add_argument('--animal', required=True, help='Animal ID')
    parser.add_argument('--model', required=True, choices=['BE', 'SC'],
                        help='Model type')
    parser.add_argument('--fit-target', required=True, choices=list(FIT_TARGETS),
                        help='Fitting target: update_matrix or conditional_psych')
    parser.add_argument('--distribution', default='uniform',
                        choices=['uniform', 'hard_a', 'hard_b'])
    parser.add_argument('--n-seeds', type=int, default=None)
    parser.add_argument('--burn-in', type=int, default=GS_BURN_IN)
    parser.add_argument('--n-bins', type=int, default=GS_N_BINS)
    parser.add_argument('--n-folds', type=int, default=GS_N_FOLDS)
    parser.add_argument('--seed-offset', type=int, default=0,
                        help='Offset added to base seed (for SLURM array indexing)')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to project config YAML (default: auto-detect)')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    n_seeds = args.n_seeds or (SMOKE_GS_N_SEEDS if args.smoke_test else GS_N_SEEDS)

    output_dir = Path(args.output_dir) if args.output_dir else (
        CV_DIR / f'{args.distribution}_{args.fit_target}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'cv_{args.animal}_{args.model}.pkl'

    print(f'=== GS CV: {args.animal} / {args.model} / {args.fit_target} ===')
    print(f'  Distribution: {args.distribution}')
    print(f'  Seeds:        {n_seeds}')
    print(f'  Folds:        {args.n_folds}')
    print(f'  Burn-in:      {args.burn_in}')
    print(f'  Output:       {output_path}')
    if args.smoke_test:
        print('  ** SMOKE TEST MODE **')

    # Load real animal data
    from behav_utils.data.selection import select_sessions

    animal = load_animal_data(args.animal, config_path=args.config)
    sessions = select_sessions(animal, f'expert_{args.distribution}')

    if not sessions:
        print(f'  No expert {args.distribution} sessions for {args.animal}. Skipping.')
        # Save empty result so gather script knows this animal was processed
        save_data = {
            'animal': args.animal, 'model': args.model,
            'fit_target': args.fit_target, 'distribution': args.distribution,
            'n_sessions': 0, 'results': [],
            'metadata': build_metadata('run_gs_single.py', vars(args)),
        }
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        return

    print(f'  Sessions: {len(sessions)}')

    from analysis.grid_search import grid_search_cv

    t0 = time.time()
    seed_results = []

    for seed in range(1, n_seeds + 1):
        actual_seed = BASE_SEED + seed + args.seed_offset
        try:
            r = grid_search_cv(
                sessions, args.model,
                n_folds=args.n_folds, seed=actual_seed,
                burn_in=args.burn_in, n_bins=args.n_bins,
                fit_target=args.fit_target,
            )
            seed_results.append({
                'seed': actual_seed,
                'avg_test_error': r['avg_test_error'],
                'test_errors': r['test_errors'],
                'best_params': r['best_params'],
                'best_params_single': r['best_params_single'],
            })
            print(f'    seed {seed}/{n_seeds}: err={r["avg_test_error"]:.6f}')
        except Exception as e:
            print(f'    seed {seed}/{n_seeds}: FAILED ({e})')
            seed_results.append({
                'seed': actual_seed, 'avg_test_error': np.nan,
                'error_msg': str(e),
            })

    elapsed = time.time() - t0

    # Summary
    valid_errors = [r['avg_test_error'] for r in seed_results
                    if not np.isnan(r.get('avg_test_error', np.nan))]
    mean_error = float(np.mean(valid_errors)) if valid_errors else np.nan

    print(f'\n  Mean error ({len(valid_errors)}/{n_seeds} valid): {mean_error:.6f}')
    print(f'  Time: {elapsed:.1f}s')

    # Save
    save_data = {
        'animal': args.animal,
        'model': args.model,
        'fit_target': args.fit_target,
        'distribution': args.distribution,
        'n_sessions': len(sessions),
        'n_seeds': n_seeds,
        'mean_error': mean_error,
        'results': seed_results,
        'metadata': build_metadata('run_gs_single.py', vars(args)),
    }
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)

    print(f'  Saved to {output_path}')


if __name__ == '__main__':
    main()
