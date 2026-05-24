#!/usr/bin/env python3
"""
Grid-search model ID on one synthetic animal.

Called as a SLURM array task. Each task processes one
(animal_index, model_type, fit_target) combination, bundling all seeds.

Usage:
    python scripts/validation/run_synth_gs.py \
        --cohort static_uniform --animal-index 0 --model BE \
        --fit-target update_matrix
    python scripts/validation/run_synth_gs.py \
        --cohort static_uniform --animal-index 0 --model BE \
        --fit-target update_matrix --smoke-test

Output:
    results/validation/synth_gs/{cohort}_{fit_target}/synth_{animal_id}_{model}.pkl
"""

import argparse
import pickle
import sys
import time
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.config import (
    SYNTH_GS_N_SEEDS, GS_BURN_IN, GS_N_BINS, GS_N_FOLDS,
    VALIDATION_DIR, SYNTH_COHORTS_DIR, BASE_SEED, FIT_TARGETS,
    SMOKE_GS_N_SEEDS,
    build_metadata,
)


def main():
    parser = argparse.ArgumentParser(description='Synthetic GS validation')
    parser.add_argument('--cohort', required=True,
                        choices=['static_uniform', 'learning_uniform'],
                        help='Which synthetic cohort to load')
    parser.add_argument('--animal-index', required=True, type=int,
                        help='Index into the cohort animal list')
    parser.add_argument('--model', required=True, choices=['BE', 'SC'])
    parser.add_argument('--fit-target', required=True, choices=list(FIT_TARGETS))
    parser.add_argument('--sessions-key', default='sessions',
                        help='Key to get sessions from animal dict')
    parser.add_argument('--n-seeds', type=int, default=None)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    n_seeds = args.n_seeds or (SMOKE_GS_N_SEEDS if args.smoke_test else SYNTH_GS_N_SEEDS)

    # Load cohort
    cohort_path = SYNTH_COHORTS_DIR / f'{args.cohort}.pkl'
    if not cohort_path.exists():
        print(f'Cohort not found: {cohort_path}. Run generate_synthetic_cohort.py first.')
        sys.exit(1)

    with open(cohort_path, 'rb') as f:
        cohort_data = pickle.load(f)
    animals = cohort_data['animals']

    if args.animal_index >= len(animals):
        print(f'Animal index {args.animal_index} out of range (cohort has {len(animals)})')
        sys.exit(1)

    sa = animals[args.animal_index]
    aid = sa['animal_id']
    sessions = sa[args.sessions_key]

    output_dir = VALIDATION_DIR / 'synth_gs' / f'{args.cohort}_{args.fit_target}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'synth_{aid}_{args.model}.pkl'

    print(f'=== Synth GS: {aid} / {args.model} / {args.fit_target} ===')
    print(f'  True model: {sa["true_model"]}')
    print(f'  Sessions:   {len(sessions)}')
    print(f'  Seeds:      {n_seeds}')
    if args.smoke_test:
        print('  ** SMOKE TEST MODE **')

    from analysis.grid_search import compute_grid_search_cv

    t0 = time.time()
    seed_results = []

    for seed in range(1, n_seeds + 1):
        actual_seed = BASE_SEED + seed
        try:
            r = compute_grid_search_cv(
                sessions, args.model,
                n_folds=GS_N_FOLDS, seed=actual_seed,
                burn_in=GS_BURN_IN, n_bins=GS_N_BINS,
                fit_target=args.fit_target,
            )
            seed_results.append({
                'seed': actual_seed,
                'avg_test_error': r['avg_test_error'],
                'best_params_single': r['best_params_single'],
            })
        except Exception as e:
            seed_results.append({
                'seed': actual_seed, 'avg_test_error': np.nan,
                'error_msg': str(e),
            })

    elapsed = time.time() - t0

    valid_errors = [r['avg_test_error'] for r in seed_results
                    if not np.isnan(r.get('avg_test_error', np.nan))]
    mean_error = float(np.mean(valid_errors)) if valid_errors else np.nan

    print(f'  Mean error: {mean_error:.6f} ({len(valid_errors)}/{n_seeds} valid)')
    print(f'  Time: {elapsed:.1f}s')

    save_data = {
        'animal_id': aid,
        'true_model': sa['true_model'],
        'true_params': sa['true_params'],
        'model': args.model,
        'fit_target': args.fit_target,
        'cohort': args.cohort,
        'n_sessions': len(sessions),
        'n_seeds': n_seeds,
        'mean_error': mean_error,
        'results': seed_results,
        'metadata': build_metadata('validation/run_synth_gs.py', vars(args)),
    }
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f'  Saved to {output_path}')


if __name__ == '__main__':
    main()
