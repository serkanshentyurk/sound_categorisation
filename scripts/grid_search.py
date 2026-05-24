#!/usr/bin/env python3
"""
Grid-Search Cross-Validation

Pools sessions, runs 2-fold CV with multiple seeds, returns a
standardised result dict compatible with compare_models.py.

Both a CLI tool and an importable module.

CLI usage:
    python scripts/grid_search.py --animal SS01 --model BE \
        --fit-target update_matrix --distribution uniform

    python scripts/grid_search.py --animal SS01 --model BE \
        --fit-target update_matrix --smoke-test

Importable:
    from scripts.grid_search import run_grid_search

    result = run_grid_search(
        sessions=clean_sessions,
        model_type='BE',
        fit_target='update_matrix',
        animal_id='SS01',
    )
    # result['cv_errors'], result['best_params'], ...

Output schema (shared with sbi.py --mode static):
    {
        'method': 'grid_search',
        'model_type': str,
        'fit_target': str,
        'animal_id': str,
        'distribution': str,

        'cv_errors': list[float],       # per-seed test errors
        'mean_error': float,
        'std_error': float,

        'best_params': dict,            # best grid point (param names → values)
        'all_results': list[dict],      # full grid: [{params, errors, mean_error}]

        'posterior_samples': None,      # GS doesn't produce posteriors
        'param_names': list[str],
        'trajectories': None,

        'n_sessions': int,
        'n_trials': int,
        'metadata': dict,
    }
"""

import argparse
import pickle
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# CORE FUNCTION (importable)
# =============================================================================

def run_grid_search(
    sessions: list,
    model_type: str,
    fit_target: str = 'update_matrix',
    animal_id: str = 'unknown',
    distribution: str = 'uniform',
    n_seeds: int = None,
    n_folds: int = None,
    burn_in: int = None,
    n_bins: int = None,
    base_seed: int = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """
    Run grid-search CV on pooled sessions.

    Args:
        sessions: Pre-filtered session list (from filter_trials).
        model_type: 'BE' or 'SC'.
        fit_target: 'update_matrix' or 'conditional_psych'.
        animal_id: For labelling only.
        distribution: For labelling only.
        n_seeds: Number of random seeds (default from config).
        n_folds: Number of CV folds (default from config).
        burn_in: Burn-in trials for simulation (default from config).
        n_bins: UM bin count (default from config).
        base_seed: Base random seed (default from config).
        smoke_test: Use fast defaults.

    Returns:
        Standardised result dict (see module docstring).
    """
    from scripts.config import (
        GS_N_SEEDS, GS_N_FOLDS, GS_BURN_IN, GS_N_BINS, BASE_SEED,
        SMOKE_GS_N_SEEDS,
    )
    from analysis.grid_search import compute_grid_search_cv
    from behav_utils.data.filtering import pool_arrays

    # Apply defaults
    n_seeds = n_seeds or (SMOKE_GS_N_SEEDS if smoke_test else GS_N_SEEDS)
    n_folds = n_folds or GS_N_FOLDS
    burn_in = burn_in or GS_BURN_IN
    n_bins = n_bins or GS_N_BINS
    base_seed = base_seed or BASE_SEED

    if not sessions:
        raise ValueError(f'No sessions provided for {animal_id}')

    # Count valid trials (for metadata)
    pooled = pool_arrays(sessions)
    no_response = pooled.get('no_response', np.isnan(pooled['choices']))
    n_trials = int((~no_response).sum())

    # Run CV across seeds
    # grid_search_cv takes sessions directly — it handles pooling and
    # block-based fold splitting internally via _sessions_to_arrays()
    all_results = []
    cv_errors = []

    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx

        result = compute_grid_search_cv(
            sessions=sessions,
            model_type=model_type,
            n_folds=n_folds,
            seed=seed,
            burn_in=burn_in,
            n_bins=n_bins,
            fit_target=fit_target,
        )

        all_results.append(result)
        cv_errors.append(result['avg_test_error'])

    # Find best overall params (lowest mean error across seeds)
    # Each result has 'best_params_single' — from the best fold of that seed
    best_idx = int(np.argmin(cv_errors))
    best_params = all_results[best_idx].get('best_params_single', {})

    # Param names
    if model_type == 'BE':
        param_names = ['sigma_percep', 'A_repulsion', 'eta_learning', 'eta_relax']
    else:
        param_names = ['sigma_percep', 'A_repulsion', 'gamma', 'sigma_update']

    return {
        'method': 'grid_search',
        'cv_type': 'held_out',
        'model_type': model_type,
        'fit_target': fit_target,
        'animal_id': animal_id,
        'distribution': distribution,

        'cv_errors': cv_errors,
        'mean_error': float(np.mean(cv_errors)),
        'std_error': float(np.std(cv_errors)),

        'best_params': best_params,
        'all_results': all_results,

        # SBI-compatibility fields (None for GS)
        'posterior_samples': None,
        'param_names': param_names,
        'trajectories': None,
        'link_type': None,

        'n_sessions': len(sessions),
        'n_trials': n_trials,
        'metadata': None,  # Filled by CLI wrapper
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    from scripts.config import (
        CV_DIR, FIT_TARGETS, build_metadata, ensure_dirs,
        load_animal_data,
    )

    parser = argparse.ArgumentParser(
        description='Grid-search CV for one animal × one model')
    parser.add_argument('--animal', required=True, help='Animal ID')
    parser.add_argument('--model', required=True, choices=['BE', 'SC'])
    parser.add_argument('--fit-target', required=True,
                        choices=list(FIT_TARGETS))
    parser.add_argument('--distribution', default='uniform',
                        choices=['uniform', 'hard_a', 'hard_b'])
    parser.add_argument('--n-seeds', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    ensure_dirs()

    output_dir = Path(args.output_dir) if args.output_dir else (
        CV_DIR / f'{args.distribution}_{args.fit_target}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{args.animal}_{args.model}.pkl'

    print(f'=== Grid-Search CV: {args.animal} / {args.model} '
          f'/ {args.fit_target} ===')
    if args.smoke_test:
        print('  ** SMOKE TEST **')

    # Load and select sessions
    from behav_utils.data.selection import select_sessions
    from behav_utils.data.filtering import filter_trials

    animal = load_animal_data(args.animal, config_path=args.config)
    sessions = select_sessions(animal, f'expert_{args.distribution}')
    clean = filter_trials(sessions)

    if not clean:
        print(f'  No expert {args.distribution} sessions. Skipping.')
        sys.exit(0)

    t0 = time.time()
    result = run_grid_search(
        sessions=clean,
        model_type=args.model,
        fit_target=args.fit_target,
        animal_id=args.animal,
        distribution=args.distribution,
        n_seeds=args.n_seeds,
        smoke_test=args.smoke_test,
    )
    dt = time.time() - t0

    result['metadata'] = build_metadata('grid_search.py', vars(args))

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    print(f'  Mean error: {result["mean_error"]:.6f} '
          f'± {result["std_error"]:.6f}')
    print(f'  {dt:.1f}s → {output_path}')


if __name__ == '__main__':
    main()
