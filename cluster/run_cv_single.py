#!/usr/bin/env python3
"""
run_cv_single.py — Run BE/SC grid-search CV for one animal, one seed.

Uses the new-architecture models (Models.BE_core, Models.SC_core) and
behav_utils update matrix computation.

Designed for SLURM array jobs:
    python run_cv_single.py --config config.yaml --animal SS01 --seed 1 \
        --output-dir ./results/cv

Pickle output format is backward-compatible with analysis/cv_utils.py.
"""

import argparse
import numpy as np
import pickle
import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    p = argparse.ArgumentParser(
        description='BE/SC grid-search CV for one animal, one seed',
    )
    p.add_argument('--config', required=True,
                   help='Path to behav_utils config.yaml')
    p.add_argument('--animal', required=True, help='Animal ID (e.g. SS01)')
    p.add_argument('--seed', required=True, type=int,
                   help='CV seed (1-indexed)')
    p.add_argument('--output-dir', required=True,
                   help='Directory for output pickles')

    p.add_argument('--grid', default='full', choices=['full', 'coarse'],
                   help='Grid resolution')

    # Session selection
    p.add_argument('--preset', default='expert_uniform',
                   help='Session selection preset name')
    p.add_argument('--min-expert-sessions', default=5, type=int)

    # CV settings
    p.add_argument('--n-folds', default=2, type=int)
    p.add_argument('--burn-in', default=1000, type=int)
    p.add_argument('--n-jobs', default=-1, type=int,
                   help='Parallelism for grid search')

    return p.parse_args()


def main():
    args = parse_args()

    from behav_utils.data.loading import load_experiment
    from behav_utils.data.selection import select_sessions
    from analysis.grid_search import (
        grid_search_cv, DEFAULT_GRID, COARSE_GRID,
    )

    # ── Grid ──────────────────────────────────────────────────────────────
    grids = DEFAULT_GRID if args.grid == 'full' else COARSE_GRID

    # ── Load data ─────────────────────────────────────────────────────────
    experiment = load_experiment(args.config)
    animal = experiment.get_animal(args.animal)

    sessions = select_sessions(animal, args.preset)

    if len(sessions) < args.min_expert_sessions:
        print(f"SKIP: {args.animal} has only {len(sessions)} sessions "
              f"(need {args.min_expert_sessions})")
        sys.exit(0)

    n_trials = sum(s.trials.valid_mask.sum() for s in sessions)
    print(f"{args.animal}: {len(sessions)} sessions, {n_trials} valid trials, "
          f"seed={args.seed}")

    # ── Run CV for both models ────────────────────────────────────────────
    results = {}
    for model_type in ['BE', 'SC']:
        t0 = time.time()
        try:
            cv_result = grid_search_cv(
                sessions=sessions,
                model_type=model_type,
                grid=grids[model_type],
                n_folds=args.n_folds,
                seed=args.seed,
                burn_in=args.burn_in,
                n_jobs=args.n_jobs,
            )

            # Store in backward-compatible format
            # (analysis/cv_utils.py expects these keys)
            results[model_type] = {
                'avg_test_error': cv_result['avg_test_error'],
                'test_errors': cv_result['test_errors'],
                'best_params': cv_result['best_params'],          # list of dicts
                'best_params_single': cv_result['best_params_single'],  # dict
                'model': model_type,
                'seed': args.seed,
            }

            elapsed = time.time() - t0
            print(f"  {model_type}: avg_error={cv_result['avg_test_error']:.6f} "
                  f"({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {model_type}: FAILED after {elapsed:.1f}s — {e}")
            results[model_type] = {
                'avg_test_error': np.nan,
                'test_errors': [],
                'best_params': None,
                'best_params_single': None,
                'model': model_type,
                'seed': args.seed,
            }

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f'cv_{args.animal}_seed{args.seed:03d}.pkl',
    )
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
