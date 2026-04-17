#!/usr/bin/env python3
"""
Condition trained SNPE on real animals (local, fast).

Loads trained SNPE pickles, conditions on each animal's observed stats,
scores via update_matrix and/or conditional_psych.

Usage:
    python scripts/condition_sbi_local.py --distribution uniform
    python scripts/condition_sbi_local.py --distribution uniform --smoke-test

Output:
    results/sbi_static/uniform/animal_{aid}.pkl
    results/sbi_static/comparisons/uniform_{fit_target}/animal_{aid}.pkl
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
    SBI_N_CV_REPEATS, SBI_STATS, SBI_BURN_IN,
    SBI_STATIC_DIR, SNPE_DIR, BASE_SEED, FIT_TARGETS, STAGE,
    SMOKE_N_ANIMALS_LIMIT,
    build_metadata, ensure_dirs,
)


def load_snpe(path):
    """Load trained SNPE, recreating simulator/prior."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    from inference.simulator import (
        create_be_simulator, create_sc_simulator,
        get_sbi_prior, wrap_for_sbi,
    )
    from behav_utils.data.synthetic import sample_stimuli

    model_type = data['model_type']
    stat_names = data['stat_names']
    burn_in = data['burn_in']

    rng = np.random.default_rng(BASE_SEED)
    stim, cat = sample_stimuli(2500, 'uniform', rng)
    creator = create_be_simulator if model_type == 'be' else create_sc_simulator
    sim = creator(stim, cat, stat_names=stat_names, burn_in=burn_in)

    data['simulator'] = sim
    data['sbi_sim'] = wrap_for_sbi(sim)
    data['prior'] = get_sbi_prior(sim)
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Condition SNPE on real animals (local)')
    parser.add_argument('--distribution', default='uniform',
                        choices=['uniform', 'hard_a', 'hard_b'])
    parser.add_argument('--animals', type=str, default=None,
                        help='Comma-separated animal IDs (default: all)')
    parser.add_argument('--n-cv-repeats', type=int, default=SBI_N_CV_REPEATS)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    if args.smoke_test:
        args.n_cv_repeats = 4

    ensure_dirs()

    # Load SNPE
    be_path = SNPE_DIR / f'{args.distribution}_be.pkl'
    sc_path = SNPE_DIR / f'{args.distribution}_sc.pkl'
    if not be_path.exists() or not sc_path.exists():
        print(f'Missing SNPE: {be_path} or {sc_path}')
        print('Run train_snpe.py first.')
        sys.exit(1)

    print(f'Loading SNPE networks...')
    be_snpe = load_snpe(be_path)
    sc_snpe = load_snpe(sc_path)

    # Load animal list
    from behav_utils.data.loading import list_animals, load_animal
    from behav_utils.data.selection import select_sessions, fitting_data_from_sessions

    if args.animals:
        animal_ids = args.animals.split(',')
    else:
        animal_ids = list_animals()

    if args.smoke_test:
        animal_ids = animal_ids[:SMOKE_N_ANIMALS_LIMIT]

    print(f'\nProcessing {len(animal_ids)} animals on {args.distribution}')

    from inference.comparison import (
        condition_on_animal, run_animal_pipeline,
    )

    meta = build_metadata('condition_sbi_local.py', vars(args))

    for aid in animal_ids:
        print(f'\n--- {aid} ---')
        try:
            animal = load_animal(aid)
            sessions = select_sessions(animal, f'expert_{args.distribution}')
            if not sessions:
                print(f'  No expert {args.distribution} sessions. Skipping.')
                continue

            fd = fitting_data_from_sessions(sessions, aid)

            # Save posterior (fit_target-agnostic)
            posterior_dir = SBI_STATIC_DIR / args.distribution
            posterior_dir.mkdir(parents=True, exist_ok=True)

            be_cond = condition_on_animal(be_snpe, fd)
            sc_cond = condition_on_animal(sc_snpe, fd)

            posterior_data = {
                'animal_id': aid,
                'distribution': args.distribution,
                'n_sessions': len(sessions),
                'be_params': be_cond['median_params'],
                'sc_params': sc_cond['median_params'],
                'metadata': meta,
            }
            posterior_path = posterior_dir / f'animal_{aid}.pkl'
            with open(posterior_path, 'wb') as f:
                pickle.dump(posterior_data, f)

            # Score via both fit targets
            for ft in FIT_TARGETS:
                comp_dir = SBI_STATIC_DIR / 'comparisons' / f'{args.distribution}_{ft}'
                comp_dir.mkdir(parents=True, exist_ok=True)

                result = run_animal_pipeline(
                    fd, be_snpe, sc_snpe,
                    n_cv_repeats=args.n_cv_repeats,
                    seed=BASE_SEED, verbose=True,
                    method=ft,
                )

                comp_data = {
                    'animal_id': aid,
                    'distribution': args.distribution,
                    'method': ft,
                    'winner': result['winner'],
                    'p': result['p'],
                    'be_mean': result['be_mean'],
                    'sc_mean': result['sc_mean'],
                    'be_params': result['be_params'],
                    'sc_params': result['sc_params'],
                    'metadata': meta,
                }
                comp_path = comp_dir / f'animal_{aid}.pkl'
                with open(comp_path, 'wb') as f:
                    pickle.dump(comp_data, f)

        except Exception as e:
            print(f'  FAILED: {e}')
            import traceback
            traceback.print_exc()

    print(f'\nDone.')


if __name__ == '__main__':
    main()
