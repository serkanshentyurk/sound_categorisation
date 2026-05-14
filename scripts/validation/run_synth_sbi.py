#!/usr/bin/env python3
"""
SBI model comparison on one synthetic animal using pre-trained SNPE.

Usage:
    python scripts/validation/run_synth_sbi.py \
        --cohort static_uniform --animal-index 0 \
        --fit-target update_matrix \
        --snpe-be results/snpe/uniform_be.pkl \
        --snpe-sc results/snpe/uniform_sc.pkl
    python scripts/validation/run_synth_sbi.py ... --smoke-test

Output:
    results/validation/synth_sbi/{cohort}_{fit_target}/synth_{animal_id}.pkl
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
    SBI_N_CV_REPEATS, SBI_BURN_IN, SBI_STATS,
    VALIDATION_DIR, SYNTH_COHORTS_DIR, SNPE_DIR, BASE_SEED, FIT_TARGETS,
    build_metadata,
)


def load_snpe(path):
    """Load trained SNPE, recreating simulator/prior from saved metadata."""
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

    # Recreate simulator with generic stimuli
    rng = np.random.default_rng(BASE_SEED)
    stim, cat = sample_stimuli(2500, 'uniform', rng)
    creator = create_be_simulator if model_type == 'be' else create_sc_simulator
    sim = creator(stim, cat, stat_names=stat_names, burn_in=burn_in)

    data['simulator'] = sim
    data['sbi_sim'] = wrap_for_sbi(sim)
    data['prior'] = get_sbi_prior(sim)
    return data


def main():
    parser = argparse.ArgumentParser(description='Synthetic SBI validation')
    parser.add_argument('--cohort', required=True,
                        choices=['static_uniform', 'learning_uniform'])
    parser.add_argument('--animal-index', required=True, type=int)
    parser.add_argument('--fit-target', required=True, choices=list(FIT_TARGETS),
                        help='Which matrix to score against (method for cv_comparison)')
    parser.add_argument('--snpe-be', type=str, default=None,
                        help='Path to trained BE SNPE pickle')
    parser.add_argument('--snpe-sc', type=str, default=None,
                        help='Path to trained SC SNPE pickle')
    parser.add_argument('--sessions-key', default='sessions')
    parser.add_argument('--n-cv-repeats', type=int, default=SBI_N_CV_REPEATS)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    if args.smoke_test:
        args.n_cv_repeats = 4

    # Paths
    snpe_be_path = Path(args.snpe_be) if args.snpe_be else SNPE_DIR / 'uniform_be.pkl'
    snpe_sc_path = Path(args.snpe_sc) if args.snpe_sc else SNPE_DIR / 'uniform_sc.pkl'

    # Load cohort
    cohort_path = SYNTH_COHORTS_DIR / f'{args.cohort}.pkl'
    with open(cohort_path, 'rb') as f:
        cohort_data = pickle.load(f)
    animals = cohort_data['animals']
    sa = animals[args.animal_index]
    aid = sa['animal_id']
    sessions = sa[args.sessions_key]

    output_dir = VALIDATION_DIR / 'synth_sbi' / f'{args.cohort}_{args.fit_target}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'synth_{aid}.pkl'

    print(f'=== Synth SBI: {aid} / {args.fit_target} ===')
    print(f'  True model: {sa["true_model"]}')
    print(f'  Sessions:   {len(sessions)}')
    if args.smoke_test:
        print('  ** SMOKE TEST MODE **')

    # Load SNPE networks
    print('  Loading SNPE networks...')
    be_snpe = load_snpe(snpe_be_path)
    sc_snpe = load_snpe(snpe_sc_path)

    # Build FittingData from synthetic sessions
    from behav_utils.data.selection import fitting_data_from_sessions
    from behav_utils.data.filtering import filter_trials
    
    sessions = filter_trials(sessions)
    fd = fitting_data_from_sessions(sessions, aid)

    # Run pipeline
    from inference.comparison import run_animal_pipeline

    t0 = time.time()
    result = run_animal_pipeline(
        fd, be_snpe, sc_snpe,
        n_cv_repeats=args.n_cv_repeats,
        seed=BASE_SEED,
        verbose=True,
        method=args.fit_target,
    )
    elapsed = time.time() - t0

    correct = result['winner'] == sa['true_model']
    print(f'\n  Winner: {result["winner"]} (true: {sa["true_model"]}) '
          f'{"✓" if correct else "✗"}')
    print(f'  Time: {elapsed:.1f}s')

    save_data = {
        'animal_id': aid,
        'true_model': sa['true_model'],
        'true_params': sa['true_params'],
        'winner': result['winner'],
        'correct': correct,
        'be_mean': result['be_mean'],
        'sc_mean': result['sc_mean'],
        'p': result['p'],
        'be_params': result['be_params'],
        'sc_params': result['sc_params'],
        'be_test_errors': result['be_cv']['test_errors'],
        'sc_test_errors': result['sc_cv']['test_errors'],
        'method': args.fit_target,
        'cohort': args.cohort,
        'n_sessions': len(sessions),
        'metadata': build_metadata('validation/run_synth_sbi.py', vars(args)),
    }
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f'  Saved to {output_path}')


if __name__ == '__main__':
    main()
