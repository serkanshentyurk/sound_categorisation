#!/usr/bin/env python3
"""
Generate synthetic BE/SC cohorts and save to disk.

Usage:
    python scripts/generate_synthetic_cohort.py
    python scripts/generate_synthetic_cohort.py --smoke-test

Output:
    results/validation/synthetic_cohorts/static_uniform.pkl
    results/validation/synthetic_cohorts/learning_uniform.pkl
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.config import (
    SYNTH_N_PER_MODEL, SYNTH_N_SESSIONS, SYNTH_TRIALS_PER_SESSION,
    GS_BURN_IN, BASE_SEED, SYNTH_COHORTS_DIR,
    SMOKE_SYNTH_N_PER_MODEL,
    build_metadata, ensure_dirs,
)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic cohorts')
    parser.add_argument('--n-per-model', type=int, default=None)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    n_per = args.n_per_model or (
        SMOKE_SYNTH_N_PER_MODEL if args.smoke_test else SYNTH_N_PER_MODEL
    )

    ensure_dirs()
    SYNTH_COHORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f'=== Generating Synthetic Cohorts ===')
    print(f'  Animals per model: {n_per}')
    if args.smoke_test:
        print('  ** SMOKE TEST MODE **')

    from analysis.validation import make_synthetic_cohort, make_learning_cohort

    meta = build_metadata('generate_synthetic_cohort.py', vars(args))

    # Static uniform
    print(f'\n--- Static Uniform ---')
    t0 = time.time()
    static = make_synthetic_cohort(
        n_per_model=n_per,
        n_sessions=SYNTH_N_SESSIONS,
        trials_per_session=SYNTH_TRIALS_PER_SESSION,
        burn_in=GS_BURN_IN,
        seed=BASE_SEED,
    )
    dt = time.time() - t0
    print(f'  {len(static)} animals in {dt:.1f}s')

    path = SYNTH_COHORTS_DIR / 'static_uniform.pkl'
    with open(path, 'wb') as f:
        pickle.dump({'animals': static, 'metadata': meta}, f)
    print(f'  Saved to {path}')

    # Learning uniform
    print(f'\n--- Learning Uniform ---')
    t0 = time.time()
    learning = make_learning_cohort(
        n_per_model=n_per,
        n_sessions=20,
        trials_per_session=SYNTH_TRIALS_PER_SESSION,
        burn_in=GS_BURN_IN,
        seed=BASE_SEED,
    )
    dt = time.time() - t0
    print(f'  {len(learning)} animals in {dt:.1f}s')

    path = SYNTH_COHORTS_DIR / 'learning_uniform.pkl'
    with open(path, 'wb') as f:
        pickle.dump({'animals': learning, 'metadata': meta}, f)
    print(f'  Saved to {path}')

    print(f'\nDone.')


if __name__ == '__main__':
    main()
