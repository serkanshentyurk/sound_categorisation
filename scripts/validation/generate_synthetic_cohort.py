#!/usr/bin/env python3
"""
Generate a synthetic BE/SC cohort and save it for GS/SBI validation.

Each animal is simulated from a BE or SC model whose parameters are sampled
uniformly within the model's own bounds. Sessions are stored as raw per-trial
arrays (stimuli, choices, categories) plus the ground-truth model/params —
NOT SessionData objects — so the file is robust to behav_utils class changes.
Reconstruct SessionData with session_from_arrays at load time.

Ground truth (true_model, true_params) rides alongside each animal's data,
never inside it, so synthetic animals are structurally identical to real ones.

Usage:
    python scripts/validation/generate_synthetic_cohort.py --cohort static_uniform
    python scripts/validation/generate_synthetic_cohort.py --cohort static_uniform --smoke-test
    python scripts/validation/generate_synthetic_cohort.py --cohort static_uniform --output /tmp/c.pkl
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.config import (
    SYNTH_N_PER_MODEL, SYNTH_N_SESSIONS, SYNTH_TRIALS_PER_SESSION,
    SMOKE_SYNTH_N_PER_MODEL, GS_BURN_IN, BASE_SEED,
    cohort_path, build_metadata,
)
from models.simulate import simulate_choices
from inference.types import ModelType, get_default_param_configs
from behav_utils.data.synthetic import sample_stimuli

# (ModelType, label) for each generator.
_GENERATORS = [
    (ModelType.BE, 'BE'),
    (ModelType.SC, 'SC'),
]

# Expert accuracy floor for the synthetic cohort. Deliberately 0.65, distinct
# from the real-data EXPERT_MIN_ACCURACY (0.70) in scripts/config.py.
DEFAULT_MIN_ACCURACY = 0.65


def sample_params(model_type, rng):
    """Sample one parameter set uniformly within the model's bounds."""
    configs = get_default_param_configs(model_type)
    return {name: float(rng.uniform(*cfg.bounds)) for name, cfg in configs.items()}


def make_sessions(model_type, params, n_sessions, n_trials, burn_in, rng):
    """Simulate one animal's sessions as raw per-session arrays (no aborts)."""
    sessions = []
    for _ in range(n_sessions):
        stimuli, categories = sample_stimuli(n_trials, distribution='uniform', rng=rng)
        seed = int(rng.integers(0, 2**31 - 1))
        choices = simulate_choices(
            model_type, params, stimuli, categories, burn_in=burn_in, seed=seed)
        sessions.append({
            'stimuli': np.asarray(stimuli, dtype=float),
            'choices': np.asarray(choices, dtype=float),
            'categories': np.asarray(categories, dtype=int),
        })
    return sessions


def _pooled_accuracy(sessions):
    """Overall choice == category accuracy across all of an animal's trials."""
    choices = np.concatenate([s['choices'] for s in sessions])
    categories = np.concatenate([s['categories'] for s in sessions])
    return float(np.mean(choices == categories))


def make_expert_animal(model_type, n_sessions, n_trials, burn_in, rng,
                       min_accuracy, max_attempts):
    """Sample params + simulate until the animal clears the expert accuracy floor.

    Burn-in already warms the model state, so low accuracy means genuinely noisy
    params (high sigma_percep), not an un-converged boundary — hence resample
    rather than simulate longer.
    """
    for _ in range(max_attempts):
        true_params = sample_params(model_type, rng)
        sessions = make_sessions(model_type, true_params, n_sessions, n_trials, burn_in, rng)
        acc = _pooled_accuracy(sessions)
        if acc >= min_accuracy:
            return true_params, sessions, acc
    raise RuntimeError(
        f'Could not sample an expert {model_type.value} animal '
        f'(>= {min_accuracy} accuracy) in {max_attempts} attempts')


def generate_cohort(n_per_model, n_sessions, n_trials, burn_in,
                    min_accuracy, max_attempts, seed):
    """Build the cohort: n_per_model BE + n_per_model SC expert animals."""
    rng = np.random.default_rng(seed)
    animals = []
    for model_type, label in _GENERATORS:
        for i in range(n_per_model):
            true_params, sessions, acc = make_expert_animal(
                model_type, n_sessions, n_trials, burn_in, rng,
                min_accuracy, max_attempts)
            animals.append({
                'animal_id': f'{label}{i:02d}',
                'true_model': label,
                'true_params': true_params,
                'accuracy': acc,
                'sessions': sessions,
            })
    return animals


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic validation cohort')
    parser.add_argument('--cohort', default='static_uniform',
                        help='Cohort name (also the output filename stem)')
    parser.add_argument('--n-per-model', type=int, default=None)
    parser.add_argument('--n-sessions', type=int, default=SYNTH_N_SESSIONS)
    parser.add_argument('--n-trials', type=int, default=SYNTH_TRIALS_PER_SESSION)
    parser.add_argument('--burn-in', type=int, default=GS_BURN_IN)
    parser.add_argument('--min-accuracy', type=float, default=DEFAULT_MIN_ACCURACY,
                        help='Expert accuracy floor; animals below it are resampled')
    parser.add_argument('--max-attempts', type=int, default=200,
                        help='Max resamples per animal to clear the accuracy floor')
    parser.add_argument('--seed', type=int, default=BASE_SEED)
    parser.add_argument('--output', type=str, default=None,
                        help='Override output path (default: cohort_path(cohort))')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    n_per_model = args.n_per_model or (
        SMOKE_SYNTH_N_PER_MODEL if args.smoke_test else SYNTH_N_PER_MODEL
    )
    out_path = Path(args.output) if args.output else cohort_path(args.cohort)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'=== Generate cohort: {args.cohort} ===')
    print(f'  {n_per_model} BE + {n_per_model} SC animals')
    print(f'  {args.n_sessions} sessions x {args.n_trials} trials, burn_in={args.burn_in}')
    print(f'  expert floor: accuracy >= {args.min_accuracy}')
    if args.smoke_test:
        print('  ** SMOKE TEST MODE **')

    t0 = time.time()
    animals = generate_cohort(
        n_per_model, args.n_sessions, args.n_trials, args.burn_in,
        args.min_accuracy, args.max_attempts, args.seed,
    )
    elapsed = time.time() - t0

    accs = [a['accuracy'] for a in animals]
    cohort_data = {
        'cohort': args.cohort,
        'animals': animals,
        'n_per_model': n_per_model,
        'n_sessions': args.n_sessions,
        'n_trials': args.n_trials,
        'metadata': build_metadata('validation/generate_synthetic_cohort.py', vars(args)),
    }
    with open(out_path, 'wb') as f:
        pickle.dump(cohort_data, f)

    print(f'  {len(animals)} animals in {elapsed:.1f}s')
    print(f'  accuracy: min {min(accs):.2f}, mean {np.mean(accs):.2f}, max {max(accs):.2f}')
    print(f'  Saved to {out_path}')


if __name__ == '__main__':
    main()
