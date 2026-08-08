#!/usr/bin/env python
"""Generate a small synthetic cohort for local smoke-testing run_gs / run_sbi.

A cohort is a pickle at ``cohort_path(name)`` with the shape ``_synthetic_records``
expects: ``{'animals': [{'animal_id', 'sessions': [{stimuli, choices, categories}],
'true_model', 'true_params'}, ...]}``. Animals are simulated from the BE and SC
priors, so the cohort carries ground truth for the recovery / identification checks.

The cohort name should encode the phase (matching run_gs / run_sbi's convention),
so make one per phase to smoke-test all three:

    python -m scripts.make_smoke_cohort --name smoke_uniform --distribution uniform
    python -m scripts.make_smoke_cohort --name smoke_hard_a  --distribution hard_a
    python -m scripts.make_smoke_cohort --name smoke_hard_b  --distribution hard_b

Then (GS is torch-free, so it runs anywhere; SBI needs torch + trained nets):

    python -m scripts.run_gs  --source synthetic --cohort smoke_uniform \
        --distribution uniform --run quick --fit-target update_matrix --task-id 0
    python -m scripts.run_sbi --source synthetic --cohort smoke_uniform \
        --distribution uniform --run smoke --smoke-test

Defaults are deliberately tiny (fast to simulate); GS grid-search is still slow,
so prefer a single ``--task-id 0`` array task over the whole cohort for a pipe-clean.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.BE_core import BEParams
from models.SC_core import SCParams
from models.simulate import simulate_choices
from utils.stimulus_distributions import sample_distribution
from scripts.config import cohort_path, DISTRIBUTIONS

_PARAMS = {'BE': BEParams, 'SC': SCParams}


def make_cohort(name, distribution='uniform', n_per_model=3, n_sessions=5,
                trials=500, burn_in=1000, seed=0):
    """Simulate a BE+SC cohort and write it to cohort_path(name); return the path."""
    animals = []
    for mi, model in enumerate(('BE', 'SC')):
        for i in range(n_per_model):
            params = _PARAMS[model].sample_prior(
                rng=np.random.default_rng(seed + 100 * mi + i)).to_dict()
            rng = np.random.default_rng(seed + 10_000 * mi + i)
            sessions = []
            for _ in range(n_sessions):
                stim, cat = sample_distribution(trials, distribution, rng=rng)
                ch = simulate_choices(model, params, stim, cat,
                                      burn_in=burn_in, seed=int(rng.integers(1, 2 ** 31)))
                sessions.append({'stimuli': stim, 'choices': ch, 'categories': cat})
            animals.append({
                'animal_id': f'{model}{i:02d}',
                'sessions': sessions,
                'true_model': model,
                'true_params': params,
            })
    path = cohort_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'animals': animals}, f)
    return path


def main():
    p = argparse.ArgumentParser(description='Make a small synthetic cohort for smoke tests.')
    p.add_argument('--name', required=True, help='Cohort name (encode the phase, e.g. smoke_uniform).')
    p.add_argument('--distribution', default='uniform', choices=list(DISTRIBUTIONS),
                   help='Stimulus distribution to simulate (default uniform).')
    p.add_argument('--n-per-model', type=int, default=3, help='Animals per model (BE, SC).')
    p.add_argument('--n-sessions', type=int, default=5, help='Sessions per animal.')
    p.add_argument('--trials', type=int, default=500, help='Trials per session.')
    p.add_argument('--burn-in', type=int, default=1000, help='Model burn-in per session.')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    path = make_cohort(args.name, distribution=args.distribution,
                       n_per_model=args.n_per_model, n_sessions=args.n_sessions,
                       trials=args.trials, burn_in=args.burn_in, seed=args.seed)
    n = 2 * args.n_per_model
    print(f'[cohort] wrote {n} animals ({args.n_per_model} BE + {args.n_per_model} SC), '
          f'{args.n_sessions}x{args.trials} trials, phase={args.distribution} -> {path}')


if __name__ == '__main__':
    main()
