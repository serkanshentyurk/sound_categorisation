#!/usr/bin/env python
"""Generate a synthetic cohort (ground truth known) for validating run_gs / run_sbi.

A cohort is a pickle at ``cohort_path(name)`` with the shape ``_synthetic_records``
expects: ``{'animals': [{'animal_id', 'sessions': [{stimuli, choices, categories}],
'true_model', 'true_params'}, ...]}``. Animals are simulated from the BE and SC
priors, so the cohort carries ground truth for the recovery / identification checks.

The cohort name encodes the phase (run_gs / run_sbi fit one phase per launch); the default name
is ``synthetic_<distribution>``:

    sc-make-synthetic-cohort --distribution uniform            # -> synthetic_uniform
    sc-make-synthetic-cohort --distribution hard_a --n-per-model 20 --name synthetic_hard_a_n20

Then (GS is torch-free, so it runs anywhere; SBI needs torch + trained nets):

    sc-run-gs  --source synthetic --cohort synthetic_uniform --distribution uniform \
        --fit-target update_matrix --fast
    sc-run-sbi --source synthetic --cohort synthetic_uniform --distribution uniform --fast
    sc-consensus --cohort synthetic_uniform --distribution uniform

A validation run is an analysis, not a test: its results (recovery, confusion) are reported. The
pipeline test lives in ``tests/`` and uses no cohort file.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np

from sound_categorisation.data.paths import cohort_path
from sound_categorisation.data.stimuli import sample_distribution
from sound_categorisation.models.be_core import BEParams
from sound_categorisation.models.sc_core import SCParams
from sound_categorisation.models.simulate import simulate_choices
from sound_categorisation.settings import DISTRIBUTIONS

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
    p = argparse.ArgumentParser(description='Make a synthetic cohort with known ground truth.')
    p.add_argument('--name', default=None, help='Cohort name (default synthetic_<distribution>).')
    p.add_argument('--distribution', default='uniform', choices=list(DISTRIBUTIONS),
                   help='Stimulus distribution to simulate (default uniform).')
    p.add_argument('--n-per-model', type=int, default=3, help='Animals per model (BE, SC).')
    p.add_argument('--n-sessions', type=int, default=5, help='Sessions per animal.')
    p.add_argument('--trials', type=int, default=500, help='Trials per session.')
    p.add_argument('--burn-in', type=int, default=1000, help='Model burn-in per session.')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    name = args.name or f'synthetic_{args.distribution}'
    path = make_cohort(name, distribution=args.distribution,
                       n_per_model=args.n_per_model, n_sessions=args.n_sessions,
                       trials=args.trials, burn_in=args.burn_in, seed=args.seed)
    n = 2 * args.n_per_model
    print(f'[cohort] wrote {n} animals ({args.n_per_model} BE + {args.n_per_model} SC), '
          f'{args.n_sessions}x{args.trials} trials, phase={args.distribution} -> {path}')


if __name__ == '__main__':
    main()
