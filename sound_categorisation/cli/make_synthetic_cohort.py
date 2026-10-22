#!/usr/bin/env python
"""Generate a synthetic cohort (ground truth known) for validating sc-grid-search / sc-sbi-condition.

A cohort is a pickle at ``cohort_path(name)`` with the shape ``_synthetic_records``
expects: ``{'animals': [{'animal_id', 'sessions': [{stimuli, choices, categories}],
'true_model', 'true_params'}, ...]}``. Animals are simulated from the BE and SC
priors, so the cohort carries ground truth for the recovery / identification checks.

The cohort name encodes the phase (sc-grid-search / sc-sbi-condition fit one phase per launch); the default name
is ``synthetic_<distribution>``:

    sc-make-synthetic-cohort --distribution uniform            # -> synthetic_uniform (10 BE + 10 SC x 5 x 500)
    sc-make-synthetic-cohort --distribution hard_a --n-per-model 20 --name synthetic_hard_a_n20

Then (GS is torch-free, so it runs anywhere; SBI needs torch + trained nets):

    sc-grid-search  --cohort synthetic_uniform --distribution uniform \
        --fit-target update_matrix --fast
    sc-sbi-condition --cohort synthetic_uniform --distribution uniform --fast
    sc-consensus --cohort synthetic_uniform --distribution uniform

A validation run is an analysis, not a test: its results (recovery, confusion) are reported. The
pipeline test lives in ``tests/`` and uses no cohort file.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np

from sound_categorisation.data.paths import cohort_path
from sound_categorisation.models.be_core import BEParams
from sound_categorisation.models.sc_core import SCParams
from sound_categorisation.settings import DISTRIBUTIONS, SBI_TRAIN_T

_PARAMS = {'BE': BEParams, 'SC': SCParams}


def make_cohort(name, distribution='uniform', n_per_model=10, n_sessions=5,
                trials=SBI_TRAIN_T, burn_in=1000, seed=0):
    """Simulate a BE+SC cohort and write it to cohort_path(name); return the path.

    ``trials`` is an int or a ``(lo, hi)`` range drawn per session (default: the networks'
    SBI_TRAIN_T, so synthetic animals have the session lengths the networks and the real
    animals have). Sessions come from ``inference.simulator.simulate_sessions`` -- the same
    generator the networks train on.
    """
    from sound_categorisation.inference.simulator import simulate_sessions
    animals = []
    for mi, model in enumerate(('BE', 'SC')):
        prior_rng = np.random.default_rng(seed + 100 * mi)     # one stream per model for the truths
        for i in range(n_per_model):
            params = _PARAMS[model].sample_prior(rng=prior_rng).to_dict()
            rng = np.random.default_rng(seed + 10_000 * mi + i)
            sessions = []
            for s in simulate_sessions(model.lower(), params, [distribution] * n_sessions,
                                       trials, burn_in, rng):
                a = s.get_arrays()
                sessions.append({'stimuli': a['stimuli'], 'choices': a['choices'],
                                 'categories': a['categories']})
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


def main(argv=None):
    p = argparse.ArgumentParser(description='Make a synthetic cohort with known ground truth.')
    p.add_argument('--name', default=None, help='Cohort name (default synthetic_<distribution>).')
    p.add_argument('--distribution', default='uniform', choices=list(DISTRIBUTIONS),
                   help='Stimulus distribution to simulate (default uniform).')
    p.add_argument('--n-per-model', type=int, default=10, help='Animals per model (BE, SC).')
    p.add_argument('--n-sessions', type=int, default=5, help='Sessions per animal.')
    p.add_argument('--trials', type=int, nargs='+', default=None,
                   help=f'Trials per session: one int, or two ints for a per-session range (default {SBI_TRAIN_T}).')
    p.add_argument('--burn-in', type=int, default=1000, help='Model burn-in per session.')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)
    if args.trials is None:
        trials = SBI_TRAIN_T
    elif len(args.trials) == 1:
        trials = args.trials[0]
    elif len(args.trials) == 2:
        trials = (args.trials[0], args.trials[1])
    else:
        p.error('--trials takes one int or two ints (lo hi)')

    name = args.name or f'synthetic_{args.distribution}'
    path = make_cohort(name, distribution=args.distribution,
                       n_per_model=args.n_per_model, n_sessions=args.n_sessions,
                       trials=trials, burn_in=args.burn_in, seed=args.seed)
    n = 2 * args.n_per_model
    print(f'[cohort] wrote {n} animals ({args.n_per_model} BE + {args.n_per_model} SC), '
          f'{args.n_sessions}x{trials} trials, phase={args.distribution} -> {path}')


if __name__ == '__main__':
    main()
