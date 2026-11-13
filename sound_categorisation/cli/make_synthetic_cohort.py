#!/usr/bin/env python
"""Generate a synthetic cohort (ground truth known) for validating sc-grid-search / sc-sbi-condition.

A cohort is a pickle at ``cohort_path(name)`` with the shape ``_synthetic_records``
expects: ``{'animals': [{'animal_id', 'sessions': [{stimuli, choices, categories}],
'true_model', 'true_params'}, ...], 'generation': {...}}``. Animals are simulated from the BE
and SC priors, so the cohort carries ground truth for the recovery / identification checks.

Two options make the synthetic animals look like the real ones the model comparison is run on:

    --shape-from <cohort>/<run_id>   each synthetic animal copies the session count and session
                                     lengths (valid trials) of a randomly chosen kept animal in
                                     that model-identification run (its sessions/<D>.csv)
    --accuracy-rule                  a parameter draw is kept only if every simulated session
                                     reaches the model_id_<D> preset's min_accuracy, as every real
                                     selected session does; rejected draws are counted in
                                     ``generation``

Seeds start at VALIDATION_SEED + 1,000,000, so no synthetic animal replays a training
simulation's (or an SBC draw's) stimulus stream.

The cohort name encodes the phase (sc-grid-search / sc-sbi-condition fit one phase per launch); the
default name is ``synthetic_<distribution>``:

    sc-make-synthetic-cohort --distribution uniform            # -> synthetic_uniform (10 BE + 10 SC x 5)
    sc-make-synthetic-cohort --distribution uniform --n-per-model 50 --name synthetic_uniform_k50 \\
        --shape-from all-mice/<run id> --accuracy-rule

Then (GS is torch-free, so it runs anywhere; SBI needs torch + trained nets):

    sc-grid-search  --cohort synthetic_uniform --distribution uniform \\
        --fit-target update_matrix --fast
    sc-sbi-condition --cohort synthetic_uniform --distribution uniform --fast
    sc-consensus --cohort synthetic_uniform --distribution uniform

A validation run is an analysis, not a test: its results (recovery, confusion) are reported. The
pipeline test lives in ``tests/`` and uses no cohort file.
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

from sound_categorisation.data.paths import cohort_path
from sound_categorisation.models.be_core import BEParams
from sound_categorisation.models.sc_core import SCParams
from sound_categorisation.settings import DISTRIBUTIONS, SBI_TRAIN_T, VALIDATION_SEED

_PARAMS = {'BE': BEParams, 'SC': SCParams}
COHORT_SEED = VALIDATION_SEED + 1_000_000
# seed layout below COHORT_SEED + 300,000: truths (+1, +2), shapes (+3, +4), and the sessions of
# candidate j (+100,000 + j for BE, +200,000 + j for SC)
_MAX_CANDIDATES = 100_000


def preset_min_accuracy(distribution: str) -> float:
    """min_accuracy of the model_id_<distribution> preset (config.yaml)."""
    import yaml
    from behav_utils.data import preset_from_spec

    from sound_categorisation.data.paths import DEFAULT_CONFIG
    name = f'model_id_{distribution}'
    spec = ((yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}).get('session_presets') or {}).get(name)
    if not isinstance(spec, dict):
        raise ValueError(f'config.yaml has no session preset {name}')
    acc = preset_from_spec(name, spec).min_accuracy
    if acc is None:
        raise ValueError(f'preset model_id_{distribution} has no min_accuracy')
    return float(acc)


def shapes_from_run(run, distribution: str = 'uniform'):
    """Per kept animal of a model-identification run, its sessions' valid-trial counts in
    session order: ``{animal_id: [n_valid, ...]}`` (from ``<run>/sessions/<distribution>.csv``)."""
    import pandas as pd
    table = pd.read_csv(run / 'sessions' / f'{distribution}.csv')
    table = table[table['kept'].astype(bool) & (table['n_valid'].fillna(0) > 0)]
    return {aid: [int(v) for v in g.sort_values('session_idx')['n_valid']]
            for aid, g in table.groupby('animal_id')}


def _session_accuracy(session) -> float:
    return float(session.summary().accuracy)


def make_cohort(name, distribution='uniform', n_per_model=10, n_sessions=5,
                trials=SBI_TRAIN_T, burn_in=1000, seed=None, shapes=None,
                min_accuracy=None, max_candidates_per_animal=50):
    """Simulate a BE+SC cohort and write it to cohort_path(name); return the path.

    ``trials`` is an int or a ``(lo, hi)`` range drawn per session (default: the networks'
    SBI_TRAIN_T). ``shapes`` (``{animal: [n_valid, ...]}``, e.g. from ``shapes_from_run``)
    replaces ``n_sessions`` and ``trials``: each synthetic animal copies one real animal's
    session count and lengths, drawn with replacement. With ``min_accuracy`` a parameter draw
    is kept only if every session's accuracy is at least that (the session filter's rule);
    otherwise a new draw is tried, up to ``max_candidates_per_animal`` per kept animal on
    average. Sessions come from ``inference.simulator.simulate_sessions`` -- the same generator
    the networks train on.
    """
    from sound_categorisation.inference.simulator import simulate_sessions
    seed = COHORT_SEED if seed is None else int(seed)
    shape_list = [list(v) for _, v in sorted(shapes.items())] if shapes else None
    max_candidates = min(_MAX_CANDIDATES, max_candidates_per_animal * n_per_model)

    animals, generation = [], {'seed': seed, 'distribution': distribution,
                               'n_per_model': n_per_model, 'min_accuracy': min_accuracy,
                               'shapes_from': sorted(shapes) if shapes else None,
                               'n_sessions': None if shapes else n_sessions,
                               'trials': None if shapes else trials, 'burn_in': burn_in}
    for mi, model in enumerate(('BE', 'SC')):
        prior_rng = np.random.default_rng(seed + 1 + mi)       # one stream per model for the truths
        shape_rng = np.random.default_rng(seed + 3 + mi)
        kept, n_candidates = 0, 0
        while kept < n_per_model:
            if n_candidates >= max_candidates:
                raise RuntimeError(
                    f'{model}: only {kept}/{n_per_model} animals passed the accuracy rule '
                    f'(>= {min_accuracy}) after {n_candidates} draws')
            params = _PARAMS[model].sample_prior(rng=prior_rng).to_dict()
            if shape_list:
                lengths = shape_list[int(shape_rng.integers(len(shape_list)))]
            else:
                lengths = None
            rng = np.random.default_rng(seed + 100_000 * (mi + 1) + n_candidates)
            n_candidates += 1
            if lengths is None:
                sims = simulate_sessions(model.lower(), params, [distribution] * n_sessions,
                                         trials, burn_in, rng)
            else:
                sims = [simulate_sessions(model.lower(), params, [distribution], int(n_valid),
                                          burn_in, rng)[0]
                        for n_valid in lengths]
            if min_accuracy is not None and min(_session_accuracy(s) for s in sims) < min_accuracy:
                continue
            sessions = []
            for s in sims:
                a = s.get_arrays()
                sessions.append({'stimuli': a['stimuli'], 'choices': a['choices'],
                                 'categories': a['categories']})
            animals.append({
                'animal_id': f'{model}{kept:02d}',
                'sessions': sessions,
                'true_model': model,
                'true_params': params,
            })
            kept += 1
        generation[f'{model}_candidates'] = n_candidates
        generation[f'{model}_rejected'] = n_candidates - kept
    path = cohort_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'animals': animals, 'generation': generation}, f)
    path.with_suffix('.json').write_text(json.dumps(generation, indent=2, default=str))
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description='Make a synthetic cohort with known ground truth.')
    p.add_argument('--name', default=None, help='Cohort name (default synthetic_<distribution>).')
    p.add_argument('--distribution', default='uniform', choices=list(DISTRIBUTIONS),
                   help='Stimulus distribution to simulate (default uniform).')
    p.add_argument('--n-per-model', type=int, default=10, help='Animals per model (BE, SC).')
    p.add_argument('--n-sessions', type=int, default=5, help='Sessions per animal (without --shape-from).')
    p.add_argument('--trials', type=int, nargs='+', default=None,
                   help=f'Trials per session: one int, or two ints for a per-session range '
                        f'(default {SBI_TRAIN_T}; without --shape-from).')
    p.add_argument('--shape-from', default=None,
                   help='<cohort>/<run_id> of a model-identification run: copy its kept animals\' '
                        'session counts and lengths')
    p.add_argument('--accuracy-rule', action='store_true',
                   help='keep a draw only if every session reaches the model_id_<D> preset\'s min_accuracy')
    p.add_argument('--burn-in', type=int, default=1000, help='Model burn-in per session.')
    p.add_argument('--seed', type=int, default=COHORT_SEED)
    args = p.parse_args(argv)
    if args.trials is None:
        trials = SBI_TRAIN_T
    elif len(args.trials) == 1:
        trials = args.trials[0]
    elif len(args.trials) == 2:
        trials = (args.trials[0], args.trials[1])
    else:
        p.error('--trials takes one int or two ints (lo hi)')

    shapes = None
    if args.shape_from:
        from sound_categorisation.data.paths import resolve_run
        if '/' not in args.shape_from:
            p.error('--shape-from takes <cohort>/<run_id>')
        cohort, run_id = args.shape_from.split('/', 1)
        shapes = shapes_from_run(resolve_run('model_identification', cohort, run_id), args.distribution)
        print(f'[cohort] session shapes from {args.shape_from}: {len(shapes)} animals, '
              f'{sum(len(v) for v in shapes.values())} sessions')
    min_accuracy = preset_min_accuracy(args.distribution) if args.accuracy_rule else None

    name = args.name or f'synthetic_{args.distribution}'
    path = make_cohort(name, distribution=args.distribution,
                       n_per_model=args.n_per_model, n_sessions=args.n_sessions,
                       trials=trials, burn_in=args.burn_in, seed=args.seed, shapes=shapes,
                       min_accuracy=min_accuracy)
    with open(path, 'rb') as f:
        gen = pickle.load(f)['generation']
    shape = f'shapes from {args.shape_from}' if shapes else f'{args.n_sessions}x{trials} trials'
    rule = (f', accuracy >= {min_accuracy} in every session (rejected {gen["BE_rejected"]} BE, '
            f'{gen["SC_rejected"]} SC draws)' if min_accuracy is not None else '')
    print(f'[cohort] wrote {2 * args.n_per_model} animals ({args.n_per_model} BE + {args.n_per_model} SC), '
          f'{shape}{rule}, phase={args.distribution} -> {path}')


if __name__ == '__main__':
    main()
