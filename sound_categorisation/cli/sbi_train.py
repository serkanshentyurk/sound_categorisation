#!/usr/bin/env python
"""Train the amortised SBI networks for model identification.

One specialist network per (representation, model, distribution). The job grid is
``settings.SBI_TRAIN_REPS x MODEL_TYPES x SBI_TRAIN_DISTRIBUTIONS`` (currently pooled / moments x
BE / SC x uniform = 4 networks); any representation or distribution in ``SBI_REPRESENTATIONS`` /
``DISTRIBUTIONS`` can still be trained by name, it is just not in the array.

    pooled    N~U{4..6} sessions pooled   mode='pooled'     x-dim D
    moments   N~U{4..6} sessions          mode='moments'    x-dim 2D (mean, variance per stat)
    single    N=1 session                 mode='pooled'     x-dim D

Each network is an ``AmortisedSBI`` saved to ``snpe_networks/snpe_{rep}_{model}_{distribution}.pkl``.
The saved file carries its own config (N / T / mode / stat_names), so the
conditioning step (sc-sbi-condition) loads and conditions without re-specifying
anything -- see ``AmortisedSBI.save``/``load``.

The heavy import (``AmortisedSBI``, which pulls in torch + sbi) is deferred into
``train_one`` so that ``--count`` and argument parsing run on a login node
without loading torch.

Local (the whole grid, serial)::

    sc-sbi-train

Cluster (SLURM array, one network per task)::

    N=$(sc-sbi-train --count)        # -> 4
    bash slurm/submit.sh sbi-train                       # task_id -> (rep, model, distribution)

Smoke test (tiny n_simulations, just checks the pipeline runs end to end)::

    sc-sbi-train --rep pooled --model BE --fast
"""

from __future__ import annotations

import argparse
import time

from sound_categorisation.data.paths import snpe_net_path
from sound_categorisation.inference.tasks import TRAIN_GRID
from sound_categorisation.settings import (
    BASE_SEED,
    DISTRIBUTIONS,
    FAST_SBI_N_SIMULATIONS,
    MODEL_TYPES,
    SBI_BURN_IN,
    SBI_REPRESENTATIONS,
    SBI_TRAIN_DISTRIBUTIONS,
    SBI_TRAIN_REPS,
)

REPRESENTATIONS = tuple(SBI_REPRESENTATIONS)            # everything trainable by name
TRAIN_DISTRIBUTIONS = tuple(SBI_TRAIN_DISTRIBUTIONS)   # the array's distributions
N_TASKS = TRAIN_GRID.n                                  # SBI_TRAIN_REPS × models × SBI_TRAIN_DISTRIBUTIONS


def decode_task(task_id):
    """SLURM array index -> (rep, model, distribution); see sound_categorisation.inference.tasks.TRAIN_GRID."""
    d = TRAIN_GRID.decode(task_id)
    return d['rep'], d['model'], d['distribution']


def train_one(rep, model, distribution, n_simulations=None, seed=BASE_SEED,
              show_progress=True):
    """Train and save one (rep, model, distribution) network. Returns the save path.

    ``dist_schedule=distribution`` makes the training curriculum match the phase the network
    will be conditioned on. ``n_simulations`` overrides the per-rep default in
    SBI_REPRESENTATIONS (used by --fast and --n-simulations).
    """
    if rep not in SBI_REPRESENTATIONS:
        raise ValueError(f'Unknown rep {rep!r}; choose from {REPRESENTATIONS}.')
    if model not in MODEL_TYPES:
        raise ValueError(f'Unknown model {model!r}; choose from {MODEL_TYPES}.')
    if distribution not in DISTRIBUTIONS:
        raise ValueError(
            f'Unknown distribution {distribution!r}; choose from {DISTRIBUTIONS}.')

    # Deferred so the module imports (and --count) work without torch.
    from sound_categorisation.inference.amortised import AmortisedSBI

    cfg = SBI_REPRESENTATIONS[rep]
    n_sims = cfg['n_simulations'] if n_simulations is None else n_simulations

    print(f'[train] {rep}/{model}/{distribution}: N={cfg["N"]} mode={cfg["mode"]} '
          f'T={cfg["T"]} burn_in={SBI_BURN_IN} n_sims={n_sims} seed={seed}')
    t0 = time.time()

    net = AmortisedSBI(
        model,
        dist_schedule=distribution,
        N=cfg['N'],
        T=cfg['T'],
        burn_in=SBI_BURN_IN,
        mode=cfg['mode'],
    )
    net.train(n_simulations=n_sims, seed=seed, show_progress=show_progress)

    out = snpe_net_path(rep, model, distribution)
    net.save(out)
    print(f'[train] {rep}/{model}/{distribution}: saved -> {out} '
          f'({(time.time() - t0) / 60:.1f} min)')
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description=f'Train the amortised SBI networks ({TRAIN_GRID.describe()}).')
    p.add_argument('--rep', default='all', choices=(*REPRESENTATIONS, 'all'),
                   help=f"Representation to train, or 'all' (= {SBI_TRAIN_REPS}).")
    p.add_argument('--model', default='all', choices=(*MODEL_TYPES, 'all'),
                   help="Model to train, or 'all'.")
    p.add_argument('--distribution', default='all', choices=(*DISTRIBUTIONS, 'all'),
                   help=f"Distribution/curriculum to train, or 'all' (= {TRAIN_DISTRIBUTIONS}).")
    p.add_argument('--task-id', type=int, default=None,
                   help=f'SLURM array index 0-{N_TASKS - 1}; '
                        'overrides --rep/--model/--distribution.')
    p.add_argument('--n-simulations', type=int, default=None,
                   help='Override the per-rep simulation budget.')
    p.add_argument('--seed', type=int, default=BASE_SEED,
                   help='Training seed (default %(default)s).')
    p.add_argument('--fast', action='store_true',
                   help=f'Use {FAST_SBI_N_SIMULATIONS} sims to check the '
                        'pipeline runs.')
    p.add_argument('--print-array', action='store_true', help='print the SLURM --array range and exit')
    p.add_argument('--count', action='store_true',
                   help='Print the number of array tasks and exit.')
    args = p.parse_args(argv)
    if args.print_array:
        print(TRAIN_GRID.slurm_range())
        return

    if args.count:
        print(N_TASKS)
        return

    if args.fast:
        n_sims = FAST_SBI_N_SIMULATIONS
    else:
        n_sims = args.n_simulations

    # Resolve which (rep, model) pairs to train.
    if args.task_id is not None:
        jobs = [decode_task(args.task_id)]
    else:
        reps = tuple(SBI_TRAIN_REPS) if args.rep == 'all' else (args.rep,)
        models = MODEL_TYPES if args.model == 'all' else (args.model,)
        dists = TRAIN_DISTRIBUTIONS if args.distribution == 'all' else (args.distribution,)
        jobs = [(r, m, d) for r in reps for m in models for d in dists]

    print(f'[train] {len(jobs)} network(s): {jobs}')
    t0 = time.time()
    for rep, model, distribution in jobs:
        train_one(rep, model, distribution, n_simulations=n_sims, seed=args.seed)
    print(f'[train] all done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
