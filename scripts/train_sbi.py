#!/usr/bin/env python
"""Train the amortised SBI networks for model-identification validation.

Three data representations x two models = six networks:

    pooled    N=15 sessions merged    mode='pooled'     x-dim 13
    moments   N=15 sessions           mode='moments'    x-dim 52
    single    N=1  session            mode='pooled'     x-dim 13

Each network is an ``AmortisedSBI`` saved to ``snpe_networks/{rep}_{model}.pkl``.
The saved file carries its own config (N / T / mode / stat_names), so the
conditioning step (run_sbi.py / NB12) loads and conditions without re-specifying
anything -- see ``AmortisedSBI.save``/``load``.

The heavy import (``AmortisedSBI``, which pulls in torch + sbi) is deferred into
``train_one`` so that ``--count`` and argument parsing run on a login node
without loading torch.

Local (all six, serial)::

    python -m scripts.train_sbi --rep all --model all

Cluster (SLURM array, one network per task)::

    N=$(python -m scripts.train_sbi --count)        # -> 6
    sbatch --array=0-$((N-1)) train_sbi.sh          # task_id -> (rep, model)

Smoke test (tiny n_simulations, just checks the pipeline runs end to end)::

    python -m scripts.train_sbi --rep pooled --model BE --smoke-test
"""

from __future__ import annotations

import argparse
import time

# Run as a plain script (python scripts/train_sbi.py) or a module
from sound_categorisation.paths import (
    BASE_SEED,
    DISTRIBUTIONS,
    MODEL_TYPES,
    SBI_BURN_IN,
    SBI_REPRESENTATIONS,
    SBI_TRAIN_DISTRIBUTIONS,
    SMOKE_SBI_N_SIMULATIONS,
    snpe_net_path,
)
from sound_categorisation.tasks import TRAIN_GRID

REPRESENTATIONS = tuple(SBI_REPRESENTATIONS)
TRAIN_DISTRIBUTIONS = tuple(SBI_TRAIN_DISTRIBUTIONS)   # per-distribution specialists
N_TASKS = TRAIN_GRID.n                                  # 3 reps × 2 models × 3 distributions = 18


def decode_task(task_id):
    """SLURM array index -> (rep, model, distribution); see sound_categorisation.tasks.TRAIN_GRID."""
    d = TRAIN_GRID.decode(task_id)
    return d['rep'], d['model'], d['distribution']


def train_one(rep, model, distribution, n_simulations=None, seed=BASE_SEED,
              show_progress=True):
    """Train and save one (rep, model, distribution) network. Returns the save path.

    ``dist_schedule=distribution`` makes the training curriculum match the phase the network
    will be conditioned on. ``n_simulations`` overrides the per-rep default in
    SBI_REPRESENTATIONS (used by --smoke-test and --n-simulations).
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


def main():
    p = argparse.ArgumentParser(
        description='Train the amortised SBI networks (3 reps x 2 models x 3 distributions = 18).')
    p.add_argument('--rep', default='all', choices=(*REPRESENTATIONS, 'all'),
                   help="Representation to train, or 'all'.")
    p.add_argument('--model', default='all', choices=(*MODEL_TYPES, 'all'),
                   help="Model to train, or 'all'.")
    p.add_argument('--distribution', default='all', choices=(*DISTRIBUTIONS, 'all'),
                   help="Distribution/curriculum to train, or 'all'.")
    p.add_argument('--task-id', type=int, default=None,
                   help=f'SLURM array index 0-{N_TASKS - 1}; '
                        'overrides --rep/--model/--distribution.')
    p.add_argument('--n-simulations', type=int, default=None,
                   help='Override the per-rep simulation budget.')
    p.add_argument('--seed', type=int, default=BASE_SEED,
                   help='Training seed (default %(default)s).')
    p.add_argument('--smoke-test', action='store_true',
                   help=f'Use {SMOKE_SBI_N_SIMULATIONS} sims to check the '
                        'pipeline runs.')
    p.add_argument('--print-array', action='store_true', help='print the SLURM --array range and exit')
    p.add_argument('--count', action='store_true',
                   help='Print the number of array tasks and exit.')
    args = p.parse_args()
    if args.print_array:
        print(TRAIN_GRID.slurm_range())
        return

    if args.count:
        print(N_TASKS)
        return

    if args.smoke_test:
        n_sims = SMOKE_SBI_N_SIMULATIONS
    else:
        n_sims = args.n_simulations

    # Resolve which (rep, model) pairs to train.
    if args.task_id is not None:
        jobs = [decode_task(args.task_id)]
    else:
        reps = REPRESENTATIONS if args.rep == 'all' else (args.rep,)
        models = MODEL_TYPES if args.model == 'all' else (args.model,)
        dists = DISTRIBUTIONS if args.distribution == 'all' else (args.distribution,)
        jobs = [(r, m, d) for r in reps for m in models for d in dists]

    print(f'[train] {len(jobs)} network(s): {jobs}')
    t0 = time.time()
    for rep, model, distribution in jobs:
        train_one(rep, model, distribution, n_simulations=n_sims, seed=args.seed)
    print(f'[train] all done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
