#!/usr/bin/env python
"""Calibrate the trained SBI networks: simulation-based calibration (SBC) + coverage.

For each (rep, model) of CONDITION_GRID this loads ``snpe_{rep}_{model}_{distribution}.pkl``,
draws SBC_N_DRAWS fresh parameter sets from the prior, simulates SBC_N_SESSIONS x T trials
from each (processed exactly like an animal), conditions, and records the rank of the true
value among SBI_N_POSTERIOR_SAMPLES_SAVED posterior samples (``inference.calibration``).

Writes, into the run it is given (normally the synthetic validation run):

    <run>/sbc/<distribution>/<rep>/<model>.pkl      the compute_sbc_ranks dict (ranks, truth,
                                                    medians, coverage, KS p-values)
    <run>/sbc/<distribution>/<rep>/<model>_coverage.csv   empirical vs nominal at 50/80/90/95 %

Local (all four, serial)::

    sc-sbi-sbc --cohort synthetic_uniform --distribution uniform --run-id ID

Cluster (array over CONDITION_GRID, one network per task)::

    bash slurm/submit.sh sbi-sbc --cohort synthetic_uniform --distribution uniform --run-id ID

--fast: FAST_SBC_N_DRAWS draws, a pipeline check; the run id gets _fast.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

from sound_categorisation.data.paths import build_metadata, snpe_net_path, start_run
from sound_categorisation.inference.tasks import CONDITION_GRID
from sound_categorisation.settings import (
    BASE_SEED,
    DISTRIBUTIONS,
    FAST_SBC_N_DRAWS,
    MODEL_TYPES,
    SBC_N_DRAWS,
    SBC_N_SESSIONS,
    SBI_N_POSTERIOR_SAMPLES_SAVED,
    SBI_REPRESENTATIONS,
    SBI_TRAIN_REPS,
)

REPRESENTATIONS = tuple(SBI_REPRESENTATIONS)
N_TASKS = CONDITION_GRID.n


def decode_task(task_id):
    """SLURM array index -> (rep, model); the same grid as sc-sbi-condition."""
    d = CONDITION_GRID.decode(task_id)
    return d['rep'], d['model']


def sbc_one(rep, model, distribution, out_dir, n_draws, n_sessions, n_posterior_samples,
            seed, metadata):
    from sound_categorisation.inference.amortised import AmortisedSBI
    from sound_categorisation.inference.calibration import compute_sbc_ranks

    net_path = snpe_net_path(rep, model, distribution)
    if not net_path.exists():
        raise FileNotFoundError(f'No trained net at {net_path}; run sc-sbi-train first.')
    net = AmortisedSBI.load(net_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f'[sbc] {rep}/{model}/{distribution}: {n_draws} draws x {n_sessions} sessions x T={net.T}, '
          f'L={n_posterior_samples}')
    res = compute_sbc_ranks(net, n_draws=n_draws, n_sessions=n_sessions,
                            n_posterior_samples=n_posterior_samples, distribution=distribution,
                            seed=seed)
    res['metadata'] = dict(metadata or {}, rep=rep, model=model, distribution=distribution,
                           network_file=net_path.name, network_stat_names=list(net.stat_names))
    with open(out_dir / f'{model}.pkl', 'wb') as f:
        pickle.dump(res, f)
    res['coverage'].to_csv(out_dir / f'{model}_coverage.csv', index=False)
    cov = res['coverage'].pivot(index='param', columns='nominal', values='empirical')
    print(cov.round(3).to_string())
    print(f'[sbc] {rep}/{model}/{distribution}: KS p={dict(zip(res["param_names"], [round(p, 3) for p in res["ks_pvalues"]]))} '
          f'skipped={res["n_skipped"]} -> {out_dir} ({(time.time() - t0) / 60:.1f} min)')
    return out_dir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cohort', default=None, help='the run\'s cohort (normally the synthetic one)')
    p.add_argument('--run-id', default=None, help='existing run id to write into (required with --task-id)')
    p.add_argument('--distribution', default=None, choices=DISTRIBUTIONS)
    p.add_argument('--rep', default='all', choices=(*REPRESENTATIONS, 'all'))
    p.add_argument('--model', default='all', choices=(*MODEL_TYPES, 'all'))
    p.add_argument('--n-draws', type=int, default=None, help=f'default {SBC_N_DRAWS}')
    p.add_argument('--n-sessions', type=int, default=SBC_N_SESSIONS)
    p.add_argument('--n-posterior-samples', type=int, default=SBI_N_POSTERIOR_SAMPLES_SAVED)
    p.add_argument('--seed', type=int, default=BASE_SEED)
    p.add_argument('--task-id', type=int, default=None, help=f'SLURM array index 0-{N_TASKS - 1}')
    p.add_argument('--print-array', action='store_true', help='print the SLURM --array range and exit')
    p.add_argument('--count', action='store_true', help='print the number of array tasks and exit')
    p.add_argument('--fast', action='store_true', help=f'{FAST_SBC_N_DRAWS} draws; run id gets _fast')
    args = p.parse_args(argv)
    if args.print_array:
        print(CONDITION_GRID.slurm_range())
        return
    if args.count:
        print(N_TASKS)
        return
    if not args.cohort or not args.distribution:
        p.error('--cohort and --distribution are required')
    if args.task_id is not None and not args.run_id:
        p.error('--task-id needs --run-id (every array task must write into the same run)')

    n_draws = args.n_draws or (FAST_SBC_N_DRAWS if args.fast else SBC_N_DRAWS)
    if args.task_id is not None:
        jobs = [decode_task(args.task_id)]
    else:
        reps = tuple(SBI_TRAIN_REPS) if args.rep == 'all' else (args.rep,)
        models = MODEL_TYPES if args.model == 'all' else (args.model,)
        jobs = [(r, m) for r in reps for m in models]

    run = start_run('model_identification', args.cohort, args.run_id, fast=args.fast)
    meta = build_metadata('sbi_sbc', vars(args), run_id=run.name)
    t0 = time.time()
    for rep, model in jobs:
        out_dir = run / 'sbc' / args.distribution / rep
        sbc_one(rep, model, args.distribution, out_dir, n_draws, args.n_sessions,
                args.n_posterior_samples, args.seed, meta)
    (run / 'sbc' / args.distribution / 'meta.json').write_text(json.dumps(meta, indent=2, default=str))
    print(f'[sbc] done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
