#!/usr/bin/env python
"""Calibrate the trained SBI networks: simulation-based calibration (SBC) + coverage.

For each (rep, model, n_sessions) of SBC_GRID this loads ``snpe_{rep}_{model}_{distribution}.pkl``,
draws SBC_DRAWS_BY_SESSIONS[n_sessions] fresh parameter sets from the prior, simulates
n_sessions x T trials from each (processed exactly like an animal), conditions, and records the
rank of the true value among SBI_N_POSTERIOR_SAMPLES_SAVED posterior samples, plus the posterior
mean, SD, quantiles and the network's acceptance rate (``inference.calibration``). Five sessions
is an animal's full data; 3 and 2 are the CV halves the same networks condition on inside the
model comparison.

Writes, into the run it is given (normally the synthetic validation run):

    <run>/sbc/<distribution>/n<N>/<rep>/<model>.pkl            the compute_sbc_ranks dict
    <run>/sbc/<distribution>/n<N>/<rep>/<model>_coverage.csv   empirical vs nominal coverage

Seeds start at VALIDATION_SEED, so no draw replays a training simulation's stimulus stream.

Local (every network and session count, serial)::

    sc-sbi-sbc --cohort synthetic_uniform_k50 --distribution uniform --run-id ID

Cluster (array over SBC_GRID, one network and session count per task)::

    bash slurm/submit.sh sbi-sbc --cohort synthetic_uniform_k50 --distribution uniform --run-id ID

--fast: FAST_SBC_N_DRAWS draws per task, a pipeline check; the run id gets _fast.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

from sound_categorisation.data.paths import build_metadata, snpe_net_path, start_run
from sound_categorisation.inference.tasks import SBC_GRID
from sound_categorisation.settings import (
    DISTRIBUTIONS,
    FAST_SBC_N_DRAWS,
    MODEL_TYPES,
    SBC_DRAWS_BY_SESSIONS,
    SBI_N_POSTERIOR_SAMPLES_SAVED,
    SBI_REPRESENTATIONS,
    SBI_TRAIN_REPS,
    VALIDATION_SEED,
)

REPRESENTATIONS = tuple(SBI_REPRESENTATIONS)
SESSION_COUNTS = tuple(SBC_DRAWS_BY_SESSIONS)
N_TASKS = SBC_GRID.n


def decode_task(task_id):
    """SLURM array index -> (rep, model, n_sessions)."""
    d = SBC_GRID.decode(task_id)
    return d['rep'], d['model'], int(d['n_sessions'])


def sbc_dir(run, distribution, n_sessions, rep):
    """Where one network's SBC at one session count is written (and read back)."""
    return run / 'sbc' / distribution / f'n{int(n_sessions)}' / rep


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
          f'L={n_posterior_samples}, design={list(net.design_features)}')
    res = compute_sbc_ranks(net, n_draws=n_draws, n_sessions=n_sessions,
                            n_posterior_samples=n_posterior_samples, distribution=distribution,
                            seed=seed)
    res['metadata'] = dict(metadata or {}, rep=rep, model=model, distribution=distribution,
                           n_sessions=int(n_sessions), network_file=net_path.name,
                           network_stat_names=list(net.stat_names),
                           network_design_features=list(net.design_features),
                           network_training_N=net.N)
    with open(out_dir / f'{model}.pkl', 'wb') as f:
        pickle.dump(res, f)
    res['coverage'].to_csv(out_dir / f'{model}_coverage.csv', index=False)
    cov = res['coverage'].pivot(index='param', columns='nominal', values='empirical')
    print(cov.round(3).to_string())
    ks = dict(zip(res['param_names'], [round(p, 3) for p in res['ks_pvalues']]))
    print(f'[sbc] {rep}/{model}/{distribution}/n{n_sessions}: KS p={ks} '
          f'skipped={res["n_skipped"]} budget hit={int(res["budget_hit"].sum())} '
          f'-> {out_dir} ({(time.time() - t0) / 60:.1f} min)')
    return out_dir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cohort', default=None, help='the run\'s cohort (normally the synthetic one)')
    p.add_argument('--run-id', default=None, help='existing run id to write into (required with --task-id)')
    p.add_argument('--distribution', default=None, choices=DISTRIBUTIONS)
    p.add_argument('--rep', default='all', choices=(*REPRESENTATIONS, 'all'))
    p.add_argument('--model', default='all', choices=(*MODEL_TYPES, 'all'))
    p.add_argument('--n-sessions', default='all', choices=(*[str(n) for n in SESSION_COUNTS], 'all'),
                   help='local runs: one session count, or all of SBC_DRAWS_BY_SESSIONS')
    p.add_argument('--n-draws', type=int, default=None,
                   help=f'default per session count: {SBC_DRAWS_BY_SESSIONS}')
    p.add_argument('--n-posterior-samples', type=int, default=SBI_N_POSTERIOR_SAMPLES_SAVED)
    p.add_argument('--seed', type=int, default=VALIDATION_SEED)
    p.add_argument('--task-id', type=int, default=None, help=f'SLURM array index 0-{N_TASKS - 1}')
    p.add_argument('--print-array', action='store_true', help='print the SLURM --array range and exit')
    p.add_argument('--count', action='store_true', help='print the number of array tasks and exit')
    p.add_argument('--fast', action='store_true', help=f'{FAST_SBC_N_DRAWS} draws; run id gets _fast')
    args = p.parse_args(argv)
    if args.print_array:
        print(SBC_GRID.slurm_range())
        return
    if args.count:
        print(N_TASKS)
        return
    if not args.cohort or not args.distribution:
        p.error('--cohort and --distribution are required')
    if args.task_id is not None and not args.run_id:
        p.error('--task-id needs --run-id (every array task must write into the same run)')

    if args.task_id is not None:
        jobs = [decode_task(args.task_id)]
    else:
        reps = tuple(SBI_TRAIN_REPS) if args.rep == 'all' else (args.rep,)
        models = MODEL_TYPES if args.model == 'all' else (args.model,)
        counts = SESSION_COUNTS if args.n_sessions == 'all' else (int(args.n_sessions),)
        jobs = [(r, m, n) for r in reps for m in models for n in counts]

    run = start_run('model_identification', args.cohort, args.run_id, fast=args.fast)
    meta = build_metadata('sbi_sbc', vars(args), run_id=run.name)
    t0 = time.time()
    for rep, model, n_sessions in jobs:
        n_draws = args.n_draws or (FAST_SBC_N_DRAWS if args.fast else SBC_DRAWS_BY_SESSIONS[n_sessions])
        sbc_one(rep, model, args.distribution, sbc_dir(run, args.distribution, n_sessions, rep),
                n_draws, n_sessions, args.n_posterior_samples, args.seed, meta)
    (run / 'sbc' / args.distribution / 'meta.json').write_text(json.dumps(meta, indent=2, default=str))
    print(f'[sbc] done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
