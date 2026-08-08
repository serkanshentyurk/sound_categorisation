#!/usr/bin/env python
"""Condition the trained SBI networks on a cohort -> held-out MSE results.

For each (rep, model) this loads the phase-matched specialist
``snpe_networks/snpe_{rep}_{model}_{distribution}.pkl`` and runs ``condition_sbi``
on every animal, writing one neutral-schema pickle per (animal, model) into the
phase's results directory:

    {data_root}/sbi/{run}/{cohort}/{distribution}/{rep}/{animal_id}_{model}.pkl

Each result stamps which network produced it (rep, model, distribution, the net's
filename and its own stat vector) into the metadata, so a file is self-identifying.

--distribution is required: condition ONE phase per launch (its networks + its
sessions). For real data the sessions default to expert_<distribution> unless
--preset is given; for synthetic the --cohort name should encode the phase.

Conditioning is cheap (forward passes + held-out simulation), so this runs
serially -- no partials/gather. The BE-vs-SC winner and recovery come afterwards
from ``load_cv_results`` + ``compare_models`` (run per rep dir); cross-rep /
cross-method consensus from ``analysis.consensus``.

Local (all six (rep, model) on a synthetic uniform cohort)::

    python -m scripts.run_sbi --source synthetic --cohort static_uniform \
        --distribution uniform --run full --fit-target update_matrix

Cluster (SLURM array, one (rep, model) per task; one phase per submit)::

    N=$(python -m scripts.run_sbi --count)          # -> 6
    sbatch --array=0-$((N-1)) slurm/run_sbi.sh --source real --distribution uniform --run expert
    sbatch --array=0-$((N-1)) slurm/run_sbi.sh --source real --distribution hard_a --run expert
    sbatch --array=0-$((N-1)) slurm/run_sbi.sh --source real --distribution hard_b --run expert

Real data (one phase)::

    python -m scripts.run_sbi --source real --distribution hard_a \
        --rep pooled --model all --run expert --fit-target update_matrix
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Run as a plain script (python scripts/run_sbi.py) or a module
# (python -m scripts.run_sbi): put the repo root on sys.path either way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.amortised import AmortisedSBI
from inference.selection import condition_sbi
from scripts.providers import load_animals
from scripts.config import (
    SBI_REPRESENTATIONS,
    SBI_N_CV_REPEATS,
    SBI_N_POSTERIOR_SAMPLES,
    GS_N_BINS,
    GS_N_FOLDS,
    FIT_TARGETS,
    MODEL_TYPES,
    DISTRIBUTIONS,
    BASE_SEED,
    snpe_networks_dir,
    snpe_net_path,
    results_dir,
    build_metadata,
)
from utils.cv_utils import save_cv_result

# Same rep-major, model-minor task order as train_sbi (so a net trained by
# task k is conditioned by task k).
REPRESENTATIONS = tuple(SBI_REPRESENTATIONS)
N_TASKS = len(REPRESENTATIONS) * len(MODEL_TYPES)
SMOKE_N_REPEATS = 2


def decode_task(task_id):
    """Map a SLURM array index in [0, N_TASKS) to a (rep, model) pair."""
    if not 0 <= task_id < N_TASKS:
        raise ValueError(f'task_id must be in [0, {N_TASKS}); got {task_id}.')
    rep = REPRESENTATIONS[task_id // len(MODEL_TYPES)]
    model = MODEL_TYPES[task_id % len(MODEL_TYPES)]
    return rep, model


def condition_cohort(records, rep, model, distribution, out_dir, fit_target,
                     n_repeats=SBI_N_CV_REPEATS,
                     n_posterior_samples=SBI_N_POSTERIOR_SAMPLES,
                     n_folds=GS_N_FOLDS, n_bins=GS_N_BINS, seed=BASE_SEED,
                     metadata=None):
    """Load the (rep, model, distribution) net and condition every animal; one pkl each.

    The network is the phase-matched specialist ``snpe_{rep}_{model}_{distribution}.pkl``.
    A per-animal failure (e.g. fewer than 2 sessions for the multi path) is
    warned and skipped, so one bad animal does not abort the cohort. An animal
    that yields no usable reps (all skipped) still writes an empty result, which
    load_cv_results drops cleanly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    net_path = snpe_net_path(rep, model, distribution)
    if not net_path.exists():
        raise FileNotFoundError(
            f'No trained net at {net_path}; run train_sbi.py first.')
    net = AmortisedSBI.load(net_path)

    # Per-result metadata: stamp exactly which network produced each file, so a
    # result is self-identifying (rep + model + phase + the net's own stat vector).
    net_meta = dict(metadata or {})
    net_meta.update(
        rep=rep, model=model, distribution=distribution,
        network_file=net_path.name,
        network_stat_names=list(net.stat_names),
        network_mode=net.mode, network_N=net.N, network_T=net.T,
    )

    written = 0
    for record in records:
        try:
            results = condition_sbi(
                record.sessions, net, model, fit_target=fit_target,
                n_folds=n_folds, n_repeats=n_repeats,
                n_posterior_samples=n_posterior_samples,
                n_bins=n_bins, seed=seed)
        except ValueError as e:
            print(f'[sbi] {rep}/{model}/{distribution}: SKIP {record.animal_id} ({e})')
            continue
        save_cv_result(
            out_dir / f'{record.animal_id}_{model}.pkl',
            record.animal_id, model, results, fit_target,
            true_model=record.true_model, true_params=record.true_params,
            metadata=net_meta)
        written += 1
        print(f'[sbi] {rep}/{model}/{distribution}: {record.animal_id} -> {len(results)} reps')
    print(f'[sbi] {rep}/{model}/{distribution}: wrote {written}/{len(records)} animals -> {out_dir}')


def main():
    p = argparse.ArgumentParser(
        description='Condition SBI nets on a cohort -> held-out MSE results.')
    p.add_argument('--source', default='synthetic',
                   choices=('synthetic', 'real'))
    p.add_argument('--cohort', default=None,
                   help='Cohort name (required for synthetic).')
    p.add_argument('--run', default='full', help='Run label (directory level).')
    p.add_argument('--fit-target', default='update_matrix', choices=FIT_TARGETS)
    p.add_argument('--rep', default='all', choices=(*REPRESENTATIONS, 'all'))
    p.add_argument('--model', default='all', choices=(*MODEL_TYPES, 'all'))
    p.add_argument('--distribution', default=None, choices=DISTRIBUTIONS,
                   help='Phase to condition. Selects the matching networks '
                        '(snpe_{rep}_{model}_{distribution}.pkl) and, for real '
                        'data unless --preset is given, the matching sessions '
                        "(expert_<distribution>). Condition one phase per launch.")
    p.add_argument('--task-id', type=int, default=None,
                   help=f'SLURM array index 0-{N_TASKS - 1}; '
                        'overrides --rep/--model.')
    p.add_argument('--config', default=None, help='config.yaml path (real).')
    p.add_argument('--preset', default=None,
                   help='Session-selection preset (real). '
                        'Default: expert_<distribution>.')
    p.add_argument('--n-repeats', type=int, default=None,
                   help='Override repeats (multi-session path).')
    p.add_argument('--seed', type=int, default=BASE_SEED)
    p.add_argument('--smoke-test', action='store_true',
                   help=f'Use {SMOKE_N_REPEATS} repeats to check the pipeline.')
    p.add_argument('--count', action='store_true',
                   help='Print the number of array tasks and exit.')
    args = p.parse_args()

    if args.count:
        print(N_TASKS)
        return

    if not args.distribution:
        p.error('--distribution is required (uniform / hard_a / hard_b) — '
                'condition one phase per launch.')

    if args.source == 'synthetic' and not args.cohort:
        p.error("--source synthetic requires --cohort")

    if args.smoke_test:
        n_repeats = SMOKE_N_REPEATS
    else:
        n_repeats = args.n_repeats or SBI_N_CV_REPEATS

    if args.task_id is not None:
        jobs = [decode_task(args.task_id)]
    else:
        reps = REPRESENTATIONS if args.rep == 'all' else (args.rep,)
        models = MODEL_TYPES if args.model == 'all' else (args.model,)
        jobs = [(r, m) for r in reps for m in models]

    # Real sessions default to the phase-matched preset; synthetic uses --cohort
    # (whose name should encode the phase). --preset overrides for real.
    preset = args.preset or f'expert_{args.distribution}'
    cohort_label = args.cohort or 'real'
    records = load_animals(args.source, cohort=args.cohort,
                           config_path=args.config, preset=preset)
    print(f'[sbi] {len(records)} animals | source={args.source} '
          f'cohort={cohort_label} phase={args.distribution} preset={preset} '
          f'| jobs={jobs} | n_repeats={n_repeats}')

    meta = build_metadata('run_sbi', vars(args))
    t0 = time.time()
    for rep, model in jobs:
        # distribution as a path level so phases never collide on disk.
        out_dir = (results_dir('sbi', args.run, cohort_label, args.fit_target)
                   / args.distribution / rep)
        condition_cohort(records, rep, model, args.distribution, out_dir,
                         args.fit_target, n_repeats=n_repeats, seed=args.seed,
                         metadata=meta)
    print(f'[sbi] done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
