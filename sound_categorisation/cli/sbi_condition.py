#!/usr/bin/env python
"""Condition the trained SBI networks on a cohort -> held-out MSE results.

For each (rep, model) this loads the phase-matched specialist
``snpe_networks/snpe_{rep}_{model}_{distribution}.pkl`` and runs ``condition_sbi``
on every animal, writing one neutral-schema pickle per (animal, model) into the
phase's results directory:

    <results root>/model_identification/<cohort>/<run_id>/sbi/<fit_target>/<distribution>/<rep>/<animal>_<model>.pkl
(``paths.model_id_dir``; run_gs writes ``grid_search/...`` into the same run, consensus reads both).

Each result stamps which network produced it (rep, model, distribution, the net's
filename and its own stat vector) into the metadata, so a file is self-identifying.

--distribution is required: condition ONE phase per launch (its networks + its
sessions). For real data the sessions default to expert_<distribution> unless
--preset is given; for synthetic the --cohort name should encode the phase.

Conditioning is cheap (forward passes + held-out simulation), so this runs
serially -- no partials/gather. The BE-vs-SC winner and recovery come afterwards
from ``load_cv_results`` + ``compare_models`` (run per rep dir); cross-rep /
cross-method consensus from ``analysis.consensus``.

Local (all six (rep, model) on a synthetic uniform cohort; a new run is created)::

    sc-sbi-condition --cohort synthetic_uniform \
        --distribution uniform --fit-target update_matrix

Cluster (SLURM array, one (rep, model) per task; one phase per submit; the run id is shared)::

    RUN=$(sc-new-run --report model_identification --cohort opto1-cohort)
    bash slurm/submit.sh sbi-condition --cohort opto1-cohort --distribution uniform --run-id $RUN
    bash slurm/submit.sh sbi-condition --cohort opto1-cohort --distribution hard_a  --run-id $RUN

--fast: FAST_N_REPEATS repeats to check the pipeline; run id suffixed _fast.
"""

from __future__ import annotations

import argparse
import time

from sound_categorisation.data.cohort import load_animals
from sound_categorisation.data.paths import build_metadata, model_id_dir, snpe_net_path, start_run
from sound_categorisation.inference.amortised import AmortisedSBI
from sound_categorisation.inference.cv_utils import save_cv_result
from sound_categorisation.inference.selection import condition_sbi
from sound_categorisation.inference.tasks import CONDITION_GRID
from sound_categorisation.settings import (
    BASE_SEED,
    DISTRIBUTIONS,
    FIT_TARGETS,
    GS_N_BINS,
    GS_N_FOLDS,
    MODEL_TYPES,
    SBI_N_CV_REPEATS,
    SBI_N_POSTERIOR_SAMPLES,
    SBI_REPRESENTATIONS,
)

# A net trained by TRAIN_GRID task (rep, model, *) is conditioned by CONDITION_GRID task (rep, model).
REPRESENTATIONS = tuple(SBI_REPRESENTATIONS)
N_TASKS = CONDITION_GRID.n
FAST_N_REPEATS = 2


def decode_task(task_id):
    """SLURM array index -> (rep, model); see sound_categorisation.inference.tasks.CONDITION_GRID."""
    d = CONDITION_GRID.decode(task_id)
    return d['rep'], d['model']


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


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Condition SBI nets on a cohort -> held-out MSE results.')
    p.add_argument('--cohort', required=True,
                   help='a synthetic cohort (sc-make-synthetic-cohort) or a cohort name from config.yaml')
    p.add_argument('--run-id', default=None, help='existing run id to write into (required with --task-id)')
    p.add_argument('--fit-target', default='update_matrix', choices=FIT_TARGETS)
    p.add_argument('--rep', default='all', choices=(*REPRESENTATIONS, 'all'))
    p.add_argument('--model', default='all', choices=(*MODEL_TYPES, 'all'))
    p.add_argument('--distribution', default=None, choices=DISTRIBUTIONS,
                   help='Phase to condition. Selects the matching networks '
                        '(snpe_{rep}_{model}_{distribution}.pkl) and, for real '
                        'data unless --preset is given, the matching sessions '
                        "(expert_<distribution>). Condition one phase per launch.")
    p.add_argument('--print-array', action='store_true', help='print the SLURM --array range and exit')
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
    p.add_argument('--fast', action='store_true',
                   help=f'Use {FAST_N_REPEATS} repeats to check the pipeline; run id gets _fast.')
    p.add_argument('--count', action='store_true',
                   help='Print the number of array tasks and exit.')
    args = p.parse_args(argv)
    if args.print_array:
        print(CONDITION_GRID.slurm_range())
        return

    if args.count:
        print(N_TASKS)
        return

    if not args.distribution:
        p.error('--distribution is required (uniform / hard_a / hard_b) — '
                'condition one phase per launch.')


    if args.task_id is not None and not args.run_id:
        p.error('--task-id needs --run-id (every array task must write into the same run)')
    if args.fast:
        n_repeats = FAST_N_REPEATS
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
    cohort_label = args.cohort
    records = load_animals(args.cohort, config_path=args.config, preset=preset)
    print(f'[sbi] {len(records)} animals | cohort={cohort_label} phase={args.distribution} preset={preset} '
          f'| jobs={jobs} | n_repeats={n_repeats}')

    run = start_run('model_identification', cohort_label, args.run_id, fast=args.fast)
    meta = build_metadata('run_sbi', vars(args), run_id=run.name)
    t0 = time.time()
    for rep, model in jobs:
        out_dir = model_id_dir(run, 'sbi', args.fit_target, args.distribution, rep)
        condition_cohort(records, rep, model, args.distribution, out_dir,
                         args.fit_target, n_repeats=n_repeats, seed=args.seed,
                         metadata=meta)
    print(f'[sbi] done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
