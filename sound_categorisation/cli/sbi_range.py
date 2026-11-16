#!/usr/bin/env python
"""Where each animal's statistics fall in what BE and SC simulate (``inference.prior_range``).

An SBI network is only as good as its training simulations near the data. For every trained
multi-session representation and both models, this draws parameters from a prior, simulates them at the
training configuration, and places each animal's observation vector in the result: feature by
feature (percentile, and whether it is outside the central 99 %) and all features together (how far
the animal is from its nearest simulations). It needs no torch and no trained network, so it runs
on a laptop from the snapshot; saved results made on the cluster are read without torch too.

The prior is one of the variants of ``models.priors`` (``--prior``):

    learning     the prior in use (settings.SBI_PRIOR): what the networks are trained on (the default)
    manuscript   the ranges the networks before mid-November 2026 were trained on
    all          both, one after the other

Writes into a model-identification run of the cohort (``latest`` by default; a new run when the
cohort has none), next to the fits the table is about:

    <run>/range/<D>/<prior>/simulated.csv   quantiles of every feature, per representation, model
                                            and set (prior = every draw; passing = draws whose
                                            every session reaches the session filter's minimum
                                            accuracy)
    <run>/range/<D>/<prior>/animals.csv     per animal, representation, model and feature: value,
                                            percentile in each set, outside or not (empty for a set
                                            of fewer than 50 draws: too few to say)
    <run>/range/<D>/<prior>/summary.csv     per animal, representation and model: features outside,
                                            the worst one, nearest_percentile, and the saved
                                            posterior's acceptance rate when the run has one from a
                                            network trained on this prior
    <run>/range/<D>/<prior>/recognised.csv  per representation and model: the share of the model's
                                            draws that a classifier trained on other draws of both
                                            models calls that model (every draw; passing draws)
    <run>/range/<D>/<prior>/meta.json       the settings, and the prior's ranges
    <run>/range/<D>/comparison.csv          the priors simulated so far side by side: per prior,
                                            representation and model, how many animals are covered
                                            and how well the model's draws are recognised
    <run>/range/<D>/comparison_animals.csv  per animal and representation, which models cover it
                                            under each prior (BE+SC, BE, SC, none)

    sc-sbi-range --cohort all-mice --distribution uniform
    sc-sbi-range --cohort all-mice --distribution uniform --prior all
    sc-sbi-range --cohort all-mice --distribution uniform --prior manuscript --run-id 2026-11-10_1432_abc1234
    sc-sbi-range --cohort synthetic_uniform_k50 --distribution uniform       # animals from the prior: a check
    sc-sbi-range --cohort all-mice --distribution uniform --fast               # 200 draws, to try it

How to read it: an animal with features outside the prior set, or a nearest_percentile near 100, is
outside what a network trained on that prior sees, and its posterior and its vote for that model
are extrapolations. An animal inside the prior set but outside the passing set is in a corner of
the prior that few simulations reach. In the comparison an animal is covered by a model when no
feature is outside the prior set and nearest_percentile is at most 99. An animal that truly comes
from a model still has each feature outside one time in a hundred by chance, so about one such
animal in ten (pooled, 12 features) to one in five (moments, 22) is not counted as covered.

Coverage is what a prior gains; ``recognised`` is what it costs. A prior can cover more animals by
letting BE and SC produce the same statistics, and then no network can tell them apart there. The
classifier behind ``recognised`` is a stand-in for that (gradient boosting on the same statistics,
each draw predicted from folds it was not in); 0.5 is chance. ``recognised_passing`` is the same
over the draws that pass the session accuracy rule, where the animals are.

Time: one draw takes about 0.7 s (BE) to 1.4 s (SC); the default 2,000 draws per model take about
70 minutes on one core per prior, and about 10 on eight (``--n-jobs``, all cores by default);
``--prior all`` takes twice that. The passing set of BE is small under the manuscript prior
(about 3 % of draws): raise ``--n-draws`` for a firmer one.
"""

from __future__ import annotations

import argparse
import json
import time

from sound_categorisation.data.paths import build_metadata, model_id_dir, resolve_run, start_run
from sound_categorisation.models.priors import PRIOR_VARIANTS, log_uniform_params, prior_bounds
from sound_categorisation.settings import (
    DISTRIBUTIONS,
    MODEL_TYPES,
    SBI_BURN_IN,
    SBI_DESIGN_FEATURES,
    SBI_PRIOR,
    SBI_REPRESENTATIONS,
    SBI_TRAIN_REPS,
    VALIDATION_SEED,
)

N_DRAWS = 2000
FAST_N_DRAWS = 200


def find_run(cohort: str, run_id: str):
    """The run to write into: the one named, or the cohort's latest; a new run if it has none."""
    try:
        return resolve_run('model_identification', cohort, run_id)
    except FileNotFoundError:
        if run_id != 'latest':
            raise
        return start_run('model_identification', cohort)


def write_comparison(directory) -> dict:
    """The comparison tables of every prior simulated so far under ``directory``
    (``<run>/range/<D>``), written there; ``{}`` when there is none."""
    import pandas as pd

    from sound_categorisation.inference.prior_range import comparison_tables

    summaries = {v: pd.read_csv(directory / v / 'summary.csv') for v in PRIOR_VARIANTS
                 if (directory / v / 'summary.csv').exists()}
    if not summaries:
        return {}
    told = {v: pd.read_csv(directory / v / 'recognised.csv') for v in summaries
            if (directory / v / 'recognised.csv').exists()}
    tables = comparison_tables(summaries, told)
    for name, table in tables.items():
        table.to_csv(directory / f'{name}.csv', index=False)
    return tables


def print_summary(summary) -> None:
    """One line per animal, representation and model, worst first."""
    cols = ['animal_id', 'rep', 'model', 'n_outside', 'n_outside_passing', 'worst_feature',
            'worst_percentile', 'nearest_percentile', 'acceptance_rate']
    order = summary.sort_values(['n_outside', 'nearest_percentile'], ascending=False)
    print(order[cols].round(2).to_string(index=False))
    flagged = summary[(summary['n_outside'] > 0) | (summary['nearest_percentile'] > 99)]
    print(f'[range] {len(flagged)} of {len(summary)} animal x representation x model rows have a feature '
          'outside the simulated range or no simulation nearby (nearest_percentile > 99)')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cohort', required=True,
                   help='a cohort name from config.yaml, or a synthetic cohort (sc-make-synthetic-cohort)')
    p.add_argument('--distribution', required=True, choices=DISTRIBUTIONS,
                   help='the phase: selects the sessions (model_id_<distribution>) and the simulations')
    p.add_argument('--run-id', default='latest',
                   help="run id under model_identification/<cohort>, or 'latest' (default)")
    p.add_argument('--rep', default='all', choices=(*SBI_REPRESENTATIONS, 'all'),
                   help="representation, or 'all' = the trained multi-session ones "
                        f"({', '.join(r for r in SBI_TRAIN_REPS if SBI_REPRESENTATIONS[r]['N'] != 1)})")
    p.add_argument('--model', default='all', choices=(*MODEL_TYPES, 'all'))
    p.add_argument('--prior', default=SBI_PRIOR, choices=(*PRIOR_VARIANTS, 'all'),
                   help="prior variant (models/priors.py); 'all' = both, then compared. Default: the "
                        f"prior in use ('{SBI_PRIOR}')")
    p.add_argument('--n-draws', type=int, default=None, help=f'prior draws per model (default {N_DRAWS})')
    p.add_argument('--n-jobs', type=int, default=-1,
                   help='joblib workers for the simulations (default all cores)')
    p.add_argument('--seed', type=int, default=None, help='default VALIDATION_SEED + 2,000,000')
    p.add_argument('--config', default=None, help='config.yaml path (real cohorts)')
    p.add_argument('--preset', default=None,
                   help='session preset (real cohorts); default model_id_<distribution>')
    p.add_argument('--fast', action='store_true', help=f'{FAST_N_DRAWS} draws, to try the command')
    args = p.parse_args(argv)

    from sound_categorisation.data.cohort import preset_min_accuracy, select_animals
    from sound_categorisation.inference.constants import SBI_STATS
    from sound_categorisation.inference.prior_range import RANGE_SEED_OFFSET, range_tables

    # an animal is placed with all its sessions together, which a single-session network never sees
    if args.rep != 'all' and SBI_REPRESENTATIONS[args.rep]['N'] == 1:
        p.error(f'{args.rep} is a single-session representation: this table places an animal\'s sessions '
                'together, so it covers the multi-session representations only')
    reps = (tuple(r for r in SBI_TRAIN_REPS if SBI_REPRESENTATIONS[r]['N'] != 1) if args.rep == 'all'
            else (args.rep,))
    models = MODEL_TYPES if args.model == 'all' else (args.model,)
    priors = PRIOR_VARIANTS if args.prior == 'all' else (args.prior,)
    n_draws = args.n_draws or (FAST_N_DRAWS if args.fast else N_DRAWS)
    seed = VALIDATION_SEED + RANGE_SEED_OFFSET if args.seed is None else args.seed
    preset = args.preset or f'model_id_{args.distribution}'
    min_accuracy = preset_min_accuracy(args.distribution)

    records, _ = select_animals(args.cohort, config_path=args.config, preset=preset)
    if not records:
        p.error(f'no animal of {args.cohort} has enough sessions under {preset}')
    run = find_run(args.cohort, args.run_id)
    root = run / 'range' / args.distribution
    print(f'[range] {len(records)} animals | cohort={args.cohort} phase={args.distribution} preset={preset} '
          f'| reps={reps} models={tuple(models)} priors={tuple(priors)} | {n_draws} draws per model '
          f'-> {root}')

    t0 = time.time()
    for prior in priors:
        out = root / prior
        out.mkdir(parents=True, exist_ok=True)
        # acceptance rates are read only from posteriors of networks trained on this prior
        result_dirs = {rep: model_id_dir(run, 'sbi', 'update_matrix', args.distribution, rep) for rep in reps}
        tables = range_tables(
            records, distribution=args.distribution, representations=SBI_REPRESENTATIONS, reps=reps,
            models=models, stat_names=SBI_STATS, design_features=SBI_DESIGN_FEATURES, burn_in=SBI_BURN_IN,
            min_accuracy=min_accuracy, n_draws=n_draws, seed=seed, n_jobs=args.n_jobs,
            result_dirs=result_dirs, prior=prior, log=lambda line: print(f'[range] {line}'))
        for name, table in tables.items():
            table.to_csv(out / f'{name}.csv', index=False)
        saved = sum(len(list(d.glob('*.pkl'))) for d in result_dirs.values() if d.is_dir())
        if not saved:
            print('[range] no saved posteriors in this run: acceptance_rate is empty '
                  '(sc-sbi-condition writes them)')
        elif tables['summary']['acceptance_rate'].isna().all():
            print(f'[range] {saved} saved results found but none gave an acceptance rate for the {prior} '
                  'prior: they come from networks of another prior, or cannot be read here (then run this '
                  'command where the results were made, the cluster)')
        meta = build_metadata('sbi_range', vars(args), run_id=run.name)
        meta.update(prior=prior, n_draws=n_draws, seed=seed, min_accuracy=min_accuracy, preset=preset,
                    reps=list(reps), models=list(models), n_animals=len(records),
                    prior_bounds={m: prior_bounds(m, prior) for m in models},
                    log_uniform={m: list(log_uniform_params(m, prior)) for m in models})
        (out / 'meta.json').write_text(json.dumps(meta, indent=2, default=str))
        print(f'[range] --- {prior} ---')
        print_summary(tables['summary'])

    comparison = write_comparison(root)
    if comparison:
        print(f'[range] priors simulated so far, side by side (covered = no feature outside the prior set and '
              f"a simulation nearby; recognised = share of the model's draws a classifier calls that model) "
              f'-> {root / "comparison.csv"}')
        print(comparison['comparison'].round(2).to_string(index=False))
        print(comparison['comparison_animals'].to_string(index=False))
    print(f'[range] done in {(time.time() - t0) / 60:.1f} min -> {root}')


if __name__ == '__main__':
    main()
