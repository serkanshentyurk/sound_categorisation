"""
What complete blocking of learning on laser trials would do to the next trial, per animal, from its fitted BE and SC
models, read against what the animal did (``inference.blocked_update``).

    sc-blocked-update --fits all-mice/<model-identification run id>
    sc-blocked-update --fits all-mice/2026-10-23_0951_64a0f89 --jobs 4

Needs the snapshot, the saved SBI posteriors of a model-identification run (``sc-fetch results
model_identification/all-mice/<run id>``) and an opto-contrasts run of the cohort with the post-laser condition
(``sc-reports battery``; ``--opto-run``, default latest). Each animal's fits are the pooled network's posterior; an
animal without a fit or without Uniform opto sessions is left out and named. Writes
``blocked_update/<cohort>/<run_id>/``:

    predicted.csv   one row per animal, model, parameter draw and statistic: the post-laser contrast of the
                    simulated opto sessions with learning blocked on laser trials (within_blocked), without
                    (within_normal), and the difference (effect)
    summary.csv     one row per animal, model and statistic: the median effect over the draws with its 10-90 % range,
                    the observed delta-of-deltas with its interval, the margin and the verdict
    sessions.csv    the opto sessions each animal was simulated on
    meta.json

Upper estimate: 1.5 h on one core for 8 animals, both models and 50 draws; --jobs N divides it by about N.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from behav_utils.config import load_cohorts

from sound_categorisation.data.cohort import collect_sessions_ppc, gather_genotypes, load_experiment_any
from sound_categorisation.data.paths import REPO_ROOT, build_metadata, resolve_run, start_run
from sound_categorisation.inference.blocked_update import (
    DRAWS,
    MODELS,
    STATS,
    blocking_table,
    observed_dod,
    posterior_draws,
    sessions_table,
    summarise,
)
from sound_categorisation.settings import BASE_SEED, EQUIVALENCE_MARGINS, SBI_BURN_IN

DISTRIBUTION = 'Uniform'


def write_run(run: Path, experiment, animals, draws: dict, opto_run: Path, *, n_jobs: int = 1,
              burn_in: int = SBI_BURN_IN, seed: int = BASE_SEED, args: dict | None = None) -> pd.DataFrame:
    """Simulate, read the observations and write the run's tables. ``draws``: ``{(animal, model): [params, ...]}``;
    animals without Uniform opto sessions are left out. Returns the summary."""
    genotypes, _ = gather_genotypes(experiment)
    opto = {aid: collect_sessions_ppc(experiment.animals[aid], DISTRIBUTION)['opto'] for aid in animals}
    if any(not s for s in opto.values()):
        print(f'left out (no {DISTRIBUTION} opto sessions): {[aid for aid, s in opto.items() if not s]}')
    jobs = [(aid, genotypes.get(aid, 'unknown'), model, opto[aid], draws[(aid, model)])
            for aid in animals for model in MODELS if opto[aid] and draws.get((aid, model))]
    predicted = blocking_table(jobs, n_jobs=n_jobs, seed=seed, burn_in=burn_in)
    simulated = sorted(set(predicted['animal']))
    observed = [o for o in (observed_dod(opto_run, aid) for aid in simulated) if len(o)]
    if len(observed) < len(simulated):
        print(f'no post-laser observation in {opto_run} for {len(simulated) - len(observed)} animal(s)')
    summary = summarise(predicted, pd.concat(observed, ignore_index=True) if observed else pd.DataFrame(),
                        EQUIVALENCE_MARGINS)
    predicted.to_csv(run / 'predicted.csv', index=False)
    summary.to_csv(run / 'summary.csv', index=False)
    sessions_table({aid: opto[aid] for aid in simulated}).to_csv(run / 'sessions.csv', index=False)
    meta = build_metadata('blocked_update', {**(args or {}), 'opto_run': str(opto_run), 'stats': list(STATS),
                                             'burn_in': burn_in, 'seed': seed, 'margins': EQUIVALENCE_MARGINS},
                          run_id=run.name)
    (run / 'meta.json').write_text(json.dumps(meta, indent=2, default=str))
    return summary


def _fit_run(spec: str) -> Path:
    if '/' not in spec:
        sys.exit(f'--fits takes <cohort>/<run_id>, e.g. all-mice/2026-10-23_0951_64a0f89; got {spec!r}')
    cohort, run_id = spec.split('/', 1)
    return resolve_run('model_identification', cohort, run_id)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fits', required=True, help='<cohort>/<run_id> of the model-identification run whose SBI '
                                                 'posteriors are used, e.g. all-mice/2026-10-23_0951_64a0f89')
    p.add_argument('--cohort', default='opto1-cohort', help='the opto cohort in config.yaml')
    p.add_argument('--opto-run', default='latest', help='the opto-contrasts run of the cohort holding the observations')
    p.add_argument('--animals', nargs='*', default=None, help='default: every animal of the cohort')
    p.add_argument('--draws', type=int, default=DRAWS, help='posterior draws per animal and model')
    p.add_argument('--jobs', type=int, default=1, help='animal-and-model pairs computed at once')
    p.add_argument('--run-id', default=None, help='write into this run id (default: a new one)')
    p.add_argument('--snapshot', type=Path, default=None)
    p.add_argument('--config', type=Path, default=None)
    a = p.parse_args(argv)

    cohorts = load_cohorts(a.config or REPO_ROOT / 'config.yaml')
    if a.cohort not in cohorts:
        sys.exit(f'unknown cohort {a.cohort!r}; cohorts in config.yaml: {sorted(cohorts)}')
    fit_run = _fit_run(a.fits)
    opto_run = resolve_run('opto_contrasts', a.cohort, a.opto_run)
    experiment = load_experiment_any(a.config, a.snapshot)
    named = list(a.animals or cohorts[a.cohort])
    animals = [aid for aid in named if aid in experiment.animals]
    if len(animals) < len(named):
        print(f'not in the snapshot: {[aid for aid in named if aid not in experiment.animals]}')
    draws, missing = {}, []
    for aid in animals:
        for model in MODELS:
            try:
                draws[(aid, model)] = posterior_draws(fit_run, aid, model, a.draws)
            except (FileNotFoundError, ValueError) as e:
                missing.append(f'  {aid} {model}: {e}')
    if missing:
        print('left out (no fit to draw from):\n' + '\n'.join(missing))
    if not draws:
        sys.exit(f'no fits in {fit_run} for {animals}')
    run = start_run('blocked_update', a.cohort, a.run_id)
    print(f'blocked_update/{a.cohort}/{run.name}: fits {fit_run}, observations {opto_run}, {a.draws} draws')
    summary = write_run(run, experiment, animals, draws, opto_run, n_jobs=a.jobs,
                        args={'fits': str(fit_run), 'cohort': a.cohort, 'draws': a.draws, 'animals': animals})
    if len(summary):
        print(summary.groupby(['genotype', 'model', 'stat'])['verdict'].value_counts().unstack(fill_value=0)
              .to_string())
    print(f'-> {run}')
    return run


if __name__ == '__main__':
    main()
