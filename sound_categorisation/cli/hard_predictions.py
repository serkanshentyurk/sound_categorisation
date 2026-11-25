"""
Which model, fitted on Uniform sessions only, predicts how the criterion moves on the Hard sessions
(``inference.hard_predictions``; thesis analysis 3).

    sc-hard-predictions --fits all-mice/<model-identification run id> --cohort behaviour1-cohort
    sc-hard-predictions --fits all-mice/2026-10-23_0951_64a0f89 --cohort opto1-cohort --jobs 8

Needs the snapshot and a model-identification run's saved SBI posteriors and fitted-session table
(``sc-fetch results model_identification/all-mice/<run id>``). Each animal of the cohort is run from its fitted
Uniform sessions straight into its Hard sessions; the mouse, and for each model and parameter draw a copy whose
choices come from the model, are measured by the switch report's own computation. On laser trials the model learns as
on any other, or not at all (``--laser-learning``; both by default, for animals with laser trials). An animal without
a fit, without fitted sessions or without Hard sessions is left out and named. Writes
``hard_predictions/<cohort>/<run_id>/``:

    observed.csv      one row per animal and readout: the animal's own value
    predicted.csv     one row per animal, model, laser variant, parameter draw and readout
    scores.csv        one row per scored readout, animal, model and laser variant: the observed value, the median and
                      10-90 % range of the predictions, the percentile, the CRPS, the bias check's group
    totals.csv        the CRPS summed per model and laser variant over the animals, per readout and group
    trajectories.csv  the criterion within each switch: the animal's own, and the median and 10-90 % range of each
                      model's
    sessions.csv      every session of each animal from its first fitted one to its last Hard one, with its role in
                      the run ('fitted Uniform', 'Hard phase', 'left out', 'Hard before the fits')
    meta.json

Upper estimate, scaled from timings on the synthetic mice: about 10 s of one core per draw and model for a pre-opto
mouse, 5 s for an opto mouse and each of its two laser variants; with 200 draws about 12 CPU-hours for
behaviour1-cohort and 9 for opto1-cohort. ``--jobs N`` divides it by about N, up to one job per animal, model and
laser variant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from behav_utils.config import load_cohorts

from sound_categorisation.behaviour.adaptation import bias_sessions
from sound_categorisation.data.cohort import gather_genotypes, load_experiment_any
from sound_categorisation.data.paths import REPO_ROOT, build_metadata, resolve_run, start_run
from sound_categorisation.inference.blocked_update import posterior_draws
from sound_categorisation.inference.hard_predictions import (
    DRAWS,
    MODELS,
    fitted_sessions,
    hard_span,
    laser_variants,
    measure,
    observed_bands,
    prediction_table,
    run_animal,
    score,
    sessions_table,
    totals,
)
from sound_categorisation.settings import BASE_SEED, SBI_BURN_IN

LASER_CHOICES = {'both': (True, False), 'on': (True,), 'off': (False,)}


def _group(animal) -> str:
    """'flagged' when the bias check (``adaptation.bias_sessions``) calls the animal biased, else 'kept'."""
    b = bias_sessions(animal)
    return 'flagged' if len(b) and bool(b['biased_animal'].astype(bool).any()) else 'kept'


def write_run(run: Path, experiment, animals, draws: dict, fitted: dict, *, laser_learning=(True, False),
              n_jobs: int = 1, burn_in: int = SBI_BURN_IN, seed: int = BASE_SEED, switch_kw: dict | None = None,
              args: dict | None = None) -> pd.DataFrame:
    """Measure each animal and its model copies and write the run's tables. ``draws``: ``{(animal, model): [params,
    ...]}``; ``fitted``: ``{animal: [fitted Uniform sessions]}``; ``switch_kw`` goes to ``compute_switches``
    (``min_block_trials``, ``min_day_trials``, ``max_trials``). Returns the totals."""
    genotypes, _ = gather_genotypes(experiment)
    switch_kw = dict(switch_kw or {})
    span_kw = {k: switch_kw[k] for k in ('min_block_trials', 'min_day_trials') if k in switch_kw}
    jobs, observed, bands, tables, groups, left = [], [], [], [], {}, []
    for aid in animals:
        if not fitted.get(aid) or not any(draws.get((aid, m)) for m in MODELS):
            continue
        animal = experiment.animals[aid]
        try:
            design, sessions, table = hard_span(animal, fitted[aid], **span_kw)
            mouse = run_animal(animal, sessions)
            o, t = measure(mouse, design, len(fitted[aid]), **switch_kw)
            group = _group(animal)
        except Exception as e:                            # one animal must not end the run
            left.append(f'  {e}' if str(e).startswith(aid) else f'  {aid}: {e!r}')
            continue
        observed.append(o)
        bands.append(observed_bands(t))
        tables.append(table)
        groups[aid] = group
        for model in MODELS:
            if not draws.get((aid, model)):
                continue
            for learns in laser_variants(sessions, laser_learning):
                jobs.append((mouse, design, len(fitted[aid]), model, learns, draws[(aid, model)]))
    if left:
        print('left out:\n' + '\n'.join(left))
    print(f'{len(observed)} animal(s), {len(jobs)} job(s) of animal, model and laser variant')
    predicted, predicted_bands = prediction_table(jobs, n_jobs=n_jobs, seed=seed, burn_in=burn_in, **switch_kw)
    obs = pd.concat(observed, ignore_index=True) if observed else pd.DataFrame()
    scores = score(predicted, obs, groups, genotypes)
    total = totals(scores)
    obs.to_csv(run / 'observed.csv', index=False)
    predicted.to_csv(run / 'predicted.csv', index=False)
    scores.to_csv(run / 'scores.csv', index=False)
    total.to_csv(run / 'totals.csv', index=False)
    pd.concat([b for b in bands + [predicted_bands] if len(b)] or [predicted_bands],
              ignore_index=True).to_csv(run / 'trajectories.csv', index=False)
    sessions_table(tables).to_csv(run / 'sessions.csv', index=False)
    meta = build_metadata('hard_predictions', {**(args or {}), 'laser_learning': list(laser_learning),
                                               'burn_in': burn_in, 'seed': seed, 'switch_kw': switch_kw,
                                               'groups': groups},
                          run_id=run.name)
    (run / 'meta.json').write_text(json.dumps(meta, indent=2, default=str))
    return total


def _fit_run(spec: str) -> Path:
    if '/' not in spec:
        sys.exit(f'--fits takes <cohort>/<run_id>, e.g. all-mice/2026-10-23_0951_64a0f89; got {spec!r}')
    cohort, run_id = spec.split('/', 1)
    return resolve_run('model_identification', cohort, run_id)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fits', required=True, help='<cohort>/<run_id> of the model-identification run whose SBI '
                                                 'posteriors are used, e.g. all-mice/2026-10-23_0951_64a0f89')
    p.add_argument('--cohort', required=True, help='a cohort in config.yaml, e.g. behaviour1-cohort or opto1-cohort')
    p.add_argument('--animals', nargs='*', default=None, help='default: every animal of the cohort')
    p.add_argument('--draws', type=int, default=DRAWS, help='posterior draws per animal and model')
    p.add_argument('--laser-learning', default='both', choices=sorted(LASER_CHOICES),
                   help='on laser trials the model learns (on), does not (off), or both are run (default)')
    p.add_argument('--jobs', type=int, default=1, help='jobs of animal, model and laser variant run at once')
    p.add_argument('--run-id', default=None, help='write into this run id (default: a new one)')
    p.add_argument('--snapshot', type=Path, default=None)
    p.add_argument('--config', type=Path, default=None)
    a = p.parse_args(argv)

    cohorts = load_cohorts(a.config or REPO_ROOT / 'config.yaml')
    if a.cohort not in cohorts:
        sys.exit(f'unknown cohort {a.cohort!r}; cohorts in config.yaml: {sorted(cohorts)}')
    fit_run = _fit_run(a.fits)
    experiment = load_experiment_any(a.config, a.snapshot)
    named = list(a.animals or cohorts[a.cohort])
    unknown = [aid for aid in named if aid not in cohorts[a.cohort]]
    if unknown:
        sys.exit(f'not in {a.cohort}: {unknown}; its animals: {cohorts[a.cohort]}')
    animals = [aid for aid in named if aid in experiment.animals]
    if len(animals) < len(named):
        print(f'not in the snapshot: {[aid for aid in named if aid not in experiment.animals]}')
    draws, fitted, missing = {}, {}, []
    for aid in animals:
        for model in MODELS:
            try:
                draws[(aid, model)] = posterior_draws(fit_run, aid, model, a.draws)
            except (FileNotFoundError, ValueError) as e:
                missing.append(f'  {aid} {model}: {e}')
        if any((aid, m) in draws for m in MODELS):
            try:
                sessions = fitted_sessions(fit_run, experiment.animals[aid])
            except ValueError as e:
                missing.append(f'  {e}')
                continue
            if sessions:
                fitted[aid] = sessions
            else:
                missing.append(f'  {aid}: no fitted sessions in {fit_run / "sessions"}')
    if missing:
        print('left out (no fit to draw from, or its sessions not found):\n' + '\n'.join(missing))
    if not fitted:
        sys.exit(f'no fits in {fit_run} for {animals}')
    run = start_run('hard_predictions', a.cohort, a.run_id)
    print(f'hard_predictions/{a.cohort}/{run.name}: fits {fit_run}, {a.draws} draws, '
          f'laser learning {a.laser_learning}')
    total = write_run(run, experiment, list(fitted), draws, fitted, laser_learning=LASER_CHOICES[a.laser_learning],
                      n_jobs=a.jobs, args={'fits': str(fit_run), 'cohort': a.cohort, 'draws': a.draws,
                                           'animals': list(fitted)})
    if len(total):
        print(total.pivot_table(index=['design', 'readout', 'kind', 'group'], columns=['model', 'laser_learning'],
                                values='crps_sum').to_string())
    print(f'-> {run}')
    return run


if __name__ == '__main__':
    main()
