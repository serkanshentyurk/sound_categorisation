"""
The noise floor of the held-out update-matrix error, and a no-history reference, per animal (``inference.noise_floor``).

    sc-noise-floor --cohort all-mice
    sc-noise-floor --cohort synthetic_uniform_learning_k100 --jobs 6

Takes each animal's sessions exactly as the fits do (``cohort.select_animals``, preset ``model_id_<distribution>``)
and writes ``noise_floor/<cohort>/<run_id>/``:

    noise_floor.csv           one row per animal and split: floor, no_history, true_model (synthetic cohorts),
                              trials per half, and for the session split the sessions in each half
    noise_floor_sessions.csv  the within-session split, one row per session (rep = the session's position, as
                              the single-session conditioning numbers its reps)
    sessions.csv              which sessions each animal has (compare with the fit run's sessions/<distribution>.csv)
    meta.json

Seconds to minutes from the snapshot, on the laptop; a synthetic cohort also simulates its true models.
"""

from __future__ import annotations

import argparse
import json

from sound_categorisation.data.cohort import select_animals
from sound_categorisation.data.paths import build_metadata, start_run
from sound_categorisation.inference.noise_floor import REPEATS, SPLITS, noise_floor_table
from sound_categorisation.settings import DISTRIBUTIONS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cohort', required=True, help='a cohort in config.yaml, or a synthetic cohort')
    p.add_argument('--distribution', default='uniform', choices=list(DISTRIBUTIONS))
    p.add_argument('--preset', default=None, help='session preset (default model_id_<distribution>, as the fits)')
    p.add_argument('--splits', nargs='+', default=list(SPLITS), choices=list(SPLITS))
    p.add_argument('--repeats', type=int, default=REPEATS, help='simulations averaged per held-out half')
    p.add_argument('--fit-target', default='update_matrix', choices=['update_matrix', 'conditional_psych'])
    p.add_argument('--jobs', type=int, default=1, help='animals at once')
    p.add_argument('--run-id', default=None, help='write into this run id (default: a new one)')
    p.add_argument('--config', default=None)
    a = p.parse_args(argv)

    preset = a.preset or f'model_id_{a.distribution}'
    records, sessions_table = select_animals(a.cohort, config_path=a.config, preset=preset)
    run = start_run('noise_floor', a.cohort, a.run_id)
    print(f'noise_floor/{a.cohort}/{run.name}: {len(records)} animals, splits {a.splits}, {a.repeats} repeats')
    table, per_session = noise_floor_table(records, n_jobs=a.jobs, splits=a.splits, fit_target=a.fit_target,
                                           repeats=a.repeats)
    table.to_csv(run / 'noise_floor.csv', index=False)
    per_session.to_csv(run / 'noise_floor_sessions.csv', index=False)
    sessions_table.to_csv(run / 'sessions.csv', index=False)
    meta = build_metadata('noise_floor', {'cohort': a.cohort, 'distribution': a.distribution, 'preset': preset,
                                          'splits': a.splits, 'repeats': a.repeats, 'fit_target': a.fit_target},
                          run_id=run.name)
    (run / 'meta.json').write_text(json.dumps(meta, indent=2, default=str))
    summary = table.groupby('split')[['floor', 'no_history', 'true_model']].median()
    print(summary.round(4).to_string())
    print(f'-> {run}')
    return run


if __name__ == '__main__':
    main()
