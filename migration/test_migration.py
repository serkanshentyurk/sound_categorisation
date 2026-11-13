"""
The capture and the comparison on the project's synthetic experiment: no data, no torch, a few minutes.

    pytest migration -q

They check the capture and the comparison themselves, so they pass on behav_utils 0.10 and 1.0 alike. Whether
the project's own code runs on the installed library is what a capture records in errors.csv, not what these
tests assert. capture.sh runs them first on the cluster's environment.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from behav_utils import select_sessions
from sound_categorisation.data.cohort import AnimalRecord
from sound_categorisation.data.paths import REPO_ROOT
from sound_categorisation.data.synthetic import synthetic_experiment

from migration.capture import capture
from migration.compare import compare, n_unexplained

CONFIG = REPO_ROOT / 'config.yaml'
NAMES = ['accuracy', 'side_bias', 'mu', 'sigma', 'win_stay', 'lose_shift']
SETS = ['all_sessions', 'uniform_opto', 'uniform_masking', 'design:Uniform:opto', 'design:Uniform:masking',
        'design:Hard-A:opto']


def _experiment():
    return synthetic_experiment(n_animals=2, n_trials=180)


def _records(experiment):
    return [AnimalRecord(aid, select_sessions(experiment.animals[aid], distribution='Uniform',
                                              session_type='regular'), None, None)
            for aid in ('ST00', 'ST01')]


def _capture(experiment, path, *, model_id=True, **kw) -> Path:
    return capture(experiment, path, config_path=CONFIG, records=_records(experiment) if model_id else None,
                   names=NAMES, set_names=SETS, n_sim=2, **kw)


def _session(animal, session_type, distribution):
    return next(s for s in animal.sessions
                if s.session_type == session_type and s.metadata.get('distribution') == distribution)


def _unexplained(tables) -> int:
    return n_unexplained({k: v for k, v in tables.items() if k != 'sessions'})


def test_two_captures_of_the_same_data_match(tmp_path):
    experiment = _experiment()
    a = _capture(experiment, tmp_path / 'a')
    b = _capture(experiment, tmp_path / 'b', n_jobs=2)            # the parallel path gives the same numbers
    pooled = pd.read_csv(a / 'stats_pooled.csv.gz')
    assert set(NAMES) <= set(pooled['stat']) and {'opto', 'post_opto'} <= set(pooled['trial_class'])
    for name in ('vectors', 'simulator', 'sim_sessions'):        # library-only paths, on any version
        assert len(pd.read_csv(a / 'model_id' / f'{name}.csv')), name
    tables = compare(a, b, tmp_path / 'cmp')
    assert _unexplained(tables) == 0
    assert (tables['errors']['cls'] == 'both').all()           # whatever failed, failed the same way twice
    assert (tables['sessions']['where'] == 'both').all() and (tables['sessions']['causes'] == '').all()
    assert (tables['stats_pooled']['cls'] == 'equal').all() and (tables['readouts']['cls'] == 'equal').all()
    assert 'Nothing moved without an expected cause' in (tmp_path / 'cmp' / 'summary.md').read_text()


def test_the_comparison_names_what_changed(tmp_path):
    before = _capture(_experiment(), tmp_path / 'before')
    changed = _experiment()
    st00, st01 = changed.animals['ST00'], changed.animals['ST01']
    # a change of score on one trial: expected under 1.0 (the task scores)
    scored = _session(st00, 'opto', 'Uniform')
    t = scored.trials
    k = int(np.flatnonzero(np.asarray(t.correct, dtype=bool) & ~np.asarray(t.opto_on, dtype=bool))[0])
    t.correct = np.asarray(t.correct, dtype=bool).copy()
    t.correct[k] = False
    t.outcome = np.asarray(t.outcome, dtype=float).copy()
    t.outcome[k] = 0.0
    # a changed choice: never expected
    flipped = _session(st01, 'masking', 'Uniform')
    flipped.trials.choice = np.asarray(flipped.trials.choice, dtype=float).copy()
    flipped.trials.choice[3] = 1.0 - flipped.trials.choice[3]
    # a session that is no longer loaded: expected (strict loading)
    gone = _session(st00, 'opto', 'Hard-A')
    st00.sessions = [s for s in st00.sessions if s is not gone]
    after = _capture(changed, tmp_path / 'after')

    tables = compare(before, after, tmp_path / 'cmp')
    sessions = tables['sessions'].set_index(['animal', 'key'])
    assert 'score' in sessions.loc[('ST00', scored.session_id), 'causes']
    assert sessions.loc[('ST01', flipped.session_id), 'causes'] == 'data'
    assert sessions.loc[('ST00', gone.session_id), 'where'] == 'before only'
    pooled = tables['stats_pooled']
    moved = pooled[pooled['cls'] != 'equal']
    masking_sets = ['uniform_masking', 'design:Uniform:masking']
    st01_moved = moved[(moved['animal'] == 'ST01') & moved['set'].isin(masking_sets)]
    assert len(st01_moved) and (st01_moved['verdict'] == 'unexplained').all()
    st00_moved = moved[moved['animal'] == 'ST00']
    assert len(st00_moved) and (st00_moved['verdict'] == 'expected').all()
    hard = st00_moved[st00_moved['set'] == 'design:Hard-A:opto']
    assert len(hard) and hard['causes'].str.contains('membership').all()
    assert hard['causes'].str.contains('gone').all()
    assert _unexplained(tables) > 0
    assert 'Unexplained' in (tmp_path / 'cmp' / 'summary.md').read_text()


def test_runs_are_matched_through_session_folders(tmp_path):
    experiment = _experiment()
    a = _capture(experiment, tmp_path / 'a', model_id=False)
    b = _capture(experiment, tmp_path / 'b', model_id=False)
    sessions = pd.read_csv(b / 'sessions.csv')                    # 1.0 renames sessions and may shift indices
    sessions['session_id'] = 'new_' + sessions['session_id'].astype(str)
    sessions['session_idx'] = sessions['session_idx'] + 1
    sessions.to_csv(b / 'sessions.csv', index=False)
    old = pd.read_csv(a / 'sessions.csv')
    rows = old[old['animal'] == 'ST00'].head(4)
    run_b, run_a = tmp_path / 'run_b' / 'Uniform', tmp_path / 'run_a' / 'Uniform'
    run_b.mkdir(parents=True)
    run_a.mkdir(parents=True)
    table = pd.DataFrame({'animal': rows['animal'], 'session_id': rows['session_id'],
                          'session_idx': rows['session_idx'], 'pse': np.linspace(0.1, 0.4, len(rows))})
    table.to_csv(run_b / 'trajectory.csv', index=False)
    renamed = table.assign(session_id='new_' + table['session_id'].astype(str),
                           session_idx=table['session_idx'] + 1)
    renamed.loc[renamed.index[1], 'pse'] += 0.05
    renamed.to_csv(run_a / 'trajectory.csv', index=False)
    tables = compare(a, b, tmp_path / 'cmp', runs=[(tmp_path / 'run_b', tmp_path / 'run_a')])
    runs = tables['runs']
    assert (runs['cls'] != 'before only').all() and (runs['cls'] != 'after only').all()
    assert (runs['cls'] == 'moved').sum() == 1 and (runs['cls'] == 'equal').sum() == len(rows) - 1
