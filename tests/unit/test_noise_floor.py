"""The noise floor (inference/noise_floor.py): the halves are the ones the fits are scored on, a correct model scores
about the floor, a model without trial history misses an animal that has one, and sc-noise-floor writes its tables."""

import numpy as np
import pandas as pd
import pytest
from behav_utils.data import pool_arrays, session_from_arrays
from behav_utils.readouts import matrix_error
from scipy.stats import norm
from sound_categorisation.data.cohort import AnimalRecord
from sound_categorisation.inference.noise_floor import (
    animal_noise_floor,
    floor_ratios,
    model_errors,
    noise_floor_table,
    true_model_band,
)
from sound_categorisation.inference.selection import empirical_target, fold_sessions, session_halves
from sound_categorisation.inference.simulator import get_bounds_arrays, simulate_sessions, theta_to_params


def _session(i, n, rng, history=0.0, mu=0.0, sigma=0.3, lapse=0.05):
    """A session whose criterion moves towards the previous stimulus by ``history`` (0: no trial history)."""
    s = rng.uniform(-1, 1, n)
    prev = np.r_[0.0, s[:-1]]
    p = lapse + (1 - 2 * lapse) * norm.cdf((s - mu + history * np.sign(prev)) / sigma)
    ch = (rng.random(n) < p).astype(float)
    return session_from_arrays(s, ch, (s > 0).astype(float), session_idx=i)


def _animal(n_sessions=5, n=600, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [_session(i, n, rng, **kw) for i in range(n_sessions)]


def test_the_halves_are_the_fits_halves():
    sessions = _animal(n_sessions=5, n=300)
    (_, first), (_, second) = fold_sessions(sessions, 2)
    assert [s.session_idx for s in first] == [0, 1, 2] and [s.session_idx for s in second] == [3, 4]
    rows, per_session = animal_noise_floor(sessions, animal='A', repeats=1)
    row = {r['split']: r for r in rows}
    assert (row['session']['sessions_a'], row['session']['sessions_b']) == ('0,1,2', '3,4')
    expected = matrix_error(*(empirical_target(pool_arrays(h), 'update_matrix', 8) for h in (first, second)))
    assert row['session']['floor'] == pytest.approx(expected)
    a, b = session_halves(sessions[2], 2)                        # the single-session network's halves
    assert len(a.get_arrays()['stimuli']) == len(b.get_arrays()['stimuli']) == 150
    d = pd.DataFrame(per_session)
    expected = matrix_error(*(empirical_target(pool_arrays([h]), 'update_matrix', 8) for h in (a, b)))
    assert d.loc[d['rep'] == 2, 'floor'].item() == pytest.approx(expected)
    assert row['within']['floor'] == pytest.approx(d['floor'].mean()) and row['within']['n_parts'] == 5


def test_no_learning_scores_the_floor_without_history_and_misses_history():
    """Without trial history the no-learning reference is the right model, so it scores about the floor; with a
    strong pull towards the previous stimulus it misses it by far."""
    ratios = []
    for seed in range(4):
        rows, _ = animal_noise_floor(_animal(seed=seed), splits=('session',), repeats=4)
        ratios.append(rows[0]['no_learning'] / rows[0]['floor'])
    assert 0.6 < np.mean(ratios) < 1.5
    rows, _ = animal_noise_floor(_animal(seed=9, history=0.4), splits=('session',), repeats=4)
    assert rows[0]['no_learning'] / rows[0]['floor'] > 3


def test_a_correct_model_scores_about_the_floor():
    """BE and SC with their true parameters, scored as a fit is (one simulation on the held-out half, the mean over
    the two folds): about the floor itself, not half of it."""
    rng = np.random.default_rng(3)
    records = []
    for model in ('be', 'sc'):
        lo, hi = get_bounds_arrays(model)
        while sum(r.true_model == model.upper() for r in records) < 3:
            params = theta_to_params(rng.uniform(lo, hi), model)
            s = simulate_sessions(model, params, ['uniform'] * 5, 350, 1000, rng, min_accuracy=0.70)
            if s is not None:
                records.append(AnimalRecord(f'{model}{len(records)}', s, model.upper(), params))
    table, _ = noise_floor_table(records, splits=('session',), repeats=4)
    r = floor_ratios(table, split='session')
    assert 0.7 < r['true_model'].mean() < 1.4
    assert set(table['true_model_name']) == {'BE', 'SC'}
    lo, hi = true_model_band(table, 'session')
    assert lo <= r['true_model'].median() <= hi


def test_ratios_put_each_model_over_the_floor():
    table = pd.DataFrame({'animal': ['A', 'B'], 'split': 'session', 'floor': [0.02, 0.01], 'no_learning': [0.04, 0.01],
                          'true_model': np.nan})
    comparison = pd.DataFrame({'animal_id': ['A', 'B'], 'be_mean': [0.03, 0.02], 'sc_mean': [0.02, 0.05]})
    r = floor_ratios(table, model_errors(comparison), 'session').set_index('animal')
    assert r.loc['A', 'BE'] == pytest.approx(1.5) and r.loc['B', 'SC'] == pytest.approx(5.0)
    assert r.loc['A', 'no_learning'] == pytest.approx(2.0) and 'true_model' not in r
    assert true_model_band(table) is None


def test_the_command_writes_its_tables(tmp_path, monkeypatch):
    monkeypatch.setenv('SC_DATA_ROOT', str(tmp_path / 'data'))
    monkeypatch.setenv('SC_RESULTS_ROOT', str(tmp_path / 'results'))
    from sound_categorisation.cli import noise_floor
    from sound_categorisation.cli.make_synthetic_cohort import make_cohort
    make_cohort('synthetic_uniform_nf', n_per_model=1, n_sessions=4, trials=200, burn_in=200)
    run = noise_floor.main(['--cohort', 'synthetic_uniform_nf', '--repeats', '1'])
    assert run.parent.name == 'synthetic_uniform_nf' and run.parent.parent.name == 'noise_floor'
    table = pd.read_csv(run / 'noise_floor.csv')
    assert set(table['split']) == {'session', 'within'} and table['animal'].nunique() == 2
    assert table['true_model'].notna().all()
    assert len(pd.read_csv(run / "noise_floor_sessions.csv")) == 8
    assert (run / 'sessions.csv').exists() and (run / 'meta.json').exists()
