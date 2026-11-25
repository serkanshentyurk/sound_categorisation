"""Predicting the Hard sessions from the Uniform fits (inference/hard_predictions.py): which sessions the run covers,
that the model carries what it learnt from session to session, that a copy keeps its session and that a copy with
the mouse's own choices measures exactly like the mouse, the scores and totals, the laser variants, jobs, and
sc-hard-predictions writing its tables."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from behav_utils.data import SessionMetadata, session_from_arrays
from sound_categorisation.data.synthetic import synthetic_experiment
from sound_categorisation.inference.blocked_update import posterior_path
from sound_categorisation.inference.cv_utils import save_cv_result
from sound_categorisation.inference.hard_predictions import (
    PREDICTED_COLUMNS,
    SCORE_COLUMNS,
    TOTAL_COLUMNS,
    TRAJECTORY_COLUMNS,
    animal_predictions,
    continuous_run,
    crps,
    hard_span,
    laser_variants,
    measure,
    percentile,
    prediction_table,
    run_animal,
    score,
    totals,
    with_choices,
)
from sound_categorisation.models import make_model
from sound_categorisation.models.priors import sample_prior

SWITCH_KW = {'min_block_trials': 300, 'max_trials': 800}      # the synthetic blocks are shorter than real ones
BE_LEARNS = dict(sigma_percep=0.15, A_repulsion=0.0, eta_learning=0.5, eta_relax=0.05)
SC_LEARNS = dict(sigma_percep=0.15, A_repulsion=0.0, gamma=0.5, sigma_update=0.2)


@pytest.fixture(scope='module')
def exp():
    return synthetic_experiment(n_animals=2, n_trials=150)


def _fitted(animal):
    return [s for s in animal.sessions if s.session_type == 'regular' and s.distribution == 'Uniform']


def _mouse(exp, aid):
    a = exp.animals[aid]
    fitted = _fitted(a)
    design, sessions, _ = hard_span(a, fitted, min_block_trials=SWITCH_KW['min_block_trials'])
    return design, run_animal(a, sessions), len(fitted)


def test_the_run_is_the_fitted_uniform_sessions_then_straight_into_the_hard_ones(exp):
    """A daily mouse: its six regular Uniform sessions, then its twelve Hard days; the Uniform masking, laser and ALM
    sessions between them are left out and listed. A blocked mouse: its Uniform sessions, then every block."""
    a = exp.animals['ST01']
    design, run, table = hard_span(a, _fitted(a))
    assert design == 'daily'
    assert table.groupby('role').size().to_dict() == {'fitted Uniform': 6, 'Hard phase': 12, 'left out': 12}
    assert [s.distribution for s in run[:6]] == ['Uniform'] * 6
    assert {s.distribution for s in run[6:]} == {'Hard-A', 'Hard-B'}
    assert set(table.loc[table['role'] == 'left out', 'session_type']) == {'masking', 'opto', 'alm_control_uni',
                                                                         'alm_control_bi'}
    b = exp.animals['SB00']
    design, run, table = hard_span(b, _fitted(b), min_block_trials=300)
    assert design == 'blocked' and len(run) == 18 and not (table['role'] == 'left out').any()


def test_a_fitted_session_in_the_hard_phase_is_an_error(exp):
    a = exp.animals['SB00']
    late = [s for s in a.sessions if s.distribution == 'Hard-A'][:1]
    with pytest.raises(ValueError, match='not before the Hard phase'):
        hard_span(a, _fitted(a) + late, min_block_trials=300)


def test_the_mouse_keeps_its_own_session_numbers(exp):
    """The run animal holds copies, renumbered from 0; the mouse's own sessions keep their indices."""
    a = exp.animals['ST01']
    before = [s.session_idx for s in a.sessions]
    _, run, _ = hard_span(a, _fitted(a))
    mouse = run_animal(a, run)
    assert [s.session_idx for s in a.sessions] == before
    assert [s.session_idx for s in mouse.sessions] == list(range(len(run)))
    assert [s.session_id for s in mouse.sessions] == [s.session_id for s in run]


def test_the_model_carries_what_it_learnt_from_one_session_into_the_next():
    """Three sessions run continuously give the same choices as one session holding all their trials, and the
    second session played on its own, from a fresh burn-in, gives other choices."""
    rng = np.random.default_rng(0)
    sessions = [session_from_arrays(rng.uniform(-1, 1, 300), np.zeros(300), session_idx=i) for i in range(3)]
    agent = make_model('BE', BE_LEARNS, burn_in=200)
    runs = continuous_run(agent, sessions, np.random.default_rng(5))
    assert [len(s.trials.choice) for s in runs] == [300] * 3
    joined = session_from_arrays(np.concatenate([s.trials.stimulus for s in sessions]), np.zeros(900))
    one = continuous_run(agent, [joined], np.random.default_rng(5))[0].trials.choice
    np.testing.assert_array_equal(np.concatenate([s.trials.choice for s in runs]), one)
    fresh = continuous_run(agent, [sessions[1]], np.random.default_rng(5))[0].trials.choice
    assert not np.array_equal(fresh, runs[1].trials.choice)


def test_a_copy_keeps_its_session_and_rescores_the_new_choices(exp):
    s = exp.animals['ST01'].sessions[10]              # a Uniform laser session
    assert s.trials.opto_on.any()
    new = 1.0 - s.trials.choice
    c = with_choices(s, new)
    assert (c.session_id, c.date, c.session_type, c.distribution) == (s.session_id, s.date, s.session_type,
                                                                      s.distribution)
    np.testing.assert_array_equal(c.trials.opto_on, s.trials.opto_on)
    np.testing.assert_array_equal(c.trials.choice, new)
    np.testing.assert_array_equal(np.asarray(c.trials.correct, bool), ~np.asarray(s.trials.correct, bool))
    np.testing.assert_array_equal(c.trials.prev_choice[1:], new[:-1])


@pytest.mark.parametrize('aid', ['SB00', 'ST01'])
def test_a_copy_with_the_mouse_own_choices_measures_like_the_mouse(exp, aid):
    """Only the choices differ between the mouse and a copy: given the mouse's own choices, a copy gives every
    readout and the trajectory exactly."""
    design, mouse, n_uniform = _mouse(exp, aid)
    observed, trajectory = measure(mouse, design, n_uniform, **SWITCH_KW)
    same = run_animal(mouse, [with_choices(s, s.trials.choice) for s in mouse.sessions])
    again, trajectory_again = measure(same, design, n_uniform, **SWITCH_KW)
    pd.testing.assert_frame_equal(observed, again)
    pd.testing.assert_frame_equal(trajectory, trajectory_again)
    expected = ({'pse_before', 'pse_block', 'block_convergence', 'hard_accuracy', 'hard_a_minus_hard_b'}
                if design == 'blocked' else {'convergence_100_300'})
    assert set(observed['readout']) == expected and len(trajectory)


def test_crps_and_percentile():
    assert crps([0.3], 1.0) == pytest.approx(0.7)                  # a point prediction scores its absolute error
    assert crps([0.0, 1.0], 0.5) == pytest.approx(0.25)
    assert crps(np.linspace(0, 1, 10001), 0.5) == pytest.approx(1 / 12, abs=1e-4)    # uniform on [0, 1]
    assert np.isnan(crps([np.nan], 0.0)) and np.isnan(crps([1.0], np.nan))
    assert percentile([0, 1, 2, 3], 1.5) == 0.5
    assert percentile([0, 1, 2, 3], 1) == 0.375                     # ties count half


def _rows(animal, model, values, readout='hard_a_minus_hard_b', level='mouse', design='blocked'):
    return pd.DataFrame({'model': model, 'laser_learning': True, 'draw': range(len(values)), 'animal': animal,
                         'design': design, 'level': level, 'switch_idx': -1, 'direction': '', 'kind': '',
                         'readout': readout, 'value': values})[PREDICTED_COLUMNS]


def test_scores_and_totals():
    """A model whose predictions sit on the observation scores zero; the totals keep kept and flagged mice apart, and
    readouts that are not scored (a single day) are left out."""
    predicted = pd.concat([_rows('M1', 'BE', [0.5] * 10), _rows('M1', 'SC', [2.0] * 10),
                           _rows('M2', 'BE', [0.0] * 10), _rows('M2', 'SC', [1.0] * 10),
                           _rows('M1', 'BE', [0.3] * 10, readout='convergence_100_300', level='switch',
                                 design='daily')], ignore_index=True)
    observed = pd.DataFrame({'animal': ['M1', 'M2'], 'design': 'blocked', 'level': 'mouse', 'switch_idx': -1,
                             'direction': '', 'kind': '', 'readout': 'hard_a_minus_hard_b', 'value': [0.5, 0.0]})
    s = score(predicted, observed, groups={'M1': 'kept', 'M2': 'flagged'})
    assert list(s.columns) == SCORE_COLUMNS and len(s) == 4
    by = s.set_index(['animal', 'model'])
    assert by.loc[('M1', 'BE'), 'crps'] == 0 and by.loc[('M1', 'SC'), 'crps'] == pytest.approx(1.5)
    assert by.loc[('M1', 'BE'), 'percentile'] == 0.5 and by.loc[('M1', 'SC'), 'percentile'] == 0.0
    t = totals(s)
    assert list(t.columns) == TOTAL_COLUMNS
    sums = t.set_index(['group', 'model'])['crps_sum']
    assert sums[('kept', 'SC')] == pytest.approx(1.5) and sums[('flagged', 'SC')] == pytest.approx(1.0)
    missing = totals(score(_rows('M3', 'BE', [0.1] * 5), observed.assign(animal='M3', value=np.nan)))
    assert missing['n'].tolist() == [0] and missing['crps_sum'].isna().all()     # nothing scored is not a zero


def test_learning_off_on_laser_trials_matters_only_where_there_are_laser_trials(exp):
    """A blocked mouse has no laser trials: both variants give the same numbers, so it is run with learning on only.
    A daily mouse has laser days: a strong learner that does not learn on laser trials chooses differently."""
    design, mouse, n = _mouse(exp, 'SB00')
    assert laser_variants(mouse.sessions) == [True]
    on = animal_predictions(mouse, design, n, 'BE', [BE_LEARNS], laser_learning=True, burn_in=100, **SWITCH_KW)
    off = animal_predictions(mouse, design, n, 'BE', [BE_LEARNS], laser_learning=False, burn_in=100, **SWITCH_KW)
    pd.testing.assert_frame_equal(on[0].drop(columns='laser_learning'), off[0].drop(columns='laser_learning'))
    design, mouse, n = _mouse(exp, 'ST01')
    assert laser_variants(mouse.sessions) == [True, False] and laser_variants(mouse.sessions, (False,)) == [False]
    on = animal_predictions(mouse, design, n, 'BE', [BE_LEARNS], laser_learning=True, burn_in=100, **SWITCH_KW)
    off = animal_predictions(mouse, design, n, 'BE', [BE_LEARNS], laser_learning=False, burn_in=100, **SWITCH_KW)
    assert list(on[0].columns) == PREDICTED_COLUMNS and list(on[1].columns) == TRAJECTORY_COLUMNS
    assert not np.allclose(on[0]['value'], off[0]['value'], equal_nan=True)


def _as_loaded(session):
    """``session`` as the loader builds it: the distribution on every trial and none in the metadata."""
    labels = np.full(session.n_trials, session.distribution, dtype=object)
    fields = {k: v for k, v in session.metadata.fields.items() if k != 'distribution'}
    return replace(session, trials=replace(session.trials, distribution=labels), metadata=SessionMetadata(fields))


@pytest.mark.parametrize('aid', ['SB00', 'ST01'])
def test_copies_of_loaded_sessions_keep_their_switches(exp, aid):
    """Loaded sessions carry the distribution per trial, synthetic ones in their metadata. A copy keeps it either
    way; otherwise its blocks and days lose their distribution and the model is never scored."""
    design, mouse, n = _mouse(exp, aid)
    loaded = run_animal(mouse, [_as_loaded(s) for s in mouse.sessions])
    assert [s.distribution for s in loaded.sessions] == [s.distribution for s in mouse.sessions]
    assert all(s.metadata.get('distribution') is None for s in loaded.sessions)
    observed, _ = measure(loaded, design, n, **SWITCH_KW)
    predicted, _ = animal_predictions(loaded, design, n, 'BE', [BE_LEARNS], burn_in=100, **SWITCH_KW)
    assert len(observed) and set(predicted['readout']) == set(observed['readout'])


def test_jobs_give_the_same_numbers(exp):
    design, mouse, n = _mouse(exp, 'SB00')
    jobs = [(mouse, design, n, 'BE', True, [BE_LEARNS]), (mouse, design, n, 'SC', True, [SC_LEARNS])]
    one = prediction_table(jobs, n_jobs=1, burn_in=100, **SWITCH_KW)
    two = prediction_table(jobs, n_jobs=2, burn_in=100, **SWITCH_KW)
    pd.testing.assert_frame_equal(one[0], two[0])
    pd.testing.assert_frame_equal(one[1], two[1])
    assert set(one[0]['model']) == {'BE', 'SC'}


def test_the_command_writes_its_tables(tmp_path, monkeypatch, exp):
    """sc-hard-predictions on the synthetic experiment: posteriors and the fitted-session table from a
    model-identification run, an animal without fits left out, a wrong cohort refused with the valid ones."""
    monkeypatch.setenv('SC_RESULTS_ROOT', str(tmp_path / 'results'))
    from sound_categorisation.cli import hard_predictions as cli
    from sound_categorisation.data.paths import start_run

    monkeypatch.setattr(cli, 'load_experiment_any', lambda config, snapshot: exp)
    monkeypatch.setattr(cli, 'load_cohorts', lambda path: {'synthetic': ['ST00', 'ST01']})
    fits = start_run('model_identification', 'all-mice', 'fits')
    rng = np.random.default_rng(0)
    for model in ('BE', 'SC'):
        names, thetas = sample_prior(model, None, 2, rng)
        save_cv_result(posterior_path(fits, 'ST01', model), 'ST01', model, [], 'update_matrix',
                       posterior={'posterior_samples': thetas, 'param_names': names})
    (fits / 'sessions').mkdir(parents=True, exist_ok=True)
    dates = [str(s.date) for s in _fitted(exp.animals['ST01'])]
    pd.DataFrame({'animal_id': 'ST01', 'kept': True, 'date': dates}).to_csv(fits / 'sessions' / 'uniform.csv',
                                                                             index=False)
    run = cli.main(['--fits', 'all-mice/fits', '--cohort', 'synthetic', '--draws', '1', '--laser-learning', 'on'])
    assert run.parent.name == 'synthetic' and run.parent.parent.name == 'hard_predictions'
    for name in ('observed', 'predicted', 'scores', 'totals', 'trajectories', 'sessions'):
        assert (run / f'{name}.csv').exists(), name
    scores = pd.read_csv(run / 'scores.csv')
    assert set(scores['animal']) == {'ST01'} and set(scores['model']) == {'BE', 'SC'}
    assert set(scores['readout']) == {'convergence_100_300'} and set(scores['laser_learning']) == {True}
    assert set(pd.read_csv(run / 'trajectories.csv')['source']) == {'observed', 'BE', 'SC'}
    assert (run / 'meta.json').exists()
    with pytest.raises(SystemExit, match='cohorts in config.yaml'):
        cli.main(['--fits', 'all-mice/fits', '--cohort', 'nope'])
