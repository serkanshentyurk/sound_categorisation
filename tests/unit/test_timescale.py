"""The timescale analysis (behaviour/timescale.py): the integrator against a trial-by-trial loop, the
steady-state swing, recovery of a known memory, the drift that block order adds, and the variants."""

import numpy as np
import pandas as pd
import pytest
from sound_categorisation.behaviour.timescale import (
    SESSION_COLUMNS,
    VARIANTS,
    _design,
    compute_timescale,
    fit_integrator,
    fit_shift,
    integrator_means,
    swing_fraction,
    timescale_tables,
)
from sound_categorisation.data.synthetic import synthetic_experiment

NORM = {'Uniform': 0.0, 'Hard-A': 0.09, 'Hard-B': -0.09}
BLOCKED = ['Hard-B'] * 5 + ['Hard-A'] * 5 + ['Hard-B'] * 4 + ['Hard-A'] * 4
DAILY = ['Hard-A', 'Hard-B'] * 6


def _sessions(schedule, memory, *, trials=400, offset=0.15, gain=1.0, drift=0.0, noise=0.01, laser=None,
              light=0.0, seed=0):
    """A session table as session_sequence builds it, its PSE made by the integrator itself."""
    rng = np.random.default_rng(seed)
    n = len(schedule)
    laser = np.zeros(n, bool) if laser is None else np.asarray(laser, bool)
    targets = np.array([NORM[d] for d in schedule])
    model = integrator_means(targets, [trials] * n, [memory])[:, 0]
    k = np.arange(n, dtype=float)
    pse = offset + gain * model + drift * k + light * laser + rng.normal(0, noise, n)
    return pd.DataFrame({
        'animal': 'X', 'order': np.arange(n), 'session_idx': np.arange(n) + 10, 'session_id': [f's{i}' for i in range(n)],
        'distribution': schedule, 'session_type': np.where(laser, 'opto', 'masking'), 'laser': laser,
        'n_trials': trials, 'normative_pse': targets, 'pse_all': pse, 'pse_laser_off': pse - light * laser,
        'lapse_low': 0.05, 'lapse_high': 0.05, 'accuracy': 0.8})[SESSION_COLUMNS]


def test_the_closed_form_is_the_trial_by_trial_integrator():
    rng = np.random.default_rng(1)
    targets, lengths = rng.normal(0, 0.1, 9), rng.integers(0, 600, 9)
    lengths[3] = 0                                                   # a session without trials
    memory = np.array([1.0, 7.0, 150.0, 2000.0, np.inf])
    got = integrator_means(targets, lengths, memory, start=0.05)
    for j, n in enumerate(memory):
        c, want = 0.05, []
        for m, t in zip(targets, lengths):
            values = []
            for _ in range(t):
                values.append(c)
                if np.isfinite(n):
                    c += (m - c) / n
            want.append(np.mean(values) if t else np.nan)
        np.testing.assert_allclose(got[:, j], want, rtol=1e-9, atol=1e-12)
    with pytest.raises(ValueError):
        integrator_means(targets, lengths, [0.5])


def test_the_steady_state_swing_matches_a_long_alternation():
    for trials, memory, want in [(400, 1000, 0.0126), (400, 200, 0.2374), (6000, 1000, 0.6683)]:
        schedule = ['Hard-A', 'Hard-B'] * 40
        means = integrator_means([NORM[d] for d in schedule], [trials] * 80, [memory])[:, 0]
        sim = (means[-20::2].mean() - means[-19::2].mean()) / (NORM['Hard-A'] - NORM['Hard-B'])
        assert sim == pytest.approx(want, abs=2e-3)
        assert float(swing_fraction(trials, memory)) == pytest.approx(want, abs=2e-3)
    assert float(swing_fraction(400, np.inf)) == 0.0 and float(swing_fraction(400, 1e-9)) == pytest.approx(1.0)


def test_a_known_memory_is_recovered_from_blocks():
    seq = _sessions(BLOCKED, memory=800, drift=0.004)
    fit = fit_integrator(seq, 'main', 'fixed')
    assert 400 < fit.n_hat < 1600 and fit.n_low > 200 and fit.n_high < 3000
    assert fit.drift == pytest.approx(0.004, abs=0.003) and fit.offset == pytest.approx(0.15, abs=0.03)
    assert fit.gain == 1.0 and fit.n_sessions == len(BLOCKED)
    assert fit.profile['f'].min() == pytest.approx(0.0) and len(fit.predicted) == len(BLOCKED)


@pytest.mark.parametrize('schedule,memory', [(BLOCKED, 800), (DAILY, 150)])
def test_the_memory_interval_covers_the_truth_about_as_often_as_it_says(schedule, memory):
    fits = [fit_integrator(_sessions(schedule, memory=memory, drift=0.004, seed=s), 'main', 'fixed')
            for s in range(120)]
    rate = np.mean([f.n_low <= memory <= f.n_high for f in fits])
    assert 0.89 <= rate <= 0.99                                      # nominal 0.95


def test_a_free_gain_reads_a_partial_shift_as_partial_and_not_as_slow():
    seq = _sessions(BLOCKED, memory=300, gain=0.5, noise=0.004, seed=3)
    free = fit_integrator(seq, 'main', 'free')
    assert free.gain == pytest.approx(0.5, abs=0.15) and free.n_low <= 300 <= free.n_high
    fixed = fit_integrator(seq, 'main', 'fixed')
    assert fixed.n_hat > free.n_hat                                  # forced to the normative shift, it slows down


def test_a_daily_alternation_without_a_swing_bounds_the_memory_from_below_only():
    seq = _sessions(DAILY, memory=20_000, seed=2)
    fit = fit_integrator(seq, 'main', 'fixed')
    assert np.isinf(fit.n_high) and fit.n_low > 200
    fast = fit_integrator(_sessions(DAILY, memory=50, seed=2), 'main', 'fixed')
    assert fast.n_high < 1000                                        # a mouse that follows is caught


def test_the_shift_with_the_drift_taken_out():
    """B blocks first and an upward drift: the plain A-B difference is positive with no real shift;
    the regression takes the drift out."""
    seq = _sessions(BLOCKED, memory=np.inf, drift=0.02, noise=0.005, seed=4)
    hard = seq[seq['distribution'].str.startswith('Hard')]
    naive = hard.loc[hard['distribution'] == 'Hard-A', 'pse_all'].mean() - hard.loc[hard['distribution'] == 'Hard-B', 'pse_all'].mean()
    assert naive > 0.05
    s = fit_shift(seq, 'main')
    assert abs(s.shift) < 0.02 and s.drift == pytest.approx(0.02, abs=0.005)
    assert s.normative_difference == pytest.approx(0.18) and s.drift_low < 0.02 < s.drift_high
    real = fit_shift(_sessions(BLOCKED, memory=30, drift=0.02, noise=0.005, seed=4), 'main')
    assert real.fraction == pytest.approx(1.0, abs=0.15) and real.row()['fraction_low'] > 0.5


@pytest.mark.parametrize('schedule', [BLOCKED, DAILY])
def test_the_shift_interval_covers_the_truth_about_as_often_as_it_says(schedule):
    fits = [fit_shift(_sessions(schedule, memory=np.inf, drift=0.02, noise=0.005, seed=s), 'main')
            for s in range(300)]
    rate = np.mean([f.shift_low <= 0 <= f.shift_high for f in fits])
    assert 0.91 <= rate <= 0.98                                      # nominal 0.95


def test_the_variants_choose_which_sessions_are_fitted():
    """Laser sessions carry an extra constant: main leaves them out, the laser variants fit it."""
    laser = np.array([True] * 6 + [False] * 6)
    seq = _sessions(DAILY, memory=20_000, laser=laser, light=0.1, noise=0.005, seed=5)
    main = fit_integrator(seq, 'main', 'fixed')
    assert main.n_sessions == 6 and np.isnan(main.light)
    for variant, light in (('all_trials', 0.1), ('laser_off', 0.0)):
        f = fit_integrator(seq, variant, 'fixed')
        assert f.n_sessions == 12 and f.light == pytest.approx(light, abs=0.02)
    assert np.isnan(fit_integrator(seq.iloc[:4], 'main', 'fixed').n_hat)     # too few sessions: no fit


def test_designs_are_told_apart_by_their_runs():
    assert _design(pd.DataFrame({'distribution': BLOCKED})) == 'blocked'
    assert _design(pd.DataFrame({'distribution': DAILY})) == 'alternation'


def test_animals_of_both_designs_from_the_synthetic_experiment():
    exp = synthetic_experiment()
    blocked = compute_timescale(exp.animals['SB00'])
    daily = compute_timescale(exp.animals['ST01'])
    assert blocked.design == 'blocked' and daily.design == 'alternation'
    assert {f.variant for f in blocked.fits} == {'main'} and {f.variant for f in daily.fits} == set(VARIANTS)
    assert daily.fit('main').n_sessions == 6 and daily.fit('laser_off').n_sessions == 12
    assert 'predicted_main' in blocked.sessions and blocked.sessions['laser'].sum() == 0
    T = timescale_tables([blocked, daily, None], {'SB00': 'wt', 'ST01': 'het'})
    assert set(T) == {'sessions', 'fits', 'shifts', 'profiles'}
    assert set(T['fits']['animal']) == {'SB00', 'ST01'} and len(T['fits']) == 2 + 6
    assert set(T['fits'].loc[T['fits']['animal'] == 'ST01', 'genotype']) == {'het'}
    assert {'design', 'biased_animal', 'sigma'} <= set(T['shifts'])
