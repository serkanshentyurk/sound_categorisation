"""Tests for behav_utils.analysis.rolling.

Covers compute_rolling_stats (per-session / pooled shapes, multi-stat,
validation, short-session fallback, cohort survival) plus a bit-identity lock
proving the _windowed_pse refactor did not move any adaptation number.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from behav_utils.analysis.rolling import (
    compute_rolling_stats, _iter_windows, _family_of)


# ── builders (real SessionData, so pool_arrays / prev_* behave as in prod) ──
def _session(session_idx, n, *, noise=0.10, distribution='Hard-A',
             session_type='regular', seed=0):
    from behav_utils.data.structures import (
        SessionData, SessionMetadata, TrialData)
    rng = np.random.default_rng(seed + session_idx)
    stimuli = rng.uniform(-1, 1, n)
    categories = (stimuli > 0).astype(float)
    choices = categories.copy()
    flip = rng.random(n) < noise
    choices[flip] = 1 - choices[flip]
    trials = TrialData(
        trial_number=np.arange(n), stimulus=stimuli, category=categories,
        choice=choices, outcome=(choices == categories).astype(float),
        correct=(choices == categories), abort=np.zeros(n, dtype=bool),
        opto_on=np.zeros(n, dtype=bool))
    return SessionData(
        session_id=f'sess_{session_idx:03d}', session_idx=session_idx,
        date=date(2026, 1, 1) + timedelta(days=session_idx),
        metadata=SessionMetadata(fields={'stage': 'Full_Task_Cont',
                                          'distribution': distribution}),
        trials=trials, session_type=session_type)


class _BadSession:
    """Duck-typed session whose arrays cannot be read (cohort-survival test)."""
    session_id = 'bad_001'
    session_idx = 9
    session_type = 'opto'

    def get_arrays(self):
        raise RuntimeError("boom")


def _n_windows(n, window, step):
    return len(range(0, n - window + 1, step))


# ── _iter_windows ──────────────────────────────────────────────────────────
def test_iter_windows_full_only():
    n, window, step = 200, 50, 10
    w = _iter_windows(n, window, step)
    assert len(w) == _n_windows(n, window, step)
    assert w[0][0] == 25.0                       # centre of first window
    assert all(sl.stop <= n for _, sl in w)      # never runs past the block
    assert all((sl.stop - sl.start) == window for _, sl in w)  # full windows


def test_iter_windows_empty_when_short():
    assert _iter_windows(30, 50, 10) == []


# ── _family_of (validator internals) ───────────────────────────────────────
def test_family_of():
    assert _family_of('mu') == 'psychometric'
    assert _family_of('accuracy') == 'accuracy'
    assert _family_of('nonsense') is None
    assert _family_of('binned_accuracy') is None   # array family, not scalar


# ── per-session shape / counts ─────────────────────────────────────────────
def test_per_session_shape_and_counts():
    ns = [200, 130, 60]
    sessions = [_session(i, n) for i, n in enumerate(ns)]
    res = compute_rolling_stats(sessions, stat_names='accuracy',
                                mode='per_session', window=50, step=10)
    assert res['mode'] == 'per_session'
    assert len(res['sessions']) == len(ns)
    for e, n in zip(res['sessions'], ns):
        assert set(e['values']) == {'accuracy'}
        assert e['n_trials'] == n
        assert len(e['trials']) == _n_windows(n, 50, 10)
        assert len(e['values']['accuracy']) == len(e['trials'])
        assert e['session_type'] == 'regular'
        assert e['distribution'] == 'Hard-A'


def test_multi_stat_same_length():
    res = compute_rolling_stats([_session(0, 200)],
                                stat_names=['accuracy', 'side_bias'],
                                mode='per_session', window=50, step=10)
    e = res['sessions'][0]
    assert set(e['values']) == {'accuracy', 'side_bias'}
    for v in e['values'].values():
        assert len(v) == len(e['trials'])


# ── validation (fail fast) ─────────────────────────────────────────────────
@pytest.mark.parametrize('bad', ['nonsense', 'binned_accuracy', []])
def test_validation_rejects_bad_stats(bad):
    with pytest.raises(ValueError):
        compute_rolling_stats([_session(0, 100)], stat_names=bad)


def test_validation_rejects_bad_mode():
    with pytest.raises(ValueError):
        compute_rolling_stats([_session(0, 100)], stat_names='accuracy',
                              mode='sideways')


# ── short-session handling ─────────────────────────────────────────────────
def test_short_session_single_point():
    res = compute_rolling_stats([_session(0, 30)], stat_names='accuracy',
                                mode='per_session', window=50, step=10,
                                min_short=10)
    e = res['sessions'][0]
    assert len(e['trials']) == 1
    assert e['trials'][0] == 15.0                 # centre = n/2
    assert e['n_trials'] == 30


def test_too_short_session_empty_but_present():
    res = compute_rolling_stats([_session(0, 5)], stat_names='accuracy',
                                window=50, step=10, min_short=10)
    e = res['sessions'][0]
    assert e['trials'].size == 0
    assert e['values']['accuracy'].size == 0
    assert e['n_trials'] == 5                      # still reported


# ── cohort survival ────────────────────────────────────────────────────────
def test_never_raises_on_bad_session():
    sessions = [_session(0, 120), _BadSession()]
    with pytest.warns(RuntimeWarning):
        res = compute_rolling_stats(sessions, stat_names='accuracy',
                                    mode='per_session', window=50, step=10)
    assert len(res['sessions']) == 2
    good, bad = res['sessions']
    assert good['n_trials'] == 120 and good['trials'].size > 0
    assert bad['session_id'] == 'bad_001'
    assert bad['trials'].size == 0


# ── pooled mode (windows cross session boundaries) ─────────────────────────
def test_pooled_mode():
    ns = [80, 90]
    sessions = [_session(i, n) for i, n in enumerate(ns)]
    res = compute_rolling_stats(sessions, stat_names='accuracy',
                                mode='pooled', window=50, step=10)
    assert res['mode'] == 'pooled'
    assert 'curve' in res and 'sessions' not in res
    total = sum(ns)
    assert res['curve']['n_trials'] == total
    assert len(res['curve']['trials']) == _n_windows(total, 50, 10)
    assert len(res['curve']['values']['accuracy']) == len(res['curve']['trials'])


def test_empty_sessions():
    res = compute_rolling_stats([], stat_names='accuracy', mode='per_session')
    assert res['sessions'] == []
    res_p = compute_rolling_stats([], stat_names='accuracy', mode='pooled')
    assert res_p['curve']['n_trials'] == 0
    assert res_p['curve']['trials'].size == 0


# ── mu goes through the guarded registry ───────────────────────────────────
def test_mu_runs_and_is_finite_on_clean_session():
    # A clean, steep session should survive the reliability guard.
    res = compute_rolling_stats([_session(0, 300, noise=0.05)],
                                stat_names='mu', mode='per_session',
                                window=50, step=10)
    mu = res['sessions'][0]['values']['mu']
    assert mu.shape == res['sessions'][0]['trials'].shape
    assert np.isfinite(mu).any()


# ── bit-identity lock: the _windowed_pse refactor changed nothing ──────────
def test_windowed_pse_bit_identical():
    from behav_utils.analysis.adaptation import _windowed_pse
    from behav_utils.analysis.psychometry import fit_psychometric

    def _reference(stimuli, choices, window, step):
        # The pre-refactor loop, verbatim.
        n = stimuli.size
        if n < window:
            return np.array([]), np.array([])
        centres, pses = [], []
        for start in range(0, n - window + 1, step):
            fit = fit_psychometric(stimuli[start:start + window],
                                   choices[start:start + window])
            mu = fit.get('mu', np.nan) if isinstance(fit, dict) else np.nan
            centres.append(start + window / 2.0)
            pses.append(mu if mu is not None else np.nan)
        return np.asarray(centres, float), np.asarray(pses, float)

    rng = np.random.default_rng(3)
    for n in (40, 120, 305):                       # incl. n < window
        stim = rng.uniform(-1, 1, n)
        p = 1 / (1 + np.exp(-6 * stim))
        ch = (rng.random(n) < p).astype(float)
        c_new, p_new = _windowed_pse(stim, ch, 50, 10)
        c_ref, p_ref = _reference(stim, ch, 50, 10)
        np.testing.assert_array_equal(c_new, c_ref)
        np.testing.assert_array_equal(np.nan_to_num(p_new, nan=-999),
                                      np.nan_to_num(p_ref, nan=-999))
