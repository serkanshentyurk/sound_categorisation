"""The adaptation measure shared by the switch report and the battery (behaviour/adaptation.py): the criterion as
the 50 % point with the shape fixed, missing at the edge, convergence that reads 0 for no shift and 1 for a full
one, switches combined by span, the overnight sign, the two Hard designs, and the one bias check."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from behav_utils.data import AnimalData, SessionData, SessionMetadata, TrialArrays, TrialData
from scipy.stats import norm
from sound_categorisation.behaviour.adaptation import (
    _pse_from_params,
    bias_sessions,
    compute_switches,
    compute_trajectory,
    convergence,
    criterion_curve,
    hard_design,
    mean_convergence,
    pinned_pse,
    pool_convergence,
    pse_fixed,
)
from sound_categorisation.data.cohort import ensure_presets
from sound_categorisation.reports.tables import _qc

SHAPE = (0.4, 0.05, 0.2)        # unequal lapses: the 50 % point sits ~0.1 above the location parameter


def _arrays(mu, n, rng, shape=SHAPE):
    sigma, lo, hi = shape
    s = rng.uniform(-1, 1, n)
    p = lo + (1 - lo - hi) * norm.cdf((s - mu) / sigma)
    return TrialArrays.from_sequence((rng.random(n) < p).astype(float), s)


def _animal(schedule, n=300, sigma=0.3, lapses=(0.03, 0.08), seed=0):
    """An animal whose sessions follow ``schedule`` = [(distribution, session_type, mu), ...]."""
    ensure_presets()
    rng = np.random.default_rng(seed)
    sessions = []
    for k, (dist, stype, mu) in enumerate(schedule):
        s = rng.uniform(-1, 1, n)
        p = lapses[0] + (1 - sum(lapses)) * norm.cdf((s - mu) / sigma)
        ch = (rng.random(n) < p).astype(float)
        c = (s > 0).astype(float)
        trials = TrialData(trial_number=np.arange(n), stimulus=s, category=c, choice=ch, outcome=(ch == c).astype(float),
                           correct=(ch == c), abort=np.zeros(n, bool),
                           opto_on=rng.random(n) < (0.3 if stype in ('opto', 'masking') else 0.0),
                           reaction_time=np.full(n, 300.0))
        sessions.append(SessionData(session_id=f'{stype}{k}', session_idx=k, date=date(2024, 1, 1) + timedelta(days=k),
                                    metadata=SessionMetadata(fields={'stage': 'Full_Task_Cont', 'distribution': dist}),
                                    trials=trials, session_type=stype))
    return AnimalData(animal_id='X', sessions=sessions, metadata={'genotype': 'wt'})


def test_the_criterion_is_the_50_percent_point_not_the_location():
    rng = np.random.default_rng(0)
    est = [pinned_pse(_arrays(0.1, 3000, rng), SHAPE) for _ in range(8)]
    loc = [pse_fixed(_arrays(0.1, 3000, rng), *SHAPE) for _ in range(8)]
    assert np.mean(est) == pytest.approx(_pse_from_params(0.1, *SHAPE), abs=0.02)      # ≈ 0.20
    assert np.mean(loc) == pytest.approx(0.1, abs=0.02)


def test_a_fit_at_the_edge_of_the_range_is_missing():
    rng = np.random.default_rng(1)
    one_sided = _arrays(3.0, 400, rng, shape=(0.4, 0.02, 0.02))
    assert pse_fixed(one_sided, *SHAPE) > 0.999 and np.isnan(pinned_pse(one_sided, SHAPE))
    assert np.isnan(pinned_pse(_arrays(0.0, 40, rng), SHAPE))                            # under 50 trials


def test_the_curve_is_running_50_trial_windows():
    c = criterion_curve(_arrays(0.0, 300, np.random.default_rng(2)), SHAPE)
    assert list(c.columns) == ['trial', 'pse'] and c['trial'].iloc[0] == 25 and c['trial'].diff().dropna().eq(10).all()
    assert len(criterion_curve(_arrays(0.0, 40, np.random.default_rng(2)), SHAPE)) == 0


@pytest.mark.parametrize('moved', [0.0, 1.0])
def test_no_shift_reads_zero_and_a_full_shift_one(moved):
    """Before and after measured the same way: with unequal lapses, mixing the location parameter with a 50 % point
    (as the switch report once did) would put no shift well away from 0."""
    rng = np.random.default_rng(3)
    true_before = _pse_from_params(0.1, *SHAPE)
    normative = true_before - 0.18
    before = pinned_pse(_arrays(0.1, 4000, rng), SHAPE)
    curve = criterion_curve(_arrays(0.1 - 0.18 * moved, 4000, rng), SHAPE)
    curve['convergence'] = convergence(curve['pse'].to_numpy(), before, normative)
    assert mean_convergence(curve, (0, 4000)) == pytest.approx(moved, abs=0.08)
    assert np.isnan(convergence(0.3, 0.2, 0.21))                                         # nothing to converge to


def test_switches_are_pooled_by_span():
    assert pool_convergence([1.0, 0.0], [0.3, 0.01]) == pytest.approx(0.3 / 0.31)
    assert pool_convergence([1.0, np.nan, 5.0], [0.2, 0.2, 0.0]) == pytest.approx(1.0)  # missing and zero span left out
    assert np.isnan(pool_convergence([np.nan], [0.2]))


def test_the_designs_are_told_apart_by_runs_of_days():
    U = [('Uniform', 'regular', 0.0)] * 4
    blocks = U + [('Hard-B', 'regular', 0.0)] * 5 + [('Hard-A', 'regular', 0.0)] * 5 + [('Hard-B', 'regular', 0.0)] * 4
    daily = U + [(d, 'masking', 0.0) for d in ['Hard-A', 'Hard-B'] * 4]
    repeat = daily + [('Hard-B', 'masking', 0.0)] * 2                                    # one run of three at the end
    assert hard_design(_animal(blocks, n=60)) == 'blocked'
    assert hard_design(_animal(daily, n=60)) == 'daily' and hard_design(_animal(repeat, n=60)) == 'daily'
    assert hard_design(_animal(U, n=60)) == 'none'


def test_the_overnight_sign_comes_from_the_switch_not_from_the_noise():
    """A mouse that does not adapt and does not change overnight: the measure reads about zero. Signed by the
    end-of-session value instead (the old rule), noise alone makes it positive."""
    schedule = [('Uniform', 'regular', 0.0)] * 6 + [('Hard-B', 'regular', 0.0)] * 40
    new, old = [], []
    for seed in range(4, 8):
        sw = compute_switches(_animal(schedule, n=200, seed=seed), max_trials=400)
        ov = sw[0].overnight.dropna(subset=['toward_pre'])
        new.append(ov['toward_pre'])
        old.append(ov['delta'] * np.sign(sw[0].pre['pse_pinned'] - ov['pse_end']))
    new, old = pd.concat(new), pd.concat(old)
    se = new.std() / np.sqrt(len(new))
    assert len(new) >= 120 and abs(new.mean()) < 3 * se
    assert old.mean() > new.mean() + 3 * se


def test_a_laser_day_is_measured_on_laser_off_trials():
    U = [('Uniform', 'regular', 0.0)] * 6
    hard = [(d, 'opto', 0.0) for d in ['Hard-A', 'Hard-B'] * 3] + [(d, 'masking', 0.0) for d in ['Hard-A', 'Hard-B'] * 3]
    days = compute_switches(_animal(U + hard, n=150, seed=5))
    assert [r.switch_kind for r in days] == ['first day'] + ['laser day'] * 5 + ['mixed day'] + ['masking day'] * 5
    assert [r.trials for r in days] == ['non_opto'] * 6 + ['all'] * 6
    assert all(r.n_sessions_after == 1 for r in days)


def test_a_trajectory_starts_each_session_where_the_last_ended():
    schedule = [('Uniform', 'regular', 0.0)] * 6 + [(d, 'masking', 0.0) for d in ['Hard-A', 'Hard-B'] * 3]
    tr = compute_trajectory(_animal(schedule, seed=6), ('Hard-A', 'Hard-B'), dynamics=False, bias=True)
    s = tr.sessions
    assert s['baseline_pse'].iloc[0] == pytest.approx(tr.baseline_pse)
    np.testing.assert_allclose(s['baseline_pse'].iloc[1:].to_numpy(), s['pse_last250'].iloc[:-1].to_numpy())
    assert {'convergence_100_300', 'convergence_final', 'span', 'pse_pinned'} <= set(s.columns)
    assert tr.curves.groupby('session_idx')['trial'].diff().dropna().eq(10).all()
    assert len(tr.bias) == 6 and not tr.bias['biased_animal'].any()


def test_the_battery_takes_the_one_bias_list():
    sessions = pd.DataFrame({'session_idx': [1, 2, 3, 40], 'distribution': ['Hard-A', 'Hard-B', 'Hard-A', 'Uniform']})
    bias = pd.DataFrame({'session_idx': [1, 2, 3], 'flagged': [True, False, True], 'biased_animal': [True] * 3})
    out = _qc(sessions, bias)
    assert out['flagged'].tolist() == [True, False, True, False] and out['biased_animal'].all()
    assert not _qc(sessions, None)['flagged'].any()


def test_bias_sessions_columns():
    schedule = [('Uniform', 'regular', 0.0)] * 6 + [(d, 'masking', 0.6) for d in ['Hard-A', 'Hard-B'] * 3]
    b = bias_sessions(_animal(schedule, seed=7))
    assert len(b) == 6 and b['flagged'].all() and b['biased_animal'].all()            # |criterion| > 0.4 on A and B
