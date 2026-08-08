"""Tests for the standardised-delta compute_adaptation / _per_session."""

from datetime import date, timedelta

import numpy as np
import pytest

from behav_utils.analysis import (
    compute_adaptation, compute_adaptation_per_session, compute_stat)


def _session(session_idx, n, dist, stype, *, noise=0.12, seed=0):
    from behav_utils.data.structures import (
        SessionData, SessionMetadata, TrialData)
    rng = np.random.default_rng(seed + session_idx)
    stim = rng.uniform(-1, 1, n)
    cat = (stim > 0).astype(float)
    ch = cat.copy()
    flip = rng.random(n) < noise
    ch[flip] = 1 - ch[flip]
    opto = np.zeros(n, dtype=bool)
    if stype == 'opto':
        opto[rng.choice(n, int(n * 0.3), replace=False)] = True
    tr = TrialData(trial_number=np.arange(n), stimulus=stim, category=cat, choice=ch,
                   outcome=(ch == cat).astype(float), correct=(ch == cat),
                   abort=np.zeros(n, dtype=bool), opto_on=opto)
    return SessionData(session_id=f'{dist}_{stype}_{session_idx}', session_idx=session_idx,
                       date=date(2026, 1, 1) + timedelta(days=session_idx),
                       metadata=SessionMetadata(fields={'stage': 'Full_Task_Cont',
                                                        'distribution': dist}),
                       trials=tr, session_type=stype)


def _animal(aid='SS16'):
    from behav_utils.data.structures import AnimalData
    sessions = ([_session(i, 300, 'Uniform', 'regular', noise=0.08) for i in range(6)] +
                [_session(6, 240, 'Hard-A', 'opto'), _session(7, 260, 'Hard-A', 'opto'),
                 _session(8, 220, 'Hard-A', 'masking'), _session(9, 200, 'Hard-A', 'masking')])
    return AnimalData(animal_id=aid, sessions=sessions)


def test_per_session_shape_and_labels():
    r = compute_adaptation_per_session(_animal(), 'Hard-A',
                                       stat_names=['pse', 'accuracy'], window=50, step=50)
    assert r['mode'] == 'per_session' and r['standardised'] is True
    assert set(r['baseline']) == {'pse', 'accuracy'}
    assert 'pse' in r['normative']                         # normative line for pse
    assert len(r['sessions']) == 4                          # opto + masking both kept
    for e in r['sessions']:
        assert set(e['values']) == {'pse', 'accuracy'}
        assert len(e['values']['pse']) == len(e['trials'])
        assert 'switch_index' in e and 'session_type' in e
    # rows tagged with session + stat_metric label
    labels = {row['stat'] for row in r['rows']}
    assert 'pse_plateau' in labels and 'accuracy_auc' in labels
    assert all('session_id' in row and 'switch_index' in row for row in r['rows'])


def test_pooled_shape():
    r = compute_adaptation(_animal(), 'Hard-A', stat_names=['pse'], mode='pooled',
                           window=50, step=50)
    assert 'curve' in r and 'sessions' not in r
    assert list(r['curve']['values']) == ['pse']
    assert len(r['curve']['values']['pse']) == len(r['curve']['trials'])
    assert {row['stat'] for row in r['rows']} == {'pse_plateau', 'pse_trials_to_plateau', 'pse_auc'}


def test_standardised_is_value_minus_baseline():
    an = _animal()
    std = compute_adaptation_per_session(an, 'Hard-A', stat_names=['pse'],
                                         standardised=True, window=50, step=50)
    raw = compute_adaptation_per_session(an, 'Hard-A', stat_names=['pse'],
                                         standardised=False, window=50, step=50)
    base = std['baseline']['pse']
    for es, er in zip(std['sessions'], raw['sessions']):
        d = es['values']['pse'] - er['values']['pse']
        finite = np.isfinite(d)
        assert np.allclose(d[finite], -base, atol=1e-9)     # std = raw − baseline


def test_normative_offset_is_baseline_subtracted():
    an = _animal()
    std = compute_adaptation(an, 'Hard-A', stat_names=['pse'], mode='pooled')
    raw = compute_adaptation(an, 'Hard-A', stat_names=['pse'], standardised=False, mode='pooled')
    assert np.isclose(std['normative']['pse'],
                      raw['normative']['pse'] - std['baseline']['pse'], atol=1e-9)


def test_non_overlapping_default_step():
    # step defaults to window → non-overlapping bins
    r = compute_adaptation(_animal(), 'Hard-A', stat_names=['pse'], mode='pooled', window=50)
    assert r['step'] == 50
    n = r['curve']['n_trials']
    assert len(r['curve']['trials']) == len(range(0, n - 50 + 1, 50))


def test_no_expert_baseline_raises():
    from behav_utils.data.structures import AnimalData
    an = AnimalData(animal_id='NOEXP', sessions=[_session(0, 200, 'Hard-A', 'opto')])
    with pytest.raises(ValueError):
        compute_adaptation(an, 'Hard-A', stat_names=['pse'])


def test_bad_mode_and_empty_stats_raise():
    an = _animal()
    with pytest.raises(ValueError):
        compute_adaptation(an, 'Hard-A', stat_names=['pse'], mode='sideways')
    with pytest.raises(ValueError):
        compute_adaptation(an, 'Hard-A', stat_names=[])

def test_fixed_sigma_passthrough():
    """sigma_source='fixed' feeds sigma_value straight to compute_normative_pse
    (the report's fixed-σ normative line), bypassing the SBI placeholder; a
    different σ gives a different normative offset."""
    from behav_utils.analysis.adaptation import compute_normative_pse
    an = _animal()
    r = compute_adaptation(an, 'Hard-A', stat_names=['pse'], mode='pooled',
                           sigma_source='fixed', sigma_value=0.2)
    base = r['baseline']['pse']
    assert np.isclose(r['normative']['pse'],
                      compute_normative_pse('Hard-A', 0.2) - base, atol=1e-9)
    r2 = compute_adaptation(an, 'Hard-A', stat_names=['pse'], mode='pooled',
                            sigma_source='fixed', sigma_value=0.35)
    assert not np.isclose(r['normative']['pse'], r2['normative']['pse'])


def test_fixed_sigma_missing_value_degrades_to_nan():
    """sigma_source='fixed' without sigma_value can't resolve σ; the normative
    offset degrades to NaN (no line drawn) rather than crashing the adaptation."""
    an = _animal()
    r = compute_adaptation(an, 'Hard-A', stat_names=['pse'], mode='pooled',
                           sigma_source='fixed')          # no sigma_value
    assert np.isnan(r['normative']['pse'])
