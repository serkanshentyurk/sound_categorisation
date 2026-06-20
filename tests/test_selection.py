"""Tests for inference.selection.condition_sbi (held-out UM/CP CV producer).

A stub net stands in for a trained AmortisedSBI (returning a fixed point
estimate), so the conditioning machinery -- fold split, within-session trial
split, fit_update_matrix, simulate_choices, matrix_error -- is exercised without
torch. condition_sbi only PRODUCES per-rep MSE results; the BE-vs-SC winner is
utils.cv_utils.compare_models and is tested there, not here.
"""

import numpy as np
import pytest

from inference.types import ModelType, get_default_param_configs
from inference.selection import condition_sbi
from models.simulate import simulate_choices
from behav_utils.data.synthetic import session_from_arrays
from utils.stimulus_distributions import sample_distribution


def _mid(m):
    return {n: 0.5 * (c.bounds[0] + c.bounds[1])
            for n, c in get_default_param_configs(m).items()}


class _StubNet:
    """Mimics AmortisedSBI.condition: returns a fixed point estimate.

    ``N`` selects the path in condition_sbi (1 -> single-session, else multi).
    ``raise_on_condition`` emulates a degenerate (NaN-stat) observation, which
    the real condition() rejects with ValueError -- used to test skipping.
    """

    def __init__(self, params, N=15, burn_in=200, raise_on_condition=False):
        self.params = dict(params)
        self.N = N
        self.burn_in = burn_in
        self._raise = raise_on_condition

    def condition(self, sessions, n_samples=50):
        if self._raise:
            raise ValueError('non-finite stats (stub)')
        return {
            'point_estimate': dict(self.params),
            'param_names': list(self.params),
            'theta_median': np.array(list(self.params.values())),
        }


def _be_sessions(n=6, n_trials=300, seed0=100):
    be = _mid(ModelType.BE)
    out = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        s, c = sample_distribution(n_trials, 'uniform', rng=rng)
        ch = simulate_choices(ModelType.BE, be, s, c, burn_in=200, seed=seed0 + i)
        out.append(session_from_arrays(s, ch, c, session_idx=i,
                                       distribution='uniform'))
    return out, be


# ── multi-session path (pooled / moments; net.N > 1) ─────────────────────────

class TestConditionMulti:

    def test_result_structure(self):
        sessions, be = _be_sessions()
        res = condition_sbi(sessions, _StubNet(be, N=15), ModelType.BE,
                            n_repeats=4, n_posterior_samples=10, seed=0)
        assert isinstance(res, list) and len(res) == 4          # one per repeat
        assert [r['rep'] for r in res] == [0, 1, 2, 3]
        for r in res:
            assert set(r) == {'rep', 'test_error', 'best_params'}
            assert np.isfinite(r['test_error'])
            assert isinstance(r['best_params'], dict)           # all-session theta

    def test_best_params_stable_across_reps(self):
        # one all-session conditioning -> identical recovered params every rep
        sessions, be = _be_sessions()
        res = condition_sbi(sessions, _StubNet(be, N=15), ModelType.BE,
                            n_repeats=3, n_posterior_samples=10)
        assert all(r['best_params'] == res[0]['best_params'] for r in res)

    def test_conditional_psych_target(self):
        sessions, be = _be_sessions()
        res = condition_sbi(sessions, _StubNet(be, N=15), ModelType.BE,
                            fit_target='conditional_psych',
                            n_repeats=2, n_posterior_samples=10)
        assert len(res) == 2 and all(np.isfinite(r['test_error']) for r in res)

    def test_too_few_sessions_raises(self):
        sessions, be = _be_sessions(n=1)
        with pytest.raises(ValueError):
            condition_sbi(sessions, _StubNet(be, N=15), ModelType.BE)

    def test_degenerate_obs_skipped(self):
        sessions, be = _be_sessions()
        res = condition_sbi(sessions, _StubNet(be, N=15, raise_on_condition=True),
                            ModelType.BE, n_repeats=4, n_posterior_samples=10)
        assert res == []                                        # every rep skipped


# ── single-session path (net.N == 1) ─────────────────────────────────────────

class TestConditionSingle:

    def test_per_session_entries(self):
        sessions, be = _be_sessions(n=5)
        res = condition_sbi(sessions, _StubNet(be, N=1), ModelType.BE,
                            n_posterior_samples=10, seed=0)
        assert isinstance(res, list) and 1 <= len(res) <= 5     # one per usable session
        for r in res:
            assert set(r) == {'rep', 'test_error', 'best_params', 'n_valid'}
            assert r['rep'] in range(5)
            assert np.isfinite(r['test_error'])
            assert r['n_valid'] > 0
            assert isinstance(r['best_params'], dict)

    def test_rep_is_session_index_and_n_valid_correct(self):
        sessions, be = _be_sessions(n=3, n_trials=300)
        res = condition_sbi(sessions, _StubNet(be, N=1), ModelType.BE,
                            n_posterior_samples=10)
        for r in res:
            ch = sessions[r['rep']].get_arrays()['choices']     # rep maps back to session
            assert r['n_valid'] == int(np.sum(~np.isnan(ch)))

    def test_single_does_not_require_two_sessions(self):
        sessions, be = _be_sessions(n=1)                        # unlike the multi path
        res = condition_sbi(sessions, _StubNet(be, N=1), ModelType.BE,
                            n_posterior_samples=10)
        assert len(res) == 1

    def test_short_session_skipped(self):
        sessions, be = _be_sessions(n=1, n_trials=6)            # < 8 trials -> unsplittable
        res = condition_sbi(sessions, _StubNet(be, N=1), ModelType.BE,
                            n_posterior_samples=10)
        assert res == []

    def test_degenerate_session_skipped(self):
        sessions, be = _be_sessions(n=3)
        res = condition_sbi(sessions, _StubNet(be, N=1, raise_on_condition=True),
                            ModelType.BE, n_posterior_samples=10)
        assert res == []                                        # all sessions skipped


# ── shared guards ────────────────────────────────────────────────────────────

class TestConditionGuards:

    def test_bad_fit_target_raises(self):
        sessions, be = _be_sessions()
        with pytest.raises(ValueError):
            condition_sbi(sessions, _StubNet(be, N=15), ModelType.BE,
                          fit_target='bogus')
