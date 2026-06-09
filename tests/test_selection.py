"""Tests for inference.selection.compare_models (held-out UM/CP CV).

A stub net stands in for a trained AmortisedSBI (returning a fixed point
estimate), so the selection machinery -- fold split, fit_update_matrix,
simulate_choices, matrix_error, Wilcoxon -- is exercised without torch.
"""

import numpy as np
import pytest

from inference.types import ModelType, get_default_param_configs
from inference.selection import compare_models
from models.simulate import simulate_choices
from behav_utils.data.synthetic import session_from_arrays
from utils.stimulus_distributions import sample_distribution


def _mid(m):
    return {n: 0.5 * (c.bounds[0] + c.bounds[1])
            for n, c in get_default_param_configs(m).items()}


class _StubNet:
    """Mimics AmortisedSBI.condition: returns a fixed point estimate."""

    def __init__(self, params, burn_in=200):
        self.params = dict(params)
        self.burn_in = burn_in

    def condition(self, sessions, n_samples=50):
        return {
            'point_estimate': dict(self.params),
            'param_names': list(self.params),
            'theta_median': np.array(list(self.params.values())),
        }


def _be_sessions(n=6, seed0=100):
    be = _mid(ModelType.BE)
    out = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        s, c = sample_distribution(300, 'uniform', rng=rng)
        ch = simulate_choices(ModelType.BE, be, s, c, burn_in=200, seed=seed0 + i)
        out.append(session_from_arrays(s, ch, c, session_idx=i,
                                       distribution='uniform'))
    return out, be


def _nets(be):
    return {ModelType.BE: _StubNet(be),
            ModelType.SC: _StubNet(_mid(ModelType.SC))}


class TestCompareModels:

    def test_be_data_be_wins(self):
        """On BE-generated data, the BE net (true params) should beat SC."""
        sessions, be = _be_sessions()
        res = compare_models(sessions, _nets(be), fit_target='update_matrix',
                             n_folds=2, n_repeats=4, seed=0)
        assert set(res['per_model'].keys()) == {'be', 'sc'}
        assert res['winner'] == 'be'
        assert (res['per_model']['be']['mean_error']
                < res['per_model']['sc']['mean_error'])

    def test_result_structure(self):
        sessions, be = _be_sessions()
        res = compare_models(sessions, _nets(be), n_repeats=2)
        for m in ('be', 'sc'):
            assert {'errors', 'mean_error', 'std_error'} \
                <= set(res['per_model'][m].keys())
            assert len(res['per_model'][m]['errors']) == 2
        assert 'winner' in res and 'p_value' in res
        assert res['n_folds'] == 2

    def test_conditional_psych_target(self):
        sessions, be = _be_sessions()
        res = compare_models(sessions, _nets(be),
                             fit_target='conditional_psych', n_repeats=2)
        assert res['winner'] in ('be', 'sc')

    def test_too_few_sessions_raises(self):
        sessions, be = _be_sessions(n=1)
        with pytest.raises(ValueError):
            compare_models(sessions, _nets(be))

    def test_bad_fit_target_raises(self):
        sessions, be = _be_sessions()
        with pytest.raises(ValueError):
            compare_models(sessions, _nets(be), fit_target='bogus')
