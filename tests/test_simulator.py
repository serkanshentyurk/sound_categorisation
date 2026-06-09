"""Tests for inference.simulator: build_simulator, theta_to_params, the
distribution schedule, and the guard conditions.

The prior is torch/sbi-backed; in a torch-free environment build_simulator
returns prior=None, so these tests assert only on sim_fn / param_names.
"""

import numpy as np
import pytest

from inference.types import ModelType
from inference.constants import SBI_STATS
from inference.simulator import (
    build_simulator, theta_to_params, get_param_names, get_bounds_arrays,
)


def _mid(model):
    lo, hi = get_bounds_arrays(model)
    return 0.5 * (lo + hi)


class TestParamLayout:

    def test_param_names(self):
        assert get_param_names(ModelType.BE) == \
            ['sigma_percep', 'A_repulsion', 'eta_learning', 'eta_relax']
        assert get_param_names(ModelType.SC) == \
            ['sigma_percep', 'A_repulsion', 'gamma', 'sigma_update']

    def test_bounds_shape_ordered(self):
        lo, hi = get_bounds_arrays(ModelType.BE)
        assert lo.shape == (4,) and hi.shape == (4,)
        assert np.all(hi > lo)

    def test_theta_to_params_roundtrip(self):
        theta = _mid(ModelType.BE)
        p = theta_to_params(theta, ModelType.BE)
        names = get_param_names(ModelType.BE)
        assert set(p.keys()) == set(names)
        for i, n in enumerate(names):
            assert p[n] == pytest.approx(theta[i])

    def test_theta_wrong_length_raises(self):
        with pytest.raises(ValueError):
            theta_to_params(np.zeros(3), ModelType.BE)


class TestBuildSimulator:

    def test_pooled_sim_finite(self):
        sim, prior, names = build_simulator(
            ModelType.BE, 'uniform', N=5, T=300, burn_in=200,
            mode='pooled', stat_names=SBI_STATS)
        x = sim(_mid(ModelType.BE), seed=1)
        assert np.all(np.isfinite(x))
        assert names == get_param_names(ModelType.BE)

    def test_reproducible_per_seed(self):
        sim, _, _ = build_simulator(
            ModelType.BE, 'uniform', N=5, T=300, burn_in=200,
            mode='pooled', stat_names=SBI_STATS)
        th = _mid(ModelType.BE)
        assert np.allclose(sim(th, seed=1), sim(th, seed=1), equal_nan=True)
        assert not np.allclose(sim(th, seed=1), sim(th, seed=2), equal_nan=True)

    def test_moments_dim_is_4D(self):
        sim_m, _, _ = build_simulator(
            ModelType.BE, 'uniform', N=5, T=300, burn_in=200,
            mode='moments', stat_names=SBI_STATS)
        sim_p, _, _ = build_simulator(
            ModelType.BE, 'uniform', N=5, T=300, burn_in=200,
            mode='pooled', stat_names=SBI_STATS)
        D = len(sim_p(_mid(ModelType.BE), seed=1))
        assert sim_m(_mid(ModelType.BE), seed=1).shape[0] == 4 * D

    def test_distribution_schedule(self):
        sim, _, _ = build_simulator(
            ModelType.BE, ['uniform', 'hard_a', 'hard_b', 'hard_a', 'uniform'],
            N=5, T=300, burn_in=200, mode='pooled', stat_names=SBI_STATS)
        assert np.all(np.isfinite(sim(_mid(ModelType.BE), seed=3)))

    def test_sc_model(self):
        sim, _, names = build_simulator(
            ModelType.SC, 'uniform', N=5, T=300, burn_in=200,
            mode='pooled', stat_names=SBI_STATS)
        assert names == get_param_names(ModelType.SC)
        assert np.all(np.isfinite(sim(_mid(ModelType.SC), seed=1)))

    def test_moments_too_few_sessions_raises(self):
        with pytest.raises(ValueError):
            build_simulator(ModelType.BE, 'uniform', N=2, mode='moments')

    def test_bad_schedule_length_raises(self):
        with pytest.raises(ValueError):
            build_simulator(ModelType.BE, ['uniform', 'hard_a'], N=5)
