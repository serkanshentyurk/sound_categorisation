"""Tests for models.BE_core and models.SC_core determinism and API."""

import numpy as np
import pytest


class TestBEModel:
    """Tests for Boundary Estimation model."""

    def test_deterministic_with_seed(self):
        """Same seed → same output."""
        from models.BE_core import BEParams, BEState, BEModel

        params = BEParams(
            sigma_percep=0.1, A_repulsion=0.3,
            eta_learning=0.5, eta_relax=0.2,
        )
        stimuli = np.linspace(-1, 1, 100)
        categories = (stimuli > 0).astype(float)

        state1 = BEState.initial_uniform()
        rng1 = np.random.default_rng(42)
        choices1, _, _, _ = BEModel.simulate_session(
            params, state1, stimuli, categories, rng1)

        state2 = BEState.initial_uniform()
        rng2 = np.random.default_rng(42)
        choices2, _, _, _ = BEModel.simulate_session(
            params, state2, stimuli, categories, rng2)

        np.testing.assert_array_equal(choices1, choices2)

    def test_choices_are_binary(self):
        """All choices should be 0 or 1."""
        from models.BE_core import BEParams, BEState, BEModel

        params = BEParams(
            sigma_percep=0.1, A_repulsion=0.3,
            eta_learning=0.5, eta_relax=0.2,
        )
        stimuli = np.random.default_rng(0).uniform(-1, 1, 200)
        categories = (stimuli > 0).astype(float)
        state = BEState.initial_uniform()
        rng = np.random.default_rng(42)

        choices, _, _, _ = BEModel.simulate_session(
            params, state, stimuli, categories, rng)
        assert set(np.unique(choices)).issubset({0.0, 1.0})

    def test_param_bounds(self):
        """get_bounds should return dict with all param names."""
        from models.BE_core import BEParams
        bounds = BEParams.get_bounds()
        names = BEParams.get_param_names()
        assert set(bounds.keys()) == set(names)
        for name, (lo, hi) in bounds.items():
            assert lo < hi

    def test_sample_prior(self):
        """sample_prior should return valid BEParams."""
        from models.BE_core import BEParams
        rng = np.random.default_rng(42)
        params = BEParams.sample_prior(rng)
        bounds = BEParams.get_bounds()
        for name in BEParams.get_param_names():
            val = getattr(params, name)
            lo, hi = bounds[name]
            assert lo <= val <= hi


class TestSCModel:
    """Tests for Stimulus Category model."""

    def test_deterministic_with_seed(self):
        """Same seed → same output."""
        from models.SC_core import SCParams, SCState, SCModel

        params = SCParams(
            sigma_percep=0.2, A_repulsion=0.2,
            gamma=0.9, sigma_update=0.5,
        )
        stimuli = np.linspace(-1, 1, 100)
        categories = (stimuli > 0).astype(float)

        state1 = SCState.initial_uniform()
        rng1 = np.random.default_rng(42)
        choices1, _, _, _ = SCModel.simulate_session(
            params, state1, stimuli, categories, rng1)

        state2 = SCState.initial_uniform()
        rng2 = np.random.default_rng(42)
        choices2, _, _, _ = SCModel.simulate_session(
            params, state2, stimuli, categories, rng2)

        np.testing.assert_array_equal(choices1, choices2)

    def test_choices_are_binary(self):
        """All choices should be 0 or 1."""
        from models.SC_core import SCParams, SCState, SCModel

        params = SCParams(
            sigma_percep=0.2, A_repulsion=0.2,
            gamma=0.9, sigma_update=0.5,
        )
        stimuli = np.random.default_rng(0).uniform(-1, 1, 200)
        categories = (stimuli > 0).astype(float)
        state = SCState.initial_uniform()
        rng = np.random.default_rng(42)

        choices, _, _, _ = SCModel.simulate_session(
            params, state, stimuli, categories, rng)
        assert set(np.unique(choices)).issubset({0.0, 1.0})

    def test_param_bounds(self):
        """get_bounds should return dict with all param names."""
        from models.SC_core import SCParams
        bounds = SCParams.get_bounds()
        names = SCParams.get_param_names()
        assert set(bounds.keys()) == set(names)

    def test_sample_prior(self):
        """sample_prior should return valid SCParams."""
        from models.SC_core import SCParams
        rng = np.random.default_rng(42)
        params = SCParams.sample_prior(rng)
        bounds = SCParams.get_bounds()
        for name in SCParams.get_param_names():
            val = getattr(params, name)
            lo, hi = bounds[name]
            assert lo <= val <= hi
