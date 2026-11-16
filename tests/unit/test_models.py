"""Tests for models.be_core and models.sc_core determinism and API."""

import numpy as np
import pytest
from sound_categorisation.models.be_core import BEModel, BEParams
from sound_categorisation.models.sc_core import SCModel, SCParams


class TestBEModel:
    """Tests for Boundary Estimation model."""

    def test_deterministic_with_seed(self):
        """Same seed → same output."""
        from sound_categorisation.models.be_core import BEModel, BEParams, BEState

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
        from sound_categorisation.models.be_core import BEModel, BEParams, BEState

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
        from sound_categorisation.models.be_core import BEParams
        bounds = BEParams.get_bounds()
        names = BEParams.get_param_names()
        assert set(bounds.keys()) == set(names)
        for lo, hi in bounds.values():
            assert lo < hi

    def test_sample_prior(self):
        """sample_prior should return valid BEParams."""
        from sound_categorisation.models.be_core import BEParams
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
        from sound_categorisation.models.sc_core import SCModel, SCParams, SCState

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
        from sound_categorisation.models.sc_core import SCModel, SCParams, SCState

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
        from sound_categorisation.models.sc_core import SCParams
        bounds = SCParams.get_bounds()
        names = SCParams.get_param_names()
        assert set(bounds.keys()) == set(names)

    def test_sample_prior(self):
        """sample_prior should return valid SCParams."""
        from sound_categorisation.models.sc_core import SCParams
        rng = np.random.default_rng(42)
        params = SCParams.sample_prior(rng)
        bounds = SCParams.get_bounds()
        for name in SCParams.get_param_names():
            val = getattr(params, name)
            lo, hi = bounds[name]
            assert lo <= val <= hi


# =============================================================================
# THE MODELS' CHOICES, PINNED
# =============================================================================

BE_MID = dict(sigma_percep=0.15, A_repulsion=0.1, eta_learning=0.3, eta_relax=0.05)
SC_MID = dict(sigma_percep=0.15, A_repulsion=0.1, gamma=0.95, sigma_update=0.3)

# 300 uniform stimuli from seed 11, burn-in 200, seed 7: the first 60 choices, how many B choices, and
# the sum of the first five P(B).
PINNED = {
    'BEParams': ('101101000000110111111001100110000100010000000000010000100000', 135, 1.5537236690369503),
    'SCParams': ('011001001001100111111010110110000110101000001100010100101000', 143, 1.8436133057103392),
}


@pytest.mark.parametrize('core, params_type, mid', [(BEModel, BEParams, BE_MID), (SCModel, SCParams, SC_MID)])
def test_the_models_choices_are_what_they_were(core, params_type, mid):
    """The trained networks, the grid search and every saved simulation assume these models: a change
    to either that alters a choice fails here."""
    stimuli = np.random.default_rng(11).uniform(-1, 1, 300)
    categories = (stimuli > 0).astype(int)
    params = params_type(**mid)
    state = core.create_initial_state(burn_in=200, params=params, seed=7)
    choices, p_B, _, _ = core.simulate_session(params, state, stimuli, categories, np.random.default_rng(7))
    first, n_b, p5 = PINNED[params_type.__name__]
    assert ''.join(str(int(c)) for c in choices[:60]) == first
    assert int(choices.sum()) == n_b and float(p_B[:5].sum()) == pytest.approx(p5, abs=1e-12)
