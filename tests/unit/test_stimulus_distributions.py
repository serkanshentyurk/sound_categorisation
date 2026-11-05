"""
Tests for utils/stimulus_distribution.py.

Covers sample_distribution, compute_distribution_density, compute_normative_pse.
"""

import numpy as np
import pytest


class TestSampleDistribution:
    """sample_distribution returns (stimuli, categories) tuples."""

    def test_uniform_returns_arrays(self, rng):
        from sound_categorisation.data.stimuli import sample_distribution
        result = sample_distribution(100, 'uniform', rng=rng)
        stim = result[0] if isinstance(result, tuple) else result
        assert isinstance(stim, np.ndarray)
        assert len(stim) == 100

    def test_hard_a_in_range(self, rng):
        from sound_categorisation.data.stimuli import sample_distribution
        result = sample_distribution(500, 'hard_a', rng=rng)
        stim = result[0] if isinstance(result, tuple) else result
        assert stim.min() >= -1.0001
        assert stim.max() <= 1.0001

    def test_hard_b_in_range(self, rng):
        from sound_categorisation.data.stimuli import sample_distribution
        result = sample_distribution(500, 'hard_b', rng=rng)
        stim = result[0] if isinstance(result, tuple) else result
        assert stim.min() >= -1.0001
        assert stim.max() <= 1.0001

    def test_reproducible(self):
        from sound_categorisation.data.stimuli import sample_distribution
        r1 = sample_distribution(100, 'hard_a', rng=np.random.default_rng(42))
        r2 = sample_distribution(100, 'hard_a', rng=np.random.default_rng(42))
        s1 = r1[0] if isinstance(r1, tuple) else r1
        s2 = r2[0] if isinstance(r2, tuple) else r2
        np.testing.assert_array_equal(s1, s2)


class TestComputeDistributionDensity:
    """compute_distribution_density(distribution, s) → {s, density_a, density_b}."""

    def test_returns_expected_keys(self):
        """Dict has s, density_a, density_b."""
        from sound_categorisation.data.stimuli import compute_distribution_density
        s = np.linspace(-1, 1, 200)
        result = compute_distribution_density('hard_a', s)
        assert isinstance(result, dict)
        assert 's' in result
        assert 'density_a' in result
        assert 'density_b' in result

    def test_density_non_negative(self):
        """All densities >= 0."""
        from sound_categorisation.data.stimuli import compute_distribution_density
        s = np.linspace(-1, 1, 200)
        for dist in ['uniform', 'hard_a', 'hard_b']:
            result = compute_distribution_density(dist, s)
            assert np.all(result['density_a'] >= 0)
            assert np.all(result['density_b'] >= 0)

    def test_a_and_b_symmetry(self):
        """For symmetric distributions, density_a(s) = density_b(-s)."""
        from sound_categorisation.data.stimuli import compute_distribution_density
        s = np.linspace(-0.99, 0.99, 200)
        # Uniform should be perfectly symmetric
        result = compute_distribution_density('uniform', s)
        # density_a(s) should equal density_b(-s) when s is symmetric around 0
        da = result['density_a']
        db = result['density_b']
        # mirror
        db_mirror = db[::-1]
        np.testing.assert_allclose(da, db_mirror, atol=1e-3)


class TestComputeNormativePSE:
    """compute_normative_pse: ideal Bayesian observer PSE."""

    def test_uniform_pse_is_zero(self):
        from sound_categorisation.data.stimuli import compute_normative_pse
        pse = compute_normative_pse(distribution='uniform', sigma_percep=0.1)
        assert abs(pse) < 0.05, f"uniform PSE should be ~0, got {pse}"

    def test_hard_a_and_hard_b_pse_opposite(self):
        from sound_categorisation.data.stimuli import compute_normative_pse
        pse_a = compute_normative_pse(distribution='hard_a', sigma_percep=0.15)
        pse_b = compute_normative_pse(distribution='hard_b', sigma_percep=0.15)
        assert abs(pse_a) > 0.01
        assert abs(pse_b) > 0.01
        assert pse_a * pse_b < 0, f"pse_a={pse_a}, pse_b={pse_b} same sign"

    def test_pse_is_scalar(self):
        from sound_categorisation.data.stimuli import compute_normative_pse
        pse = compute_normative_pse(distribution='uniform', sigma_percep=0.15)
        assert isinstance(pse, float) or np.isscalar(pse)


class TestRigHardDensity:
    """The hard side is the rig's truncated exponential (Distribution_Exp_Rate = 2.15329), not the legacy density."""

    def test_hard_side_mean_distance_is_a_third(self):
        """The rig's rate puts the mean distance from the boundary at 1/3 (the snapshot gives 0.328-0.334
        after the stable start); the legacy density gave 0.378."""
        from sound_categorisation.data.stimuli import sample_distribution
        rng = np.random.default_rng(0)
        x_a, _ = sample_distribution(200_000, 'hard_a', rng=rng)
        x_b, _ = sample_distribution(200_000, 'hard_b', rng=rng)
        assert abs(-x_a[x_a < 0].mean() - 1 / 3) < 0.005
        assert abs(x_b[x_b > 0].mean() - 1 / 3) < 0.005
        assert abs(x_a[x_a > 0].mean() - 0.5) < 0.005           # the other side stays uniform

    def test_density_integrates_to_one_and_peaks_at_the_boundary(self):
        from scipy.integrate import quad
        from sound_categorisation.data.stimuli import (
            HARD_EXP_RATE,
            compute_distribution_density,
            hard_side_density,
        )
        assert quad(lambda d: float(hard_side_density(d)), 0, 1)[0] == pytest.approx(1.0, abs=1e-9)
        peak = HARD_EXP_RATE / (1 - np.exp(-HARD_EXP_RATE))
        dens = compute_distribution_density('hard_b', np.array([0.0, 1.0, -0.5]))
        assert dens['density_b'][0] == pytest.approx(peak)
        assert dens['density_b'][1] == pytest.approx(peak * np.exp(-HARD_EXP_RATE))
        assert dens['density_a'][2] == pytest.approx(1.0)

    def test_normative_pse_pinned(self):
        """Ideal-observer PSE for the rig's density (independently computed: +0.0779 at sigma 0.2; the
        legacy density gave +0.059)."""
        from sound_categorisation.data.stimuli import compute_normative_pse
        assert compute_normative_pse('hard_a', 0.2) == pytest.approx(0.0779, abs=1e-3)
        assert compute_normative_pse('hard_b', 0.2) == pytest.approx(-0.0779, abs=1e-3)

    def test_rate_is_a_parameter(self):
        from sound_categorisation.data.stimuli import compute_normative_pse
        assert compute_normative_pse('hard_a', 0.2, rate=4.0) > compute_normative_pse('hard_a', 0.2)
