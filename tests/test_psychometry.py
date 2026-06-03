"""
Tests for behav_utils/analysis/psychometry.py.

Covers fit_psychometric (low-level) and compute_psychometric (session-level).
"""

import numpy as np
import pytest


class TestFitPsychometric:
    """fit_psychometric on raw arrays returns mu/sigma keys."""

    def test_returns_mu_sigma_keys(self, rng):
        """Post-rename: returns mu, sigma, lapse_low, lapse_high — NOT pse, slope."""
        from behav_utils.analysis.psychometry import fit_psychometric

        n = 500
        stim = rng.uniform(-1, 1, n)
        ch = (stim > 0).astype(float) + (rng.random(n) < 0.1) * (rng.random(n) > 0.5)
        ch = np.clip(ch, 0, 1)

        result = fit_psychometric(stim, ch)

        # Naming convention (mu/sigma everywhere, no pse/slope)
        assert 'mu' in result
        assert 'sigma' in result
        assert 'lapse_low' in result
        assert 'lapse_high' in result
        assert 'pse' not in result, "should use 'mu' not 'pse'"
        assert 'slope' not in result, "should use 'sigma' not 'slope'"

    def test_recovers_known_boundary(self, rng):
        """Synthetic data with known boundary should be recovered."""
        from behav_utils.analysis.psychometry import fit_psychometric

        # Generate data with boundary at 0.2
        n = 2000
        stim = rng.uniform(-1, 1, n)
        true_mu = 0.2
        true_sigma = 0.15
        from scipy.stats import norm
        p_right = norm.cdf((stim - true_mu) / true_sigma)
        ch = (rng.random(n) < p_right).astype(float)

        result = fit_psychometric(stim, ch)

        # mu within ±0.05 of truth
        assert abs(result['mu'] - true_mu) < 0.05, f"mu={result['mu']}, true={true_mu}"
        # sigma within reasonable range
        assert 0.05 < result['sigma'] < 0.5

    def test_returns_success_flag(self, rng):
        """Fit returns a success flag."""
        from behav_utils.analysis.psychometry import fit_psychometric

        n = 300
        stim = rng.uniform(-1, 1, n)
        ch = (stim > 0).astype(float)
        result = fit_psychometric(stim, ch)
        assert 'success' in result

    def test_lapses_in_unit_interval(self, rng):
        """Lapse rates must be valid probabilities."""
        from behav_utils.analysis.psychometry import fit_psychometric

        n = 500
        stim = rng.uniform(-1, 1, n)
        ch = (stim > 0).astype(float) + (rng.random(n) < 0.05)
        ch = np.clip(ch, 0, 1)
        result = fit_psychometric(stim, ch)

        if result.get('success', False):
            assert 0 <= result['lapse_low'] <= 1
            assert 0 <= result['lapse_high'] <= 1


class TestComputePsychometric:
    """compute_psychometric at session level."""

    def test_pooled_mode(self, synthetic_animal):
        """Pooled mode returns one curve from all sessions."""
        from behav_utils.analysis.psychometry import compute_psychometric

        clean = [s for s in synthetic_animal.sessions if not s.masking]
        result = compute_psychometric(clean, mode='pooled')

        assert 'mode' in result
        assert result['mode'] == 'pooled'
        assert 'params' in result
        assert 'mu' in result['params']

    def test_overlay_mode(self, synthetic_animal):
        """Overlay mode returns one curve per session."""
        from behav_utils.analysis.psychometry import compute_psychometric

        clean = [s for s in synthetic_animal.sessions if not s.masking][:5]
        result = compute_psychometric(clean, mode='overlay')

        assert result['mode'] == 'overlay'
        # Overlay returns per_session list of psychometric fits
        assert 'per_session' in result
        assert result['n_sessions'] == 5

    def test_per_session_mode(self, synthetic_animal):
        """Per-session mode: median over session fits + across-session CI."""
        from behav_utils.analysis.psychometry import compute_psychometric

        clean = [s for s in synthetic_animal.sessions if not s.masking][:5]
        result = compute_psychometric(clean, mode='per_session')

        assert result['mode'] == 'per_session'
        assert 'mu' in result['params']
        assert 'params_ci' in result and 'curve_band' in result and 'n_fits' in result

    def test_pooled_has_param_ci(self, synthetic_animal):
        """Pooled mode with bootstrap returns parameter CIs and a curve band."""
        from behav_utils.analysis.psychometry import compute_psychometric

        clean = [s for s in synthetic_animal.sessions if not s.masking]
        result = compute_psychometric(clean, mode='pooled', n_bootstrap=100)
        if result['success'] and result['n_fits'] > 0:
            assert result['params_ci'] is not None
            assert 'mu' in result['params_ci']
            assert result['curve_band'] is not None

    def test_per_session_ci_none_below_three(self, synthetic_animal):
        """Per-session CI is None when fewer than 3 sessions fit."""
        from behav_utils.analysis.psychometry import compute_psychometric

        clean = [s for s in synthetic_animal.sessions if not s.masking][:2]
        result = compute_psychometric(clean, mode='per_session')
        if result['n_fits'] < 3:
            assert result['params_ci'] is None
