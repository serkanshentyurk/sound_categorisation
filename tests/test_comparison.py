"""
Tests for behav_utils/analysis/comparison.py.

Covers the pair-comparison primitives (compare_conditions, compute_comparison).
Group-level statistics (per-animal extraction + cross-animal tests) live in the
Tier A/B layer (stats_table + group) and are tested in test_stats_table.py and
test_group.py.
"""

import numpy as np


# ============================================================================
# compare_conditions — the low-level array primitive
# ============================================================================

class TestCompareConditions:
    """compare_conditions takes raw arrays for two conditions."""

    def test_returns_expected_keys(self, rng):
        """compare_conditions returns the documented dict keys with mu/sigma naming."""
        from behav_utils.analysis.comparison import compare_conditions

        n = 500
        stim_a = rng.uniform(-1, 1, n)
        ch_a = (stim_a > 0.05).astype(float)  # boundary slightly off
        cat_a = (stim_a > 0).astype(float)

        stim_b = rng.uniform(-1, 1, n)
        ch_b = (stim_b > -0.05).astype(float)  # boundary other side
        cat_b = (stim_b > 0).astype(float)

        result = compare_conditions(stim_a, ch_a, cat_a, stim_b, ch_b, cat_b)

        # mu/sigma keys (post-rename), not pse/slope
        for top in ['params_a', 'params_b', 'diffs', 'perm_p', 'boot_ci']:
            assert top in result, f"missing top-level key: {top}"
            assert 'mu' in result[top], f"missing 'mu' in {top}"
            assert 'sigma' in result[top], f"missing 'sigma' in {top}"
        assert 'um_rmse' in result
        assert 'boot_band_a' in result
        assert 'boot_band_b' in result

    def test_boot_band_shape(self, rng):
        """Bootstrap band has expected shape."""
        from behav_utils.analysis.comparison import compare_conditions

        n = 400
        stim_a = rng.uniform(-1, 1, n)
        ch_a = (stim_a > 0).astype(float)
        cat_a = (stim_a > 0).astype(float)

        result = compare_conditions(
            stim_a, ch_a, cat_a, stim_a, ch_a, cat_a,
            n_bootstrap=50,
        )

        band_a = result['boot_band_a']
        assert 'x' in band_a
        assert 'lo' in band_a
        assert 'hi' in band_a
        assert len(band_a['x']) == len(band_a['lo']) == len(band_a['hi'])
        # Upper >= lower at every x
        assert np.all(band_a['hi'] >= band_a['lo'])

    def test_perm_p_values_in_unit_interval(self, rng):
        """All permutation p-values should be valid probabilities."""
        from behav_utils.analysis.comparison import compare_conditions

        n = 500
        stim_a = rng.uniform(-1, 1, n)
        ch_a = (stim_a > 0).astype(float)
        cat_a = (stim_a > 0).astype(float)

        stim_b = rng.uniform(-1, 1, n)
        ch_b = (stim_b > 0.3).astype(float)
        cat_b = (stim_b > 0).astype(float)

        result = compare_conditions(stim_a, ch_a, cat_a, stim_b, ch_b, cat_b,
                                     n_permutations=100)

        for key, p in result['perm_p'].items():
            if not np.isnan(p):
                assert 0 <= p <= 1, f"p_{key} = {p} out of [0,1]"


# ============================================================================
# compute_comparison — the session-level wrapper
# ============================================================================

class TestComputeComparison:
    """compute_comparison takes sessions, pools, calls compare_conditions."""

    def test_basic_call(self, synthetic_animal):
        """Returns dict with same shape as compare_conditions."""
        from behav_utils.analysis.comparison import compute_comparison
        from behav_utils.data.ops.filtering import filter_session, opto_mask

        sessions_with_opto = [s for s in synthetic_animal.sessions
                              if not s.masking and s.trials.opto_on.any()]
        assert len(sessions_with_opto) >= 2, "fixture must provide opto sessions"

        on = [filter_session(s, opto_mask(s.trials, 0))
              for s in sessions_with_opto]
        off = [filter_session(s, opto_mask(s.trials, 'control'))
               for s in sessions_with_opto]

        result = compute_comparison(
            on, off,
            label_a='opto_on', label_b='opto_off',
            n_bootstrap=50, n_permutations=50,
        )

        assert 'diffs' in result
        assert 'mu' in result['diffs']
        assert 'um_rmse' in result
        assert result['label_a'] == 'opto_on'
        assert result['label_b'] == 'opto_off'

    def test_identical_inputs_give_small_diff(self, synthetic_animal):
        """Comparing a session to itself should give near-zero diffs."""
        from behav_utils.analysis.comparison import compute_comparison

        sess = [s for s in synthetic_animal.sessions if not s.masking][:3]
        result = compute_comparison(
            sess, sess,
            n_bootstrap=20, n_permutations=20,
        )

        # mu/sigma diffs should be exactly zero (same data, both psychometric fits)
        assert abs(result['diffs']['mu']) < 1e-9
        assert abs(result['diffs']['sigma']) < 1e-9
