"""Tests for inference.representation: to_stat_vector + _nan_moments.

The NaN contract: to_stat_vector produces NaN where a stat is undefined (never
raises for that), so the moments path mirrors the pooled path.
"""

import numpy as np
import pytest

from inference.representation import (
    to_stat_vector, _nan_moments, compute_feature_medians, impute_with_medians)
from inference.constants import SBI_STATS
from behav_utils.data.ops.filtering import filter_trials, pool_arrays
from behav_utils.analysis.summary_stats import fit_summary_stats


class TestToStatVectorPooled:

    def test_pooled_matches_direct_pool_and_fit(self, synthetic_animal):
        """Pooled to_stat_vector == direct pool_arrays + fit_summary_stats,
        i.e. it reuses the real seam-aware path (no separate pooling)."""
        clean = filter_trials(synthetic_animal.sessions[:5])
        v = to_stat_vector(clean, mode='pooled', stat_names=SBI_STATS)
        p = pool_arrays(clean)
        direct = fit_summary_stats(
            p['choices'], p['stimuli'], p['categories'],
            prev_choices=p['prev_choices'], prev_stimuli=p['prev_stimuli'],
            prev_categories=p['prev_categories'], stat_names=SBI_STATS,
            return_dict=False,
        )
        assert np.allclose(v, direct, equal_nan=True)

    def test_pooled_finite(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        v = to_stat_vector(clean, mode='pooled', stat_names=SBI_STATS)
        assert np.all(np.isfinite(v))


class TestToStatVectorMoments:

    def test_moments_dim_is_2D(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        D = len(to_stat_vector(clean, mode='pooled', stat_names=SBI_STATS))
        m = to_stat_vector(clean, mode='moments', stat_names=SBI_STATS)
        assert m.shape[0] == 2 * D                       # [mean(D), var(D)]

    def test_too_few_sessions_raises(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:1])   # 1 session < 2
        with pytest.raises(ValueError):
            to_stat_vector(clean, mode='moments', stat_names=SBI_STATS)

    def test_two_sessions_ok(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:2])   # >= 2 is fine now
        D = len(to_stat_vector(clean, mode='pooled', stat_names=SBI_STATS))
        m = to_stat_vector(clean, mode='moments', stat_names=SBI_STATS)
        assert m.shape[0] == 2 * D

    def test_invalid_mode_raises(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        with pytest.raises(ValueError):
            to_stat_vector(clean, mode='bogus', stat_names=SBI_STATS)


class TestNanMoments:

    def test_supported_matches_plain_moments(self):
        X = np.random.default_rng(0).normal(size=(8, 3))
        expected = np.concatenate([X.mean(0), X.var(0)])     # mean, var (ddof=0)
        assert np.allclose(_nan_moments(X), expected)

    def test_undersupported_column_is_nan_not_raise(self):
        """A column with < 2 finite values -> its mean+var are NaN (not a raise),
        handled downstream by the median-impute."""
        # col0: 3 finite; col1: only 1 finite
        X = np.array([[1., np.nan], [2., np.nan], [3., 5.]])
        m = _nan_moments(X)   # [mean0, mean1, var0, var1]
        assert np.all(np.isfinite(m[[0, 2]]))    # col0 supported (mean0, var0)
        assert np.all(np.isnan(m[[1, 3]]))       # col1 under-supported (mean1, var1)

    def test_nan_aware_ignores_occasional_nans(self):
        # column with >= 2 finite (one NaN of four) -> finite moments
        X = np.array([[1.], [np.nan], [4.], [5.]])
        assert np.all(np.isfinite(_nan_moments(X)))


class TestImpute:

    def test_compute_feature_medians_ignores_nan(self):
        X = np.array([[1., 10.], [3., np.nan], [np.nan, 30.], [5., 50.]])
        med = compute_feature_medians(X)
        assert np.allclose(med, [np.nanmedian([1, 3, 5]), np.nanmedian([10, 30, 50])])

    def test_all_nan_column_raises(self):
        X = np.array([[1., np.nan], [2., np.nan], [3., np.nan]])
        with pytest.raises(ValueError):
            compute_feature_medians(X)

    def test_impute_vector_fills_with_median(self):
        out = impute_with_medians(np.array([1., np.nan, 3.]), np.array([0., 5., 9.]))
        assert np.allclose(out, [1., 5., 3.]) and np.all(np.isfinite(out))

    def test_impute_matrix_fills_per_column(self):
        X = np.array([[1., np.nan], [np.nan, 2.]])
        out = impute_with_medians(X, np.array([100., 200.]))
        assert np.allclose(out, [[1., 200.], [100., 2.]])

    def test_impute_copies_and_leaves_finite(self):
        x = np.array([1., 2.])
        out = impute_with_medians(x, np.array([0., 0.]))
        assert np.allclose(out, x) and out is not x
