"""Tests for inference.representation: to_stat_vector + _nan_moments.

The NaN contract: to_stat_vector produces NaN where a stat is undefined (never
raises for that), so the moments path mirrors the pooled path.
"""

import numpy as np
import pytest

from inference.representation import to_stat_vector, _nan_moments
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

    def test_moments_dim_is_4D(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        D = len(to_stat_vector(clean, mode='pooled', stat_names=SBI_STATS))
        m = to_stat_vector(clean, mode='moments', stat_names=SBI_STATS)
        assert m.shape[0] == 4 * D

    def test_too_few_sessions_raises(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:3])
        with pytest.raises(ValueError):
            to_stat_vector(clean, mode='moments', stat_names=SBI_STATS)

    def test_invalid_mode_raises(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        with pytest.raises(ValueError):
            to_stat_vector(clean, mode='bogus', stat_names=SBI_STATS)


class TestNanMoments:

    def test_supported_matches_plain_moments(self):
        from scipy.stats import skew, kurtosis
        X = np.random.default_rng(0).normal(size=(8, 3))
        expected = np.concatenate([X.mean(0), X.var(0), skew(X, 0),
                                   kurtosis(X, 0)])
        assert np.allclose(_nan_moments(X), expected)

    def test_undersupported_column_is_nan_not_raise(self):
        """A column with < 4 finite values -> its four moments are NaN
        (not a raise), so the bad row is row-filtered at train time."""
        # col0: 5 finite; col1: only 3 finite
        X = np.array([[1., np.nan], [2., np.nan], [3., 5.], [4., 6.], [5., 7.]])
        m = _nan_moments(X)   # [mean0, mean1, var0, var1, sk0, sk1, ku0, ku1]
        assert np.all(np.isfinite(m[[0, 2, 4, 6]]))   # col0 supported
        assert np.all(np.isnan(m[[1, 3, 5, 7]]))       # col1 under-supported

    def test_nan_aware_ignores_occasional_nans(self):
        # column with >= 4 finite (one NaN of six) -> finite moments
        X = np.array([[1.], [2.], [np.nan], [4.], [5.], [6.]])
        m = _nan_moments(X)
        assert np.all(np.isfinite(m))
