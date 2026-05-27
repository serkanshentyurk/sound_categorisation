"""
Tests for utils/cv_utils.py.

Update matrix CV helpers used by GS-CV and SBI-CV pipelines.
"""

import numpy as np
import pandas as pd
import pytest


def _make_trial_df(rng, n=500):
    """Build a flat trial DataFrame with the columns compute_empirical_um expects."""
    stim = rng.uniform(-1, 1, n)
    cat = (stim > 0).astype(int)
    ch = cat.copy()
    flip = rng.random(n) < 0.2
    ch[flip] = 1 - ch[flip]
    return pd.DataFrame({
        'stim_relative': stim,
        'choice': ch,
        'No_response': np.zeros(n, dtype=bool),
        'is_not_start_of_block': np.ones(n, dtype=bool),
    })


class TestComputeEmpiricalUM:
    """compute_empirical_um returns (update_matrix, conditional_matrix)."""

    def test_returns_tuple(self, rng):
        from utils.cv_utils import compute_empirical_um
        df = _make_trial_df(rng, n=400)
        result = compute_empirical_um(df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_update_matrix_shape(self, rng):
        from utils.cv_utils import compute_empirical_um
        df = _make_trial_df(rng, n=400)
        um, cm = compute_empirical_um(df)
        assert um.shape == (8, 8)

    def test_finite_or_nan(self, rng):
        from utils.cv_utils import compute_empirical_um
        df = _make_trial_df(rng, n=400)
        um, cm = compute_empirical_um(df)
        assert np.all(np.isfinite(um) | np.isnan(um))


class TestParamsToStr:
    def test_returns_string(self):
        from utils.cv_utils import params_to_str
        params = {'sigma_percep': 0.1, 'eta_learning': 0.3}
        result = params_to_str(params)
        assert isinstance(result, str)
        assert len(result) > 0
