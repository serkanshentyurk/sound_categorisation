"""Tests for behav_utils.analysis.update_matrix."""

import numpy as np
import pytest

from behav_utils.analysis.update_matrix import (
    compute_update_matrix, matrix_error,
)


class TestComputeUpdateMatrix:
    """Tests for update matrix computation."""

    def test_shape(self, rng):
        """Output should be (n_bins, n_bins)."""
        n = 500
        stimuli = rng.uniform(-1, 1, n)
        categories = (stimuli > 0).astype(float)
        choices = categories.copy()

        um, _, _ = compute_update_matrix(stimuli, choices, categories, n_bins=8)
        assert um.shape == (8, 8)

    def test_symmetric_data_symmetric_um(self, rng):
        """Symmetric behaviour should give roughly symmetric UM."""
        n = 5000
        stimuli = rng.uniform(-1, 1, n)
        categories = (stimuli > 0).astype(float)
        choices = categories.copy()  # perfect observer

        um, _, _ = compute_update_matrix(stimuli, choices, categories, n_bins=8)
        # UM should be close to zero for a perfect observer
        assert np.nanmax(np.abs(um)) < 0.05

    def test_different_n_bins(self, rng):
        """Should work with different bin counts."""
        n = 500
        stimuli = rng.uniform(-1, 1, n)
        categories = (stimuli > 0).astype(float)
        choices = rng.choice([0.0, 1.0], n)

        for n_bins in [4, 6, 8, 10]:
            um, _, _ = compute_update_matrix(
                stimuli, choices, categories, n_bins=n_bins)
            assert um.shape == (n_bins, n_bins)


class TestMatrixError:
    """Tests for matrix_error (MSE between UMs)."""

    def test_identical_matrices(self):
        """Error between identical matrices should be zero."""
        um = np.random.default_rng(42).uniform(-0.1, 0.1, (8, 8))
        err = matrix_error(um, um)
        assert err == pytest.approx(0.0)

    def test_nan_handling(self):
        """NaN cells should be excluded from error."""
        um1 = np.ones((8, 8)) * 0.1
        um2 = np.ones((8, 8)) * 0.2
        um1[0, 0] = np.nan
        um2[0, 0] = np.nan

        err = matrix_error(um1, um2)
        assert not np.isnan(err)
        assert err == pytest.approx(0.01)  # (0.1)^2

    def test_commutative(self):
        """matrix_error(a, b) == matrix_error(b, a)."""
        rng = np.random.default_rng(42)
        um1 = rng.uniform(-0.1, 0.1, (8, 8))
        um2 = rng.uniform(-0.1, 0.1, (8, 8))
        assert matrix_error(um1, um2) == pytest.approx(matrix_error(um2, um1))
