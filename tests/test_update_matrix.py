"""Tests for behav_utils.analysis.update_matrix."""

import numpy as np
import pytest

from behav_utils.analysis.update_matrix import (
    fit_update_matrix, compute_um, matrix_error,
)
from behav_utils.data.ops.filtering import filter_trials, pool_arrays


class TestFitUpdateMatrix:
    """Tests for update matrix computation."""

    def test_shape(self, rng):
        """Output should be (n_bins, n_bins)."""
        n = 500
        stimuli = rng.uniform(-1, 1, n)
        categories = (stimuli > 0).astype(float)
        choices = categories.copy()

        um, _, _ = fit_update_matrix(stimuli, choices, categories, n_bins=8)
        assert um.shape == (8, 8)

    def test_symmetric_data_symmetric_um(self, rng):
        """Symmetric behaviour should give roughly symmetric UM."""
        n = 5000
        stimuli = rng.uniform(-1, 1, n)
        categories = (stimuli > 0).astype(float)
        choices = categories.copy()  # perfect observer

        um, _, _ = fit_update_matrix(stimuli, choices, categories, n_bins=8)
        # UM should be close to zero for a perfect observer
        assert np.nanmax(np.abs(um)) < 0.05

    def test_different_n_bins(self, rng):
        """Should work with different bin counts."""
        n = 500
        stimuli = rng.uniform(-1, 1, n)
        categories = (stimuli > 0).astype(float)
        choices = rng.choice([0.0, 1.0], n)

        for n_bins in [4, 6, 8, 10]:
            um, _, _ = fit_update_matrix(
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


class TestComputeUm:
    """Session-level compute_um: pooling via pool_arrays + the frozen prev_*."""

    def test_pooled_shape(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        r = compute_um(clean, mode='pooled')
        assert r['mode'] == 'pooled'
        assert r['um'].shape == (8, 8)
        assert r['n_sessions'] == 5

    def test_per_session_returns_tagged_list(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        r = compute_um(clean, mode='per_session')
        assert r['mode'] == 'per_session'
        assert isinstance(r['per_session'], list)
        assert len(r['per_session']) == 5
        entry = r['per_session'][0]
        for k in ('session_id', 'session_idx', 'um', 'conditional_matrix', 'n_trials'):
            assert k in entry
        assert entry['um'].shape == (8, 8)

    def test_prev_equals_adjacency_on_opto_free(self, synthetic_animal):
        """On opto-free data the frozen-prev UM equals the old adjacency UM
        (prev_has_prev is the adjacency not_blockstart)."""
        clean = filter_trials(synthetic_animal.sessions[:5])
        p = pool_arrays(clean)
        um_adj, _, _ = fit_update_matrix(
            p['stimuli'], p['choices'], p['categories'],
            no_response=p['no_response'], not_blockstart=p['prev_has_prev'])
        um_cu = compute_um(clean, mode='pooled')['um']
        assert np.allclose(um_cu, um_adj, equal_nan=True)

    def test_invalid_mode_raises(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        with pytest.raises(ValueError):
            compute_um(clean, mode='average')


class TestComputeUmOpto:
    """The reason the frozen prev_* exists: correct pairing on opto subsets."""

    def _opto_sessions(self, animal):
        return [s for s in animal.sessions if np.asarray(s.trials.opto_on).any()]

    def test_opto_subset_uses_frozen_prev(self, synthetic_opto_animal):
        """compute_um on an opto-only subset uses the real (frozen) predecessor,
        not subset array-adjacency — so it must differ from the adjacency UM
        computed on the same non-consecutive subset."""
        opto = self._opto_sessions(synthetic_opto_animal)
        trials_opto = filter_trials(opto, mask_fn=lambda s: s.trials.opto_on == 1)
        um = compute_um(trials_opto, mode='pooled')['um']
        # subset array-adjacency = the WRONG pairing
        p = pool_arrays(trials_opto)
        um_wrong, _, _ = fit_update_matrix(
            p['stimuli'], p['choices'], p['categories'],
            no_response=p['no_response'], not_blockstart=p['prev_has_prev'])
        assert np.isfinite(um).any(), 'UM all-NaN: not enough opto trials to test'
        assert not np.allclose(um, um_wrong, equal_nan=True)

    def test_post_opto_mask_runs(self, synthetic_opto_animal):
        """The complementary selection (previous trial was opto) also works."""
        opto = self._opto_sessions(synthetic_opto_animal)
        post = filter_trials(opto, mask_fn=lambda s: s.trials.prev_opto_on == 1)
        r = compute_um(post, mode='pooled')
        assert r['mode'] == 'pooled'
        assert r['um'].shape == (8, 8)


class TestPlotUm:
    """plot_um dispatch: pooled vs per_session selection and its error paths."""

    @staticmethod
    def _agg():
        import matplotlib
        matplotlib.use('Agg')

    def test_pooled_draws(self, synthetic_animal):
        self._agg()
        from behav_utils.plotting.update_matrix import plot_um
        clean = filter_trials(synthetic_animal.sessions[:5])
        fig, ax = plot_um(compute_um(clean, mode='pooled'))
        assert ax is not None

    def test_per_session_select_and_all(self, synthetic_animal):
        self._agg()
        from behav_utils.plotting.update_matrix import plot_um
        clean = filter_trials(synthetic_animal.sessions[:5])
        r = compute_um(clean, mode='per_session')
        idx = r['per_session'][0]['session_idx']
        fig, ax = plot_um(r, session_idx=idx)   # select one session by its tag
        assert ax is not None
        fig, ax = plot_um(r, session_idx='all')  # nan-aware mean across sessions
        assert ax is not None

    def test_per_session_requires_session_idx(self, synthetic_animal):
        self._agg()
        from behav_utils.plotting.update_matrix import plot_um
        clean = filter_trials(synthetic_animal.sessions[:5])
        r = compute_um(clean, mode='per_session')
        with pytest.raises(ValueError):
            plot_um(r)  # per_session result without session_idx

    def test_pooled_with_session_idx_raises(self, synthetic_animal):
        self._agg()
        from behav_utils.plotting.update_matrix import plot_um
        clean = filter_trials(synthetic_animal.sessions[:5])
        with pytest.raises(ValueError):
            plot_um(compute_um(clean, mode='pooled'), session_idx=0)
