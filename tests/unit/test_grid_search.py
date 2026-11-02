"""
Tests for analysis/grid_search.py.

Smoke tests + parameter recovery on tiny synthetic data with COARSE_GRID.
Full validation against the manuscript protocol is in NB 11/12.
"""

import numpy as np


class TestParameterGrid:
    """ParameterGrid construction and inspection."""

    def test_default_grid_has_be_and_sc(self):
        """DEFAULT_GRID has presets for both models."""
        from sound_categorisation.inference.grid_search import COARSE_GRID, DEFAULT_GRID
        assert DEFAULT_GRID is not None
        assert COARSE_GRID is not None

    def test_coarse_grid_smaller_than_default(self):
        """Per model, COARSE_GRID has fewer points than DEFAULT_GRID (and FAST_GRID fewer still)."""
        from sound_categorisation.inference.grid_search import COARSE_GRID, DEFAULT_GRID, FAST_GRID
        for model in ('BE', 'SC'):
            assert FAST_GRID[model].n_combinations < COARSE_GRID[model].n_combinations < DEFAULT_GRID[model].n_combinations


class TestSimulateModelMatrices:
    """simulate_model_matrices generates per-θ update matrices."""

    def test_returns_arrays(self):
        """Returns numpy arrays of expected shape."""
        from sound_categorisation.inference.grid_search import simulate_model_matrices

        # Minimal call: a single param set
        rng = np.random.default_rng(0)
        stimuli = rng.uniform(-1, 1, 500)
        categories = (stimuli > 0).astype(int)
        no_response = np.zeros(500, dtype=bool)
        not_blockstart = np.ones(500, dtype=bool)
        um, cm = simulate_model_matrices(
            'be', stimuli, categories, no_response, not_blockstart,
            sigma_percep=0.1, A_repulsion=0.1,
            param1=0.3, param2=0.1,
            param1_name='eta_learning', param2_name='eta_relax',
            seed=0, burn_in=50, n_bins=8,
        )
        assert um.shape == (8, 8) and cm.shape == (8, 8)


class TestComputeGridSearchCV:
    """compute_grid_search_cv on tiny synthetic data."""

    def test_runs_on_synthetic_animal(self, synthetic_animal):
        """Function runs without error on small synthetic data."""
        from sound_categorisation.inference.grid_search import FAST_GRID, compute_grid_search_cv

        clean = [s for s in synthetic_animal.sessions if s.session_type != 'masking'][:4]

        # Minimum effort: the smallest grid, two folds. No try/except: a failure here is a failure.
        result = compute_grid_search_cv(
            clean, model_type='BE', grid=FAST_GRID['BE'],
            n_folds=2, fit_target='update_matrix',
        )
        assert isinstance(result, dict)
        assert {'best_params', 'avg_test_error', 'test_errors', 'model'} <= set(result)
        assert len(result['test_errors']) == 2
