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


class TestGridFollowsThePrior:
    """The grid search and the SBI networks search the same space."""

    def test_default_grid_spans_the_prior_in_use(self):
        from sound_categorisation.inference.grid_search import DEFAULT_GRID, GRIDS
        from sound_categorisation.models.priors import log_uniform_params, prior_bounds
        from sound_categorisation.settings import SBI_PRIOR
        assert DEFAULT_GRID is GRIDS[SBI_PRIOR]
        for model in ('BE', 'SC'):
            grid, bounds = DEFAULT_GRID[model], prior_bounds(model.lower())
            names = list(bounds)
            assert (grid.model_param1_name, grid.model_param2_name) == (names[2], names[3])
            axes = [grid.sigma_percep_values, grid.A_repulsion_values, grid.model_param1_values,
                    grid.model_param2_values]
            assert [len(a) for a in axes] == [10, 4, 20, 10] and grid.n_combinations == 8000
            for name, values in zip(names, axes):
                np.testing.assert_allclose([values[0], values[-1]], bounds[name])
                steps = np.diff(np.log(values)) if name in log_uniform_params(model.lower()) else np.diff(values)
                np.testing.assert_allclose(steps, steps[0])       # even, in the logarithm where the prior is

    def test_be_rates_are_log_spaced_under_learning(self):
        from sound_categorisation.inference.grid_search import GRIDS
        eta = GRIDS['learning']['BE'].model_param1_values
        assert eta[0] == 0.005 and np.sum(eta < 0.05) >= 8        # the slow end is searched

    def test_manuscript_grid_is_kept_as_it_was(self):
        from sound_categorisation.inference.grid_search import GRIDS
        be, sc = GRIDS['manuscript']['BE'], GRIDS['manuscript']['SC']
        np.testing.assert_allclose(be.model_param1_values, np.linspace(0.1, 0.9, 20))
        np.testing.assert_allclose(be.model_param2_values, np.linspace(0.05, 0.4, 10))
        np.testing.assert_allclose(sc.model_param1_values, np.linspace(0.1, 1.0, 20))
        np.testing.assert_allclose(be.sigma_percep_values, np.linspace(0.05, 0.30, 10))


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
