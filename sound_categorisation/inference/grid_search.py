"""
Grid-Search Cross-Validation (New Architecture)

Reimplements the manuscript's grid-search CV procedure using:
- models/ (BEModel, SCModel) for simulation
- behav_utils.readouts.fit for UM computation
- np.random.default_rng for reproducible noise


Protocol (matching manuscript):
    1. Pool expert sessions into a flat trial sequence
    2. Split by blocks into k folds
    3. For each fold:
        a. Compute empirical update matrix on training data
        b. Grid search: for each (sigma_percep, A_repulsion, model_param1, model_param2):
            - Simulate model on training stimuli (with burn-in)
            - Compute model UM on training data
            - MSE between model UM and empirical UM
        c. Evaluate best training params on test fold
    4. Return per-fold test errors, best params

Usage:
    from sound_categorisation.inference.grid_search import compute_grid_search_cv, DEFAULT_GRID

    results = compute_grid_search_cv(
        sessions=expert_sessions,
        model_type='BE',
        grid=DEFAULT_GRID['BE'],
        n_folds=2,
        seed=1,
        burn_in=1000,
    )
    print(f"Avg test error: {results['avg_test_error']:.6f}")
    print(f"Best params: {results['best_params']}")
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
from behav_utils.data import pool_arrays
from behav_utils.readouts import fit_update_matrix, matrix_error
from behav_utils.tasks import Outcome, TwoAFC
from joblib import Parallel, delayed

from sound_categorisation.inference.fold_utils import split_folds_by_block

if TYPE_CHECKING:
    from behav_utils.data import SessionData

# =============================================================================
# PARAMETER GRID DEFINITIONS
# =============================================================================

@dataclass(frozen=True)
class ParameterGrid:
    """Grid-search parameter ranges for one model."""
    sigma_percep_values: np.ndarray
    A_repulsion_values: np.ndarray
    model_param1_values: np.ndarray   # BE: eta_learning, SC: gamma
    model_param2_values: np.ndarray   # BE: eta_relax,    SC: sigma_update
    model_param1_name: str
    model_param2_name: str

    @property
    def n_combinations(self) -> int:
        return (len(self.sigma_percep_values) *
                len(self.A_repulsion_values) *
                len(self.model_param1_values) *
                len(self.model_param2_values))


# Manuscript grids
DEFAULT_GRID = {
    'BE': ParameterGrid(
        sigma_percep_values=np.linspace(0.05, 0.30, 10),
        A_repulsion_values=np.linspace(0.0, 0.5, 4),
        model_param1_values=np.linspace(0.1, 0.9, 20),   # eta_learning
        model_param2_values=np.linspace(0.05, 0.4, 10),   # eta_relax
        model_param1_name='eta_learning',
        model_param2_name='eta_relax',
    ),
    'SC': ParameterGrid(
        sigma_percep_values=np.linspace(0.05, 0.30, 10),
        A_repulsion_values=np.linspace(0.0, 0.5, 4),
        model_param1_values=np.linspace(0.1, 1.0, 20),   # gamma
        model_param2_values=np.linspace(0.1, 1.0, 10),   # sigma_update
        model_param1_name='gamma',
        model_param2_name='sigma_update',
    ),
}

COARSE_GRID = {
    'BE': ParameterGrid(
        sigma_percep_values=np.linspace(0.05, 0.30, 4),
        A_repulsion_values=np.array([0.0, 0.25, 0.5]),
        model_param1_values=np.linspace(0.1, 0.9, 8),
        model_param2_values=np.linspace(0.05, 0.4, 4),
        model_param1_name='eta_learning',
        model_param2_name='eta_relax',
    ),
    'SC': ParameterGrid(
        sigma_percep_values=np.linspace(0.05, 0.30, 4),
        A_repulsion_values=np.array([0.0, 0.25, 0.5]),
        model_param1_values=np.linspace(0.1, 1.0, 8),
        model_param2_values=np.linspace(0.1, 1.0, 4),
        model_param1_name='gamma',
        model_param2_name='sigma_update',
    ),
}


# FAST_GRID: a deliberately tiny sweep for "does it run" checks only. Two params
# vary (sigma_percep x the first model param), the other two are pinned -> 3x1x3x1
# = 9 points (vs COARSE 384, DEFAULT 8000), so a single unit finishes in seconds.
# NOT for real fits -- the winner is meaningless at this resolution.
FAST_GRID = {
    'BE': ParameterGrid(
        sigma_percep_values=np.linspace(0.10, 0.25, 3),
        A_repulsion_values=np.array([0.2]),               # pinned
        model_param1_values=np.linspace(0.3, 0.7, 3),     # eta_learning
        model_param2_values=np.array([0.2]),              # pinned (eta_relax)
        model_param1_name='eta_learning',
        model_param2_name='eta_relax',
    ),
    'SC': ParameterGrid(
        sigma_percep_values=np.linspace(0.10, 0.25, 3),
        A_repulsion_values=np.array([0.2]),               # pinned
        model_param1_values=np.linspace(0.3, 0.7, 3),     # gamma
        model_param2_values=np.array([0.5]),              # pinned (sigma_update)
        model_param1_name='gamma',
        model_param2_name='sigma_update',
    ),
}


# =============================================================================
# CORE: SIMULATE → UPDATE MATRIX
# =============================================================================

def score_choices(choices: np.ndarray, truth: np.ndarray, task=None) -> np.ndarray:
    """The task's score of ``choices`` as ``fit_update_matrix`` takes it: 1 correct, 0 error, NaN where
    there is no score (no response, or no correct response). In behav_utils 1.0 the task decides what is
    correct; ``task`` defaults to the project's two-choice task (TwoAFC: B above 0, A on it)."""
    task = task if task is not None else TwoAFC()
    choices = np.asarray(choices, dtype=float)
    codes = task.score(choices, truth, np.zeros(len(choices), dtype=bool))
    return np.where(codes == Outcome.CORRECT, 1.0, np.where(codes == Outcome.ERROR, 0.0, np.nan))


def simulate_model_matrices(
    model_type: str,
    stimuli: np.ndarray,
    categories: np.ndarray,
    no_response: np.ndarray,
    not_blockstart: np.ndarray,
    sigma_percep: float,
    A_repulsion: float,
    param1: float,
    param2: float,
    param1_name: str,
    param2_name: str,
    seed: int,
    burn_in: int = 1000,
    n_bins: int = 8,
    task=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate model and compute both update matrix and conditional psychometric matrix.

    ``task`` scores the simulated choices (default: the project's TwoAFC).

    Returns:
        (update_matrix, conditional_matrix): each (n_bins, n_bins)
    """
    rng = np.random.default_rng(seed)

    if model_type.upper() == 'BE':
        from sound_categorisation.models.be_core import BEModel, BEParams

        params = BEParams(
            sigma_percep=sigma_percep,
            A_repulsion=A_repulsion,
            **{param1_name: param1, param2_name: param2},
        )
        state = BEModel.create_initial_state(
            burn_in=burn_in, params=params, seed=seed,
        )
        choices, _, _, _ = BEModel.simulate_session(
            params, state, stimuli, categories, rng,
            no_response=no_response,
            not_blockstart=not_blockstart,
            return_history=False,
        )

    elif model_type.upper() == 'SC':
        from sound_categorisation.models.sc_core import SCModel, SCParams

        params = SCParams(
            sigma_percep=sigma_percep,
            A_repulsion=A_repulsion,
            **{param1_name: param1, param2_name: param2},
        )
        state = SCModel.create_initial_state(
            burn_in=burn_in, params=params, seed=seed,
        )
        choices, _, _, _ = SCModel.simulate_session(
            params, state, stimuli, categories, rng,
            no_response=no_response,
            not_blockstart=not_blockstart,
            return_history=False,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Compute both matrices using behav_utils
    truth = (task if task is not None else TwoAFC()).truth(stimuli)
    um, cm, _ = fit_update_matrix(
        stimuli, choices, correct=score_choices(choices, truth, task),
        n_bins=n_bins,
        trial_filter='post_correct',
        no_response=np.isnan(choices),
        not_blockstart=not_blockstart,
    )
    return um, cm



# =============================================================================
# GRID SEARCH (single data split)
# =============================================================================

def _evaluate_single_point(
    model_type: str,
    stimuli: np.ndarray,
    categories: np.ndarray,
    no_response: np.ndarray,
    not_blockstart: np.ndarray,
    target_matrix: np.ndarray,
    sigma_percep: float,
    A_repulsion: float,
    param1: float,
    param2: float,
    param1_name: str,
    param2_name: str,
    seed: int,
    burn_in: int,
    n_bins: int,
    fit_target: str = 'update_matrix',
    task=None,
) -> float:
    """Evaluate one grid point: simulate → select matrix → MSE against target.

    Args:
        fit_target: 'update_matrix' or 'conditional_psych'.
    """
    try:
        um, cm = simulate_model_matrices(
            model_type, stimuli, categories, no_response, not_blockstart,
            sigma_percep, A_repulsion, param1, param2,
            param1_name, param2_name, seed, burn_in, n_bins, task=task,
        )
        if fit_target == 'update_matrix':
            model_matrix = um
        elif fit_target == 'conditional_psych':
            model_matrix = cm
        else:
            raise ValueError(
                f"Unknown fit_target '{fit_target}'. "
                f"Use 'update_matrix' or 'conditional_psych'."
            )
        return matrix_error(model_matrix, target_matrix)
    except (ValueError, KeyError):
        return np.nan


def _grid_sweep(
    model_type: str,
    grid: ParameterGrid,
    stimuli: np.ndarray,
    categories: np.ndarray,
    no_response: np.ndarray,
    not_blockstart: np.ndarray,
    target_matrix: np.ndarray,
    seed: int,
    burn_in: int = 1000,
    n_bins: int = 8,
    n_jobs: int = -1,
    fit_target: str = 'update_matrix',
    task=None,
) -> Dict[str, Any]:
    """
    Grid search over all parameter combinations.

    Parallelised via joblib. Returns the best parameters and their error.

    Args:
        target_matrix: The empirical matrix to fit against (UM or conditional).
                       Must match fit_target.
        fit_target: 'update_matrix' or 'conditional_psych'.

    Returns:
        {
            'best_params': dict of named parameters,
            'best_error': float,
            'errors': 4D array (sigma × A × param1 × param2),
            'fit_target': str,
        }
    """
    sp_vals = grid.sigma_percep_values
    ar_vals = grid.A_repulsion_values
    p1_vals = grid.model_param1_values
    p2_vals = grid.model_param2_values

    # Build flat list of all grid points
    jobs = []
    for i, sp in enumerate(sp_vals):
        for j, ar in enumerate(ar_vals):
            for k, p1 in enumerate(p1_vals):
                for m, p2 in enumerate(p2_vals):
                    jobs.append((i, j, k, m, sp, ar, p1, p2))

    # Evaluate in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_single_point)(
            model_type, stimuli, categories, no_response, not_blockstart,
            target_matrix, sp, ar, p1, p2,
            grid.model_param1_name, grid.model_param2_name,
            seed, burn_in, n_bins, fit_target, task=task,
        )
        for _, _, _, _, sp, ar, p1, p2 in jobs
    )

    # Reshape into 4D tensor
    errors = np.full(
        (len(sp_vals), len(ar_vals), len(p1_vals), len(p2_vals)),
        np.nan,
    )
    for idx, (i, j, k, m, _sp, _ar, _p1, _p2) in enumerate(jobs):
        errors[i, j, k, m] = results[idx]

    # Find best
    best_idx = np.unravel_index(np.nanargmin(errors), errors.shape)
    best_error = errors[best_idx]

    best_params = {
        'sigma_percep': float(sp_vals[best_idx[0]]),
        'A_repulsion': float(ar_vals[best_idx[1]]),
        grid.model_param1_name: float(p1_vals[best_idx[2]]),
        grid.model_param2_name: float(p2_vals[best_idx[3]]),
    }

    return {
        'best_params': best_params,
        'best_error': float(best_error),
        'errors': errors,
        'fit_target': fit_target,
    }

# =============================================================================
# SESSION DATA → FLAT ARRAYS
# =============================================================================

def sessions_to_arrays(
    sessions: List['SessionData'],
) -> Dict[str, np.ndarray]:
    """
    Pool pre-filtered sessions into flat arrays for CV.

    Thin wrapper around ``pool_arrays``: each session is a block.
    ``not_blockstart`` is ``pool_arrays``' ``prev_has_prev`` (False at each
    block's first trial, identical on abort/opto-filtered data); ``block_ids``
    is the per-trial expansion of ``session_boundaries``.

    Returns:
        {stimuli, categories, choices, correct, truth, task, no_response, not_blockstart, block_ids};
        ``correct`` and ``truth`` are the session task's (behav_utils 1.0), ``task`` the task itself.
    """
    pooled = pool_arrays(sessions)
    boundaries = pooled['session_boundaries']      # cumulative offsets, len n_sessions + 1
    counts = np.diff(boundaries)
    block_ids = np.repeat(np.arange(len(counts)), counts)
    return {
        'stimuli': pooled['stimuli'],
        'categories': pooled['categories'],
        'choices': pooled['choices'],
        'correct': pooled['correct'],
        'truth': pooled['truth'],
        'task': pooled['task'],
        'no_response': pooled['no_response'],
        'not_blockstart': pooled['prev_has_prev'],
        'block_ids': block_ids,
    }




# =============================================================================
# FULL k-FOLD CV
# =============================================================================

def compute_grid_search_cv(
    sessions: List['SessionData'],
    model_type: str,
    grid: ParameterGrid | None = None,
    n_folds: int = 2,
    seed: int = 1,
    burn_in: int = 1000,
    n_bins: int = 8,
    n_jobs: int = -1,
    fit_target: str = 'update_matrix',
) -> Dict[str, Any]:
    """
    Full grid-search cross-validation for one model, one seed.

    Protocol:
        1. Pool sessions, split into k folds by block
        2. For each fold: grid search on train → evaluate on test
        3. Return avg test error and best params

    Args:
        sessions: Expert SessionData objects (from select_sessions)
        model_type: 'BE' or 'SC'
        grid: ParameterGrid (default: manuscript grid for model_type)
        n_folds: Number of CV folds
        seed: Random seed (affects burn-in noise and choice stochasticity)
        burn_in: Burn-in trials for model initialisation
        n_bins: Number of bins for update matrix
        n_jobs: Parallelism for grid search (-1 = all cores)
        fit_target: 'update_matrix' or 'conditional_psych'.
                    Determines which matrix is used as the fitting target
                    and the test-fold error metric.

    Returns:
        {
            'avg_test_error': float,
            'test_errors': list of per-fold test errors,
            'best_params': list of per-fold best params (named dicts),
            'best_params_single': best params from best fold,
            'model': model_type,
            'seed': seed,
            'fit_target': str,
        }
    """

    if grid is None:
        grid = DEFAULT_GRID[model_type]

    if fit_target not in ('update_matrix', 'conditional_psych'):
        raise ValueError(
            f"Unknown fit_target '{fit_target}'. "
            f"Use 'update_matrix' or 'conditional_psych'."
        )

    data = sessions_to_arrays(sessions)
    stim = data['stimuli']
    cat = data['categories']
    choices = data['choices']
    no_resp = data['no_response']
    nbs = data['not_blockstart']
    blocks = data['block_ids']
    correct = data['correct']
    task = data.get('task')

    folds = split_folds_by_block(blocks, n_folds)

    test_errors = []
    fold_params = []

    for train_mask, test_mask in folds:
        # Empirical matrices on training data
        train_um, train_cm, _ = fit_update_matrix(
            stim[train_mask], choices[train_mask], correct=correct[train_mask],
            n_bins=n_bins, trial_filter='post_correct',
            no_response=no_resp[train_mask],
            not_blockstart=nbs[train_mask],
        )
        train_target = train_um if fit_target == 'update_matrix' else train_cm

        # Grid search on training data
        sweep = _grid_sweep(
            model_type, grid,
            stim[train_mask], cat[train_mask],
            no_resp[train_mask], nbs[train_mask],
            train_target, seed, burn_in, n_bins, n_jobs,
            fit_target=fit_target, task=task,
        )

        best = sweep['best_params']
        fold_params.append(best)

        # Evaluate on test fold: simulate both matrices, pick the right one
        test_um_emp, test_cm_emp, _ = fit_update_matrix(
            stim[test_mask], choices[test_mask], correct=correct[test_mask],
            n_bins=n_bins, trial_filter='post_correct',
            no_response=no_resp[test_mask],
            not_blockstart=nbs[test_mask],
        )

        test_um_model, test_cm_model = simulate_model_matrices(
            model_type,
            stim[test_mask], cat[test_mask],
            no_resp[test_mask], nbs[test_mask],
            best['sigma_percep'], best['A_repulsion'],
            best[grid.model_param1_name], best[grid.model_param2_name],
            grid.model_param1_name, grid.model_param2_name,
            seed, burn_in, n_bins, task=task,
        )

        if fit_target == 'update_matrix':
            test_errors.append(matrix_error(test_um_model, test_um_emp))
        else:
            test_errors.append(matrix_error(test_cm_model, test_cm_emp))

    avg_error = float(np.mean(test_errors))
    best_fold_idx = int(np.argmin(test_errors))

    return {
        'avg_test_error': avg_error,
        'test_errors': test_errors,
        'best_params': fold_params,
        'best_params_single': fold_params[best_fold_idx],
        'model': model_type,
        'seed': seed,
        'fit_target': fit_target,
    }
