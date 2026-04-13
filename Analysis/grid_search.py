"""
Grid-Search Cross-Validation (New Architecture)

Reimplements the manuscript's grid-search CV procedure using:
- models/ (BEModel, SCModel) for simulation
- behav_utils.analysis.update_matrix for UM computation
- np.random.default_rng for reproducible noise

This replaces the legacy code path (legacy/fitter.py + legacy/be.py + legacy/sc.py)
while producing equivalent results (up to RNG differences across seeds).

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
    from analysis.grid_search import grid_search_cv, DEFAULT_GRID

    results = grid_search_cv(
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

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from joblib import Parallel, delayed

from behav_utils.analysis.update_matrix import compute_update_matrix, matrix_error
from analysis.fold_utils import split_folds_by_block

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData

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


# =============================================================================
# CORE: SIMULATE → UPDATE MATRIX
# =============================================================================

def _simulate_um(
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
) -> np.ndarray:
    """
    Simulate model and compute update matrix.

    This is the core bridge: new-architecture model → behav_utils UM.

    Returns:
        update_matrix: (n_bins, n_bins) array
    """
    rng = np.random.default_rng(seed)

    if model_type == 'BE':
        from models.BE_core import BEParams, BEModel

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

    elif model_type == 'SC':
        from models.SC_core import SCParams, SCModel

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

    # Compute update matrix using behav_utils
    um, _, _ = compute_update_matrix(
        stimuli, choices, categories,
        n_bins=n_bins,
        trial_filter='post_correct',
        no_response=np.isnan(choices),
        not_blockstart=not_blockstart,
    )

    return um


# =============================================================================
# GRID SEARCH (single data split)
# =============================================================================

def _evaluate_single_point(
    model_type: str,
    stimuli: np.ndarray,
    categories: np.ndarray,
    no_response: np.ndarray,
    not_blockstart: np.ndarray,
    target_um: np.ndarray,
    sigma_percep: float,
    A_repulsion: float,
    param1: float,
    param2: float,
    param1_name: str,
    param2_name: str,
    seed: int,
    burn_in: int,
    n_bins: int,
) -> float:
    """Evaluate one grid point: simulate → UM → MSE against target."""
    try:
        model_um = _simulate_um(
            model_type, stimuli, categories, no_response, not_blockstart,
            sigma_percep, A_repulsion, param1, param2,
            param1_name, param2_name, seed, burn_in, n_bins,
        )
        return matrix_error(model_um, target_um)
    except Exception:
        return np.nan


def parameter_sweep(
    model_type: str,
    grid: ParameterGrid,
    stimuli: np.ndarray,
    categories: np.ndarray,
    no_response: np.ndarray,
    not_blockstart: np.ndarray,
    target_um: np.ndarray,
    seed: int,
    burn_in: int = 1000,
    n_bins: int = 8,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Grid search over all parameter combinations.

    Parallelised via joblib. Returns the best parameters and their error.

    Returns:
        {
            'best_params': dict of named parameters,
            'best_error': float,
            'errors': 4D array (sigma × A × param1 × param2),
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
                for l, p2 in enumerate(p2_vals):
                    jobs.append((i, j, k, l, sp, ar, p1, p2))

    # Evaluate in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_single_point)(
            model_type, stimuli, categories, no_response, not_blockstart,
            target_um, sp, ar, p1, p2,
            grid.model_param1_name, grid.model_param2_name,
            seed, burn_in, n_bins,
        )
        for _, _, _, _, sp, ar, p1, p2 in jobs
    )

    # Reshape into 4D tensor
    errors = np.full(
        (len(sp_vals), len(ar_vals), len(p1_vals), len(p2_vals)),
        np.nan,
    )
    for idx, (i, j, k, l, sp, ar, p1, p2) in enumerate(jobs):
        errors[i, j, k, l] = results[idx]

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
    }


# =============================================================================
# SESSION DATA → FLAT ARRAYS
# =============================================================================

def _sessions_to_arrays(
    sessions: List['SessionData'],
) -> Dict[str, np.ndarray]:
    """
    Pool sessions into flat arrays for CV.

    Each session becomes a block. Aborts and opto trials excluded.

    Returns:
        {stimuli, categories, choices, no_response, not_blockstart, block_ids}
    """
    all_stim, all_cat, all_choice = [], [], []
    all_no_resp, all_nbs, all_block = [], [], []

    for block_id, session in enumerate(sessions):
        arrays = session.trials.get_arrays(
            exclude_abort=True, exclude_opto=True,
        )
        n = len(arrays['stimuli'])
        if n == 0:
            continue

        all_stim.append(arrays['stimuli'])
        all_cat.append(arrays['categories'])
        all_choice.append(arrays['choices'])
        all_no_resp.append(arrays['no_response'])

        nbs = np.ones(n, dtype=bool)
        nbs[0] = False
        all_nbs.append(nbs)
        all_block.append(np.full(n, block_id))

    return {
        'stimuli': np.concatenate(all_stim),
        'categories': np.concatenate(all_cat),
        'choices': np.concatenate(all_choice),
        'no_response': np.concatenate(all_no_resp),
        'not_blockstart': np.concatenate(all_nbs),
        'block_ids': np.concatenate(all_block),
    }




# =============================================================================
# FULL k-FOLD CV
# =============================================================================

def grid_search_cv(
    sessions: List['SessionData'],
    model_type: str,
    grid: Optional[ParameterGrid] = None,
    n_folds: int = 2,
    seed: int = 1,
    burn_in: int = 1000,
    n_bins: int = 8,
    n_jobs: int = -1,
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

    Returns:
        {
            'avg_test_error': float,
            'test_errors': list of per-fold test errors,
            'best_params': list of per-fold best params (named dicts),
            'best_params_single': best params from best fold,
            'model': model_type,
            'seed': seed,
        }
    """
        
    if grid is None:
        grid = DEFAULT_GRID[model_type]

    data = _sessions_to_arrays(sessions)
    stim = data['stimuli']
    cat = data['categories']
    choices = data['choices']
    no_resp = data['no_response']
    nbs = data['not_blockstart']
    blocks = data['block_ids']

    folds = split_folds_by_block(blocks, n_folds)

    test_errors = []
    fold_params = []

    for train_mask, test_mask in folds:
        # Empirical UM on training data
        train_um, _, _ = compute_update_matrix(
            stim[train_mask], choices[train_mask], cat[train_mask],
            n_bins=n_bins, trial_filter='post_correct',
            no_response=no_resp[train_mask],
            not_blockstart=nbs[train_mask],
        )

        # Grid search on training data
        sweep = parameter_sweep(
            model_type, grid,
            stim[train_mask], cat[train_mask],
            no_resp[train_mask], nbs[train_mask],
            train_um, seed, burn_in, n_bins, n_jobs,
        )

        best = sweep['best_params']
        fold_params.append(best)

        # Evaluate on test fold
        test_um_emp, _, _ = compute_update_matrix(
            stim[test_mask], choices[test_mask], cat[test_mask],
            n_bins=n_bins, trial_filter='post_correct',
            no_response=no_resp[test_mask],
            not_blockstart=nbs[test_mask],
        )

        test_um_model = _simulate_um(
            model_type,
            stim[test_mask], cat[test_mask],
            no_resp[test_mask], nbs[test_mask],
            best['sigma_percep'], best['A_repulsion'],
            best[grid.model_param1_name], best[grid.model_param2_name],
            grid.model_param1_name, grid.model_param2_name,
            seed, burn_in, n_bins,
        )

        test_errors.append(matrix_error(test_um_model, test_um_emp))

    avg_error = float(np.mean(test_errors))
    best_fold_idx = int(np.argmin(test_errors))

    return {
        'avg_test_error': avg_error,
        'test_errors': test_errors,
        'best_params': fold_params,
        'best_params_single': fold_params[best_fold_idx],
        'model': model_type,
        'seed': seed,
    }


# =============================================================================
# CONVENIENCE: RUN BOTH MODELS
# =============================================================================

def run_cv_both_models(
    sessions: List['SessionData'],
    grid_be: Optional[ParameterGrid] = None,
    grid_sc: Optional[ParameterGrid] = None,
    n_folds: int = 2,
    seed: int = 1,
    burn_in: int = 1000,
    n_bins: int = 8,
    n_jobs: int = -1,
) -> Dict[str, Dict]:
    """
    Run grid-search CV for both BE and SC on the same data.

    Returns:
        {'BE': {cv_result_dict}, 'SC': {cv_result_dict}}
    """
    results = {}
    for model_type, grid in [('BE', grid_be), ('SC', grid_sc)]:
        results[model_type] = grid_search_cv(
            sessions, model_type, grid,
            n_folds, seed, burn_in, n_bins, n_jobs,
        )
    return results

# ─────────────────────────────────────────────────────────────────────────────
# PHASE-BLOCKED FITTING
# ─────────────────────────────────────────────────────────────────────────────

def fit_sessions_blocked(
    phase_blocks: Dict[str, List['SessionData']],
    model_type: str,
    grid: 'ParameterGrid' = None,
    burn_in: int = 1000,
    seed: int = 42,
    n_seeds: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit model parameters to phase-blocked groups of sessions.

    Each block is a named group of sessions (e.g. 'naive', 'expert',
    'early_post', 'late_post'). Parameters are fit independently per
    block by pooling sessions within each block.

    Args:
        phase_blocks: {phase_name: [SessionData, ...]}
        model_type: 'BE' or 'SC'
        grid: Parameter grid (default: DEFAULT_GRID[model_type])
        burn_in: Burn-in trials for simulation
        seed: Base random seed
        n_seeds: Number of seeds to average over

    Returns:
        {phase_name: {
            'best_params': dict,
            'train_error': float,
            'n_sessions': int,
            'n_trials': int,
            'session_indices': list,
            'per_seed_errors': list,
            'per_seed_params': list,
        }}
    """
    from analysis.grid_search import (
        grid_search_cv, _sessions_to_arrays, DEFAULT_GRID, ParameterGrid,
        _simulate_um, parameter_sweep,
    )
    from behav_utils.analysis.update_matrix import (
        compute_update_matrix, matrix_error,
    )

    if grid is None:
        grid = DEFAULT_GRID[model_type]

    results = {}

    for phase_name, sessions in phase_blocks.items():
        if not sessions:
            results[phase_name] = {
                'best_params': {}, 'train_error': np.nan,
                'n_sessions': 0, 'n_trials': 0,
                'session_indices': [],
            }
            continue

        # Pool sessions into flat arrays
        stimuli, choices, categories, block_ids = _sessions_to_arrays(sessions)
        n_trials = len(stimuli)

        if n_trials < 50:
            results[phase_name] = {
                'best_params': {}, 'train_error': np.nan,
                'n_sessions': len(sessions), 'n_trials': n_trials,
                'session_indices': [s.session_idx for s in sessions],
            }
            continue

        # Compute empirical UM
        emp_um, _, _ = compute_update_matrix(stimuli, choices, categories)

        # Grid search (no CV — fit to all data in this block)
        best_error = np.inf
        best_params = None
        per_seed_errors = []
        per_seed_params = []

        for s_offset in range(n_seeds):
            sweep_result = parameter_sweep(
                model_type=model_type,
                grid=grid,
                stimuli=stimuli,
                categories=categories,
                empirical_um=emp_um,
                burn_in=burn_in,
                seed=seed + s_offset,
            )
            per_seed_errors.append(sweep_result['best_error'])
            per_seed_params.append(sweep_result['best_params'])

            if sweep_result['best_error'] < best_error:
                best_error = sweep_result['best_error']
                best_params = sweep_result['best_params']

        results[phase_name] = {
            'best_params': best_params,
            'train_error': float(best_error),
            'mean_error': float(np.mean(per_seed_errors)),
            'n_sessions': len(sessions),
            'n_trials': n_trials,
            'session_indices': [s.session_idx for s in sessions],
            'per_seed_errors': per_seed_errors,
            'per_seed_params': per_seed_params,
        }

    return results


def fit_sessions_individual(
    sessions: List['SessionData'],
    model_type: str,
    grid: 'ParameterGrid' = None,
    burn_in: int = 1000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Fit model parameters to each session independently.

    Noisy but assumption-free — gives the raw per-session parameter
    trajectory. Sessions with too few trials return NaN.

    Args:
        sessions: List of SessionData
        model_type: 'BE' or 'SC'
        grid: Parameter grid (default: DEFAULT_GRID[model_type])
        burn_in: Burn-in trials for simulation
        seed: Random seed

    Returns:
        List of dicts (one per session), each with:
            'session_idx', 'session_id', 'best_params', 'error',
            'n_trials', 'converged'
    """
    results = []

    for sess in sessions:
        result = {
            'session_idx': sess.session_idx,
            'session_id': sess.session_id,
            'date': sess.date,
        }

        # Use fit_sessions_blocked with a single-session block
        block_result = fit_sessions_blocked(
            phase_blocks={'single': [sess]},
            model_type=model_type,
            grid=grid,
            burn_in=burn_in,
            seed=seed,
            n_seeds=1,
        )
        r = block_result['single']
        result.update({
            'best_params': r['best_params'],
            'error': r['train_error'],
            'n_trials': r['n_trials'],
            'converged': not np.isnan(r['train_error']),
        })
        results.append(result)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STATIC VS DYNAMIC COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_static_vs_dynamic(
    sessions: List['SessionData'],
    model_type: str,
    phase_blocks: Dict[str, List['SessionData']],
    grid: 'ParameterGrid' = None,
    burn_in: int = 1000,
    seed: int = 42,
    n_seeds: int = 5,
) -> Dict[str, Any]:
    """
    Compare a single static fit (all sessions pooled) against
    phase-blocked fits. Tests whether allowing parameters to vary
    across phases improves fit quality.

    Args:
        sessions: All sessions (for the static fit)
        model_type: 'BE' or 'SC'
        phase_blocks: {phase_name: [SessionData, ...]} for dynamic fit
        grid: Parameter grid
        burn_in: Burn-in trials
        seed: Random seed
        n_seeds: Seeds to average

    Returns:
        {
            'static': {fit results for all-sessions-pooled},
            'dynamic': {phase_name: {fit results}, ...},
            'static_total_error': float,
            'dynamic_total_error': float,
            'improvement_ratio': float,  # dynamic/static (< 1 means dynamic is better)
            'per_phase_comparison': [{phase, static_error, dynamic_error, ...}],
        }
    """
    from behav_utils.analysis.update_matrix import (
        compute_update_matrix, matrix_error,
    )

    # Static fit: pool everything
    static_result = fit_sessions_blocked(
        phase_blocks={'all': sessions},
        model_type=model_type,
        grid=grid,
        burn_in=burn_in,
        seed=seed,
        n_seeds=n_seeds,
    )['all']

    # Dynamic fit: per phase
    dynamic_results = fit_sessions_blocked(
        phase_blocks=phase_blocks,
        model_type=model_type,
        grid=grid,
        burn_in=burn_in,
        seed=seed,
        n_seeds=n_seeds,
    )

    # Evaluate static params on each phase separately
    from analysis.grid_search import _sessions_to_arrays, _simulate_um

    per_phase_comparison = []
    dynamic_total_weighted_error = 0.0
    static_total_weighted_error = 0.0
    total_trials = 0

    for phase_name, phase_sessions in phase_blocks.items():
        if not phase_sessions:
            continue

        stimuli, choices, categories, _ = _sessions_to_arrays(phase_sessions)
        n_trials = len(stimuli)
        if n_trials < 50:
            continue

        emp_um, _, _ = compute_update_matrix(stimuli, choices, categories)

        # Static params evaluated on this phase
        if static_result['best_params']:
            static_um = _simulate_um(
                model_type=model_type,
                params=static_result['best_params'],
                stimuli=stimuli,
                categories=categories,
                burn_in=burn_in,
                seed=seed,
            )
            static_phase_error = matrix_error(static_um, emp_um)
        else:
            static_phase_error = np.nan

        # Dynamic params for this phase
        dyn_r = dynamic_results.get(phase_name, {})
        dynamic_phase_error = dyn_r.get('train_error', np.nan)

        per_phase_comparison.append({
            'phase': phase_name,
            'n_trials': n_trials,
            'static_error': static_phase_error,
            'dynamic_error': dynamic_phase_error,
            'static_params': static_result['best_params'],
            'dynamic_params': dyn_r.get('best_params', {}),
        })

        if not np.isnan(static_phase_error) and not np.isnan(dynamic_phase_error):
            static_total_weighted_error += static_phase_error * n_trials
            dynamic_total_weighted_error += dynamic_phase_error * n_trials
            total_trials += n_trials

    if total_trials > 0:
        static_total = static_total_weighted_error / total_trials
        dynamic_total = dynamic_total_weighted_error / total_trials
        improvement = dynamic_total / static_total if static_total > 0 else np.nan
    else:
        static_total = np.nan
        dynamic_total = np.nan
        improvement = np.nan

    return {
        'static': static_result,
        'dynamic': dynamic_results,
        'static_total_error': static_total,
        'dynamic_total_error': dynamic_total,
        'improvement_ratio': improvement,
        'per_phase_comparison': per_phase_comparison,
    }
