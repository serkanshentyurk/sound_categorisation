"""SBI-based model selection (BE vs SC), using the SAME held-out protocol as
grid_search so the result is directly comparable in the four-method consensus.

Per fold: condition each model's (pre-trained) posterior on the train fold to
get a point estimate theta*, simulate choices on the test fold with theta*, and
score the held-out update-matrix (or conditional-psychometric) MSE against the
empirical test matrix. The ONLY difference from grid_search is where theta*
comes from (SBI posterior median vs grid sweep); the UM metric, the simulator,
matrix_error, and the block-aware fold split are shared.

Nets are PRE-TRAINED (train-once / condition-many). compare_models never trains.

Reuses:
    utils.fold_utils.split_folds_by_block                block-level CV folds
    behav_utils.data.ops.filtering.pool_arrays           block-aware pooling
    behav_utils.analysis.update_matrix.fit_update_matrix, matrix_error
    models.simulate.simulate_choices                     params -> choices
"""

import numpy as np
from typing import Any, Dict, List, Mapping

from inference.types import ModelType
from utils.fold_utils import split_folds_by_block
from behav_utils.data.ops.filtering import pool_arrays
from behav_utils.analysis.update_matrix import fit_update_matrix, matrix_error
from models.simulate import simulate_choices


def _as_model(model) -> ModelType:
    if isinstance(model, ModelType):
        return model
    return ModelType(str(getattr(model, 'value', model)).lower())


def _block_ids(sessions: List) -> np.ndarray:
    """Per-trial session(block) index, in session order."""
    pooled = pool_arrays(sessions)
    boundaries = pooled['session_boundaries']
    sizes = np.diff(boundaries)
    return np.repeat(np.arange(len(sizes)), sizes)


def _empirical_target(pooled: Dict[str, np.ndarray], fit_target: str, n_bins: int):
    um, cm, _ = fit_update_matrix(
        pooled['stimuli'], pooled['choices'], pooled['categories'],
        n_bins=n_bins, trial_filter='post_correct',
        no_response=pooled['no_response'],
        not_blockstart=pooled['prev_has_prev'],
    )
    return um if fit_target == 'update_matrix' else cm


def _simulated_target(model, params: Dict[str, float],
                      pooled: Dict[str, np.ndarray], fit_target: str,
                      n_bins: int, burn_in: int, seed: int):
    sim_ch = simulate_choices(
        model, params, pooled['stimuli'], pooled['categories'],
        burn_in=burn_in, seed=seed)
    um, cm, _ = fit_update_matrix(
        pooled['stimuli'], sim_ch, pooled['categories'],
        n_bins=n_bins, trial_filter='post_correct',
        no_response=pooled['no_response'],
        not_blockstart=pooled['prev_has_prev'],
    )
    return um if fit_target == 'update_matrix' else cm


def compare_models(
    sessions: List,
    nets: Mapping,
    fit_target: str = 'update_matrix',
    n_folds: int = 2,
    n_repeats: int = 64,
    n_posterior_samples: int = 50,
    n_bins: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Held-out CV model comparison via trained SBI posteriors.

    Args:
        sessions: One animal's (pre-filtered) SessionData list.
        nets: {model -> trained AmortisedSBI}. Keys may be ModelType or 'be'/'sc'.
        fit_target: 'update_matrix' or 'conditional_psych'.
        n_folds: Block-level CV folds (default 2, matching grid_search).
        n_repeats: Repeats varying posterior sampling + simulation seed
            (the spread for the Wilcoxon test).
        n_posterior_samples: Samples drawn per conditioning (median = point est).
        n_bins: Update-matrix bins.
        seed: Base seed.

    Returns:
        {fit_target, n_folds, n_sessions, per_model: {name: {fold/rep errors,
        mean_error, std_error}}, and (for exactly two models) winner, p_value,
        <a>_mean, <b>_mean}.
    """
    if fit_target not in ('update_matrix', 'conditional_psych'):
        raise ValueError(f"Unknown fit_target {fit_target!r}")
    if len(sessions) < 2:
        raise ValueError(f'Need >= 2 sessions for CV, got {len(sessions)}')

    nets = {_as_model(k): v for k, v in nets.items()}
    models = list(nets.keys())

    block_ids = _block_ids(sessions)
    folds = split_folds_by_block(block_ids, n_folds)

    # Precompute per-fold train sessions, pooled test arrays, and empirical
    # targets (fixed across repeats).
    fold_data = []
    for train_mask, test_mask in folds:
        train_blocks = np.unique(block_ids[train_mask])
        test_blocks = np.unique(block_ids[test_mask])
        train_sessions = [sessions[int(b)] for b in train_blocks]
        test_sessions = [sessions[int(b)] for b in test_blocks]
        test_pooled = pool_arrays(test_sessions)
        emp_target = _empirical_target(test_pooled, fit_target, n_bins)
        fold_data.append((train_sessions, test_pooled, emp_target))

    per_model = {m: [] for m in models}
    for rep in range(n_repeats):
        rep_seed = seed + rep
        for m in models:
            net = nets[m]
            burn_in = getattr(net, 'burn_in', 1000)
            fold_errs = []
            for train_sessions, test_pooled, emp_target in fold_data:
                cond = net.condition(train_sessions, n_samples=n_posterior_samples)
                theta_star = cond['point_estimate']
                sim_target = _simulated_target(
                    m, theta_star, test_pooled, fit_target,
                    n_bins, burn_in, rep_seed)
                fold_errs.append(float(matrix_error(emp_target, sim_target)))
            per_model[m].append(float(np.mean(fold_errs)))

    summary = {
        m.value: {
            'errors': per_model[m],
            'mean_error': float(np.mean(per_model[m])),
            'std_error': float(np.std(per_model[m])),
        }
        for m in models
    }
    result = {
        'method': 'sbi_static',
        'fit_target': fit_target,
        'n_folds': len(folds),
        'n_repeats': n_repeats,
        'n_sessions': len(sessions),
        'per_model': summary,
    }

    if len(models) == 2:
        a, b = models
        ea, eb = np.asarray(per_model[a]), np.asarray(per_model[b])
        winner = a.value if ea.mean() < eb.mean() else b.value
        p_value = np.nan
        if len(ea) >= 2 and np.any(ea != eb):
            from scipy.stats import wilcoxon
            try:
                _, p_value = wilcoxon(ea, eb)
            except ValueError:
                p_value = np.nan
        result.update({
            'winner': winner,
            'p_value': float(p_value) if p_value == p_value else np.nan,
            f'{a.value}_mean': float(ea.mean()),
            f'{b.value}_mean': float(eb.mean()),
        })

    return result
