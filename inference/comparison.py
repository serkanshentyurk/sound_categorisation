"""
SBI Model Comparison

Cross-validated BE vs SC comparison using trained SNPE posteriors.

Single responsibility: given a trained posterior and fitting data,
compute cross-validated errors and compare two models.

Training → inference/fitting.py (train_per_animal_snpe, SBIFitter)
Simulation → inference/simulation.py (simulate_all_sessions, etc.)
Fold splitting → utils/fold_utils.py (merge_smallest_adjacent)

Key fix (from original sbi_comparison_utils.py):
    The original cv_um_comparison shuffled individual trials, which
    destroys sequential structure that the update matrix depends on.
    The fixed version splits by sessions (blocks), preserving trial
    order within each fold.

Public API:
    compute_cv_comparison    — Cross-validated matrix comparison
    compute_model_comparison — Wilcoxon/ANOVA on paired CV errors


Usage:
    from inference.comparison import compute_cv_comparison, compute_model_comparison

    be_cv = compute_cv_comparison(be_snpe, fitting_data, method='update_matrix')
    sc_cv = compute_cv_comparison(sc_snpe, fitting_data, method='update_matrix')
    result = compute_model_comparison(be_cv, sc_cv)
    print(f"Winner: {result['winner']} (p={result['p_value']:.4f})")
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Any

from behav_utils.data.structures import FittingData
from behav_utils.analysis.summary_stats import compute_summary_stats
from behav_utils.analysis.update_matrix import compute_update_matrix, matrix_error
from utils.fold_utils import merge_smallest_adjacent


# =============================================================================
# CROSS-VALIDATED COMPARISON
# =============================================================================

def compute_cv_comparison(
    snpe_result: Dict[str, Any],
    fitting_data: FittingData,
    n_folds: int = 2,
    n_repeats: int = 64,
    n_posterior_samples: int = 50,
    n_stochastic_reps: int = 10,
    n_bins: int = 8,
    seed: int = 42,
    sample_timeout: int = 200,
    method: str = 'update_matrix',
) -> Dict[str, Any]:
    """
    Cross-validated matrix comparison using trained SNPE posterior.

    Splits by SESSION (block), not by trial, preserving sequential
    structure needed for update matrix computation.

    For each repeat:
        1. Split sessions into k folds (block-level)
        2. For each fold:
            a. Condition posterior on training-fold observed stats
            b. Simulate on test-fold stimuli with posterior median params
            c. Compute matrix MSE on test fold
        3. Average across folds

    Args:
        snpe_result: Output from train_per_animal_snpe or AmortisedSBI.
                     Must contain 'posterior', 'model_type', 'stat_names',
                     'burn_in', 'param_names'.
        fitting_data: FittingData for one animal.
        n_folds: Number of CV folds.
        n_repeats: Number of repetitions with different fold splits.
        n_posterior_samples: Samples from posterior for parameter estimation.
        n_stochastic_reps: Stochastic simulations per fold.
        n_bins: Bins for matrices.
        seed: Base random seed.
        sample_timeout: Max sampling batch size for posterior.
        method: 'update_matrix' or 'conditional_psych'.

    Returns:
        Dict with 'test_errors', 'mean_error', 'std_error', 'method'.
    """
    import torch
    from inference.simulator import (
        create_be_simulator, create_sc_simulator,
    )

    if method not in ('update_matrix', 'conditional_psych'):
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Use 'update_matrix' or 'conditional_psych'."
        )

    n_sessions = fitting_data.n_sessions
    model_type = snpe_result['model_type']
    stat_names = snpe_result['stat_names']
    burn_in = snpe_result['burn_in']
    param_names = snpe_result['param_names']
    creator = (create_be_simulator if model_type == 'be'
               else create_sc_simulator)

    rng = np.random.default_rng(seed)

    if n_sessions < n_folds:
        warnings.warn(
            f"Only {n_sessions} sessions but {n_folds} folds requested. "
            f"Reducing to {n_sessions} folds."
        )
        n_folds = n_sessions

    if n_sessions < 2:
        return {
            'test_errors': np.array([]),
            'mean_error': np.nan,
            'std_error': np.nan,
            'method': method,
        }

    # Pre-compute per-session arrays
    session_stim = []
    session_cat = []
    session_choices = []
    for i in range(n_sessions):
        sa = fitting_data.get_session(i)
        v = ~sa['no_response']
        session_stim.append(sa['stimuli'][v])
        session_cat.append(sa['categories'][v])
        session_choices.append(sa['choices'][v])

    session_sizes = [len(s) for s in session_stim]
    session_labels = list(range(n_sessions))

    test_errors = []

    for rep in range(n_repeats):
        # Shuffle session order for this repeat
        perm = rng.permutation(n_sessions)
        perm_sizes = [session_sizes[i] for i in perm]
        perm_labels = [session_labels[i] for i in perm]

        # Split sessions into folds
        actual_folds = min(n_folds, len(perm_sizes))
        if actual_folds < 2:
            continue

        fold_groups = merge_smallest_adjacent(
            perm_sizes, perm_labels, actual_folds,
        )

        fold_errors = []

        for fold_idx in range(actual_folds):
            test_sessions = set(fold_groups[fold_idx])
            train_sessions = set(range(n_sessions)) - test_sessions

            if not test_sessions or not train_sessions:
                continue

            # Training fold: pool trials, compute observed stats
            train_stim = np.concatenate([session_stim[i] for i in train_sessions])
            train_cat = np.concatenate([session_cat[i] for i in train_sessions])
            train_ch = np.concatenate([session_choices[i] for i in train_sessions])

            if len(train_stim) < 100:
                continue

            train_obs = compute_summary_stats(
                train_ch, train_stim, train_cat,
                stat_names=stat_names, return_dict=False,
            )
            train_obs = np.nan_to_num(train_obs, nan=0.0)

            # Condition posterior on training fold
            x_train = torch.tensor(train_obs, dtype=torch.float32)
            try:
                post_samples = snpe_result['posterior'].sample(
                    (n_posterior_samples,), x=x_train,
                    max_sampling_batch_size=sample_timeout,
                ).numpy()
            except (RuntimeError, ValueError):
                continue

            fold_params = {
                name: float(np.median(post_samples[:, i]))
                for i, name in enumerate(param_names)
            }

            # Test fold: pool trials, compute empirical matrices
            test_stim = np.concatenate([session_stim[i] for i in test_sessions])
            test_cat = np.concatenate([session_cat[i] for i in test_sessions])
            test_ch = np.concatenate([session_choices[i] for i in test_sessions])

            if len(test_stim) < 50:
                continue

            emp_um, emp_cm, _ = compute_update_matrix(
                test_stim, test_ch, test_cat, n_bins=n_bins,
            )
            emp_target = emp_um if method == 'update_matrix' else emp_cm

            # Simulate on test fold stimuli with posterior median params
            sim = creator(
                test_stim, test_cat,
                fixed_params=fold_params,
                stat_names=['accuracy'],
                burn_in=burn_in,
            )

            sim_targets = []
            for j in range(n_stochastic_reps):
                try:
                    _, sim_ch = sim.simulate(
                        sim.sample_prior(
                            seed=rep * 1000 + fold_idx * 100 + j,
                        ),
                        seed=rep * 1000 + fold_idx * 100 + j,
                        return_choices=True,
                    )
                    sim_um, sim_cm, _ = compute_update_matrix(
                        test_stim, sim_ch.flatten(), test_cat,
                        n_bins=n_bins,
                    )
                    sim_target = sim_um if method == 'update_matrix' else sim_cm
                    if not np.all(np.isnan(sim_target)):
                        sim_targets.append(sim_target)
                except (RuntimeError, ValueError):
                    continue

            if sim_targets:
                fold_errors.append(
                    matrix_error(np.nanmean(sim_targets, axis=0), emp_target)
                )

        if fold_errors:
            test_errors.append(np.mean(fold_errors))

    test_errors = np.array(test_errors)
    return {
        'test_errors': test_errors,
        'mean_error': float(np.nanmean(test_errors)) if len(test_errors) > 0 else np.nan,
        'std_error': float(np.nanstd(test_errors)) if len(test_errors) > 0 else np.nan,
        'method': method,
    }


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compute_model_comparison(
    be_cv: Dict,
    sc_cv: Dict,
    alpha: float = 0.05,
) -> Dict:
    """
    Statistical comparison of CV test errors between two models.

    Uses one-way ANOVA (F-test) on paired test errors.

    Args:
        be_cv: Dict from compute_cv_comparison for BE model.
        sc_cv: Dict from compute_cv_comparison for SC model.
        alpha: Significance threshold.

    Returns:
        Dict with f_stat, p_value, winner, be_mean, sc_mean, be_std, sc_std.
    """
    from scipy.stats import f_oneway

    be = be_cv['test_errors']
    sc = sc_cv['test_errors']

    if len(be) < 2 or len(sc) < 2:
        return {
            'f_stat': np.nan, 'p_value': np.nan,
            'winner': 'insufficient_data',
            'be_mean': float(np.nanmean(be)),
            'sc_mean': float(np.nanmean(sc)),
            'be_std': float(np.nanstd(be)),
            'sc_std': float(np.nanstd(sc)),
        }

    f_stat, p = f_oneway(be, sc)
    winner = ('BE' if np.mean(be) < np.mean(sc) else 'SC') if p < alpha else 'Inconclusive'

    return {
        'f_stat': float(f_stat), 'p_value': float(p), 'winner': winner,
        'be_mean': float(np.mean(be)), 'be_std': float(np.std(be)),
        'sc_mean': float(np.mean(sc)), 'sc_std': float(np.std(sc)),
    }