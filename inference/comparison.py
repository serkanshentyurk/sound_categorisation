"""
SBI Model Comparison

Cross-validated BE vs SC comparison using trained SNPE posteriors.
Extracted from sbi_comparison_utils.py with:
- FIXED CV splitting (block-level, not trial-level)
- Plotting removed (see plotting/comparison.py)
- Session selection delegated to behav_utils.data.selection
- No duplicated ANOVA (use analysis.cv_utils.run_anova)

Key fix:
    The original cv_um_comparison shuffled individual trials:
        perm = rng.permutation(n_trials)
        test_idx = perm[fold * fold_size:(fold + 1) * fold_size]
    This destroys sequential structure that the update matrix depends on.

    The fixed version splits by sessions (blocks), preserving trial order
    within each fold. Uses the same merge_smallest_adjacent logic as
    the grid-search CV.

Functions:
    train_amortised_snpe     — Train SNPE with generic stimuli (Parts 1 & 3)
    train_per_animal_snpe    — Train SNPE with real stimuli (Part 2)
    condition_on_animal      — Condition posterior on observed stats
    cv_um_comparison         — FIXED cross-validated UM comparison
    compare_models           — ANOVA wrapper
    run_animal_pipeline      — Full per-animal pipeline (Parts 1 & 3)
    run_animal_pipeline_part2 — Full per-animal pipeline (Part 2)
    simulate_all_sessions    — Session-by-session simulation for plotting

Usage:
    from inference.comparison import (
        train_amortised_snpe, condition_on_animal,
        cv_um_comparison, compare_models,
    )
"""

import numpy as np
import warnings
import time
from typing import Dict, List, Tuple, Optional, Any

from behav_utils.data.structures import FittingData
from behav_utils.analysis.summary_stats import (
    compute_summary_stats, get_stat_names_expanded,
)
from behav_utils.analysis.update_matrix import compute_update_matrix, matrix_error
from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.data.synthetic import sample_stimuli
from analysis.fold_utils import merge_smallest_adjacent

from inference.types import ModelType

# Lazy imports for simulator (avoids torch dependency at import time)
# from inference.simulator import create_be_simulator, create_sc_simulator, ...


# =============================================================================
# TRAINING
# =============================================================================

def train_amortised_snpe(
    model_type: str,
    stat_names: List[str],
    n_simulations: int = 50_000,
    n_trials: int = 2500,
    burn_in: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train amortised SNPE with generic Uniform stimuli.
    Condition on any animal's stats without retraining.

    stat_names should NOT include update_matrix (sequence-dependent).
    """
    import torch
    from sbi.inference import SNPE
    from inference.simulator import (
        create_be_simulator, create_sc_simulator,
        get_sbi_prior, wrap_for_sbi,
    )

    name = model_type.upper()
    print(f"\nTraining amortised SNPE [{name}] "
          f"({n_simulations:,} sims, {n_trials} trials, burn_in={burn_in})...")

    stim, cat = sample_stimuli(n_trials, 'uniform', np.random.default_rng(seed))
    creator = create_be_simulator if model_type == 'be' else create_sc_simulator
    sim = creator(stim, cat, stat_names=stat_names, burn_in=burn_in)
    prior = get_sbi_prior(sim)
    sbi_sim = wrap_for_sbi(sim)

    t0 = time.time()
    theta = prior.sample((n_simulations,))
    print(f"  Simulating...")
    x = torch.stack([sbi_sim(t) for t in theta])

    valid = ~torch.any(torch.isnan(x), dim=1)
    n_valid = valid.sum().item()
    print(f"  {n_valid}/{n_simulations} valid "
          f"({100 * n_valid / n_simulations:.0f}%)")

    inference = SNPE(prior=prior)
    inference.append_simulations(theta[valid], x[valid])
    posterior = inference.build_posterior(inference.train())

    dt = time.time() - t0
    print(f"  Done in {dt / 60:.1f} min")

    return {
        'posterior': posterior, 'prior': prior,
        'simulator': sim, 'sbi_sim': sbi_sim,
        'param_names': sim.get_param_names(),
        'model_type': model_type, 'stat_names': stat_names,
        'burn_in': burn_in, 'training_time': dt, 'n_valid': n_valid,
    }


def train_per_animal_snpe(
    model_type: str,
    fitting_data: FittingData,
    stat_names: List[str],
    n_simulations: int = 10_000,
    burn_in: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train SNPE for one animal using its real stimulus sequence.
    stat_names CAN include update_matrix here.
    """
    import torch
    from sbi.inference import SNPE
    from inference.simulator import (
        create_be_simulator, create_sc_simulator,
        get_sbi_prior, wrap_for_sbi,
    )

    name = model_type.upper()
    aid = fitting_data.animal_id
    pooled = fitting_data.pool()
    stim, cat = pooled['stimuli'], pooled['categories']

    print(f"  Training per-animal SNPE [{name}] for {aid} "
          f"({n_simulations:,} sims, {len(stim)} trials)...")

    creator = create_be_simulator if model_type == 'be' else create_sc_simulator
    sim = creator(stim, cat, stat_names=stat_names, burn_in=burn_in)
    prior = get_sbi_prior(sim)
    sbi_sim = wrap_for_sbi(sim)

    t0 = time.time()
    theta = prior.sample((n_simulations,))
    x = torch.stack([sbi_sim(t) for t in theta])

    valid = ~torch.any(torch.isnan(x), dim=1)
    n_valid = valid.sum().item()
    print(f"    {n_valid}/{n_simulations} valid "
          f"({100 * n_valid / n_simulations:.0f}%)")

    inference = SNPE(prior=prior)
    inference.append_simulations(theta[valid], x[valid])
    posterior = inference.build_posterior(inference.train())

    dt = time.time() - t0
    print(f"    Done in {dt / 60:.1f} min")

    return {
        'posterior': posterior, 'prior': prior,
        'simulator': sim, 'sbi_sim': sbi_sim,
        'param_names': sim.get_param_names(),
        'model_type': model_type, 'stat_names': stat_names,
        'burn_in': burn_in, 'training_time': dt, 'n_valid': n_valid,
    }


# =============================================================================
# POSTERIOR CONDITIONING
# =============================================================================

def condition_on_animal(
    snpe_result: Dict[str, Any],
    fitting_data: FittingData,
    n_samples: int = 2000,
) -> Dict[str, Any]:
    """Condition posterior on one animal's observed stats."""
    import torch

    pooled = fitting_data.pool()
    obs = compute_summary_stats(
        pooled['choices'], pooled['stimuli'], pooled['categories'],
        stat_names=snpe_result['stat_names'], return_dict=False,
    )
    obs = np.nan_to_num(obs, nan=0.0)

    x_obs = torch.tensor(obs, dtype=torch.float32)
    samples = snpe_result['posterior'].sample(
        (n_samples,), x=x_obs,
    ).numpy()

    param_names = snpe_result['param_names']
    median_params = {
        name: float(np.median(samples[:, i]))
        for i, name in enumerate(param_names)
    }

    return {
        'samples': samples, 'median_params': median_params,
        'param_names': param_names, 'observed_stats': obs,
        'animal_id': fitting_data.animal_id,
    }


# =============================================================================
# CROSS-VALIDATED UM COMPARISON (FIXED: block-level splitting)
# =============================================================================


def cv_comparison(
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

    FIXED: splits by SESSION (block), not by trial.

    For each repeat:
        1. Split sessions into k folds (block-level)
        2. For each fold:
            a. Condition posterior on training-fold observed stats
            b. Simulate on test-fold stimuli with posterior median params
            c. Compute matrix MSE on test fold (UM or conditional psych)
        3. Average across folds

    Args:
        snpe_result: Output from train_amortised_snpe or train_per_animal_snpe
        fitting_data: FittingData for one animal
        n_folds: Number of CV folds
        n_repeats: Number of repetitions with different fold splits
        n_posterior_samples: Samples from posterior for parameter estimation
        n_stochastic_reps: Stochastic simulations per fold
        n_bins: Bins for matrices
        seed: Base random seed
        sample_timeout: Max sampling batch size for posterior
        method: 'update_matrix' or 'conditional_psych' — which matrix
                to use for the MSE score.

    Returns:
        {'test_errors': array, 'mean_error': float, 'std_error': float, 'method': str}
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
            except Exception:
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
                except Exception:
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


# Backwards-compatible wrapper
def cv_um_comparison(*args, **kwargs) -> Dict[str, Any]:
    """Backwards-compatible wrapper: calls cv_comparison with method='update_matrix'."""
    kwargs.pop('method', None)  # ignore if passed
    return cv_comparison(*args, method='update_matrix', **kwargs)


# =============================================================================
# MODEL COMPARISON (ANOVA)
# =============================================================================

def compare_models(
    be_cv: Dict,
    sc_cv: Dict,
    alpha: float = 0.05,
) -> Dict:
    """ANOVA on CV test errors. Returns winner + p-value."""
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


# =============================================================================
# FULL PIPELINE (Parts 1 & 3)
# =============================================================================

def run_animal_pipeline(
    fitting_data: FittingData,
    be_snpe: Dict,
    sc_snpe: Dict,
    n_cv_repeats: int = 64,
    seed: int = 42,
    verbose: bool = True,
    method: str = 'update_matrix',
) -> Dict[str, Any]:
    """
    Full comparison for one animal using pre-trained amortised SNPE.

    Args:
        method: 'update_matrix' or 'conditional_psych' — which matrix
                to score BE vs SC against during CV.
    """
    aid = fitting_data.animal_id
    if verbose:
        print(f"\n  {aid}: {fitting_data.n_sessions} sessions, "
              f"{fitting_data.trials_per_session.sum()} trials")

    be_cond = condition_on_animal(be_snpe, fitting_data)
    sc_cond = condition_on_animal(sc_snpe, fitting_data)

    if verbose:
        _print_params(be_cond['median_params'], 'BE')
        _print_params(sc_cond['median_params'], 'SC')

    be_cv = cv_comparison(
        be_snpe, fitting_data, n_repeats=n_cv_repeats, seed=seed, method=method,
    )
    sc_cv = cv_comparison(
        sc_snpe, fitting_data, n_repeats=n_cv_repeats, seed=seed, method=method,
    )
    comp = compare_models(be_cv, sc_cv)

    if verbose:
        print(f"    CV ({method}): BE={comp['be_mean']:.5f} SC={comp['sc_mean']:.5f} "
              f"p={comp['p_value']:.3g} → {comp['winner']}")

    return {
        'animal_id': aid, 'n_sessions': fitting_data.n_sessions,
        'n_trials': int(fitting_data.trials_per_session.sum()),
        'be_params': be_cond['median_params'],
        'sc_params': sc_cond['median_params'],
        'winner': comp['winner'], 'p': comp['p_value'],
        'be_mean': comp['be_mean'], 'sc_mean': comp['sc_mean'],
        'be_cv': be_cv, 'sc_cv': sc_cv,
        'method': method,
    }


def run_animal_pipeline_part2(
    fitting_data: FittingData,
    stat_names_with_um: List[str],
    n_sbi_sims: int = 10_000,
    n_cv_repeats: int = 64,
    burn_in: int = 1000,
    seed: int = 42,
    verbose: bool = True,
    method: str = 'update_matrix',
) -> Dict[str, Any]:
    """
    Full comparison for one animal using per-animal SNPE with UM.

    Args:
        method: 'update_matrix' or 'conditional_psych' — which matrix
                to score BE vs SC against during CV.
    """
    aid = fitting_data.animal_id
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"  {aid}: {fitting_data.n_sessions} sessions, "
              f"{fitting_data.trials_per_session.sum()} trials")

    be_snpe = train_per_animal_snpe(
        'be', fitting_data, stat_names_with_um,
        n_sbi_sims, burn_in, seed,
    )
    sc_snpe = train_per_animal_snpe(
        'sc', fitting_data, stat_names_with_um,
        n_sbi_sims, burn_in, seed + 1,
    )

    be_cond = condition_on_animal(be_snpe, fitting_data)
    sc_cond = condition_on_animal(sc_snpe, fitting_data)

    be_cv = cv_comparison(
        be_snpe, fitting_data, n_repeats=n_cv_repeats, seed=seed, method=method,
    )
    sc_cv = cv_comparison(
        sc_snpe, fitting_data, n_repeats=n_cv_repeats, seed=seed, method=method,
    )
    comp = compare_models(be_cv, sc_cv)

    if verbose:
        print(f"    CV ({method}): BE={comp['be_mean']:.5f} SC={comp['sc_mean']:.5f} "
              f"p={comp['p_value']:.3g} → {comp['winner']}")

    return {
        'animal_id': aid, 'n_sessions': fitting_data.n_sessions,
        'n_trials': int(fitting_data.trials_per_session.sum()),
        'be_params': be_cond['median_params'],
        'sc_params': sc_cond['median_params'],
        'winner': comp['winner'], 'p': comp['p_value'],
        'be_mean': comp['be_mean'], 'sc_mean': comp['sc_mean'],
        'be_cv': be_cv, 'sc_cv': sc_cv,
        'method': method,
    }


# =============================================================================
# SESSION-BY-SESSION SIMULATION
# =============================================================================

def simulate_all_sessions(
    fitting_data: FittingData,
    be_params: Dict[str, float],
    sc_params: Dict[str, float],
    burn_in: int = 1000,
    n_reps: int = 20,
    n_bins: int = 8,
    min_valid_trials: int = 30,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Simulate BE and SC on every session for visual comparison.

    Returns list of dicts, one per session, containing:
        stimuli, categories, choices (real),
        session_id, session_idx, accuracy, n_trials,
        real_um, be_um, sc_um,
        real_psych, be_psych, sc_psych,
        be_um_mse, sc_um_mse
    """
    from models.BE_core import BEParams, BEModel
    from models.SC_core import SCParams, SCModel

    be_p = BEParams.from_dict(be_params)
    sc_p = SCParams.from_dict(sc_params)

    results = []

    for i in range(fitting_data.n_sessions):
        v = ~fitting_data.no_response[i]
        stim = fitting_data.stimuli[i][v]
        cat = fitting_data.categories[i][v]
        ch = fitting_data.choices[i][v]

        if len(stim) < min_valid_trials:
            continue

        acc = float(np.mean(ch == cat))
        real_um, _, _ = compute_update_matrix(stim, ch, cat, n_bins=n_bins)
        real_psych = fit_psychometric(stim, ch)

        # BE simulation
        be_ums, be_psychs = _simulate_model_reps(
            'BE', be_p, stim, cat, burn_in, n_reps, n_bins, seed,
            BEModel, None,
        )
        be_mean_um = (np.nanmean(be_ums, axis=0) if be_ums
                      else np.full((n_bins, n_bins), np.nan))
        be_psych = _fit_mean_psychometric(be_psychs, stim)

        # SC simulation
        sc_ums, sc_psychs = _simulate_model_reps(
            'SC', sc_p, stim, cat, burn_in, n_reps, n_bins, seed,
            None, SCModel,
        )
        sc_mean_um = (np.nanmean(sc_ums, axis=0) if sc_ums
                      else np.full((n_bins, n_bins), np.nan))
        sc_psych = _fit_mean_psychometric(sc_psychs, stim)

        results.append({
            'stimuli': stim, 'categories': cat, 'choices': ch,
            'session_id': fitting_data.session_ids[i],
            'session_idx': int(fitting_data.session_indices[i]),
            'accuracy': acc, 'n_trials': len(stim),
            'real_um': real_um, 'be_um': be_mean_um, 'sc_um': sc_mean_um,
            'real_psych': real_psych, 'be_psych': be_psych,
            'sc_psych': sc_psych,
            'be_um_mse': matrix_error(be_mean_um, real_um),
            'sc_um_mse': matrix_error(sc_mean_um, real_um),
        })

    return results


def _simulate_model_reps(
    model_name, params, stim, cat, burn_in, n_reps, n_bins, seed,
    BEModel_cls, SCModel_cls,
):
    """Helper: run n_reps simulations, collect UMs and choice arrays."""
    ums, all_choices = [], []

    for r in range(n_reps):
        rng_r = np.random.default_rng(seed + r + 1)

        if model_name == 'BE':
            state = BEModel_cls.create_initial_state(
                params=params, burn_in=burn_in, seed=seed,
            )
            c, _, _, _ = BEModel_cls.simulate_session(
                params, state, stim, cat, rng_r, return_history=False,
            )
        else:
            state = SCModel_cls.create_initial_state(
                params=params, burn_in=burn_in, seed=seed,
            )
            c, _, _, _ = SCModel_cls.simulate_session(
                params, state, stim, cat, rng_r, return_history=False,
            )

        vv = ~np.isnan(c)
        if vv.sum() > 50:
            um, _, _ = compute_update_matrix(stim[vv], c[vv], cat[vv], n_bins)
            ums.append(um)
        all_choices.append(c)

    return ums, all_choices


def _fit_mean_psychometric(all_choices, stim):
    """Fit psychometric to mean choice probabilities across reps."""
    if not all_choices:
        return {'success': False}
    mean_choices = np.nanmean(all_choices, axis=0)
    return fit_psychometric(stim, mean_choices)


def _print_params(params, name):
    parts = ', '.join(f'{k}={v:.3f}' for k, v in params.items())
    print(f"    {name}: {parts}")



# =============================================================================
# TIMING UTILITIES
# =============================================================================

def estimate_timing(
    stat_names: List[str],
    n_trials: int = 2500,
    burn_in: int = 1000,
    n_sbi_sims: int = 50_000,
    n_test: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Estimate per-simulation cost for BE and SC.

    Runs n_test forward simulations with each model and reports
    timing + NaN rate + projected total training time.
    """
    from inference.simulator import (
        create_be_simulator, create_sc_simulator,
    )

    stim, cat = sample_stimuli(n_trials, 'uniform', np.random.default_rng(seed))
    results = {}

    for model_type in ['be', 'sc']:
        creator = create_be_simulator if model_type == 'be' else create_sc_simulator
        sim = creator(stim, cat, stat_names=stat_names, burn_in=burn_in)

        times = []
        nan_count = 0
        for i in range(n_test):
            theta = sim.sample_prior(seed=seed + i)
            t0 = time.time()
            stats = sim(theta, seed=seed + i)
            times.append(time.time() - t0)
            if np.any(np.isnan(stats)):
                nan_count += 1

        ms_per_sim = np.mean(times) * 1000
        total_min = np.mean(times) * n_sbi_sims / 60
        n_stat_dims = len(stats)

        results[model_type] = {
            'ms_per_sim': ms_per_sim,
            'total_minutes': total_min,
            'total_hours': total_min / 60,
            'nan_rate': nan_count / n_test,
            'stat_dims': n_stat_dims,
            'theta_dims': sim.n_free_params,
        }

    return results


def print_timing_report(
    timing: Dict[str, Any],
    n_sbi_sims: int,
    n_animals: int = 1,
    label: str = '',
):
    """Print a formatted timing report."""
    print(f"\n{'=' * 60}")
    if label:
        print(f"  Timing estimate: {label}")
    print(f"  {n_sbi_sims:,} simulations")
    print(f"{'=' * 60}")
    print(f"  {'Model':<6s} {'ms/sim':>8s} {'Total':>10s} {'NaN%':>6s} "
          f"{'θ dims':>7s} {'Stat dims':>10s}")
    print(f"  {'-' * 50}")

    for mt in ['be', 'sc']:
        t = timing[mt]
        total_str = (f"{t['total_hours']:.1f}h" if t['total_hours'] >= 1
                     else f"{t['total_minutes']:.0f}min")
        print(f"  {mt.upper():<6s} {t['ms_per_sim']:8.0f} {total_str:>10s} "
              f"{t['nan_rate']:5.0%} {t['theta_dims']:>7d} {t['stat_dims']:>10d}")

    if n_animals > 1:
        be_h = timing['be']['total_hours']
        sc_h = timing['sc']['total_hours']
        total = (be_h + sc_h) * n_animals
        print(f"\n  {n_animals} animals × 2 models = ~{total:.0f} hours total")


# =============================================================================
# SINGLE-SESSION EXAMPLE SIMULATION
# =============================================================================

def simulate_example_session(
    animal: Any, session_idx: int,
    be_params: Dict, sc_params: Dict,
    stage: str = 'Full_Task_Cont', distribution: str = 'Uniform',
    burn_in: int = 1000, n_reps: int = 20, seed: int = 42,
) -> Dict[str, Any]:
    """Simulate BE and SC on one real session for visualisation."""
    from models.BE_core import BEParams, BEModel
    from models.SC_core import SCParams, SCModel

    sessions = animal.get_sessions(stage=stage, distribution=distribution)
    sess = sessions[session_idx]
    # Pre-filter then extract
    from behav_utils.data.filtering import filter_session
    clean = filter_session(sess)
    arrays = clean.get_arrays()
    valid = ~arrays['no_response']
    stim = arrays['stimuli'][valid]
    cat = arrays['categories'][valid]
    ch = arrays['choices'][valid]

    be_p = BEParams(**be_params)
    be_state = BEModel.create_initial_state(params=be_p, burn_in=burn_in, seed=seed)
    _, be_pB, _, _ = BEModel.simulate_session(
        be_p, be_state, stim, cat,
        np.random.default_rng(seed), return_history=False,
    )

    sc_p = SCParams(**sc_params)
    sc_state = SCModel.create_initial_state(params=sc_p, burn_in=burn_in, seed=seed)
    _, sc_pB, _, _ = SCModel.simulate_session(
        sc_p, sc_state, stim, cat,
        np.random.default_rng(seed), return_history=False,
    )

    be_all, sc_all = [], []
    for r in range(n_reps):
        rng_r = np.random.default_rng(seed + r + 1)
        s1 = BEModel.create_initial_state(params=be_p, burn_in=burn_in, seed=seed)
        c1, _, _, _ = BEModel.simulate_session(
            be_p, s1, stim, cat, rng_r, return_history=False,
        )
        be_all.append(c1)
        s2 = SCModel.create_initial_state(params=sc_p, burn_in=burn_in, seed=seed)
        c2, _, _, _ = SCModel.simulate_session(
            sc_p, s2, stim, cat, rng_r, return_history=False,
        )
        sc_all.append(c2)

    return {
        'stimuli': stim, 'categories': cat, 'choices': ch,
        'be_pB': be_pB, 'sc_pB': sc_pB,
        'be_choices_all': be_all, 'sc_choices_all': sc_all,
        'session_info': {
            'session_id': sess.session_id, 'n_trials': len(stim),
            'accuracy': float(np.mean(ch == cat)),
        },
    }


# =============================================================================
# FITTING DATA CONVENIENCE WRAPPERS
# =============================================================================
# These provide backward-compatible signatures for notebook code that
# calls select_expert_sessions(animal, stage, dist, min_acc, last_frac)
# and expects FittingData back.

def select_expert_fitting_data(
    animal: Any,
    stage: str = 'Full_Task_Cont',
    distribution: str = 'Uniform',
    min_accuracy: float = 0.70,
    last_fraction: float = 0.50,
    min_valid_trials: int = 30,
) -> FittingData:
    """
    Select expert sessions and return as FittingData.

    Drop-in replacement for the old sbi_comparison_utils.select_expert_sessions.
    """
    from behav_utils.data.selection import select_sessions, fitting_data_from_sessions

    sessions = select_sessions(
        animal,
        stage=stage,
        distribution=distribution,
        min_accuracy=min_accuracy,
        last_fraction=last_fraction,
        min_trials=min_valid_trials,
    )
    if len(sessions) == 0:
        raise ValueError(
            f"No expert sessions for {animal.animal_id} "
            f"(acc>={min_accuracy}, last {last_fraction:.0%})"
        )
    return fitting_data_from_sessions(
        sessions, animal.animal_id, min_valid_trials=min_valid_trials,
    )


def select_all_fitting_data(
    animal: Any,
    stage: str = 'Full_Task_Cont',
    distribution: str = 'Uniform',
    min_valid_trials: int = 30,
) -> FittingData:
    """
    Select all qualifying sessions and return as FittingData.

    Drop-in replacement for the old sbi_comparison_utils.select_all_sessions.
    """
    from behav_utils.data.selection import select_sessions, fitting_data_from_sessions

    sessions = select_sessions(
        animal,
        stage=stage,
        distribution=distribution,
        min_trials=min_valid_trials,
    )
    return fitting_data_from_sessions(
        sessions, animal.animal_id, min_valid_trials=min_valid_trials,
    )
