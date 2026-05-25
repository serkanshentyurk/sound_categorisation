"""
analysis/sbi_validation.py — SBI validation: SBC, parameter recovery,
parameter-stat correlations.

Three compute functions, all returning result dicts ready for the
matching plotters in plotting/sbi_validation.py.

    compute_sbc_ranks(posterior, simulator, prior, ...)
        Simulation-Based Calibration: tests whether the posterior is
        well-calibrated by sampling theta from the prior, simulating
        data, drawing posterior samples, and ranking the true theta
        among them. Well-calibrated → ranks uniform.

    compute_parameter_recovery(posterior, simulator, prior, ...)
        Tests whether posterior point estimates (median) are close to
        the true values. Returns correlation, RMSE, bias, 90% coverage
        per parameter.

    compute_param_stat_correlations(model_type, stat_names, ...)
        Regenerates a small training-like batch and reports
        parameter ↔ summary-stat correlations. Useful for choosing
        which stats are informative for which parameters.

Naming note: the third function takes (model_type, stat_names) rather
than (posterior, simulator, prior). It's a property of the model class,
not a property of a trained posterior.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# Tunable: minimum number of valid bootstrap replicates before a CI is reported.
_MIN_VALID_RUNS = 10


# =============================================================================
# SBC
# =============================================================================

def compute_sbc_ranks(
    posterior: Any,
    simulator: Callable,
    prior: Any,
    n_sbc_runs: int = 1000,
    n_posterior_samples: int = 1000,
    seed: int = 42,
    show_progress: bool = True,
    param_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Simulation-Based Calibration.

    For each iteration:
        1. theta* ~ prior
        2. x* = simulator(theta*)
        3. theta_post[i] ~ posterior(theta | x*)
        4. rank(theta*) = #{i: theta_post[i] < theta*}

    Well-calibrated posteriors produce uniformly distributed ranks.
    U-shape → overconfident; inverted-U → underconfident; skew → biased.

    Args:
        posterior: SBIResult or raw sbi posterior with .sample(n, x=x_obs).
        simulator: Callable theta → summary stats (numpy array).
        prior: Object with .sample((n,)) returning (n, n_params) tensor/array.
        n_sbc_runs: SBC iterations.
        n_posterior_samples: Posterior samples per iteration.
        seed: RNG seed.
        show_progress: Show a tqdm bar if available.
        param_names: Optional names for the n_params dimensions.

    Returns:
        Dict ready for plot_sbc_ranks / plot_sbc_ecdf:
            'ranks':                (n_valid, n_params) int — rank per param per run
            'thetas':               (n_valid, n_params) — true thetas
            'n_posterior_samples':  int
            'param_names':          List[str] or None
            'ks_pvalues':           (n_params,) Kolmogorov-Smirnov p per param
    """
    import torch

    sbi_posterior = posterior.posterior if hasattr(posterior, 'posterior') else posterior

    # Try sbi built-in first; fall back to manual on any failure.
    try:
        from sbi.analysis import run_sbc as _sbi_run_sbc

        sbc_result = _sbi_run_sbc(
            sbi_posterior, simulator, prior,
            num_sbc_runs=n_sbc_runs,
            num_posterior_samples=n_posterior_samples,
        )
        if isinstance(sbc_result, tuple):
            ranks, thetas = sbc_result[0], sbc_result[1]
        else:
            ranks, thetas = sbc_result, None

        ranks_np = ranks.numpy() if hasattr(ranks, 'numpy') else np.asarray(ranks)
        thetas_np = (
            thetas.numpy() if thetas is not None and hasattr(thetas, 'numpy')
            else thetas
        )
        return {
            'ranks':               ranks_np,
            'thetas':              thetas_np,
            'n_posterior_samples': n_posterior_samples,
            'param_names':         param_names,
            'ks_pvalues':          _ks_pvalues(ranks_np, n_posterior_samples),
        }
    except Exception as e:
        if show_progress:
            print(f"sbi built-in SBC failed ({e}); using manual implementation.")

    # Manual implementation
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    prior_samples = prior.sample((n_sbc_runs,))
    prior_np = (
        prior_samples.numpy() if hasattr(prior_samples, 'numpy')
        else np.asarray(prior_samples)
    )

    n_params = prior_np.shape[1]
    ranks = np.zeros((n_sbc_runs, n_params), dtype=int)

    iterator = _maybe_progress(range(n_sbc_runs), show_progress, 'SBC')

    for i in iterator:
        theta_star = prior_np[i]
        x_star = np.asarray(simulator(theta_star))

        if not np.all(np.isfinite(x_star)):
            ranks[i] = -1
            continue

        try:
            x_tensor = torch.tensor(x_star, dtype=torch.float32).unsqueeze(0)
            post_samples = sbi_posterior.sample((n_posterior_samples,), x=x_tensor)
            post_np = (
                post_samples.numpy() if hasattr(post_samples, 'numpy')
                else np.asarray(post_samples)
            )
        except (RuntimeError, ValueError, torch.linalg.LinAlgError):
            ranks[i] = -1
            continue

        for p in range(n_params):
            ranks[i, p] = int(np.sum(post_np[:, p] < theta_star[p]))

    valid = np.all(ranks >= 0, axis=1)
    if not valid.all():
        warnings.warn(f"Removed {(~valid).sum()}/{n_sbc_runs} invalid SBC runs")
    ranks = ranks[valid]
    prior_np = prior_np[valid]

    return {
        'ranks':               ranks,
        'thetas':              prior_np,
        'n_posterior_samples': n_posterior_samples,
        'param_names':         param_names,
        'ks_pvalues':          _ks_pvalues(ranks, n_posterior_samples),
    }


# =============================================================================
# PARAMETER RECOVERY
# =============================================================================

def compute_parameter_recovery(
    posterior: Any,
    simulator: Callable,
    prior: Any,
    n_recoveries: int = 100,
    n_posterior_samples: int = 1000,
    seed: int = 42,
    show_progress: bool = True,
    param_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Parameter recovery test.

    Unlike SBC (calibration), this asks whether the posterior MEDIAN
    is close to the true theta. Returns per-parameter diagnostics:
    correlation between true and recovered, RMSE, bias, 90% coverage.

    Args:
        posterior: SBIResult or raw sbi posterior.
        simulator: Callable theta → summary stats.
        prior: Object with .sample((n,)).
        n_recoveries: Number of (theta_true, theta_recovered) pairs.
        n_posterior_samples: Posterior samples per recovery.
        seed: RNG seed.
        show_progress: tqdm bar if available.
        param_names: Optional parameter names.

    Returns:
        Dict ready for plot_recovery_scatter / plot_recovery_bias /
        recovery_summary_table:
            'true_params':       (n_valid, n_params)
            'recovered_median':  (n_valid, n_params)
            'recovered_mean':    (n_valid, n_params)
            'recovered_ci_low':  (n_valid, n_params) 5th percentile
            'recovered_ci_high': (n_valid, n_params) 95th percentile
            'coverage_90':       (n_params,) fraction within 90% CI
            'rmse':              (n_params,)
            'bias':              (n_params,) mean(recovered - true)
            'correlation':       (n_params,) Pearson r
            'param_names':       List[str]
            'n_valid':           int
    """
    import torch

    sbi_posterior = posterior.posterior if hasattr(posterior, 'posterior') else posterior

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    prior_samples = prior.sample((n_recoveries,))
    prior_np = (
        prior_samples.numpy() if hasattr(prior_samples, 'numpy')
        else np.asarray(prior_samples)
    )
    n_params = prior_np.shape[1]

    recovered_median = np.zeros((n_recoveries, n_params))
    recovered_mean   = np.zeros((n_recoveries, n_params))
    recovered_ci_lo  = np.zeros((n_recoveries, n_params))
    recovered_ci_hi  = np.zeros((n_recoveries, n_params))
    valid_mask       = np.ones(n_recoveries, dtype=bool)

    iterator = _maybe_progress(range(n_recoveries), show_progress, 'Recovery')

    for i in iterator:
        theta_true = prior_np[i]
        x_sim = np.asarray(simulator(theta_true))
        if not np.all(np.isfinite(x_sim)):
            valid_mask[i] = False
            continue

        try:
            x_tensor = torch.tensor(x_sim, dtype=torch.float32).unsqueeze(0)
            post = sbi_posterior.sample((n_posterior_samples,), x=x_tensor)
            post_np = post.numpy() if hasattr(post, 'numpy') else np.asarray(post)

            recovered_median[i] = np.median(post_np, axis=0)
            recovered_mean[i]   = np.mean(post_np, axis=0)
            recovered_ci_lo[i]  = np.percentile(post_np, 5,  axis=0)
            recovered_ci_hi[i]  = np.percentile(post_np, 95, axis=0)
        except (RuntimeError, ValueError, torch.linalg.LinAlgError):
            valid_mask[i] = False

    if not valid_mask.all():
        warnings.warn(f"Removed {(~valid_mask).sum()}/{n_recoveries} invalid recovery runs")

    true_v     = prior_np[valid_mask]
    med_v      = recovered_median[valid_mask]
    mean_v     = recovered_mean[valid_mask]
    ci_lo_v    = recovered_ci_lo[valid_mask]
    ci_hi_v    = recovered_ci_hi[valid_mask]

    coverage_90 = np.zeros(n_params)
    rmse        = np.zeros(n_params)
    bias        = np.zeros(n_params)
    correlation = np.zeros(n_params)
    for p in range(n_params):
        in_ci = (true_v[:, p] >= ci_lo_v[:, p]) & (true_v[:, p] <= ci_hi_v[:, p])
        coverage_90[p] = float(np.mean(in_ci))
        rmse[p]        = float(np.sqrt(np.mean((med_v[:, p] - true_v[:, p]) ** 2)))
        bias[p]        = float(np.mean(med_v[:, p] - true_v[:, p]))
        if np.std(true_v[:, p]) > 1e-10 and np.std(med_v[:, p]) > 1e-10:
            correlation[p] = float(np.corrcoef(true_v[:, p], med_v[:, p])[0, 1])
        else:
            correlation[p] = np.nan

    if param_names is None:
        param_names = [f'θ_{p}' for p in range(n_params)]

    return {
        'true_params':       true_v,
        'recovered_median':  med_v,
        'recovered_mean':    mean_v,
        'recovered_ci_low':  ci_lo_v,
        'recovered_ci_high': ci_hi_v,
        'coverage_90':       coverage_90,
        'rmse':              rmse,
        'bias':              bias,
        'correlation':       correlation,
        'param_names':       param_names,
        'n_valid':           int(valid_mask.sum()),
    }


def recovery_summary_table(
    recovery_result: Dict[str, Any],
    print_table: bool = True,
) -> Optional[str]:
    """
    Format the recovery diagnostics as a text table.

    Args:
        recovery_result: Output of compute_parameter_recovery.
        print_table: If True, print to stdout and return None.
            If False, return the text without printing.

    Returns:
        Text string when print_table is False, else None.
    """
    names = recovery_result['param_names']
    n_params = len(recovery_result['rmse'])

    lines = [
        f"Parameter Recovery Summary ({recovery_result['n_valid']} valid runs)",
        "-" * 65,
        f"{'Parameter':<20} {'Corr r':>8} {'RMSE':>8} {'Bias':>8} {'90% Cov':>8}",
        "-" * 65,
    ]

    for p in range(n_params):
        name = names[p] if p < len(names) else f'θ_{p}'
        r    = recovery_result['correlation'][p]
        rmse = recovery_result['rmse'][p]
        bias = recovery_result['bias'][p]
        cov  = recovery_result['coverage_90'][p]

        r_str = f'{r:.3f}' if np.isfinite(r) else 'N/A'
        flag  = '⚠' if cov < 0.8 else ''
        lines.append(f'{name:<20} {r_str:>8} {rmse:>8.4f} {bias:>8.4f} {cov:>7.0%} {flag}')

    lines.append("-" * 65)
    text = '\n'.join(lines)

    if print_table:
        print(text)
        return None
    return text


# =============================================================================
# PARAMETER ↔ STAT CORRELATIONS
# =============================================================================

def compute_param_stat_correlations(
    model_type: str,
    stat_names: List[str],
    burn_in: int = 1000,
    n_samples: int = 1000,
    n_trials: int = 2500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Per-parameter / per-summary-stat correlation matrix.

    Samples theta from the prior, simulates one session per theta,
    computes summary stats, then reports Pearson r between each
    (theta_i, x_j) pair. Useful for diagnosing whether the chosen
    stat set is informative for each parameter.

    Args:
        model_type: 'be' or 'sc'.
        stat_names: Raw (un-expanded) stat names — see behav_utils
            list_available_stats().
        burn_in: Burn-in trials per simulation.
        n_samples: Number of (theta, x) pairs.
        n_trials: Trials per simulation (after burn-in).
        seed: RNG seed.

    Returns:
        Dict ready for plot_param_stat_correlations:
            'corr_matrix':         (n_params, n_stats_expanded) Pearson r
            'param_names':         list
            'stat_names_expanded': list (psychometric expands to 4 etc.)
            'theta':               (n_valid, n_params) raw samples
            'x':                   (n_valid, n_stats_expanded) summary stats
            'n_valid':             int
    """
    import torch
    from behav_utils.data.synthetic import sample_stimuli
    from behav_utils.analysis.summary_stats import get_stat_names_expanded
    from inference.simulator import (
        create_be_simulator, create_sc_simulator,
        get_sbi_prior, wrap_for_sbi,
    )

    rng = np.random.default_rng(seed)
    stim, cat = sample_stimuli(n_trials, 'uniform', rng)

    creator = create_be_simulator if model_type == 'be' else create_sc_simulator
    sim = creator(stim, cat, stat_names=stat_names, burn_in=burn_in, seed=seed)
    prior = get_sbi_prior(sim)
    sbi_sim = wrap_for_sbi(sim)

    param_names = sim.get_param_names()
    stat_names_expanded = get_stat_names_expanded(stat_names)

    theta_samples = prior.sample((n_samples,))
    x_list, theta_valid = [], []
    for t in theta_samples:
        try:
            x = sbi_sim(t)
            if x is not None and not torch.any(torch.isnan(x)):
                x_list.append(x.numpy())
                theta_valid.append(t.numpy())
        except (RuntimeError, ValueError, torch.linalg.LinAlgError):
            pass

    if not x_list:
        raise RuntimeError('All simulations failed during parameter-stat correlation')

    theta_arr = np.asarray(theta_valid)
    x_arr     = np.asarray(x_list)

    n_params = theta_arr.shape[1]
    n_stats  = x_arr.shape[1]
    corr_matrix = np.zeros((n_params, n_stats))
    for i in range(n_params):
        for j in range(n_stats):
            corr_matrix[i, j] = float(np.corrcoef(theta_arr[:, i], x_arr[:, j])[0, 1])

    return {
        'corr_matrix':         corr_matrix,
        'param_names':         param_names,
        'stat_names_expanded': stat_names_expanded,
        'theta':               theta_arr,
        'x':                   x_arr,
        'n_valid':             int(theta_arr.shape[0]),
    }


# =============================================================================
# private helpers
# =============================================================================

def _ks_pvalues(ranks: np.ndarray, n_posterior_samples: int) -> np.ndarray:
    """KS test of rank uniformity, one p-value per parameter."""
    from scipy.stats import kstest
    n_params = ranks.shape[1]
    out = np.zeros(n_params)
    for p in range(n_params):
        normalised = ranks[:, p] / (n_posterior_samples + 1)
        _, out[p] = kstest(normalised, 'uniform')
    return out


def _maybe_progress(iterator, show: bool, desc: str):
    if not show:
        return iterator
    try:
        from tqdm import tqdm
        return tqdm(iterator, desc=desc)
    except ImportError:
        return iterator
