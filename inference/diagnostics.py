"""
SBI diagnostic tools: Simulation-Based Calibration (SBC), parameter recovery,
and associated plotting functions.

Wraps sbi's built-in diagnostics where available and adds custom recovery
analysis tailored to multi-session BE model inference.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union,
)
import warnings


# =============================================================================
# SIMULATION-BASED CALIBRATION (SBC)
# =============================================================================

def run_sbc(
    posterior: Any,
    simulator: Callable,
    prior: Any,
    n_sbc_runs: int = 1000,
    n_posterior_samples: int = 1000,
    observed_stats: Optional[np.ndarray] = None,
    seed: int = 42,
    show_progress: bool = True,
    param_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run Simulation-Based Calibration (SBC) to validate posterior calibration.
    
    For each run:
      1. Sample theta* from prior
      2. Simulate x* = simulator(theta*)
      3. Draw posterior samples given x*
      4. Compute rank of theta* among posterior samples
    
    If posterior is well-calibrated, ranks should be uniformly distributed.
    
    Tries to use sbi's built-in run_sbc; falls back to manual implementation.
    
    Args:
        posterior: Trained SBI posterior (SBIResult or raw sbi posterior).
                   Must support .sample(n, x=x_obs).
        simulator: Callable theta -> summary_stats (numpy arrays).
        prior: Prior with .sample() method.
        n_sbc_runs: Number of SBC iterations.
        n_posterior_samples: Posterior samples per iteration.
        observed_stats: Not used directly but kept for API consistency.
        seed: Random seed.
        show_progress: Show progress bar.
        param_names: Names for parameter dimensions.
    
    Returns:
        Dict with:
            'ranks': (n_sbc_runs, n_params) array of ranks
            'thetas': (n_sbc_runs, n_params) ground truth thetas
            'n_posterior_samples': number of posterior samples used
            'param_names': parameter names
            'ks_pvalues': KS test p-values per parameter
    """
    import torch
    
    # Unwrap SBIResult if needed
    if hasattr(posterior, 'posterior'):
        sbi_posterior = posterior.posterior
    else:
        sbi_posterior = posterior
    
    # Try sbi's built-in SBC first
    try:
        from sbi.analysis import run_sbc as _sbi_run_sbc, check_sbc
        
        # sbi's run_sbc expects a specific interface
        # It may not work with our custom priors, so wrap in try
        sbc_result = _sbi_run_sbc(
            sbi_posterior,
            simulator,
            prior,
            num_sbc_runs=n_sbc_runs,
            num_posterior_samples=n_posterior_samples,
        )
        # sbi returns (ranks, thetas) or similar
        if isinstance(sbc_result, tuple):
            ranks, thetas = sbc_result[0], sbc_result[1]
        else:
            ranks = sbc_result
            thetas = None
        
        ranks_np = ranks.numpy() if hasattr(ranks, 'numpy') else np.asarray(ranks)
        
        # KS test
        ks_pvalues = _compute_ks_pvalues(ranks_np, n_posterior_samples)
        
        return {
            'ranks': ranks_np,
            'thetas': thetas.numpy() if thetas is not None and hasattr(thetas, 'numpy') else thetas,
            'n_posterior_samples': n_posterior_samples,
            'param_names': param_names,
            'ks_pvalues': ks_pvalues,
        }
    
    except Exception as e:
        if show_progress:
            print(f"sbi built-in SBC failed ({e}), using manual implementation...")
    
    # Manual SBC implementation
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    
    # Sample from prior
    prior_samples = prior.sample((n_sbc_runs,))
    if hasattr(prior_samples, 'numpy'):
        prior_np = prior_samples.numpy()
    else:
        prior_np = np.asarray(prior_samples)
    
    n_params = prior_np.shape[1]
    ranks = np.zeros((n_sbc_runs, n_params), dtype=int)
    
    iterator = range(n_sbc_runs)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc='SBC')
        except ImportError:
            print(f"Running {n_sbc_runs} SBC iterations...")
    
    for i in iterator:
        theta_star = prior_np[i]
        
        # Simulate
        x_star = simulator(theta_star)
        x_tensor = torch.tensor(x_star, dtype=torch.float32).unsqueeze(0)
        
        # Check for NaN/inf
        if not np.all(np.isfinite(np.asarray(x_star))):
            ranks[i] = -1  # Mark as invalid
            continue
        
        # Sample from posterior
        try:
            post_samples = sbi_posterior.sample(
                (n_posterior_samples,), x=x_tensor
            )
            if hasattr(post_samples, 'numpy'):
                post_np = post_samples.numpy()
            else:
                post_np = np.asarray(post_samples)
        except Exception:
            ranks[i] = -1
            continue
        
        # Compute ranks
        for p in range(n_params):
            ranks[i, p] = np.sum(post_np[:, p] < theta_star[p])
    
    # Remove invalid runs
    valid = np.all(ranks >= 0, axis=1)
    if not valid.all():
        n_invalid = (~valid).sum()
        warnings.warn(f"Removed {n_invalid}/{n_sbc_runs} invalid SBC runs")
    ranks = ranks[valid]
    prior_np = prior_np[valid]
    
    ks_pvalues = _compute_ks_pvalues(ranks, n_posterior_samples)
    
    return {
        'ranks': ranks,
        'thetas': prior_np,
        'n_posterior_samples': n_posterior_samples,
        'param_names': param_names,
        'ks_pvalues': ks_pvalues,
    }


def _compute_ks_pvalues(ranks: np.ndarray, n_posterior_samples: int) -> np.ndarray:
    """KS test of rank uniformity for each parameter."""
    from scipy.stats import kstest
    n_params = ranks.shape[1]
    pvalues = np.zeros(n_params)
    for p in range(n_params):
        # Normalise ranks to [0, 1]
        normalised = ranks[:, p] / (n_posterior_samples + 1)
        stat, pval = kstest(normalised, 'uniform')
        pvalues[p] = pval
    return pvalues


# =============================================================================
# SBC PLOTTING
# =============================================================================

def plot_sbc_ranks(
    sbc_result: Dict[str, Any],
    param_indices: Optional[List[int]] = None,
    n_bins: int = 20,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot SBC rank histograms.
    
    Uniform ranks indicate well-calibrated posterior.
    Deviations indicate:
        - U-shape: posterior too narrow (overconfident)
        - Inverted U: posterior too wide (underconfident)
        - Skewed: systematic bias
    
    Args:
        sbc_result: Output from run_sbc().
        param_indices: Which parameters to plot (indices). Default: all.
        n_bins: Number of histogram bins.
        figsize: Figure size.
        title: Overall title.
    
    Returns:
        Matplotlib figure.
    """
    ranks = sbc_result['ranks']
    n_posterior_samples = sbc_result['n_posterior_samples']
    names = sbc_result.get('param_names')
    ks_pvals = sbc_result.get('ks_pvalues')
    
    n_params = ranks.shape[1]
    if param_indices is None:
        param_indices = list(range(n_params))
    
    n_plot = len(param_indices)
    n_cols = min(4, n_plot)
    n_rows = int(np.ceil(n_plot / n_cols))
    
    if figsize is None:
        figsize = (4 * n_cols, 3.5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    
    expected_count = len(ranks) / n_bins
    
    for idx, p in enumerate(param_indices):
        ax = axes_flat[idx]
        
        ax.hist(ranks[:, p], bins=n_bins, density=False,
                color='steelblue', edgecolor='white', alpha=0.8)
        
        # Expected uniform line
        ax.axhline(expected_count, color='red', linestyle='--',
                   linewidth=1.5, alpha=0.7, label='Expected (uniform)')
        
        # 95% CI for uniform
        from scipy.stats import binom
        ci_lo = binom.ppf(0.025, len(ranks), 1 / n_bins)
        ci_hi = binom.ppf(0.975, len(ranks), 1 / n_bins)
        ax.axhspan(ci_lo, ci_hi, alpha=0.1, color='red', zorder=0)
        
        label = names[p] if names is not None else f'θ_{p}'
        ks_str = ''
        if ks_pvals is not None:
            pval = ks_pvals[p]
            ks_str = f'\nKS p={pval:.3f}'
            if pval < 0.05:
                ks_str += ' ⚠'
        
        ax.set_title(f'{label}{ks_str}', fontsize=9)
        ax.set_xlabel('Rank')
        ax.set_ylabel('Count')
    
    for j in range(n_plot, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    else:
        fig.suptitle('SBC Rank Histograms', fontsize=13, y=1.02)
    
    fig.tight_layout()
    return fig


def plot_sbc_ecdf(
    sbc_result: Dict[str, Any],
    param_indices: Optional[List[int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot SBC empirical CDF of ranks.
    
    Deviation from diagonal indicates miscalibration.
    
    Args:
        sbc_result: Output from run_sbc().
        param_indices: Which parameters to plot. Default: all.
        figsize: Figure size.
        title: Overall title.
    
    Returns:
        Matplotlib figure.
    """
    ranks = sbc_result['ranks']
    n_posterior_samples = sbc_result['n_posterior_samples']
    names = sbc_result.get('param_names')
    
    n_params = ranks.shape[1]
    if param_indices is None:
        param_indices = list(range(n_params))
    
    n_plot = len(param_indices)
    n_cols = min(4, n_plot)
    n_rows = int(np.ceil(n_plot / n_cols))
    
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    
    n_sbc = len(ranks)
    
    for idx, p in enumerate(param_indices):
        ax = axes_flat[idx]
        
        normalised = np.sort(ranks[:, p]) / (n_posterior_samples + 1)
        ecdf_y = np.arange(1, n_sbc + 1) / n_sbc
        
        ax.plot(normalised, ecdf_y, color='steelblue', linewidth=1.5)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=1)
        
        # Kolmogorov-Smirnov band (approximate 95%)
        ks_crit = 1.36 / np.sqrt(n_sbc)
        ax.fill_between([0, 1],
                        [0 - ks_crit, 1 - ks_crit],
                        [0 + ks_crit, 1 + ks_crit],
                        alpha=0.1, color='red')
        
        label = names[p] if names is not None else f'θ_{p}'
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Normalised rank')
        ax.set_ylabel('ECDF')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
    
    for j in range(n_plot, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    
    fig.tight_layout()
    return fig


# =============================================================================
# PARAMETER RECOVERY
# =============================================================================

def parameter_recovery(
    posterior: Any,
    simulator: Callable,
    prior: Any,
    layout: Any = None,
    n_recoveries: int = 100,
    n_posterior_samples: int = 1000,
    seed: int = 42,
    show_progress: bool = True,
    param_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run parameter recovery: sample from prior, simulate, infer, compare.
    
    Unlike SBC (which tests calibration), parameter recovery tests whether
    the posterior point estimates (median) are close to the true values.
    
    Args:
        posterior: Trained SBI posterior (SBIResult or raw sbi posterior).
        simulator: Callable theta -> summary_stats.
        prior: Prior with .sample() method.
        layout: Optional ThetaLayout for extracting per-parameter trajectories.
        n_recoveries: Number of recovery tests.
        n_posterior_samples: Posterior samples per recovery.
        seed: Random seed.
        show_progress: Show progress bar.
        param_names: Names for all theta dimensions.
    
    Returns:
        Dict with:
            'true_params': (n_recoveries, n_params) ground truth
            'recovered_median': (n_recoveries, n_params) posterior medians
            'recovered_mean': (n_recoveries, n_params) posterior means
            'recovered_ci_low': (n_recoveries, n_params) 5th percentile
            'recovered_ci_high': (n_recoveries, n_params) 95th percentile
            'coverage_90': fraction of true values within 90% CI per param
            'rmse': RMSE per parameter
            'correlation': Pearson r per parameter
            'param_names': parameter names
    """
    import torch
    
    # Unwrap SBIResult
    if hasattr(posterior, 'posterior'):
        sbi_posterior = posterior.posterior
    else:
        sbi_posterior = posterior
    
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    
    # Sample ground truths from prior
    prior_samples = prior.sample((n_recoveries,))
    if hasattr(prior_samples, 'numpy'):
        prior_np = prior_samples.numpy()
    else:
        prior_np = np.asarray(prior_samples)
    
    n_params = prior_np.shape[1]
    
    recovered_median = np.zeros((n_recoveries, n_params))
    recovered_mean = np.zeros((n_recoveries, n_params))
    recovered_ci_low = np.zeros((n_recoveries, n_params))
    recovered_ci_high = np.zeros((n_recoveries, n_params))
    valid_mask = np.ones(n_recoveries, dtype=bool)
    
    iterator = range(n_recoveries)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc='Recovery')
        except ImportError:
            print(f"Running {n_recoveries} recovery tests...")
    
    for i in iterator:
        theta_true = prior_np[i]
        
        # Simulate
        x_sim = simulator(theta_true)
        x_sim = np.asarray(x_sim)
        if not np.all(np.isfinite(x_sim)):
            valid_mask[i] = False
            continue
        
        x_tensor = torch.tensor(x_sim, dtype=torch.float32).unsqueeze(0)
        
        # Posterior samples
        try:
            post_samples = sbi_posterior.sample(
                (n_posterior_samples,), x=x_tensor
            )
            if hasattr(post_samples, 'numpy'):
                post_np = post_samples.numpy()
            else:
                post_np = np.asarray(post_samples)
            
            recovered_median[i] = np.median(post_np, axis=0)
            recovered_mean[i] = np.mean(post_np, axis=0)
            recovered_ci_low[i] = np.percentile(post_np, 5, axis=0)
            recovered_ci_high[i] = np.percentile(post_np, 95, axis=0)
        except Exception:
            valid_mask[i] = False
    
    # Filter invalid
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        warnings.warn(f"Removed {n_invalid}/{n_recoveries} invalid recovery runs")
    
    true_valid = prior_np[valid_mask]
    med_valid = recovered_median[valid_mask]
    mean_valid = recovered_mean[valid_mask]
    ci_lo_valid = recovered_ci_low[valid_mask]
    ci_hi_valid = recovered_ci_high[valid_mask]
    
    # Compute diagnostics per parameter
    coverage_90 = np.zeros(n_params)
    rmse = np.zeros(n_params)
    correlation = np.zeros(n_params)
    bias = np.zeros(n_params)
    
    for p in range(n_params):
        in_ci = (true_valid[:, p] >= ci_lo_valid[:, p]) & \
                (true_valid[:, p] <= ci_hi_valid[:, p])
        coverage_90[p] = np.mean(in_ci)
        
        rmse[p] = np.sqrt(np.mean((med_valid[:, p] - true_valid[:, p]) ** 2))
        bias[p] = np.mean(med_valid[:, p] - true_valid[:, p])
        
        if np.std(true_valid[:, p]) > 1e-10 and np.std(med_valid[:, p]) > 1e-10:
            correlation[p] = np.corrcoef(true_valid[:, p], med_valid[:, p])[0, 1]
        else:
            correlation[p] = np.nan
    
    if param_names is None:
        param_names = [f'θ_{p}' for p in range(n_params)]
    
    return {
        'true_params': true_valid,
        'recovered_median': med_valid,
        'recovered_mean': mean_valid,
        'recovered_ci_low': ci_lo_valid,
        'recovered_ci_high': ci_hi_valid,
        'coverage_90': coverage_90,
        'rmse': rmse,
        'bias': bias,
        'correlation': correlation,
        'param_names': param_names,
        'n_valid': int(valid_mask.sum()),
    }


# =============================================================================
# RECOVERY PLOTTING
# =============================================================================

def plot_recovery_scatter(
    recovery_result: Dict[str, Any],
    param_indices: Optional[List[int]] = None,
    prior_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    param_links: Optional[Dict[str, Any]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of true vs recovered parameter values.
    
    Points should cluster around the identity line for good recovery.
    
    Args:
        recovery_result: Output from parameter_recovery().
        param_indices: Which parameters to plot. Default: all.
        prior_bounds: Dict mapping param name -> (low, high). Sets axis range.
        param_links: Dict of link specs; bounds extracted if prior_bounds is None.
        figsize: Figure size.
        title: Overall title.
    
    Returns:
        Matplotlib figure.
    """
    true_params = recovery_result['true_params']
    recovered = recovery_result['recovered_median']
    ci_lo = recovery_result['recovered_ci_low']
    ci_hi = recovery_result['recovered_ci_high']
    names = recovery_result['param_names']
    corrs = recovery_result['correlation']
    coverage = recovery_result['coverage_90']
    
    # Auto-extract bounds from param_links
    if prior_bounds is None and param_links is not None:
        prior_bounds = {}
        for name, link in param_links.items():
            if hasattr(link, 'bounds'):
                prior_bounds[name] = link.bounds
    
    n_params = true_params.shape[1]
    if param_indices is None:
        param_indices = list(range(n_params))
    
    n_plot = len(param_indices)
    n_cols = min(4, n_plot)
    n_rows = int(np.ceil(n_plot / n_cols))
    
    if figsize is None:
        figsize = (4.5 * n_cols, 4.5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    
    for idx, p in enumerate(param_indices):
        ax = axes_flat[idx]
        
        name = names[p] if p < len(names) else f'θ_{p}'
        
        # Error bars
        yerr_lo = recovered[:, p] - ci_lo[:, p]
        yerr_hi = ci_hi[:, p] - recovered[:, p]
        
        ax.errorbar(
            true_params[:, p], recovered[:, p],
            yerr=[yerr_lo, yerr_hi],
            fmt='o', markersize=3, alpha=0.4,
            elinewidth=0.5, capsize=0,
            color='steelblue',
        )
        
        # Set axis range to prior bounds
        if prior_bounds is not None and name in prior_bounds:
            lo, hi = prior_bounds[name]
            padding = (hi - lo) * 0.08
            ax_lo, ax_hi = lo - padding, hi + padding
        else:
            # Fall back to data range
            all_vals = np.concatenate([true_params[:, p], recovered[:, p]])
            lo, hi = np.min(all_vals), np.max(all_vals)
            padding = (hi - lo) * 0.1
            ax_lo, ax_hi = lo - padding, hi + padding
        
        ax.set_xlim(ax_lo, ax_hi)
        ax.set_ylim(ax_lo, ax_hi)
        
        # Identity line
        ax.plot([ax_lo, ax_hi], [ax_lo, ax_hi], 'k--',
                linewidth=1, alpha=0.5)
        
        r = corrs[p]
        cov = coverage[p]
        r_str = f'r={r:.2f}' if np.isfinite(r) else 'r=N/A'
        ax.set_title(f'{name}\n{r_str}, 90% cov={cov:.0%}', fontsize=9)
        ax.set_xlabel('True')
        ax.set_ylabel('Recovered (median)')
        ax.set_aspect('equal')
    
    for j in range(n_plot, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    else:
        fig.suptitle('Parameter Recovery: True vs Recovered', fontsize=13, y=1.02)
    
    fig.tight_layout()
    return fig


def plot_recovery_bias(
    recovery_result: Dict[str, Any],
    param_indices: Optional[List[int]] = None,
    prior_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    param_links: Optional[Dict[str, Any]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot recovery bias (recovered - true) as a function of true value.
    
    Horizontal band around zero indicates unbiased recovery.
    Systematic trends indicate parameter-dependent bias.
    
    Args:
        recovery_result: Output from parameter_recovery().
        param_indices: Which parameters to plot. Default: all.
        prior_bounds: Dict mapping param name -> (low, high). Sets x-axis range.
        param_links: Dict of link specs; bounds extracted if prior_bounds is None.
        figsize: Figure size.
        title: Overall title.
    
    Returns:
        Matplotlib figure.
    """
    true_params = recovery_result['true_params']
    recovered = recovery_result['recovered_median']
    names = recovery_result['param_names']
    rmse_vals = recovery_result['rmse']
    bias_vals = recovery_result['bias']
    
    # Auto-extract bounds
    if prior_bounds is None and param_links is not None:
        prior_bounds = {}
        for name, link in param_links.items():
            if hasattr(link, 'bounds'):
                prior_bounds[name] = link.bounds
    
    n_params = true_params.shape[1]
    if param_indices is None:
        param_indices = list(range(n_params))
    
    n_plot = len(param_indices)
    n_cols = min(4, n_plot)
    n_rows = int(np.ceil(n_plot / n_cols))
    
    if figsize is None:
        figsize = (4.5 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    
    for idx, p in enumerate(param_indices):
        ax = axes_flat[idx]
        
        name = names[p] if p < len(names) else f'θ_{p}'
        error = recovered[:, p] - true_params[:, p]
        
        ax.scatter(true_params[:, p], error, s=15, alpha=0.4,
                   color='steelblue', edgecolors='none')
        ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        
        # LOWESS or running mean trend
        try:
            sort_idx = np.argsort(true_params[:, p])
            x_sorted = true_params[sort_idx, p]
            e_sorted = error[sort_idx]
            # Running mean with window
            window = max(5, len(e_sorted) // 10)
            if len(e_sorted) > window:
                running_mean = np.convolve(e_sorted, np.ones(window) / window, mode='valid')
                x_running = x_sorted[window // 2: window // 2 + len(running_mean)]
                ax.plot(x_running, running_mean, 'r-', linewidth=2, alpha=0.7)
        except Exception:
            pass
        
        # Set x-axis to prior bounds
        if prior_bounds is not None and name in prior_bounds:
            lo, hi = prior_bounds[name]
            padding = (hi - lo) * 0.08
            ax.set_xlim(lo - padding, hi + padding)
        
        ax.set_title(
            f'{name}\nbias={bias_vals[p]:.4f}, RMSE={rmse_vals[p]:.4f}',
            fontsize=9,
        )
        ax.set_xlabel('True value')
        ax.set_ylabel('Error (recovered − true)')
    
    for j in range(n_plot, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    else:
        fig.suptitle('Parameter Recovery: Bias Analysis', fontsize=13, y=1.02)
    
    fig.tight_layout()
    return fig


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def recovery_summary_table(
    recovery_result: Dict[str, Any],
    print_table: bool = True,
) -> Optional[str]:
    """
    Print / return a formatted summary of parameter recovery diagnostics.
    
    Args:
        recovery_result: Output from parameter_recovery().
        print_table: Whether to print immediately.
    
    Returns:
        Formatted string if print_table is False.
    """
    names = recovery_result['param_names']
    n_params = len(recovery_result['rmse'])
    
    lines = []
    lines.append(f"Parameter Recovery Summary ({recovery_result['n_valid']} valid runs)")
    lines.append("-" * 65)
    lines.append(f"{'Parameter':<20} {'Corr r':>8} {'RMSE':>8} {'Bias':>8} {'90% Cov':>8}")
    lines.append("-" * 65)
    
    for p in range(n_params):
        name = names[p] if p < len(names) else f'θ_{p}'
        r = recovery_result['correlation'][p]
        rmse = recovery_result['rmse'][p]
        bias = recovery_result['bias'][p]
        cov = recovery_result['coverage_90'][p]
        
        r_str = f'{r:.3f}' if np.isfinite(r) else 'N/A'
        flag = '⚠' if cov < 0.8 else ''
        
        lines.append(f'{name:<20} {r_str:>8} {rmse:>8.4f} {bias:>8.4f} {cov:>7.0%} {flag}')
    
    lines.append("-" * 65)
    
    text = '\n'.join(lines)
    if print_table:
        print(text)
    else:
        return text


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'run_sbc',
    'plot_sbc_ranks',
    'plot_sbc_ecdf',
    'parameter_recovery',
    'plot_recovery_scatter',
    'plot_recovery_bias',
    'recovery_summary_table',
]
