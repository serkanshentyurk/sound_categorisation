"""
Diagnostics for SBI Inference.

Provides validation tools for simulation-based inference:
- Simulation-Based Calibration (SBC)
- Parameter recovery analysis
- Posterior predictive checks
- Coverage diagnostics
- Visualisation functions

Usage:
    from Inference.diagnostics import run_sbc, plot_sbc_ranks, parameter_recovery
    
    # SBC
    sbc_result = run_sbc(simulator, prior, posterior, n_sbc=500)
    plot_sbc_ranks(sbc_result)
    
    # Parameter recovery
    recovery = parameter_recovery(simulator, prior, posterior, n_tests=100)
    plot_recovery_scatter(recovery)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
import warnings
from scipy import stats


# =============================================================================
# RESULT CONTAINERS
# =============================================================================

@dataclass
class SBCResult:
    """
    Results from Simulation-Based Calibration.
    
    Attributes:
        ranks: Rank statistics for each parameter, shape (n_sbc, n_params)
        theta_true: True parameter values, shape (n_sbc, n_params)
        theta_samples: Posterior samples, shape (n_sbc, n_posterior_samples, n_params)
        n_sbc: Number of SBC iterations
        n_posterior_samples: Number of posterior samples per iteration
        param_names: Parameter names
        uniformity_pvalues: p-values from uniformity tests
        ks_statistics: KS test statistics
    """
    ranks: np.ndarray
    theta_true: np.ndarray
    theta_samples: Optional[np.ndarray]
    n_sbc: int
    n_posterior_samples: int
    param_names: List[str]
    uniformity_pvalues: Optional[Dict[str, float]] = None
    ks_statistics: Optional[Dict[str, float]] = None
    
    def is_calibrated(self, alpha: float = 0.05) -> Dict[str, bool]:
        """
        Check if posterior is calibrated for each parameter.
        
        Uses KS test for uniformity of ranks.
        Returns True if we cannot reject uniformity at level alpha.
        """
        if self.uniformity_pvalues is None:
            self._compute_uniformity_tests()
        
        return {name: pval > alpha for name, pval in self.uniformity_pvalues.items()}
    
    def _compute_uniformity_tests(self):
        """Compute KS tests for uniformity of ranks."""
        self.uniformity_pvalues = {}
        self.ks_statistics = {}
        
        # Ranks should be uniform on [0, n_posterior_samples]
        # Normalize to [0, 1] for KS test
        normalized_ranks = self.ranks / self.n_posterior_samples
        
        for i, name in enumerate(self.param_names):
            ks_stat, pval = stats.kstest(normalized_ranks[:, i], 'uniform')
            self.uniformity_pvalues[name] = pval
            self.ks_statistics[name] = ks_stat
    
    def summary(self) -> str:
        """Return text summary of SBC results."""
        if self.uniformity_pvalues is None:
            self._compute_uniformity_tests()
        
        lines = ["SBC Summary", "=" * 40]
        lines.append(f"N iterations: {self.n_sbc}")
        lines.append(f"Posterior samples per iteration: {self.n_posterior_samples}")
        lines.append("")
        lines.append("Uniformity Tests (KS):")
        lines.append("-" * 40)
        
        for name in self.param_names:
            pval = self.uniformity_pvalues[name]
            ks = self.ks_statistics[name]
            status = "✓" if pval > 0.05 else "✗"
            lines.append(f"  {name:20s}: KS={ks:.3f}, p={pval:.3f} {status}")
        
        return "\n".join(lines)


@dataclass
class RecoveryResult:
    """
    Results from parameter recovery analysis.
    
    Attributes:
        theta_true: True parameter values, shape (n_tests, n_params)
        theta_estimated: Point estimates (posterior mean), shape (n_tests, n_params)
        theta_lower: Lower CI bound, shape (n_tests, n_params)
        theta_upper: Upper CI bound, shape (n_tests, n_params)
        ci_level: Credible interval level (e.g., 0.95)
        param_names: Parameter names
        correlations: Correlation between true and estimated
        biases: Mean bias for each parameter
        rmses: Root mean squared error
        coverages: Empirical coverage of credible intervals
    """
    theta_true: np.ndarray
    theta_estimated: np.ndarray
    theta_lower: np.ndarray
    theta_upper: np.ndarray
    ci_level: float
    param_names: List[str]
    correlations: Optional[Dict[str, float]] = None
    biases: Optional[Dict[str, float]] = None
    rmses: Optional[Dict[str, float]] = None
    coverages: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        self._compute_metrics()
    
    def _compute_metrics(self):
        """Compute recovery metrics."""
        self.correlations = {}
        self.biases = {}
        self.rmses = {}
        self.coverages = {}
        
        for i, name in enumerate(self.param_names):
            true = self.theta_true[:, i]
            est = self.theta_estimated[:, i]
            lower = self.theta_lower[:, i]
            upper = self.theta_upper[:, i]
            
            # Correlation
            self.correlations[name] = float(np.corrcoef(true, est)[0, 1])
            
            # Bias
            self.biases[name] = float(np.mean(est - true))
            
            # RMSE
            self.rmses[name] = float(np.sqrt(np.mean((est - true) ** 2)))
            
            # Coverage
            covered = (true >= lower) & (true <= upper)
            self.coverages[name] = float(np.mean(covered))
    
    def summary(self) -> str:
        """Return text summary of recovery results."""
        lines = ["Parameter Recovery Summary", "=" * 50]
        lines.append(f"N tests: {len(self.theta_true)}")
        lines.append(f"CI level: {self.ci_level:.0%}")
        lines.append("")
        lines.append(f"{'Parameter':<20} {'Corr':>8} {'Bias':>10} {'RMSE':>10} {'Coverage':>10}")
        lines.append("-" * 60)
        
        for name in self.param_names:
            corr = self.correlations[name]
            bias = self.biases[name]
            rmse = self.rmses[name]
            cov = self.coverages[name]
            expected_cov = self.ci_level
            cov_status = "✓" if abs(cov - expected_cov) < 0.1 else "✗"
            
            lines.append(f"{name:<20} {corr:>8.3f} {bias:>10.4f} {rmse:>10.4f} {cov:>9.1%} {cov_status}")
        
        return "\n".join(lines)


@dataclass
class PosteriorPredictiveResult:
    """
    Results from posterior predictive checks.
    
    Attributes:
        observed_stats: Observed summary statistics
        predicted_stats: Predicted statistics from posterior samples, shape (n_samples, n_stats)
        stat_names: Names of summary statistics
        ppc_pvalues: Bayesian p-values (proportion of predictions >= observed)
    """
    observed_stats: np.ndarray
    predicted_stats: np.ndarray
    stat_names: List[str]
    ppc_pvalues: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        self._compute_pvalues()
    
    def _compute_pvalues(self):
        """Compute posterior predictive p-values."""
        self.ppc_pvalues = {}
        
        for i, name in enumerate(self.stat_names):
            obs = self.observed_stats[i]
            pred = self.predicted_stats[:, i]
            # Two-sided: proportion in tails
            pval = 2 * min(np.mean(pred >= obs), np.mean(pred <= obs))
            self.ppc_pvalues[name] = float(pval)
    
    def summary(self) -> str:
        """Return text summary."""
        lines = ["Posterior Predictive Check Summary", "=" * 50]
        lines.append(f"N posterior samples: {len(self.predicted_stats)}")
        lines.append("")
        lines.append(f"{'Statistic':<25} {'Observed':>12} {'Pred Mean':>12} {'Pred Std':>10} {'p-value':>10}")
        lines.append("-" * 70)
        
        for i, name in enumerate(self.stat_names):
            obs = self.observed_stats[i]
            pred_mean = np.mean(self.predicted_stats[:, i])
            pred_std = np.std(self.predicted_stats[:, i])
            pval = self.ppc_pvalues[name]
            status = "✓" if pval > 0.05 else "?"
            
            lines.append(f"{name:<25} {obs:>12.4f} {pred_mean:>12.4f} {pred_std:>10.4f} {pval:>9.3f} {status}")
        
        return "\n".join(lines)


# =============================================================================
# SIMULATION-BASED CALIBRATION
# =============================================================================

def run_sbc(
    simulator: Callable,
    prior: Any,
    posterior: Any,
    n_sbc: int = 500,
    n_posterior_samples: int = 1000,
    param_names: Optional[List[str]] = None,
    seed: Optional[int] = None,
    show_progress: bool = True
) -> SBCResult:
    """
    Run Simulation-Based Calibration.
    
    SBC tests whether the posterior is correctly calibrated by checking
    if rank statistics are uniformly distributed.
    
    Algorithm:
        1. Sample θ_true from prior
        2. Simulate data x from model with θ_true
        3. Compute summary stats
        4. Sample from posterior given summary stats
        5. Compute rank of θ_true among posterior samples
        6. Repeat and check ranks are uniform
    
    Args:
        simulator: Callable(theta) -> summary_stats
        prior: Prior distribution (must have .sample())
        posterior: Posterior (must have .sample(n, x=...))
        n_sbc: Number of SBC iterations
        n_posterior_samples: Posterior samples per iteration
        param_names: Parameter names
        seed: Random seed
        show_progress: Print progress
    
    Returns:
        SBCResult with rank statistics and diagnostics
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Get parameter dimensionality
    test_sample = prior.sample((1,))
    if isinstance(test_sample, torch.Tensor):
        n_params = test_sample.shape[-1]
    else:
        n_params = len(test_sample.flatten())
    
    if param_names is None:
        if hasattr(prior, 'param_names'):
            param_names = prior.param_names
        else:
            param_names = [f'param_{i}' for i in range(n_params)]
    
    # Storage
    ranks = np.zeros((n_sbc, n_params))
    theta_true_all = np.zeros((n_sbc, n_params))
    
    for i in range(n_sbc):
        if show_progress and (i + 1) % 50 == 0:
            print(f"SBC iteration {i + 1}/{n_sbc}")
        
        # 1. Sample from prior
        theta_true = prior.sample((1,))
        if isinstance(theta_true, torch.Tensor):
            theta_true_np = theta_true.numpy().flatten()
            theta_true = theta_true.squeeze()
        else:
            theta_true_np = np.array(theta_true).flatten()
        
        theta_true_all[i] = theta_true_np
        
        # 2-3. Simulate and get summary stats
        if isinstance(theta_true, torch.Tensor):
            x = simulator(theta_true.numpy())
        else:
            x = simulator(theta_true)
        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        # 4. Sample from posterior
        try:
            posterior_samples = posterior.sample((n_posterior_samples,), x=x)
            if isinstance(posterior_samples, torch.Tensor):
                posterior_samples = posterior_samples.numpy()
        except Exception as e:
            warnings.warn(f"Posterior sampling failed at iteration {i}: {e}")
            ranks[i] = np.nan
            continue
        
        # 5. Compute ranks
        for j in range(n_params):
            ranks[i, j] = np.sum(posterior_samples[:, j] < theta_true_np[j])
    
    # Remove failed iterations
    valid = ~np.isnan(ranks[:, 0])
    if not valid.all():
        warnings.warn(f"Removed {(~valid).sum()} failed SBC iterations")
        ranks = ranks[valid]
        theta_true_all = theta_true_all[valid]
    
    result = SBCResult(
        ranks=ranks,
        theta_true=theta_true_all,
        theta_samples=None,  # Don't store all samples by default
        n_sbc=len(ranks),
        n_posterior_samples=n_posterior_samples,
        param_names=param_names
    )
    
    return result


# =============================================================================
# PARAMETER RECOVERY
# =============================================================================

def parameter_recovery(
    simulator: Callable,
    prior: Any,
    posterior: Any,
    n_tests: int = 100,
    n_posterior_samples: int = 2000,
    ci_level: float = 0.95,
    param_names: Optional[List[str]] = None,
    seed: Optional[int] = None,
    show_progress: bool = True
) -> RecoveryResult:
    """
    Test parameter recovery across many simulated datasets.
    
    For each test:
        1. Sample true parameters from prior
        2. Simulate data
        3. Estimate posterior
        4. Compare point estimates and CIs to true values
    
    Args:
        simulator: Callable(theta) -> summary_stats
        prior: Prior distribution
        posterior: Posterior (from trained SBI)
        n_tests: Number of test datasets
        n_posterior_samples: Samples for posterior summaries
        ci_level: Credible interval level
        param_names: Parameter names
        seed: Random seed
        show_progress: Print progress
    
    Returns:
        RecoveryResult with correlations, biases, RMSE, coverage
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Get dimensionality
    test_sample = prior.sample((1,))
    if isinstance(test_sample, torch.Tensor):
        n_params = test_sample.shape[-1]
    else:
        n_params = len(test_sample.flatten())
    
    if param_names is None:
        if hasattr(prior, 'param_names'):
            param_names = prior.param_names
        else:
            param_names = [f'param_{i}' for i in range(n_params)]
    
    # Storage
    theta_true = np.zeros((n_tests, n_params))
    theta_estimated = np.zeros((n_tests, n_params))
    theta_lower = np.zeros((n_tests, n_params))
    theta_upper = np.zeros((n_tests, n_params))
    
    alpha = (1 - ci_level) / 2
    
    for i in range(n_tests):
        if show_progress and (i + 1) % 20 == 0:
            print(f"Recovery test {i + 1}/{n_tests}")
        
        # Sample true parameters
        theta = prior.sample((1,))
        if isinstance(theta, torch.Tensor):
            theta_np = theta.numpy().flatten()
            theta = theta.squeeze()
        else:
            theta_np = np.array(theta).flatten()
        
        theta_true[i] = theta_np
        
        # Simulate
        if isinstance(theta, torch.Tensor):
            x = simulator(theta.numpy())
        else:
            x = simulator(theta)
        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Get posterior samples
        try:
            samples = posterior.sample((n_posterior_samples,), x=x)
            if isinstance(samples, torch.Tensor):
                samples = samples.numpy()
            
            # Point estimate (posterior mean)
            theta_estimated[i] = samples.mean(axis=0)
            
            # Credible interval
            theta_lower[i] = np.quantile(samples, alpha, axis=0)
            theta_upper[i] = np.quantile(samples, 1 - alpha, axis=0)
            
        except Exception as e:
            warnings.warn(f"Recovery failed at test {i}: {e}")
            theta_estimated[i] = np.nan
            theta_lower[i] = np.nan
            theta_upper[i] = np.nan
    
    # Remove failed tests
    valid = ~np.isnan(theta_estimated[:, 0])
    if not valid.all():
        warnings.warn(f"Removed {(~valid).sum()} failed tests")
        theta_true = theta_true[valid]
        theta_estimated = theta_estimated[valid]
        theta_lower = theta_lower[valid]
        theta_upper = theta_upper[valid]
    
    return RecoveryResult(
        theta_true=theta_true,
        theta_estimated=theta_estimated,
        theta_lower=theta_lower,
        theta_upper=theta_upper,
        ci_level=ci_level,
        param_names=param_names
    )


# =============================================================================
# POSTERIOR PREDICTIVE CHECKS
# =============================================================================

def posterior_predictive_check(
    simulator: Callable,
    posterior: Any,
    observed_stats: Union[np.ndarray, torch.Tensor],
    n_samples: int = 1000,
    stat_names: Optional[List[str]] = None,
    seed: Optional[int] = None,
    show_progress: bool = True
) -> PosteriorPredictiveResult:
    """
    Perform posterior predictive check.
    
    Samples parameters from posterior, simulates data, and compares
    predicted statistics to observed.
    
    Args:
        simulator: Callable(theta) -> summary_stats
        posterior: Posterior (from trained SBI)
        observed_stats: Observed summary statistics
        n_samples: Number of posterior samples to use
        stat_names: Names of statistics
        seed: Random seed
        show_progress: Print progress
    
    Returns:
        PosteriorPredictiveResult with observed vs predicted statistics
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Convert observed_stats
    if isinstance(observed_stats, torch.Tensor):
        observed_stats = observed_stats.numpy()
    observed_stats = np.atleast_1d(observed_stats)
    
    n_stats = len(observed_stats)
    
    if stat_names is None:
        stat_names = [f'stat_{i}' for i in range(n_stats)]
    
    # Sample from posterior
    x_tensor = torch.tensor(observed_stats, dtype=torch.float32)
    posterior_samples = posterior.sample((n_samples,), x=x_tensor)
    if isinstance(posterior_samples, torch.Tensor):
        posterior_samples = posterior_samples.numpy()
    
    # Simulate from each posterior sample
    predicted_stats = np.zeros((n_samples, n_stats))
    
    for i in range(n_samples):
        if show_progress and (i + 1) % 200 == 0:
            print(f"PPC simulation {i + 1}/{n_samples}")
        
        try:
            stats = simulator(posterior_samples[i])
            if isinstance(stats, torch.Tensor):
                stats = stats.numpy()
            predicted_stats[i] = stats
        except Exception as e:
            warnings.warn(f"PPC simulation failed at sample {i}: {e}")
            predicted_stats[i] = np.nan
    
    # Remove failed simulations
    valid = ~np.isnan(predicted_stats[:, 0])
    if not valid.all():
        warnings.warn(f"Removed {(~valid).sum()} failed PPC simulations")
        predicted_stats = predicted_stats[valid]
    
    return PosteriorPredictiveResult(
        observed_stats=observed_stats,
        predicted_stats=predicted_stats,
        stat_names=stat_names
    )


# =============================================================================
# VISUALISATION: SBC
# =============================================================================

def plot_sbc_ranks(
    sbc_result: SBCResult,
    params: Optional[List[str]] = None,
    n_bins: int = 20,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot SBC rank histograms.
    
    Ranks should be approximately uniform if posterior is calibrated.
    
    Args:
        sbc_result: SBCResult from run_sbc
        params: Which parameters to plot (None = all)
        n_bins: Number of histogram bins
        figsize: Figure size
        title: Overall title
    
    Returns:
        Matplotlib figure
    """
    if params is None:
        params = sbc_result.param_names
    
    n_params = len(params)
    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Expected count per bin under uniformity
    expected = sbc_result.n_sbc / n_bins
    
    # 95% CI for uniform (approximate)
    ci_low = expected - 2 * np.sqrt(expected * (1 - 1/n_bins))
    ci_high = expected + 2 * np.sqrt(expected * (1 - 1/n_bins))
    
    for i, param in enumerate(params):
        ax = axes[i]
        param_idx = sbc_result.param_names.index(param)
        ranks = sbc_result.ranks[:, param_idx]
        
        # Histogram
        ax.hist(ranks, bins=n_bins, range=(0, sbc_result.n_posterior_samples),
                color='steelblue', edgecolor='white', alpha=0.7)
        
        # Expected line and CI band
        ax.axhline(expected, color='red', linestyle='--', linewidth=1.5, label='Expected')
        ax.axhspan(ci_low, ci_high, color='red', alpha=0.1, label='95% CI')
        
        # Labels
        ax.set_xlabel('Rank')
        ax.set_ylabel('Count')
        ax.set_title(param)
        
        # Add p-value annotation
        if sbc_result.uniformity_pvalues:
            pval = sbc_result.uniformity_pvalues.get(param, np.nan)
            ax.text(0.95, 0.95, f'p={pval:.3f}', transform=ax.transAxes,
                   ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    
    fig.tight_layout()
    return fig


def plot_sbc_ecdf(
    sbc_result: SBCResult,
    params: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """
    Plot SBC empirical CDF vs uniform.
    
    Alternative visualisation to histograms.
    
    Args:
        sbc_result: SBCResult from run_sbc
        params: Which parameters to plot
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if params is None:
        params = sbc_result.param_names
    
    n_params = len(params)
    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, param in enumerate(params):
        ax = axes[i]
        param_idx = sbc_result.param_names.index(param)
        ranks = sbc_result.ranks[:, param_idx]
        
        # Normalize ranks to [0, 1]
        normalized = ranks / sbc_result.n_posterior_samples
        sorted_ranks = np.sort(normalized)
        ecdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
        
        # Plot ECDF
        ax.plot(sorted_ranks, ecdf, 'b-', linewidth=2, label='Observed')
        
        # Plot uniform reference
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Uniform')
        
        # 95% confidence band (Kolmogorov-Smirnov)
        n = len(sorted_ranks)
        ks_crit = 1.36 / np.sqrt(n)  # ~95% level
        ax.fill_between([0, 1], [0 - ks_crit, 1 - ks_crit], [0 + ks_crit, 1 + ks_crit],
                       color='red', alpha=0.1)
        
        ax.set_xlabel('Normalized Rank')
        ax.set_ylabel('ECDF')
        ax.set_title(param)
        ax.legend(loc='lower right', fontsize=8)
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.tight_layout()
    return fig


# =============================================================================
# VISUALISATION: PARAMETER RECOVERY
# =============================================================================

def plot_recovery_scatter(
    recovery_result: RecoveryResult,
    params: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show_ci: bool = True,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot parameter recovery scatter plots (true vs estimated).
    
    Args:
        recovery_result: RecoveryResult from parameter_recovery
        params: Which parameters to plot
        figsize: Figure size
        show_ci: Show credible intervals as error bars
        title: Overall title
    
    Returns:
        Matplotlib figure
    """
    if params is None:
        params = recovery_result.param_names
    
    n_params = len(params)
    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, param in enumerate(params):
        ax = axes[i]
        param_idx = recovery_result.param_names.index(param)
        
        true = recovery_result.theta_true[:, param_idx]
        est = recovery_result.theta_estimated[:, param_idx]
        lower = recovery_result.theta_lower[:, param_idx]
        upper = recovery_result.theta_upper[:, param_idx]
        
        # Error bars (CI)
        if show_ci:
            yerr = np.array([est - lower, upper - est])
            ax.errorbar(true, est, yerr=yerr, fmt='o', markersize=4,
                       alpha=0.5, capsize=2, color='steelblue')
        else:
            ax.scatter(true, est, alpha=0.5, s=20, color='steelblue')
        
        # Identity line
        lims = [min(true.min(), est.min()), max(true.max(), est.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'r--', linewidth=1.5, label='Identity')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        # Labels
        ax.set_xlabel('True')
        ax.set_ylabel('Estimated')
        ax.set_title(param)
        
        # Add metrics
        corr = recovery_result.correlations[param]
        coverage = recovery_result.coverages[param]
        ax.text(0.05, 0.95, f'r={corr:.3f}\ncov={coverage:.1%}',
               transform=ax.transAxes, va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    
    fig.tight_layout()
    return fig


def plot_recovery_bias(
    recovery_result: RecoveryResult,
    params: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """
    Plot bias as function of true parameter value.
    
    Args:
        recovery_result: RecoveryResult
        params: Which parameters to plot
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if params is None:
        params = recovery_result.param_names
    
    n_params = len(params)
    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, param in enumerate(params):
        ax = axes[i]
        param_idx = recovery_result.param_names.index(param)
        
        true = recovery_result.theta_true[:, param_idx]
        est = recovery_result.theta_estimated[:, param_idx]
        bias = est - true
        
        ax.scatter(true, bias, alpha=0.5, s=20, color='steelblue')
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
        
        ax.set_xlabel('True')
        ax.set_ylabel('Bias (Est - True)')
        ax.set_title(param)
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.tight_layout()
    return fig


# =============================================================================
# VISUALISATION: POSTERIOR PREDICTIVE
# =============================================================================

def plot_posterior_predictive(
    ppc_result: PosteriorPredictiveResult,
    stats: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot posterior predictive distributions vs observed.
    
    Args:
        ppc_result: PosteriorPredictiveResult
        stats: Which statistics to plot
        figsize: Figure size
        title: Overall title
    
    Returns:
        Matplotlib figure
    """
    if stats is None:
        stats = ppc_result.stat_names
    
    n_stats = len(stats)
    n_cols = min(4, n_stats)
    n_rows = int(np.ceil(n_stats / n_cols))
    
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, stat in enumerate(stats):
        ax = axes[i]
        stat_idx = ppc_result.stat_names.index(stat)
        
        observed = ppc_result.observed_stats[stat_idx]
        predicted = ppc_result.predicted_stats[:, stat_idx]
        
        # Histogram of predictions
        ax.hist(predicted, bins=30, color='steelblue', alpha=0.7,
               edgecolor='white', density=True)
        
        # Observed value
        ax.axvline(observed, color='red', linewidth=2, label='Observed')
        
        # Labels
        ax.set_xlabel(stat)
        ax.set_ylabel('Density')
        ax.set_title(stat)
        
        # Add p-value
        pval = ppc_result.ppc_pvalues.get(stat, np.nan)
        ax.text(0.95, 0.95, f'p={pval:.3f}', transform=ax.transAxes,
               ha='right', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    
    fig.tight_layout()
    return fig


# =============================================================================
# VISUALISATION: PSYCHOMETRIC CURVES (MODEL-SPECIFIC)
# =============================================================================

def plot_posterior_predictive_psychometric(
    simulator: Callable,
    posterior: Any,
    observed_choices: np.ndarray,
    stimuli: np.ndarray,
    observed_stats: Union[np.ndarray, torch.Tensor],
    n_samples: int = 100,
    n_bins: int = 8,
    figsize: Tuple[float, float] = (8, 6),
    seed: Optional[int] = None
) -> plt.Figure:
    """
    Plot posterior predictive psychometric curves.
    
    Shows ensemble of psychometric curves from posterior samples
    compared to observed data.
    
    Args:
        simulator: Must have a way to return choices (not just stats)
        posterior: Trained posterior
        observed_choices: Observed choice data
        stimuli: Stimulus values
        observed_stats: Summary stats for conditioning
        n_samples: Number of posterior samples
        n_bins: Number of bins for psychometric curve
        figsize: Figure size
        seed: Random seed
    
    Returns:
        Matplotlib figure
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Convert observed_stats
    if isinstance(observed_stats, np.ndarray):
        x_tensor = torch.tensor(observed_stats, dtype=torch.float32)
    else:
        x_tensor = observed_stats
    
    # Sample from posterior
    posterior_samples = posterior.sample((n_samples,), x=x_tensor)
    if isinstance(posterior_samples, torch.Tensor):
        posterior_samples = posterior_samples.numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Bin stimuli
    bin_edges = np.linspace(stimuli.min(), stimuli.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Observed psychometric curve
    obs_prop = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (stimuli >= bin_edges[b]) & (stimuli < bin_edges[b + 1])
        if mask.sum() > 0:
            obs_prop[b] = np.mean(observed_choices[mask])
    
    ax.scatter(bin_centers, obs_prop, s=100, c='red', zorder=10, 
               label='Observed', edgecolors='darkred')
    
    # Note: This requires the simulator to have a method to return choices
    # not just summary stats. This is a placeholder showing the structure.
    ax.text(0.5, 0.02, 
            'Note: Requires simulator with return_choices option\n'
            'to generate predictive psychometric curves',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('P(choose B)')
    ax.set_title('Posterior Predictive Psychometric Curves')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    
    fig.tight_layout()
    return fig


# =============================================================================
# COVERAGE DIAGNOSTICS
# =============================================================================

def compute_coverage(
    recovery_result: RecoveryResult,
    levels: List[float] = [0.50, 0.80, 0.90, 0.95]
) -> Dict[str, Dict[float, float]]:
    """
    Compute empirical coverage at multiple credible levels.
    
    Note: This requires re-running recovery with different CI levels,
    or using stored posterior samples.
    
    Args:
        recovery_result: RecoveryResult (uses stored CI level)
        levels: Coverage levels to report (uses only stored level currently)
    
    Returns:
        Dict mapping param names to coverage at each level
    """
    # For now, just return the single computed coverage
    return {
        param: {recovery_result.ci_level: cov}
        for param, cov in recovery_result.coverages.items()
    }


def plot_coverage(
    coverages: Dict[str, Dict[float, float]],
    figsize: Tuple[float, float] = (8, 5)
) -> plt.Figure:
    """
    Plot coverage calibration.
    
    Args:
        coverages: Dict from compute_coverage
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    params = list(coverages.keys())
    levels = list(next(iter(coverages.values())).keys())
    
    x = np.arange(len(params))
    width = 0.8 / len(levels)
    
    for i, level in enumerate(levels):
        empirical = [coverages[p][level] for p in params]
        offset = (i - len(levels)/2 + 0.5) * width
        bars = ax.bar(x + offset, empirical, width, label=f'{level:.0%} CI')
        
        # Add expected line
        ax.axhline(level, color='red', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title('Coverage Calibration')
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    fig.tight_layout()
    return fig


# =============================================================================
# CONVENIENCE: RUN ALL DIAGNOSTICS
# =============================================================================

def run_all_diagnostics(
    simulator: Callable,
    prior: Any,
    posterior: Any,
    observed_stats: Union[np.ndarray, torch.Tensor],
    n_sbc: int = 300,
    n_recovery: int = 100,
    n_ppc: int = 500,
    param_names: Optional[List[str]] = None,
    stat_names: Optional[List[str]] = None,
    seed: Optional[int] = None,
    show_progress: bool = True,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete diagnostic suite.
    
    Args:
        simulator: Simulator callable
        prior: Prior distribution
        posterior: Trained posterior
        observed_stats: Observed summary statistics
        n_sbc: Number of SBC iterations
        n_recovery: Number of recovery tests
        n_ppc: Number of PPC samples
        param_names: Parameter names
        stat_names: Statistic names
        seed: Random seed
        show_progress: Print progress
        save_dir: If provided, save figures to this directory
    
    Returns:
        Dict with 'sbc', 'recovery', 'ppc' results and figures
    """
    results = {}
    
    # SBC
    print("\n" + "=" * 60)
    print("Running Simulation-Based Calibration...")
    print("=" * 60)
    sbc = run_sbc(simulator, prior, posterior, n_sbc=n_sbc,
                  param_names=param_names, seed=seed, show_progress=show_progress)
    print(sbc.summary())
    results['sbc'] = sbc
    results['fig_sbc_ranks'] = plot_sbc_ranks(sbc)
    results['fig_sbc_ecdf'] = plot_sbc_ecdf(sbc)
    
    # Parameter recovery
    print("\n" + "=" * 60)
    print("Running Parameter Recovery...")
    print("=" * 60)
    recovery = parameter_recovery(simulator, prior, posterior, n_tests=n_recovery,
                                  param_names=param_names, seed=seed, 
                                  show_progress=show_progress)
    print(recovery.summary())
    results['recovery'] = recovery
    results['fig_recovery'] = plot_recovery_scatter(recovery)
    results['fig_recovery_bias'] = plot_recovery_bias(recovery)
    
    # Posterior predictive check
    print("\n" + "=" * 60)
    print("Running Posterior Predictive Check...")
    print("=" * 60)
    ppc = posterior_predictive_check(simulator, posterior, observed_stats,
                                     n_samples=n_ppc, stat_names=stat_names,
                                     seed=seed, show_progress=show_progress)
    print(ppc.summary())
    results['ppc'] = ppc
    results['fig_ppc'] = plot_posterior_predictive(ppc)
    
    # Save figures if requested
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in results.items():
            if name.startswith('fig_'):
                fig.savefig(os.path.join(save_dir, f'{name}.png'), dpi=150, bbox_inches='tight')
                print(f"Saved {name}.png")
    
    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Result containers
    'SBCResult',
    'RecoveryResult',
    'PosteriorPredictiveResult',
    # Main functions
    'run_sbc',
    'parameter_recovery',
    'posterior_predictive_check',
    # Visualisation
    'plot_sbc_ranks',
    'plot_sbc_ecdf',
    'plot_recovery_scatter',
    'plot_recovery_bias',
    'plot_posterior_predictive',
    'plot_posterior_predictive_psychometric',
    'plot_coverage',
    # Coverage
    'compute_coverage',
    # Convenience
    'run_all_diagnostics',
]
