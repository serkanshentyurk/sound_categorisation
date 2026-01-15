"""
Parameter recovery plotting utilities.

Functions for visualising parameter recovery analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


def plot_burn_in_recovery(
    results: Dict, 
    figsize: Tuple[int, int] = (16, 12),
    show_lapse: bool = True
) -> plt.Figure:
    """
    Plot parameter and psychometric recovery as a function of burn-in.
    
    Shows how well parameters are recovered when fitter assumes naive (burn_in=0)
    but true model has varying experience levels.
    
    Layout:
        Row 1: BE model parameter recovery (sigma_percep, A_repulsion, mu_learning, mu_relax)
        Row 2: Psychometric parameter recovery (μ, σ) + optionally (λ_low, λ_high)
        Row 3: Curve error + Fit quality (NLL)
    
    Args:
        results: Output from burn_in_recovery_analysis()
        figsize: Figure size
        show_lapse: Whether to show lapse parameter recovery (default True)
    
    Returns:
        Matplotlib figure
    """
    burn_in_values = results['config']['burn_in_values']
    true_params = results['config']['true_params']
    param_names = list(true_params.keys())
    
    n_rows = 3 if show_lapse else 2
    fig = plt.figure(figsize=figsize)
    
    # --- Row 1: BE Model Parameter recovery (4 params) ---
    for i, name in enumerate(param_names):
        ax = fig.add_subplot(n_rows, 4, i + 1)
        
        means = []
        stds = []
        true_val = true_params[name]
        
        for burn_in in burn_in_values:
            fitted = results['param_recovery'][burn_in][name]['fitted']
            means.append(np.nanmean(fitted))
            stds.append(np.nanstd(fitted))
        
        means = np.array(means)
        stds = np.array(stds)
        
        # Plot
        ax.errorbar(burn_in_values, means, yerr=stds, fmt='o-', capsize=3, 
                    color='C0', label='Fitted')
        ax.axhline(true_val, color='k', linestyle='--', alpha=0.7, label='True')
        
        ax.set_xlabel('True burn-in')
        ax.set_ylabel(name)
        ax.set_title(f'BE: {name}')
        ax.set_xscale('symlog', linthresh=10)
        
        if i == 0:
            ax.legend(loc='best', fontsize=8)
    
    # --- Row 2: Psychometric parameter recovery (μ, σ, and optionally λ) ---
    psych_params = ['mu', 'sigma']
    if show_lapse:
        psych_params.extend(['lapse_low', 'lapse_high'])
    
    psych_labels = {
        'mu': ('PSE (μ)', 'PSE error (fitted - true)'),
        'sigma': ('Slope (σ)', 'Slope error (fitted - true)'),
        'lapse_low': ('Lapse low (λ_lo)', 'λ_lo error (fitted - true)'),
        'lapse_high': ('Lapse high (λ_hi)', 'λ_hi error (fitted - true)')
    }
    psych_colors = {'mu': 'C1', 'sigma': 'C2', 'lapse_low': 'C3', 'lapse_high': 'C4'}
    
    for i, param in enumerate(psych_params):
        ax = fig.add_subplot(n_rows, 4, 5 + i)
        means, stds = [], []
        
        for burn_in in burn_in_values:
            if param in results['psych_recovery'][burn_in]:
                errors = results['psych_recovery'][burn_in][param]['error']
                means.append(np.nanmean(errors))
                stds.append(np.nanstd(errors))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        ax.errorbar(burn_in_values, means, yerr=stds, fmt='o-', capsize=3, 
                    color=psych_colors[param])
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('True burn-in')
        ax.set_ylabel(psych_labels[param][1])
        ax.set_title(f'Psych: {psych_labels[param][0]}')
        ax.set_xscale('symlog', linthresh=10)
    
    # --- Row 3: Curve error + Fit quality ---
    row3_start = 9 if show_lapse else 7
    
    # Curve MAE
    ax = fig.add_subplot(n_rows, 4, row3_start)
    means, stds = [], []
    for burn_in in burn_in_values:
        mae = results['psych_recovery'][burn_in]['curve_mae']
        means.append(np.nanmean(mae))
        stds.append(np.nanstd(mae))
    ax.errorbar(burn_in_values, means, yerr=stds, fmt='o-', capsize=3, color='C5')
    ax.set_xlabel('True burn-in')
    ax.set_ylabel('Curve MAE')
    ax.set_title('Psych curve error\n(true vs fitted model)')
    ax.set_xscale('symlog', linthresh=10)
    
    # Fit quality (train NLL)
    ax = fig.add_subplot(n_rows, 4, row3_start + 1)
    train_means, train_stds = [], []
    test_means, test_stds = [], []
    for burn_in in burn_in_values:
        train_nll = results['fit_quality'][burn_in]['train_nll']
        train_means.append(np.nanmean(train_nll))
        train_stds.append(np.nanstd(train_nll))
        
        test_nll = results['fit_quality'][burn_in]['test_nll']
        if len(test_nll) > 0:
            test_means.append(np.nanmean(test_nll))
            test_stds.append(np.nanstd(test_nll))
    
    ax.errorbar(burn_in_values, train_means, yerr=train_stds, fmt='o-', 
                capsize=3, color='C6', label='Train')
    if test_means:
        ax.errorbar(burn_in_values, test_means, yerr=test_stds, fmt='s-', 
                    capsize=3, color='C7', label='Test')
    ax.set_xlabel('True burn-in')
    ax.set_ylabel('NLL per trial')
    ax.set_title('BE model fit quality\n(lower = better)')
    ax.set_xscale('symlog', linthresh=10)
    ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    
    # Add overall title
    fitter_burn_in = results['config']['fitter_burn_in']
    fig.suptitle(f'Recovery analysis: Fitter assumes burn-in = {fitter_burn_in} (naive)', 
                 y=1.02, fontsize=12)
    
    return fig
    ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    
    # Add overall title
    fitter_burn_in = results['config']['fitter_burn_in']
    fig.suptitle(f'Recovery analysis: Fitter assumes burn-in = {fitter_burn_in} (naive)', 
                 y=1.02, fontsize=12)
    
    return fig


def plot_burn_in_param_distributions(
    results: Dict, 
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot distributions of fitted parameters for each burn-in condition.
    
    Shows boxplots of fitted parameter values across replicates.
    
    Args:
        results: Output from burn_in_recovery_analysis()
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    burn_in_values = results['config']['burn_in_values']
    true_params = results['config']['true_params']
    param_names = list(true_params.keys())
    
    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params, figsize=figsize)
    
    if n_params == 1:
        axes = [axes]
    
    for ax, name in zip(axes, param_names):
        data = []
        positions = []
        
        for i, burn_in in enumerate(burn_in_values):
            fitted = results['param_recovery'][burn_in][name]['fitted']
            data.append(fitted[~np.isnan(fitted)])
            positions.append(i)
        
        bp = ax.boxplot(data, positions=positions, widths=0.6, 
                        patch_artist=True)
        
        # Colour boxes
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(burn_in_values)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # True value line
        ax.axhline(true_params[name], color='red', linestyle='--', 
                   linewidth=2, label=f'True = {true_params[name]:.3f}')
        
        ax.set_xticks(positions)
        ax.set_xticklabels([str(b) for b in burn_in_values])
        ax.set_xlabel('True burn-in')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    fig.suptitle('Parameter distributions across burn-in conditions', y=1.02)
    
    return fig


def plot_param_recovery_scatter(
    true_values: np.ndarray,
    fitted_values: np.ndarray,
    param_name: str,
    ax: Optional[plt.Axes] = None,
    color: str = 'C0',
    show_identity: bool = True,
    show_correlation: bool = True,
    title: Optional[str] = None
) -> plt.Axes:
    """
    Plot true vs fitted parameter scatter with identity line.
    
    Args:
        true_values: Array of true parameter values
        fitted_values: Array of fitted parameter values
        param_name: Parameter name for labels
        ax: Matplotlib axes
        color: Point colour
        show_identity: Whether to show identity line
        show_correlation: Whether to show correlation coefficient
        title: Axes title
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    # Remove NaNs
    valid = ~np.isnan(true_values) & ~np.isnan(fitted_values)
    true_valid = true_values[valid]
    fitted_valid = fitted_values[valid]
    
    ax.scatter(true_valid, fitted_valid, alpha=0.6, color=color)
    
    if show_identity:
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Identity')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    
    if show_correlation and len(true_valid) > 2:
        corr = np.corrcoef(true_valid, fitted_valid)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=10, va='top')
    
    ax.set_xlabel(f'True {param_name}')
    ax.set_ylabel(f'Fitted {param_name}')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(param_name)
    
    ax.set_aspect('equal')
    
    return ax


def plot_param_recovery_grid(
    results: Dict,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot true vs fitted scatter for all parameters in a grid.
    
    Args:
        results: Output from parameter_recovery analysis
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    param_names = list(results['true'].keys())
    n_params = len(param_names)
    
    ncols = min(4, n_params)
    nrows = int(np.ceil(n_params / ncols))
    
    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.arange(n_params))
    
    for i, name in enumerate(param_names):
        ax = axes[i]
        plot_param_recovery_scatter(
            results['true'][name],
            results['fitted'][name],
            name,
            ax=ax,
            color=colors[i]
        )
    
    # Hide empty subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    return fig


def plot_recovery_summary(
    results: Dict,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot summary of parameter recovery: bias and correlation for each parameter.
    
    Args:
        results: Output from parameter_recovery analysis with 'true' and 'fitted' keys
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    param_names = list(results['true'].keys())
    
    biases = []
    correlations = []
    
    for name in param_names:
        true_vals = results['true'][name]
        fitted_vals = results['fitted'][name]
        
        valid = ~np.isnan(true_vals) & ~np.isnan(fitted_vals)
        
        bias = np.mean(fitted_vals[valid] - true_vals[valid])
        biases.append(bias)
        
        if np.sum(valid) > 2:
            corr = np.corrcoef(true_vals[valid], fitted_vals[valid])[0, 1]
        else:
            corr = np.nan
        correlations.append(corr)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bias
    ax = axes[0]
    x = np.arange(len(param_names))
    colors = ['C0' if b >= 0 else 'C3' for b in biases]
    ax.bar(x, biases, color=colors, alpha=0.7)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_ylabel('Mean bias (fitted - true)')
    ax.set_title('Parameter bias')
    
    # Correlation
    ax = axes[1]
    ax.bar(x, correlations, color='C2', alpha=0.7)
    ax.axhline(1, color='k', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_ylabel('Correlation (true vs fitted)')
    ax.set_title('Parameter recovery')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    return fig
