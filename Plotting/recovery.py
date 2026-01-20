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
        Row 2: Psychometric parameter recovery (Î¼, Ïƒ) + optionally (Î»_low, Î»_high)
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
    
    # --- Row 2: Psychometric parameter recovery (Î¼, Ïƒ, and optionally Î») ---
    psych_params = ['mu', 'sigma']
    if show_lapse:
        psych_params.extend(['lapse_low', 'lapse_high'])
    
    psych_labels = {
        'mu': ('PSE (Î¼)', 'PSE error (fitted - true)'),
        'sigma': ('Slope (Ïƒ)', 'Slope error (fitted - true)'),
        'lapse_low': ('Lapse low (Î»_lo)', 'Î»_lo error (fitted - true)'),
        'lapse_high': ('Lapse high (Î»_hi)', 'Î»_hi error (fitted - true)')
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


# =============================================================================
# MIXED AGENT RECOVERY PLOTTING
# =============================================================================

def plot_mixed_agent_recovery(
    results: Dict, 
    figsize: Tuple[int, int] = (16, 12),
    show_lapse: bool = False
) -> plt.Figure:
    """
    Plot parameter and psychometric recovery as a function of α (BE weight).
    
    Shows how well BE parameters are recovered when fitting BE model to MixedAgent
    data with varying α values. Key diagnostic for detecting BE vs heuristic behaviour.
    
    Layout:
        Row 1: BE model parameter recovery (sigma_percep, A_repulsion, mu_learning, mu_relax)
        Row 2: Psychometric metrics (accuracy, R², σ, NLL)
        Row 3 (optional): PSE and lapse parameters
    
    Args:
        results: Output from mixed_agent_recovery_analysis()
        figsize: Figure size
        show_lapse: Whether to show lapse parameter plots (adds row)
    
    Returns:
        Matplotlib figure
    """
    alpha_values = results['config']['alpha_values']
    true_be_params = results['config']['true_be_params']
    param_names = list(true_be_params.keys())
    
    n_rows = 3 if show_lapse else 2
    n_cols = 4
    fig = plt.figure(figsize=figsize)
    
    # --- Row 1: BE Model Parameter recovery (4 params) ---
    for i, name in enumerate(param_names):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        
        means = []
        stds = []
        true_val = true_be_params[name]
        
        for alpha in alpha_values:
            data = results['param_recovery'][alpha][name]
            means.append(data['fitted_mean'])
            stds.append(data['fitted_std'])
        
        means = np.array(means)
        stds = np.array(stds)
        
        # Plot
        ax.errorbar(alpha_values, means, yerr=stds, fmt='o-', capsize=3, 
                    color='C0', label='Fitted', linewidth=2, markersize=6)
        ax.axhline(true_val, color='red', linestyle='--', alpha=0.7, 
                   linewidth=2, label=f'True = {true_val:.3f}')
        
        ax.set_xlabel('α (BE weight)')
        ax.set_ylabel(name)
        ax.set_title(f'{name}')
        ax.set_xlim(-0.05, 1.05)
        
        if i == 0:
            ax.legend(loc='best', fontsize=8)
        
        # Add shading for regions
        ax.axvspan(-0.05, 0.3, alpha=0.1, color='red', label='_Heuristic-dominated')
        ax.axvspan(0.7, 1.05, alpha=0.1, color='green', label='_BE-dominated')
    
    # --- Row 2: Psychometric and fit metrics ---
    row2_start = n_cols + 1
    
    # Accuracy
    ax = fig.add_subplot(n_rows, n_cols, row2_start)
    acc_means = [results['psych_metrics'][a]['accuracy_mean'] for a in alpha_values]
    acc_stds = [results['psych_metrics'][a]['accuracy_std'] for a in alpha_values]
    ax.errorbar(alpha_values, acc_means, yerr=acc_stds, fmt='o-', capsize=3, 
                color='C1', linewidth=2, markersize=6)
    ax.set_xlabel('α (BE weight)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Task accuracy')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.4, 0.95)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvspan(-0.05, 0.3, alpha=0.1, color='red')
    ax.axvspan(0.7, 1.05, alpha=0.1, color='green')
    
    # R-squared
    ax = fig.add_subplot(n_rows, n_cols, row2_start + 1)
    r2_means = [results['psych_metrics'][a]['r_squared_mean'] for a in alpha_values]
    r2_stds = [results['psych_metrics'][a]['r_squared_std'] for a in alpha_values]
    ax.errorbar(alpha_values, r2_means, yerr=r2_stds, fmt='o-', capsize=3, 
                color='C2', linewidth=2, markersize=6)
    ax.set_xlabel('α (BE weight)')
    ax.set_ylabel('R²')
    ax.set_title('Psychometric R²\n(goodness of fit)')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color='green', linestyle='--', alpha=0.5, label='Good fit')
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Poor fit')
    ax.legend(loc='lower right', fontsize=8)
    ax.axvspan(-0.05, 0.3, alpha=0.1, color='red')
    ax.axvspan(0.7, 1.05, alpha=0.1, color='green')
    
    # Psychometric sigma (slope)
    ax = fig.add_subplot(n_rows, n_cols, row2_start + 2)
    sig_means = [results['psych_metrics'][a]['sigma_mean'] for a in alpha_values]
    sig_stds = [results['psych_metrics'][a]['sigma_std'] for a in alpha_values]
    ax.errorbar(alpha_values, sig_means, yerr=sig_stds, fmt='o-', capsize=3, 
                color='C3', linewidth=2, markersize=6)
    ax.set_xlabel('α (BE weight)')
    ax.set_ylabel('σ (psychometric)')
    ax.set_title('Psychometric slope\n(lower = steeper)')
    ax.set_xlim(-0.05, 1.05)
    ax.axvspan(-0.05, 0.3, alpha=0.1, color='red')
    ax.axvspan(0.7, 1.05, alpha=0.1, color='green')
    
    # NLL (fit quality)
    ax = fig.add_subplot(n_rows, n_cols, row2_start + 3)
    nll_means = [results['fit_quality'][a].get('train_nll_mean', np.nan) for a in alpha_values]
    nll_stds = [results['fit_quality'][a].get('train_nll_std', np.nan) for a in alpha_values]
    ax.errorbar(alpha_values, nll_means, yerr=nll_stds, fmt='o-', capsize=3, 
                color='C4', linewidth=2, markersize=6)
    
    # Test NLL if available
    test_nll_means = [results['fit_quality'][a].get('test_nll_mean', np.nan) for a in alpha_values]
    if not all(np.isnan(test_nll_means)):
        test_nll_stds = [results['fit_quality'][a].get('test_nll_std', np.nan) for a in alpha_values]
        ax.errorbar(alpha_values, test_nll_means, yerr=test_nll_stds, fmt='s--', 
                    capsize=3, color='C5', linewidth=2, markersize=6, label='Test')
        ax.legend(loc='best', fontsize=8)
    
    ax.set_xlabel('α (BE weight)')
    ax.set_ylabel('NLL per trial')
    ax.set_title('BE model fit quality\n(lower = better)')
    ax.set_xlim(-0.05, 1.05)
    ax.axvspan(-0.05, 0.3, alpha=0.1, color='red')
    ax.axvspan(0.7, 1.05, alpha=0.1, color='green')
    
    # --- Row 3 (optional): PSE and lapse ---
    if show_lapse:
        row3_start = 2 * n_cols + 1
        
        # PSE (mu)
        ax = fig.add_subplot(n_rows, n_cols, row3_start)
        mu_means = [results['psych_metrics'][a]['mu_mean'] for a in alpha_values]
        mu_stds = [results['psych_metrics'][a]['mu_std'] for a in alpha_values]
        ax.errorbar(alpha_values, mu_means, yerr=mu_stds, fmt='o-', capsize=3, 
                    color='C6', linewidth=2, markersize=6)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('α (BE weight)')
        ax.set_ylabel('μ (PSE)')
        ax.set_title('PSE (point of subj. equality)')
        ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    
    # Add overall title
    agent_burn_in = results['config']['agent_burn_in']
    fitter_burn_in = results['config']['fitter_burn_in']
    fig.suptitle(f'BE recovery from MixedAgent data\n'
                 f'Agent burn-in={agent_burn_in}, Fitter burn-in={fitter_burn_in}', 
                 y=1.02, fontsize=12)
    
    return fig


def plot_mixed_agent_param_distributions(
    results: Dict, 
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot distributions of fitted BE parameters for each α condition.
    
    Shows boxplots/violin plots of fitted parameter values across replicates.
    Useful for visualising parameter uncertainty as α varies.
    
    Args:
        results: Output from mixed_agent_recovery_analysis()
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    alpha_values = results['config']['alpha_values']
    true_be_params = results['config']['true_be_params']
    param_names = list(true_be_params.keys())
    
    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params, figsize=figsize)
    
    if n_params == 1:
        axes = [axes]
    
    # Colour gradient from red (low α) to green (high α)
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(alpha_values)))
    
    for ax, name in zip(axes, param_names):
        data = []
        positions = []
        
        for i, alpha in enumerate(alpha_values):
            fitted = results['param_recovery'][alpha][name]['fitted']
            data.append(fitted[~np.isnan(fitted)])
            positions.append(i)
        
        bp = ax.boxplot(data, positions=positions, widths=0.6, 
                        patch_artist=True)
        
        # Colour boxes by α value
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # True value line
        ax.axhline(true_be_params[name], color='blue', linestyle='--', 
                   linewidth=2, label=f'True = {true_be_params[name]:.3f}')
        
        ax.set_xticks(positions)
        ax.set_xticklabels([f'{a:.1f}' for a in alpha_values])
        ax.set_xlabel('α (BE weight)')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    fig.suptitle('Fitted BE parameter distributions by α\n'
                 '(Red = heuristic-dominated, Green = BE-dominated)', y=1.02)
    
    return fig


def plot_mixed_agent_bias_heatmap(
    results: Dict,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot heatmap of parameter bias across α values.
    
    Each cell shows the bias (fitted - true) for a parameter at a given α.
    Useful for identifying which parameters are most affected by heuristic contamination.
    
    Args:
        results: Output from mixed_agent_recovery_analysis()
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    alpha_values = results['config']['alpha_values']
    true_be_params = results['config']['true_be_params']
    param_names = list(true_be_params.keys())
    
    # Build bias matrix (rows = params, cols = α)
    bias_matrix = np.zeros((len(param_names), len(alpha_values)))
    
    for i, name in enumerate(param_names):
        for j, alpha in enumerate(alpha_values):
            bias_matrix[i, j] = results['param_recovery'][alpha][name]['mean_error']
    
    # Normalise by true parameter value for comparability
    true_vals = np.array([true_be_params[name] for name in param_names])
    # Avoid division by zero
    normalised_bias = bias_matrix / np.where(true_vals[:, None] != 0, true_vals[:, None], 1)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Absolute bias
    ax = axes[0]
    im = ax.imshow(bias_matrix, aspect='auto', cmap='RdBu_r', 
                   vmin=-np.max(np.abs(bias_matrix)), vmax=np.max(np.abs(bias_matrix)))
    ax.set_xticks(range(len(alpha_values)))
    ax.set_xticklabels([f'{a:.1f}' for a in alpha_values])
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names)
    ax.set_xlabel('α (BE weight)')
    ax.set_ylabel('Parameter')
    ax.set_title('Absolute bias (fitted - true)')
    plt.colorbar(im, ax=ax, label='Bias')
    
    # Add text annotations
    for i in range(len(param_names)):
        for j in range(len(alpha_values)):
            text = ax.text(j, i, f'{bias_matrix[i, j]:.3f}',
                          ha='center', va='center', fontsize=8,
                          color='white' if abs(bias_matrix[i, j]) > np.max(np.abs(bias_matrix))/2 else 'black')
    
    # Normalised bias (% of true value)
    ax = axes[1]
    im = ax.imshow(normalised_bias * 100, aspect='auto', cmap='RdBu_r',
                   vmin=-100, vmax=100)
    ax.set_xticks(range(len(alpha_values)))
    ax.set_xticklabels([f'{a:.1f}' for a in alpha_values])
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names)
    ax.set_xlabel('α (BE weight)')
    ax.set_ylabel('Parameter')
    ax.set_title('Normalised bias (% of true value)')
    plt.colorbar(im, ax=ax, label='Bias (%)')
    
    # Add text annotations
    for i in range(len(param_names)):
        for j in range(len(alpha_values)):
            text = ax.text(j, i, f'{normalised_bias[i, j]*100:.0f}%',
                          ha='center', va='center', fontsize=8,
                          color='white' if abs(normalised_bias[i, j]) > 0.5 else 'black')
    
    plt.tight_layout()
    fig.suptitle('Parameter bias across α values', y=1.02)
    
    return fig


def plot_mixed_agent_sweep(
    results: Dict,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot parameter sweep results showing recovery across sweep values and α.
    
    Args:
        results: Output from mixed_agent_parameter_sweep()
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    sweep_param = results['config']['sweep_param']
    sweep_values = results['config']['sweep_values']
    alpha_values = results['config']['alpha_values']
    param_names = ['sigma_percep', 'A_repulsion', 'mu_learning', 'mu_relax']
    
    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(alpha_values)))
    
    # Plot bias for each BE param
    for i, param in enumerate(param_names):
        if i >= len(axes) - 2:  # Leave room for summary plots
            break
        ax = axes[i]
        
        for j, alpha in enumerate(alpha_values):
            biases = []
            for sweep_val in sweep_values:
                sweep_result = results['sweep_results'][sweep_val]
                biases.append(sweep_result['param_recovery'][alpha][param]['mean_error'])
            
            ax.plot(sweep_values, biases, 'o-', color=colors[j], 
                   label=f'α={alpha:.1f}', linewidth=2, markersize=6)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(sweep_param)
        ax.set_ylabel(f'{param} bias')
        ax.set_title(f'{param} recovery')
        if i == 0:
            ax.legend(loc='best', fontsize=8)
    
    # Accuracy across sweep
    ax = axes[-2]
    for j, alpha in enumerate(alpha_values):
        accs = []
        for sweep_val in sweep_values:
            sweep_result = results['sweep_results'][sweep_val]
            accs.append(sweep_result['psych_metrics'][alpha]['accuracy_mean'])
        ax.plot(sweep_values, accs, 'o-', color=colors[j], 
               label=f'α={alpha:.1f}', linewidth=2, markersize=6)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel('Accuracy')
    ax.set_title('Task accuracy')
    
    # R² across sweep
    ax = axes[-1]
    for j, alpha in enumerate(alpha_values):
        r2s = []
        for sweep_val in sweep_values:
            sweep_result = results['sweep_results'][sweep_val]
            r2s.append(sweep_result['psych_metrics'][alpha]['r_squared_mean'])
        ax.plot(sweep_values, r2s, 'o-', color=colors[j], 
               label=f'α={alpha:.1f}', linewidth=2, markersize=6)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel('R²')
    ax.set_title('Psychometric R²')
    ax.axhline(0.9, color='green', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    fig.suptitle(f'Parameter sweep: {sweep_param}', y=1.02, fontsize=14)
    
    return fig


# =============================================================================
# BE PARAMETER EFFECT PLOTTING
# =============================================================================

def plot_be_param_behaviour_sweep(
    results: Dict,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot effects of BE parameter on behaviour.
    
    Shows accuracy, R², psychometric sigma, and statistical tests vs chance.
    
    Args:
        results: Output from be_param_behaviour_sweep()
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    sweep_param = results['config']['sweep_param']
    sweep_values = results['config']['sweep_values']
    burn_in_values = results['config']['burn_in_values']
    
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(burn_in_values)))
    
    # Accuracy
    ax = axes[0]
    for i, burn_in in enumerate(burn_in_values):
        means = [results['behaviour'][sv][burn_in]['accuracy_mean'] 
                for sv in sweep_values]
        stds = [results['behaviour'][sv][burn_in]['accuracy_std'] 
               for sv in sweep_values]
        ax.errorbar(sweep_values, means, yerr=stds, fmt='o-', 
                   color=colors[i], label=f'burn_in={burn_in}', 
                   capsize=3, linewidth=2, markersize=6)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel(sweep_param)
    ax.set_ylabel('Accuracy')
    ax.set_title('Task accuracy')
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0)
    
    # R²
    ax = axes[1]
    for i, burn_in in enumerate(burn_in_values):
        means = [results['behaviour'][sv][burn_in]['r_squared_mean'] 
                for sv in sweep_values]
        stds = [results['behaviour'][sv][burn_in]['r_squared_std'] 
               for sv in sweep_values]
        ax.errorbar(sweep_values, means, yerr=stds, fmt='o-', 
                   color=colors[i], capsize=3, linewidth=2, markersize=6)
    ax.axhline(0.9, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel('R²')
    ax.set_title('Psychometric R²')
    ax.set_ylim(0, 1.05)
    
    # Psychometric sigma
    ax = axes[2]
    for i, burn_in in enumerate(burn_in_values):
        means = [results['behaviour'][sv][burn_in]['sigma_mean'] 
                for sv in sweep_values]
        stds = [results['behaviour'][sv][burn_in]['sigma_std'] 
               for sv in sweep_values]
        ax.errorbar(sweep_values, means, yerr=stds, fmt='o-', 
                   color=colors[i], capsize=3, linewidth=2, markersize=6)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel('σ (psychometric)')
    ax.set_title('Psychometric slope\n(lower = steeper)')
    
    # PSE (mu)
    ax = axes[3]
    for i, burn_in in enumerate(burn_in_values):
        means = [results['behaviour'][sv][burn_in]['mu_mean'] 
                for sv in sweep_values]
        stds = [results['behaviour'][sv][burn_in]['mu_std'] 
               for sv in sweep_values]
        ax.errorbar(sweep_values, means, yerr=stds, fmt='o-', 
                   color=colors[i], capsize=3, linewidth=2, markersize=6)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel('μ (PSE)')
    ax.set_title('Point of subjective equality')
    
    # P-value vs chance (accuracy)
    ax = axes[4]
    for i, burn_in in enumerate(burn_in_values):
        p_vals = [results['behaviour'][sv][burn_in]['accuracy_vs_chance_p'] 
                 for sv in sweep_values]
        # Convert to -log10(p) for better visualisation
        neg_log_p = [-np.log10(p) if p > 0 else 10 for p in p_vals]
        ax.plot(sweep_values, neg_log_p, 'o-', color=colors[i], 
               linewidth=2, markersize=6)
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', 
              alpha=0.5, label='p=0.05')
    ax.axhline(-np.log10(0.001), color='orange', linestyle='--', 
              alpha=0.5, label='p=0.001')
    ax.set_xlabel(sweep_param)
    ax.set_ylabel('-log₁₀(p)')
    ax.set_title('Significance vs chance\n(higher = more significant)')
    ax.legend(fontsize=8)
    
    # Deviance explained
    ax = axes[5]
    for i, burn_in in enumerate(burn_in_values):
        means = [results['behaviour'][sv][burn_in]['deviance_explained_mean'] 
                for sv in sweep_values]
        stds = [results['behaviour'][sv][burn_in]['deviance_explained_std'] 
               for sv in sweep_values]
        ax.errorbar(sweep_values, means, yerr=stds, fmt='o-', 
                   color=colors[i], capsize=3, linewidth=2, markersize=6)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel('Deviance explained')
    ax.set_title('Deviance explained\n(model vs null)')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    fig.suptitle(f'BE parameter effect on behaviour: {sweep_param}', 
                y=1.02, fontsize=14)
    
    return fig


def plot_be_param_recovery_sweep(
    results: Dict,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Plot effects of true BE parameter value on recovery.
    
    Shows bias for each parameter as function of sweep value and burn_in.
    
    Args:
        results: Output from be_param_recovery_sweep()
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    sweep_param = results['config']['sweep_param']
    sweep_values = results['config']['sweep_values']
    burn_in_values = results['config']['burn_in_values']
    param_names = ['sigma_percep', 'A_repulsion', 'mu_learning', 'mu_relax']
    
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(burn_in_values)))
    
    # Plot bias for each parameter
    for i, param in enumerate(param_names):
        ax = axes[i]
        
        for j, burn_in in enumerate(burn_in_values):
            biases = [results['recovery'][sv][burn_in]['params'][param]['bias'] 
                     for sv in sweep_values]
            stds = [results['recovery'][sv][burn_in]['params'][param]['std'] 
                   for sv in sweep_values]
            ax.errorbar(sweep_values, biases, yerr=stds, fmt='o-', 
                       color=colors[j], label=f'burn_in={burn_in}',
                       capsize=3, linewidth=2, markersize=6)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(sweep_param)
        ax.set_ylabel(f'{param} bias')
        ax.set_title(f'{param} recovery bias')
        
        if i == 0:
            ax.legend(fontsize=8)
        
        # Highlight swept parameter
        if param == sweep_param:
            ax.set_facecolor('#f0f0f0')
    
    # NLL (fit quality)
    ax = axes[4]
    for j, burn_in in enumerate(burn_in_values):
        nlls = [np.nanmean(results['recovery'][sv][burn_in]['fit']['train_nll']) 
               for sv in sweep_values]
        ax.plot(sweep_values, nlls, 'o-', color=colors[j], 
               linewidth=2, markersize=6)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel('NLL per trial')
    ax.set_title('Fit quality (lower = better)')
    
    # Summary: absolute error averaged across params
    ax = axes[5]
    for j, burn_in in enumerate(burn_in_values):
        mean_abs_errors = []
        for sv in sweep_values:
            errors = [results['recovery'][sv][burn_in]['params'][p]['abs_error'] 
                     for p in param_names]
            mean_abs_errors.append(np.nanmean(errors))
        ax.plot(sweep_values, mean_abs_errors, 'o-', color=colors[j], 
               linewidth=2, markersize=6)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel('Mean |error|')
    ax.set_title('Average absolute error\n(across all params)')
    
    plt.tight_layout()
    fig.suptitle(f'Parameter recovery vs true {sweep_param} value', 
                y=1.02, fontsize=14)
    
    return fig


def plot_mixed_agent_be_param_sweep(
    results: Dict,
    metric: str = 'accuracy',
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot mixed agent BE param sweep results as heatmaps.
    
    Creates a grid of heatmaps: rows = burn_in, cols = metric type,
    each heatmap shows sweep_value × alpha.
    
    Args:
        results: Output from mixed_agent_be_param_sweep()
        metric: 'accuracy', 'r_squared', or param name for bias
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    sweep_param = results['config']['sweep_param']
    sweep_values = results['config']['sweep_values']
    alpha_values = results['config']['alpha_values']
    burn_in_values = results['config']['burn_in_values']
    
    n_burn_ins = len(burn_in_values)
    fig, axes = plt.subplots(1, n_burn_ins, figsize=figsize)
    
    if n_burn_ins == 1:
        axes = [axes]
    
    for i, burn_in in enumerate(burn_in_values):
        ax = axes[i]
        
        # Build matrix
        matrix = np.zeros((len(sweep_values), len(alpha_values)))
        for j, sv in enumerate(sweep_values):
            for k, alpha in enumerate(alpha_values):
                data = results['data'][sv][burn_in][alpha]
                
                if metric in ['accuracy', 'r_squared', 'sigma_psych']:
                    if metric == 'sigma_psych':
                        matrix[j, k] = data['behaviour']['sigma_mean']
                    else:
                        matrix[j, k] = data['behaviour'][f'{metric}_mean']
                else:
                    # Assume it's a parameter bias
                    matrix[j, k] = data['recovery'][metric]['bias']
        
        # Plot heatmap
        if metric in ['accuracy', 'r_squared']:
            vmin, vmax = 0, 1
            cmap = 'RdYlGn'
        elif 'bias' in metric or metric in ['sigma_percep', 'A_repulsion', 
                                             'mu_learning', 'mu_relax']:
            vmax = np.nanmax(np.abs(matrix))
            vmin, vmax = -vmax, vmax
            cmap = 'RdBu_r'
        else:
            vmin, vmax = None, None
            cmap = 'viridis'
        
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                      origin='lower')
        
        ax.set_xticks(range(len(alpha_values)))
        ax.set_xticklabels([f'{a:.1f}' for a in alpha_values])
        ax.set_yticks(range(len(sweep_values)))
        ax.set_yticklabels([f'{sv:.2f}' for sv in sweep_values])
        ax.set_xlabel('α (BE weight)')
        ax.set_ylabel(sweep_param)
        ax.set_title(f'burn_in = {burn_in}')
        
        plt.colorbar(im, ax=ax, label=metric)
    
    plt.tight_layout()
    fig.suptitle(f'{metric} across {sweep_param} × α', y=1.02, fontsize=14)
    
    return fig


# =============================================================================
# PSYCHOMETRIC CURVE PLOTTING
# =============================================================================

def plot_behaviour_sweep_psychometrics(
    results: Dict,
    burn_in: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
    show_data: bool = True,
    n_bins: int = 8
) -> plt.Figure:
    """
    Plot psychometric curves from behaviour sweep results.
    
    Shows how the psychometric curve changes as a BE parameter is swept.
    
    Args:
        results: Output from be_param_behaviour_sweep()
        burn_in: Which burn_in to plot (None = first one)
        figsize: Figure size (auto-calculated if None)
        show_data: Whether to show binned data points
        n_bins: Number of bins for data points
    
    Returns:
        Matplotlib figure
    """
    sweep_param = results['config']['sweep_param']
    sweep_values = results['config']['sweep_values']
    burn_in_values = results['config']['burn_in_values']
    x_eval = results.get('x_eval', np.linspace(-1, 1, 100))
    
    if burn_in is None:
        burn_in = burn_in_values[0]
    
    n_values = len(sweep_values)
    ncols = min(4, n_values)
    nrows = int(np.ceil(n_values / ncols))
    
    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_values))
    
    for i, sweep_val in enumerate(sweep_values):
        ax = axes[i]
        metrics = results['behaviour'][sweep_val][burn_in]
        
        # Plot mean psychometric curve
        if metrics.get('psych_curve_mean') is not None:
            curve_mean = metrics['psych_curve_mean']
            curve_std = metrics.get('psych_curve_std')
            
            ax.plot(x_eval, curve_mean, color=colors[i], linewidth=2)
            
            if curve_std is not None:
                ax.fill_between(x_eval, curve_mean - curve_std, curve_mean + curve_std,
                               color=colors[i], alpha=0.2)
        
        # Plot binned data from first replicate if requested
        if show_data and len(metrics['choices']) > 0:
            # Use first replicate for data points
            stim = metrics['stimuli'][0]
            choices = metrics['choices'][0]
            valid = ~np.isnan(choices)
            
            bin_edges = np.linspace(-1, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_indices = np.digitize(stim[valid], bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            props = []
            for b in range(n_bins):
                mask = bin_indices == b
                if np.sum(mask) > 0:
                    props.append(np.mean(choices[valid][mask]))
                else:
                    props.append(np.nan)
            
            ax.scatter(bin_centers, props, color=colors[i], s=30, alpha=0.7, 
                      edgecolors='white', linewidths=0.5)
        
        # Labels and formatting
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Stimulus')
        ax.set_ylabel('P(B)')
        
        # Title with key metrics
        acc = metrics['accuracy_mean']
        r2 = metrics['r_squared_mean']
        sig = metrics['sigma_mean']
        ax.set_title(f'{sweep_param}={sweep_val:.2f}\n'
                    f'Acc={acc:.2f}, R²={r2:.2f}, σ={sig:.2f}')
    
    # Hide unused axes
    for i in range(n_values, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    fig.suptitle(f'Psychometric curves across {sweep_param} (burn_in={burn_in})', 
                y=1.02, fontsize=12)
    
    return fig


def plot_behaviour_sweep_psychometrics_overlay(
    results: Dict,
    burn_in: Optional[int] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot all psychometric curves overlaid on a single plot.
    
    Useful for directly comparing how the curve shape changes.
    
    Args:
        results: Output from be_param_behaviour_sweep()
        burn_in: Which burn_in to plot (None = first one)
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    sweep_param = results['config']['sweep_param']
    sweep_values = results['config']['sweep_values']
    burn_in_values = results['config']['burn_in_values']
    x_eval = results.get('x_eval', np.linspace(-1, 1, 100))
    
    if burn_in is None:
        burn_in = burn_in_values[0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sweep_values)))
    
    for i, sweep_val in enumerate(sweep_values):
        metrics = results['behaviour'][sweep_val][burn_in]
        
        if metrics.get('psych_curve_mean') is not None:
            curve_mean = metrics['psych_curve_mean']
            curve_std = metrics.get('psych_curve_std')
            
            ax.plot(x_eval, curve_mean, color=colors[i], linewidth=2,
                   label=f'{sweep_param}={sweep_val:.2f}')
            
            if curve_std is not None:
                ax.fill_between(x_eval, curve_mean - curve_std, curve_mean + curve_std,
                               color=colors[i], alpha=0.15)
    
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('P(B)')
    ax.legend(loc='lower right')
    ax.set_title(f'Psychometric curves across {sweep_param} (burn_in={burn_in})')
    
    plt.tight_layout()
    
    return fig


# =============================================================================
# JOINT PARAMETER SWEEP PLOTTING
# =============================================================================

def plot_joint_sweep_heatmap(
    results: Dict,
    metric: str = 'accuracy',
    figsize: Tuple[int, int] = (8, 6),
    annotate: bool = True
) -> plt.Figure:
    """
    Plot 2D heatmap of metric across parameter grid.
    
    Args:
        results: Output from be_param_joint_sweep()
        metric: Which metric to plot ('accuracy', 'r_squared', 'sigma', 'mu')
        figsize: Figure size
        annotate: Whether to add text annotations
    
    Returns:
        Matplotlib figure
    """
    param1 = results['config']['param1']
    param2 = results['config']['param2']
    param1_values = results['config']['param1_values']
    param2_values = results['config']['param2_values']
    
    matrix = results['matrices'][metric]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose colormap based on metric
    if metric == 'accuracy':
        vmin, vmax = 0.5, 1.0
        cmap = 'RdYlGn'
    elif metric == 'r_squared':
        vmin, vmax = 0, 1
        cmap = 'RdYlGn'
    elif metric == 'mu':
        vmax = np.nanmax(np.abs(matrix))
        vmin, vmax = -vmax, vmax
        cmap = 'RdBu_r'
    else:
        vmin, vmax = None, None
        cmap = 'viridis'
    
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                  origin='lower')
    
    ax.set_xticks(range(len(param2_values)))
    ax.set_xticklabels([f'{v:.2f}' for v in param2_values])
    ax.set_yticks(range(len(param1_values)))
    ax.set_yticklabels([f'{v:.2f}' for v in param1_values])
    ax.set_xlabel(param2)
    ax.set_ylabel(param1)
    
    plt.colorbar(im, ax=ax, label=metric)
    
    # Add annotations
    if annotate:
        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = 'white' if (vmin is not None and 
                                            abs(val - (vmin + vmax)/2) > (vmax - vmin)/3) else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=8, color=text_color)
    
    ax.set_title(f'{metric} across {param1} × {param2}')
    plt.tight_layout()
    
    return fig


def plot_joint_sweep_all_metrics(
    results: Dict,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot all metrics from joint sweep in a grid of heatmaps.
    
    Args:
        results: Output from be_param_joint_sweep()
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    param1 = results['config']['param1']
    param2 = results['config']['param2']
    param1_values = results['config']['param1_values']
    param2_values = results['config']['param2_values']
    
    metrics = ['accuracy', 'r_squared', 'sigma', 'mu', 'deviance_explained']
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    cmaps = {
        'accuracy': ('RdYlGn', 0.5, 1.0),
        'r_squared': ('RdYlGn', 0, 1),
        'sigma': ('viridis_r', None, None),
        'mu': ('RdBu_r', None, None),
        'deviance_explained': ('RdYlGn', 0, 1)
    }
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        matrix = results['matrices'][metric]
        
        cmap, vmin, vmax = cmaps.get(metric, ('viridis', None, None))
        
        if vmin is None and metric == 'mu':
            vmax = np.nanmax(np.abs(matrix))
            vmin = -vmax
        
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                      origin='lower')
        
        ax.set_xticks(range(len(param2_values)))
        ax.set_xticklabels([f'{v:.2f}' for v in param2_values], fontsize=8)
        ax.set_yticks(range(len(param1_values)))
        ax.set_yticklabels([f'{v:.2f}' for v in param1_values], fontsize=8)
        ax.set_xlabel(param2, fontsize=9)
        ax.set_ylabel(param1, fontsize=9)
        ax.set_title(metric)
        
        plt.colorbar(im, ax=ax)
    
    # Hide last subplot
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    fig.suptitle(f'Joint effects: {param1} × {param2}', y=1.02, fontsize=14)
    
    return fig


def plot_joint_sweep_psychometrics(
    results: Dict,
    n_samples: int = 9,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot psychometric curves from selected points in the joint sweep grid.
    
    Args:
        results: Output from be_param_joint_sweep()
        n_samples: Number of curves to show (will sample corners + center)
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    param1 = results['config']['param1']
    param2 = results['config']['param2']
    param1_values = results['config']['param1_values']
    param2_values = results['config']['param2_values']
    x_eval = results.get('x_eval', np.linspace(-1, 1, 100))
    
    # Select points: corners + middle
    n1, n2 = len(param1_values), len(param2_values)
    if n1 >= 3 and n2 >= 3:
        indices = [
            (0, 0), (0, n2//2), (0, n2-1),
            (n1//2, 0), (n1//2, n2//2), (n1//2, n2-1),
            (n1-1, 0), (n1-1, n2//2), (n1-1, n2-1)
        ]
    else:
        indices = [(i, j) for i in range(n1) for j in range(n2)]
    
    ncols = 3
    nrows = int(np.ceil(len(indices) / ncols))
    
    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for idx, (i, j) in enumerate(indices):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        p1_val = param1_values[i]
        p2_val = param2_values[j]
        
        metrics = results['grid'][(p1_val, p2_val)]
        
        if metrics.get('psych_curve_mean') is not None:
            ax.plot(x_eval, metrics['psych_curve_mean'], 'b-', linewidth=2)
        
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Stimulus')
        ax.set_ylabel('P(B)')
        
        acc = metrics['accuracy_mean']
        ax.set_title(f'{param1}={p1_val:.2f}, {param2}={p2_val:.2f}\nAcc={acc:.2f}')
    
    # Hide unused
    for idx in range(len(indices), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    fig.suptitle(f'Psychometric curves: {param1} × {param2}', y=1.02)
    
    return fig
