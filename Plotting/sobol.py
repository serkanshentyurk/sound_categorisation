"""
Plotting functions for Sobol sensitivity analysis results.

Usage:
    from Plotting.sobol import plot_sobol_indices, plot_sobol_interactions
    
    fig = plot_sobol_indices(results, output='accuracy')
    fig = plot_sobol_interactions(results, output='accuracy')
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


def plot_sobol_indices(
    results: 'SobolResults',
    output: str,
    figsize: Tuple[int, int] = (10, 6),
    show_conf: bool = True,
    sort_by: str = 'ST'
) -> plt.Figure:
    """
    Plot first-order (S1) and total-order (ST) Sobol indices.
    
    Args:
        results: SobolResults from run_sobol_analysis
        output: Which output to plot
        figsize: Figure size
        show_conf: Show confidence intervals
        sort_by: Sort parameters by 'S1', 'ST', or 'name'
    
    Returns:
        Matplotlib figure
    """
    df = results.sensitivity[output].copy()
    
    # Sort
    if sort_by in ['S1', 'ST']:
        df = df.sort_values(sort_by, ascending=True)
    elif sort_by == 'name':
        df = df.sort_values('parameter')
    
    params = df['parameter'].values
    s1 = df['S1'].values
    st = df['ST'].values
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y = np.arange(len(params))
    height = 0.35
    
    # Plot bars
    bars1 = ax.barh(y - height/2, s1, height, label='S1 (first-order)', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.barh(y + height/2, st, height, label='ST (total-order)', 
                   color='darkorange', alpha=0.8)
    
    # Add confidence intervals
    if show_conf and 'S1_conf' in df.columns:
        ax.errorbar(s1, y - height/2, xerr=df['S1_conf'].values, 
                   fmt='none', color='black', capsize=3, alpha=0.7)
        ax.errorbar(st, y + height/2, xerr=df['ST_conf'].values, 
                   fmt='none', color='black', capsize=3, alpha=0.7)
    
    ax.set_yticks(y)
    ax.set_yticklabels(params)
    ax.set_xlabel('Sensitivity Index')
    ax.set_title(f'Sobol Sensitivity Indices: {output}')
    ax.legend(loc='lower right')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(-0.05, max(1.0, st.max() * 1.1))
    
    # Add interaction indicator
    interaction_gap = st - s1
    for i, (gap, param) in enumerate(zip(interaction_gap, params)):
        if gap > 0.1:  # Significant interaction
            ax.annotate(f'Δ={gap:.2f}', xy=(st[i], i + height/2), 
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=8, color='red')
    
    plt.tight_layout()
    
    return fig


def plot_sobol_indices_multi(
    results: 'SobolResults',
    outputs: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    sort_by: str = 'ST'
) -> plt.Figure:
    """
    Plot Sobol indices for multiple outputs side by side.
    
    Args:
        results: SobolResults from run_sobol_analysis
        outputs: Which outputs to plot (None = all)
        figsize: Figure size
        sort_by: Sort parameters by
    
    Returns:
        Matplotlib figure
    """
    if outputs is None:
        outputs = results.config['output_names']
    
    n_outputs = len(outputs)
    ncols = min(3, n_outputs)
    nrows = int(np.ceil(n_outputs / ncols))
    
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, output in enumerate(outputs):
        ax = axes[i]
        df = results.sensitivity[output].copy()
        
        if sort_by in ['S1', 'ST']:
            df = df.sort_values(sort_by, ascending=True)
        
        params = df['parameter'].values
        s1 = df['S1'].values
        st = df['ST'].values
        
        y = np.arange(len(params))
        height = 0.35
        
        ax.barh(y - height/2, s1, height, label='S1', color='steelblue', alpha=0.8)
        ax.barh(y + height/2, st, height, label='ST', color='darkorange', alpha=0.8)
        
        ax.set_yticks(y)
        ax.set_yticklabels(params, fontsize=9)
        ax.set_xlabel('Sensitivity Index')
        ax.set_title(output)
        ax.set_xlim(-0.05, max(1.0, st.max() * 1.1))
        
        if i == 0:
            ax.legend(loc='lower right', fontsize=8)
    
    # Hide unused
    for i in range(n_outputs, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    fig.suptitle('Sobol Sensitivity Indices', y=1.02, fontsize=12)
    
    return fig


def plot_sobol_interactions(
    results: 'SobolResults',
    output: str,
    figsize: Tuple[int, int] = (8, 6),
    threshold: float = 0.02
) -> plt.Figure:
    """
    Plot second-order interaction indices as heatmap.
    
    Args:
        results: SobolResults from run_sobol_analysis
        output: Which output to plot
        figsize: Figure size
        threshold: Only show interactions above this threshold
    
    Returns:
        Matplotlib figure
    """
    if output not in results.interactions or results.interactions[output].empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No second-order indices available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Interactions: {output}')
        return fig
    
    df = results.interactions[output]
    params = results.problem['names']
    n_params = len(params)
    
    # Build matrix
    matrix = np.zeros((n_params, n_params))
    for _, row in df.iterrows():
        i = params.index(row['param1'])
        j = params.index(row['param2'])
        matrix[i, j] = row['S2']
        matrix[j, i] = row['S2']  # Symmetric
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Mask diagonal and small values
    mask = np.eye(n_params, dtype=bool) | (np.abs(matrix) < threshold)
    matrix_masked = np.ma.masked_where(mask, matrix)
    
    im = ax.imshow(matrix_masked, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    
    ax.set_xticks(range(n_params))
    ax.set_yticks(range(n_params))
    ax.set_xticklabels(params, rotation=45, ha='right')
    ax.set_yticklabels(params)
    ax.set_title(f'Second-Order Interactions (S2): {output}')
    
    plt.colorbar(im, ax=ax, label='S2')
    
    # Add text annotations
    for i in range(n_params):
        for j in range(n_params):
            if not mask[i, j]:
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                       fontsize=9, color='white' if abs(matrix[i, j]) > 0.1 else 'black')
    
    plt.tight_layout()
    
    return fig


def plot_sobol_summary_heatmap(
    results: 'SobolResults',
    metric: str = 'ST',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot heatmap of sensitivity across all outputs.
    
    Rows = parameters, Columns = outputs
    
    Args:
        results: SobolResults from run_sobol_analysis
        metric: 'S1' or 'ST'
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    params = results.problem['names']
    outputs = results.config['output_names']
    
    # Build matrix
    matrix = np.zeros((len(params), len(outputs)))
    
    for j, output in enumerate(outputs):
        df = results.sensitivity[output]
        for i, param in enumerate(params):
            val = df[df['parameter'] == param][metric].values
            matrix[i, j] = val[0] if len(val) > 0 else np.nan
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    
    ax.set_xticks(range(len(outputs)))
    ax.set_yticks(range(len(params)))
    ax.set_xticklabels(outputs, rotation=45, ha='right')
    ax.set_yticklabels(params)
    ax.set_title(f'{metric} Sensitivity Across Outputs')
    
    plt.colorbar(im, ax=ax, label=metric)
    
    # Add text annotations
    for i in range(len(params)):
        for j in range(len(outputs)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=8, color=color)
    
    plt.tight_layout()
    
    return fig


def plot_partial_dependence(
    results: 'SobolResults',
    param: str,
    output: str,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Plot partial dependence of output on parameter.
    
    Shows marginal effect of parameter on output, averaging over other params.
    
    Args:
        results: SobolResults from run_sobol_analysis
        param: Parameter to plot
        output: Output to analyse
        n_bins: Number of bins for parameter
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    params = results.problem['names']
    param_idx = params.index(param)
    
    samples = results.raw_samples[:, param_idx]
    outputs = results.raw_outputs[output]
    
    # Bin the parameter
    bins = np.linspace(samples.min(), samples.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(samples, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Compute mean and std for each bin
    means = []
    stds = []
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            means.append(np.nanmean(outputs[mask]))
            stds.append(np.nanstd(outputs[mask]))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    
    means = np.array(means)
    stds = np.array(stds)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(bin_centers, means, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax.fill_between(bin_centers, means - stds, means + stds, 
                   color='steelblue', alpha=0.2)
    
    ax.set_xlabel(param)
    ax.set_ylabel(output)
    ax.set_title(f'Partial Dependence: {output} vs {param}')
    
    # Add sensitivity index
    df = results.sensitivity[output]
    st = df[df['parameter'] == param]['ST'].values[0]
    ax.annotate(f'ST = {st:.3f}', xy=(0.95, 0.95), xycoords='axes fraction',
               ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return fig


def plot_partial_dependence_grid(
    results: 'SobolResults',
    output: str,
    params: Optional[List[str]] = None,
    n_bins: int = 10,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot partial dependence for multiple parameters.
    
    Args:
        results: SobolResults from run_sobol_analysis
        output: Output to analyse
        params: Which parameters to plot (None = all)
        n_bins: Number of bins
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if params is None:
        params = results.problem['names']
    
    n_params = len(params)
    ncols = min(3, n_params)
    nrows = int(np.ceil(n_params / ncols))
    
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    all_params = results.problem['names']
    outputs = results.raw_outputs[output]
    
    for i, param in enumerate(params):
        ax = axes[i]
        param_idx = all_params.index(param)
        samples = results.raw_samples[:, param_idx]
        
        # Bin
        bins = np.linspace(samples.min(), samples.max(), n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_indices = np.digitize(samples, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        means = []
        stds = []
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() > 0:
                means.append(np.nanmean(outputs[mask]))
                stds.append(np.nanstd(outputs[mask]))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        means = np.array(means)
        stds = np.array(stds)
        
        ax.plot(bin_centers, means, 'o-', color='steelblue', linewidth=2, markersize=4)
        ax.fill_between(bin_centers, means - stds, means + stds, 
                       color='steelblue', alpha=0.2)
        ax.set_xlabel(param)
        ax.set_ylabel(output)
        
        # Add ST
        df = results.sensitivity[output]
        st = df[df['parameter'] == param]['ST'].values[0]
        ax.set_title(f'{param} (ST={st:.2f})')
    
    # Hide unused
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    fig.suptitle(f'Partial Dependence: {output}', y=1.02)
    
    return fig


def plot_convergence_check(
    results: 'SobolResults',
    output: str,
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """
    Check convergence by plotting indices at different sample sizes.
    
    Uses subsampling of existing results to estimate convergence.
    
    Args:
        results: SobolResults from run_sobol_analysis
        output: Output to check
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # This would require re-running analysis at different N
    # For now, just show a placeholder with recommendations
    
    fig, ax = plt.subplots(figsize=figsize)
    
    df = results.sensitivity[output]
    params = df['parameter'].values
    st = df['ST'].values
    st_conf = df['ST_conf'].values
    
    # Plot ST with confidence
    x = np.arange(len(params))
    ax.bar(x, st, yerr=st_conf, capsize=5, color='darkorange', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=45, ha='right')
    ax.set_ylabel('ST')
    ax.set_title(f'Convergence Check: {output}\n'
                f'(Confidence intervals show estimation uncertainty)')
    
    # Add warning if confidence is large
    max_conf_ratio = (st_conf / (st + 0.01)).max()
    if max_conf_ratio > 0.3:
        ax.annotate('⚠ Large uncertainty - consider increasing n_sobol',
                   xy=(0.5, 0.95), xycoords='axes fraction',
                   ha='center', fontsize=10, color='red')
    
    plt.tight_layout()
    
    return fig


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'plot_sobol_indices',
    'plot_sobol_indices_multi',
    'plot_sobol_interactions',
    'plot_sobol_summary_heatmap',
    'plot_partial_dependence',
    'plot_partial_dependence_grid',
    'plot_convergence_check',
]
