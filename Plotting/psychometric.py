"""
Psychometric curve plotting utilities.

Core functions for visualising psychometric curves with binned data and fitted curves.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Tuple, List, Union

from Helpers.psychometry import fit_psychometric, compute_psychometric_gof


def plot_psychometric(
    stimuli: np.ndarray,
    choices: np.ndarray,
    ax: Optional[plt.Axes] = None,
    n_bins: int = 8,
    show_fit: bool = True,
    show_gof: bool = False,
    show_params: bool = True,
    show_lapse: bool = False,
    show_reference_lines: bool = True,
    n_bootstrap: int = 0,
    show_ci: bool = True,
    color: Optional[str] = None,
    label: Optional[str] = None,
    marker: str = 'o',
    markersize: int = 8,
    linewidth: int = 2,
    capsize: int = 3,
    x_fine: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    gof_position: str = 'upper left',
    seed: int = 42
) -> Tuple[plt.Axes, Dict]:
    """
    Plot a single psychometric curve with binned data and optional fitted curve.
    
    Args:
        stimuli: Array of stimulus values
        choices: Array of binary choices (0=A, 1=B)
        ax: Matplotlib axes (creates new if None)
        n_bins: Number of bins for data points
        show_fit: Whether to show fitted cumulative Gaussian
        show_gof: Whether to show goodness-of-fit metrics (Acc, R², RMSE)
        show_params: Whether to show fitted parameters (μ, σ)
        show_lapse: Whether to show lapse parameters (λ_low, λ_high)
        show_reference_lines: Whether to show reference lines at 0.5 and 0
        n_bootstrap: Number of bootstrap samples for CIs (0 = no bootstrap)
        show_ci: Whether to show CI band (requires n_bootstrap > 0)
        color: Colour for data points and curve
        label: Legend label for data points
        marker: Marker style
        markersize: Marker size
        linewidth: Line width for fitted curve
        capsize: Error bar cap size
        x_fine: Fine x values for fitted curve (default: linspace(-1, 1, 100))
        title: Axes title
        gof_position: Position for text box ('upper left', 'upper right', etc.)
        seed: Random seed for bootstrap
    
    Returns:
        ax: Matplotlib axes
        info: Dict with:
            'psych_params': Fitted psychometric parameters (including CIs if bootstrap)
            'gof': Goodness-of-fit metrics
            'bin_centers': Bin center values
            'prop_B': Proportion choosing B per bin
            'prop_B_se': Standard error per bin
    """
    stimuli = np.asarray(stimuli)
    choices = np.asarray(choices)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    
    if x_fine is None:
        x_fine = np.linspace(-1, 1, 100)
    
    if color is None:
        color = 'C0'
    
    # Remove NaNs
    valid = ~np.isnan(stimuli) & ~np.isnan(choices)
    stimuli_valid = stimuli[valid]
    choices_valid = choices[valid]
    
    # Compute binned proportions
    bin_edges = np.linspace(-1, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(stimuli_valid, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    prop_B = np.zeros(n_bins)
    prop_B_se = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    for b in range(n_bins):
        mask = bin_indices == b
        counts[b] = np.sum(mask)
        if counts[b] > 0:
            prop_B[b] = np.mean(choices_valid[mask])
            # Standard error of proportion
            prop_B_se[b] = np.sqrt(prop_B[b] * (1 - prop_B[b]) / counts[b])
    
    # Plot binned data
    ax.errorbar(bin_centers, prop_B, yerr=prop_B_se, fmt=marker,
                color=color, markersize=markersize, capsize=capsize, 
                label=label)
    
    # Fit psychometric curve (with optional bootstrap)
    psych_params = fit_psychometric(stimuli_valid, choices_valid, x_fine, 
                                     n_bootstrap=n_bootstrap, seed=seed)
    gof = compute_psychometric_gof(stimuli_valid, choices_valid, psych_params, n_bins)
    
    # Plot CI band first (so it's behind the line)
    if show_fit and show_ci and n_bootstrap > 0 and psych_params['success']:
        y_ci = psych_params.get('y_fit_ci', (None, None))
        if y_ci[0] is not None and y_ci[1] is not None:
            ax.fill_between(x_fine, y_ci[0], y_ci[1], color=color, alpha=0.2)
    
    # Plot fitted curve
    if show_fit and psych_params['success']:
        ax.plot(x_fine, psych_params['y_fit'], '-', color=color, 
                linewidth=linewidth)
    
    # Reference lines
    if show_reference_lines:
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    
    # Text box with metrics
    text_lines = []
    
    if show_gof:
        # Compute accuracy
        categories = (stimuli_valid > 0).astype(int)
        accuracy = np.mean(choices_valid == categories)
        text_lines.extend([
            f"Acc: {accuracy:.1%}",
            f"R²: {gof['r_squared']:.3f}",
            f"RMSE: {gof['rmse']:.3f}"
        ])
    
    if show_params and psych_params['success']:
        mu_str = f"μ: {psych_params['mu']:.3f}"
        sigma_str = f"σ: {psych_params['sigma']:.3f}"
        
        # Add CIs if available
        if n_bootstrap > 0 and 'mu_ci' in psych_params:
            mu_ci = psych_params['mu_ci']
            sigma_ci = psych_params['sigma_ci']
            mu_str = f"μ: {psych_params['mu']:.3f} [{mu_ci[0]:.3f}, {mu_ci[1]:.3f}]"
            sigma_str = f"σ: {psych_params['sigma']:.3f} [{sigma_ci[0]:.3f}, {sigma_ci[1]:.3f}]"
        
        text_lines.extend([mu_str, sigma_str])
    
    if show_lapse and psych_params['success']:
        lapse_low_str = f"λ_lo: {psych_params['lapse_low']:.3f}"
        lapse_high_str = f"λ_hi: {psych_params['lapse_high']:.3f}"
        
        # Add CIs if available
        if n_bootstrap > 0 and 'lapse_low_ci' in psych_params:
            ll_ci = psych_params['lapse_low_ci']
            lh_ci = psych_params['lapse_high_ci']
            lapse_low_str = f"λ_lo: {psych_params['lapse_low']:.3f} [{ll_ci[0]:.3f}, {ll_ci[1]:.3f}]"
            lapse_high_str = f"λ_hi: {psych_params['lapse_high']:.3f} [{lh_ci[0]:.3f}, {lh_ci[1]:.3f}]"
        
        text_lines.extend([lapse_low_str, lapse_high_str])
    
    if text_lines:
        # Position mapping
        pos_map = {
            'upper left': (0.05, 0.95, 'left', 'top'),
            'upper right': (0.95, 0.95, 'right', 'top'),
            'lower left': (0.05, 0.05, 'left', 'bottom'),
            'lower right': (0.95, 0.05, 'right', 'bottom')
        }
        x_pos, y_pos, ha, va = pos_map.get(gof_position, (0.05, 0.95, 'left', 'top'))
        
        ax.text(x_pos, y_pos, '\n'.join(text_lines), transform=ax.transAxes,
                fontsize=8, ha=ha, va=va, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Labels
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('P(choose B)')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    
    if title:
        ax.set_title(title)
    
    info = {
        'psych_params': psych_params,
        'gof': gof,
        'bin_centers': bin_centers,
        'prop_B': prop_B,
        'prop_B_se': prop_B_se,
        'counts': counts
    }
    
    return ax, info


def plot_psychometric_grid(
    stimuli_list: List[np.ndarray],
    choices_list: List[np.ndarray],
    labels: Optional[List[str]] = None,
    ncols: int = 4,
    figsize: Optional[Tuple[int, int]] = None,
    n_bins: int = 8,
    show_fit: bool = True,
    show_gof: bool = True,
    show_params: bool = False,
    show_lapse: bool = False,
    share_y: bool = True,
    colors: Optional[List[str]] = None,
    suptitle: Optional[str] = None,
    n_bootstrap: int = 0,
    show_ci: bool = True,
    seed: int = 42
) -> Tuple[plt.Figure, List[Dict]]:
    """
    Plot multiple psychometric curves in a grid layout.
    
    Args:
        stimuli_list: List of stimulus arrays (one per condition)
        choices_list: List of choice arrays (one per condition)
        labels: Labels for each condition (used as subplot titles)
        ncols: Number of columns in grid
        figsize: Figure size (auto-calculated if None)
        n_bins: Number of bins per plot
        show_fit: Whether to show fitted curves
        show_gof: Whether to show GOF metrics (Acc, R², RMSE)
        show_params: Whether to show μ, σ parameters
        show_lapse: Whether to show lapse parameters
        share_y: Whether to share y-axis across subplots
        colors: List of colours (auto-generated if None)
        suptitle: Overall figure title
        n_bootstrap: Number of bootstrap samples for CIs
        show_ci: Whether to show CI bands
        seed: Random seed for bootstrap
    
    Returns:
        fig: Matplotlib figure
        infos: List of info dicts from each plot_psychometric call
    """
    n_conditions = len(stimuli_list)
    
    if labels is None:
        labels = [f'Condition {i+1}' for i in range(n_conditions)]
    
    nrows = int(np.ceil(n_conditions / ncols))
    
    if figsize is None:
        figsize = (3 * ncols, 3 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                              squeeze=False, sharey=share_y)
    axes = axes.flatten()
    
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_conditions))
    
    infos = []
    
    for i, (stim, choices, label) in enumerate(zip(stimuli_list, choices_list, labels)):
        ax = axes[i]
        color = colors[i] if isinstance(colors[i], str) else colors[i]
        
        _, info = plot_psychometric(
            stim, choices, ax=ax,
            n_bins=n_bins,
            show_fit=show_fit,
            show_gof=show_gof,
            show_params=show_params,
            show_lapse=show_lapse,
            color=color,
            title=label,
            n_bootstrap=n_bootstrap,
            show_ci=show_ci,
            seed=seed + i
        )
        infos.append(info)
        
        # Only show y-label on leftmost plots
        if i % ncols != 0:
            ax.set_ylabel('')
    
    # Hide empty subplots
    for i in range(n_conditions, len(axes)):
        axes[i].set_visible(False)
    
    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    
    plt.tight_layout()
    
    return fig, infos


def plot_psychometric_comparison(
    stimuli_list: List[np.ndarray],
    choices_list: List[np.ndarray],
    labels: List[str],
    ax: Optional[plt.Axes] = None,
    n_bins: int = 8,
    show_fit: bool = True,
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (6, 5),
    title: Optional[str] = None,
    n_bootstrap: int = 0,
    show_ci: bool = True,
    seed: int = 42
) -> Tuple[plt.Axes, List[Dict]]:
    """
    Plot multiple psychometric curves overlaid on the same axes.
    
    Args:
        stimuli_list: List of stimulus arrays
        choices_list: List of choice arrays
        labels: Labels for legend
        ax: Matplotlib axes
        n_bins: Number of bins
        show_fit: Whether to show fitted curves
        colors: List of colours
        figsize: Figure size if creating new figure
        title: Axes title
        n_bootstrap: Number of bootstrap samples for CIs
        show_ci: Whether to show CI bands
        seed: Random seed for bootstrap
    
    Returns:
        ax: Matplotlib axes
        infos: List of info dicts
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if colors is None:
        colors = [f'C{i}' for i in range(len(stimuli_list))]
    
    infos = []
    
    for i, (stim, choices, label, color) in enumerate(zip(stimuli_list, choices_list, labels, colors)):
        _, info = plot_psychometric(
            stim, choices, ax=ax,
            n_bins=n_bins,
            show_fit=show_fit,
            show_gof=False,
            show_params=False,
            show_reference_lines=False,
            color=color,
            label=label,
            n_bootstrap=n_bootstrap,
            show_ci=show_ci,
            seed=seed + i
        )
        infos.append(info)
    
    # Add reference lines once
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    
    ax.legend(loc='lower right')
    
    if title:
        ax.set_title(title)
    
    return ax, infos
