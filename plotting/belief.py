"""
Belief distribution plotting utilities.

Functions for visualising the agent's boundary belief distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from typing import Optional, List, Tuple, Dict


def plot_belief_distribution(
    x: np.ndarray,
    belief: np.ndarray,
    ax: Optional[plt.Axes] = None,
    true_boundary: Optional[float] = 0.0,
    color: Optional[str] = None,
    label: Optional[str] = None,
    linewidth: int = 2,
    fill: bool = False,
    fill_alpha: float = 0.3,
    show_stats: bool = False,
    title: Optional[str] = None
) -> Tuple[plt.Axes, Dict]:
    """
    Plot a single belief distribution over the boundary location.
    
    Args:
        x: Stimulus space grid
        belief: Belief density values (PDF that integrates to 1)
        ax: Matplotlib axes (creates new if None)
        true_boundary: True boundary location for reference line
        color: Line colour
        label: Legend label
        linewidth: Line width
        fill: Whether to fill under the curve
        fill_alpha: Fill transparency
        show_stats: Whether to show mean and std
        title: Axes title
    
    Returns:
        ax: Matplotlib axes
        stats: Dict with 'mean' and 'std' of belief
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    if color is None:
        color = 'C0'
    
    # Belief should already be a normalised PDF from the model (integrates to 1)
    # We use trapezoid integration for computing statistics (not np.sum!)
    belief_mean = trapezoid(x * belief, x)
    belief_var = trapezoid((x - belief_mean)**2 * belief, x)
    belief_std = np.sqrt(belief_var)
    
    stats = {'mean': belief_mean, 'std': belief_std}
    
    # Plot
    ax.plot(x, belief, color=color, linewidth=linewidth, label=label)
    
    if fill:
        ax.fill_between(x, belief, alpha=fill_alpha, color=color)
    
    if true_boundary is not None:
        ax.axvline(true_boundary, color='red', linestyle='--', 
                   alpha=0.7, label='True boundary')
    
    if show_stats:
        ax.axvline(belief_mean, color=color, linestyle=':', alpha=0.7)
        text = f"μ = {belief_mean:.3f}\nσ = {belief_std:.3f}"
        ax.text(0.95, 0.95, text, transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Boundary location')
    ax.set_ylabel('Belief density')
    ax.set_xlim(x.min(), x.max())
    
    if title:
        ax.set_title(title)
    
    return ax, stats


def plot_belief_distributions(
    x: np.ndarray,
    beliefs: List[np.ndarray],
    labels: List[str],
    ax: Optional[plt.Axes] = None,
    true_boundary: Optional[float] = 0.0,
    colors: Optional[List] = None,
    linewidth: int = 2,
    figsize: Tuple[int, int] = (7, 5),
    title: Optional[str] = None,
    legend_title: Optional[str] = None
) -> Tuple[plt.Axes, List[Dict]]:
    """
    Plot multiple belief distributions overlaid.
    
    Args:
        x: Stimulus space grid
        beliefs: List of belief density arrays
        labels: Labels for each belief
        ax: Matplotlib axes
        true_boundary: True boundary for reference line
        colors: List of colours
        linewidth: Line width
        figsize: Figure size
        title: Axes title
        legend_title: Legend title
    
    Returns:
        ax: Matplotlib axes
        stats_list: List of stats dicts for each belief
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(beliefs)))
    
    stats_list = []
    
    for belief, label, color in zip(beliefs, labels, colors):
        _, stats = plot_belief_distribution(
            x, belief, ax=ax,
            true_boundary=None,  # Add once at the end
            color=color,
            label=label,
            linewidth=linewidth,
            show_stats=False
        )
        stats_list.append(stats)
    
    if true_boundary is not None:
        ax.axvline(true_boundary, color='red', linestyle='--', 
                   alpha=0.7, label='True boundary')
    
    ax.legend(title=legend_title, loc='upper right')
    
    if title:
        ax.set_title(title)
    
    return ax, stats_list


def plot_belief_evolution(
    x: np.ndarray,
    belief_history: np.ndarray,
    trial_indices: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    true_boundary: float = 0.0,
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (8, 5),
    title: Optional[str] = None
) -> plt.Axes:
    """
    Plot belief evolution over trials as a heatmap.
    
    Args:
        x: Stimulus space grid (n_points,)
        belief_history: Belief over trials (n_trials, n_points)
        trial_indices: Trial indices to show (default: all)
        ax: Matplotlib axes
        true_boundary: True boundary location
        cmap: Colourmap
        figsize: Figure size
        title: Axes title
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    n_trials = belief_history.shape[0]
    
    if trial_indices is None:
        trial_indices = np.arange(n_trials)
    
    # Plot heatmap
    im = ax.imshow(
        belief_history[trial_indices, :].T,
        aspect='auto',
        origin='lower',
        extent=[trial_indices[0], trial_indices[-1], x.min(), x.max()],
        cmap=cmap
    )
    
    # True boundary line
    ax.axhline(true_boundary, color='red', linestyle='--', linewidth=2,
               label='True boundary')
    
    ax.set_xlabel('Trial')
    ax.set_ylabel('Boundary location')
    plt.colorbar(im, ax=ax, label='Belief density')
    ax.legend(loc='upper right')
    
    if title:
        ax.set_title(title)
    
    return ax


def plot_belief_uncertainty(
    burn_in_values: List[int],
    uncertainties: List[float],
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (6, 4),
    title: Optional[str] = None,
    show_uniform_reference: bool = True,
    uniform_std: Optional[float] = None
) -> plt.Axes:
    """
    Plot belief uncertainty (std) as a function of experience (burn-in).
    
    Args:
        burn_in_values: List of burn-in trial counts
        uncertainties: Corresponding belief standard deviations
        ax: Matplotlib axes
        figsize: Figure size
        title: Axes title
        show_uniform_reference: Whether to show uniform distribution reference
        uniform_std: Std of uniform distribution (computed if None)
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(burn_in_values, uncertainties, 'o-', color='C0', 
            linewidth=2, markersize=8)
    
    if show_uniform_reference:
        if uniform_std is None:
            # Std of uniform distribution on [-1, 1]
            uniform_std = np.sqrt(1/3)  # = 2/sqrt(12) ≈ 0.577
        ax.axhline(uniform_std, color='red', linestyle='--', alpha=0.5)
        ax.text(0.95, 0.95, f'Uniform std ≈ {uniform_std:.2f}', 
                transform=ax.transAxes, ha='right', va='top', 
                fontsize=9, color='red')
    
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Burn-in trials')
    ax.set_ylabel('Belief std (uncertainty)')
    ax.set_xscale('symlog', linthresh=10)
    
    if title:
        ax.set_title(title)
    
    return ax
