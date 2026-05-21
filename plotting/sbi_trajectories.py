"""
SBI Trajectory Plotting

Parameter evolution over sessions: recovered trajectories with credible
intervals, performance curves, and learning rate plots.

Usage:
    from plotting.sbi_trajectories import (
        plot_parameter_trajectories,
        plot_performance_trajectory,
        plot_learning_trajectory,
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any


# =============================================================================
# SHARED COLOUR PALETTES
# =============================================================================

PARAM_COLOURS = {
    'sigma_percep': '#1f77b4',   # blue
    'A_repulsion': '#ff7f0e',    # orange
    'eta_learning': '#2ca02c',   # green
    'eta_relax': '#d62728',      # red
}

PHASE_COLOURS = {
    'naive': '#e74c3c',
    'expert': '#2ecc71',
    'post_shift': '#f39c12',
}


# =============================================================================
# PARAMETER TRAJECTORY PLOTS
# =============================================================================

def plot_parameter_trajectories(
    trajectories: Dict[str, Dict[str, np.ndarray]],
    ground_truth: Optional[Dict[str, np.ndarray]] = None,
    prior_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    param_links: Optional[Dict[str, Any]] = None,
    param_names: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ci_level: float = 0.95,
    title: Optional[str] = None,
    x_label: str = 'Session',
    show_samples: int = 0,
) -> plt.Figure:
    """
    Plot recovered parameter trajectories across sessions with credible intervals.

    For constant parameters, shows marginal posterior as horizontal band.
    For varying parameters, shows trajectory with CI envelope.

    Args:
        trajectories: Output from SBIFitter.extract_trajectories().
                      Dict[param_name] -> {'mean', 'median', 'ci_low', 'ci_high',
                      'samples', 'session_indices', 'link_type'}
        ground_truth: Optional dict mapping param names to arrays (per-session)
                      or scalars. If provided, overlaid as dashed line.
        prior_bounds: Dict mapping param names to (low, high) bounds.
        param_links: Dict of link specs. If provided and prior_bounds is None,
                     bounds are extracted automatically.
        param_names: Which params to plot. Default: all.
        figsize: Figure size.
        ci_level: For annotation only (actual CI from trajectories).
        title: Overall figure title.
        x_label: Label for x-axis.
        show_samples: Number of individual posterior trajectory samples to show.

    Returns:
        Matplotlib figure.
    """
    # Auto-extract prior_bounds from param_links if not given directly
    if prior_bounds is None and param_links is not None:
        prior_bounds = {}
        for name, link in param_links.items():
            if hasattr(link, 'bounds'):
                prior_bounds[name] = link.bounds
    if param_names is None:
        param_names = list(trajectories.keys())

    n_params = len(param_names)

    if figsize is None:
        figsize = (5 * min(n_params, 4), 4 * max(1, (n_params + 3) // 4))

    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for i, name in enumerate(param_names):
        ax = axes_flat[i]
        traj = trajectories[name]
        colour = PARAM_COLOURS.get(name, f'C{i}')
        x = traj['session_indices']

        if traj['link_type'] == 'constant':
            # Horizontal band for constant parameter
            mean = traj['mean']
            ci_lo = traj['ci_low']
            ci_hi = traj['ci_high']

            ax.axhspan(ci_lo, ci_hi, alpha=0.25, color=colour,
                       label=f'{ci_level:.0%} CI')
            ax.axhline(mean, color=colour, linewidth=2,
                       label='Posterior median')

            if ground_truth is not None and name in ground_truth:
                gt = ground_truth[name]
                gt_val = gt[0] if hasattr(gt, '__len__') else gt
                ax.axhline(gt_val, color='k', linestyle='--', linewidth=1.5,
                           label='Ground truth')

            ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

        else:
            # Trajectory with CI envelope
            median = traj['median']
            ci_lo = traj['ci_low']
            ci_hi = traj['ci_high']

            ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color=colour,
                            label=f'{ci_level:.0%} CI')

            if show_samples > 0 and 'samples' in traj:
                samples = traj['samples']
                n_avail = min(show_samples, len(samples))
                for j in range(n_avail):
                    ax.plot(x, samples[j], color=colour, alpha=0.03,
                            linewidth=0.5)

            ax.plot(x, median, color=colour, linewidth=2,
                    label='Posterior median')

            if ground_truth is not None and name in ground_truth:
                gt = np.atleast_1d(ground_truth[name])
                if len(gt) == len(x):
                    ax.plot(x, gt, 'k--', linewidth=1.5, label='Ground truth')
                else:
                    ax.axhline(float(gt[0]), color='k', linestyle='--',
                               linewidth=1.5, label='Ground truth')

        # Set y-axis to full prior range if provided
        if prior_bounds is not None and name in prior_bounds:
            lo, hi = prior_bounds[name]
            padding = (hi - lo) * 0.05
            ax.set_ylim(lo - padding, hi + padding)
            ax.axhspan(lo, hi, alpha=0.04, color='grey', zorder=0)

        ax.set_xlabel(x_label)
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend(loc='best', fontsize=7)

    for j in range(n_params, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    fig.tight_layout()
    return fig


# =============================================================================
# PERFORMANCE TRAJECTORY
# =============================================================================

def plot_performance_trajectory(
    performance_per_session: np.ndarray,
    session_indices: Optional[np.ndarray] = None,
    predicted_performance: Optional[np.ndarray] = None,
    predicted_ci: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    figsize: Tuple[float, float] = (10, 4),
    title: Optional[str] = None,
    chance_level: float = 0.5,
) -> plt.Figure:
    """
    Plot performance (accuracy) trajectory across sessions.

    Args:
        performance_per_session: Observed accuracy per session.
        session_indices: X-axis values (default: 0, 1, 2, ...).
        predicted_performance: Optional model-predicted accuracy.
        predicted_ci: Optional (lower, upper) CI arrays.
        figsize: Figure size.
        title: Plot title.
        chance_level: Chance performance level.

    Returns:
        Matplotlib figure.
    """
    n_sessions = len(performance_per_session)
    if session_indices is None:
        session_indices = np.arange(n_sessions)

    fig, ax = plt.subplots(figsize=figsize)

    if predicted_performance is not None:
        ax.plot(session_indices, predicted_performance, 'o-',
                color='steelblue', linewidth=2, markersize=6,
                label='Model predicted')
        if predicted_ci is not None:
            ax.fill_between(session_indices, predicted_ci[0], predicted_ci[1],
                            alpha=0.2, color='steelblue')

    ax.plot(session_indices, performance_per_session, 's-',
            color='k', linewidth=2, markersize=7, label='Observed')
    ax.axhline(chance_level, color='grey', linestyle=':', alpha=0.5,
               label='Chance')

    ax.set_xlabel('Session')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.3, 1.05)
    ax.legend(loc='lower right')

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


# =============================================================================
# MULTI-SESSION LEARNING CURVE
# =============================================================================

def plot_learning_trajectory(
    performance: np.ndarray,
    eta_trajectory: Optional[np.ndarray] = None,
    eta_ci: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    eta_true: Optional[np.ndarray] = None,
    eta_bounds: Optional[Tuple[float, float]] = None,
    session_indices: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (12, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Combined plot of performance + learning rate trajectory.

    Top panel: accuracy over sessions.
    Bottom panel: eta_learning trajectory (recovered vs true).

    Args:
        performance: Accuracy per session.
        eta_trajectory: Recovered eta_learning (median).
        eta_ci: (lower, upper) CI for eta.
        eta_true: Ground truth eta per session.
        eta_bounds: (low, high) prior bounds for eta.
        session_indices: X-axis values.
        figsize: Figure size.
        title: Overall title.

    Returns:
        Matplotlib figure.
    """
    n_panels = 1 + (1 if eta_trajectory is not None else 0)
    n_sessions = len(performance)
    if session_indices is None:
        session_indices = np.arange(n_sessions)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]

    # Panel 1: Performance
    ax = axes[0]
    ax.plot(session_indices, performance, 's-k', linewidth=2, markersize=6)
    ax.axhline(0.5, color='grey', linestyle=':', alpha=0.5)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.3, 1.05)
    ax.set_title('Performance trajectory' if title is None else title)

    # Panel 2: eta_learning
    if eta_trajectory is not None:
        ax = axes[1]

        if eta_ci is not None:
            ax.fill_between(session_indices, eta_ci[0], eta_ci[1],
                            alpha=0.2, color=PARAM_COLOURS['eta_learning'])

        ax.plot(session_indices, eta_trajectory, 'o-',
                color=PARAM_COLOURS['eta_learning'], linewidth=2,
                markersize=5, label='Recovered (median)')

        if eta_true is not None:
            ax.plot(session_indices, eta_true, 's--k', linewidth=1.5,
                    markersize=5, label='Ground truth')

        ax.set_ylabel('η_learning')
        ax.set_xlabel('Session')
        ax.legend(loc='best', fontsize=8)

        if eta_bounds is not None:
            lo, hi = eta_bounds
            padding = (hi - lo) * 0.05
            ax.set_ylim(lo - padding, hi + padding)
    else:
        axes[0].set_xlabel('Session')

    fig.tight_layout()
    return fig
