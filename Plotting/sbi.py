"""
SBI-specific plotting utilities.

Visualisation functions for multi-session SBI inference results:
- Parameter trajectory recovery (GP-linked params over sessions)
- Marginal posterior distributions
- Corner / pairplot of posteriors
- Psychometric curve overlays from posterior samples vs observed
- Performance and learning rate trajectories

Usage:
    from Plotting.sbi import (
        plot_parameter_trajectories,
        plot_marginal_posteriors,
        plot_psychometric_overlay,
        plot_performance_trajectory,
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings


# =============================================================================
# COLOUR PALETTE
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
        prior_bounds: Dict mapping param names to (low, high) bounds from the
                      prior / search range. If provided, y-axis spans this full
                      range (with small padding) so the visual context is clear.
        param_links: Dict of link specs (e.g. from SBIFitter). If provided
                     and prior_bounds is None, bounds are extracted automatically.
        param_names: Which params to plot. Default: all.
        figsize: Figure size.
        ci_level: For annotation only (actual CI from trajectories).
        title: Overall figure title.
        x_label: Label for x-axis.
        show_samples: Number of individual posterior trajectory samples to show.
    
    Returns:
        Matplotlib figure
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
            
            ax.axhspan(ci_lo, ci_hi, alpha=0.25, color=colour, label=f'{ci_level:.0%} CI')
            ax.axhline(mean, color=colour, linewidth=2, label='Posterior median')
            
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
            
            # CI envelope
            ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color=colour,
                           label=f'{ci_level:.0%} CI')
            
            # Individual samples
            if show_samples > 0 and 'samples' in traj:
                samples = traj['samples']
                n_avail = min(show_samples, len(samples))
                for j in range(n_avail):
                    ax.plot(x, samples[j], color=colour, alpha=0.03, linewidth=0.5)
            
            # Posterior median
            ax.plot(x, median, color=colour, linewidth=2, label='Posterior median')
            
            # Ground truth
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
            # Shade prior bounds lightly
            ax.axhspan(lo, hi, alpha=0.04, color='grey', zorder=0)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend(loc='best', fontsize=7)
    
    # Hide unused axes
    for j in range(n_params, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    
    fig.tight_layout()
    return fig


# =============================================================================
# MARGINAL POSTERIORS
# =============================================================================

def plot_marginal_posteriors(
    trajectories: Dict[str, Dict[str, np.ndarray]],
    ground_truth: Optional[Dict[str, Any]] = None,
    param_names: Optional[List[str]] = None,
    sessions_to_show: Optional[List[int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    n_bins: int = 40,
) -> plt.Figure:
    """
    Plot marginal posterior distributions for each parameter.
    
    For constant params: single histogram.
    For varying params: one histogram per selected session, colour-coded.
    
    Args:
        trajectories: From SBIFitter.extract_trajectories()
        ground_truth: Optional ground truth values
        param_names: Which params to plot
        sessions_to_show: Which session indices to show for varying params.
                         Default: first, middle, last.
        figsize: Figure size
        n_bins: Histogram bins
    
    Returns:
        Matplotlib figure
    """
    if param_names is None:
        param_names = list(trajectories.keys())
    
    n_params = len(param_names)
    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    
    for i, name in enumerate(param_names):
        ax = axes_flat[i]
        traj = trajectories[name]
        colour = PARAM_COLOURS.get(name, f'C{i}')
        samples = traj['samples']
        
        if traj['link_type'] == 'constant':
            ax.hist(samples, bins=n_bins, color=colour, alpha=0.7,
                   edgecolor='white', density=True)
            
            if ground_truth is not None and name in ground_truth:
                gt = ground_truth[name]
                gt_val = gt[0] if hasattr(gt, '__len__') else gt
                ax.axvline(gt_val, color='k', linestyle='--', linewidth=2,
                          label=f'True: {gt_val:.3f}')
            
            ax.set_xlabel(name)
            ax.set_ylabel('Density')
            ax.set_title(name)
            ax.legend(fontsize=8)
        
        else:
            # Select sessions to show
            n_sess = samples.shape[1]
            if sessions_to_show is None:
                if n_sess <= 5:
                    sess_idx = list(range(n_sess))
                else:
                    sess_idx = [0, n_sess // 4, n_sess // 2, 3 * n_sess // 4, n_sess - 1]
            else:
                sess_idx = sessions_to_show
            
            cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(sess_idx)))
            
            for j, s in enumerate(sess_idx):
                ax.hist(samples[:, s], bins=n_bins, alpha=0.5,
                       color=cmap[j], density=True, label=f'S{s}')
                
                if ground_truth is not None and name in ground_truth:
                    gt = np.atleast_1d(ground_truth[name])
                    if s < len(gt):
                        ax.axvline(gt[s], color=cmap[j], linestyle='--',
                                  linewidth=1.5)
            
            ax.set_xlabel(name)
            ax.set_ylabel('Density')
            ax.set_title(f'{name} (selected sessions)')
            ax.legend(fontsize=7, loc='upper right')
    
    for j in range(n_params, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    fig.tight_layout()
    return fig


# =============================================================================
# CORNER / PAIRPLOT
# =============================================================================

def plot_pairplot(
    samples: np.ndarray,
    param_names: List[str],
    ground_truth: Optional[np.ndarray] = None,
    figsize: Optional[Tuple[float, float]] = None,
    n_bins: int = 30,
    max_params: int = 8,
) -> plt.Figure:
    """
    Corner plot (pairwise scatter + marginals) for posterior samples.
    
    For high-dimensional posteriors (multi-session varying params),
    only the constant params + a few session slices are shown.
    
    Args:
        samples: (n_samples, n_params) posterior samples
        param_names: Parameter names matching columns
        ground_truth: Optional true parameter values
        figsize: Figure size
        n_bins: Bins for marginal histograms
        max_params: Maximum number of params to include
    
    Returns:
        Matplotlib figure
    """
    n_params = min(len(param_names), max_params, samples.shape[1])
    samples = samples[:, :n_params]
    names = param_names[:n_params]
    
    if figsize is None:
        figsize = (2.5 * n_params, 2.5 * n_params)
    
    fig, axes = plt.subplots(n_params, n_params, figsize=figsize)
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: marginal histogram
                ax.hist(samples[:, i], bins=n_bins, color='steelblue',
                       alpha=0.7, edgecolor='white', density=True)
                if ground_truth is not None and i < len(ground_truth):
                    ax.axvline(ground_truth[i], color='red', linewidth=2)
            
            elif i > j:
                # Lower triangle: scatter
                ax.scatter(samples[:, j], samples[:, i], alpha=0.05,
                          s=1, color='steelblue')
                if ground_truth is not None:
                    if j < len(ground_truth) and i < len(ground_truth):
                        ax.axvline(ground_truth[j], color='red',
                                  linewidth=1, alpha=0.7)
                        ax.axhline(ground_truth[i], color='red',
                                  linewidth=1, alpha=0.7)
            else:
                # Upper triangle: hide
                ax.set_visible(False)
            
            # Labels
            if i == n_params - 1:
                ax.set_xlabel(names[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            
            if j == 0 and i != 0:
                ax.set_ylabel(names[i], fontsize=8)
            elif j != 0:
                ax.set_yticklabels([])
            
            ax.tick_params(labelsize=6)
    
    fig.tight_layout()
    return fig


# =============================================================================
# PSYCHOMETRIC OVERLAY
# =============================================================================

def plot_psychometric_overlay(
    stimuli_per_session: List[np.ndarray],
    choices_per_session: List[np.ndarray],
    posterior_choices: Optional[List[List[np.ndarray]]] = None,
    sessions_to_show: Optional[List[int]] = None,
    n_bins: int = 8,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay observed psychometric curves with posterior predictive samples.
    
    Shows binned data points, fitted cumulative Gaussian, and (optionally)
    posterior predictive CI band.
    
    Args:
        stimuli_per_session: List of stimulus arrays
        choices_per_session: List of observed choice arrays
        posterior_choices: List (sessions) of lists (samples) of choice arrays.
                         If provided, draws posterior predictive psychometric curves.
        sessions_to_show: Which sessions to plot. Default: up to 9.
        n_bins: Bins for psychometric curves
        figsize: Figure size
        title: Overall title
    
    Returns:
        Matplotlib figure
    """
    from Helpers.psychometry import fit_psychometric
    
    n_sessions = len(stimuli_per_session)
    
    if sessions_to_show is None:
        if n_sessions <= 9:
            sessions_to_show = list(range(n_sessions))
        else:
            step = max(1, n_sessions // 9)
            sessions_to_show = list(range(0, n_sessions, step))[:9]
    
    n_show = len(sessions_to_show)
    n_cols = min(3, n_show)
    n_rows = int(np.ceil(n_show / n_cols))
    
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    
    x_fine = np.linspace(-1, 1, 200)
    
    for idx, s in enumerate(sessions_to_show):
        ax = axes_flat[idx]
        stim = stimuli_per_session[s]
        choices = choices_per_session[s]
        
        # Remove NaN choices
        valid = ~np.isnan(choices)
        stim_v = stim[valid]
        choices_v = choices[valid].astype(int)
        
        # Bin observed data
        bin_edges = np.linspace(stim_v.min() - 0.01, stim_v.max() + 0.01, n_bins + 1)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_idx = np.digitize(stim_v, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        
        obs_prob = np.zeros(n_bins)
        obs_count = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                obs_prob[b] = choices_v[mask].mean()
                obs_count[b] = mask.sum()
            else:
                obs_prob[b] = np.nan
        
        # Posterior predictive curves
        if posterior_choices is not None and s < len(posterior_choices):
            pred_curves = []
            for pc in posterior_choices[s]:
                pc_v = pc[valid].astype(int)
                pred_prob = np.zeros(n_bins)
                for b in range(n_bins):
                    mask = bin_idx == b
                    if mask.sum() > 0:
                        pred_prob[b] = pc_v[mask].mean()
                    else:
                        pred_prob[b] = np.nan
                pred_curves.append(pred_prob)
            
            pred_curves = np.array(pred_curves)
            pred_mean = np.nanmean(pred_curves, axis=0)
            pred_lo = np.nanpercentile(pred_curves, 2.5, axis=0)
            pred_hi = np.nanpercentile(pred_curves, 97.5, axis=0)
            
            ax.fill_between(bin_centres, pred_lo, pred_hi,
                           alpha=0.2, color='steelblue', label='95% pred. CI')
            ax.plot(bin_centres, pred_mean, '-', color='steelblue',
                   linewidth=1.5, label='Pred. mean')
        
        # Fit cumulative Gaussian to observed data
        fit_result = fit_psychometric(stim_v, choices_v.astype(float),
                                      x_eval=x_fine)
        if fit_result.get('success', False):
            ax.plot(fit_result['x_fit'], fit_result['y_fit'], '-',
                   color='k', linewidth=2, alpha=0.8, label='Fit')
        
        # Observed data points
        valid_bins = obs_count > 0
        sizes = np.clip(obs_count[valid_bins] * 3, 20, 150)
        ax.scatter(bin_centres[valid_bins], obs_prob[valid_bins],
                  s=sizes, color='k', zorder=5, label='Observed')
        
        # Reference lines
        ax.axhline(0.5, color='grey', linestyle=':', alpha=0.5)
        ax.axvline(0, color='grey', linestyle=':', alpha=0.5)
        
        ax.set_xlim(stim_v.min() - 0.1, stim_v.max() + 0.1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Stimulus')
        ax.set_ylabel('P(choose B)')
        
        acc = np.mean(choices_v == (stim_v > 0).astype(int))
        # Add fit params to title if available
        if fit_result.get('success', False):
            mu = fit_result['mu']
            sigma = fit_result['sigma']
            ax.set_title(f'Session {s} (acc={acc:.2f}, μ={mu:.2f}, σ={sigma:.2f})')
        else:
            ax.set_title(f'Session {s} (acc={acc:.2f}, n={len(choices_v)})')
        
        if idx == 0:
            ax.legend(fontsize=7, loc='lower right')
    
    for j in range(n_show, len(axes_flat)):
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
        performance_per_session: Observed accuracy per session
        session_indices: X-axis values (default: 0, 1, 2, ...)
        predicted_performance: Optional model-predicted accuracy
        predicted_ci: Optional (lower, upper) CI arrays
        figsize: Figure size
        title: Plot title
        chance_level: Chance performance level
    
    Returns:
        Matplotlib figure
    """
    n_sessions = len(performance_per_session)
    if session_indices is None:
        session_indices = np.arange(n_sessions)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Predicted
    if predicted_performance is not None:
        ax.plot(session_indices, predicted_performance, 'o-',
               color='steelblue', linewidth=2, markersize=6,
               label='Model predicted')
        
        if predicted_ci is not None:
            ax.fill_between(session_indices, predicted_ci[0], predicted_ci[1],
                           alpha=0.2, color='steelblue')
    
    # Observed
    ax.plot(session_indices, performance_per_session, 's-',
           color='k', linewidth=2, markersize=7, label='Observed')
    
    # Chance
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
# SUMMARY STATS COMPARISON
# =============================================================================

def plot_summary_stats_comparison(
    observed: np.ndarray,
    simulated: np.ndarray,
    stat_names: Optional[List[str]] = None,
    n_per_session: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Compare observed vs posterior predictive summary statistics.
    
    Shows violin plots of simulated stats with observed as red dots.
    
    Args:
        observed: (n_stats,) observed summary stats
        simulated: (n_sims, n_stats) posterior predictive stats
        stat_names: Names for each stat
        n_per_session: If set, groups stats by session
        figsize: Figure size
        title: Overall title
    
    Returns:
        Matplotlib figure
    """
    n_stats = len(observed)
    
    if stat_names is None:
        stat_names = [f'stat_{i}' for i in range(n_stats)]
    
    if n_per_session is not None and n_stats > n_per_session:
        # Group by session
        n_sessions = n_stats // n_per_session
        n_cols = min(3, n_sessions)
        n_rows = int(np.ceil(n_sessions / n_cols))
        
        if figsize is None:
            figsize = (6 * n_cols, 4 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()
        
        base_names = stat_names[:n_per_session]
        
        for s in range(n_sessions):
            ax = axes_flat[s]
            start = s * n_per_session
            end = start + n_per_session
            
            obs_s = observed[start:end]
            sim_s = simulated[:, start:end]
            
            parts = ax.violinplot(sim_s, positions=range(n_per_session),
                                 showmeans=True, showmedians=False)
            for pc in parts['bodies']:
                pc.set_alpha(0.5)
            
            ax.scatter(range(n_per_session), obs_s, color='red',
                      zorder=5, s=40, label='Observed')
            
            ax.set_xticks(range(n_per_session))
            ax.set_xticklabels(base_names, rotation=45, ha='right', fontsize=7)
            ax.set_title(f'Session {s}')
            
            if s == 0:
                ax.legend(fontsize=8)
        
        for j in range(n_sessions, len(axes_flat)):
            axes_flat[j].set_visible(False)
    
    else:
        # All stats in one plot
        if figsize is None:
            figsize = (max(8, n_stats * 0.8), 5)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        parts = ax.violinplot(simulated, positions=range(n_stats),
                             showmeans=True, showmedians=False)
        for pc in parts['bodies']:
            pc.set_alpha(0.5)
        
        ax.scatter(range(n_stats), observed, color='red',
                  zorder=5, s=40, label='Observed')
        
        ax.set_xticks(range(n_stats))
        ax.set_xticklabels(stat_names, rotation=45, ha='right', fontsize=7)
        ax.legend()
    
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    
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
        performance: Accuracy per session
        eta_trajectory: Recovered eta_learning (median)
        eta_ci: (lower, upper) CI for eta
        eta_true: Ground truth eta per session
        eta_bounds: (low, high) prior bounds for eta — sets y-axis range
        session_indices: X-axis values
        figsize: Figure size
        title: Overall title
    
    Returns:
        Matplotlib figure
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
        
        # Set y-axis to full prior range if provided
        if eta_bounds is not None:
            lo, hi = eta_bounds
            padding = (hi - lo) * 0.05
            ax.set_ylim(lo - padding, hi + padding)
    else:
        axes[0].set_xlabel('Session')
    
    fig.tight_layout()
    return fig


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'plot_parameter_trajectories',
    'plot_marginal_posteriors',
    'plot_pairplot',
    'plot_psychometric_overlay',
    'plot_performance_trajectory',
    'plot_summary_stats_comparison',
    'plot_learning_trajectory',
    'PARAM_COLOURS',
    'PHASE_COLOURS',
]
