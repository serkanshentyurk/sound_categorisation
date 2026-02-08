"""
Plotting functions for update matrix analysis.

The update matrix captures serial dependence: how does the previous trial's
stimulus location shift the current psychometric curve?
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING

from Analysis.update_matrix import matrix_error

if TYPE_CHECKING:
    from Models.BE_core import ModelTrace


def plot_update_matrix(update_matrix: np.ndarray, 
                       title: str = 'Update Matrix',
                       vmin: Optional[float] = None, 
                       vmax: Optional[float] = None,
                       cmap: str = 'RdBu_r',
                       ax: Optional[plt.Axes] = None,
                       show_colorbar: bool = True,
                       colorbar_label: str = r'$\Delta$P(B)') -> plt.Axes:
    """
    Plot update matrix as a heatmap.
    
    Args:
        update_matrix: (n_bins, n_bins) update matrix
        title: Plot title
        vmin, vmax: Colour scale limits (auto-computed if None)
        cmap: Colourmap
        ax: Matplotlib axes (creates new if None)
        show_colorbar: Whether to show colourbar
        colorbar_label: Label for colourbar
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    n_bins = update_matrix.shape[0]
    
    # Auto-compute colour limits if not provided
    if vmin is None or vmax is None:
        max_abs = np.nanmax(np.abs(update_matrix))
        if np.isnan(max_abs) or max_abs == 0:
            max_abs = 0.3
        vmin = -max_abs if vmin is None else vmin
        vmax = max_abs if vmax is None else vmax
    
    im = ax.imshow(update_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin='lower', aspect='equal')
    
    ax.set_xlabel('Previous stimulus bin')
    ax.set_ylabel('Current stimulus bin')
    ax.set_title(title)
    
    # Tick labels
    tick_labels = [f'{i+1}' for i in range(n_bins)]
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels(tick_labels)
    
    if show_colorbar:
        plt.colorbar(im, ax=ax, label=colorbar_label)
    
    return ax


def plot_update_matrix_comparison(data_matrix: np.ndarray, 
                                  model_matrix: np.ndarray,
                                  data_title: str = 'Data',
                                  model_title: str = 'Model',
                                  figsize: Tuple[int, int] = (14, 4),
                                  show_mse: bool = True) -> plt.Figure:
    """
    Plot data and model update matrices side by side with difference.
    
    Args:
        data_matrix: Data update matrix
        model_matrix: Model update matrix
        data_title: Title for data panel
        model_title: Title for model panel
        figsize: Figure size
        show_mse: Whether to show MSE in title
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Use same colour scale for data and model
    vmax = max(np.nanmax(np.abs(data_matrix)), np.nanmax(np.abs(model_matrix)))
    if np.isnan(vmax) or vmax == 0:
        vmax = 0.3
    vmin = -vmax
    
    plot_update_matrix(data_matrix, data_title, vmin, vmax, ax=axes[0])
    plot_update_matrix(model_matrix, model_title, vmin, vmax, ax=axes[1])
    
    diff = model_matrix - data_matrix
    diff_max = np.nanmax(np.abs(diff))
    if np.isnan(diff_max) or diff_max == 0:
        diff_max = 0.1
    plot_update_matrix(diff, 'Model - Data', -diff_max, diff_max, ax=axes[2])
    
    if show_mse:
        error = matrix_error(model_matrix, data_matrix)
        fig.suptitle(f'Update Matrix Comparison (MSE = {error:.4f})', y=1.02)
    
    plt.tight_layout()
    return fig


def plot_conditional_psychometrics(conditional_matrix: np.ndarray,
                                   midpoints: Optional[np.ndarray] = None,
                                   overall_curve: Optional[np.ndarray] = None,
                                   title: str = 'Conditional Psychometric Curves',
                                   ax: Optional[plt.Axes] = None,
                                   cmap: str = 'coolwarm') -> plt.Axes:
    """
    Plot conditional psychometric curves (one per previous-stimulus bin).
    
    This shows how the psychometric curve shifts depending on where the 
    previous stimulus was located.
    
    Args:
        conditional_matrix: (n_bins, n_bins) where column j is the psychometric
                           curve for trials where previous stimulus was in bin j
        midpoints: Stimulus bin midpoints (x-axis)
        overall_curve: Overall psychometric curve (no conditioning)
        title: Plot title
        ax: Matplotlib axes
        cmap: Colourmap for different conditions
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    n_bins = conditional_matrix.shape[0]
    
    if midpoints is None:
        bin_edges = np.linspace(-1, 1, n_bins + 1)
        midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Colour gradient for previous stimulus bins
    colors = plt.cm.get_cmap(cmap)(np.linspace(0.1, 0.9, n_bins))
    
    # Plot overall curve first (if provided)
    if overall_curve is not None:
        ax.plot(midpoints, overall_curve, 'k-', linewidth=2.5, 
                label='Overall', zorder=10)
    
    # Plot conditional curves
    for j in range(n_bins):
        curve = conditional_matrix[:, j]
        if not np.all(np.isnan(curve)):
            ax.plot(midpoints, curve, '-', color=colors[j], 
                    linewidth=1.5, alpha=0.8,
                    label=f'Prev bin {j+1}')
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Current stimulus')
    ax.set_ylabel('P(choose B)')
    ax.set_title(title)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    
    return ax


def plot_method_comparison(trace: 'ModelTrace',
                           n_bins: int = 8,
                           trial_filter: str = 'post_correct',
                           seed: int = 42,
                           figsize: Tuple[int, int] = (14, 4)) -> plt.Figure:
    """
    Compare deterministic vs stochastic update matrix computation methods.
    
    Args:
        trace: ModelTrace object from simulation
        n_bins: Number of bins
        trial_filter: 'post_correct' or 'all'
        seed: Random seed for stochastic method
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    from Analysis.update_matrix import compute_model_update_matrix
    
    # Compute both methods
    det_update, det_cond, det_info = compute_model_update_matrix(
        trace, method='deterministic', n_bins=n_bins, trial_filter=trial_filter
    )
    stoch_update, stoch_cond, stoch_info = compute_model_update_matrix(
        trace, method='stochastic', n_bins=n_bins, trial_filter=trial_filter, seed=seed
    )
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Use same colour scale
    vmax = max(np.nanmax(np.abs(det_update)), np.nanmax(np.abs(stoch_update)))
    if np.isnan(vmax) or vmax == 0:
        vmax = 0.3
    vmin = -vmax
    
    plot_update_matrix(det_update, 'Deterministic', vmin, vmax, ax=axes[0])
    plot_update_matrix(stoch_update, 'Stochastic', vmin, vmax, ax=axes[1])
    
    diff = stoch_update - det_update
    diff_max = np.nanmax(np.abs(diff))
    if np.isnan(diff_max) or diff_max == 0:
        diff_max = 0.1
    plot_update_matrix(diff, 'Stochastic - Deterministic', -diff_max, diff_max, ax=axes[2])
    
    error = matrix_error(stoch_update, det_update)
    fig.suptitle(f'Method Comparison (MSE = {error:.4f})', y=1.02)
    
    plt.tight_layout()
    return fig


def plot_update_matrix_summary(data_matrix: np.ndarray,
                               model_det_matrix: np.ndarray,
                               model_stoch_matrix: Optional[np.ndarray] = None,
                               info: Optional[Dict] = None,
                               figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
    """
    Comprehensive update matrix summary plot.
    
    Shows data, model (deterministic), optionally model (stochastic),
    and differences with key metrics.
    
    Args:
        data_matrix: Data update matrix
        model_det_matrix: Model deterministic update matrix
        model_stoch_matrix: Model stochastic update matrix (optional)
        info: Dict with additional info (midpoints, overall curves, etc.)
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    n_panels = 4 if model_stoch_matrix is not None else 3
    fig, axes = plt.subplots(2, n_panels, figsize=figsize)
    
    # Determine colour scale
    all_matrices = [data_matrix, model_det_matrix]
    if model_stoch_matrix is not None:
        all_matrices.append(model_stoch_matrix)
    
    vmax = max(np.nanmax(np.abs(m)) for m in all_matrices if not np.all(np.isnan(m)))
    if np.isnan(vmax) or vmax == 0:
        vmax = 0.3
    vmin = -vmax
    
    # Top row: update matrices
    plot_update_matrix(data_matrix, 'Data', vmin, vmax, ax=axes[0, 0])
    plot_update_matrix(model_det_matrix, 'Model (Deterministic)', vmin, vmax, ax=axes[0, 1])
    
    if model_stoch_matrix is not None:
        plot_update_matrix(model_stoch_matrix, 'Model (Stochastic)', vmin, vmax, ax=axes[0, 2])
        
        # Model comparison
        diff_models = model_stoch_matrix - model_det_matrix
        diff_max = np.nanmax(np.abs(diff_models))
        if np.isnan(diff_max) or diff_max == 0:
            diff_max = 0.1
        plot_update_matrix(diff_models, 'Stoch - Det', -diff_max, diff_max, ax=axes[0, 3])
    else:
        # Difference: model - data
        diff = model_det_matrix - data_matrix
        diff_max = np.nanmax(np.abs(diff))
        if np.isnan(diff_max) or diff_max == 0:
            diff_max = 0.1
        plot_update_matrix(diff, 'Model - Data', -diff_max, diff_max, ax=axes[0, 2])
    
    # Bottom row: column averages (serial dependence profile)
    # Average update by previous bin position
    n_bins = data_matrix.shape[0]
    bin_indices = np.arange(n_bins) + 1
    
    # Data profile
    ax = axes[1, 0]
    data_profile = np.nanmean(data_matrix, axis=0)
    ax.bar(bin_indices, data_profile, color='steelblue', alpha=0.7)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Previous stimulus bin')
    ax.set_ylabel(r'Mean $\Delta$P(B)')
    ax.set_title('Data: Serial Dependence Profile')
    ax.set_xticks(bin_indices)
    
    # Model deterministic profile
    ax = axes[1, 1]
    det_profile = np.nanmean(model_det_matrix, axis=0)
    ax.bar(bin_indices, det_profile, color='darkorange', alpha=0.7)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Previous stimulus bin')
    ax.set_ylabel(r'Mean $\Delta$P(B)')
    ax.set_title('Model (Det): Serial Dependence Profile')
    ax.set_xticks(bin_indices)
    
    if model_stoch_matrix is not None:
        # Model stochastic profile
        ax = axes[1, 2]
        stoch_profile = np.nanmean(model_stoch_matrix, axis=0)
        ax.bar(bin_indices, stoch_profile, color='forestgreen', alpha=0.7)
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Previous stimulus bin')
        ax.set_ylabel(r'Mean $\Delta$P(B)')
        ax.set_title('Model (Stoch): Serial Dependence Profile')
        ax.set_xticks(bin_indices)
        
        # Profile comparison
        ax = axes[1, 3]
        width = 0.25
        ax.bar(bin_indices - width, data_profile, width, label='Data', color='steelblue', alpha=0.7)
        ax.bar(bin_indices, det_profile, width, label='Model (Det)', color='darkorange', alpha=0.7)
        ax.bar(bin_indices + width, stoch_profile, width, label='Model (Stoch)', color='forestgreen', alpha=0.7)
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Previous stimulus bin')
        ax.set_ylabel(r'Mean $\Delta$P(B)')
        ax.set_title('Profile Comparison')
        ax.set_xticks(bin_indices)
        ax.legend(fontsize=8)
    else:
        # Profile overlay comparison
        ax = axes[1, 2]
        width = 0.35
        ax.bar(bin_indices - width/2, data_profile, width, label='Data', color='steelblue', alpha=0.7)
        ax.bar(bin_indices + width/2, det_profile, width, label='Model', color='darkorange', alpha=0.7)
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Previous stimulus bin')
        ax.set_ylabel(r'Mean $\Delta$P(B)')
        ax.set_title('Profile Comparison')
        ax.set_xticks(bin_indices)
        ax.legend()
    
    # Add error metrics to title
    mse_det = matrix_error(model_det_matrix, data_matrix)
    title_text = f'Update Matrix Analysis (MSE Det = {mse_det:.4f}'
    if model_stoch_matrix is not None:
        mse_stoch = matrix_error(model_stoch_matrix, data_matrix)
        title_text += f', MSE Stoch = {mse_stoch:.4f}'
    title_text += ')'
    fig.suptitle(title_text, y=1.02, fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_eta_effect_on_update_matrix(eta_values: List[float],
                                     stimuli: np.ndarray,
                                     categories: np.ndarray,
                                     base_params: Dict,
                                     burn_in: int = 1000,
                                     n_bins: int = 8,
                                     seed: int = 42,
                                     figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
    """
    Show how learning rate (eta) affects the update matrix.
    
    Higher eta -> stronger serial dependence (larger update matrix values).
    This is key for your research: context learning (high eta) vs inference (low eta).
    
    Args:
        eta_values: List of eta_learning values to compare
        stimuli: Stimulus array
        categories: Category array
        base_params: Dict of other parameters (sigma_percep, A_repulsion, eta_relax)
        burn_in: Burn-in trials
        n_bins: Number of bins
        seed: Random seed
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    from Models.BE_core import BEParams, BEState, BEModel
    from Analysis.update_matrix import compute_model_update_matrix
    
    n_etas = len(eta_values)
    if figsize is None:
        figsize = (4 * n_etas, 4)
    
    fig, axes = plt.subplots(1, n_etas, figsize=figsize)
    if n_etas == 1:
        axes = [axes]
    
    update_matrices = []
    
    for i, eta in enumerate(eta_values):
        # Create params with this eta
        params = BEParams(
            sigma_percep=base_params['sigma_percep'],
            A_repulsion=base_params['A_repulsion'],
            eta_learning=eta,
            eta_relax=base_params['eta_relax']
        )
        
        # Create initial state with burn-in
        initial_state = BEModel.create_initial_state(
            burn_in=burn_in, params=params, seed=seed
        )
        
        # Simulate with history
        rng = np.random.default_rng(seed + i)
        choices, p_B, final_state, history = BEModel.simulate_session(
            params, initial_state, stimuli, categories, rng, return_history=True
        )
        
        # Compute update matrix
        update_mat, _, _ = compute_model_update_matrix(
            history, method='deterministic', n_bins=n_bins
        )
        update_matrices.append(update_mat)
    
    # Determine common colour scale
    vmax = max(np.nanmax(np.abs(m)) for m in update_matrices if not np.all(np.isnan(m)))
    if np.isnan(vmax) or vmax == 0:
        vmax = 0.3
    vmin = -vmax
    
    # Plot each
    for i, (eta, update_mat) in enumerate(zip(eta_values, update_matrices)):
        plot_update_matrix(update_mat, f'$\\eta$ = {eta}', vmin, vmax, ax=axes[i])
    
    fig.suptitle(r'Effect of Learning Rate ($\eta$) on Update Matrix', y=1.02)
    plt.tight_layout()
    return fig
