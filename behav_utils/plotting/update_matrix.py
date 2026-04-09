"""
Update Matrix Plotting

Heatmaps, serial dependence profiles, conditional psychometrics.
All functions return (fig, ax) or (fig, axes).

Usage:
    from behav_utils.plotting.update_matrix import (
        plot_update_matrix,
        plot_update_matrix_comparison,
        plot_phase_update_matrices,
        plot_sd_profile,
        plot_conditional_psychometrics,
        plot_update_matrix_summary,
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Union

from behav_utils.plotting.styles import UM_CMAP, get_bin_colours, COLOURS
from behav_utils.analysis.utils import cumulative_gaussian


# =============================================================================
# SINGLE UPDATE MATRIX
# =============================================================================

def plot_update_matrix(
    update_matrix: np.ndarray,
    title: str = 'Update Matrix',
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    cmap=None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot update matrix heatmap.

    Args:
        update_matrix: (n_bins, n_bins) array
        title: Plot title
        ax: Existing axes
        vmin, vmax: Colour limits (default: symmetric around 0)
        show_colorbar: Whether to add colourbar
        cmap: Colourmap (default: UM_CMAP)

    Returns:
        (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.get_figure()

    if cmap is None:
        cmap = UM_CMAP

    if vmax is None:
        vmax = np.nanmax(np.abs(update_matrix))
        if np.isnan(vmax) or vmax == 0:
            vmax = 0.3
    if vmin is None:
        vmin = -vmax

    n_bins = update_matrix.shape[0]

    im = ax.imshow(
        update_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
        origin='lower', aspect='equal',
    )

    ax.set_xlabel('Previous stimulus bin')
    ax.set_ylabel('Current stimulus bin')
    ax.set_title(title)

    # Tick labels at edges and centre
    ax.set_xticks([0, (n_bins - 1) / 2, n_bins - 1])
    ax.set_xticklabels(['-1', '0', '1'])
    ax.set_yticks([0, (n_bins - 1) / 2, n_bins - 1])
    ax.set_yticklabels(['-1', '0', '1'])

    if show_colorbar:
        plt.colorbar(im, ax=ax, label='\u0394P(B)')

    return fig, ax


# =============================================================================
# UPDATE MATRIX COMPARISON (data vs model, 2 panels + difference)
# =============================================================================

def plot_update_matrix_comparison(
    data_matrix: np.ndarray,
    model_matrix: np.ndarray,
    data_title: str = 'Data',
    model_title: str = 'Model',
    suptitle: str = '',
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Side-by-side data vs model update matrices with difference.

    Returns:
        (fig, axes) where axes is (1, 3) array
    """
    diff = data_matrix - model_matrix
    vmax = max(
        np.nanmax(np.abs(data_matrix)),
        np.nanmax(np.abs(model_matrix)),
        np.nanmax(np.abs(diff)),
    )
    if np.isnan(vmax) or vmax == 0:
        vmax = 0.3

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, um, title in [
        (axes[0], data_matrix, data_title),
        (axes[1], model_matrix, model_title),
        (axes[2], diff, f'Difference ({data_title} \u2212 {model_title})'),
    ]:
        plot_update_matrix(um, title=title, ax=ax, vmin=-vmax, vmax=vmax,
                           show_colorbar=True)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()

    return fig, axes


# =============================================================================
# N-PHASE UPDATE MATRIX COMPARISON
# =============================================================================

def plot_phase_update_matrices(
    phase_matrices: Dict[str, np.ndarray],
    suptitle: str = '',
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    figsize_per_panel: Tuple[float, float] = (3.5, 3.5),
    cmap=None,
    annotations: Optional[Dict[str, str]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot update matrices for multiple phases side-by-side.

    Shared colour scale across all panels for direct comparison.

    Args:
        phase_matrices: Dict of {label: (n_bins, n_bins) array}.
            Insertion order determines panel order.
        suptitle: Figure title
        vmax: Symmetric colour limit. If None, computed from data.
        show_colorbar: Add shared colourbar
        figsize_per_panel: (width, height) per panel
        cmap: Colourmap (default: UM_CMAP)
        annotations: Optional dict {label: annotation_string} shown
            below each panel (e.g., trial counts, session info)

    Returns:
        (fig, axes) where axes is 1D array

    Usage:
        from behav_utils.analysis.update_matrix import (
            compute_update_matrix_from_sessions,
        )

        phases = {
            'Baseline (expert)': compute_update_matrix_from_sessions(baseline[-5:])[0],
            'Early post-shift':  compute_update_matrix_from_sessions(post[:3])[0],
            'Late post-shift':   compute_update_matrix_from_sessions(post[-3:])[0],
        }
        plot_phase_update_matrices(phases, suptitle='SS05: Distribution Shift')
    """
    if cmap is None:
        cmap = UM_CMAP

    labels = list(phase_matrices.keys())
    n = len(labels)

    if vmax is None:
        all_vals = [np.nanmax(np.abs(m)) for m in phase_matrices.values()
                    if not np.all(np.isnan(m))]
        vmax = max(all_vals) if all_vals else 0.3
        if np.isnan(vmax) or vmax == 0:
            vmax = 0.3

    from matplotlib.gridspec import GridSpec

    # GridSpec: n columns for panels + 1 narrow column for colorbar
    if show_colorbar:
        width_ratios = [1] * n + [0.05]
        gs = GridSpec(1, n + 1, width_ratios=width_ratios, wspace=0.25)
        fig = plt.figure(figsize=(figsize_per_panel[0] * n + 1.5,
                                  figsize_per_panel[1]))
        axes = np.array([fig.add_subplot(gs[0, i]) for i in range(n)])
        cbar_ax = fig.add_subplot(gs[0, n])
    else:
        fig, axes = plt.subplots(
            1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
        )
        cbar_ax = None
    if n == 1:
        axes = np.array([axes])

    im = None
    for idx, (ax, label) in enumerate(zip(axes, labels)):
        um = phase_matrices[label]
        n_bins = um.shape[0]

        im_obj = ax.imshow(
            um, cmap=cmap, vmin=-vmax, vmax=vmax,
            origin='lower', aspect='equal',
        )
        ax.set_title(label, fontsize=9, pad=4)
        ax.set_xticks([0, (n_bins - 1) / 2, n_bins - 1])
        ax.set_xticklabels(['-1', '0', '1'], fontsize=8)
        ax.set_yticks([0, (n_bins - 1) / 2, n_bins - 1])

        if idx == 0:
            ax.set_yticklabels(['-1', '0', '1'], fontsize=8)
            ax.set_ylabel('Current stim bin', fontsize=8)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')
        ax.set_xlabel('')

        im = im_obj

        if annotations and label in annotations:
            ax.annotate(
                annotations[label], xy=(0.5, -0.12),
                xycoords='axes fraction', ha='center', va='top',
                fontsize=7, color='grey',
            )

    if show_colorbar and im is not None and cbar_ax is not None:
        fig.colorbar(im, cax=cbar_ax, label='\u0394P(B)')

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight='bold')

    return fig, axes


# =============================================================================
# SERIAL DEPENDENCE PROFILE
# =============================================================================

def plot_sd_profile(
    update_matrix: np.ndarray,
    title: str = 'Serial Dependence Profile',
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot mean column values of update matrix (serial dependence profile).

    Returns:
        (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    n_bins = update_matrix.shape[1]
    profile = np.nanmean(update_matrix, axis=0)
    bin_indices = np.arange(n_bins) + 1
    colours = get_bin_colours(n_bins)

    if color is not None:
        ax.bar(bin_indices, profile, color=color, edgecolor='black',
               linewidth=0.5, label=label)
    else:
        ax.bar(bin_indices, profile, color=colours, edgecolor='black',
               linewidth=0.5, label=label)

    ax.axhline(0, color='k', alpha=0.3)
    ax.set_xlabel('Previous stimulus bin')
    ax.set_ylabel('Mean \u0394P(B)')
    ax.set_title(title)
    ax.set_xticks(bin_indices)

    if label is not None:
        ax.legend()

    return fig, ax


# =============================================================================
# CONDITIONAL PSYCHOMETRICS
# =============================================================================

def plot_conditional_psychometrics(
    conditional_matrix: np.ndarray,
    info: Optional[Dict] = None,
    title: str = 'Conditional Psychometrics',
    ax: Optional[plt.Axes] = None,
    show_overall: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot conditional psychometric curves (one per previous-stimulus bin).

    Args:
        conditional_matrix: (n_bins, n_bins) — rows = current stim, cols = prev stim
        info: Dict from compute_update_matrix (for overall curve + conditional fits)
        title: Plot title
        ax: Existing axes
        show_overall: Plot the overall (unconditional) curve

    Returns:
        (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    n_bins = conditional_matrix.shape[0]
    midpoints = np.linspace(-0.875, 0.875, n_bins)
    if info is not None:
        midpoints = info.get('midpoints', midpoints)

    colours = get_bin_colours(n_bins)

    # Overall curve
    if show_overall:
        if info is not None and 'total_psychometric' in info:
            tp = info['total_psychometric']
            if tp.get('success', False):
                x_fine = np.linspace(-1.1, 1.1, 200)
                y_total = cumulative_gaussian(
                    x_fine, tp['mu'], tp['sigma'],
                    tp['lapse_low'], tp['lapse_high'],
                )
                ax.plot(x_fine, y_total, 'k-', linewidth=2.5,
                        label='Overall', zorder=10)
        else:
            # Fallback: mean of conditional curves
            overall = np.nanmean(conditional_matrix, axis=1)
            ax.plot(midpoints, overall, 'k-', linewidth=2.5,
                    label='Overall', zorder=10)

    # Conditional curves
    if info is not None and 'conditional_psychometrics' in info:
        x_fine = np.linspace(-1.1, 1.1, 200)
        for j, cond_psych in enumerate(info['conditional_psychometrics']):
            if cond_psych is None or not cond_psych.get('success', False):
                continue
            y_cond = cumulative_gaussian(
                x_fine, cond_psych['mu'], cond_psych['sigma'],
                cond_psych['lapse_low'], cond_psych['lapse_high'],
            )
            ax.plot(x_fine, y_cond, '-', color=colours[j], linewidth=1.2,
                    alpha=0.7, label=f'Prev bin {j} ({midpoints[j]:.2f})')
    else:
        # Plot from matrix directly
        for j in range(n_bins):
            curve = conditional_matrix[:, j]
            if not np.all(np.isnan(curve)):
                ax.plot(midpoints, curve, '-', color=colours[j],
                        linewidth=1.2, alpha=0.7,
                        label=f'Prev bin {j}')

    ax.axhline(0.5, color='grey', ls='--', alpha=0.3)
    ax.axvline(0, color='grey', ls='--', alpha=0.3)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Current stimulus')
    ax.set_ylabel('P(choose B)')
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc='lower right')

    return fig, ax


# =============================================================================
# UPDATE MATRIX SUMMARY (matrix + profile + conditional in one figure)
# =============================================================================

def plot_update_matrix_summary(
    update_matrix: np.ndarray,
    conditional_matrix: np.ndarray,
    info: Optional[Dict] = None,
    title: str = '',
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Three-panel summary: update matrix, SD profile, conditional psychometrics.

    Returns:
        (fig, axes)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    plot_update_matrix(update_matrix, title='Update Matrix', ax=axes[0])
    plot_sd_profile(update_matrix, title='SD Profile', ax=axes[1])
    plot_conditional_psychometrics(
        conditional_matrix, info=info,
        title='Conditional Psychometrics', ax=axes[2],
    )

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()

    return fig, axes
