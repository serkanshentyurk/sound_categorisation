"""
SBI Validation Plotting

Posterior predictive checks and parameter-statistic relationship plots.

Usage:
    from plotting.sbi_validation import (
        plot_summary_stats_comparison,
        plot_param_stat_correlations,
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


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
        observed: (n_stats,) observed summary stats.
        simulated: (n_sims, n_stats) posterior predictive stats.
        stat_names: Names for each stat.
        n_per_session: If set, groups stats by session.
        figsize: Figure size.
        title: Overall title.

    Returns:
        Matplotlib figure.
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
            ax.set_xticklabels(base_names, rotation=45, ha='right',
                               fontsize=7)
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
# PARAMETER-STAT CORRELATIONS
# =============================================================================

def plot_param_stat_correlations(
    corr_data: dict,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of parameter–summary stat correlations.

    Args:
        corr_data: Dict with 'corr_matrix', 'param_names',
                   'stat_names_expanded'.
        title: Optional title.

    Returns:
        Matplotlib figure.
    """
    corr = corr_data['corr_matrix']
    pnames = corr_data['param_names']
    snames = corr_data['stat_names_expanded']

    fig, ax = plt.subplots(
        figsize=(max(8, len(snames) * 0.8), len(pnames) * 1.2 + 1))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    for i in range(len(pnames)):
        for j in range(len(snames)):
            val = corr[i, j]
            colour = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=colour)

    ax.set_xticks(range(len(snames)))
    ax.set_xticklabels(snames, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(pnames)))
    ax.set_yticklabels(pnames, fontsize=10)

    plt.colorbar(im, ax=ax, label='Correlation', shrink=0.8)
    ax.set_title(title or 'Parameter–Summary Stat Correlations',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig
