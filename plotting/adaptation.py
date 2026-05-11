"""
Adaptation / shift analysis plotting.

Visualisations for distribution shift responses: per-animal
trajectory plots, UM evolution, psychometric comparison across
phases, and group-level trajectory heatmaps.

Extracted from NB 30 inline code. Delegates to behav_utils
plotting functions where possible.

Public API:
    plot_animal_trajectory      — Per-stat trajectory around a shift
    plot_shift_um_evolution     — UMs across expert → early post → late post
    plot_shift_psychometric     — Psychometric overlay across phases
    plot_group_trajectories     — Group-mean trajectory with SEM

Usage:
    from analysis.adaptation import (
        detect_all_manipulations, adaptation_trajectory, aggregate_trajectories,
    )
    from plotting.adaptation import plot_animal_trajectory, plot_group_trajectories
"""

from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

from behav_utils.analysis.update_matrix import compute_update_matrix_from_sessions
from behav_utils.plotting.update_matrix import plot_phase_update_matrices
from behav_utils.plotting.psychometric import plot_psychometric_overlay
from behav_utils.plotting.styles import COLOURS, apply_style

apply_style()

# ─── Colours ─────────────────────────────────────────────────────────────────

BASELINE_COLOUR = 'steelblue'
POST_COLOUR = 'darkorange'


# ─── Per-animal trajectory ───────────────────────────────────────────────────

def plot_animal_trajectory(
    trajectory_df,
    stats: List[str],
    shift_info: Optional[Dict] = None,
    animal_id: Optional[str] = None,
    n_cols: int = 3,
    figsize_per_panel: tuple = (5, 3.5),
) -> plt.Figure:
    """
    Plot per-stat trajectory around a distribution shift for one animal.

    Args:
        trajectory_df: DataFrame from adaptation_trajectory() with columns
            'relative_session', 'phase', stat columns, 'baseline_{stat}_mean/std'.
        stats: List of stat names to plot.
        shift_info: Dict with 'details' and 'shift_type' (from detect_all_manipulations).
        animal_id: Animal identifier for title.
        n_cols: Max columns in subplot grid.
    """
    n_stats = len(stats)
    n_cols = min(n_cols, n_stats)
    n_rows = int(np.ceil(n_stats / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharex=True,
    )
    axes = np.atleast_2d(axes)

    for idx, stat in enumerate(stats):
        if stat not in trajectory_df.columns:
            continue
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        bl_mask = trajectory_df['phase'] == 'baseline'
        post_mask = trajectory_df['phase'] == 'post'

        ax.plot(
            trajectory_df.loc[bl_mask, 'relative_session'],
            trajectory_df.loc[bl_mask, stat],
            'o-', ms=4, color=BASELINE_COLOUR,
        )
        ax.plot(
            trajectory_df.loc[post_mask, 'relative_session'],
            trajectory_df.loc[post_mask, stat],
            'o-', ms=4, color=POST_COLOUR,
        )

        # Baseline band
        bl_mean_col = f'baseline_{stat}_mean'
        bl_std_col = f'baseline_{stat}_std'
        if bl_mean_col in trajectory_df.columns:
            bl_mean = trajectory_df[bl_mean_col].iloc[0]
            bl_std = trajectory_df[bl_std_col].iloc[0]
            if not np.isnan(bl_mean):
                ax.axhline(bl_mean, color=BASELINE_COLOUR, ls='--', lw=0.8, alpha=0.5)
                ax.axhspan(
                    bl_mean - bl_std, bl_mean + bl_std,
                    color=BASELINE_COLOUR, alpha=0.08,
                )

        ax.axvline(0, color='black', ls=':', lw=0.8)
        ax.set_title(stat, fontsize=10)

    # Hide unused panels
    for idx in range(n_stats, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    axes[-1, 0].set_xlabel('Sessions relative to shift')

    # Suptitle
    title_parts = []
    if animal_id:
        title_parts.append(animal_id)
    if shift_info:
        details = shift_info.get('details', {})
        title_parts.append(
            f"{details.get('before', '?')} → {details.get('after', '?')}"
        )
        title_parts.append(f"({shift_info.get('shift_type', '')})")
    if title_parts:
        fig.suptitle(' '.join(title_parts), fontsize=12, fontweight='bold')

    fig.tight_layout()
    return fig


# ─── UM evolution across phases ──────────────────────────────────────────────

def plot_shift_um_evolution(
    baseline_sessions: list,
    post_sessions: list,
    n_baseline: int = 5,
    n_early_post: int = 3,
    animal_id: Optional[str] = None,
    shift_info: Optional[Dict] = None,
) -> Optional[plt.Figure]:
    """
    Plot update matrices across expert → early post → late post.

    Delegates to behav_utils.plotting.update_matrix.plot_phase_update_matrices.

    Args:
        baseline_sessions: Expert-phase sessions.
        post_sessions: Post-shift sessions.
        n_baseline: Number of baseline sessions to pool.
        n_early_post: Number of early post-shift sessions.
    """
    phases = {}

    bl = baseline_sessions[-n_baseline:] if len(baseline_sessions) >= n_baseline \
        else baseline_sessions
    um, _, info = compute_update_matrix_from_sessions(bl, method='pool')
    phases[f'Expert\n({info["n_sessions"]}s)'] = um

    n_early = min(n_early_post, len(post_sessions))
    if n_early >= 2:
        um, _, info = compute_update_matrix_from_sessions(
            post_sessions[:n_early], method='pool')
        phases[f'Early post\n({info["n_sessions"]}s)'] = um

    if len(post_sessions) > n_early_post:
        late = post_sessions[n_early_post:]
        um, _, info = compute_update_matrix_from_sessions(late, method='pool')
        phases[f'Late post\n({info["n_sessions"]}s)'] = um

    if len(phases) < 2:
        return None

    title_parts = []
    if animal_id:
        title_parts.append(animal_id)
    if shift_info:
        details = shift_info.get('details', {})
        title_parts.append(
            f"{details.get('before', '?')} → {details.get('after', '?')}"
        )

    fig, axes = plot_phase_update_matrices(
        phases, suptitle=': '.join(title_parts) if title_parts else None)
    fig.tight_layout()
    return fig


# ─── Psychometric overlay across phases ──────────────────────────────────────

def plot_shift_psychometric(
    baseline_sessions: list,
    post_sessions: list,
    n_baseline: int = 5,
    animal_id: Optional[str] = None,
    shift_info: Optional[Dict] = None,
    n_bootstrap: int = 200,
) -> Optional[plt.Figure]:
    """
    Psychometric curves overlaid across shift phases.

    Delegates to behav_utils.plotting.psychometric.plot_psychometric_overlay.
    """
    groups = {'Expert baseline': baseline_sessions[-n_baseline:]}

    if len(post_sessions) >= 3:
        groups['Early post (1–3)'] = post_sessions[:3]
    if len(post_sessions) >= 6:
        groups['Mid post'] = post_sessions[3:min(6, len(post_sessions))]
    if len(post_sessions) > 6:
        groups['Late post'] = post_sessions[-3:]

    if len(groups) < 2:
        return None

    title_parts = []
    if animal_id:
        title_parts.append(animal_id)
    if shift_info:
        details = shift_info.get('details', {})
        title_parts.append(
            f"{details.get('before', '?')} → {details.get('after', '?')}"
        )

    fig, ax, infos = plot_psychometric_overlay(
        groups, mode='pooled', n_bootstrap=n_bootstrap, show_ci=True,
        title=': '.join(title_parts) if title_parts else None)
    return fig


# ─── Group-level trajectory ─────────────────────────────────────────────────

def plot_group_trajectories(
    aggregated_df,
    stats: List[str],
    n_animals: Optional[int] = None,
    n_cols: int = 3,
    figsize_per_panel: tuple = (5, 3.5),
) -> plt.Figure:
    """
    Plot group-mean trajectory with SEM error bars.

    Args:
        aggregated_df: DataFrame from aggregate_trajectories() with columns
            'relative_session', 'stat', 'mean', 'sem'.
        stats: List of stat names to plot.
        n_animals: For title annotation.
    """
    n_stats = len(stats)
    n_cols = min(n_cols, n_stats)
    n_rows = int(np.ceil(n_stats / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharex=True,
    )
    axes = np.atleast_2d(axes)

    for idx, stat in enumerate(stats):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        stat_data = aggregated_df[
            aggregated_df['stat'] == stat
        ].sort_values('relative_session')
        if stat_data.empty:
            ax.set_title(f'{stat} (no data)', fontsize=10)
            continue

        x = stat_data['relative_session'].values
        y = stat_data['mean'].values
        yerr = stat_data['sem'].values

        bl_mask = x < 0
        post_mask = x >= 0

        ax.errorbar(
            x[bl_mask], y[bl_mask], yerr=yerr[bl_mask],
            fmt='o-', ms=4, color=BASELINE_COLOUR, capsize=2,
        )
        ax.errorbar(
            x[post_mask], y[post_mask], yerr=yerr[post_mask],
            fmt='o-', ms=4, color=POST_COLOUR, capsize=2,
        )
        ax.axvline(0, color='black', ls=':', lw=0.8)
        ax.set_title(stat, fontsize=10)

    for idx in range(n_stats, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    axes[-1, 0].set_xlabel('Sessions relative to shift')

    title = 'Group mean'
    if n_animals is not None:
        title += f' (n={n_animals} animals)'
    fig.suptitle(title, fontsize=12, fontweight='bold')

    fig.tight_layout()
    return fig
