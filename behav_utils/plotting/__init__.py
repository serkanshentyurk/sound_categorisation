"""
behav_utils.plotting — Behavioural data visualisation.

Three core plotting functions:
    plot_psychometric   — Psychometric curves (single, pooled, overlay, session mean)
    plot_um             — Update matrix heatmaps
    plot_trajectory     — Summary stat trajectories across sessions

All accept SessionData, List[SessionData], or AnimalData.
All draw on a user-provided axes (create one if ax=None).
User controls layout via plt.subplots().

Usage:
    from behav_utils.plotting import plot_psychometric, plot_um, plot_trajectory
    from behav_utils.plotting.styles import PALETTE, UM_CMAP, apply_style

    apply_style()

    fig, axes = plt.subplots(1, 2)
    plot_psychometric(early_sessions, ax=axes[0], color=PALETTE[0], label='Early')
    plot_psychometric(late_sessions, ax=axes[1], color=PALETTE[1], label='Late')
"""

from behav_utils.plotting.psychometric import plot_psychometric
from behav_utils.plotting.update_matrix import plot_um
from behav_utils.plotting.trajectory import plot_trajectory
from behav_utils.plotting.styles import (
    PALETTE, COLOURS, UM_CMAP,
    apply_style, get_colour,
    get_animal_colours, get_session_colours, get_bin_colours,
)

__all__ = [
    # Core plotting
    'plot_psychometric',
    'plot_um',
    'plot_trajectory',
    # Styles
    'PALETTE',
    'COLOURS',
    'UM_CMAP',
    'apply_style',
    'get_colour',
    'get_animal_colours',
    'get_session_colours',
    'get_bin_colours',
]
