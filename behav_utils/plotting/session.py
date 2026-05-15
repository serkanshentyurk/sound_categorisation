"""
Session-Level Trial Plots

Trial rasters, within-session choice/stimulus visualisation.
All functions return (fig, ax).

Usage:
    from behav_utils.plotting.session import plot_session_trials

    fig, ax = plot_session_trials(session)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Optional, Tuple, TYPE_CHECKING

from behav_utils.plotting.styles import COLOURS

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData

def plot_session_raster(
    result: dict,
    ax: Optional[plt.Axes] = None,
    window: Optional[int] = None,
    title: str = '',
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot trial-by-trial raster from a compute_session_raster() result.

    Shows stimulus values as a line, correct/incorrect as green/red markers,
    and no-response trials as grey.

    Args:
        result: Dict from compute_session_raster().
        ax: Matplotlib axes (creates one if None).
        window: Rolling accuracy window size (None = no rolling line).
        title: Axes title.

    Returns:
        (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    else:
        fig = ax.get_figure()

    n = result['n_trials']
    trials = np.arange(n)
    stimuli = result['stimuli']
    correct = result['correct']
    no_resp = result['no_response']

    # Stimulus trace
    ax.plot(trials, stimuli, color='grey', alpha=0.3, lw=0.5, zorder=1)

    # Correct / incorrect / no-response markers
    corr_mask = correct & ~no_resp
    incorr_mask = ~correct & ~no_resp
    ax.scatter(trials[corr_mask], stimuli[corr_mask],
               c='#60BD68', s=8, zorder=2, label='Correct')
    ax.scatter(trials[incorr_mask], stimuli[incorr_mask],
               c='#E24A33', s=8, zorder=2, label='Incorrect')
    if no_resp.any():
        ax.scatter(trials[no_resp], stimuli[no_resp],
                   c='#AAAAAA', s=6, marker='x', zorder=2, label='No response')

    # Category boundary
    ax.axhline(0, ls='--', color='black', alpha=0.3, lw=0.8)

    # Rolling accuracy
    if window and n >= window:
        valid = ~no_resp
        rolling = np.convolve(correct[valid].astype(float),
                              np.ones(window) / window, mode='valid')
        x_roll = np.arange(window - 1, valid.sum())
        ax2 = ax.twinx()
        ax2.plot(x_roll, rolling, color='steelblue', alpha=0.6, lw=1.5)
        ax2.set_ylabel('Rolling accuracy', color='steelblue')
        ax2.set_ylim(0, 1.05)
        ax2.axhline(0.5, ls=':', color='steelblue', alpha=0.3)

    ax.set_xlabel('Trial')
    ax.set_ylabel('Stimulus')
    ax.set_xlim(0, n)

    if not title:
        title = result.get('session_id', '')
    ax.set_title(title)

    return fig, ax