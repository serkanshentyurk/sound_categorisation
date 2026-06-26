"""Cohort-level overview plots for the sound-categorisation task."""
from typing import Literal, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from behav_utils.data.structures import AnimalData
from behav_utils.analysis.summary_stats import list_available_stats

from plotting._style import DIST_COLOURS, TYPE_MARKERS


def plot_timeline(animal: AnimalData, stat: str = 'accuracy',
                  x_axis: Literal['date', 'session_idx'] = 'date',
                  ylim: Optional[tuple] = None, hline: Optional[float] = None,
                  ax: Optional[plt.Axes] = None, fig_size=(12, 3)):
    """Scatter a per-session statistic over time for one animal.

    Colour encodes distribution, marker encodes session type. Takes an ``AnimalData``
    object directly (no reliance on a notebook-global ``experiment``).
    """
    if stat not in list_available_stats():
        raise ValueError(f"stat '{stat}' not available. Choose from: {list_available_stats()}")
    if x_axis not in ('date', 'session_idx'):
        raise ValueError("x_axis must be 'date' or 'session_idx'.")

    table = animal.session_table

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    for _, row in table.iterrows():
        ax.scatter(row[x_axis], row[stat],
                   c=DIST_COLOURS.get(row['distribution'], 'grey'),
                   marker=TYPE_MARKERS.get(row['session_type'], 'o'),
                   s=60, edgecolors='k', linewidths=0.5, zorder=3)

    ax.set_xlabel('Session date' if x_axis == 'date' else 'Session index')
    ax.set_ylabel(stat)
    ax.set_ylim(ylim)
    if hline is not None:
        ax.axhline(hline, ls='--', c='grey', lw=0.5)

    handles = ([Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=d)
                for d, c in DIST_COLOURS.items()] +
               [Line2D([0], [0], marker=m, color='w', markerfacecolor='grey', markersize=8, label=t)
                for t, m in TYPE_MARKERS.items()])
    ax.legend(handles=handles, ncol=4, fontsize=8, loc='lower right')

    fig.suptitle(f'{animal.animal_id} ({animal.genotype.upper()}) — Session Timeline', fontsize=14)
    fig.tight_layout()
    return fig
