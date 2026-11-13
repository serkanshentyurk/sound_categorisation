"""
plotting/daily_tracker.py — draw-only single-panel plotters for the daily tracker.

Contract: every plot_x(data, …, ax=None) draws on one Axes and does no analysis; the numbers come from
behaviour.daily_tracker.DailyTracker.

    plot_daily_values        <- tracker.sessions (one animal): values against date, gaps for days off
    plot_condition_curves    <- a CurveSet: session curves faint, their mean dashed, the pooled fit solid
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from behav_utils.plotting import plot_psychometric_curve

_COLOURS = ('#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b')
LIGHT_COLOURS = {'no light': '#1f77b4', 'light': '#d62728'}


def plot_daily_values(sessions: pd.DataFrame, animal: str, stats: Sequence[str], ax: plt.Axes | None = None, *,
                      labels: Sequence[str] | None = None, ylabel: str = '', hline: float | None = None,
                      ylim: tuple | None = None, changes: Sequence[tuple] | None = None) -> plt.Axes:
    """One line per statistic against date for one animal; days without a value leave a gap.

    ``changes``: (date, label) pairs drawn as dashed lines, e.g. the days the preset changed.
    """
    ax = ax or plt.gca()
    d = sessions[sessions['animal'] == animal].sort_values('date')
    x = pd.to_datetime(d['date'])
    for k, stat in enumerate(stats):
        y = pd.to_numeric(d[stat], errors='coerce') if stat in d else pd.Series(np.nan, index=d.index)
        ax.plot(x, y, marker='o', markersize=3, linewidth=1.2, color=_COLOURS[k % len(_COLOURS)],
                label=labels[k] if labels else stat)
    if hline is not None:
        ax.axhline(hline, color='0.6', linewidth=0.8, linestyle=':')
    if ylim is not None:
        ax.set_ylim(*ylim)
    for when, text in changes or ():
        ax.axvline(pd.to_datetime(when), color='0.7', linewidth=0.8, linestyle='--', zorder=0)
        ax.annotate(text, (pd.to_datetime(when), 1.0), xycoords=('data', 'axes fraction'), rotation=90,
                    fontsize=6, color='0.45', ha='right', va='top')
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', labelrotation=45, labelsize=7)
    if len(x):      # the axis is the window itself, whatever has a value: a lone session cannot stretch it
        half = pd.Timedelta(hours=12)
        ax.set_xlim(x.min() - half, x.max() + half)
        span = (x.max() - x.min()).days
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, -(-span // 7))))   # whole days only
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    if len(stats) > 1 or labels:
        ax.legend(fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    return ax


def plot_condition_curves(curves, ax: plt.Axes | None = None, *, color: str = '#1f77b4',
                          label: str | None = None) -> plt.Axes:
    """A condition's psychometrics on one panel: each session's fit faint, their point-wise mean dashed,
    and the fit to the pooled trials ("mega session") solid with its binned data."""
    ax = ax or plt.gca()
    for _, c in curves.sessions:
        if c.success:
            plot_psychometric_curve(c, ax, color=color, alpha=0.25, linewidth=0.8, show_data=False,
                                    show_band=False)
    if curves.mean_y is not None:
        ax.plot(curves.x, curves.mean_y, color=color, linestyle='--', linewidth=1.2)
    if curves.pooled is not None and curves.pooled.success:
        plot_psychometric_curve(curves.pooled, ax, color=color, label=label, linewidth=2.0, show_band=False)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('stimulus')
    ax.set_ylabel('P(choose B)')
    ax.spines[['top', 'right']].set_visible(False)
    return ax
