"""
Stat Trajectory Plotting

plot_trajectory(data, stat_name, ax, **kwargs)

Accepts AnimalData, List[AnimalData], or List[SessionData].
NO FILTERING. Data must be pre-filtered via filter_trials / session.filter.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, TYPE_CHECKING

from behav_utils.plotting.styles import (
    COLOURS, PALETTE, DEFAULT_ALPHA, SEM_ALPHA, DEFAULT_LINE_WIDTH,
)

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData, AnimalData


def plot_trajectory(
    data, stat_name: str, ax=None,
    combine='none',
    color=None, label=None, alpha=DEFAULT_ALPHA,
    linewidth=DEFAULT_LINE_WIDTH, marker='o', markersize=4,
    title='', xlabel='Session', ylabel=None,
) -> Tuple[plt.Figure, plt.Axes, dict]:
    """
    Plot a summary stat across sessions. Data must be pre-filtered.

    Args:
        data: AnimalData, List[AnimalData], or List[SessionData].
        stat_name: Registered stat name ('accuracy', 'pse', etc.).
        combine: For multi-animal:
            'none' — overlay individual animals
            'mean_sem' — cohort mean ± SEM
            'median_iqr' — cohort median ± IQR
            'mean_only' — mean, no error band
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    ylabel = ylabel or stat_name
    animals, session_list = _resolve(data)

    if session_list is not None:
        info = _draw_sessions(session_list, stat_name, ax,
            color=color, label=label, alpha=alpha,
            linewidth=linewidth, marker=marker, markersize=markersize)
    elif len(animals) == 1 or combine == 'none':
        info = _draw_animals(animals, stat_name, ax,
            color=color, label=label, alpha=alpha,
            linewidth=linewidth, marker=marker, markersize=markersize)
    else:
        info = _draw_combined(animals, stat_name, ax, combine,
            color=color, label=label, alpha=alpha, linewidth=linewidth)

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax, info


def _resolve(data):
    from behav_utils.data.structures import SessionData, AnimalData
    if isinstance(data, AnimalData):
        return [data], None
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return [], None
        if isinstance(data[0], AnimalData):
            return list(data), None
        if hasattr(data[0], 'trials'):
            return None, list(data)
    try:
        items = list(data)
        if items and hasattr(items[0], 'trials'):
            return None, items
        if items and hasattr(items[0], 'sessions'):
            return items, None
    except TypeError:
        pass
    raise TypeError(f"Expected AnimalData/List/SessionData, got {type(data).__name__}")


def _get_stat(session, stat_name):
    """Get a scalar stat from one session. No filtering."""
    try:
        result = session.stats([stat_name])
    except Exception:
        return np.nan
    val = result.get(stat_name, np.nan)
    if isinstance(val, dict):
        for k in ['value', 'mean', 'accuracy', stat_name]:
            if k in val:
                return float(val[k])
        return np.nan
    return float(val)


def _draw_sessions(sessions, stat_name, ax, color=None, label=None,
                   alpha=DEFAULT_ALPHA, linewidth=DEFAULT_LINE_WIDTH,
                   marker='o', markersize=4):
    color = color or COLOURS['default']
    vals = [_get_stat(s, stat_name) for s in sessions]
    ax.plot(range(len(vals)), vals, marker=marker, markersize=markersize,
            color=color, lw=linewidth, alpha=alpha, label=label, zorder=2)
    return {'values': np.array(vals), 'n_sessions': len(vals)}


def _draw_animals(animals, stat_name, ax, color=None, label=None,
                  alpha=DEFAULT_ALPHA, linewidth=DEFAULT_LINE_WIDTH,
                  marker='o', markersize=4):
    infos = {}
    for i, animal in enumerate(animals):
        vals = [_get_stat(s, stat_name) for s in animal.sessions]
        c = color or PALETTE[i % len(PALETTE)]
        lbl = label if len(animals) == 1 else getattr(animal, 'animal_id', f'Animal {i}')
        ax.plot(range(len(vals)), vals, marker=marker, markersize=markersize,
                color=c, lw=linewidth, alpha=alpha, label=lbl, zorder=2)
        infos[lbl] = np.array(vals)
    return {'per_animal': infos, 'n_animals': len(animals)}


def _draw_combined(animals, stat_name, ax, combine,
                   color=None, label=None, alpha=DEFAULT_ALPHA,
                   linewidth=DEFAULT_LINE_WIDTH):
    color = color or COLOURS['mean_line']
    all_trajs = [[_get_stat(s, stat_name) for s in a.sessions] for a in animals]
    max_len = max(len(t) for t in all_trajs)
    padded = np.full((len(all_trajs), max_len), np.nan)
    for i, t in enumerate(all_trajs):
        padded[i, :len(t)] = t

    x = np.arange(max_len)
    n_valid = np.sum(~np.isnan(padded), axis=0)
    mask = n_valid >= 2

    with np.errstate(all='ignore'):
        if combine in ('mean_sem', 'mean_only'):
            centre = np.nanmean(padded, axis=0)
            err = np.nanstd(padded, axis=0, ddof=1) / np.sqrt(n_valid)
        elif combine == 'median_iqr':
            centre = np.nanmedian(padded, axis=0)
            q25 = np.nanpercentile(padded, 25, axis=0)
            q75 = np.nanpercentile(padded, 75, axis=0)
        else:
            raise ValueError(f"Unknown combine: {combine!r}")

    ax.plot(x[mask], centre[mask], '-', color=color, lw=linewidth * 1.5,
            alpha=alpha, label=label or combine.replace('_', ' '), zorder=3)

    if combine == 'mean_sem':
        ax.fill_between(x[mask], (centre-err)[mask], (centre+err)[mask],
                        color=color, alpha=SEM_ALPHA, zorder=1)
    elif combine == 'median_iqr':
        ax.fill_between(x[mask], q25[mask], q75[mask],
                        color=color, alpha=SEM_ALPHA, zorder=1)

    return {'centre': centre, 'n_animals': len(animals), 'padded': padded}
