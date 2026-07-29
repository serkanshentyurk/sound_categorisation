"""
plot_session_stats — per-session values across groups, the "eyeball" plot.

The qualitative companion to every pooled result. A pooled estimate with a
bootstrap CI can hide the fact that it rests on one atypical session; this plot
shows the per-session spread so you can see that, and — because every point is
labelled with its session id — decide which session to omit and re-run.

It is deliberately descriptive, not inferential. With a handful of sessions per
animal the spread is a handful of numbers; do not read a test into it. The
inferential claim lives at the trial level (within animal) or the animal level
(across animals), never the session level — 4–6 sessions cannot support a test.

    from behav_utils.analysis import compute_stat
    from behav_utils.plotting import plot_session_stats, plot_session_stats_single

    # two conditions from one phase
    opto = compute_stat(cond_opto,     stats, mode='per_session', animal_id='SS15')
    ctrl = compute_stat(cond_non_opto, stats, mode='per_session', animal_id='SS15')
    plot_session_stats({'opto': opto['sessions'], 'non_opto': ctrl['sessions']})

    # one stat into an axis you own (paper figures):
    plot_session_stats_single({'opto': opto['sessions'], 'non_opto': ctrl['sessions']},
                              'side_bias', ax=ax)

Input is ``{group_label: sessions_frame}`` where each frame is the ``sessions``
field of a ``compute_stat(mode='per_session')`` result (columns: animal,
session, stat, value, n_trials). Sessions shared across groups by id are joined
with a faint line, so the within-session direction — the readable signal in a
paired design — is visible rather than hidden behind the marginal spread.
"""

from typing import Mapping, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from behav_utils.plotting.styles import get_colour

__all__ = ['plot_session_stats', 'plot_session_stats_single']


def plot_session_stats_single(
    groups: Mapping[str, 'object'],
    stat: str,
    ax: Optional[plt.Axes] = None,
    group_order: Optional[Sequence[str]] = None,
    palette: Optional[Sequence[str]] = None,
    connect: bool = True,
    show_session_ids: bool = True,
    centre: str = 'mean',
    jitter: float = 0.06,
    seed: int = 0,
) -> plt.Axes:
    """One stat's per-session values by group, drawn on a single Axes.

    The single-panel core of :func:`plot_session_stats`; use it to place one
    stat in a figure you lay out yourself. ``plot_session_stats`` calls this
    once per stat to build its grid. Descriptive, not inferential — see the
    module docstring.

    Args:
        groups:      ``{group_label: sessions_frame}``. Each frame is the
                     ``sessions`` field of ``compute_stat(mode='per_session')``.
        stat:        the stat to draw.
        ax:          Axes to draw on; a new one is made if None.
        group_order: x-axis order of the groups; default is dict order.
        palette:     colours per group; defaults to the house palette.
        connect:     join points sharing a session id across adjacent groups.
        show_session_ids: annotate each point with its session id.
        centre:      'mean' or 'median' — the session-level centre bar (NOT the
                     pooled estimate; a visual summary of spread only).
        jitter:      horizontal scatter half-width, in x-units.
        seed:        RNG seed for the jitter.

    Returns:
        the Axes drawn on.

    Raises:
        ValueError: if there are no non-empty group frames.
    """
    frames = {k: v for k, v in groups.items() if v is not None and len(v)}
    if not frames:
        raise ValueError("plot_session_stats_single: no non-empty group frames")

    order = list(group_order) if group_order else list(frames)
    order = [g for g in order if g in frames]
    colours = list(palette) if palette else [get_colour(i) for i in range(len(order))]

    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 3.0))
    rng = np.random.default_rng(seed)

    # per group: session ids, values, jittered x — reused for the lines
    xs, ys, ids = {}, {}, {}
    for i, group in enumerate(order):
        frame = frames[group]
        rows = frame[frame['stat'] == stat]
        if rows.empty:
            continue
        values = rows['value'].to_numpy(dtype=float)
        session_ids = rows['session'].to_numpy()
        keep = np.isfinite(values)
        values, session_ids = values[keep], session_ids[keep]
        if not values.size:
            continue
        xs[group] = i + (rng.random(values.size) - 0.5) * 2 * jitter
        ys[group] = values
        ids[group] = session_ids

    # connecting lines first (same session id across adjacent groups)
    if connect and len(ys) > 1:
        for a, b in zip(order[:-1], order[1:]):
            if a not in ids or b not in ids:
                continue
            pos_b = {sid: k for k, sid in enumerate(ids[b])}
            for k, sid in enumerate(ids[a]):
                j = pos_b.get(sid)
                if j is None:
                    continue
                ax.plot([xs[a][k], xs[b][j]], [ys[a][k], ys[b][j]],
                        color='0.8', lw=0.8, alpha=0.7, zorder=1)

    for i, group in enumerate(order):
        if group not in ys:
            continue
        colour = colours[i % len(colours)]
        ax.scatter(xs[group], ys[group], color=colour, s=40, alpha=0.85,
                   edgecolor='white', linewidth=0.6, zorder=3)
        if show_session_ids:
            for x, y, sid in zip(xs[group], ys[group], ids[group]):
                ax.annotate(str(sid), xy=(x, y), xytext=(3, 0),
                            textcoords='offset points', fontsize=6,
                            color='0.4', va='center', zorder=3)
        c = float(np.mean(ys[group])) if centre == 'mean' else float(np.median(ys[group]))
        ax.plot([i - 0.24, i + 0.24], [c, c], color=colour, lw=2.4, zorder=4)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{g}\n(n={len(ys.get(g, []))})" for g in order], fontsize=8)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_title(stat, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    return ax


def plot_session_stats(
    groups: Mapping[str, 'object'],
    stats: Optional[Sequence[str]] = None,
    ncols: int = 4,
    group_order: Optional[Sequence[str]] = None,
    palette: Optional[Sequence[str]] = None,
    panel_size: tuple = (3.0, 3.0),
    suptitle: Optional[str] = None,
    connect: bool = True,
    show_session_ids: bool = True,
    centre: str = 'mean',
    jitter: float = 0.06,
    seed: int = 0,
):
    """Grid of per-session values by group, one panel per stat.

    Each panel is :func:`plot_session_stats_single`; call that directly to
    place a single stat in a figure you lay out yourself.

    Args:
        groups:      ``{group_label: sessions_frame}``. Each frame is the
                     ``sessions`` field of ``compute_stat(mode='per_session')``.
        stats:       which stats to draw; default is every stat present.
        ncols:       panels per row; unused axes removed.
        group_order: x-axis order of the groups; default is dict order.
        palette:     colours per group; defaults to the house palette.
        panel_size:  (width, height) inches per panel.
        suptitle:    figure title.
        connect:     join points sharing a session id across adjacent groups.
        show_session_ids: annotate each point with its session id.
        centre:      'mean' or 'median' — the session-level centre bar.
        jitter:      horizontal scatter half-width, in x-units.
        seed:        RNG seed for the jitter.

    Returns:
        ``(fig, axes)`` with ``axes`` a flat list of the axes used.

    Raises:
        ValueError: if no groups, or no stats to plot.
    """
    frames = {k: v for k, v in groups.items() if v is not None and len(v)}
    if not frames:
        raise ValueError("plot_session_stats: no non-empty group frames")

    order = list(group_order) if group_order else list(frames)
    order = [g for g in order if g in frames]
    colours = list(palette) if palette else [get_colour(i) for i in range(len(order))]

    if stats is None:
        seen = []
        for frame in frames.values():
            for s in frame['stat'].tolist():
                if s not in seen:
                    seen.append(s)
        stats = seen
    stats = list(stats)
    if not stats:
        raise ValueError("plot_session_stats: no stats to plot")

    ncols = max(1, min(ncols, len(stats)))
    nrows = int(np.ceil(len(stats) / ncols))
    fig, axarr = plt.subplots(nrows, ncols, squeeze=False,
                              figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    flat = axarr.ravel()

    for ax, stat in zip(flat, stats):
        plot_session_stats_single(frames, stat, ax=ax, group_order=order,
                                  palette=colours, connect=connect,
                                  show_session_ids=show_session_ids,
                                  centre=centre, jitter=jitter, seed=seed)

    for ax in flat[len(stats):]:
        fig.delaxes(ax)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig, list(flat[:len(stats)])
