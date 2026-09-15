"""
Adaptation trajectory plots from compute_adaptation / compute_adaptation_per_session.

Both draw the *standardised* trajectory: value(t) − expert-uniform baseline, so
0 is expert-uniform behaviour (no ratio, no clip). For the 'pse' stat the
normative optimum is drawn as a dashed reference line (already baseline-
subtracted when the result is standardised). Pass ``stat=`` to choose which
rolled stat to plot (default 'pse').
"""

from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from behav_utils.plotting.styles import get_colour

__all__ = ['plot_adaptation', 'plot_adaptation_sessions']


def _ylabel(stat: str, standardised: bool) -> str:
    return f'rolling {stat}' + (' − expert' if standardised else '')


def plot_adaptation(
    result: Dict,
    ax: Optional[plt.Axes] = None,
    stat: str = 'pse',
    color: Optional[str] = None,
    label: Optional[str] = None,
    show_band: bool = True,
    show_plateau: bool = True,
) -> plt.Axes:
    """Plot one pooled adaptation trajectory from ``compute_adaptation`` (mode='pooled').

    Call repeatedly on the same ``ax`` with different colours/labels to overlay
    animals or distributions.

    Args:
        result:       output of ``compute_adaptation`` with ``mode='pooled'``.
        ax:           axis to draw on (creates one if None).
        stat:         which rolled stat to plot (default 'pse').
        color:        curve colour (house palette[0] if None).
        label:        legend label.
        show_band:    shade the bootstrap CI band when present (absent for now).
        show_plateau: dashed horizontal line at the plateau value.

    Returns:
        The axis used.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    colour = color or get_colour(0)

    curve = result['curve']
    trials = np.asarray(curve['trials'], dtype=float)
    values = np.asarray(curve['values'].get(stat, []), dtype=float)
    standardised = result.get('standardised', True)

    ax.axhline(0.0, color='0.6', lw=1, ls='-', zorder=0)   # baseline (expert-uniform)
    norm = result.get('normative', {}).get('pse', np.nan)
    if stat == 'pse' and np.isfinite(norm):
        ax.axhline(norm, color='crimson', lw=1, ls='--', zorder=0, label='normative')

    if show_band and curve.get('ci_lo') is not None:
        lo = np.asarray(curve['ci_lo'], dtype=float)
        hi = np.asarray(curve['ci_hi'], dtype=float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        if ok.any():
            ax.fill_between(trials[ok], lo[ok], hi[ok], color=colour, alpha=0.18, linewidth=0)

    ok = np.isfinite(values)
    ax.plot(trials[ok], values[ok], color=colour, lw=2, marker='o', ms=3, label=label, zorder=3)

    if show_plateau:
        rows = {r['stat']: r['value'] for r in result.get('rows', [])}
        plateau = rows.get(f'{stat}_plateau')
        if plateau is not None and np.isfinite(plateau):
            ax.axhline(plateau, color=colour, lw=1, ls='--', alpha=0.5, zorder=1)

    ax.set_xlabel('trial in block')
    ax.set_ylabel(_ylabel(stat, standardised))
    if label:
        ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    return ax


_TYPE_COLOUR = {'regular': '#30638e', 'opto': '#d1495b', 'masking': '#e9a13b', 'washout': '#8a8a8a'}


def plot_adaptation_sessions(
    result: Dict,
    ax: Optional[plt.Axes] = None,
    stat: str = 'pse',
    layout: str = 'concat',
    colour_by_type: bool = True,
    show_normative: bool = True,
    label_sessions: bool = True,
) -> plt.Axes:
    """Per-session adaptation trajectories from ``compute_adaptation_per_session``.

    Each session's within-session rolling ``stat`` (as ``value − expert``) is
    drawn as its own segment — no window crosses a boundary. y=0 is expert-
    uniform; for 'pse' the dashed line is the normative target
    (``result['normative']['pse']``, already baseline-subtracted).

    Args:
        result:         a ``compute_adaptation_per_session`` result.
        ax:             Axes to draw on; a new one is made if None.
        stat:           which rolled stat to plot (default 'pse').
        layout:         'concat' lays sessions end-to-end along x with a boundary
                        line between them (the run as it happened, real trial
                        counts); 'rezero' re-zeros each session to x=0 and
                        overlays them (start/end offsets directly comparable).
        colour_by_type: colour each segment by session_type (opto/masking/…).
        show_normative: draw the normative target line (pse only).
        label_sessions: annotate each segment with its switch_index.

    Returns:
        the Axes drawn on.
    """
    entries = result.get('sessions', [])
    norm = result.get('normative', {}).get('pse', np.nan)
    standardised = result.get('standardised', True)

    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 3.6))

    ax.axhline(0.0, color='0.55', lw=1, ls='-', zorder=1)          # expert-uniform
    show_norm = show_normative and stat == 'pse' and np.isfinite(norm)
    if show_norm:
        ax.axhline(norm, color='crimson', lw=1, ls='--', zorder=1, label='normative')

    cursor = 0.0
    boundaries = []
    for e in entries:
        x = np.asarray(e['trials'], dtype=float)
        y = np.asarray(e['values'].get(stat, []), dtype=float)
        if layout == 'rezero':
            xs = x - (x[0] if x.size else 0.0)
        else:
            xs = x - (x[0] if x.size else 0.0) + cursor
        colour = _TYPE_COLOUR.get(e.get('session_type', ''), get_colour(0)) if colour_by_type else get_colour(0)
        ax.plot(xs, y, color=colour, lw=1.6, marker='o', ms=2.5, alpha=0.9, zorder=3)
        if layout != 'rezero':
            span = (x[-1] - x[0]) if x.size > 1 else float(e.get('n_trials', 50))
            cursor += span + 0.05 * max(span, 1.0)
            boundaries.append(cursor)
            if label_sessions and x.size:
                si = e.get('switch_index')
                ax.annotate('' if si is None or not np.isfinite(si) else f"#{int(si)}",
                            xy=(xs[0], ax.get_ylim()[1]), xytext=(2, -2),
                            textcoords='offset points', fontsize=7, color='0.4', va='top')

    if layout != 'rezero':
        for b in boundaries[:-1]:
            ax.axvline(b - 0.025 * max(b, 1.0), color='0.85', lw=0.8, zorder=0)
        ax.set_xlabel('trials (sessions laid end to end)')
    else:
        ax.set_xlabel('trials since session start')

    ax.set_ylabel(_ylabel(stat, standardised))
    ax.spines[['top', 'right']].set_visible(False)
    if colour_by_type:
        seen = {}
        for e in entries:
            t = e.get('session_type', '')
            if t not in seen:
                seen[t] = _TYPE_COLOUR.get(t, get_colour(0))
        handles = [plt.Line2D([0], [0], color=c, lw=2, label=t) for t, c in seen.items()]
        if show_norm:
            handles.append(plt.Line2D([0], [0], color='crimson', lw=1, ls='--', label='normative'))
        ax.legend(handles=handles, frameon=False, fontsize=8, ncol=len(handles))
    return ax
