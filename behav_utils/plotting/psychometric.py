"""
Psychometric Curve Plotting

plot_psychometric(data, ax, mode, **kwargs)

Accepts SessionData, List[SessionData], AnimalData, or (stimuli, choices) tuple.
NO FILTERING. Data must be pre-filtered via filter_trials / session.filter.

Modes for multi-session: 'pooled', 'overlay', 'session_mean'.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, TYPE_CHECKING

from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.analysis.utils import cumulative_gaussian
from behav_utils.plotting.styles import (
    PALETTE, COLOURS, DEFAULT_ALPHA, SEM_ALPHA,
    DEFAULT_LINE_WIDTH, DEFAULT_MARKER_SIZE,
    get_session_colours,
)

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData, AnimalData


def plot_psychometric(
    data, ax=None, mode='pooled',
    color=None, label=None, alpha=DEFAULT_ALPHA,
    linewidth=DEFAULT_LINE_WIDTH, linestyle='-',
    n_bins=8, n_bootstrap=0, show_ci=True, show_data=True,
    show_params=False, show_lapse=False, title='',
    session_colours=None,
    show_individual=True, individual_alpha=0.12,
    show_reference=True,
) -> Tuple[plt.Figure, plt.Axes, dict]:
    """Plot psychometric curve(s). Data must be pre-filtered."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    sessions, raw = _resolve_data(data)

    if raw is not None:
        info = _draw_single(raw[0], raw[1], ax, color=color, label=label,
            alpha=alpha, linewidth=linewidth, linestyle=linestyle,
            n_bins=n_bins, n_bootstrap=n_bootstrap, show_ci=show_ci,
            show_data=show_data, show_params=show_params, show_lapse=show_lapse)
    elif len(sessions) == 1 or mode == 'pooled':
        stim, ch = _pool(sessions)
        info = _draw_single(stim, ch, ax, color=color, label=label,
            alpha=alpha, linewidth=linewidth, linestyle=linestyle,
            n_bins=n_bins, n_bootstrap=n_bootstrap, show_ci=show_ci,
            show_data=show_data, show_params=show_params,
            show_lapse=show_lapse) if len(stim) > 0 else {}
    elif mode == 'overlay':
        info = _draw_overlay(sessions, ax, color=color, alpha=alpha,
            linewidth=linewidth, n_bins=n_bins, session_colours=session_colours)
    elif mode == 'session_mean':
        info = _draw_session_mean(sessions, ax, color=color, label=label,
            alpha=alpha, linewidth=linewidth, n_bins=n_bins, show_ci=show_ci,
            show_data=show_data, show_individual=show_individual,
            individual_alpha=individual_alpha)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    if show_reference:
        ax.axhline(0.5, color='grey', ls='--', alpha=0.3, zorder=0)
        ax.axvline(0, color='grey', ls='--', alpha=0.3, zorder=0)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Stimulus'); ax.set_ylabel('P(choose B)')
    if title:
        ax.set_title(title)
    return fig, ax, info


# ── Data resolution (NO filtering) ──────────────────────────────────────

def _resolve_data(data):
    from behav_utils.data.structures import SessionData, AnimalData
    if isinstance(data, SessionData):
        return [data], None
    if isinstance(data, AnimalData):
        return list(data.sessions), None
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return [], None
        if hasattr(data[0], 'trials'):
            return list(data), None
        if len(data) == 2 and isinstance(data[0], np.ndarray):
            return None, (data[0], data[1])
    try:
        items = list(data)
        if items and hasattr(items[0], 'trials'):
            return items, None
    except TypeError:
        pass
    raise TypeError(f"Expected SessionData/List/AnimalData/(stim,ch), got {type(data).__name__}")


def _stim_choice(session):
    """Get (stimuli, choices) from one session. No filtering."""
    arr = session.get_arrays()
    v = ~arr['no_response']
    return arr['stimuli'][v], arr['choices'][v]


def _pool(sessions):
    """Concatenate across sessions. No filtering."""
    ss, cc = [], []
    for s in sessions:
        st, ch = _stim_choice(s)
        if len(st) > 0:
            ss.append(st); cc.append(ch)
    return (np.concatenate(ss), np.concatenate(cc)) if ss else (np.array([]), np.array([]))


def _bin(stimuli, choices, n_bins=8):
    bins = np.linspace(-1, 1, n_bins + 1)
    centres = (bins[:-1] + bins[1:]) / 2
    means = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = (stimuli >= bins[b]) & (stimuli < bins[b + 1])
        if b == n_bins - 1:
            m |= (stimuli == bins[b + 1])
        if m.sum() > 0:
            means[b] = np.mean(choices[m])
    return centres, means


def _bootstrap_ci(stimuli, choices, n_bootstrap, seed=42):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, 200)
    curves = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(stimuli), len(stimuli), replace=True)
        try:
            pf = fit_psychometric(stimuli[idx], choices[idx])
            if not np.isnan(pf.get('mu', np.nan)):
                curves.append(cumulative_gaussian(x, pf['mu'], pf['sigma'],
                    pf['lapse_low'], pf['lapse_high']))
        except Exception:
            pass
    if len(curves) < 10:
        return x, None, None
    a = np.array(curves)
    return x, np.percentile(a, 2.5, axis=0), np.percentile(a, 97.5, axis=0)


# ── Drawing ──────────────────────────────────────────────────────────────

def _draw_single(stimuli, choices, ax, color=None, label=None,
                 alpha=DEFAULT_ALPHA, linewidth=DEFAULT_LINE_WIDTH,
                 linestyle='-', n_bins=8, n_bootstrap=0, show_ci=True,
                 show_data=True, show_params=False, show_lapse=False):
    color = color or COLOURS['default']
    x = np.linspace(-1, 1, 200)
    pf = fit_psychometric(stimuli, choices)
    info = {**pf, 'n_trials': len(stimuli)}

    if np.isnan(pf.get('mu', np.nan)):
        if show_data:
            c, m = _bin(stimuli, choices, n_bins)
            ax.plot(c, m, 'o', color=color, markersize=DEFAULT_MARKER_SIZE, alpha=alpha, label=label)
        return info

    y = cumulative_gaussian(x, pf['mu'], pf['sigma'], pf['lapse_low'], pf['lapse_high'])
    lbl = label
    if show_params:
        lbl = f"{label or ''} (PSE={pf['mu']:.2f}, σ={pf['sigma']:.2f})".strip()
    ax.plot(x, y, color=color, lw=linewidth, ls=linestyle, alpha=alpha, label=lbl, zorder=2)

    if show_data:
        c, m = _bin(stimuli, choices, n_bins)
        v = ~np.isnan(m)
        ax.plot(c[v], m[v], 'o', color=color, markersize=DEFAULT_MARKER_SIZE, alpha=alpha*0.7, zorder=3)

    if n_bootstrap > 0 and show_ci:
        _, lo, hi = _bootstrap_ci(stimuli, choices, n_bootstrap)
        if lo is not None:
            ax.fill_between(np.linspace(-1,1,200), lo, hi, color=color, alpha=SEM_ALPHA, zorder=1)

    if show_lapse:
        ax.axhline(pf['lapse_low'], color='grey', ls=':', alpha=0.4)
        ax.axhline(1-pf['lapse_high'], color='grey', ls=':', alpha=0.4)
    return info


def _draw_overlay(sessions, ax, color=None, alpha=DEFAULT_ALPHA,
                  linewidth=1.0, n_bins=8, session_colours=None):
    n = len(sessions)
    colours = session_colours or ([color]*n if color else get_session_colours(n))
    x = np.linspace(-1, 1, 200)
    infos = []
    for i, s in enumerate(sessions):
        st, ch = _stim_choice(s)
        if len(st) < 10:
            infos.append({}); continue
        pf = fit_psychometric(st, ch)
        infos.append(dict(pf))
        if not np.isnan(pf.get('mu', np.nan)):
            y = cumulative_gaussian(x, pf['mu'], pf['sigma'], pf['lapse_low'], pf['lapse_high'])
            a = 0.3 + 0.5 * (i / max(n-1, 1))
            ax.plot(x, y, color=colours[i], lw=linewidth, alpha=a, zorder=2)
    return {'per_session': infos, 'n_sessions': n}


def _draw_session_mean(sessions, ax, color=None, label=None,
                       alpha=DEFAULT_ALPHA, linewidth=DEFAULT_LINE_WIDTH,
                       n_bins=8, show_ci=True, show_data=True,
                       show_individual=True, individual_alpha=0.12):
    color = color or COLOURS['default']
    x = np.linspace(-1, 1, 200)
    all_b, centres = [], None
    for s in sessions:
        st, ch = _stim_choice(s)
        if len(st) < 10: continue
        c, m = _bin(st, ch, n_bins)
        all_b.append(m); centres = c
    if not all_b:
        return {'n_sessions': 0}
    arr = np.array(all_b)
    mean_p = np.nanmean(arr, axis=0)
    sem_p = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(arr), axis=0))

    if show_individual:
        for s in sessions:
            st, ch = _stim_choice(s)
            if len(st) < 10: continue
            try:
                pf = fit_psychometric(st, ch)
                y = cumulative_gaussian(x, pf['mu'], pf['sigma'], pf['lapse_low'], pf['lapse_high'])
                ax.plot(x, y, color=color, alpha=individual_alpha, lw=0.8, zorder=1)
            except Exception: pass

    v = ~np.isnan(mean_p)
    if show_data:
        ax.plot(centres[v], mean_p[v], 'o-', color=color, markersize=DEFAULT_MARKER_SIZE,
                lw=linewidth, alpha=alpha, label=label, zorder=3)
    if show_ci:
        ax.fill_between(centres, mean_p-sem_p, mean_p+sem_p, color=color, alpha=SEM_ALPHA, zorder=1)

    pf = fit_psychometric(centres[v], mean_p[v])
    return {'n_sessions': len(all_b), 'mean_p': mean_p, 'sem_p': sem_p, 'centres': centres, **dict(pf)}
