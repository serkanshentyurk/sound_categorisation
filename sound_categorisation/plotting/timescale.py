"""
Plots for the timescale analysis (behaviour/timescale.py), drawn from its tables; nothing is computed.

    from sound_categorisation.plotting.timescale import plot_session_criterion, plot_memory

    plot_session_criterion(sessions[sessions.animal == 'SS04'])     # one animal's sessions table
    plot_memory(fits)                                               # the fits table, both cohorts
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from behav_utils.plotting import PALETTE

from sound_categorisation.plotting._style import DIST_COLOURS

__all__ = ['plot_session_criterion', 'plot_memory']

PSE_COLUMN = {'main': 'pse_all', 'laser_off': 'pse_laser_off', 'all_trials': 'pse_all'}
DESIGN_COLOURS = {'blocked': PALETTE[0], 'alternation': PALETTE[1]}


def plot_session_criterion(sessions: pd.DataFrame, variant: str = 'main', ax: plt.Axes | None = None) -> plt.Axes:
    """One animal's criterion session by session: the observed PSE (filled where the variant fits it,
    open where it does not, triangles for laser sessions), the fitted integrator (line) and each
    distribution's normative PSE (dashes). ``sessions``: the timescale ``sessions`` table for one animal."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    s = sessions.sort_values('order')
    hard = s[s['distribution'].isin(('Hard-A', 'Hard-B'))]
    ax.step(s['order'], s['normative_pse'], where='mid', ls='--', lw=0.8, color='0.55', label='normative')
    pred = f'predicted_{variant}'
    if pred in hard:
        ax.plot(hard['order'], hard[pred], '-', color='k', lw=1.2, label='integrator')
    observed = hard[PSE_COLUMN[variant]]
    fitted = ~hard['laser'] if variant == 'main' else pd.Series(True, index=hard.index)
    for dist in ('Hard-A', 'Hard-B'):
        for laser in (False, True):
            for fit in (True, False):
                m = (hard['distribution'] == dist) & (hard['laser'] == laser) & (fitted == fit)
                if not m.any():
                    continue
                c = DIST_COLOURS.get(dist, '0.3')
                ax.scatter(hard.loc[m, 'order'], observed[m], s=24, marker='^' if laser else 'o',
                           facecolors=c if fit else 'none', edgecolors=c, lw=1, zorder=3,
                           label=f'{dist}{" laser" if laser else ""}{"" if fit else " (not fitted)"}')
    ax.axhline(0, color='0.85', lw=0.6, zorder=0)
    ax.set_xlabel('session, in order')
    ax.set_ylabel('criterion (PSE)')
    return ax


def plot_memory(fits: pd.DataFrame, variant: str = 'main', gain_mode: str = 'fixed',
                ax: plt.Axes | None = None) -> plt.Axes:
    """Memory N per animal on a log axis: the best fit with its 95 % interval, blocked animals first.
    An interval with no upper end (no movement cannot be excluded) is drawn as an arrow up from its
    lower end. Open markers: animals the bias check flags. ``fits``: the timescale ``fits`` table."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    f = fits[(fits['variant'] == variant) & (fits['gain_mode'] == gain_mode) & fits['n_low'].notna()].copy()
    order = {'blocked': 0, 'alternation': 1}
    f = f.assign(_d=f['design'].map(order).fillna(2)).sort_values(['_d', 'animal']).reset_index(drop=True)
    x = np.arange(len(f)) + f['_d'].to_numpy()            # a gap between the designs
    values = f[['n_low', 'n_high', 'n_hat']].to_numpy(float).ravel()
    values = values[np.isfinite(values)]
    bottom = values.min() / 2 if len(values) else 1.0
    top = max(1e4, 10 * values.max()) if len(values) else 1e4      # unbounded intervals reach up to here
    for xi, (_, r) in zip(x, f.iterrows()):
        c = DESIGN_COLOURS.get(r['design'], '0.3')
        face = 'none' if bool(r.get('biased_animal', False)) else c
        if np.isfinite(r['n_high']):
            ax.plot([xi, xi], [r['n_low'], r['n_high']], color=c, lw=1.5)
            if np.isfinite(r['n_hat']):
                ax.scatter([xi], [r['n_hat']], s=30, facecolors=face, edgecolors=c, zorder=3)
        else:
            ax.annotate('', xy=(xi, top), xytext=(xi, r['n_low']),
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.5))
            ax.scatter([xi], [r['n_low']], s=30, marker='_', color=c, zorder=3)
    ax.set_yscale('log')
    ax.set_ylim(bottom, top * 1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(f['animal'], rotation=90, fontsize=8)
    ax.set_ylabel('memory N (trials)')
    for design, c in DESIGN_COLOURS.items():
        ax.scatter([], [], color=c, label=design)
    return ax
