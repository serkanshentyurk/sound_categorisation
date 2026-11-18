"""
How close each model's held-out error gets to the noise floor, per animal (``inference.noise_floor``). Draw-only.

    from sound_categorisation.inference.noise_floor import floor_ratios, model_errors
    from sound_categorisation.plotting.noise_floor import plot_noise_floor

    ratios = floor_ratios(table, model_errors(load_cv_results(d).comparison), split='session')
    plot_noise_floor(ratios)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from behav_utils.plotting import COLOURS
from matplotlib.ticker import FuncFormatter

__all__ = ['plot_noise_floor']


def _tick(v, _):
    """Plain numbers on the log axis, at 1, 2, 3 and 5 times a power of ten."""
    if not (np.isfinite(v) and v > 0):
        return ''
    lead = round(v / 10 ** np.floor(np.log10(v)), 6)
    return f'{v:g}' if lead in (1, 2, 3, 5) else ''

STYLE = {'no_learning': dict(marker='x', color='0.45', ms=6, label='no learning'),
         'true_model': dict(marker='+', color='k', ms=8, label='true model'),
         'BE': dict(marker='o', color=COLOURS['BE'], ms=5, label='BE'),
         'SC': dict(marker='s', color=COLOURS['SC'], ms=5, label='SC')}


def plot_noise_floor(ratios: pd.DataFrame, *, band: tuple | None = None, ax=None):
    """One column per animal (ordered by its floor): each error in ``ratios`` (``floor_ratios``) over the floor, on a
    log scale; the line at 1 is the floor. ``band`` = (low, high) shades where a correct model's error falls (e.g.
    the 10-90 % range of ``true_model`` on a synthetic cohort). Returns the axes."""
    ax = ax if ax is not None else plt.gca()
    d = ratios.sort_values('floor').reset_index(drop=True)
    x = np.arange(len(d))
    if band is not None:
        ax.axhspan(*band, color='0.92', zorder=0, lw=0)
    ax.axhline(1.0, color='k', lw=1, label='floor')
    shift = {k: v for k, v in zip(('no_learning', 'true_model', 'BE', 'SC'), (-0.18, -0.06, 0.06, 0.18))}
    for col, style in STYLE.items():
        if col in d and d[col].notna().any():
            ax.plot(x + shift[col], d[col], ls='', **style)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(_tick))
    ax.yaxis.set_minor_formatter(FuncFormatter(_tick))
    ax.set_xticks(x)
    ax.set_xticklabels(d['animal'], rotation=90, fontsize=7)
    ax.set_xlim(-0.6, len(d) - 0.4)
    ax.set_ylabel('held-out error / noise floor')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=5, loc='upper left')
    return ax
