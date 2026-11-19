"""
The observed post-laser effect against what complete blocking would give, per animal (``inference.blocked_update``).
Draw-only.

    summary = pd.read_csv(run / 'summary.csv')
    plot_blocked_update(summary, 'win_stay')
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from behav_utils.plotting import COLOURS

__all__ = ['plot_blocked_update']

STYLE = {'BE': dict(marker='o', color=COLOURS['BE'], ms=5, label='BE, complete blocking'),
         'SC': dict(marker='s', color=COLOURS['SC'], ms=5, label='SC, complete blocking')}
OFFSET = {'BE': -0.22, 'SC': 0.22}


def plot_blocked_update(summary: pd.DataFrame, stat: str, *, ax=None):
    """One row per animal, HET first: the observed delta-of-deltas of ``stat`` after a laser trial with its interval
    (black), and the effect complete blocking would have under each model (median over the parameter draws, bar:
    their 10-90 % range). The grey band is the margin: a predicted effect inside it is too weak to test. Returns the
    axes."""
    ax = ax if ax is not None else plt.gca()
    d = summary[summary['stat'] == stat]
    animals = (d[['animal', 'genotype']].drop_duplicates()
               .sort_values(['genotype', 'animal'], key=lambda c: c.ne('het') if c.name == 'genotype' else c))
    row = {aid: i for i, aid in enumerate(animals['animal'])}
    margin = d['margin'].dropna()
    if len(margin):
        ax.axvspan(-margin.iloc[0], margin.iloc[0], color='0.92', zorder=0, lw=0)
    ax.axvline(0, color='0.4', lw=0.8)
    obs = d.drop_duplicates('animal')
    y = obs['animal'].map(row).to_numpy(float)
    err = np.clip(np.vstack([obs['observed'] - obs['ci_lo'], obs['ci_hi'] - obs['observed']]), 0, None)
    ax.errorbar(obs['observed'], y, xerr=err, fmt='D', color='k', ms=4, lw=1.2,
                label='observed (opto − masking sessions)')
    for model, style in STYLE.items():
        m = d[d['model'] == model]
        if not len(m):
            continue
        ym = m['animal'].map(row).to_numpy(float) + OFFSET[model]
        ax.errorbar(m['effect'], ym, xerr=np.vstack([m['effect'] - m['effect_lo'], m['effect_hi'] - m['effect']]),
                    ls='', lw=1, **style)
    ax.set_yticks(list(row.values()))
    ax.set_yticklabels([f'{a} ({g.upper()})' for a, g in zip(animals['animal'], animals['genotype'])], fontsize=8)
    ax.set_ylim(len(row) - 0.5, -0.5)
    ax.set_xlabel(f'Δ {stat}: after a laser trial − other laser-off trials')
    ax.set_title(stat)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc='lower left')
    return ax
