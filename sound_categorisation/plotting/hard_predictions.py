"""
The Hard sessions as the Uniform fits predict them, against what the mice did (``inference.hard_predictions``).
Draw-only.

    scores = pd.read_csv(run / 'scores.csv')
    plot_readout(scores, 'pse_block')
    plot_readout(scores, 'convergence_100_300', kind='masking day', laser_learning=False)
    plot_trajectory(pd.read_csv(run / 'trajectories.csv'), 'SS04', 1)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from behav_utils.plotting import COLOURS

__all__ = ['plot_readout', 'plot_trajectory', 'READOUT_LABELS']

STYLE = {'BE': dict(marker='o', color=COLOURS['BE'], ms=5), 'SC': dict(marker='s', color=COLOURS['SC'], ms=5)}
OFFSET = {'BE': -0.22, 'SC': 0.22}
READOUT_LABELS = {
    'pse_before': 'PSE before the switch',
    'pse_block': "the block's PSE",
    'block_convergence': 'share of the normative shift (0 = before, 1 = normative)',
    'hard_accuracy': 'accuracy on the hard stimuli',
    'hard_a_minus_hard_b': 'mean block PSE: Hard-A − Hard-B',
    'convergence_100_300': 'convergence over trials 100–300 (0 = before, 1 = normative)',
}
_REFERENCE = {'block_convergence': (0, 1), 'convergence_100_300': (0, 1), 'pse_before': (0,), 'pse_block': (0,),
              'hard_a_minus_hard_b': (0,)}


def _as_bool(values: pd.Series) -> pd.Series:
    """True/False whether the column came from a table in memory or from a CSV (where it can read as text)."""
    return values.map(lambda v: str(v).strip().lower() == 'true')


def _text(values: pd.Series) -> pd.Series:
    """Empty text columns come back from a CSV as NaN."""
    return values.fillna('').astype(str)


def _variant(d: pd.DataFrame, laser_learning: bool) -> pd.DataFrame:
    """The rows of one laser variant; an animal run with learning on only (no laser trials) keeps its rows."""
    learns = _as_bool(d['laser_learning'])
    single = d.assign(_l=learns).groupby('animal')['_l'].transform('nunique') == 1
    return d[(learns == bool(laser_learning)) | single]


def plot_readout(scores: pd.DataFrame, readout: str, *, kind: str | None = None, laser_learning: bool = True,
                 ax=None):
    """One row per animal (per animal and switch for a per-switch readout), the animals the bias check keeps first:
    the observed value (black) and each model's predictions (median; bar: the 10-90 % range of the draws). ``kind``
    picks a kind of day for the daily readout; ``laser_learning`` the laser variant. Returns the axes."""
    ax = ax if ax is not None else plt.gca()
    d = scores.assign(kind=_text(scores['kind']), direction=_text(scores['direction']))
    d = d[d['readout'] == readout]
    if kind is not None:
        d = d[d['kind'] == kind]
    d = _variant(d, laser_learning)
    rows = (d[['animal', 'group', 'switch_idx', 'direction']].drop_duplicates()
            .assign(_flagged=lambda r: r['group'].ne('kept')).sort_values(['_flagged', 'animal', 'switch_idx']))
    at = {(a, int(k)): i for i, (a, k) in enumerate(zip(rows['animal'], rows['switch_idx']))}
    for ref in _REFERENCE.get(readout, ()):
        ax.axvline(ref, color='0.6', lw=0.8, ls='-' if ref == 0 else '--')
    obs = d.drop_duplicates(['animal', 'switch_idx'])
    ax.plot(obs['observed'], [at[(a, int(k))] for a, k in zip(obs['animal'], obs['switch_idx'])], 'D', color='k',
            ms=4, label='observed', zorder=3)
    for model, style in STYLE.items():
        m = d[d['model'] == model]
        if not len(m):
            continue
        y = np.array([at[(a, int(k))] for a, k in zip(m['animal'], m['switch_idx'])], float) + OFFSET[model]
        err = np.clip(np.vstack([m['median'] - m['lo'], m['hi'] - m['median']]), 0, None)
        ax.errorbar(m['median'], y, xerr=err, ls='', lw=1, label=f'{model}, predicted', **style)
    labels = [f'{a} · {dr}' if dr else a for a, dr in zip(rows['animal'], rows['direction'])]
    labels = [f'{t} (flagged)' if f else t for t, f in zip(labels, rows['_flagged'])]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlabel(READOUT_LABELS.get(readout, readout))
    ax.set_title(readout + (f' · {kind}' if kind else ''))
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc='best')
    return ax


def plot_trajectory(trajectories: pd.DataFrame, animal: str, switch_idx: int, *, laser_learning: bool = True,
                    ax=None):
    """The criterion within one switch, by trial after it: the animal's own (black) and each model's median over the
    draws with its 10-90 % range. Returns the axes."""
    ax = ax if ax is not None else plt.gca()
    t = trajectories[(trajectories['animal'] == animal) & (trajectories['switch_idx'] == switch_idx)]
    t = t.assign(direction=_text(t['direction']))
    obs = t[t['source'] == 'observed'].sort_values('trial')
    ax.plot(obs['trial'], obs['median'], color='k', marker='D', ms=3, lw=1.2, label='observed', zorder=3)
    for model, style in STYLE.items():
        m = t[t['source'] == model]
        if not len(m):
            continue
        m = _variant(m, laser_learning).sort_values('trial')
        ax.fill_between(m['trial'], m['lo'], m['hi'], color=style['color'], alpha=0.2, lw=0)
        ax.plot(m['trial'], m['median'], color=style['color'], lw=1.5, label=f'{model}, predicted')
    ax.axhline(0, color='0.6', lw=0.6)
    ax.set_xlabel('trial after the switch')
    ax.set_ylabel('criterion (PSE)')
    direction = t['direction'].iloc[0] if len(t) else ''
    ax.set_title(f'{animal} · {direction}' if direction else animal, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    return ax
