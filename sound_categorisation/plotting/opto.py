"""
plotting/opto.py — draw-only single-panel plotters for the opto analysis.

Contract: every plot_x(data, …, ax=None) draws on a single Axes and does no
analysis (no pooling, fitting, or statistical tests — those live in
``reports.compute``). Inputs are tidy frames from the library analysis layer:

    plot_delta_swarm     <- paired_diff  (the opto − nonopto Δ frame, one stat)

Genotype palette: het warm, wt cool.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

_GENO_COLOUR = {'het': '#d1495b', 'wt': '#30638e'}
_GENO_ORDER = ['wt', 'het']


def _geno_colour(g) -> str:
    return _GENO_COLOUR.get(str(g).lower(), '#888888')


def plot_delta_swarm(delta_df, stat: str, ax: plt.Axes | None = None,
                     p_value: float | None = None,
                     genotype_order: Sequence[str] | None = None,
                     group_col: str = 'genotype', value_col: str = 'delta',
                     seed: int = 0) -> plt.Axes:
    """Per-animal Δ (opto − nonopto) for one stat, split by genotype.

    delta_df: a per-animal Δ frame (e.g. paired_diff output, opto − nonopto). Draws
    jittered per-animal points, a zero reference line, and a per-group median
    bar. `p_value`, if given, is annotated as text — it is NOT computed here
    (run the test in the notebook).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.6))
    sub = delta_df[delta_df['stat'] == stat]
    present = set(sub[group_col])
    order = list(genotype_order) if genotype_order else \
        [g for g in _GENO_ORDER if g in present] + \
        [g for g in sorted(present) if g not in _GENO_ORDER]
    rng = np.random.default_rng(seed)

    ax.axhline(0.0, color='0.6', lw=1, ls='--', zorder=0)
    for i, g in enumerate(order):
        vals = sub[sub[group_col] == g][value_col].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if not len(vals):
            continue
        jit = (rng.random(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full(len(vals), i) + jit, vals,
                   color=_geno_colour(g), s=42, alpha=0.85,
                   edgecolor='white', linewidth=0.6, zorder=3)
        med = np.median(vals)
        ax.plot([i - 0.22, i + 0.22], [med, med],
                color=_geno_colour(g), lw=2.4, zorder=4)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{g}\n(n={int((sub[group_col] == g).sum())})" for g in order])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel(f"Δ {stat}  (opto − nonopto)")
    ax.set_title(stat)
    if p_value is not None:
        ax.annotate(f"p = {p_value:.3g}", xy=(0.5, 0.98), xycoords='axes fraction',
                    ha='center', va='top', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    return ax
