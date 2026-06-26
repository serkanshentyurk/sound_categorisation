"""
plotting/opto.py — draw-only single-panel plotters for the opto analysis.

Contract: every plot_x(data, …, ax=None) draws on a single Axes and does no
analysis (no pooling, fitting, or statistical tests — those live in the
notebooks). Inputs are the tidy frames from analysis/opto.py:

    plot_delta_swarm     <- paired_diff (the opto − nonopto Δ frame)
    plot_delta_paired    <- paired_diff per phase, concatenated with a 'phase' column
    plot_stat_trajectory <- compute_opto_trajectory

Genotype palette: het warm, wt cool.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence

_GENO_COLOUR = {'het': '#d1495b', 'wt': '#30638e'}
_GENO_ORDER = ['wt', 'het']


def _geno_colour(g) -> str:
    return _GENO_COLOUR.get(str(g).lower(), '#888888')


def plot_delta_swarm(delta_df, stat: str, ax: Optional[plt.Axes] = None,
                     p_value: Optional[float] = None,
                     genotype_order: Optional[Sequence[str]] = None,
                     seed: int = 0) -> plt.Axes:
    """Per-animal Δ (opto − nonopto) for one stat, split by genotype.

    delta_df: a per-animal Δ frame (e.g. paired_diff output, opto − nonopto). Draws
    jittered per-animal points, a
    zero reference line, and a per-group median bar. `p_value`, if given, is
    annotated as text — it is NOT computed here (run the test in the notebook).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.6))
    sub = delta_df[delta_df['stat'] == stat]
    present = set(sub['genotype'])
    order = list(genotype_order) if genotype_order else \
        [g for g in _GENO_ORDER if g in present] + \
        [g for g in sorted(present) if g not in _GENO_ORDER]
    rng = np.random.default_rng(seed)

    ax.axhline(0.0, color='0.6', lw=1, ls='--', zorder=0)
    for i, g in enumerate(order):
        vals = sub[sub['genotype'] == g]['delta'].to_numpy(dtype=float)
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
    ax.set_xticklabels([f"{g}\n(n={int((sub['genotype'] == g).sum())})" for g in order])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel(f"Δ {stat}  (opto − nonopto)")
    ax.set_title(stat)
    if p_value is not None:
        ax.annotate(f"p = {p_value:.3g}", xy=(0.5, 0.98), xycoords='axes fraction',
                    ha='center', va='top', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    return ax


def plot_stat_trajectory(traj_df, stat: str, ax: Optional[plt.Axes] = None,
                         condition: str = 'opto', linestyle: str = '-',
                         show_legend: bool = True) -> plt.Axes:
    """Per-session trajectory of one stat for one condition, one line per animal.

    traj_df: output of compute_opto_trajectory. Lines coloured by genotype.
    Failed per-session fits (success=False) are dropped before plotting, so
    curve-stat trajectories show only the sessions that fitted.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.4))
    sub = traj_df[(traj_df['stat'] == stat) & (traj_df['condition'] == condition)]
    if 'success' in sub.columns:
        sub = sub[sub['success'].astype(bool)]

    seen = set()
    for aid, g in sub[['animal', 'genotype']].drop_duplicates().itertuples(index=False):
        a = sub[sub['animal'] == aid].sort_values('session_idx')
        x = a['session_idx'].to_numpy(dtype=float)
        y = a['value'].to_numpy(dtype=float)
        m = ~np.isnan(y)
        if not m.any():
            continue
        label = g if g not in seen else None
        seen.add(g)
        ax.plot(x[m], y[m], marker='o', ms=4, lw=1.4, ls=linestyle,
                color=_geno_colour(g), alpha=0.8, label=label)

    ax.set_xlabel(f"{condition} session (ordinal)")
    ax.set_ylabel(stat)
    ax.set_title(f"{stat} — {condition}")
    ax.spines[['top', 'right']].set_visible(False)
    if show_legend and seen:
        ax.legend(frameon=False, fontsize=9, title='genotype')
    return ax


def plot_delta_paired(delta_df, stat: str, ax: Optional[plt.Axes] = None,
                      phase_a: str = 'uniform', phase_b: str = 'hard',
                      p_value: Optional[float] = None,
                      genotype_order: Optional[Sequence[str]] = None) -> plt.Axes:
    """Per-animal Δ at phase_a vs phase_b, connected — the dispensability view.

    delta_df: per-phase paired_diff Δ frames concatenated with a
    'phase' column. For one stat, each animal contributes a line from its
    phase_a Δ to its phase_b Δ (coloured by genotype), so a steepening of the
    opto effect from expert to post-shift shows up per animal. Only animals with
    BOTH phases are drawn (the paired set). Genotypes are offset horizontally.
    `p_value` (the interaction test) is annotated, not computed here.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 3.8))
    sub = delta_df[delta_df['stat'] == stat]
    wide = sub.pivot_table(index=['animal', 'genotype'], columns='phase',
                           values='delta').rename_axis(columns=None)
    for ph in (phase_a, phase_b):
        if ph not in wide.columns:
            ax.set_title(f"{stat} (missing '{ph}')")
            ax.spines[['top', 'right']].set_visible(False)
            return ax
    wide = wide.dropna(subset=[phase_a, phase_b]).reset_index()
    present = set(wide['genotype'])
    order = list(genotype_order) if genotype_order else \
        [g for g in _GENO_ORDER if g in present] + \
        [g for g in sorted(present) if g not in _GENO_ORDER]
    off = {g: (i - (len(order) - 1) / 2) * 0.12 for i, g in enumerate(order)}
    xa, xb = 0.0, 1.0

    ax.axhline(0.0, color='0.6', lw=1, ls='--', zorder=0)
    for g in order:
        gw = wide[wide['genotype'] == g]
        c = _geno_colour(g)
        xpa, xpb = xa + off[g], xb + off[g]
        for _, r in gw.iterrows():
            ax.plot([xpa, xpb], [r[phase_a], r[phase_b]],
                    color=c, lw=1.2, alpha=0.6, zorder=2)
            ax.scatter([xpa, xpb], [r[phase_a], r[phase_b]], color=c, s=34,
                       alpha=0.9, edgecolor='white', linewidth=0.5, zorder=3)
        for xp, ph in [(xpa, phase_a), (xpb, phase_b)]:
            v = gw[ph].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            if len(v):
                ax.plot([xp - 0.07, xp + 0.07], [np.median(v), np.median(v)],
                        color=c, lw=2.4, zorder=4)

    ax.set_xticks([xa, xb])
    ax.set_xticklabels([phase_a, phase_b])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel(f"Δ {stat}  (opto − nonopto)")
    ax.set_title(stat)
    if p_value is not None:
        ax.annotate(f"interaction p = {p_value:.3g}", xy=(0.5, 0.98),
                    xycoords='axes fraction', ha='center', va='top', fontsize=9)
    handles = [plt.Line2D([0], [0], color=_geno_colour(g), lw=2,
                          label=f"{g} (n={int((wide['genotype'] == g).sum())})")
               for g in order]
    ax.legend(handles=handles, frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    return ax
