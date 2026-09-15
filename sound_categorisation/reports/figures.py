"""
Report figures. Every function takes an AnimalResult / GroupResult (or its
tables) and returns a matplotlib Figure; nothing here computes a statistic.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from behav_utils.plotting import (
    plot_interaction_single, plot_psychometric_curve, plot_stat_comparison_single, plot_update_matrix,
)
from behav_utils.readouts import UpdateMatrix

from sound_categorisation.adaptation import SwitchAdaptation
from sound_categorisation.plotting.opto import plot_delta_swarm
from sound_categorisation.reports.compute import PHASES, AnimalResult, GroupResult, Settings

__all__ = ['contrast_grid', 'interaction_grid', 'psychometric_page', 'update_matrix_page',
           'group_psychometric_page', 'group_update_matrix_page', 'swarm_page',
           'adaptation_page', 'group_adaptation_page']

OFF, ON, ALL = '#7f7f7f', '#1f77b4', '#2ca02c'
GENO_COL = {'het': '#d62728', 'wt': '#2ca02c'}
TYPE_COL = {'opto': ON, 'masking': OFF}
TRIAL_UNITS = ('trials',)


def _grid(n: int, ncols: int, size=(3.3, 3.0)):
    nrows = math.ceil(max(n, 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False)
    return fig, axes.ravel()


def contrast_grid(result, stats: Sequence[str], units: Sequence[str], title: str, ncols: int = 4):
    """One panel per stat: point + CI per phase, p above (a DeltaStats)."""
    fig, axf = _grid(len(stats), ncols)
    units = [u for u in units if u in result.units] or None
    for ax, stat in zip(axf, stats):
        plot_stat_comparison_single(result, stat, ax=ax, units=units)
    for ax in axf[len(stats):]:
        fig.delaxes(ax)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def interaction_grid(interaction, stats: Sequence[str], units: Sequence[str], title: str, ncols: int = 4):
    fig, axf = _grid(len(stats), ncols)
    units = [u for u in units if u in interaction.units] or None
    stats = [s for s in stats if s in interaction.interaction.index]
    for ax, stat in zip(axf, stats):
        plot_interaction_single(interaction, stat, ax=ax, units=units)
    for ax in axf[len(stats):]:
        fig.delaxes(ax)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def psychometric_page(r: AnimalResult, title: str):
    """One panel per phase: all trials (green), non_opto (grey), toi (blue)."""
    phases = PHASES[r.design]
    fig, axes = plt.subplots(1, len(phases), figsize=(4.7 * len(phases), 4.0), squeeze=False)
    for ax, phase in zip(axes[0], phases):
        for tt, colour, label in (('all', ALL, 'all trials'), ('non_opto', OFF, 'non_opto'), (r.toi, ON, r.toi)):
            ro = r.readouts.get((phase, tt))
            if ro is not None and ro.curve is not None:
                plot_psychometric_curve(ro.curve, ax=ax, color=colour, label=label)
        ax.set_title(f'{phase} (n={r.n_sessions.get(phase, 0)} sessions)', fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def update_matrix_page(r: AnimalResult, title: str):
    phases = PHASES[r.design]
    fig, axes = plt.subplots(len(phases), 2, figsize=(8, 4 * len(phases)), squeeze=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for row, phase in enumerate(phases):
            for col, tt in enumerate(('non_opto', r.toi)):
                ro = r.readouts.get((phase, tt))
                if ro is not None and ro.update_matrix is not None:
                    plot_update_matrix(ro.update_matrix, ax=axes[row, col])
                axes[row, col].set_title(f'{phase} · {tt}', fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ── group pages (from per-animal results) ──────────────────────────────────

def group_psychometric_page(results: Dict[str, AnimalResult], title: str):
    """Per phase: the mean of per-animal curves per genotype (equal weight per animal)."""
    design = next(iter(results.values())).design
    phases = PHASES[design]
    fig, axes = plt.subplots(1, len(phases), figsize=(4.7 * len(phases), 4.0), squeeze=False)
    for ax, phase in zip(axes[0], phases):
        for g, col in (('wt', OFF), ('het', ON)):
            ys, x = [], None
            for r in results.values():
                ro = r.readouts.get((phase, 'all'))
                if r.genotype == g and ro is not None and ro.curve is not None and ro.curve.success:
                    ys.append(ro.curve.y)
                    x = ro.curve.x
            if ys:
                Y = np.stack(ys)
                ax.plot(x, Y.mean(0), color=col, lw=2, label=f'{g} (n={len(ys)})')
                if len(ys) > 1:
                    sem = Y.std(0, ddof=1) / np.sqrt(len(ys))
                    ax.fill_between(x, Y.mean(0) - sem, Y.mean(0) + sem, color=col, alpha=0.2)
        ax.axhline(0.5, ls='--', color='grey', alpha=0.3)
        ax.axvline(0, ls='--', color='grey', alpha=0.3)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Stimulus')
        ax.set_ylabel('P(choose B)')
        ax.set_title(f'{phase} · all trials', fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def group_update_matrix_page(results: Dict[str, AnimalResult], phase: str, toi: str, title: str):
    """Genotype-mean UM: rows het/wt, cols non_opto/toi."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), squeeze=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for row, g in enumerate(('het', 'wt')):
            for col, tt in enumerate(('non_opto', toi)):
                ums = [r.readouts[(phase, tt)].update_matrix for r in results.values()
                       if r.genotype == g and (phase, tt) in r.readouts
                       and r.readouts[(phase, tt)].update_matrix is not None]
                if ums:
                    plot_update_matrix(UpdateMatrix.average(ums, min_sources=2), ax=axes[row, col])
                axes[row, col].set_title(f'{g} · {tt} (n={len(ums)})', fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def swarm_page(g: GroupResult, kind: str, display: Sequence[str], title: str, ncols: int = 4):
    """Per-animal point differences by genotype, rank-test p in the corner."""
    df = g.rows[g.rows['kind'] == kind]
    tests = g.tests[g.tests['kind'] == kind].set_index('stat') if len(g.tests) and 'kind' in g.tests else pd.DataFrame()
    fig, axf = _grid(len(display), ncols, size=(3.3, 3.3))
    for ax, stat in zip(axf, display):
        p = tests.loc[stat, 'p'] if stat in tests.index else None
        plot_delta_swarm(df, stat, ax=ax, p_value=p, group_col='group', value_col='value')
    for ax in axf[len(display):]:
        fig.delaxes(ax)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ── adaptation (convergence index, trials since switch, log axis) ──────────

def _finish_convergence(ax, has_data: bool, empty_msg: str):
    ax.axhline(0.0, color='0.6', lw=1, zorder=1, label='expert-Uniform PSE')
    ax.axhline(1.0, color='crimson', ls='--', lw=1.2, zorder=1, label='normative PSE')
    ax.set_xscale('log')
    ax.set_xlabel('trials since switch (log)')
    ax.set_ylabel('convergence  (PSE − PSE_uni) / (PSE_norm − PSE_uni)')
    if not has_data and empty_msg:
        ax.text(0.5, 0.5, empty_msg, transform=ax.transAxes, ha='center', va='center', color='0.5')
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8)


def _draw_curve(ax, ad: SwitchAdaptation, colour: str, lw: float, alpha: float, label: Optional[str] = None):
    c = ad.curve.dropna(subset=['convergence'])
    if len(c):
        ax.plot(c['trial'], c['convergence'], color=colour, lw=lw, alpha=alpha, label=label, zorder=2)
        return True
    return False


def _mean_curve(ads: Sequence[SwitchAdaptation]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± SEM of convergence at each shared trial centre across items (equal weight each)."""
    frames = [ad.curve[['trial', 'convergence']].dropna().assign(i=k) for k, ad in enumerate(ads)]
    if not frames:
        return np.array([]), np.array([]), np.array([])
    df = pd.concat(frames)
    g = df.groupby('trial')['convergence']
    x = g.mean().index.to_numpy()
    return x, g.mean().to_numpy(), (g.std(ddof=1) / np.sqrt(g.count())).to_numpy()


def _dyn_text(ad: SwitchAdaptation) -> str:
    d = ad.dynamics
    parts = [f"τ={d.get('pse_tau', np.nan):.0f}{'*' if d.get('pse_censored', 0) else ''}",
             f"end={d.get('convergence_final', np.nan):.2f}",
             f"ΔAIC={d.get('pse_shape_daic', np.nan):+.1f}"]
    return ', '.join(parts)


def adaptation_page(r: AnimalResult, title: str):
    """Per animal: thin line per session type (opto vs masking) with the fit summary in the legend."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    drew = False
    for stype, ad in r.adaptation.items():
        col = TYPE_COL.get(stype, '0.4')
        drew |= _draw_curve(ax, ad, col, 2.0, 0.9, label=f'{stype} ({ad.n_sessions} sess) · {_dyn_text(ad)}')
    _finish_convergence(ax, drew, 'no adaptation data at this phase')
    fig.suptitle(title + '   (* = τ censored: not plateaued within the block)', fontsize=10)
    fig.tight_layout()
    return fig


def group_adaptation_page(g: GroupResult, title: str):
    """1×4: HET opto-vs-masking | WT opto-vs-masking | masking HET-vs-WT | opto HET-vs-WT.
    Thin = each animal, thick = mean across animals with SEM band."""
    het = [a for a in g.animals if g.by_animal.get(a) == 'het']
    wt = [a for a in g.animals if g.by_animal.get(a) == 'wt']

    def panel(ax, series, empty_msg):
        has = False
        for label, aids, stype, col in series:
            ads = [g.adaptation[(a, stype)] for a in aids if (a, stype) in g.adaptation]
            for ad in ads:
                has |= _draw_curve(ax, ad, col, 0.7, 0.35)
            x, m, sem = _mean_curve(ads)
            if x.size:
                ax.plot(x, m, color=col, lw=2.6, zorder=4, label=f'{label} (n={len(ads)})')
                ax.fill_between(x, m - sem, m + sem, color=col, alpha=0.15, zorder=3)
        _finish_convergence(ax, has, empty_msg)

    fig, ax = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
    panel(ax[0], [(st, het, st, TYPE_COL[st]) for st in TYPE_COL], 'no HET sessions')
    ax[0].set_title(f'HET (n={len(het)}) · opto vs masking', fontsize=10)
    panel(ax[1], [(st, wt, st, TYPE_COL[st]) for st in TYPE_COL], 'no WT sessions')
    ax[1].set_title(f'WT (n={len(wt)}) · opto vs masking', fontsize=10)
    panel(ax[2], [('het', het, 'masking', GENO_COL['het']), ('wt', wt, 'masking', GENO_COL['wt'])], 'no masking')
    ax[2].set_title('masking · HET vs WT', fontsize=10)
    panel(ax[3], [('het', het, 'opto', GENO_COL['het']), ('wt', wt, 'opto', GENO_COL['wt'])], 'no opto')
    ax[3].set_title('opto · HET vs WT', fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig