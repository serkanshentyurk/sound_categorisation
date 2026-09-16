"""
Report figures. Every function takes an AnimalResult / GroupResult (or its
tables) and returns a matplotlib Figure; nothing here computes a statistic.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from behav_utils.plotting import (
    plot_interaction_single,
    plot_psychometric_curve,
    plot_stat_comparison_single,
    plot_update_matrix,
)
from behav_utils.readouts import UpdateMatrix

from sound_categorisation.plotting.opto import plot_delta_swarm
from sound_categorisation.reports.compute import PHASES, AnimalResult, GroupResult

__all__ = ['contrast_grid', 'interaction_grid', 'psychometric_page', 'update_matrix_page',
           'group_psychometric_page', 'group_update_matrix_page', 'swarm_page',
           'trajectory_page', 'group_trajectory_page']

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


# ── session trajectory (per session, in order) ─────────────────────────────

DIST_COL = {'Hard-A': '#1f77b4', 'Hard-B': '#ff7f0e', 'Uniform': '#7f7f7f'}


def _laser_boundary(ax, sess: pd.DataFrame):
    """Vertical line where the session type changes (laser half → masking half)."""
    types = list(sess['session_type'])
    for i in range(1, len(types)):
        if types[i] != types[i - 1]:
            ax.axvline(sess['order'].iloc[i] - 0.5, color='0.5', ls=':', lw=1)


def _trajectory_axis(ax, sess: pd.DataFrame, y: str, colour_by_dist=True, label=None, lw=1.6, alpha=0.9, marker='o'):
    x = sess['order'].to_numpy()
    v = sess[y].to_numpy(dtype=float)
    ax.plot(x, v, '-', color='0.4', lw=lw * 0.6, alpha=alpha * 0.6, zorder=1)
    if colour_by_dist:
        for d, c in DIST_COL.items():
            m = sess['distribution'] == d
            if m.any():
                ax.plot(x[m], v[m], marker, color=c, ms=5, alpha=alpha, zorder=3, label=d if label is None else None)
    else:
        ax.plot(x, v, marker + '-', color=label[1], lw=lw, ms=4, alpha=alpha, zorder=2, label=label[0])


def trajectory_page(r: AnimalResult, title: str):
    """Per session in order: PSE, delta from previous session, tau, convergence; laser/masking boundary marked."""
    tr = r.trajectory
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    if tr is None or not len(tr.sessions):
        for ax in axes.ravel():
            ax.text(0.5, 0.5, 'no trajectory', transform=ax.transAxes, ha='center', color='0.5')
        fig.suptitle(title, fontsize=11)
        return fig
    sess = tr.sessions
    panels = [('pse', 'criterion (PSE) fitted per session, all trials'),
              ('delta_from_prev', 'PSE change from the previous day (switch amplitude)'),
              ('pse_tau_fixed', 'τ per session: trials for the within-session PSE to move ~63% of the way to its endpoint\n'
                                '(exponential fit, shape pinned; × = did not plateau within the session)'),
              ('convergence_final', 'convergence at session end: 0 = previous day\'s PSE, 1 = normative PSE')]
    for ax, (y, lab) in zip(axes.ravel(), panels):
        if y not in sess:
            ax.set_visible(False)
            continue
        _trajectory_axis(ax, sess, y)
        _laser_boundary(ax, sess)
        ax.axhline(0 if y != 'pse_tau' else 0, color='0.7', lw=0.8, zorder=0)
        if y == 'pse':
            ax.axhline(tr.baseline_pse, color='0.3', ls='--', lw=0.8, label='expert-Uniform PSE')
            for d, c in DIST_COL.items():
                m = sess['distribution'] == d
                if m.any() and np.isfinite(sess.loc[m, 'normative_pse']).any():
                    ax.axhline(sess.loc[m, 'normative_pse'].iloc[0], color=c, ls=':', lw=0.8)
        if y == 'pse_tau_fixed':
            ax.set_yscale('log')
            cens = sess.get('pse_censored_fixed', pd.Series(dtype=float)) == 1
            if cens.any():
                ax.plot(sess.loc[cens, 'order'], sess.loc[cens, 'pse_tau_fixed'], 'x', color='k', ms=7, label='censored')
        if y == 'convergence_final':
            ax.axhline(1, color='crimson', ls='--', lw=0.8, label='normative')
            ax.set_ylim(-3, 3)
        ax.set_xticks(sess['order'])
        ax.set_xticklabels([f"{t[:1]}{'L' if st in ('opto', 'alm_control_uni', 'alm_control_bi') else 'm'}"
                            for t, st in zip(sess['distribution'].astype(str), sess['session_type'])], fontsize=7)
        ax.set_xlabel('session (A/B = distribution, L = laser, m = masking)', fontsize=8)
        ax.set_title(lab, fontsize=8.5)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f'{title}   (σ={tr.sigma:.2f} for the normative PSE)', fontsize=11)
    fig.tight_layout()
    return fig


def group_trajectory_page(g: GroupResult, title: str):
    """PSE and delta-from-previous per session, one line per animal (genotype colour), WT and HET panels."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex='col')
    for col, geno in enumerate(('wt', 'het')):
        ids = [a for a in g.animals if g.by_animal.get(a) == geno and a in g.trajectories]
        for row, (y, lab) in enumerate((('pse', 'PSE per session'), ('delta_from_prev', 'PSE − previous session'))):
            ax = axes[row, col]
            for k, aid in enumerate(ids):
                sess = g.trajectories[aid].sessions
                if y not in sess:
                    continue
                ax.plot(sess['order'], sess[y], '-', color=GENO_COL[geno], alpha=0.35, lw=1)
                for d, c in DIST_COL.items():
                    m = sess['distribution'] == d
                    ax.plot(sess.loc[m, 'order'], sess.loc[m, y], 'o', color=c, ms=4, alpha=0.8)
                if k == 0:
                    _laser_boundary(ax, sess)
            if ids:
                # mean across animals at each order
                stack = pd.concat([g.trajectories[a].sessions[['order', y]].assign(a=a) for a in ids if y in g.trajectories[a].sessions])
                m = stack.groupby('order')[y].mean()
                ax.plot(m.index, m.to_numpy(), '-', color=GENO_COL[geno], lw=2.5, zorder=4, label=f'{geno} mean (n={len(ids)})')
            ax.axhline(0, color='0.7', lw=0.8)
            ax.set_title(f'{geno.upper()} · {lab}', fontsize=9)
            ax.legend(frameon=False, fontsize=7)
            if row == 1:
                ax.set_xlabel('session order (laser half | masking half)', fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig
