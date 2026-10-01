"""
Switch report (pre-opto cohorts): adaptation across whole blocks after a distribution switch.

    python -m sound_categorisation.reports switches --cohort behaviour1-cohort

Per animal: every qualifying switch (blocks ≥ 1000 trials) → SwitchResult; per cohort:
tables (switches.csv, convergence.csv, sessions.csv, overnight.csv), a per-animal PDF, and
a two-page summary (Fig. 5C in three methods; speed/plateau/overnight by transition).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from sound_categorisation.adaptation import (
    SwitchResult,
    animal_switches,
    compute_phase_curves,
    compute_switches,
    flag_biased_sessions,
)
from sound_categorisation.reports.tables import write_result

__all__ = ['SwitchesResult', 'compute_switches_cohort', 'switch_tables', 'switches_summary', 'run_switches']

TRANS_COL = {'first': '#7f7f7f', 'novel': '#1f77b4', 'return': '#ff7f0e'}
METHOD_TITLE = {'manuscript': 'manuscript method: 4-param fit per 50-trial bin, clipped to [0, 1]',
                'pinned': 'shape pinned (only the criterion fitted per 50-trial bin), unclipped',
                'pinned_running': 'shape pinned, running 50-trial windows (step 10), unclipped'}


@dataclass(frozen=True)
class SwitchesResult:
    cohort: str
    results: Dict[str, List[SwitchResult]] = field(default_factory=dict)    # animal → switches
    phase_params: pd.DataFrame = field(default_factory=pd.DataFrame)        # per animal × phase psychometric fit
    phase_curves: pd.DataFrame = field(default_factory=pd.DataFrame)        # per animal × phase curve (x, y)

    @property
    def animals(self) -> List[str]:
        return list(self.results)

    def __repr__(self) -> str:
        n = sum(len(v) for v in self.results.values())
        return f'SwitchesResult({self.cohort!r}, n_animals={len(self.results)}, n_switches={n})'


def compute_switches_cohort(experiment, animals: Sequence[str], *, cohort: str = '', **kw) -> SwitchesResult:
    out, params, curves = {}, [], []
    for aid in animals:
        if aid not in experiment.animals:
            continue
        animal = experiment.animals[aid]
        try:
            res = compute_switches(animal, **kw)
            sessions, switches = animal_switches(animal, min_block_trials=kw.get('min_block_trials', 1000))
            p, c = compute_phase_curves(animal, sessions, switches)
        except Exception as exc:                          # one animal must not kill the batch
            import warnings
            warnings.warn(f'{aid}: switches failed: {exc}', stacklevel=2)
            continue
        if res:
            out[aid] = res
            params.append(p)
            curves.append(c)
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    return SwitchesResult(cohort, out, cat(params), cat(curves))


def switch_tables(g: SwitchesResult) -> Dict[str, pd.DataFrame]:
    lab = {'cohort': g.cohort}
    rows, conv, sess, over = [], [], [], []
    for aid, results in g.results.items():
        for r in results:
            key = {'animal': aid, 'switch_idx': r.switch_idx, 'from_distribution': r.from_distribution,
                   'to_distribution': r.to_distribution, 'transition': r.transition, **lab}
            rows.append(r.to_rows().assign(**lab))
            conv.append(r.convergence.assign(**key))
            sess.append(r.sessions.assign(**key))
            over.append(r.overnight.assign(**key))
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    sessions = cat(sess)
    if len(sessions):
        sessions = flag_biased_sessions(sessions)
    sw = cat(rows)
    # pre/post per switch from the block-pooled 4-parameter fits (robust) + the pinned last-250-trial values
    pre_post = pd.DataFrame()
    if len(sw) and len(g.phase_params):
        keys = ['animal', 'switch_idx', 'transition', 'from_distribution', 'to_distribution']
        pv = (sw.groupby(keys + ['stat'])['value'].first().unstack('stat').reset_index())
        pp = pv[keys].copy()
        pp['pse_pre_last250'] = pv.get('pre_pse')
        pp['pse_post_last250'] = pv.get('pse_post')
        pp['normative_pse'] = pv.get('normative_pse')
        P = g.phase_params
        pooled = {(r.animal, int(r.order)): r.pse for r in P.itertuples()}   # order 0 = Uniform ref, k = switch k-1's block
        pp['pse_pre'] = [pooled.get((a, int(k))) for a, k in zip(pp['animal'], pp['switch_idx'])]
        pp['pse_post'] = [pooled.get((a, int(k) + 1)) for a, k in zip(pp['animal'], pp['switch_idx'])]
        pp['delta'] = pp['pse_post'] - pp['pse_pre']
        pre_post = pp.assign(**lab)
    return {'switches': sw, 'pre_post': pre_post, 'convergence': cat(conv), 'sessions': sessions, 'overnight': cat(over),
            'psychometrics': g.phase_params.assign(**lab) if len(g.phase_params) else pd.DataFrame(),
            'psychometric_curves': g.phase_curves.assign(**lab) if len(g.phase_curves) else pd.DataFrame()}


# ── figures (read tables only) ──────────────────────────────────────────────

def _biased(T: Dict[str, pd.DataFrame]) -> set:
    se = T.get('sessions', pd.DataFrame())
    return set(se.loc[se['biased_animal'].astype(bool), 'animal']) if len(se) and 'biased_animal' in se else set()


def _subset(df: pd.DataFrame, T, clean: bool) -> pd.DataFrame:
    if not clean or not len(df):
        return df
    return df[~df['animal'].isin(_biased(T))]


def _subset_label(T, clean: bool) -> str:
    b = sorted(_biased(T))
    return (f'clean animals only (excluded: {", ".join(b)})' if b else 'all animals (none flagged)') if clean else 'all animals'

def _band(df: pd.DataFrame, value: str, by: str = 'animal', B: int = 1000, seed: int = 42):
    """Per-animal mean per trial bin, then mean and bootstrap-over-animals band per bin."""
    pv = df.groupby([by, 'trial'])[value].mean().unstack('trial')
    x = pv.columns.to_numpy(float)
    V = pv.to_numpy(float)
    mean = np.nanmean(V, axis=0)
    rng = np.random.default_rng(seed)
    lo, hi = np.full(len(x), np.nan), np.full(len(x), np.nan)
    for j in range(len(x)):
        col = V[:, j][np.isfinite(V[:, j])]
        if col.size < 2:
            continue
        boots = np.array([rng.choice(col, col.size, replace=True).mean() for _ in range(B)])
        lo[j], hi[j] = np.percentile(boots, [2.5, 97.5])
    return x, mean, lo, hi, V.shape[0]


def _curve_with_anchor(sub: pd.DataFrame, value: str):
    """Manuscript aggregation: per animal, mean over its switches per bin; a q=0 anchor at x=1 with value 0;
    mean and bootstrap band over animals."""
    per = sub.groupby(['animal', 'trial'])[value].mean().unstack('trial')
    per.insert(0, 1.0, 0.0)                                   # the pre-switch point, convergence = 0 by definition
    x = per.columns.to_numpy(float)
    V = per.to_numpy(float)
    mean = np.nanmean(V, axis=0)
    rng = np.random.default_rng(42)
    lo, hi = np.full(len(x), np.nan), np.full(len(x), np.nan)
    for j in range(len(x)):
        col = V[:, j][np.isfinite(V[:, j])]
        if col.size < 2:
            continue
        boots = np.array([rng.choice(col, col.size, replace=True).mean() for _ in range(1000)])
        lo[j], hi[j] = np.percentile(boots, [2.5, 97.5])
    return x, mean, lo, hi, V.shape[0]


def convergence_page(T: Dict[str, pd.DataFrame], cohort: str, clean: bool = True):
    """Fig. 5C, two ways: (left) the manuscript's aggregation — asymmetric switches pooled per animal, anchored at 0 —
    with the first switch as a separate line; (right) the same data split by transition type."""
    conv = _subset(T['convergence'], T, clean)
    d = conv[(conv['method'] == 'manuscript') & np.isfinite(conv['convergence_clipped'])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)

    ax = axes[0]
    asym = d[d['transition'].isin(['novel', 'return'])]
    if len(asym):
        x, m, lo, hi, n = _curve_with_anchor(asym, 'convergence_clipped')
        ax.plot(x, m, color='#5db9bb', lw=2.4, label=f'Hard↔Hard switches pooled (n={n} mice)')
        ax.fill_between(x, lo, hi, color='#5db9bb', alpha=0.25)
    first = d[d['transition'] == 'first']
    if len(first):
        x, m, lo, hi, n = _curve_with_anchor(first, 'convergence_clipped')
        ax.plot(x, m, color=TRANS_COL['first'], lw=1.8, ls='--', label=f'first switch Uniform→Hard-B (n={n}; not in the manuscript)')
        ax.fill_between(x, lo, hi, color=TRANS_COL['first'], alpha=0.12)
    ax.set_title('as in the manuscript: asymmetric switches pooled per mouse, pre-switch point fixed at 0', fontsize=9)

    ax = axes[1]
    for tr, sub in d.groupby('transition'):
        x, m, lo, hi, n = _curve_with_anchor(sub, 'convergence_clipped')
        ax.plot(x, m, color=TRANS_COL.get(tr, '0.3'), lw=2.0, label=f'{tr} (n={n} mice)')
        ax.fill_between(x, lo, hi, color=TRANS_COL.get(tr, '0.3'), alpha=0.15)
    ax.set_title('the same, split by transition (first = Uniform→B, novel = B→A, return = A→B)', fontsize=9)

    for ax in axes:
        ax.axhline(1.0, ls='--', color='k', lw=1, label='converged to normative value')
        ax.axhline(0.0, color='0.6', lw=0.8)
        ax.set_xscale('log')
        ax.set_xlim(1, 3000)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('trial number after the switch (log scale)')
        ax.legend(frameon=False, fontsize=8, loc='lower right')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('convergence  (0 = pre-switch PSE, 1 = normative PSE)')
    fig.suptitle(f'{cohort} — dynamics of adaptation (manuscript Fig. 5C recipe), {_subset_label(T, clean)}\n'
                 'PSE from a 4-parameter fit on each 50-trial bin after the switch, clipped to [0, 1]; pre = last 250 trials before the switch; '
                 'normative = constant-σ observer at each mouse\'s σ; mean ± 95% bootstrap over mice', fontsize=10)
    fig.text(0.5, 0.01, 'The point at trial 1 is the pre-switch value and is 0 by definition (as in the manuscript). Each bin is clipped '
             'to [0, 1] before averaging; with noisy 50-trial fits this alone puts a bin near 0.5, so the level of the first fitted bin '
             'is not an adaptation measure. Unclipped and shape-pinned versions are in convergence.csv and the per-animal PDFs.',
             ha='center', va='bottom', fontsize=8, color='0.3', wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def _swarm(ax, df, value, ylabel, logy=False):
    trs = [t for t in ('first', 'novel', 'return') if t in set(df['transition'])]
    for i, tr in enumerate(trs):
        d = df[df['transition'] == tr][value].dropna()
        if d.empty:
            continue
        j = np.linspace(-0.2, 0.2, len(d)) if len(d) > 1 else np.zeros(1)
        ax.plot(i + j, d.to_numpy(), 'o', color=TRANS_COL[tr], ms=6, alpha=0.8)
        ax.plot([i - 0.28, i + 0.28], [d.median()] * 2, color='k', lw=1.5)
    ax.set_xticks(range(len(trs)))
    ax.set_xticklabels(trs)
    ax.set_ylabel(ylabel)
    if logy and (df[value] > 0).any():
        ax.set_yscale('log')
    ax.spines[['top', 'right']].set_visible(False)


def dynamics_page(T: Dict[str, pd.DataFrame], cohort: str, clean: bool = True):
    sw = _subset(T['switches'], T, clean).pivot_table(index=['animal', 'switch_idx', 'transition', 'to_distribution'], columns='stat',
                                   values='value', dropna=False).reset_index()
    for col in ('pse_tau', 'trials_to_criterion', 'plateau', 'pse_shape_daic'):
        if col not in sw:
            sw[col] = np.nan
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    _swarm(axes[0, 0], sw, 'pse_tau', 'τ (trials) — exponential fit on the block, shape pinned', logy=True)
    axes[0, 0].set_title('speed of the criterion shift (log)', fontsize=10)
    _swarm(axes[0, 1], sw, 'trials_to_criterion', 'trials to reach 80 % convergence (running windows)', logy=True)
    axes[0, 1].set_title('trials to criterion', fontsize=10)
    _swarm(axes[0, 2], sw, 'plateau', 'mean convergence over the last third of the block (pinned, unclipped)')
    axes[0, 2].axhline(1, ls='--', color='k', lw=1)
    axes[0, 2].axhline(0, color='0.6', lw=0.8)
    axes[0, 2].set_title('plateau relative to normative', fontsize=10)
    _swarm(axes[1, 0], sw, 'pse_shape_daic', 'AIC(exponential) − AIC(step);  < 0 gradual, > 0 step')
    axes[1, 0].axhline(0, color='0.6', lw=0.8)
    axes[1, 0].set_title('trajectory shape', fontsize=10)
    # overnight: change in PSE from the end of one session to the start of the next, signed toward the pre-switch value
    ov = _subset(T['overnight'], T, clean)
    ax = axes[1, 1]
    if len(ov):
        d = ov.dropna(subset=['toward_pre'])
        per = d.groupby(['animal', 'transition'])['toward_pre'].mean().reset_index()
        _swarm(ax, per, 'toward_pre', 'overnight PSE change, signed toward pre-switch PSE\n(> 0 = forgets toward yesterday\'s criterion)')
        ax.axhline(0, color='0.6', lw=0.8)
    ax.set_title('overnight: last 100 trials → first 100 of the next session', fontsize=9)
    # within-block per-session PSE (pinned), order in block, one thin line per animal×switch, mean per transition
    ax = axes[1, 2]
    se = _subset(T['sessions'], T, clean)
    if len(se):
        for tr, sub in se.groupby('transition'):
            for (a, k), s in sub.groupby(['animal', 'switch_idx']):
                s = s.sort_values('order_in_block')
                ax.plot(s['order_in_block'], s['pse_fixed'] - s['pse_fixed'].iloc[0], color=TRANS_COL[tr], alpha=0.25, lw=0.8)
            m = sub.groupby('order_in_block')['pse_fixed'].mean()
            m0 = sub.groupby(['animal', 'switch_idx'])['pse_fixed'].first().mean()
            ax.plot(m.index, m.to_numpy() - m0, color=TRANS_COL[tr], lw=2.2, label=tr)
        ax.axhline(0, color='0.6', lw=0.8)
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel('session within the block')
    ax.set_ylabel('PSE (shape pinned) − first session of the block')
    ax.set_title('within-block drift, session by session', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'{cohort} — switch dynamics by transition type, {_subset_label(T, clean)} (first = Uniform→Hard-B, novel = Hard-B→Hard-A, '
                 'return = Hard-A→Hard-B; type and destination are confounded by design). One dot = one mouse × switch, bar = median', fontsize=10)
    fig.tight_layout()
    return fig


def animal_page(results: List[SwitchResult], aid: str):
    fig, axes = plt.subplots(len(results), 2, figsize=(12, 3.2 * len(results)), squeeze=False)
    for row, r in enumerate(results):
        ax = axes[row, 0]
        for method, c in (('manuscript', '0.5'), ('pinned', '#d62728'), ('pinned_running', '#1f77b4')):
            d = r.convergence[r.convergence['method'] == method]
            v = 'convergence_clipped' if method == 'manuscript' else 'convergence'
            ax.plot(d['trial'], d[v], '-' if method == 'pinned_running' else 'o-', color=c, ms=3, lw=1.2, label=method)
        ax.axhline(1, ls='--', color='k', lw=0.8)
        ax.axhline(0, color='0.6', lw=0.8)
        ax.set_xlim(1, max(10, int(r.convergence['trial'].max()) if len(r.convergence) else 10))
        ax.set_xscale('log')
        ax.set_ylim(-1, 2)
        ax.set_title(f'{aid} switch {r.switch_idx}: {r.from_distribution}→{r.to_distribution} [{r.transition}]  '
                     f'pre PSE {r.pre["pse"]:+.2f}, normative {r.normative_pse:+.2f}, τ={r.dynamics["pse_tau"]:.0f}'
                     f'{"*" if r.dynamics["pse_censored"] else ""}, plateau {r.dynamics["plateau"]:.2f}', fontsize=8)
        ax.legend(frameon=False, fontsize=7)
        ax = axes[row, 1]
        s = r.sessions
        ax.plot(s['order_in_block'], s['pse_fixed'], 'o-', color='#d62728', ms=4, label='PSE, shape pinned')
        ax.plot(s['order_in_block'], s['pse'], 's--', color='0.5', ms=3, label='PSE, 4-param')
        ax.axhline(r.pre['pse'], color='0.3', ls=':', label='pre-switch PSE')
        ax.axhline(r.normative_pse, color='crimson', ls='--', label='normative')
        ax.set_xlabel('session within block')
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


# ── QC and psychometrics pages ───────────────────────────────────────────────

DIST_COL = {'Uniform': '#7f7f7f', 'Hard-A': '#c23f63', 'Hard-B': '#007dc7'}


def qc_page(T: Dict[str, pd.DataFrame], cohort: str):
    """Per animal × session timeline of accuracy, |PSE|, lapses; flagged sessions marked; biased animals named."""
    se = T['sessions'].sort_values(['animal', 'switch_idx', 'order_in_block'])
    animals = list(dict.fromkeys(se['animal']))
    fig, axes = plt.subplots(len(animals), 1, figsize=(14, 2.0 * len(animals)), sharex=False, squeeze=False)
    for ax, aid in zip(axes[:, 0], animals):
        d = se[se['animal'] == aid].reset_index(drop=True)
        x = np.arange(len(d))
        biased = bool(d['biased_animal'].iloc[0])
        for dist, c in DIST_COL.items():
            m = d['to_distribution'] == dist
            ax.scatter(x[m], d.loc[m, 'accuracy'], s=14, color=c, label=dist if m.any() else None, zorder=3)
        ax.plot(x, d['pse_fixed'].clip(-1.0, 1.0), '-', color='0.4', lw=1, label='PSE (pinned)')
        ax.axhline(0.4, color='0.85', lw=0.6, ls='--')
        ax.axhline(-0.4, color='0.85', lw=0.6, ls='--')
        ax.axhline(0, color='0.85', lw=0.6)
        ax.plot(x, np.maximum(d['lapse_low'], d['lapse_high']), ':', color='0.6', lw=1, label='max lapse')
        fl = d['flagged'].to_numpy(bool)
        if fl.any():
            ax.scatter(x[fl], np.full(fl.sum(), 1.05), marker='v', s=18, color='k', label='flagged')
        ax.set_ylim(-1.05, 1.12)
        ax.set_ylabel(aid + ('\nBIASED' if biased else ''), fontsize=8, color='crimson' if biased else 'k')
        ax.set_yticks([-1, -0.4, 0, 0.4, 1])
        ax.tick_params(axis='y', labelsize=6)
        for b in d.index[d['order_in_block'] == 0][1:]:
            ax.axvline(b - 0.5, color='0.5', ls='--', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=6, loc='lower left', bbox_to_anchor=(0, 1.05))
    axes[-1, 0].set_xlabel('session within the Hard phase (blocks separated by dashed lines)')
    fig.suptitle(f'{cohort} — cohort QC. Dots = accuracy (colour = distribution); line = signed PSE (dashed = ±0.4); dotted = larger lapse; '
                 f'▼ = flagged (|PSE| > 0.4, lapse > 0.3 or accuracy < 0.6). BIASED = majority flagged with the same PSE sign on A and B days.', fontsize=9)
    fig.tight_layout()
    return fig


def _phase_order(params: pd.DataFrame) -> List[str]:
    return list(params.drop_duplicates('phase').sort_values('order')['phase'])


def psychometrics_animal_page(T: Dict[str, pd.DataFrame], aid: str):
    P = T['psychometrics']
    C = T['psychometric_curves']
    p, c = P[P['animal'] == aid], C[C['animal'] == aid]
    phases = _phase_order(p)
    fig, axes = plt.subplots(1, len(phases), figsize=(2.9 * len(phases), 3.2), squeeze=False)
    for ax, ph in zip(axes[0], phases):
        r = p[p['phase'] == ph].iloc[0]
        cc = c[c['phase'] == ph]
        ax.plot(cc['x'], cc['y'], color=DIST_COL.get(r['distribution'], '0.3'), lw=2)
        ax.axhline(0.5, ls='--', color='0.7', lw=0.7)
        ax.axvline(0, ls='--', color='0.7', lw=0.7)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.set_title(ph, fontsize=9, color=DIST_COL.get(r['distribution'], 'k'))
        ax.text(0.03, 0.97, f"PSE {r['pse']:+.2f}\nσ {r['sigma']:.2f}\nacc {r['accuracy']:.2f}\nn {int(r['n_trials'])} / {int(r['n_sessions'])} s",
                transform=ax.transAxes, va='top', fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_ylabel('P(choose B)')
    biased = bool(T['sessions'].loc[T['sessions']['animal'] == aid, 'biased_animal'].iloc[0]) if len(T['sessions']) else False
    fig.suptitle(f'{aid} — psychometrics by phase' + ('   [BIASED — excluded from the combined pages]' if biased else ''),
                 fontsize=10, color='crimson' if biased else 'k')
    fig.tight_layout()
    return fig


def psychometrics_combined_page(T: Dict[str, pd.DataFrame], cohort: str, exclude_biased: bool = True):
    P, C, se = T['psychometrics'], T['psychometric_curves'], T['sessions']
    biased = set(se.loc[se['biased_animal'].astype(bool), 'animal']) if len(se) else set()
    keep = [a for a in P['animal'].unique() if not (exclude_biased and a in biased)]
    P, C = P[P['animal'].isin(keep)], C[C['animal'].isin(keep)]
    phases = _phase_order(P)
    fig = plt.figure(figsize=(2.9 * len(phases) + 4, 3.6))
    gs = fig.add_gridspec(1, len(phases) + 1, width_ratios=[1] * len(phases) + [1.4])
    for k, ph in enumerate(phases):
        ax = fig.add_subplot(gs[0, k])
        cc = C[C['phase'] == ph]
        dist = P.loc[P['phase'] == ph, 'distribution'].iloc[0]
        Y = []
        for a, d in cc.groupby('animal'):
            ax.plot(d['x'], d['y'], color=DIST_COL.get(dist, '0.3'), lw=0.6, alpha=0.3)
            Y.append(d['y'].to_numpy())
        if Y:
            Y = np.stack(Y)
            x = cc[cc['animal'] == cc['animal'].iloc[0]]['x'].to_numpy()
            m = Y.mean(0)
            ax.plot(x, m, color=DIST_COL.get(dist, '0.3'), lw=2.2)
            if len(Y) > 1:
                sem = Y.std(0, ddof=1) / np.sqrt(len(Y))
                ax.fill_between(x, m - sem, m + sem, color=DIST_COL.get(dist, '0.3'), alpha=0.2)
        ax.axhline(0.5, ls='--', color='0.7', lw=0.7)
        ax.axvline(0, ls='--', color='0.7', lw=0.7)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{ph} (n={len(Y)})', fontsize=9, color=DIST_COL.get(dist, 'k'))
        ax.spines[['top', 'right']].set_visible(False)
        if k == 0:
            ax.set_ylabel('P(choose B)')
    # paired PSE per animal: Hard-B #k vs Hard-A #k, then all-B vs all-A (Fig. 5B right style)
    ax = fig.add_subplot(gs[0, -1])
    pairs = []
    for k in (1, 2):
        if f'Hard-B #{k}' in phases and f'Hard-A #{k}' in phases:
            pairs.append((f'Hard-B #{k}', f'Hard-A #{k}'))
    if 'Hard-B all' in phases and 'Hard-A all' in phases:
        pairs.append(('Hard-B all', 'Hard-A all'))
    x0 = 0
    for b, a in pairs:
        pv = P[P['phase'].isin([b, a])].pivot_table(index='animal', columns='phase', values='pse')
        for _, r in pv.iterrows():
            ax.plot([x0, x0 + 1], [r[b], r[a]], '-', color='0.6', lw=0.7)
        ax.plot([x0] * len(pv), pv[b], 'o', color=DIST_COL['Hard-B'], ms=5)
        ax.plot([x0 + 1] * len(pv), pv[a], 'o', color=DIST_COL['Hard-A'], ms=5)
        ax.errorbar([x0, x0 + 1], [pv[b].mean(), pv[a].mean()], yerr=[pv[b].sem(), pv[a].sem()], color='k',
                    marker='s', ms=6, capsize=3, lw=1.5)
        ax.text(x0 + 0.5, ax.get_ylim()[1] if False else -0.9, f"{b.split()[-1]}\nΔ={(pv[a] - pv[b]).mean():+.2f}",
                ha='center', fontsize=7)
        x0 += 2.5
    ax.axhline(0, color='0.7', lw=0.7)
    ax.set_xticks([i * 2.5 + 0.5 for i in range(len(pairs))])
    ax.set_xticklabels([f'{b.split()[-1]}' for b, _ in pairs])
    ax.set_ylim(-1, 1)
    ax.set_ylabel('PSE (blue = Hard-B, red = Hard-A)')
    ax.set_title('paired PSE per mouse, by block pair', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    note = f'excluding biased: {sorted(biased)}' if exclude_biased and biased else 'all animals'
    fig.suptitle(f'{cohort} — psychometrics by phase, thin = mouse, thick = mean ± SEM over mice ({note})', fontsize=10)
    fig.tight_layout()
    return fig


def pre_post_page(T: Dict[str, pd.DataFrame], cohort: str, clean: bool = True):
    """Per transition type: PSE before the switch vs at the end of the block, one line per mouse, normative dashed."""
    pp = _subset(T['pre_post'], T, clean)
    trs = [t for t in ('first', 'novel', 'return') if t in set(pp['transition'])]
    fig, axes = plt.subplots(1, max(len(trs), 1), figsize=(4.2 * max(len(trs), 1), 4.4), squeeze=False)
    for ax, tr in zip(axes[0], trs):
        d = pp[pp['transition'] == tr].dropna(subset=['pse_pre', 'pse_post'])
        for r in d.itertuples():
            ax.plot([0, 1], [r.pse_pre, r.pse_post], '-', color='0.6', lw=0.8)
            ax.annotate(str(r.animal).replace('SS', ''), (1, r.pse_post), textcoords='offset points', xytext=(4, 0), fontsize=6, color='0.4', va='center')
        ax.plot([0] * len(d), d['pse_pre'], 'o', color='0.3', ms=5)
        for dist, c in DIST_COL.items():
            m = d['to_distribution'] == dist
            ax.plot([1] * m.sum(), d.loc[m, 'pse_post'], 'o', color=c, ms=5)
            if m.any():
                ax.hlines(d.loc[m, 'normative_pse'].mean(), 0.85, 1.15, colors=c, linestyles='--', lw=1)
        if len(d):
            ax.errorbar([0, 1], [d['pse_pre'].mean(), d['pse_post'].mean()], yerr=[d['pse_pre'].sem(), d['pse_post'].sem()],
                        color='k', marker='s', ms=6, capsize=3, lw=1.5, zorder=4)
            delta = (d['pse_post'] - d['pse_pre'])
            # direction of the switch: +1 when the new distribution is Hard-A (criterion should rise), -1 for Hard-B
            direction = np.where(d['to_distribution'] == 'Hard-A', 1.0, -1.0)
            with_switch = np.sign(delta.to_numpy()) == direction
            ax.set_title(f"{tr}: {' / '.join(sorted(set(d['from_distribution'])))} → {' / '.join(sorted(set(d['to_distribution'])))}\n"
                         f"Δ = {delta.mean():+.2f}, moved with the switch in {int(with_switch.sum())}/{len(d)}", fontsize=9)
        ax.axhline(0, color='0.8', lw=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['before the switch\n(previous phase, pooled)', 'after the switch\n(whole block, pooled)'], fontsize=8)
        ax.set_xlim(-0.4, 1.5)
        ax.set_ylim(-1, 1.3)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_ylabel('PSE (4-parameter fit on pooled trials)')
    fig.suptitle(f'{cohort} — criterion before and after each switch, {_subset_label(T, clean)}. '
                 'Pre = the previous phase pooled (Uniform: its last 5 sessions), post = the whole new block pooled. Dashed = normative PSE (colour = distribution)', fontsize=10)
    fig.tight_layout()
    return fig


def switches_summary(T: Dict[str, pd.DataFrame], cohort: str, out_dir: Path) -> Path:
    path = Path(out_dir) / 'summary_switches.pdf'
    has_psy = len(T.get('psychometrics', []))
    pages = [qc_page(T, cohort)]
    if has_psy:
        pages.append(psychometrics_combined_page(T, cohort, exclude_biased=True))
    pages += [pre_post_page(T, cohort, clean=True), convergence_page(T, cohort, clean=True), dynamics_page(T, cohort, clean=True)]
    if has_psy:
        pages.append(psychometrics_combined_page(T, cohort, exclude_biased=False))
    pages.append(convergence_page(T, cohort, clean=False))
    with PdfPages(path) as pdf:
        for fig in pages:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    return path


def run_switches(experiment, animals: Sequence[str], out_root: Path, cohort: str, meta: dict | None = None,
                 **kw) -> Path:
    matplotlib.use('Agg')
    g = compute_switches_cohort(experiment, animals, cohort=cohort, **kw)
    out = Path(out_root) / cohort / 'switches'
    (out / 'pdf').mkdir(parents=True, exist_ok=True)
    T = switch_tables(g)
    write_result(out, T, None, {**(meta or {}), 'animals': g.animals, 'kind': 'switches'})
    for aid, results in g.results.items():
        with PdfPages(out / 'pdf' / f'{aid}_switches.pdf') as pdf:
            for fig in (animal_page(results, aid),) + ((psychometrics_animal_page(T, aid),) if len(T['psychometrics']) else ()):
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    if len(T['switches']):
        switches_summary(T, cohort, out)
    print(f'{g}  -> {out}')
    return out
