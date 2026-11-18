"""
Switch report: how fast the criterion follows a change of stimulus distribution, in both Hard designs.

    sc-reports switch-adaptation --cohort behaviour1-cohort     # blocks of days (the pre-opto mice)
    sc-reports switch-adaptation --cohort opto1-cohort          # the daily alternation of the opto Hard phase

Per animal, ``adaptation.compute_switches``: every switch into a Hard distribution, which is the start of a
block of at least 1000 trials or, in the daily alternation, each day (``adaptation.hard_design`` decides per
animal), measured one way: the criterion with the shape fixed at the animal's expert-Uniform fit, in running
50-trial windows, 0 = the criterion before the switch, 1 = the normative criterion, not clipped. In the daily
design a laser day is measured on laser-off trials. The manuscript's Fig. 5C recipe stays beside it for
comparison. ``bias.csv`` is ``adaptation.bias_sessions``, the one bias check every report uses.

Writes into ``<run>/switches/``: switches.csv, convergence.csv, sessions.csv, overnight.csv, pre_post.csv
(blocks), psychometrics.csv, psychometric_curves.csv, bias.csv, animals.csv (one number per animal and kind
of switch: ``convergence_100_300``), per-animal PDFs and summary_switches.pdf. Notebook 13 draws chapter 3's
Fig 3.6 from one run per cohort.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy import stats

from sound_categorisation.behaviour.adaptation import (
    BIAS_COLUMNS,
    DAY_MIN_TRIALS,
    HARD,
    SWITCH_MIN_BLOCK,
    BiasRule,
    SwitchResult,
    bias_sessions,
    compute_phase_curves,
    compute_switches,
    design_switches,
    hard_design,
    pool_convergence,
)
from sound_categorisation.reports.tables import write_result

__all__ = ['SwitchesResult', 'compute_switches_cohort', 'switch_tables', 'switches_summary', 'run_switches',
           'pooled_curve', 't_interval', 'LEVEL', 'DAY_KINDS', 'BLOCK_KINDS']

LEVEL = 0.95
BLOCK_KINDS = ('first', 'novel', 'return')          # blocked design: switch_kind = the transition
DAY_KINDS = ('masking day', 'laser day')            # daily design: the two kinds the summary compares
TRANS_COL = {'first': '#7f7f7f', 'novel': '#1f77b4', 'return': '#ff7f0e'}
KIND_COL = {**TRANS_COL, 'masking day': '#2ca02c', 'laser day': '#d62728', 'first day': '#7f7f7f',
            'mixed day': '#9467bd'}
METHOD_TITLE = {'manuscript': 'manuscript method: 4-parameter fit per 50-trial bin, clipped to [0, 1]',
                'pinned': 'shape fixed (only the criterion fitted per 50-trial bin), unclipped',
                'pinned_running': 'shape fixed, running 50-trial windows (step 10), unclipped'}
SWITCH_LABELS = ['animal', 'switch_idx', 'from_distribution', 'to_distribution', 'transition', 'design',
                 'switch_kind', 'trials', 'session_type', 'prev_session_type']


@dataclass(frozen=True)
class SwitchesResult:
    cohort: str
    results: Dict[str, List[SwitchResult]] = field(default_factory=dict)    # animal → switches
    phase_params: pd.DataFrame = field(default_factory=pd.DataFrame)        # per animal × phase psychometric fit
    phase_curves: pd.DataFrame = field(default_factory=pd.DataFrame)        # per animal × phase curve (x, y)
    bias: pd.DataFrame = field(default_factory=pd.DataFrame)                # adaptation.bias_sessions, every animal
    designs: Dict[str, str] = field(default_factory=dict)                   # animal → 'blocked' | 'daily'

    @property
    def animals(self) -> List[str]:
        return list(self.results)

    def __repr__(self) -> str:
        n = sum(len(v) for v in self.results.values())
        designs = {d: list(self.designs.values()).count(d) for d in sorted(set(self.designs.values()))}
        return f'SwitchesResult({self.cohort!r}, n_animals={len(self.results)}, designs={designs}, n_switches={n})'


def _cat(frames) -> pd.DataFrame:
    """Concatenate a list of frames; an empty list gives an empty frame."""
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_switches_cohort(experiment, animals: Sequence[str], *, cohort: str = '', bias_rule=None,
                            **kw) -> SwitchesResult:
    """Every animal's switches, phase psychometrics and bias check. ``kw`` goes to ``compute_switches``
    (``min_block_trials``, ``min_day_trials``, ``max_trials``, …). An animal that fails is left out with a
    warning; one without switches into a Hard distribution is skipped."""
    out, params, curves, bias, designs = {}, [], [], [], {}
    for aid in animals:
        if aid not in experiment.animals:
            continue
        animal = experiment.animals[aid]
        try:
            design = hard_design(animal)
            res = compute_switches(animal, design=design, **kw)
            if not res:
                continue
            sessions, switches = design_switches(animal, design,
                                                 min_block_trials=kw.get('min_block_trials', SWITCH_MIN_BLOCK),
                                                 min_day_trials=kw.get('min_day_trials', DAY_MIN_TRIALS))
            switches = [s for s in switches if s['to_distribution'] in HARD]
            p, c = compute_phase_curves(animal, sessions, switches, per_block=(design == 'blocked'),
                                        trials='all' if design == 'blocked' else 'non_opto')
            b = bias_sessions(animal, bias_rule)
        except Exception as exc:                          # one animal must not kill the batch
            warnings.warn(f'{aid}: switches failed: {exc!r}', stacklevel=2)
            continue
        out[aid] = res
        params.append(p)
        curves.append(c)
        bias.append(b)
        designs[aid] = design
    return SwitchesResult(cohort, out, _cat(params), _cat(curves), _cat(bias), designs)


def _with_flags(sessions: pd.DataFrame, bias: pd.DataFrame) -> pd.DataFrame:
    """The bias check's ``flagged`` (by animal and session) and ``biased_animal`` on another sessions table."""
    if not len(sessions):
        return sessions
    if not len(bias):
        return sessions.assign(flagged=False, biased_animal=False)
    flags = bias.set_index(['animal', 'session_idx'])['flagged'].astype(bool)
    biased = bias.groupby('animal')['biased_animal'].first().astype(bool)
    key = list(zip(sessions['animal'], sessions['session_idx']))
    return sessions.assign(flagged=[bool(flags.get(k, False)) for k in key],
                           biased_animal=sessions['animal'].map(biased).fillna(False).astype(bool))


def _wide(sw: pd.DataFrame) -> pd.DataFrame:
    """switches.csv one row per switch (its stats as columns)."""
    if not len(sw):
        return pd.DataFrame(columns=SWITCH_LABELS)
    return sw.groupby(SWITCH_LABELS + ['stat'], dropna=False)['value'].first().unstack('stat').reset_index()


def _animal_summary(sw: pd.DataFrame, bias: pd.DataFrame) -> pd.DataFrame:
    """One row per animal × design × kind of switch: the switches combined by ``pool_convergence``."""
    cols = ['animal', 'design', 'switch_kind', 'n_switches', 'convergence_100_300', 'convergence_final', 'biased_animal']
    w = _wide(sw)
    if not len(w):
        return pd.DataFrame(columns=cols)
    for c in ('convergence_100_300', 'convergence_final', 'span'):
        if c not in w:
            w[c] = np.nan
    rows = [{'animal': a, 'design': d, 'switch_kind': k, 'n_switches': len(g),
             'convergence_100_300': pool_convergence(g['convergence_100_300'], g['span']),
             'convergence_final': pool_convergence(g['convergence_final'], g['span'])}
            for (a, d, k), g in w.groupby(['animal', 'design', 'switch_kind'])]
    out = pd.DataFrame(rows)
    biased = bias.groupby('animal')['biased_animal'].first().astype(bool) if len(bias) else pd.Series(dtype=bool)
    out['biased_animal'] = out['animal'].map(biased).fillna(False).astype(bool)
    return out[cols]


def switch_tables(g: SwitchesResult) -> Dict[str, pd.DataFrame]:
    """The switch report's tables (see the module notes)."""
    lab = {'cohort': g.cohort}
    rows, conv, sess, over = [], [], [], []
    for results in g.results.values():
        for r in results:
            key = {**r.labels, **lab}
            rows.append(r.to_rows().assign(**lab))
            conv.append(r.convergence.assign(**key))
            sess.append(r.sessions.assign(**key))
            over.append(r.overnight.assign(**key))
    bias = g.bias.assign(**lab) if len(g.bias) else pd.DataFrame(columns=BIAS_COLUMNS + ['cohort'])
    sw = _cat(rows)
    # pre/post per blocked switch: the block-pooled 4-parameter fits (robust) and the criterion over the
    # last 250 trials on each side (shape fixed)
    pre_post = pd.DataFrame()
    blocked = sw[sw['design'] == 'blocked'] if len(sw) else sw
    if len(blocked) and len(g.phase_params):
        keys = ['animal', 'switch_idx', 'transition', 'from_distribution', 'to_distribution']
        pv = blocked.groupby(keys + ['stat'])['value'].first().unstack('stat').reset_index()
        pp = pv[keys].copy()
        pp['pse_pre_last250'] = pv.get('pre_pse_pinned')
        pp['pse_post_last250'] = pv.get('pse_post')
        pp['normative_pse'] = pv.get('normative_pse')
        P = g.phase_params
        pooled = {(r.animal, int(r.order)): r.pse for r in P.itertuples()}   # order 0 = Uniform ref, k = switch k-1's block
        pp['pse_pre'] = [pooled.get((a, int(k))) for a, k in zip(pp['animal'], pp['switch_idx'])]
        pp['pse_post'] = [pooled.get((a, int(k) + 1)) for a, k in zip(pp['animal'], pp['switch_idx'])]
        pp['delta'] = pp['pse_post'] - pp['pse_pre']
        pre_post = pp.assign(**lab)
    return {'switches': sw, 'pre_post': pre_post, 'convergence': _cat(conv),
            'sessions': _with_flags(_cat(sess), bias), 'overnight': _cat(over),
            'psychometrics': g.phase_params.assign(**lab) if len(g.phase_params) else pd.DataFrame(),
            'psychometric_curves': g.phase_curves.assign(**lab) if len(g.phase_curves) else pd.DataFrame(),
            'bias': bias, 'animals': _animal_summary(sw, bias).assign(**lab)}


# ── group statistics (read tables only) ─────────────────────────────────────

def t_interval(values, level: float = LEVEL) -> tuple:
    """(mean, low, high, n) over the finite ``values``: the t interval, the unit being whatever the values are
    (here: mice). Fewer than two values: no interval."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return np.nan, np.nan, np.nan, 0
    m = float(v.mean())
    if v.size < 2:
        return m, np.nan, np.nan, 1
    h = float(stats.t.ppf(0.5 + level / 2, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
    return m, m - h, m + h, int(v.size)


def pooled_curve(conv: pd.DataFrame) -> pd.DataFrame:
    """A convergence table (one method) → per mouse and window, its switches combined like
    ``adaptation.pool_convergence`` (weighted by span); columns animal, trial, convergence."""
    d = conv.dropna(subset=['convergence', 'span'])
    d = d[d['span'] > 0]
    if not len(d):
        return pd.DataFrame(columns=['animal', 'trial', 'convergence'])
    d = d.assign(num=d['convergence'] * d['span'])
    g = d.groupby(['animal', 'trial']).agg(num=('num', 'sum'), den=('span', 'sum')).reset_index()
    return g.assign(convergence=g['num'] / g['den'])[['animal', 'trial', 'convergence']]


def _group_curve(conv: pd.DataFrame) -> pd.DataFrame:
    """Per window: mean over mice of ``pooled_curve`` with its t interval; columns trial, mean, lo, hi, n."""
    per = pooled_curve(conv)
    rows = [(t, *t_interval(g['convergence'])) for t, g in per.groupby('trial')]
    return pd.DataFrame(rows, columns=['trial', 'mean', 'lo', 'hi', 'n'])


# ── figures (read tables only) ──────────────────────────────────────────────

def _biased(T: Dict[str, pd.DataFrame]) -> set:
    b = T.get('bias', pd.DataFrame())
    if len(b) and 'biased_animal' in b:
        return set(b.loc[b['biased_animal'].astype(bool), 'animal'])
    se = T.get('sessions', pd.DataFrame())
    return set(se.loc[se['biased_animal'].astype(bool), 'animal']) if len(se) and 'biased_animal' in se else set()


def _subset(df: pd.DataFrame, T, clean: bool) -> pd.DataFrame:
    if not clean or not len(df):
        return df
    return df[~df['animal'].isin(_biased(T))]


def _subset_label(T, clean: bool) -> str:
    b = sorted(_biased(T))
    return (f'clean animals only (excluded: {", ".join(b)})' if b else 'all animals (none flagged)') if clean else 'all animals'


def _designs(T) -> List[str]:
    sw = T.get('switches', pd.DataFrame())
    present = set(sw['design']) if len(sw) and 'design' in sw else set()
    return [d for d in ('blocked', 'daily') if d in present]


def _curve_with_anchor(sub: pd.DataFrame, value: str):
    """Manuscript aggregation: per animal, mean over its switches per bin; a q=0 anchor at x=1 with value 0;
    mean and 95 % t interval over animals."""
    per = sub.groupby(['animal', 'trial'])[value].mean().unstack('trial')
    per.insert(0, 1.0, 0.0)                                   # the pre-switch point, convergence = 0 by definition
    x = per.columns.to_numpy(float)
    rows = [t_interval(per[c]) for c in per.columns]
    m, lo, hi, n = (np.array(v, float) for v in zip(*rows))
    return x, m, lo, hi, per.shape[0]


def convergence_page(T: Dict[str, pd.DataFrame], cohort: str, clean: bool = True):
    """Fig. 5C, the manuscript's recipe (blocked switches): (left) the manuscript's aggregation — asymmetric
    switches pooled per animal, anchored at 0 — with the first switch as a separate line; (right) the same
    split by transition type."""
    conv = _subset(T['convergence'], T, clean)
    d = conv[(conv['method'] == 'manuscript') & (conv['design'] == 'blocked') & np.isfinite(conv['convergence_clipped'])]
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
                 'normative = constant-σ observer at each mouse\'s expert-Uniform σ; mean ± 95 % t interval over mice', fontsize=10)
    fig.text(0.5, 0.01, 'The point at trial 1 is the pre-switch value and is 0 by definition (as in the manuscript). Each bin is clipped '
             'to [0, 1] before averaging; with noisy 50-trial fits this alone puts a bin near 0.5, so the level of the first fitted bin '
             'is not an adaptation measure. The measure the report uses (shape fixed, unclipped) is on the speed page.',
             ha='center', va='bottom', fontsize=8, color='0.3', wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def _dots(ax, i, values, colour):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size:
        j = np.linspace(-0.15, 0.15, v.size) if v.size > 1 else np.zeros(1)
        ax.plot(i + j, v, 'o', color=colour, ms=6, alpha=0.75)
    m, lo, hi, n = t_interval(v)
    if np.isfinite(m):
        ax.plot([i - 0.25, i + 0.25], [m, m], color='k', lw=1.6)
        if np.isfinite(lo):
            ax.plot([i, i], [lo, hi], color='k', lw=1.2)
    return n


def speed_page(T: Dict[str, pd.DataFrame], cohort: str, design: str, clean: bool = True):
    """How fast the criterion moves, the report's measure: the criterion with the shape fixed in running 50-trial
    windows, 0 = before the switch, 1 = normative, unclipped. Each mouse's switches of one kind combined
    (weighted by span), then mean ± 95 % t interval over mice; per mouse the mean over trials 100-300 (grey
    band); for blocks, where the Hard-A → Hard-B block ends (drift works against it)."""
    kinds = BLOCK_KINDS if design == 'blocked' else DAY_KINDS
    conv = _subset(T['convergence'], T, clean)
    conv = conv[(conv['method'] == 'pinned_running') & (conv['design'] == design)]
    an = _subset(T['animals'], T, clean)
    an = an[an['design'] == design]
    ncols = 3 if design == 'blocked' else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols, 4.8), squeeze=False)
    axes = axes[0]
    ax = axes[0]
    for kind in kinds:
        gc = _group_curve(conv[conv['switch_kind'] == kind])
        if not len(gc):
            continue
        c = KIND_COL.get(kind, '0.3')
        ax.plot(gc['trial'], gc['mean'], color=c, lw=2, label=f'{kind} (n = {int(gc["n"].max())} mice)')
        ax.fill_between(gc['trial'], gc['lo'], gc['hi'], color=c, alpha=0.15, lw=0)
    ax.axvspan(100, 300, color='0.93', zorder=0)
    ax.axhline(1, ls='--', color='k', lw=0.9)
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlim(20, 3000 if design == 'blocked' else 600)
    ax.xaxis.set_major_locator(FixedLocator([25, 50, 100, 300, 1000, 3000] if design == 'blocked' else [25, 50, 100, 200, 400]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-1, 2)
    ax.set_xlabel('trials after the switch (window centre, log scale)')
    ax.set_ylabel('convergence  (0 = before the switch, 1 = normative)')
    ax.set_title('criterion after the switch, mean ± 95 % t interval over mice', fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax = axes[1]
    for i, kind in enumerate(kinds):
        n = _dots(ax, i, an.loc[an['switch_kind'] == kind, 'convergence_100_300'], KIND_COL.get(kind, '0.3'))
        ax.text(i, -0.95, f'n = {n}', ha='center', fontsize=8, color='0.4')
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds)
    ax.axhline(1, ls='--', color='k', lw=0.9)
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_ylim(-1, 2)
    ax.set_ylabel('convergence over trials 100-300')
    ax.set_title('one number per mouse; bar = mean, line = 95 % t interval', fontsize=9)
    if design == 'blocked':
        ax = axes[2]
        w = _subset(_wide(T['switches']), T, clean)
        ab = w[(w['design'] == 'blocked') & (w['from_distribution'] == 'Hard-A') & (w['to_distribution'] == 'Hard-B')]
        per = [pool_convergence(g['convergence_final'], g['span']) for _, g in ab.groupby('animal')] if len(ab) else []
        n = _dots(ax, 0, per, TRANS_COL['return'])
        ax.set_xticks([0])
        ax.set_xticklabels([f'Hard-A → Hard-B\n(n = {n} mice)'])
        ax.axhline(1, ls='--', color='k', lw=0.9)
        ax.axhline(0, color='0.6', lw=0.8)
        ax.set_xlim(-1, 1)
        ax.set_ylabel('convergence over the last 250 trials of the block')
        ax.set_title('where the block ends; the slow upward drift works against\n'
                     'this switch, so > 0 is evidence, ≈ 0 is not', fontsize=9)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    what = 'blocks of days' if design == 'blocked' else 'the daily alternation (laser days on laser-off trials)'
    fig.suptitle(f'{cohort} — how fast the criterion follows the distribution: {what}; {_subset_label(T, clean)}.\n'
                 'Criterion with the shape fixed at the expert-Uniform fit, running 50-trial windows, unclipped; '
                 'a mouse\'s switches of one kind combined weighted by |normative − before|', fontsize=10)
    fig.tight_layout()
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
    """Blocked switches by transition: speed, shape, plateau, overnight change and within-block drift."""
    sw = _subset(_wide(T['switches']), T, clean)
    sw = sw[sw['design'] == 'blocked']
    for col in ('pse_tau', 'trials_to_criterion', 'plateau', 'pse_shape_daic'):
        if col not in sw:
            sw[col] = np.nan
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    _swarm(axes[0, 0], sw, 'pse_tau', 'τ (trials) — exponential fit on the block, shape fixed', logy=True)
    axes[0, 0].set_title('speed of the criterion shift (log)', fontsize=10)
    _swarm(axes[0, 1], sw, 'trials_to_criterion', 'trials to reach 80 % convergence (running windows)', logy=True)
    axes[0, 1].set_title('trials to criterion', fontsize=10)
    _swarm(axes[0, 2], sw, 'plateau', 'mean convergence over the last third of the block')
    axes[0, 2].axhline(1, ls='--', color='k', lw=1)
    axes[0, 2].axhline(0, color='0.6', lw=0.8)
    axes[0, 2].set_title('plateau relative to normative', fontsize=10)
    _swarm(axes[1, 0], sw, 'pse_shape_daic', 'AIC(exponential) − AIC(step);  < 0 gradual, > 0 step')
    axes[1, 0].axhline(0, color='0.6', lw=0.8)
    axes[1, 0].set_title('trajectory shape', fontsize=10)
    # overnight: the criterion from the end of one session to the start of the next, signed by the direction
    # of the switch, so that noise in the end-of-session window cannot set its sign
    ov = _subset(T['overnight'], T, clean)
    ov = ov[ov['design'] == 'blocked'] if len(ov) else ov
    ax = axes[1, 1]
    if len(ov):
        d = ov.dropna(subset=['toward_pre'])
        per = d.groupby(['animal', 'transition'])['toward_pre'].mean().reset_index()
        _swarm(ax, per, 'toward_pre', 'overnight change, signed by the switch direction\n'
                                      '(> 0 = back towards the previous distribution)')
        ax.axhline(0, color='0.6', lw=0.8)
    ax.set_title('overnight: last 100 trials → first 100 of the next session', fontsize=9)
    # within-block criterion session by session, order in block, one thin line per animal × switch
    ax = axes[1, 2]
    se = _subset(T['sessions'], T, clean)
    se = se[se['design'] == 'blocked'] if len(se) else se
    if len(se):
        for tr, sub in se.groupby('transition'):
            for _, s in sub.groupby(['animal', 'switch_idx']):
                s = s.sort_values('order_in_block')
                ax.plot(s['order_in_block'], s['pse_pinned'] - s['pse_pinned'].iloc[0], color=TRANS_COL.get(tr, '0.3'),
                        alpha=0.25, lw=0.8)
            m = sub.groupby('order_in_block')['pse_pinned'].mean()
            m0 = sub.groupby(['animal', 'switch_idx'])['pse_pinned'].first().mean()
            ax.plot(m.index, m.to_numpy() - m0, color=TRANS_COL.get(tr, '0.3'), lw=2.2, label=tr)
        ax.axhline(0, color='0.6', lw=0.8)
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel('session within the block')
    ax.set_ylabel('criterion (shape fixed) − first session of the block')
    ax.set_title('within-block drift, session by session', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'{cohort} — blocked switches by transition, {_subset_label(T, clean)} (first = Uniform→Hard-B, novel = Hard-B→Hard-A, '
                 'return = Hard-A→Hard-B; type and destination are confounded by design). One dot = one mouse × switch, bar = median', fontsize=10)
    fig.tight_layout()
    return fig


ANIMAL_ROWS = 6        # switches per page of an animal's PDF


def animal_pages(results: List[SwitchResult], aid: str) -> list:
    """Per switch: convergence by method (left) and the criterion session by session in the block (right);
    ``ANIMAL_ROWS`` switches per page."""
    figs = []
    for first in range(0, len(results), ANIMAL_ROWS):
        chunk = results[first:first + ANIMAL_ROWS]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(12, 3.2 * len(chunk)), squeeze=False)
        for row, r in enumerate(chunk):
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
            dyn = r.dynamics
            ax.set_title(f'{aid} switch {r.switch_idx}: {r.from_distribution}→{r.to_distribution} [{r.switch_kind}, '
                         f'{r.trials} trials]  before {r.pre["pse_pinned"]:+.2f}, normative {r.normative_pse:+.2f}, '
                         f'100-300: {dyn.get("convergence_100_300", np.nan):.2f}, end: {dyn.get("convergence_final", np.nan):.2f}',
                         fontsize=8)
            ax.legend(frameon=False, fontsize=7)
            ax = axes[row, 1]
            s = r.sessions
            if len(s):
                ax.plot(s['order_in_block'], s['pse_pinned'], 'o-', color='#d62728', ms=4, label='criterion, shape fixed')
                ax.plot(s['order_in_block'], s['pse'], 's--', color='0.5', ms=3, label='PSE, 4-parameter')
            ax.axhline(r.pre['pse_pinned'], color='0.3', ls=':', label='before the switch')
            ax.axhline(r.normative_pse, color='crimson', ls='--', label='normative')
            ax.set_xlabel('session within block')
            ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        figs.append(fig)
    return figs


# ── QC and psychometrics pages ───────────────────────────────────────────────

DIST_COL = {'Uniform': '#7f7f7f', 'Hard-A': '#c23f63', 'Hard-B': '#007dc7'}


def qc_page(T: Dict[str, pd.DataFrame], cohort: str):
    """The bias check (``bias.csv``, every Hard session in order): accuracy, criterion, lapses; flagged sessions
    marked; biased animals named."""
    se = T['bias'].sort_values(['animal', 'order'])
    animals = list(dict.fromkeys(se['animal']))
    fig, axes = plt.subplots(max(len(animals), 1), 1, figsize=(14, 2.0 * max(len(animals), 1)), sharex=False, squeeze=False)
    for ax, aid in zip(axes[:, 0], animals):
        d = se[se['animal'] == aid].reset_index(drop=True)
        x = np.arange(len(d))
        biased = bool(d['biased_animal'].iloc[0])
        for dist, c in DIST_COL.items():
            m = (d['to_distribution'] == dist).to_numpy()
            ax.scatter(x[m], d.loc[m, 'accuracy'], s=14, color=c, label=dist if m.any() else None, zorder=3)
        ax.plot(x, d['pse_fixed'].clip(-1.0, 1.0), '-', color='0.4', lw=1, label='criterion (shape fixed)')
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
        changes = np.flatnonzero(d['to_distribution'].to_numpy()[1:] != d['to_distribution'].to_numpy()[:-1]) + 1
        if len(changes) <= 8:                      # blocks; a daily alternation would be all lines
            for b in changes:
                ax.axvline(b - 0.5, color='0.5', ls='--', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=6, loc='lower left', bbox_to_anchor=(0, 1.05))
    axes[-1, 0].set_xlabel('Hard session, in order (blocks separated by dashed lines), laser-off trials')
    fig.suptitle(f'{cohort} — the bias check (bias.csv). Dots = accuracy (colour = distribution); line = signed criterion (dashed = ±0.4); '
                 f'dotted = larger lapse; ▼ = flagged (|criterion| > 0.4, lapse > 0.3 or accuracy < 0.6). BIASED = majority flagged '
                 f'with the same sign on A and B days.', fontsize=9)
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
    biased = aid in _biased(T)
    fig.suptitle(f'{aid} — psychometrics by phase' + ('   [BIASED — excluded from the combined pages]' if biased else ''),
                 fontsize=10, color='crimson' if biased else 'k')
    fig.tight_layout()
    return fig


def psychometrics_combined_page(T: Dict[str, pd.DataFrame], cohort: str, exclude_biased: bool = True):
    P, C = T['psychometrics'], T['psychometric_curves']
    biased = _biased(T)
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
        for _, d in cc.groupby('animal'):
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
    """Blocked switches, per transition type: PSE before the switch vs at the end of the block, one line per mouse,
    normative dashed."""
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
    designs = _designs(T)
    pages = [qc_page(T, cohort)]
    if has_psy:
        pages.append(psychometrics_combined_page(T, cohort, exclude_biased=True))
    pages += [speed_page(T, cohort, d, clean=True) for d in designs]
    if 'blocked' in designs:
        if len(T.get('pre_post', [])):
            pages.append(pre_post_page(T, cohort, clean=True))
        pages += [convergence_page(T, cohort, clean=True), dynamics_page(T, cohort, clean=True)]
    if has_psy:
        pages.append(psychometrics_combined_page(T, cohort, exclude_biased=False))
    pages += [speed_page(T, cohort, d, clean=False) for d in designs]
    if 'blocked' in designs:
        pages.append(convergence_page(T, cohort, clean=False))
    with PdfPages(path) as pdf:
        for fig in pages:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    return path


def run_switches(experiment, animals: Sequence[str], run: Path, cohort: str, meta: dict | None = None,
                 bias_rule=None, **kw) -> Path:
    """Write the switch report into ``<run>/switches/`` (``run`` is a run directory from ``start_run``).
    ``bias_rule``: the cohort QC (adaptation.BiasRule; default BiasRule()), recorded in meta.json; ``kw`` goes
    to ``compute_switches``."""
    matplotlib.use('Agg')
    rule = bias_rule if bias_rule is not None else BiasRule()
    g = compute_switches_cohort(experiment, animals, cohort=cohort, bias_rule=rule, **kw)
    out = Path(run) / 'switches'
    (out / 'pdf').mkdir(parents=True, exist_ok=True)
    T = switch_tables(g)
    write_result(out, T, None, {**(meta or {}), 'animals': g.animals, 'designs': g.designs, 'kind': 'switches',
                                'bias_rule': asdict(rule)})
    for aid, results in g.results.items():
        with PdfPages(out / 'pdf' / f'{aid}_switches.pdf') as pdf:
            figs = animal_pages(results, aid) + ([psychometrics_animal_page(T, aid)] if len(T['psychometrics']) else [])
            for fig in figs:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    if len(T['switches']):
        switches_summary(T, cohort, out)
    print(f'{g}  -> {out}')
    return out
