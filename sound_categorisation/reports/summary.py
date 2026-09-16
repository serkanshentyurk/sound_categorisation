"""
Summary pages, drawn from the report tables only.

    python -m sound_categorisation.reports summary [--out results/reports] [--cohort opto1-cohort]

Writes ``<out>/<cohort>/summary.pdf`` with four pages (and a PNG of each):

    Uniform   within · between (laser sessions vs masking sessions) · fake-opto null · post-laser · session order
    Hard-A    within · between · session trajectory (A/B alternation, laser half | masking half) · per-session τ
    Hard-B    the same
    Overall   within μ × 3 distributions · between μ × 3 · ALM control · post-laser

Conventions on every page: one marker per animal, WT green / HET red, vertical
line = 95 % CI. Within-session contrasts use the trial bootstrap and a filled
marker means permutation p < 0.05 (laser randomised per trial). Between-session
contrasts use the session bootstrap and a filled marker means the session-level
CI excludes 0 (no permutation: session type was not randomised per trial).
Nothing here is computed; only read and drawn.
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

__all__ = ['load_tables', 'uniform_page', 'hard_page', 'overall_page', 'write_summary']

GENO_COL = {'wt': '#2ca02c', 'het': '#d62728'}
GENO_LABEL = {'wt': 'WT (light only)', 'het': 'HET (PPC silenced)'}
DIST_COL = {'Hard-A': '#1f77b4', 'Hard-B': '#ff7f0e', 'Uniform': '#7f7f7f'}
LASER_TYPES = ('opto', 'alm_control_uni', 'alm_control_bi')


# ── loading ─────────────────────────────────────────────────────────────────

def _read_all(root: Path, name: str) -> pd.DataFrame:
    files = [f for f in glob.glob(str(root / '**' / f'{name}.csv'), recursive=True) if '/group/' not in f]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if 'site' in df:
        df['site'] = df['site'].fillna('')
    return df


def load_tables(root: Path) -> dict:
    root = Path(root)
    contrasts = _read_all(root, 'contrasts')
    if not len(contrasts):
        raise FileNotFoundError(f'no contrasts.csv under {root}')
    gfiles = glob.glob(str(root / '**' / 'group_tests.csv'), recursive=True)
    traj = _read_all(root, 'trajectory')
    if len(traj):   # the same sessions are written under every phase folder they belong to: keep one copy
        traj = traj.drop_duplicates(subset=['animal', 'session_idx', 'toi']).query("toi == 'opto'")
    return {
        'contrasts': contrasts,
        'trajectory': traj,
        'group_tests': pd.concat([pd.read_csv(f) for f in gfiles], ignore_index=True) if gfiles else pd.DataFrame(),
    }


# ── primitives ──────────────────────────────────────────────────────────────

def _sel(df, *, design='ppc', site='', distribution, toi='opto', kind='within', unit=None, stat):
    unit = unit or ('trials' if kind in ('within', 'within_masking') else 'sessions')
    d = df[(df.design == design) & (df.site == site) & (df.distribution == distribution) & (df.toi == toi)
           & (df.kind == kind) & (df.unit == unit) & (df.stat == stat)]
    return d.sort_values(['genotype', 'animal'], ascending=[False, True])


def _dots(ax, d: pd.DataFrame, *, x0: float = 0.0, label_animals: bool = True, ylabel: str = ''):
    xs = {'wt': x0, 'het': x0 + 1}
    for g in ('wt', 'het'):
        sub = d[d.genotype == g]
        n = len(sub)
        jitter = np.linspace(-0.22, 0.22, n) if n > 1 else np.zeros(n)
        for j, r in zip(jitter, sub.itertuples()):
            if not np.isfinite(r.diff):
                continue
            if np.isfinite(r.perm_p):
                sig = r.perm_p < 0.05
            else:
                sig = np.isfinite(r.ci_lo) and (r.ci_lo > 0 or r.ci_hi < 0)
            x = xs[g] + j
            if np.isfinite(r.ci_lo):
                ax.plot([x, x], [r.ci_lo, r.ci_hi], color=GENO_COL[g], lw=1.2, alpha=0.8, zorder=2)
            ax.plot(x, r.diff, 'o', ms=6.5, mfc=GENO_COL[g] if sig else 'white', mec=GENO_COL[g], mew=1.5, zorder=3)
            if label_animals:
                ax.annotate(str(r.animal).replace('SS', ''), (x, r.diff), textcoords='offset points', xytext=(5, 0),
                            fontsize=6, color='0.35', va='center')
    ax.axhline(0, color='0.6', lw=0.8, ls='--', zorder=1)
    ax.set_xticks([x0, x0 + 1])
    ax.set_xticklabels(['WT', 'HET'], fontsize=9)
    ax.set_xlim(x0 - 0.6, x0 + 1.6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)


def _group_p(gt: pd.DataFrame, **key) -> float | None:
    if gt is None or not len(gt):
        return None
    d = gt.copy()
    if 'site' in d:
        d['site'] = d['site'].fillna('')
    for k, v in key.items():
        d = d[d[k] == v]
    return float(d['p'].iloc[0]) if len(d) and 'p' in d else None


def _ptxt(ax, p):
    if p is not None and np.isfinite(p):
        ax.text(0.98, 0.98, f'WT vs HET p = {p:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=8)


def _stat_row(fig, gs, row, df, gt, *, distribution, kind, stats, title, toi='opto'):
    """One row of panels: one stat each, for one contrast kind."""
    axes = []
    for k, (stat, lab) in enumerate(stats):
        ax = fig.add_subplot(gs[row, k])
        _dots(ax, _sel(df, distribution=distribution, kind=kind, stat=stat, toi=toi), ylabel=lab, label_animals=(k == 0))
        _ptxt(ax, _group_p(gt, design='ppc', distribution=distribution, toi=toi, kind=kind, stat=stat))
        if k == 0:
            ax.set_title(title, fontsize=10, loc='left')
        axes.append(ax)
    return axes


def _legend(fig, between=False):
    handles = [plt.Line2D([], [], marker='o', ls='', mfc=GENO_COL[g], mec=GENO_COL[g], label=GENO_LABEL[g])
               for g in ('wt', 'het')]
    sig = 'filled = session CI excludes 0' if between else 'filled = perm p < 0.05'
    handles += [plt.Line2D([], [], marker='o', ls='', mfc='white', mec='0.3', label='open = n.s.'),
                plt.Line2D([], [], marker='o', ls='', mfc='0.3', mec='0.3', label=sig)]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.005))


STATS4 = (('mu', 'Δ criterion μ'), ('side_bias', 'Δ side bias'), ('sigma', 'Δ slope σ'), ('accuracy', 'Δ accuracy'))


# ── trajectory panels ──────────────────────────────────────────────────────

def _boundary(ax, sess):
    types = list(sess['session_type'])
    for i in range(1, len(types)):
        if (types[i] in LASER_TYPES) != (types[i - 1] in LASER_TYPES):
            ax.axvline(sess['order'].iloc[i] - 0.5, color='0.5', ls=':', lw=1)


def _trajectory_panel(ax, tr: pd.DataFrame, y: str, geno: str, title: str, logy=False):
    """One line per animal of one genotype, distribution-coloured markers, boundary line, genotype mean."""
    d = tr[tr['genotype'] == geno] if len(tr) and 'genotype' in tr else pd.DataFrame()
    if not len(d) or y not in d:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', color='0.5')
        ax.set_title(title, fontsize=9)
        return
    for k, (_aid, sess) in enumerate(d.groupby('animal')):
        sess = sess.sort_values('order')
        ax.plot(sess['order'], sess[y], '-', color=GENO_COL[geno], alpha=0.3, lw=1)
        for dist, c in DIST_COL.items():
            m = sess['distribution'] == dist
            ax.plot(sess.loc[m, 'order'], sess.loc[m, y], 'o', color=c, ms=3.5, alpha=0.8)
        if k == 0:
            _boundary(ax, sess)
    mean = d.groupby('order')[y].mean()
    ax.plot(mean.index, mean.to_numpy(), '-', color=GENO_COL[geno], lw=2.4, zorder=4,
            label=f'{geno.upper()} mean (n={d.animal.nunique()})')
    if logy:
        ax.set_yscale('log')
    else:
        ax.axhline(0, color='0.7', lw=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('session order  (laser half | masking half)', fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)


# ── pages ───────────────────────────────────────────────────────────────────

def uniform_page(T: dict, cohort: str):
    df, gt, tr = T['contrasts'], T['group_tests'], T['trajectory']
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.6, wspace=0.35, height_ratios=[1, 1, 1, 0.55])
    _stat_row(fig, gs, 0, df, gt, distribution='Uniform', kind='within', stats=STATS4,
              title='1  ONLY OPTO SESSIONS: opto (laser-on) trials − non-opto (laser-off) trials.   One dot per mouse, bar = 95% CI, filled = permutation p < 0.05')
    _stat_row(fig, gs, 1, df, gt, distribution='Uniform', kind='between', stats=STATS4,
              title='2  OPTO SESSIONS − MASKING SESSIONS, no trial filtering (opto sessions include laser-on and laser-off trials).   Session bootstrap; filled = CI excludes 0')
    _stat_row(fig, gs, 2, df, gt, distribution='Uniform', kind='compensation', stats=STATS4,
              title='3  LASER-OFF TRIALS OF OPTO SESSIONS − MASKING SESSIONS (all trials): does the baseline move in a lasered session? (compensation)')
    # small: fake-opto null, post-laser
    ax = fig.add_subplot(gs[3, 0])
    _dots(ax, _sel(df, distribution='Uniform', kind='within_masking', stat='mu'), ylabel='Δ μ', label_animals=False)
    ax.set_title('4  ONLY MASKING SESSIONS:\nopto-flagged (laser at 0) − non-opto trials', fontsize=8.5, loc='left')
    ax = fig.add_subplot(gs[3, 1])
    _dots(ax, _sel(df, distribution='Uniform', toi='post_opto', kind='within', stat='mu'), ylabel='Δ μ', label_animals=False)
    ax.set_title('5  ONLY OPTO SESSIONS:\ntrial after a laser-on trial − laser-off trials', fontsize=8.5, loc='left')
    # session-order strip
    ax = fig.add_subplot(gs[3, 2:])
    u = tr[(tr['phase'] == 'Uniform')] if len(tr) and 'phase' in tr else pd.DataFrame()
    if len(u):
        for k, (_aid, sess) in enumerate(u.sort_values(['genotype', 'animal']).groupby('animal', sort=False)):
            sess = sess.sort_values('order')
            geno = sess['genotype'].iloc[0]
            laser = sess['session_type'].isin(LASER_TYPES)
            ax.scatter(sess['order'], [k] * len(sess), c=np.where(laser, GENO_COL[geno], '0.75'), s=28, marker='s')
        ax.set_yticks(range(k + 1))
        ax.set_yticklabels(u.sort_values(['genotype', 'animal'])['animal'].unique(), fontsize=7)
        ax.set_xlabel('session order (coloured = laser sessions, grey = masking)', fontsize=8)
    ax.set_title('6  Session order per mouse\n(coloured = opto block, grey = masking block)', fontsize=8.5, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    _legend(fig)
    fig.suptitle(f'{cohort} — Expert Uniform · laser at PPC', fontsize=13, y=0.995)
    return fig


def hard_page(T: dict, cohort: str, distribution: str):
    df, gt, tr = T['contrasts'], T['group_tests'], T['trajectory']
    fig = plt.figure(figsize=(16, 17))
    gs = fig.add_gridspec(5, 4, hspace=0.7, wspace=0.35, bottom=0.09, top=0.95)
    _stat_row(fig, gs, 0, df, gt, distribution=distribution, kind='within', stats=STATS4,
              title=f'1  {distribution}, ONLY OPTO SESSIONS: opto (laser-on) trials − non-opto (laser-off) trials, pooled over the six opto sessions.   Filled = permutation p < 0.05')
    _stat_row(fig, gs, 1, df, gt, distribution=distribution, kind='between', stats=STATS4,
              title=f'2  {distribution}, OPTO SESSIONS − MASKING SESSIONS, no trial filtering (six vs six; masking recorded later, so drift is inside this number)')
    _stat_row(fig, gs, 2, df, gt, distribution=distribution, kind='compensation', stats=STATS4,
              title=f'3  {distribution}, LASER-OFF TRIALS OF OPTO SESSIONS − MASKING SESSIONS (all trials): the compensation')
    hard = tr[tr['phase'].isin(['Hard-A', 'Hard-B'])] if len(tr) and 'phase' in tr else pd.DataFrame()
    # trajectory: the whole A/B alternation, both distributions, one panel per genotype
    for c, geno in enumerate(('wt', 'het')):
        ax = fig.add_subplot(gs[3, 2 * c:2 * c + 2])
        _trajectory_panel(ax, hard, 'pse', geno, f'4  {geno.upper()}: PSE fitted per session (slope and lapses free)')
        if c == 0:
            ax.set_ylabel('PSE (all trials)', fontsize=9)
    for c, geno in enumerate(('wt', 'het')):
        ax = fig.add_subplot(gs[4, c])
        _trajectory_panel(ax, hard, 'delta_from_prev', geno, f'5  {geno.upper()}: PSE change from the previous day')
        if c == 0:
            ax.set_ylabel('ΔPSE (switch amplitude)', fontsize=9)
    for c, geno in enumerate(('wt', 'het')):
        ax = fig.add_subplot(gs[4, 2 + c])
        _trajectory_panel(ax, hard, 'pse_fixed', geno, f'6  {geno.upper()}: PSE per session, slope and lapses fixed at Uniform values')
        if c == 0:
            ax.set_ylabel('PSE, shape pinned', fontsize=9)
    fig.text(0.5, 0.045, 'Rows 4–6: one thin line per mouse, thick = genotype mean; blue dot = Hard-A day, orange = Hard-B day; '
             'dotted line = last opto session | first masking session.\nRow 5 is the day-to-day switch amplitude (a tracking mouse '
             'alternates sign). Row 6 refits the PSE with slope and lapses held at the mouse\'s Uniform values, so only the '
             'criterion moves.', ha='center', va='top', fontsize=8, color='0.3')
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=d) for d, c in DIST_COL.items() if d != 'Uniform']
    fig.legend(handles=handles, loc='lower right', ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.98, 0.005))
    _legend(fig)
    fig.suptitle(f'{cohort} — {distribution} · laser at PPC.   Rows 1–3: pooled contrasts.   Rows 4–6: the full daily A/B alternation, session by session', fontsize=12, y=0.995)
    return fig


def overall_page(T: dict, cohort: str):
    df, gt = T['contrasts'], T['group_tests']
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.35)
    for k, dist in enumerate(('Uniform', 'Hard-A', 'Hard-B')):
        ax = fig.add_subplot(gs[0, k])
        _dots(ax, _sel(df, distribution=dist, kind='within', stat='mu'), ylabel='Δ μ, laser − no-laser trials' if k == 0 else '')
        ax.set_title(f'{dist}: opto sessions, laser-on − laser-off trials', fontsize=10, loc='left')
        _ptxt(ax, _group_p(gt, design='ppc', distribution=dist, toi='opto', kind='within', stat='mu'))
        ax = fig.add_subplot(gs[1, k])
        _dots(ax, _sel(df, distribution=dist, kind='between', stat='mu'), ylabel='Δ μ, laser − masking sessions' if k == 0 else '')
        ax.set_title(f'{dist}: opto sessions − masking sessions (all trials)', fontsize=10, loc='left')
        _ptxt(ax, _group_p(gt, design='ppc', distribution=dist, toi='opto', kind='between', stat='mu'))
    ax = fig.add_subplot(gs[0, 3])
    _dots(ax, _sel(df, design='alm', site='bi', distribution='Uniform', stat='accuracy'), ylabel='Δ accuracy', label_animals=False)
    ax.set_title('ALM bilateral sessions: laser-on − laser-off, accuracy', fontsize=10, loc='left')
    ax = fig.add_subplot(gs[1, 3])
    _dots(ax, _sel(df, distribution='Uniform', toi='post_opto', kind='within', stat='mu'), ylabel='Δ μ', label_animals=False)
    ax.set_title('Uniform opto sessions: post-laser trials − laser-off, μ', fontsize=10, loc='left')
    _legend(fig)
    fig.suptitle(f'{cohort} — overall: change in criterion μ per mouse (positive = shifted toward A).   Top row: only opto sessions, laser-on − laser-off trials.   Bottom row: opto sessions − masking sessions, no trial filtering', fontsize=11, y=0.995)
    return fig


def write_summary(out_root: Path, cohort: str) -> Path:
    from sound_categorisation.reports.readme import write_readme
    root = Path(out_root) / cohort
    write_readme(root)
    T = load_tables(root)
    pages = [('overall', overall_page(T, cohort)), ('uniform', uniform_page(T, cohort)),
             ('hard_a', hard_page(T, cohort, 'Hard-A')), ('hard_b', hard_page(T, cohort, 'Hard-B'))]
    path = root / 'summary.pdf'
    with PdfPages(path) as pdf:
        for name, fig in pages:
            pdf.savefig(fig, bbox_inches='tight')
            fig.savefig(root / f'summary_{name}.png', dpi=140, bbox_inches='tight')
            plt.close(fig)
    return path
