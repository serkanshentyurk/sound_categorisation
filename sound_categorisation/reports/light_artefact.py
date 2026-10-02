"""
The light-artefact report — ``sc-reports light-artefact``.

What the light does on its own, per animal and across genotypes, at every site that carries light
(``behaviour/light_artefact.py``). Writes, under ``light_artefact/<cohort>/<run_id>/<distribution>/``:

    <animal>/light_contrasts.csv   light-on − light-off per session set (kind), stat and unit — the schema of
                                   contrasts.csv, plus the ppc_sham − alm interaction as kind='site_dod'
    <animal>/levels.csv            the observed on / off values behind each contrast
    <animal>/choice_by_bin.csv     P(B) per stimulus bin, on vs off, with a CI on the difference
    <animal>/meta.json
    group/group_rows.csv           one per-animal point difference per (set, stat)
    group/group_tests.csv          WT vs HET rank test per (set, stat), with min_p
    pdf/<animal>_light.pdf, pdf/group_light.pdf

Reading: for WT animals every set is light-only; a criterion shift (``criterion``, ``mu``) with intact
``dprime`` on ``ppc_sham`` but not on ``alm_*`` is a site-specific light effect, and ``site_dod`` is its
test. On HET animals ``ppc_opto`` adds inactivation to the light; the opto-contrasts report's ``dod`` is
where that is separated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import matplotlib
import numpy as np
import pandas as pd
from behav_utils.analysis import collect_rows, compare_groups
from behav_utils.plotting import plot_psychometric_curve

from sound_categorisation.behaviour.light_artefact import (
    KEY,
    LIGHT_SETS,
    LIGHT_STATS,
    LightArtefact,
    compute_light_artefact,
)
from sound_categorisation.data.cohort import gather_genotypes
from sound_categorisation.data.paths import git_state

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

COLUMNS = ['cohort', 'animal', 'genotype', 'distribution', 'kind', 'contrast', 'unit', 'stat', 'diff', 'ci_lo', 'ci_hi',
           'boot_p', 'perm_p', 'n_on', 'n_off', 'n_sessions']
SET_COLOUR = {'ppc_opto': '#d62728', 'ppc_sham': '#1f77b4', 'alm_uni': '#2ca02c', 'alm_bi': '#9467bd'}
GENOTYPE_COLOUR = {'wt': '#1f77b4', 'het': '#ff7f0e', 'unknown': '0.5'}


# ── tables ───────────────────────────────────────────────────────────────────

def to_tables(r: LightArtefact, cohort: str, genotype: str) -> Dict[str, pd.DataFrame]:
    lab = {'cohort': cohort, 'animal': r.animal, 'genotype': genotype, 'distribution': r.distribution}
    frames, levels = [], []
    for name, d in r.contrasts.items():
        c = d.contrasts[KEY]
        for u in (c.units or (None,)):
            t = c.table(u) if u else c.table()
            frames.append(t.assign(kind=name, contrast=KEY, n_on=c.n_a, n_off=c.n_b, n_sessions=r.n_sessions.get(name), **lab))
        for label, ph in d.phases.items():
            for stat, value in ph.stats.items():
                levels.append({'kind': name, 'phase': label, 'n_trials': ph.n_trials, 'n_sessions': ph.n_sessions,
                               'stat': stat, 'value': float(value), **lab})
    if r.site_interaction is not None:
        ix = r.site_interaction
        for u in ix.units:
            t = ix.table(u).rename(columns={'interaction': 'diff', 'p': 'boot_p'})
            frames.append(t.assign(perm_p=np.nan, kind='site_dod', contrast=f'{ix.label_a}_minus_{ix.label_b}',
                                   n_on=np.nan, n_off=np.nan, n_sessions=np.nan, **lab))
    contrasts = pd.concat(frames, ignore_index=True)[COLUMNS] if frames else pd.DataFrame(columns=COLUMNS)
    return {'light_contrasts': contrasts,
            'levels': pd.DataFrame(levels),
            'choice_by_bin': r.bins.assign(**lab)}


# ── figures ──────────────────────────────────────────────────────────────────

def animal_pdf(r: LightArtefact, genotype: str, out: Path) -> Path:
    sets = [s for s in LIGHT_SETS if s in r.contrasts]
    with PdfPages(out) as pdf:
        n = max(1, len(sets))
        fig, axes = plt.subplots(n, 3, figsize=(13, 3.3 * n), squeeze=False)
        for i, name in enumerate(sets):
            d = r.contrasts[name]
            plot_psychometric_curve(r.curves[name]['off'], ax=axes[i, 0], color='0.3')
            plot_psychometric_curve(r.curves[name]['on'], ax=axes[i, 0], color=SET_COLOUR[name])
            axes[i, 0].set_title(f'{name}: light on (colour) vs off (grey)', fontsize=9)
            b = r.bins[r.bins['set'] == name]
            axes[i, 1].axhline(0, color='0.7', lw=0.8)
            axes[i, 1].errorbar(b['centre'], b['diff'], yerr=[b['diff'] - b['ci_lo'], b['ci_hi'] - b['diff']],
                                fmt='o-', color=SET_COLOUR[name], ms=4)
            axes[i, 1].set(xlabel='stimulus bin centre', ylabel='P(B) on − off', title='choice shift per bin', ylim=(-0.5, 0.5))
            t = d.contrasts[KEY].table('trials') if 'trials' in (d.contrasts[KEY].units or ()) else d.contrasts[KEY].table()
            t = t.set_index('stat').reindex(list(LIGHT_STATS))
            y = np.arange(len(t))
            axes[i, 2].axvline(0, color='0.7', lw=0.8)
            axes[i, 2].errorbar(t['diff'], y, xerr=[t['diff'] - t['ci_lo'], t['ci_hi'] - t['diff']], fmt='o', color=SET_COLOUR[name], ms=4)
            axes[i, 2].set(yticks=y, yticklabels=t.index, title='on − off, 95 % CI (trials)')
            axes[i, 2].invert_yaxis()
        if not sets:
            axes[0, 0].text(0.5, 0.5, 'no light-carrying sessions', ha='center', transform=axes[0, 0].transAxes)
        fig.suptitle(f'{r.animal} · {genotype} · {r.distribution} — light-on vs light-off', fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    return out


def group_pdf(rows: pd.DataFrame, tests: pd.DataFrame, distribution: str, out: Path, stats=('criterion', 'dprime', 'mu', 'side_bias')) -> Path:
    sets = [s for s in LIGHT_SETS if s in set(rows['kind'])]
    with PdfPages(out) as pdf:
        fig, axes = plt.subplots(len(stats), max(1, len(sets)), figsize=(3 * max(1, len(sets)), 2.8 * len(stats)), squeeze=False)
        rng = np.random.default_rng(0)
        for i, st in enumerate(stats):
            for j, name in enumerate(sets):
                ax = axes[i, j]
                d = rows[(rows['kind'] == name) & (rows['stat'] == st)]
                for k, g in enumerate(('wt', 'het')):
                    v = d[d['group'] == g]['value'].to_numpy()
                    ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, color=GENOTYPE_COLOUR[g], s=26, zorder=3)
                    if len(v):
                        ax.hlines(np.median(v), k - 0.25, k + 0.25, color=GENOTYPE_COLOUR[g], lw=2)
                ax.axhline(0, color='0.7', lw=0.6)
                t = tests[(tests['kind'] == name) & (tests['stat'] == st)] if len(tests) else tests
                p = f"WT vs HET p={t['p'].iloc[0]:.3f} (min {t['min_p'].iloc[0]:.3f})" if len(t) else 'no test'
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['WT', 'HET'])
                ax.set_xlim(-0.6, 1.6)
                ax.set_title(f'{name} · Δ{st}\n{p}', fontsize=8)
        fig.suptitle(f'light-on − light-off per animal, {distribution} (WT: light only)', fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    return out


# ── the run ──────────────────────────────────────────────────────────────────

def run_light_artefact(experiment, animals: Sequence[str], run: Path, cohort: str, distribution: str = 'Uniform', *,
                       fast: bool = False, meta: dict | None = None) -> Path:
    """Per-animal tables + PDFs, then the WT-vs-HET fold, under ``<run>/<distribution>/``."""
    out = Path(run) / distribution
    (out / 'pdf').mkdir(parents=True, exist_ok=True)
    (out / 'group').mkdir(exist_ok=True)
    by_animal, _ = gather_genotypes(experiment)
    n_boot, n_perm = (50, 50) if fast else (1000, 1000)
    g = git_state()
    rows = []
    for aid in animals:
        if aid not in experiment.animals:
            continue
        r = compute_light_artefact(experiment.animals[aid], distribution, n_boot=n_boot, n_perm=n_perm,
                                   curve_boot=20 if fast else 200)
        if not r.contrasts:
            print(f'  {aid}: no light-carrying sessions at {distribution}, skipped')
            continue
        geno = by_animal.get(aid, 'unknown')
        tables = to_tables(r, cohort, geno)
        d = out / aid
        d.mkdir(exist_ok=True)
        for name, t in tables.items():
            t.to_csv(d / f'{name}.csv', index=False)
        (d / 'meta.json').write_text(json.dumps({'run_id': run.name, 'argv': list(sys.argv), 'git_sha': g['sha'],
                                                 'git_dirty': g['dirty'], 'cohort': cohort, 'animal': aid,
                                                 'distribution': distribution, 'fast': fast, 'n_boot': n_boot,
                                                 'n_perm': n_perm, 'sets': list(r.contrasts), **(meta or {})},
                                                indent=2, default=str))
        animal_pdf(r, geno, out / 'pdf' / f'{aid}_light.pdf')
        for name, dlt in r.contrasts.items():
            rows += [x | {'kind': name} for x in collect_rows(
                [{'stat': k, 'value': float(v)} for k, v in dlt.contrasts[KEY].diff.items()], animal=aid, group=geno)]
        print(f'  {aid}: {list(r.contrasts)}')
    rows_df = pd.DataFrame(rows, columns=['animal', 'group', 'kind', 'stat', 'value'])
    tests = []
    for kind, sub in rows_df.groupby('kind', sort=False):
        if sub['group'].nunique() < 2:
            continue
        for stat, res in compare_groups(sub, group_col='group', groups=('wt', 'het')).items():
            tests.append({'kind': kind, 'stat': stat, **{k: v for k, v in res.items() if np.isscalar(v)}})
    tests_df = pd.DataFrame(tests)
    rows_df.assign(cohort=cohort, distribution=distribution).to_csv(out / 'group' / 'group_rows.csv', index=False)
    tests_df.assign(cohort=cohort, distribution=distribution).to_csv(out / 'group' / 'group_tests.csv', index=False)
    group_pdf(rows_df, tests_df, distribution, out / 'pdf' / 'group_light.pdf')
    (Path(run) / 'README.md').write_text(README.format(cohort=cohort))
    return out


README = '''# light_artefact — {cohort}

What the light does on its own, at every site that carries light. Light-on and light-off trials are
randomised per trial within a session, so `on − off` is a valid per-trial contrast on every set:

| set | sessions | for a WT animal | for a HET animal |
|---|---|---|---|
| `ppc_opto` | PPC laser sessions | light only | light + inactivation |
| `ppc_sham` | PPC blue-light, no laser (session type `masking`) | light only | light only |
| `alm_uni` / `alm_bi` | ALM laser sessions | light only | light + ALM inactivation |

Per animal, under `<distribution>/<animal>/`: `light_contrasts.csv` (kind = set; `site_dod` = ppc_sham − alm,
the site-specificity test), `levels.csv` (observed on/off values), `choice_by_bin.csv` (P(B) per stimulus
bin, on vs off, CI on the difference). `group/` holds the per-animal point differences and the WT-vs-HET
rank tests (`min_p` = the smallest p the group sizes allow). Stats: `side_bias accuracy mu sigma dprime
criterion` — a light effect on the decision rule moves `criterion` / `mu` and leaves `dprime` / `sigma`.

Produced by `sc-reports light-artefact`; read by notebook `20` §3.
'''
