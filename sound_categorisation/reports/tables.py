"""
Tidy tables from report results, and their persistence.

    tables = to_tables(animal_result)             # {'contrasts': df, 'levels': df, 'equivalence': df,
                                                  #  'trajectory': df, 'trajectory_curves': df}
    write_result(out_dir, tables, readouts, meta)  # tables/*.csv, readouts.npz, meta.json
    tables, readouts, meta = read_result(out_dir)

One schema for every contrast row, whatever the site:

    cohort, animal, genotype, distribution, site, trial_class, kind, contrast, unit,
    stat, diff, ci_lo, ci_hi, boot_p, perm_p, n_a, n_b, n_sessions_a, n_sessions_b

``kind`` ∈ within, within_masking, between, compensation, dod, vs_ppc. Group tables add
``group`` and the rank-test columns. CSV + JSON, not pickle: a table written
today is readable by any code tomorrow.

``equivalence`` reads the ``within`` contrast (trials unit) against the margins the run was given
(``Settings.margins``): one row per statistic that has a margin, with the verdict of
``contrasts.equivalence_verdict``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sound_categorisation.behaviour.contrasts import equivalence_verdict
from sound_categorisation.data.cohort import site_label
from sound_categorisation.reports.compute import AnimalResult, GroupResult

__all__ = ['to_tables', 'group_tables', 'readout_arrays', 'write_result', 'read_result', 'CONTRAST_COLUMNS',
           'EQUIVALENCE_COLUMNS', 'equivalence_table']

LEVEL_COLUMNS = ['cohort', 'animal', 'genotype', 'distribution', 'site', 'trial_class', 'kind', 'phase',
                 'n_trials', 'n_sessions', 'stat', 'value']

CONTRAST_COLUMNS = ['cohort', 'animal', 'genotype', 'distribution', 'site', 'trial_class', 'kind', 'contrast',
                    'unit', 'stat', 'diff', 'ci_lo', 'ci_hi', 'boot_p', 'perm_p',
                    'n_a', 'n_b', 'n_sessions_a', 'n_sessions_b']


EQUIVALENCE_COLUMNS = ['cohort', 'animal', 'genotype', 'distribution', 'site', 'trial_class', 'stat',
                       'diff', 'ci_lo', 'ci_hi', 'margin', 'verdict']


def equivalence_table(contrasts: pd.DataFrame, margins: Dict[str, float]) -> pd.DataFrame:
    """The ``within`` contrast (laser-on − laser-off trials of the laser sessions, trials unit) read against
    the equivalence margins: one row per statistic that has a margin, with its interval, the margin and the
    verdict (``contrasts.equivalence_verdict``). Empty when there is no such contrast or no margin applies."""
    rows = contrasts[(contrasts['kind'] == 'within') & (contrasts['unit'] == 'trials')
                     & contrasts['stat'].isin(list(margins))] if len(contrasts) else contrasts
    if not len(rows):
        return pd.DataFrame(columns=EQUIVALENCE_COLUMNS)
    out = rows.copy()
    out['margin'] = out['stat'].map(margins).astype(float)
    out['verdict'] = [equivalence_verdict(lo, hi, m)
                      for lo, hi, m in zip(out['ci_lo'], out['ci_hi'], out['margin'])]
    return out[EQUIVALENCE_COLUMNS].reset_index(drop=True)


def _qc(sessions: pd.DataFrame, bias: pd.DataFrame | None) -> pd.DataFrame:
    """The cohort QC on a trajectory frame: ``flagged`` per session and ``biased_animal`` from the one bias
    check (``adaptation.bias_sessions``, carried as ``Trajectory.bias``), the same list the switch report
    uses. Sessions it does not cover (Uniform) are not flagged."""
    if not len(sessions):
        return sessions
    if bias is None or not len(bias):
        return sessions.assign(flagged=False, biased_animal=False)
    flags = bias.set_index('session_idx')['flagged'].astype(bool)
    biased = bool(bias['biased_animal'].astype(bool).iloc[0])
    return sessions.assign(flagged=sessions['session_idx'].map(flags).fillna(False).astype(bool),
                           biased_animal=biased)


def _labels(r: AnimalResult) -> dict:
    return {'cohort': r.cohort, 'animal': r.animal, 'genotype': r.genotype, 'distribution': r.distribution,
            'site': site_label(r.design, r.site), 'trial_class': r.trial_class}


def to_tables(r: AnimalResult) -> Dict[str, pd.DataFrame]:
    """Flatten an AnimalResult: every contrast for every available unit, plus adaptation."""
    lab = _labels(r)
    frames = []
    for kind in ('within', 'within_masking', 'between', 'compensation', 'vs_ppc'):
        if kind not in r.contrasts:
            continue
        d = r.contrasts[kind]
        for key, c in d.contrasts.items():
            units = c.units or (None,)
            for u in units:
                t = c.table(u) if u else c.table()
                t = t.assign(kind=kind, contrast=key, n_a=c.n_a, n_b=c.n_b,
                             n_sessions_a=c.n_sessions_a, n_sessions_b=c.n_sessions_b, **lab)
                frames.append(t)
    if 'dod' in r.contrasts:
        ix = r.contrasts['dod']
        for u in ix.units:
            t = ix.table(u).rename(columns={'interaction': 'diff', 'p': 'boot_p'})
            t = t.assign(perm_p=np.nan, kind='dod', contrast=f'dod_{r.contrasts.key}',
                         n_a=np.nan, n_b=np.nan, n_sessions_a=np.nan, n_sessions_b=np.nan, **lab)
            frames.append(t[['stat', 'diff', 'ci_lo', 'ci_hi', 'boot_p', 'perm_p', 'unit', 'kind', 'contrast',
                             'n_a', 'n_b', 'n_sessions_a', 'n_sessions_b', *lab]])
    contrasts = (pd.concat(frames, ignore_index=True)[CONTRAST_COLUMNS] if frames
                 else pd.DataFrame(columns=CONTRAST_COLUMNS))

    # levels: the observed value of every stat in every condition the contrasts were built from —
    # (kind, phase) is unambiguous: within/opto = laser-on trials of opto sessions, within/non_opto = laser-off
    # trials of opto sessions, within_masking/* = the same on sham sessions, between/* = all trials of each
    # session set, compensation/laser_off = laser-off trials of opto sessions.
    lrows = []
    for kind in ('within', 'within_masking', 'between', 'compensation', 'vs_ppc'):
        if kind not in r.contrasts:
            continue
        for label, ph in r.contrasts[kind].phases.items():
            for stat, value in ph.stats.items():
                lrows.append({'kind': kind, 'phase': label, 'n_trials': ph.n_trials, 'n_sessions': ph.n_sessions,
                              'stat': stat, 'value': float(value), **lab})
    levels = pd.DataFrame(lrows, columns=LEVEL_COLUMNS) if lrows else pd.DataFrame(columns=LEVEL_COLUMNS)

    tr = r.trajectory
    # trajectory rows keep their own per-session `distribution`; the page's distribution is `phase`
    tlab = {k: v for k, v in lab.items() if k != 'distribution'} | {'phase': r.distribution}
    return {
        'contrasts': contrasts,
        'levels': levels,
        'equivalence': equivalence_table(contrasts, r.margins),
        'trajectory': (_qc(tr.sessions.assign(expert_pse=tr.baseline_pse, sigma=tr.sigma, **tlab),
                           getattr(tr, 'bias', None))
                       if tr is not None else pd.DataFrame()),
        'trajectory_curves': tr.curves.assign(**tlab) if tr is not None else pd.DataFrame(),
    }


def group_tables(g: GroupResult) -> Dict[str, pd.DataFrame]:
    lab = {'cohort': g.cohort, 'distribution': g.distribution, 'site': site_label(g.design, g.site), 'trial_class': g.trial_class}
    tlab = {k: v for k, v in lab.items() if k != 'distribution'} | {'phase': g.distribution}
    sess = [_qc(tr.sessions.assign(animal=aid, genotype=g.by_animal.get(aid, 'unknown'), expert_pse=tr.baseline_pse,
                                   sigma=tr.sigma, **tlab), getattr(tr, 'bias', None))
            for aid, tr in g.trajectories.items()]
    curves = [tr.curves.assign(animal=aid, genotype=g.by_animal.get(aid, 'unknown'), **tlab)
              for aid, tr in g.trajectories.items()]
    return {
        'group_rows': g.rows.assign(**lab),
        'group_tests': g.tests.assign(**lab),
        'trajectory': pd.concat(sess, ignore_index=True) if sess else pd.DataFrame(),
        'trajectory_curves': pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
    }


def readout_arrays(r: AnimalResult) -> Dict[str, np.ndarray]:
    """Curves and matrices as flat arrays keyed ``<phase>__<trial_type>__<field>`` for an .npz."""
    out = {}
    for (phase, tt), ro in r.readouts.items():
        k = f'{phase}__{tt}'
        if ro.curve is not None:
            c = ro.curve
            out[f'{k}__curve_x'] = c.x
            out[f'{k}__curve_y'] = c.y
            out[f'{k}__curve_params'] = c.params.to_numpy()
            if c.band is not None:
                out[f'{k}__curve_band'] = c.band
            out[f'{k}__bin_centres'] = c.bin_centres
            out[f'{k}__bin_means'] = c.bin_means
            out[f'{k}__bin_counts'] = c.bin_counts
        if ro.update_matrix is not None:
            out[f'{k}__um'] = ro.update_matrix.matrix
            out[f'{k}__um_n_pairs'] = ro.update_matrix.n_pairs
    return out


def _versions() -> dict:
    import behav_utils

    import sound_categorisation
    v = {'behav_utils': behav_utils.__version__, 'sound_categorisation': sound_categorisation.__version__}
    try:
        import subprocess
        v['git_sha'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
                                              text=True).strip()
    except Exception:
        v['git_sha'] = None
    return v


def normalise_site(df: pd.DataFrame) -> pd.DataFrame:
    """Readers' compatibility shim for tables written before the single `site` column: a missing or empty
    `site` means PPC, and an old (`design`, `site`) pair becomes `ppc` / `alm_uni` / `alm_bi`. Writers never
    produce the old form. Returns the frame with `site` in {ppc, alm_uni, alm_bi} and no `design` column."""
    if not len(df):
        return df
    df = df.copy()
    if 'design' in df:
        sub = df['site'].fillna('').astype(str) if 'site' in df else ''
        df['site'] = np.where(df['design'].astype(str) == 'alm', 'alm_' + sub, 'ppc')
        df = df.drop(columns=['design'])
    elif 'site' in df:
        df['site'] = df['site'].fillna('').astype(str).replace({'': 'ppc', 'nan': 'ppc'})
    else:
        df['site'] = 'ppc'
    return df


def write_result(out_dir, tables: Dict[str, pd.DataFrame], readouts: Dict[str, np.ndarray] | None = None,
                 meta: dict | None = None) -> Path:
    """Write tables as CSV, readouts as npz, and a metadata sidecar. Overwrites."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in tables.items():
        if len(df.columns) == 0:          # an empty table has no schema to write
            continue
        df.to_csv(out_dir / f'{name}.csv', index=False)
        written.append(name)
    if readouts:
        np.savez_compressed(out_dir / 'readouts.npz', **readouts)
    m = {'written_utc': datetime.now(UTC).isoformat(), 'versions': _versions(),
         'tables': sorted(written), 'has_readouts': bool(readouts), **(meta or {})}
    (out_dir / 'meta.json').write_text(json.dumps(m, indent=2, default=str))
    return out_dir


def read_result(out_dir) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray], dict]:
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / 'meta.json').read_text())
    tables = {p.stem: pd.read_csv(p) for p in sorted(out_dir.glob('*.csv')) if p.stat().st_size > 0}
    readouts = {}
    if (out_dir / 'readouts.npz').exists():
        with np.load(out_dir / 'readouts.npz') as z:
            readouts = {k: z[k] for k in z.files}
    return tables, readouts, meta
