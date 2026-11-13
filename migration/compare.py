"""
Compare two migration captures, and pairs of report runs, number by number.

    python -m migration.compare BEFORE AFTER --out DIR [--runs BEFORE_RUN AFTER_RUN ...] [--loose]

BEFORE and AFTER are capture folders (``python -m migration.capture``). Each number in one is matched to
the other by keys that survive the 1.0 change of session ids (animal and session folder, session set, trial
class, statistic, readout row and column) and classed as equal, close (a fitted value within tolerance),
moved, or on one side only. Every number that is not equal is tagged with what changed underneath it, found
by comparing the sessions' trials:

    expected with 1.0 (marked ▲ in its changelog)
      gone, added    the session is in one capture only (strict loading; min_trials_per_file now also applies
                     to single-file sessions)
      date, type     the session's date or type changed (the date comes from the folder name only; a
                     session_types list overrides a recorded type)
      boundary       a trial with its stimulus on the boundary changed category or score
      nan_stimulus   a trial with a NaN stimulus changed category or score
      score          another trial's score changed (the task scores; a recorded column is only checked)
      prev_score     a trial's previous-trial score changed (NaN after a trial without a score)
      membership     the set holds other sessions, with one of the causes above somewhere in the animal
    not expected: a bug until explained
      rows           the session's trial count changed
      data           stimulus, choice, abort or laser differ on the same trials, or a category changed away
                     from the boundary
      history        the lag-1 view (previous stimulus, choice, has-previous) differs
      (none)         nothing underneath changed

``--runs`` pairs two report-run folders (``results/<report>/<cohort>/<run_id>``): every CSV and NPZ in both
is matched, with session ids and indices mapped to session folders through the captures' session tables.

Writes summary.md, sessions.csv and one table per comparison holding the rows that are not equal; exits 1 when
any number moved without an expected cause.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_CAUSES = frozenset({'gone', 'added', 'date', 'type', 'boundary', 'nan_stimulus', 'score',
                             'prev_score'})
UNEXPECTED_CAUSES = frozenset({'rows', 'data', 'history'})
EQUAL_RTOL = 1e-12
TOLERANCE = {'same': (1e-6, 1e-9), 'loose': (2e-2, 1e-3)}     # (rtol, atol) for fitted values
FITTED_STATS = frozenset({
    'mu', 'sigma', 'lapse_low', 'lapse_high', 'pse', 'psychometric_gof', 'history_interaction_r2',
    'w_stimulus', 'w_prev_choice_1', 'w_prev_choice_2', 'w_prev_choice_3', 'w_prev_outcome_1',
    'w_prev_outcome_2', 'w_prev_outcome_3', 'history_decay', 'pse_start', 'pse_end', 'pse_final', 'pse_tau',
    'pse_trials_to_90', 'pse_censored', 'pse_shape_daic', 'pse_step_switch', 'sd_slope', 'sd_curvature',
    'sd_range',
})
FITTED_READOUTS = frozenset({'psychometric', 'update_matrix', 'conditional_psychometric'})
TEXT_COLUMNS = frozenset({
    'animal', 'animal_id', 'key', 'set', 'trial_class', 'stat', 'readout', 'column', 'text', 'session_id',
    'date', 'start', 'session_type', 'type_source', 'stage', 'distribution', 'contingency', 'source',
    'digest_data', 'digest_view', 'digest_correct', 'outcome_label', 'choice_raw', 'scope', 'item', 'error',
    'fit_target', 'rep', 'feature', 'model', 'matrix', 'reason', 'true_model', 'digest_stimulus',
    'digest_choice',
})
SESSION_ID_COLUMNS = ('session_id', 'session_idx', 'next_session_idx')
KEY_INT_COLUMNS = frozenset({'order', 'trial', 'bin', 'cur_bin', 'prev_bin', 'row', 'draw', 'point',
                             'switch_idx', 'order_in_block', 'session', 'fold', 'rep_idx'})
SHOW = 25


# ── reading ───────────────────────────────────────────────────────────────────

def read_table(path) -> pd.DataFrame:
    """A capture table, with text columns as text ('' for missing) and floats read back exactly."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        head = pd.read_csv(path, nrows=0)
    except pd.errors.EmptyDataError:                  # an empty table written without a header
        return pd.DataFrame()
    dtype = {c: str for c in head.columns if c in TEXT_COLUMNS}
    table = pd.read_csv(path, dtype=dtype, keep_default_na=False, na_values=[''],
                        float_precision='round_trip', low_memory=False)
    for c in dtype:
        table[c] = table[c].fillna('')
    return table


class Capture:
    """One capture folder: its manifest, tables and per-animal trials (read on first use)."""

    def __init__(self, path):
        self.path = Path(path)
        manifest = self.path / 'manifest.json'
        if not manifest.exists():
            raise FileNotFoundError(f'{self.path} is not a capture (no manifest.json)')
        self.manifest = json.loads(manifest.read_text())
        self.library = str(self.manifest.get('library', ''))
        self.sessions = read_table(self.path / 'sessions.csv')
        self._trials: dict = {}

    def table(self, name: str) -> pd.DataFrame:
        return read_table(self.path / name)

    def trials(self, animal: str) -> dict:
        """{session key: that session's trials, in order}."""
        if animal not in self._trials:
            table = read_table(self.path / 'trials' / f'{animal}.csv.gz')
            self._trials[animal] = ({k: g.sort_values('row').reset_index(drop=True)
                                     for k, g in table.groupby('key', sort=False)} if len(table) else {})
        return self._trials[animal]

    def describe(self) -> str:
        git = self.manifest.get('git') or {}
        sha = (git.get('short') or 'no git') + (' (dirty)' if git.get('dirty') else '')
        return f"{self.path} — behav_utils {self.library}, {sha}, {self.manifest.get('created_utc', '?')}"


# ── numbers ───────────────────────────────────────────────────────────────────

def _same(a, b) -> np.ndarray:
    """Element-wise equality, NaN equal to NaN."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return (a == b) | (np.isnan(a) & np.isnan(b))


def scores(trials: pd.DataFrame) -> np.ndarray:
    """The score statistics read: the task's outcome codes where the capture has them (1.0), else
    choice == category (0.10, whose statistics scored that way)."""
    code = trials['outcome_code'].to_numpy(dtype=float)
    if np.isfinite(code).any():
        return np.where(code == 1, 1.0, np.where(code == 0, 0.0, np.nan))
    choice, category = trials['choice'].to_numpy(dtype=float), trials['category'].to_numpy(dtype=float)
    out = (choice == category).astype(float)
    out[np.isnan(choice) | np.isnan(category)] = np.nan
    return out


def classify(before, after, fitted, tolerance) -> np.ndarray:
    """'equal', 'close' (fitted, within tolerance) or 'moved', element-wise."""
    b, a = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    fitted = np.broadcast_to(np.asarray(fitted, dtype=bool), b.shape)
    with np.errstate(invalid='ignore'):
        diff = np.abs(a - b)
        equal = _same(b, a) | (diff <= EQUAL_RTOL * (1.0 + np.abs(b)))
        rtol, atol = tolerance
        close = fitted & ~equal & (diff <= rtol * np.abs(b) + atol)
    return np.where(equal, 'equal', np.where(close, 'close', 'moved'))


def verdict(cls: str, causes: str) -> str:
    """'' for equal and close numbers; else 'expected' or 'unexplained'."""
    if cls in ('equal', 'close'):
        return ''
    found = {c for c in causes.split(';') if c}
    if not found or found & UNEXPECTED_CAUSES or found == {'membership'}:
        return 'unexplained'
    return 'expected'


def _join(*parts) -> str:
    out = set()
    for p in parts:
        out |= set(p.split(';')) if isinstance(p, str) else set(p)
    return ';'.join(sorted(c for c in out if c))


# ── sessions and trials ───────────────────────────────────────────────────────

def trial_causes(tb: pd.DataFrame, ta: pd.DataFrame) -> tuple:
    """(causes, counts) from two versions of the same session's trials, row for row."""
    def col(t, name):
        return t[name].to_numpy(dtype=float)

    sb, sa = col(tb, 'stimulus'), col(ta, 'stimulus')
    boundary = (sb == 0) | (sa == 0)
    nan_stimulus = np.isnan(sb) | np.isnan(sa)
    edge = boundary | nan_stimulus
    data = ~_same(sb, sa)
    for name in ('stimulus_raw', 'choice', 'abort', 'opto_on'):
        data |= ~_same(col(tb, name), col(ta, name))
    category = ~_same(col(tb, 'category'), col(ta, 'category'))
    score = ~_same(scores(tb), scores(ta))
    score |= ~_same(col(tb, 'correct'), col(ta, 'correct'))
    view = np.zeros(len(tb), dtype=bool)
    for name in ('prev_stimulus', 'prev_choice', 'prev_has_prev'):
        view |= ~_same(col(tb, name), col(ta, name))
    prev_score = ~_same(col(tb, 'prev_correct'), col(ta, 'prev_correct'))
    causes = set()
    if data.any() or (category & ~edge).any():
        causes.add('data')
    if ((category | score) & boundary).any():
        causes.add('boundary')
    if ((category | score) & nan_stimulus & ~boundary).any():
        causes.add('nan_stimulus')
    if (score & ~edge).any():
        causes.add('score')
    if view.any():
        causes.add('history')
    if prev_score.any():
        causes.add('prev_score')
    counts = {'n_data': int(data.sum()), 'n_category': int(category.sum()), 'n_score': int(score.sum()),
              'n_history': int(view.sum()), 'n_prev_score': int(prev_score.sum()),
              'n_boundary': int(boundary.sum()), 'n_nan_stimulus': int(nan_stimulus.sum())}
    return causes, counts


def compare_sessions(before: Capture, after: Capture) -> pd.DataFrame:
    """One row per session in either capture: where it is, and the causes found in its trials."""
    columns = ['animal', 'key', 'date', 'session_type', 'n_trials']
    b = before.sessions[columns] if len(before.sessions) else pd.DataFrame(columns=columns)
    a = after.sessions[columns] if len(after.sessions) else pd.DataFrame(columns=columns)
    m = b.merge(a, on=['animal', 'key'], how='outer', suffixes=('_before', '_after'), indicator='side')
    sides = {'left_only': 'before only', 'right_only': 'after only', 'both': 'both'}
    m['where'] = m['side'].map(sides).astype(str)
    m = m.drop(columns='side')
    rows = []
    for animal, group in m.groupby('animal', sort=True):
        trials_b, trials_a = before.trials(animal), after.trials(animal)
        for r in group.itertuples(index=False):
            causes, counts = set(), {}
            where = r.where
            if where == 'before only':
                causes.add('gone')
            elif where == 'after only':
                causes.add('added')
            else:
                if r.date_before != r.date_after:
                    causes.add('date')
                if r.session_type_before != r.session_type_after:
                    causes.add('type')
                tb, ta = trials_b.get(r.key), trials_a.get(r.key)
                if tb is None or ta is None or len(tb) != len(ta):
                    causes.add('rows')
                else:
                    found, counts = trial_causes(tb, ta)
                    causes |= found
            rows.append({'animal': animal, 'key': r.key, 'where': where, 'causes': _join(causes),
                         'date_before': r.date_before, 'date_after': r.date_after,
                         'type_before': r.session_type_before, 'type_after': r.session_type_after,
                         'n_trials_before': r.n_trials_before, 'n_trials_after': r.n_trials_after, **counts})
    return pd.DataFrame(rows)


class Causes:
    """Causes by session, by animal and by session set, from the session comparison and set membership."""

    def __init__(self, sessions: pd.DataFrame, before: Capture, after: Capture):
        self.by_session = {(r.animal, r.key): r.causes for r in sessions.itertuples(index=False)}
        self.by_animal: dict = {}
        for (animal, _), causes in self.by_session.items():
            self.by_animal[animal] = _join(self.by_animal.get(animal, ''), causes)
        self.expected_in = {a: _join(set(c.split(';')) & EXPECTED_CAUSES) for a, c in self.by_animal.items()}
        self.members = ({}, {})
        for side, capture in enumerate((before, after)):
            sets = capture.table('sets.csv')
            if len(sets):
                for (animal, name), g in sets.groupby(['animal', 'set'], sort=False):
                    self.members[side][(animal, name)] = frozenset(g['key'])

    def of_sessions(self, animal: str, keys_before, keys_after) -> str:
        """The causes under a number computed from these sessions."""
        keys_before, keys_after = frozenset(keys_before), frozenset(keys_after)
        causes = _join(*(self.by_session.get((animal, k), '') for k in keys_before | keys_after))
        if keys_before != keys_after:
            causes = _join(causes, {'membership'}, self.expected_in.get(animal, ''))
        return causes

    def of_set(self, animal: str, name: str) -> str:
        return self.of_sessions(animal, self.members[0].get((animal, name), ()),
                                self.members[1].get((animal, name), ()))


# ── keyed tables ─────────────────────────────────────────────────────────────

def compare_keyed(before: pd.DataFrame, after: pd.DataFrame, keys: list, fitted, tolerance,
                  value: str = 'value', text: str | None = None) -> pd.DataFrame:
    """Outer-join two tables on ``keys`` and class each value. ``fitted`` is a bool or a function of the
    joined frame giving one bool per row."""
    if not len(before) and not len(after):
        return pd.DataFrame(columns=keys + ['before', 'after', 'cls'])
    take = keys + [value] + ([text] if text else [])
    b = before[take] if len(before) else after[take].iloc[0:0]
    a = after[take] if len(after) else b.iloc[0:0]
    m = b.merge(a, on=keys, how='outer', suffixes=('_before', '_after'), indicator=True)
    m = m.rename(columns={f'{value}_before': 'before', f'{value}_after': 'after'})
    flags = fitted(m) if callable(fitted) else np.full(len(m), bool(fitted))
    cls = classify(m['before'], m['after'], flags, tolerance)
    if text:
        tb, ta = m[f'{text}_before'].fillna(''), m[f'{text}_after'].fillna('')
        cls = np.where((cls == 'equal') & (tb != ta), 'moved', cls)
    side = m['_merge'].astype(str)
    cls = np.where(side == 'left_only', 'before only', np.where(side == 'right_only', 'after only', cls))
    m['cls'] = cls
    return m.drop(columns='_merge')


def attach(table: pd.DataFrame, columns: list, causes) -> pd.DataFrame:
    """Add ``causes`` (``causes(*values of columns)``, called once per distinct combination) and
    ``verdict`` (once per distinct class and causes)."""
    if not len(table):
        return table.assign(causes=pd.Series(dtype=str), verdict=pd.Series(dtype=str))
    distinct = table[columns].drop_duplicates()
    distinct = distinct.assign(causes=[causes(*row) for row in distinct.itertuples(index=False, name=None)])
    table = table.merge(distinct, on=columns, how='left')
    pairs = table[['cls', 'causes']].drop_duplicates()
    pairs = pairs.assign(verdict=[verdict(c, x) for c, x in zip(pairs['cls'], pairs['causes'])])
    return table.merge(pairs, on=['cls', 'causes'], how='left')


def mark_names(table: pd.DataFrame, names_before, names_after) -> pd.DataFrame:
    """A statistic only one library computes is a name change, not a number change."""
    if not len(table):
        return table
    table = table.copy()
    new = ~table['stat'].isin(set(names_before))
    dropped = ~table['stat'].isin(set(names_after))
    table.loc[new, 'verdict'] = 'new statistic'
    table.loc[dropped & ~new, 'verdict'] = 'unexplained'
    table.loc[dropped & ~new, 'causes'] = 'statistic dropped'
    return table


def compare_errors(before: Capture, after: Capture) -> pd.DataFrame:
    def signature(e):
        if not len(e):
            return pd.DataFrame(columns=['scope', 'animal', 'set', 'trial_class', 'key', 'item', 'kind',
                                         'error'])
        e = e.copy()
        e['kind'] = e['error'].str.split(':').str[0]
        return e
    keys = ['scope', 'animal', 'set', 'trial_class', 'key', 'item', 'kind']
    b, a = signature(before.table('errors.csv')), signature(after.table('errors.csv'))
    m = b.merge(a, on=keys, how='outer', suffixes=('_before', '_after'), indicator=True)
    m['cls'] = m['_merge'].map({'left_only': 'fixed', 'right_only': 'new', 'both': 'both'})
    m['verdict'] = np.where(m['cls'] == 'new', 'unexplained', '')
    return m.drop(columns='_merge')


# ── report runs ──────────────────────────────────────────────────────────────

def _session_maps(capture: Capture) -> dict:
    s = capture.sessions
    if not len(s):
        return {'id': {}, 'animal_id': {}, 'idx': {}}
    return {'id': dict(zip(s['session_id'], s['key'])),
            'animal_id': {(a, i): k for a, i, k in zip(s['animal'], s['session_id'], s['key'])},
            'idx': {(a, int(i)): k for a, i, k in zip(s['animal'], s['session_idx'], s['key'])}}


def _map_sessions(table: pd.DataFrame, maps: dict) -> pd.DataFrame:
    """Session ids and indices become session folders, so the two versions' rows meet."""
    animal = next((c for c in ('animal', 'animal_id') if c in table.columns), None)
    table = table.copy()
    for column in SESSION_ID_COLUMNS:
        if column not in table.columns:
            continue
        values = table[column]
        if column == 'session_id':
            mapped = [maps['animal_id'].get((a, str(v)), maps['id'].get(str(v), f'?{v}'))
                      for a, v in zip(table[animal] if animal else [None] * len(table), values)]
        elif animal:
            mapped = [maps['idx'].get((a, int(v)), f'?{v}') if pd.notna(v) else ''
                      for a, v in zip(table[animal], values)]
        else:
            continue
        table[column] = mapped
    return table


def _read_run_csv(path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, float_precision='round_trip', low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _csv_rows(path_b, path_a, maps_b, maps_a, tolerance) -> pd.DataFrame:
    b, a = _read_run_csv(path_b), _read_run_csv(path_a)
    if b.empty or a.empty:                          # an empty file on either side: compare the file only
        cls = 'equal' if b.empty and a.empty else ('after only' if b.empty else 'before only')
        return pd.DataFrame([{'animal': '', 'row_key': '', 'column': '(rows)', 'before': float(len(b)),
                              'after': float(len(a)), 'cls': cls}])
    b, a = _map_sessions(b, maps_b), _map_sessions(a, maps_a)
    keys = [c for c in b.columns if c in a.columns and
            (not pd.api.types.is_float_dtype(b[c]) or c in KEY_INT_COLUMNS) and
            (not pd.api.types.is_float_dtype(a[c]) or c in KEY_INT_COLUMNS)]
    values = [c for c in b.columns if c in a.columns and c not in keys]
    for t in (b, a):
        for c in keys:
            t[c] = t[c].astype(str)
        t['_n'] = t.groupby(keys, sort=False).cumcount() if keys else np.arange(len(t))
    keys = keys + ['_n']
    parts = []
    for c in values:
        if not (pd.api.types.is_numeric_dtype(b[c]) and pd.api.types.is_numeric_dtype(a[c])):
            continue
        part = compare_keyed(b[keys + [c]].rename(columns={c: 'value'}),
                             a[keys + [c]].rename(columns={c: 'value'}), keys, True, tolerance)
        part['column'] = c
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out['row_key'] = ['|'.join(f'{k}={v}' for k, v in zip(keys[:-1], row))
                      for row in out[keys[:-1]].itertuples(index=False)]
    if 'animal' not in out.columns:
        out['animal'] = out['animal_id'] if 'animal_id' in out.columns else ''
    return out[['animal', 'row_key', 'column', 'before', 'after', 'cls']]


def _npz_rows(path_b, path_a, tolerance) -> pd.DataFrame:
    rows = []
    with np.load(path_b, allow_pickle=False) as fb, np.load(path_a, allow_pickle=False) as fa:
        for name in sorted(set(fb.files) | set(fa.files)):
            if name not in fa.files or name not in fb.files:
                rows.append({'row_key': name, 'column': '(array)', 'before': np.nan, 'after': np.nan,
                             'cls': 'before only' if name in fb.files else 'after only'})
                continue
            xb, xa = fb[name], fa[name]
            if xb.shape != xa.shape or xb.dtype.kind not in 'biuf' or xa.dtype.kind not in 'biuf':
                same = xb.shape == xa.shape and np.array_equal(xb, xa)
                rows.append({'row_key': name, 'column': '(array)', 'before': np.nan, 'after': np.nan,
                             'cls': 'equal' if same else 'moved'})
                continue
            cls = classify(xb.ravel(), xa.ravel(), True, tolerance)
            worst = 'moved' if (cls == 'moved').any() else ('close' if (cls == 'close').any() else 'equal')
            with np.errstate(invalid='ignore'):
                d = np.nanmax(np.abs(xa.astype(float) - xb.astype(float))) if xb.size else 0.0
            rows.append({'row_key': name, 'column': f'max |diff| {d:.3g}', 'before': np.nan, 'after': np.nan,
                         'cls': worst})
    return pd.DataFrame(rows)


def compare_runs(run_b, run_a, before: Capture, after: Capture, causes: Causes, tolerance) -> pd.DataFrame:
    """Every CSV and NPZ of two report runs, matched file by file."""
    run_b, run_a = Path(run_b), Path(run_a)
    maps_b, maps_a = _session_maps(before), _session_maps(after)
    animals = set(causes.by_animal)
    every = _join(*causes.by_animal.values())
    files = sorted({p.relative_to(run_b) for p in run_b.rglob('*') if p.suffix in ('.csv', '.npz')} |
                   {p.relative_to(run_a) for p in run_a.rglob('*') if p.suffix in ('.csv', '.npz')})
    parts = []
    for rel in files:
        pb, pa = run_b / rel, run_a / rel
        if not pb.exists() or not pa.exists():
            part = pd.DataFrame([{'animal': '', 'row_key': '', 'column': '(file)', 'before': np.nan,
                                  'after': np.nan, 'cls': 'before only' if pb.exists() else 'after only'}])
        elif rel.suffix == '.csv':
            part = _csv_rows(pb, pa, maps_b, maps_a, tolerance)
        else:
            part = _npz_rows(pb, pa, tolerance)
            part['animal'] = next((p for p in rel.parts if p in animals), '')
        if not len(part):
            continue
        part['file'] = str(rel)
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=['file', 'animal', 'row_key', 'column', 'before', 'after', 'cls',
                                     'causes', 'verdict'])
    out = pd.concat(parts, ignore_index=True)
    out['animal'] = out['animal'].fillna('').astype(str)
    out['causes'] = [causes.by_animal.get(a, '') if a in animals else every for a in out['animal']]
    out['verdict'] = [verdict(c, x) for c, x in zip(out['cls'], out['causes'])]
    return out[['file', 'animal', 'row_key', 'column', 'before', 'after', 'cls', 'causes', 'verdict']]


# ── the whole comparison ─────────────────────────────────────────────────────

def compare(before_path, after_path, out_dir, *, runs=(), loose: bool = False) -> dict:
    """Compare two captures (and run pairs); write the tables and summary.md into ``out_dir``. Returns
    ``{table: DataFrame}`` of the rows that are not equal, plus ``sessions``."""
    before, after = Capture(before_path), Capture(after_path)
    tolerance = TOLERANCE['loose' if loose else 'same']
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = compare_sessions(before, after)
    causes = Causes(sessions, before, after)
    names_b, names_a = before.manifest.get('stat_names', []), after.manifest.get('stat_names', [])

    def fitted_stat(m):
        return m['stat'].isin(FITTED_STATS).to_numpy()

    def fitted_readout(m):
        return m['readout'].isin(FITTED_READOUTS).to_numpy()

    tables = {}
    pooled = compare_keyed(before.table('stats_pooled.csv.gz'), after.table('stats_pooled.csv.gz'),
                           ['animal', 'set', 'trial_class', 'stat'], fitted_stat, tolerance)
    tables['stats_pooled'] = mark_names(attach(pooled, ['animal', 'set'], causes.of_set), names_b, names_a)
    per = compare_keyed(before.table('stats_sessions.csv.gz'), after.table('stats_sessions.csv.gz'),
                        ['animal', 'key', 'trial_class', 'stat'], fitted_stat, tolerance)
    tables['stats_sessions'] = mark_names(
        attach(per, ['animal', 'key'], lambda a, k: causes.of_sessions(a, [k], [k])), names_b, names_a)
    readouts = compare_keyed(before.table('readouts.csv.gz'), after.table('readouts.csv.gz'),
                             ['animal', 'set', 'trial_class', 'readout', 'row', 'column'], fitted_readout,
                             tolerance, text='text')
    tables['readouts'] = attach(readouts, ['animal', 'set'], causes.of_set)

    # model identification: the sessions each animal is fitted on, then its numbers
    mid_b, mid_a = before.table('model_id/sessions.csv'), after.table('model_id/sessions.csv')

    def model_keys(t):
        if not len(t) or 'key' not in t.columns:
            return {}
        t = t[t['key'] != '']
        return {a: frozenset(g['key']) for a, g in t.groupby('animal_id')}
    keys_b, keys_a = model_keys(mid_b), model_keys(mid_a)

    def of_model_animal(animal):
        return causes.of_sessions(animal, keys_b.get(animal, ()), keys_a.get(animal, ()))

    def kept(t, animal):
        rows = t[t['animal_id'] == animal] if len(t) and 'kept' in t.columns else t.iloc[0:0]
        return bool(rows['kept'].iloc[0]) if len(rows) else None

    rows = []
    for animal in sorted(set(keys_b) | set(keys_a)):
        kb, ka = kept(mid_b, animal), kept(mid_a, animal)
        cls = 'equal' if keys_b.get(animal) == keys_a.get(animal) and kb == ka else 'moved'
        c = of_model_animal(animal)
        rows.append({'animal': animal, 'kept_before': kb, 'kept_after': ka,
                     'n_before': len(keys_b.get(animal, ())), 'n_after': len(keys_a.get(animal, ())),
                     'cls': cls, 'causes': c, 'verdict': verdict(cls, c)})
    tables['model_sessions'] = pd.DataFrame(rows, columns=['animal', 'kept_before', 'kept_after', 'n_before',
                                                           'n_after', 'cls', 'causes', 'verdict'])
    for name, keys in (('targets', ['animal', 'fit_target', 'cur_bin', 'prev_bin']),
                       ('vectors', ['animal', 'rep', 'feature']),
                       ('gs_points', ['animal', 'model', 'point', 'matrix', 'cur_bin', 'prev_bin'])):
        t = compare_keyed(before.table(f'model_id/{name}.csv'), after.table(f'model_id/{name}.csv'), keys,
                          True, tolerance)
        tables[f'model_{name}'] = attach(t, ['animal'], of_model_animal)
    sim = compare_keyed(before.table('model_id/simulator.csv'), after.table('model_id/simulator.csv'),
                        ['model', 'rep', 'draw', 'feature'], True, tolerance)
    tables['model_simulator'] = attach(sim, ['model'], lambda m: '')
    sim_b, sim_a = before.table('model_id/sim_sessions.csv'), after.table('model_id/sim_sessions.csv')
    parts = []
    for value, text in (('accuracy', 'digest_stimulus'), ('n_trials', 'digest_choice')):
        if len(sim_b) or len(sim_a):
            part = compare_keyed(sim_b, sim_a, ['model', 'distribution', 'session'], False, tolerance,
                                 value=value, text=text)
            parts.append(part.assign(column=f'{value} / {text}'))
    sims = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['model', 'cls'])
    tables['model_sim_sessions'] = attach(sims, ['model'], lambda m: '')
    tables['errors'] = compare_errors(before, after)
    run_tables = [compare_runs(rb, ra, before, after, causes, tolerance).assign(run=f'{rb} | {ra}')
                  for rb, ra in runs]
    if run_tables:
        tables['runs'] = pd.concat(run_tables, ignore_index=True)

    sessions.to_csv(out_dir / 'sessions.csv', index=False)
    for name, t in tables.items():
        if 'cls' in t.columns:
            t = t[~t['cls'].isin(['equal', 'both'])]
        t.to_csv(out_dir / f'{name}.csv.gz', index=False)
    (out_dir / 'summary.md').write_text(summary(before, after, sessions, tables, loose))
    return {'sessions': sessions, **tables}


def n_unexplained(tables: dict) -> int:
    return int(sum((t['verdict'] == 'unexplained').sum() for name, t in tables.items()
                   if name != 'sessions' and 'verdict' in t.columns))


def summary(before: Capture, after: Capture, sessions: pd.DataFrame, tables: dict, loose: bool) -> str:
    lines = ['# Migration comparison', '', f'- before: {before.describe()}', f'- after: {after.describe()}',
             f"- tolerance for fitted values: {'loose (another machine)' if loose else 'same machine'}", '']
    if len(sessions):
        where = sessions['where'].value_counts()
        lines += ['## Sessions', '',
                  f"{int(where.get('both', 0))} in both, {int(where.get('before only', 0))} before only, "
                  f"{int(where.get('after only', 0))} after only.", '']
        exploded = sessions['causes'].str.split(';').explode()
        counts = exploded[exploded != ''].value_counts()
        if len(counts):
            lines += ['| cause | sessions |', '|---|---|']
            lines += [f'| {c} | {n} |' for c, n in counts.items()] + ['']
    lines += ['## Numbers', '', '| table | equal | close | moved, expected | moved, unexplained | one side |',
              '|---|---|---|---|---|---|']
    for name, t in tables.items():
        if name == 'errors' or 'cls' not in t.columns:
            continue
        cls, v = t['cls'], t.get('verdict', pd.Series('', index=t.index))
        moved = cls == 'moved'
        counts = [(cls == 'equal').sum(), (cls == 'close').sum(), (moved & (v == 'expected')).sum(),
                  (moved & (v == 'unexplained')).sum(), cls.isin(['before only', 'after only']).sum()]
        lines.append(f'| {name} | ' + ' | '.join(str(int(c)) for c in counts) + ' |')
    errors = tables.get('errors', pd.DataFrame())
    if len(errors):
        n = errors['cls'].value_counts()
        lines += ['', f"Errors: {int(n.get('new', 0))} new, {int(n.get('fixed', 0))} fixed, "
                      f"{int(n.get('both', 0))} in both (errors.csv.gz)."]
    total = n_unexplained(tables)
    lines += ['', f'## Unexplained: {total}', '']
    for name, t in tables.items():
        if 'verdict' not in t.columns:
            continue
        bad = t[t['verdict'] == 'unexplained']
        if not len(bad):
            continue
        shown = bad.head(SHOW).to_string(index=False, max_colwidth=60)
        lines += [f'### {name} ({len(bad)})', '', '```', shown, '```', '']
    if total == 0:
        lines.append('Nothing moved without an expected cause.')
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('before', type=Path)
    p.add_argument('after', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--runs', nargs=2, action='append', default=[], metavar=('BEFORE_RUN', 'AFTER_RUN'),
                   help='a pair of report-run folders to compare; repeat for more pairs')
    p.add_argument('--loose', action='store_true', help='fitted values from another machine or library build')
    a = p.parse_args(argv)
    tables = compare(a.before, a.after, a.out, runs=[tuple(r) for r in a.runs], loose=a.loose)
    total = n_unexplained({k: v for k, v in tables.items() if k != 'sessions'})
    print((a.out / 'summary.md').read_text())
    print(f'{total} numbers moved without an expected cause' if total else 'nothing unexplained')
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main())
