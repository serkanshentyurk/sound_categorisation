"""
Baseline capture for moving the project from behav_utils 0.10 to 1.0.

Records what the project computes from the experiment its analyses load (the snapshot): every session and
every trial, point statistics and readouts per animal x session set x trial class, statistics per session,
and the model-identification inputs. Run it once on behav_utils 0.10.0 before switching and again after each
step of the move; ``python -m migration.compare`` then matches two captures number by number.

The same file runs on both library versions. The library names it imports are ones both export at the top
level (the golden test pinned its values from 0.10.0 through them), and the fields that differ between the
two versions are read defensively. The project functions it calls keep their signatures through the move:
``load_experiment_any``, ``ensure_presets``, ``select_animals``, ``to_stat_vector``, ``feature_names``,
``build_simulator``, ``simulate_sessions``, ``sessions_to_arrays``, ``simulate_model_matrices`` and
``selection._empirical_target``.

    python -m migration.capture --out DIR [--snapshot PATH] [--config PATH] [--cohort all-mice]
                                [--n-jobs N] [--n-sim 25] [--skip-model-id] [--overwrite]

Everything is CSV (gzip where large) and JSON; nothing is pickled. README.md lists the tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import warnings
from datetime import date, datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from behav_utils import (
    TrialArrays,
    compute_binned_curve,
    compute_conditional_psychometric,
    compute_psychometric_curve,
    compute_sd_profile,
    compute_stats,
    compute_update_matrix,
    filter_trials,
    select_sessions,
)

STAT_SEED = 0                    # every compute_stats call gets a fresh generator with this seed
FINGERPRINT_SEED = 20_000_000    # simulator fingerprints: outside the training and validation seed ranges
GS_SEED = 30_000_000
TRIAL_CLASSES = ('all', 'non_opto', 'opto', 'post_opto')
SESSION_CLASSES = ('all', 'non_opto', 'opto')
DESIGN_DISTRIBUTIONS = ('Uniform', 'Hard-A', 'Hard-B')
DESIGN_TYPES = ('opto', 'masking', 'alm_control_uni', 'alm_control_bi', 'regular')
READOUTS = (
    ('psychometric', compute_psychometric_curve, {'n_bootstrap': 0}),
    ('update_matrix', compute_update_matrix, {}),
    ('conditional_psychometric', compute_conditional_psychometric, {}),
    ('sd_profile', compute_sd_profile, {}),
    ('binned_choice', compute_binned_curve, {}),
    ('binned_accuracy', compute_binned_curve, {'kind': 'accuracy'}),
)
# Grid-search fingerprints: (sigma_percep, A_repulsion, param 1, param 2), inside DEFAULT_GRID's ranges
GS_POINTS = {
    'BE': ((0.15, 0.2, 0.5, 0.2), (0.25, 0.0, 0.3, 0.1), (0.10, 0.4, 0.7, 0.3)),
    'SC': ((0.15, 0.2, 0.5, 0.5), (0.25, 0.0, 0.3, 0.8), (0.10, 0.4, 0.7, 0.3)),
}
SIM_DISTRIBUTIONS = ('uniform', 'hard_a', 'hard_b')

ERROR_COLUMNS = ['scope', 'animal', 'set', 'trial_class', 'key', 'item', 'error']
COLUMNS = {                      # every table keeps its header when empty, so compare.py can read it
    'sessions': ['animal', 'key', 'session_id', 'session_idx', 'date', 'start', 'session_type', 'type_source',
                 'stage', 'distribution', 'contingency', 'source', 'n_trials', 'n_valid', 'n_abort',
                 'n_no_response', 'accuracy', 'n_laser', 'n_check_sent', 'n_check_unknown', 'digest_data',
                 'digest_view', 'digest_correct'],
    'sets': ['animal', 'set', 'key'],
    'pooled': ['animal', 'set', 'trial_class', 'stat', 'value', 'n_trials', 'n_sessions'],
    'per_session': ['animal', 'key', 'trial_class', 'stat', 'value', 'n_trials'],
    'readouts': ['row', 'column', 'value', 'text', 'animal', 'set', 'trial_class', 'readout'],
    'targets': ['animal', 'fit_target', 'cur_bin', 'prev_bin', 'value'],
    'vectors': ['animal', 'rep', 'feature', 'value'],
    'gs_points': ['animal', 'model', 'point', 'matrix', 'cur_bin', 'prev_bin', 'value'],
    'simulator': ['model', 'rep', 'draw', 'feature', 'value'],
    'sim_sessions': ['model', 'distribution', 'session', 'n_trials', 'accuracy', 'digest_stimulus',
                     'digest_choice'],
    'model_sessions': ['animal_id', 'kept', 'session_idx', 'key'],
}
_FLOAT_FIELDS = ('stimulus_raw', 'stimulus', 'choice', 'category', 'reaction_time',
                 'prev_stimulus', 'prev_choice', 'prev_correct', 'prev_category')
_BOOL_FIELDS = ('abort', 'opto_on', 'prev_has_prev')


# ── small helpers ─────────────────────────────────────────────────────────────

def _missing(value) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value))


def _as_float(values) -> np.ndarray:
    """Numbers and booleans as float, anything that is not a number as NaN."""
    arr = np.asarray(values)
    if arr.dtype.kind in 'biuf':
        return arr.astype(float)
    return pd.to_numeric(pd.Series(arr, dtype=object), errors='coerce').to_numpy(dtype=float)


def _as_text(values) -> np.ndarray:
    return np.array(['' if _missing(v) else str(v) for v in np.asarray(values, dtype=object)], dtype=object)


def _iso(value) -> str:
    return value.isoformat() if isinstance(value, (date, time, datetime)) else ''


def _digest(*arrays) -> str:
    """A short hash of numeric arrays, NaN-safe (every NaN hashes alike)."""
    h = hashlib.sha1()
    for a in arrays:
        a = _as_float(a)
        h.update(np.where(np.isnan(a), -1.2345e300, a).tobytes())
        h.update(b'|')
    return h.hexdigest()[:16]


def _field(trials, name: str, n: int) -> np.ndarray:
    value = getattr(trials, name, None)
    if value is None:
        return np.full(n, np.nan)
    value = np.asarray(value)
    return value if len(value) == n else np.full(n, np.nan)


def _error(errors: list, scope: str, exc: Exception, **where) -> None:
    errors.append({'scope': scope, 'animal': where.get('animal', ''), 'set': where.get('set', ''),
                   'trial_class': where.get('trial_class', ''), 'key': where.get('key', ''),
                   'item': where.get('item', ''), 'error': f'{type(exc).__name__}: {exc}'[:300]})


def library_version() -> str:
    import behav_utils
    return str(getattr(behav_utils, '__version__', 'unknown'))


def stat_names() -> list:
    """Every scalar statistic the installed library computes for a two-alternative task."""
    import inspect
    try:
        from behav_utils import list_stats
    except ImportError:
        from behav_utils.stats import list_stats
    if 'task' in inspect.signature(list_stats).parameters:
        from behav_utils import TwoAFC
        return list(list_stats(task=TwoAFC()))
    return list(list_stats())


def compute_scalars(arrays, names) -> pd.Series:
    """compute_stats with a fresh, seeded generator; a failing producer gives NaN."""
    try:
        return compute_stats(arrays, list(names), rng=np.random.default_rng(STAT_SEED), strict=False)
    except TypeError:                                    # a library without ``strict``
        return compute_stats(arrays, list(names), rng=np.random.default_rng(STAT_SEED))


def long_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """A readout's ``to_rows()`` in long form: one row per (row, column), numbers in ``value``, text in
    ``text``. Row positions are the readout's own order (bins, parameters), stable across versions."""
    keep_index = not isinstance(frame.index, pd.RangeIndex)
    frame = frame.reset_index() if keep_index else frame.reset_index(drop=True)
    parts = []
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=object)
        present = [v for v in values if not _missing(v)]
        numeric = all(isinstance(v, (bool, int, float, np.bool_, np.integer, np.floating)) for v in present)
        n = len(frame)
        parts.append(pd.DataFrame({'row': np.arange(n), 'column': str(column),
                                   'value': _as_float(values) if numeric else np.full(n, np.nan),
                                   'text': np.full(n, '', dtype=object) if numeric else _as_text(values)}))
    if not parts:
        return pd.DataFrame(columns=['row', 'column', 'value', 'text'])
    return pd.concat(parts, ignore_index=True)


# ── sessions and trials ───────────────────────────────────────────────────────

def session_key(session) -> str:
    """The session's folder name, the identity that survives the 1.0 change of session ids: 1.0 keeps it
    as ``source_id``, 0.10 has the path of a CSV inside it. Synthetic sessions have neither: their id."""
    source_id = getattr(session, 'source_id', None)
    if source_id:
        return str(source_id)
    path = getattr(session, 'csv_path', None)
    if path:
        return Path(path).parent.name
    return str(session.session_id)


def trial_frame(session) -> pd.DataFrame:
    """Every trial of one session: the fields statistics read, the lag-1 view and the stimulus check.

    ``outcome_label`` and ``correct_recorded`` are the CSV's columns in both versions (0.10 keeps them in
    ``outcome`` and ``correct``, 1.0 in ``outcome_recorded`` and ``correct_recorded``); ``outcome_code``
    is 1.0's score (NaN on 0.10).
    """
    t = session.trials
    n = t.n_trials
    if hasattr(t, 'outcome_recorded'):
        label, code = _field(t, 'outcome_recorded', n), _as_float(t.outcome)
        recorded = _field(t, 'correct_recorded', n)
    else:
        label, code, recorded = np.asarray(t.outcome), np.full(n, np.nan), np.asarray(t.correct)
    frame = {'row': np.arange(n), 'trial_number': _as_float(t.trial_number)}
    for name in _FLOAT_FIELDS + _BOOL_FIELDS:
        frame[name] = _as_float(_field(t, name, n))
    frame['correct'] = _as_float(t.correct)
    frame['correct_recorded'] = _as_float(recorded)
    frame['outcome_code'] = code
    frame['outcome_label'] = _as_text(label)
    frame['choice_raw'] = _as_text(_field(t, 'choice_raw', n))
    frame['distribution'] = _as_text(_field(t, 'distribution', n))
    check = (getattr(t, 'extra', None) or {}).get('stimulus_check')
    known = check is not None and len(check) == n
    frame['stimulus_check'] = _as_float(check) if known else np.full(n, np.nan)
    return pd.DataFrame(frame)


def session_row(animal_id: str, session) -> dict:
    t = session.trials
    n = t.n_trials
    summary = session.summary()
    check = (getattr(t, 'extra', None) or {}).get('stimulus_check')
    check = _as_float(check) if check is not None and len(check) == n else np.full(n, np.nan)
    source = getattr(session, 'source', None) or getattr(session, 'csv_path', None) or ''
    def summarised(name):
        value = getattr(summary, name, '')
        return '' if value is None else str(value)

    return {
        'animal': animal_id, 'key': session_key(session), 'session_id': str(session.session_id),
        'session_idx': int(session.session_idx), 'date': _iso(session.date),
        'start': _iso(getattr(session, 'start', None)), 'session_type': str(session.session_type),
        'type_source': str(getattr(session, 'session_type_source', '')), 'stage': summarised('stage'),
        'distribution': summarised('distribution'),
        'contingency': str(session.metadata.get('sound_contingency', '') or ''), 'source': str(source),
        'n_trials': int(n), 'n_valid': getattr(summary, 'n_valid', np.nan),
        'n_abort': getattr(summary, 'n_abort', np.nan),
        'n_no_response': getattr(summary, 'n_no_response', np.nan),
        'accuracy': getattr(summary, 'accuracy', np.nan),
        'n_laser': int(np.nansum(_as_float(_field(t, 'opto_on', n)))),
        'n_check_sent': int(np.sum(check == 1)), 'n_check_unknown': int(np.sum(np.isnan(check))),
        'digest_data': _digest(_field(t, 'stimulus_raw', n), t.choice, _field(t, 'abort', n),
                               _field(t, 'opto_on', n)),
        'digest_view': _digest(_field(t, 'prev_stimulus', n), _field(t, 'prev_choice', n),
                               _field(t, 'prev_has_prev', n)),
        'digest_correct': _digest(t.correct),
    }


# ── session sets ──────────────────────────────────────────────────────────────

def preset_names(config_path) -> list:
    """The session presets the project config defines (anchors such as ``_controls`` left out)."""
    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    presets = raw.get('session_presets') or {}
    return [str(k) for k, v in presets.items() if isinstance(v, dict) and not str(k).startswith('_')]


def default_sets(config_path) -> list:
    """Every set the analyses select from: all sessions, each config preset, and the opto designs'
    sets (distribution x session type, as ``data/cohort.collect_sessions_*`` select them)."""
    design = [f'design:{d}:{t}' for d in DESIGN_DISTRIBUTIONS for t in DESIGN_TYPES]
    return ['all_sessions'] + preset_names(config_path) + design


def select_set(animal, name: str) -> list:
    if name == 'all_sessions':
        return select_sessions(animal)
    if name.startswith('design:'):
        _, distribution, session_type = name.split(':')
        return select_sessions(animal, distribution=distribution, session_type=session_type)
    return select_sessions(animal, preset=name)


# ── one animal ────────────────────────────────────────────────────────────────

def _ensure_presets(config_path) -> None:
    from sound_categorisation.data.cohort import ensure_presets
    ensure_presets(Path(config_path))


def capture_animal(animal, set_names, names, *, config_path=None, readouts: bool = True) -> dict:
    """Everything for one animal: its sessions and trials, set membership, pooled statistics and readouts
    per set x trial class, and statistics per session x trial class. Returns ``{table: DataFrame}``."""
    if config_path is not None:                          # a worker process starts without the presets
        _ensure_presets(config_path)
    aid = str(animal.animal_id)
    out = {k: [] for k in ('sessions', 'trials', 'sets', 'pooled', 'per_session', 'readouts')}
    errors: list = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for s in animal.sessions:
            key = session_key(s)
            try:
                out['sessions'].append(pd.DataFrame([session_row(aid, s)]))
                out['trials'].append(trial_frame(s).assign(animal=aid, key=key))
            except Exception as e:
                _error(errors, 'session', e, animal=aid, key=key)
            for tc in SESSION_CLASSES:
                try:
                    phase = filter_trials([s], trial_type=tc)
                    if not phase:
                        continue
                    arrays = TrialArrays.from_sessions(phase)
                    values = compute_scalars(arrays, names)
                    out['per_session'].append(pd.DataFrame({
                        'animal': aid, 'key': key, 'trial_class': tc, 'stat': values.index,
                        'value': values.to_numpy(dtype=float), 'n_trials': len(arrays.choice)}))
                except Exception as e:
                    _error(errors, 'per_session', e, animal=aid, key=key, trial_class=tc)
        for set_name in set_names:
            try:
                sessions = select_set(animal, set_name)
            except Exception as e:
                _error(errors, 'set', e, animal=aid, set=set_name)
                continue
            out['sets'].append(pd.DataFrame({'animal': aid, 'set': set_name,
                                             'key': [session_key(s) for s in sessions]}))
            for tc in TRIAL_CLASSES:
                try:
                    phase = filter_trials(sessions, trial_type=tc) if sessions else []
                    if not phase:
                        continue
                    arrays = TrialArrays.from_sessions(phase)
                    values = compute_scalars(arrays, names)
                except Exception as e:
                    _error(errors, 'pooled', e, animal=aid, set=set_name, trial_class=tc)
                    continue
                out['pooled'].append(pd.DataFrame({
                    'animal': aid, 'set': set_name, 'trial_class': tc, 'stat': values.index,
                    'value': values.to_numpy(dtype=float), 'n_trials': len(arrays.choice),
                    'n_sessions': len(phase)}))
                if not readouts:
                    continue
                for readout, fn, kwargs in READOUTS:
                    try:
                        rows = long_rows(fn(arrays, **kwargs).to_rows())
                    except Exception as e:
                        _error(errors, 'readout', e, animal=aid, set=set_name, trial_class=tc, item=readout)
                        continue
                    out['readouts'].append(rows.assign(animal=aid, set=set_name, trial_class=tc,
                                                       readout=readout))
    tables = {k: _stacked(v, k) for k, v in out.items()}
    tables['errors'] = pd.DataFrame(errors, columns=ERROR_COLUMNS)
    return tables


def _stacked(frames: list, table: str) -> pd.DataFrame:
    frames = [f for f in frames if len(f)]
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=COLUMNS.get(table, []))


# ── model identification ─────────────────────────────────────────────────────

def capture_model_id(experiment, records, sessions_table, n_sim: int) -> dict:
    """The inputs model identification computes from: the selected sessions, the empirical update-matrix
    and conditional-psychometric targets, the SBI observation vectors, and fingerprints of the simulators
    (fixed parameters and seeds), which must not move while the data do not."""
    try:
        from behav_utils import pool_arrays
    except ImportError:                                  # 0.10 kept it below the top level
        from behav_utils.data.ops.filtering import pool_arrays

    from sound_categorisation.inference.constants import SBI_STATS
    from sound_categorisation.inference.grid_search import (
        DEFAULT_GRID,
        sessions_to_arrays,
        simulate_model_matrices,
    )
    from sound_categorisation.inference.representation import feature_names, to_stat_vector
    from sound_categorisation.inference.selection import _empirical_target
    from sound_categorisation.inference.simulator import (
        build_simulator,
        get_bounds_arrays,
        simulate_sessions,
        theta_to_params,
    )
    from sound_categorisation.settings import (
        FIT_TARGETS,
        GS_BURN_IN,
        GS_N_BINS,
        SBI_BURN_IN,
        SBI_DESIGN_FEATURES,
        SBI_REPRESENTATIONS,
        SBI_TRAIN_REPS,
    )

    errors: list = []
    out = {k: [] for k in ('sessions', 'targets', 'vectors', 'gs_points', 'simulator', 'sim_sessions')}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        keys_by_idx = {(aid, s.session_idx): session_key(s)
                       for aid, a in experiment.animals.items() for s in a.sessions}
        if sessions_table is not None and len(sessions_table):
            table = sessions_table.copy()
            table['key'] = [keys_by_idx.get((a, int(i)), '') if pd.notna(i) else ''
                            for a, i in zip(table['animal_id'], table['session_idx'])]
            out['sessions'].append(table)
        else:
            out['sessions'].append(pd.DataFrame(
                [{'animal_id': r.animal_id, 'kept': True, 'session_idx': s.session_idx, 'key': session_key(s)}
                 for r in records for s in r.sessions]))

        for r in records:
            aid = str(r.animal_id)
            try:
                pooled = pool_arrays(r.sessions)
                for target in FIT_TARGETS:
                    m = np.asarray(_empirical_target(pooled, target, GS_N_BINS), dtype=float)
                    i, j = np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), indexing='ij')
                    out['targets'].append(pd.DataFrame({'animal': aid, 'fit_target': target,
                                                        'cur_bin': i.ravel(), 'prev_bin': j.ravel(),
                                                        'value': m.ravel()}))
            except Exception as e:
                _error(errors, 'model_id', e, animal=aid, item='targets')
            for rep in SBI_TRAIN_REPS:
                mode = SBI_REPRESENTATIONS[rep]['mode']
                try:
                    x = to_stat_vector(r.sessions, mode=mode, design_features=SBI_DESIGN_FEATURES)
                    names = feature_names(mode, SBI_STATS, SBI_DESIGN_FEATURES)
                    out['vectors'].append(pd.DataFrame({'animal': aid, 'rep': rep, 'feature': names,
                                                        'value': np.asarray(x, dtype=float)}))
                except Exception as e:
                    _error(errors, 'model_id', e, animal=aid, item=f'vector_{rep}')

        for r in records[:2]:                                     # grid-search simulation on real designs
            aid = str(r.animal_id)
            try:
                d = sessions_to_arrays(r.sessions)
            except Exception as e:
                _error(errors, 'model_id', e, animal=aid, item='gs_arrays')
                continue
            for model, points in GS_POINTS.items():
                grid = DEFAULT_GRID[model]
                for k, (sigma, a_rep, p1, p2) in enumerate(points):
                    try:
                        um, cm = simulate_model_matrices(
                            model, d['stimuli'], d['categories'], d['no_response'], d['not_blockstart'],
                            sigma, a_rep, p1, p2, grid.model_param1_name, grid.model_param2_name,
                            seed=GS_SEED + k, burn_in=GS_BURN_IN, n_bins=GS_N_BINS)
                    except Exception as e:
                        _error(errors, 'model_id', e, animal=aid, item=f'gs_{model}_{k}')
                        continue
                    for matrix, m in (('um', um), ('cm', cm)):
                        m = np.asarray(m, dtype=float)
                        i, j = np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), indexing='ij')
                        out['gs_points'].append(pd.DataFrame({
                            'animal': aid, 'model': model, 'point': k, 'matrix': matrix,
                            'cur_bin': i.ravel(), 'prev_bin': j.ravel(), 'value': m.ravel()}))

        for model in ('be', 'sc'):
            lo, hi = get_bounds_arrays(model)
            thetas = np.random.default_rng(FINGERPRINT_SEED).uniform(lo, hi, size=(max(n_sim, 1), len(lo)))
            for rep in SBI_TRAIN_REPS:
                cfg = SBI_REPRESENTATIONS[rep]
                try:
                    sim_fn, _, _ = build_simulator(model, 'uniform', N=cfg['N'], T=cfg['T'],
                                                   burn_in=SBI_BURN_IN, mode=cfg['mode'],
                                                   design_features=SBI_DESIGN_FEATURES)
                    names = feature_names(cfg['mode'], SBI_STATS, SBI_DESIGN_FEATURES)
                    for k in range(n_sim):
                        x = sim_fn(thetas[k], seed=FINGERPRINT_SEED + k)
                        out['simulator'].append(pd.DataFrame({'model': model, 'rep': rep, 'draw': k,
                                                              'feature': names,
                                                              'value': np.asarray(x, dtype=float)}))
                except Exception as e:
                    _error(errors, 'model_id', e, item=f'simulator_{model}_{rep}')
            for distribution in SIM_DISTRIBUTIONS:
                try:
                    sims = simulate_sessions(model, theta_to_params(thetas[0], model), [distribution] * 2,
                                             300, SBI_BURN_IN, np.random.default_rng(FINGERPRINT_SEED))
                except Exception as e:
                    _error(errors, 'model_id', e, item=f'sim_sessions_{model}_{distribution}')
                    continue
                for k, s in enumerate(sims):
                    t = s.trials
                    choice, category = _as_float(t.choice), _as_float(t.category)
                    ok = ~np.isnan(choice)
                    out['sim_sessions'].append(pd.DataFrame([{
                        'model': model, 'distribution': distribution, 'session': k, 'n_trials': t.n_trials,
                        'accuracy': float(np.mean(choice[ok] == category[ok])) if ok.any() else np.nan,
                        'digest_stimulus': _digest(t.stimulus), 'digest_choice': _digest(t.choice)}]))
    tables = {k: _stacked(v, 'model_sessions' if k == 'sessions' else k) for k, v in out.items()}
    tables['errors'] = pd.DataFrame(errors, columns=ERROR_COLUMNS)
    return tables


# ── the whole capture ────────────────────────────────────────────────────────

def _versions() -> dict:
    out = {'python': platform.python_version(), 'platform': f'{platform.system()}-{platform.machine()}'}
    for name in ('behav_utils', 'sound_categorisation', 'numpy', 'scipy', 'pandas'):
        try:
            out[name] = str(getattr(__import__(name), '__version__', 'unknown'))
        except Exception:
            out[name] = 'not installed'
    return out


def _write(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def capture(experiment, out_dir, *, config_path, records=None, sessions_table=None, names=None,
            set_names=None, n_jobs: int = 1, n_sim: int = 25, readouts: bool = True, meta: dict | None = None,
            overwrite: bool = False) -> Path:
    """Capture ``experiment`` into ``out_dir``. ``records`` (``AnimalRecord``s, as ``select_animals`` gives)
    switch on the model-identification tables; ``names`` and ``set_names`` default to every statistic and
    every set. Returns ``out_dir``."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f'{out_dir} is not empty; pass overwrite=True (--overwrite) to replace it')
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)
    _ensure_presets(config_path)
    names = list(names) if names is not None else stat_names()
    set_names = list(set_names) if set_names is not None else default_sets(config_path)
    animals = [experiment.animals[a] for a in sorted(experiment.animals)]

    if n_jobs == 1:
        per_animal = [capture_animal(a, set_names, names, readouts=readouts) for a in animals]
    else:
        from joblib import Parallel, delayed
        per_animal = Parallel(n_jobs=n_jobs)(
            delayed(capture_animal)(a, set_names, names, config_path=config_path, readouts=readouts)
            for a in animals)

    def stack(table):
        return _stacked([t[table] for t in per_animal], table)

    _write(stack('sessions'), out_dir / 'sessions.csv')
    _write(stack('sets'), out_dir / 'sets.csv')
    _write(stack('pooled'), out_dir / 'stats_pooled.csv.gz')
    _write(stack('per_session'), out_dir / 'stats_sessions.csv.gz')
    _write(stack('readouts'), out_dir / 'readouts.csv.gz')
    for a, tables in zip(animals, per_animal):
        if len(tables['trials']):
            _write(tables['trials'], out_dir / 'trials' / f'{a.animal_id}.csv.gz')
    errors = [stack('errors')]

    if records is not None:
        model_id = capture_model_id(experiment, records, sessions_table, n_sim)
        for name, table in model_id.items():
            if name != 'errors':
                _write(table, out_dir / 'model_id' / f'{name}.csv')
        errors.append(model_id['errors'])
    errors = [e for e in errors if len(e)]
    errors = pd.concat(errors, ignore_index=True) if errors else pd.DataFrame(columns=ERROR_COLUMNS)
    _write(errors, out_dir / 'errors.csv')

    from sound_categorisation.data.paths import git_state
    sessions = stack('sessions')
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'library': library_version(), 'versions': _versions(), 'git': git_state(),
        'argv': list(sys.argv), 'config': str(config_path.resolve()),
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'stat_names': names, 'sets': set_names, 'trial_classes': list(TRIAL_CLASSES),
        'readouts': [r[0] for r in READOUTS] if readouts else [],
        'n_animals': len(animals), 'n_sessions': len(sessions),
        'n_trials': int(sessions['n_trials'].sum()) if len(sessions) else 0,
        'model_id': records is not None, 'n_sim': n_sim if records is not None else 0,
        'n_errors': len(errors), **(meta or {}),
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1, default=str) + '\n')
    return out_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True, help='an empty or new folder')
    p.add_argument('--snapshot', type=Path, default=None, help='default: the project default snapshot')
    p.add_argument('--config', type=Path, default=None, help='default: config.yaml at the repository root')
    p.add_argument('--cohort', default='all-mice', help='cohort for the model-identification tables')
    p.add_argument('--n-jobs', type=int, default=1, help='animals captured at once (e.g. the cores you have)')
    p.add_argument('--n-sim', type=int, default=25,
                   help='simulator fingerprints per model and representation')
    p.add_argument('--skip-model-id', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args(argv)
    if a.out.exists() and any(a.out.iterdir()) and not a.overwrite:
        p.error(f'{a.out} is not empty; pass --overwrite to replace it')    # before the slow load

    from sound_categorisation.data.cohort import load_experiment_any, select_animals
    from sound_categorisation.data.paths import DEFAULT_CONFIG
    config_path = a.config or DEFAULT_CONFIG
    print(f'behav_utils {library_version()}; loading the experiment …', flush=True)
    experiment = load_experiment_any(config_path=config_path, snapshot_path=a.snapshot)
    records = sessions_table = None
    if not a.skip_model_id:
        records, sessions_table = select_animals(a.cohort, config_path=config_path, experiment=experiment)
    out = capture(experiment, a.out, config_path=config_path, records=records, sessions_table=sessions_table,
                  n_jobs=a.n_jobs, n_sim=a.n_sim, overwrite=a.overwrite,
                  meta={'snapshot': str(a.snapshot) if a.snapshot else 'default', 'cohort': a.cohort})
    manifest = json.loads((out / 'manifest.json').read_text())
    print(f"captured {manifest['n_animals']} animals, {manifest['n_sessions']} sessions, "
          f"{manifest['n_trials']} trials into {out}; {manifest['n_errors']} errors (errors.csv)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
