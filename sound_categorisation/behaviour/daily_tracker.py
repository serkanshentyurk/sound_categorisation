"""
The daily tracker: how each animal is doing, from the raw session files (data.raw_mirror), not the snapshot.

    from sound_categorisation.behaviour.daily_tracker import (
        compute_daily_tracker, load_mirror, load_tracker_settings)
    settings = load_tracker_settings()             # config.yaml, block daily_tracker
    experiment, notes = load_mirror()              # the behav_utils loader, unchanged, on the local copy
    tracker = compute_daily_tracker(experiment, settings, notes=notes)

What a ``DailyTracker`` holds:
    sessions            one row per animal and day of the window, days without a session included
    conditions          per animal and condition (a preset), its last ``last_sessions`` sessions: pooled
                        ("mega session") and the mean and SD of the per-session values; trials without
                        light, and with light where the condition has any
    condition_sessions  the per-session values behind those summaries (the individual sessions)
    flags               side bias, low trials, accuracy drop, missing day
    move_on             the ready-to-move-on mark from TRAINING.md's criteria (training steps only)
    curves, latest      psychometric readouts per condition, and each animal's latest session

Statistics use trials without light, so laser and masking days compare with regular ones. The spatial
columns turn category B into the right spout with the session's sound contingency:
``right_choices_minus_50`` is side_bias and ``accuracy_right_minus_left`` is accuracy_b_minus_a.

Every number comes from config.yaml's daily_tracker block (no defaults here), except the side-bias test,
which is adaptation.BiasRule's (shared with the switch report) applied to the free-fit PSE: pse_fixed needs
the animal's expert Uniform sessions, and on this data the expert preset cannot yet find them.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from behav_utils.analysis.group import average_arrays
from behav_utils.analysis.phase import compute_phase_stats
from behav_utils.config.schema import load_config
from behav_utils.data.arrays import TrialArrays
from behav_utils.data.loading import load_experiment
from behav_utils.data.ops.filtering import filter_trials
from behav_utils.data.ops.selection import select_sessions
from behav_utils.readouts import compute_psychometric_curve
from behav_utils.stats import PSYCHOMETRIC

from sound_categorisation.behaviour.adaptation import BiasRule, session_bias_checks
from sound_categorisation.data.paths import REPO_ROOT, raw_mirror_dir
from sound_categorisation.data.raw_mirror import session_parameters

EARLY_STAGES = ('Habituation', 'Lick_To_Release', 'Three_And_Three')
STATS = ('accuracy', 'hard_accuracy', 'easy_accuracy', 'side_bias', 'accuracy_b_minus_a', 'pse', *PSYCHOMETRIC)
COUNTS = ('trials', 'responded', 'correct', 'no_response', 'early', 'light_trials')
SPATIAL = {'side_bias': 'right_choices_minus_50', 'accuracy_b_minus_a': 'accuracy_right_minus_left'}
B_IS_RIGHT = 'Low_Left_High_Right'      # category B (high sounds) is rewarded on the right spout
CONDITION_COLUMNS = ['animal', 'preset', 'trials', 'n_sessions', 'first_date', 'last_date', 'stat', 'pooled',
                     'mean', 'sd']
CONDITION_SESSION_COLUMNS = ['animal', 'preset', 'trials', 'date', 'session_no', 'stat', 'value']
MISSING_COLUMNS = ['animal', 'last_session', 'days_since']


# ═══════════════════════════════════════════════════════════════════════════
# Settings: config.yaml, block daily_tracker
# ═══════════════════════════════════════════════════════════════════════════

_KEYS = {
    'days': ('days',),
    'last_sessions': ('last_sessions',),
    'summary_days': ('summary_days',),
    'early_steps': ('early_steps',),
    'full_task_training': ('full_task_training',),
    'expected_cohorts': ('expected_cohorts',),
    'min_correct_trials': ('flags', 'min_correct_trials'),
    'accuracy_drop': ('flags', 'accuracy_drop'),
    'accuracy_drop_reference_sessions': ('flags', 'accuracy_drop_reference_sessions'),
    'early_step_min_trials': ('move_on', 'early_step_min_trials'),
    'sessions_in_a_row': ('move_on', 'full_task', 'sessions_in_a_row'),
    'move_on_min_correct_trials': ('move_on', 'full_task', 'min_correct_trials'),
    'min_accuracy': ('move_on', 'full_task', 'min_accuracy'),
    'min_hard_accuracy': ('move_on', 'full_task', 'min_hard_accuracy'),
    'max_accuracy_difference': ('move_on', 'full_task', 'max_accuracy_difference'),
}


@dataclass(frozen=True)
class TrackerSettings:
    """The numbers in config.yaml's daily_tracker block (see the block's comments for their meaning)."""
    days: int
    last_sessions: int
    summary_days: int | None            # None: the whole history ('all' in the YAML)
    early_steps: tuple
    full_task_training: tuple
    expected_cohorts: tuple
    min_correct_trials: int
    accuracy_drop: float
    accuracy_drop_reference_sessions: int
    early_step_min_trials: int
    sessions_in_a_row: int
    move_on_min_correct_trials: int
    min_accuracy: float
    min_hard_accuracy: float
    max_accuracy_difference: float

    @classmethod
    def from_block(cls, block: dict) -> TrackerSettings:
        values, missing = {}, []
        for name, path in _KEYS.items():
            node = block
            for key in path:
                if not isinstance(node, dict) or key not in node:
                    missing.append('daily_tracker.' + '.'.join(path))
                    break
                node = node[key]
            else:
                values[name] = node
        if missing:
            raise ValueError('config.yaml, block daily_tracker, is missing: ' + ', '.join(missing))
        sd = values['summary_days']
        values['summary_days'] = None if str(sd).strip().lower() == 'all' else int(sd)
        values['early_steps'] = tuple(values['early_steps'])
        values['full_task_training'] = tuple(values['full_task_training'])
        values['expected_cohorts'] = tuple(values['expected_cohorts'] or ())
        return cls(**values)

    def table(self, rule: BiasRule | None = None) -> pd.DataFrame:
        """Every number in use, with the place it is set (the report's last section)."""
        rows = []
        for name, path in _KEYS.items():
            v = getattr(self, name)
            if name == 'summary_days' and v is None:
                v = 'all'
            if isinstance(v, tuple):
                v = ', '.join(v)
            rows.append({'setting': 'daily_tracker.' + '.'.join(path), 'value': v, 'set in': 'config.yaml'})
        r = rule or BiasRule()
        for name in ('pse', 'lapse', 'accuracy'):
            rows.append({'setting': f'BiasRule.{name}', 'value': getattr(r, name),
                         'set in': 'sound_categorisation/behaviour/adaptation.py'})
        return pd.DataFrame(rows)


def load_tracker_settings(config_path: Path | None = None) -> TrackerSettings:
    """The daily_tracker block of config.yaml. Stops with the missing keys rather than guessing them."""
    path = Path(config_path) if config_path else REPO_ROOT / 'config.yaml'
    block = (yaml.safe_load(path.read_text()) or {}).get('daily_tracker')
    if not isinstance(block, dict):
        raise ValueError(f'{path} has no daily_tracker block; see START_HERE.md, "Daily behaviour tracker".')
    return TrackerSettings.from_block(block)


# ═══════════════════════════════════════════════════════════════════════════
# Loading the local copy
# ═══════════════════════════════════════════════════════════════════════════

class _Collect(logging.Handler):
    def __init__(self, notes):
        super().__init__(level=logging.WARNING)
        self.notes = notes

    def emit(self, record):
        self.notes.append(record.getMessage())


def load_mirror(config_path: Path | None = None, data_dir: Path | None = None) -> tuple:
    """(ExperimentData, notes): the behav_utils loader, unchanged, on the local copy.

    A session that fails to load is skipped and named in ``notes``, with the loader's other warnings. Its
    warning that every trial is a no-response is dropped: in the early steps no response is asked for.
    """
    cfg = load_config(Path(config_path) if config_path else REPO_ROOT / 'config.yaml')
    cfg.file_structure.data_dir = str(data_dir or raw_mirror_dir())
    notes: list = []
    handler = _Collect(notes)
    logger = logging.getLogger('behav_utils')
    logger.addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            experiment = load_experiment(cfg, strict=False)
    finally:
        logger.removeHandler(handler)
    notes += [str(w.message) for w in caught if 'no-response' not in str(w.message)]
    return experiment, notes


# ═══════════════════════════════════════════════════════════════════════════
# The result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CurveSet:
    """Psychometric readouts for one condition and trial class: the pooled fit, each session's fit, and
    the point-wise mean of the session curves on ``x``."""
    pooled: object | None
    sessions: tuple = ()            # ((date, PsychometricCurve), ...), oldest first
    x: np.ndarray | None = None
    mean_y: np.ndarray | None = None


@dataclass(frozen=True)
class DailyTracker:
    today: date
    days: int
    sessions: pd.DataFrame
    conditions: pd.DataFrame
    condition_sessions: pd.DataFrame
    flags: pd.DataFrame
    move_on: pd.DataFrame
    curves: dict = field(default_factory=dict)      # (animal, preset, trials) -> CurveSet
    latest: dict = field(default_factory=dict)      # animal -> (date, preset, PsychometricCurve)
    settings: TrackerSettings | None = None
    rule: BiasRule | None = None
    notes: tuple = ()
    missing: pd.DataFrame | None = None     # expected animals with no session in the window
    newest_file: object = None              # when the newest session file in the window was written

    @property
    def animals(self) -> list:
        return sorted(self.sessions['animal'].unique()) if len(self.sessions) else []

    def to_rows(self) -> pd.DataFrame:
        """Tidy rows of the window's sessions: {animal, date, preset, stat, value}."""
        cols = [*COUNTS, 'accuracy', 'hard_accuracy', 'easy_accuracy', *SPATIAL.values(), 'pse',
                *PSYCHOMETRIC]
        s = self.sessions.dropna(subset=['folder'])
        rows = s.melt(id_vars=['animal', 'date', 'preset'], value_vars=cols, var_name='stat', value_name='value')
        return rows.sort_values(['animal', 'date', 'stat']).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Per session
# ═══════════════════════════════════════════════════════════════════════════

def _first_text(values) -> str:
    if values is None:
        return ''
    for v in np.asarray(values, dtype=object):
        if v is not None and str(v).strip() and str(v).lower() != 'nan':
            return str(v).strip()
    return ''


def _any_true(values) -> bool:
    if values is None:
        return False
    return any(str(v).strip().lower() in ('true', '1') for v in np.asarray(values, dtype=object))


def _record(animal_id: str, s, settings: TrackerSettings) -> dict:
    t = s.trials
    extra = getattr(t, 'extra', None) or {}
    preset = _first_text(extra.get('Session_Type'))
    stage = s.stage or ''
    if preset in settings.early_steps or stage in EARLY_STAGES:
        group = 'early step'
    elif preset in settings.full_task_training:
        group = 'training'
    else:
        group = 'experiment'
    folder = Path(s.csv_path).parent if s.csv_path else None
    params = session_parameters(folder) if folder is not None else {}
    outcome = np.asarray(getattr(t, 'outcome', np.array([])), dtype=str)
    choice, category = np.asarray(t.choice, dtype=float), np.asarray(t.category, dtype=float)
    valid = np.asarray(t.valid_mask, dtype=bool)
    light = np.asarray(getattr(t, 'opto_on', np.zeros(len(choice))), dtype=bool)
    rec = {
        'animal': animal_id, 'date': s.date, 'session_idx': s.session_idx,
        'folder': folder.name if folder is not None else '', 'stage': stage, 'preset': preset, 'group': group,
        'distribution': s.distribution or '', 'contingency': str(s.metadata.get('sound_contingency', '') or ''),
        'rig': params.get('Rig_ID', ''),
        'test_run': _any_true(extra.get('Emulator')) or str(params.get('Agent_Sim', '')).lower() == 'true',
        'trials': len(choice), 'responded': int(valid.sum()),
        'correct': int(np.sum(valid & (choice == category))),
        'no_response': int(np.sum(outcome == 'Abort')), 'early': int(np.sum(outcome == 'Early')),
        'light_trials': int(light.sum()),
        'file_time': datetime.fromtimestamp(os.path.getmtime(s.csv_path)) if s.csv_path else None,
    }
    if group == 'early step':          # no response is asked for: not measured, rather than zero
        for key in ('responded', 'correct', 'no_response', 'early'):
            rec[key] = np.nan
    return rec


def _spatial_sign(contingency: str) -> float:
    return 1.0 if contingency == B_IS_RIGHT else -1.0


def _with_spatial(values: dict, contingency: str) -> dict:
    out = dict(values)
    sign = _spatial_sign(contingency)
    for cat, spatial in SPATIAL.items():
        out[spatial] = sign * out[cat] if np.isfinite(out.get(cat, np.nan)) else np.nan
    return out


def _per_session_stats(sessions: list, trial_type: str) -> dict:
    """{session_idx: {stat: value}} on the given trial class, one compute_phase_stats call."""
    if not sessions:
        return {}
    phase = filter_trials(sessions, trial_type=trial_type, min_trials=0)
    if not phase:
        return {}
    r = compute_phase_stats(phase, list(STATS), per_session=True)
    out: dict = {}
    for row in r.sessions.itertuples(index=False):
        out.setdefault(row.session, {})[row.stat] = float(row.value)
    return out


def _curve(sessions: list):
    arrays = TrialArrays.from_sessions(sessions)
    if arrays.valid().n_trials < 10:
        return None
    return compute_psychometric_curve(arrays, n_bootstrap=0)


# ═══════════════════════════════════════════════════════════════════════════
# The computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_daily_tracker(experiment, settings: TrackerSettings, *, today: date | None = None,
                          days: int | None = None, rule: BiasRule | None = None, notes=(), expected=(),
                          last_seen: dict | None = None, animals=None) -> DailyTracker:
    """Everything the report shows, for every animal with a session in the last ``days`` days.

    ``expected``: animals to list even without a session in the window (with ``last_seen``, their newest
    session folder on ceph, from MirrorResult.last_session). ``animals``: only these (e.g. a cohort).
    """
    today = today or date.today()
    days = days or settings.days
    rule = rule or BiasRule()
    window = [today - timedelta(days=i) for i in range(days - 1, -1, -1)]
    start = window[0]

    records, cond_rows, cond_session_rows = [], [], []
    curves, latest = {}, {}
    for aid in sorted(experiment.animals):
        if animals is not None and aid not in animals:
            continue
        animal = experiment.animals[aid]
        sel = select_sessions(animal, min_trials=0)
        if not any(start <= s.date <= today for s in sel):
            continue
        by_idx = {s.session_idx: s for s in sel}
        recs = {s.session_idx: _record(aid, s, settings) for s in sel if s.date <= today}
        n_all, n_preset = 0, {}
        for i in sorted(recs, key=lambda i: recs[i]['date']):    # session numbers; test runs not counted
            r = recs[i]
            if r['test_run']:
                r['session_no'] = r['preset_no'] = np.nan
                continue
            n_all += 1
            n_preset[r['preset']] = n_preset.get(r['preset'], 0) + 1
            r['session_no'], r['preset_no'] = n_all, n_preset[r['preset']]
        full = [i for i, r in recs.items() if r['group'] != 'early step']
        real_full = [i for i in full if not recs[i]['test_run']]

        # conditions: the last N real full-task sessions of each preset
        by_preset: dict = {}
        for i in sorted(real_full, key=lambda i: recs[i]['date']):
            if recs[i]['preset']:
                by_preset.setdefault(recs[i]['preset'], []).append(i)
        last_n = {p: idx[-settings.last_sessions:] for p, idx in by_preset.items()}

        # per-session statistics (trials without light) for what the report needs
        needed = {i for i in full if recs[i]['date'] >= start}
        needed |= {i for idx in last_n.values() for i in idx}
        for i in [i for i in full if recs[i]['date'] >= start and not recs[i]['test_run']]:
            same = [j for j in real_full if recs[j]['preset'] == recs[i]['preset'] and recs[j]['date'] < recs[i]['date']]
            needed |= set(sorted(same, key=lambda j: recs[j]['date'])[-settings.accuracy_drop_reference_sessions:])
        needed |= set(sorted(real_full, key=lambda j: recs[j]['date'])[-settings.sessions_in_a_row:])
        stats = _per_session_stats([by_idx[i] for i in sorted(needed)], 'non_opto')
        for i, r in recs.items():
            r.update(_with_spatial({k: stats.get(i, {}).get(k, np.nan) for k in STATS}, r['contingency']))
        records += list(recs.values())

        for preset, idx in last_n.items():
            group_sessions = [by_idx[i] for i in idx]
            contingency = pd.Series([recs[i]['contingency'] for i in idx]).mode().iloc[0]
            classes = [('no light', 'non_opto')]
            if sum(recs[i]['light_trials'] for i in idx) > 0:
                classes.append(('light', 'opto'))
            for label, trial_type in classes:
                phase = filter_trials(group_sessions, trial_type=trial_type, min_trials=0)
                if not phase:
                    continue
                per = stats if trial_type == 'non_opto' else _per_session_stats(group_sessions, trial_type)
                pooled = _with_spatial(compute_phase_stats(phase, list(STATS)).pooled.to_dict(), contingency)
                per_rows = []
                for s in phase:
                    vals = _with_spatial({k: per.get(s.session_idx, {}).get(k, np.nan) for k in STATS},
                                         recs[s.session_idx]['contingency'])
                    for stat in (*STATS, *SPATIAL.values()):
                        per_rows.append({'animal': aid, 'preset': preset, 'trials': label, 'date': s.date,
                                         'session_no': recs[s.session_idx]['session_no'], 'stat': stat,
                                         'value': vals[stat]})
                cond_session_rows += per_rows
                per_df = pd.DataFrame(per_rows)
                for stat in (*STATS, *SPATIAL.values()):
                    v = per_df.loc[per_df['stat'] == stat, 'value'].to_numpy(dtype=float)
                    v = v[np.isfinite(v)]
                    cond_rows.append({
                        'animal': aid, 'preset': preset, 'trials': label, 'n_sessions': len(phase),
                        'first_date': min(s.date for s in phase), 'last_date': max(s.date for s in phase),
                        'stat': stat, 'pooled': pooled.get(stat, np.nan),
                        'mean': float(v.mean()) if len(v) else np.nan,
                        'sd': float(v.std(ddof=1)) if len(v) > 1 else np.nan})
                session_curves = [(s.date, c) for s in phase if (c := _curve([s])) is not None]
                pooled_curve = _curve(phase)
                x = mean_y = None
                ok = [c for _, c in session_curves if c.success]
                if ok:
                    x = ok[0].x
                    ys = [c.y if np.allclose(c.x, x) else np.interp(x, c.x, c.y) for c in ok]
                    mean_y = average_arrays(ys).mean
                curves[(aid, preset, label)] = CurveSet(pooled_curve, tuple(session_curves), x, mean_y)
                if label == 'no light' and session_curves:
                    latest_curve = session_curves[-1]
                    prev = latest.get(aid)
                    if prev is None or latest_curve[0] > prev[0]:
                        latest[aid] = (latest_curve[0], preset, latest_curve[1])

    history = pd.DataFrame(records)
    sessions = _window_rows(history, window)
    flags = _flags(history, sessions, window, settings, rule)
    move_on = _move_on(history, sessions, settings)
    shown = set(sessions['animal']) if len(sessions) else set()
    seen = last_seen or {}
    missing = pd.DataFrame([(a, seen.get(a), (today - seen[a]).days if seen.get(a) else None)
                            for a in sorted(set(expected) - shown)], columns=MISSING_COLUMNS)
    times = history['file_time'].dropna() if 'file_time' in history else pd.Series(dtype=object)
    return DailyTracker(
        today=today, days=days, sessions=sessions,
        conditions=pd.DataFrame(cond_rows, columns=CONDITION_COLUMNS),
        condition_sessions=pd.DataFrame(cond_session_rows, columns=CONDITION_SESSION_COLUMNS),
        flags=flags, move_on=move_on, curves=curves, latest=latest, settings=settings, rule=rule,
        notes=tuple(notes), missing=missing, newest_file=times.max() if len(times) else None)


def _window_rows(history: pd.DataFrame, window: list) -> pd.DataFrame:
    """One row per shown animal and day of the window; days without a session have only animal and date."""
    if history.empty:
        return pd.DataFrame(columns=['animal', 'date', 'folder'])
    in_window = history[history['date'].between(window[0], window[-1])]
    grid = pd.DataFrame([(a, d) for a in sorted(in_window['animal'].unique()) for d in window],
                        columns=['animal', 'date'])
    rows = grid.merge(in_window, on=['animal', 'date'], how='left')
    return rows.sort_values(['animal', 'date']).reset_index(drop=True)


def _flags(history, sessions, window, settings, rule) -> pd.DataFrame:
    cols = ['animal', 'date', 'flag', 'detail']
    if sessions.empty:
        return pd.DataFrame(columns=cols)
    out = []
    real = sessions[sessions['folder'].notna() & ~sessions['test_run'].fillna(False).astype(bool)]
    full = real[real['group'] != 'early step']

    if len(full):
        checks = session_bias_checks(full, rule, pse_column='pse')
        for (_, r), (_, c) in zip(full.iterrows(), checks.iterrows()):
            if c['flagged']:
                parts = []
                if c['pse_out']:
                    parts.append(f"PSE {r['pse']:+.2f}")
                if c['lapse_out']:
                    parts.append(f"lapses {r['lapse_low']:.2f} / {r['lapse_high']:.2f}")
                if c['accuracy_out']:
                    parts.append(f"accuracy {r['accuracy']:.2f}")
                out.append((r['animal'], r['date'], 'side bias', ', '.join(parts)))
            if r['correct'] < settings.min_correct_trials:
                out.append((r['animal'], r['date'], 'low trials',
                            f"{int(r['correct'])} correct (< {settings.min_correct_trials})"))

    hist_real = history[~history['test_run'] & (history['group'] != 'early step')]
    for _, r in full.iterrows():
        ref = hist_real[(hist_real['animal'] == r['animal']) & (hist_real['preset'] == r['preset'])
                        & (hist_real['date'] < r['date'])].sort_values('date')
        ref = ref['accuracy'].dropna().iloc[-settings.accuracy_drop_reference_sessions:]
        if len(ref) >= 2 and np.isfinite(r['accuracy']) and r['accuracy'] <= ref.median() - settings.accuracy_drop:
            out.append((r['animal'], r['date'], 'accuracy drop',
                        f"{r['accuracy']:.2f} vs median {ref.median():.2f} of {len(ref)} earlier {r['preset'] or 'sessions'}"))

    ran = real.groupby('date')['animal'].apply(set).to_dict()
    first = history[~history['test_run']].groupby('animal')['date'].min().to_dict()
    for animal in sessions['animal'].unique():
        mine = set(real.loc[real['animal'] == animal, 'date'])
        for d in window:
            others = ran.get(d, set()) - {animal}
            if d not in mine and others and animal in first and d >= first[animal]:
                out.append((animal, d, 'missing day', 'no session on ceph: not run, or not pushed'))
    return pd.DataFrame(out, columns=cols).sort_values(['animal', 'date', 'flag']).reset_index(drop=True)


def _move_on(history, sessions, settings) -> pd.DataFrame:
    cols = ['animal', 'preset', 'group', 'ready', 'detail']
    rows = []
    if history.empty:
        return pd.DataFrame(columns=cols)
    for animal in sessions['animal'].unique():
        h = history[(history['animal'] == animal) & ~history['test_run']].sort_values('date')
        if h.empty:
            continue
        last = h.iloc[-1]
        key = last['preset'] or last['stage']
        run = []
        for _, r in h.iloc[::-1].iterrows():
            if (r['preset'] or r['stage']) != key:
                break
            run.append(r)
        if last['group'] == 'early step':
            total = int(sum(r['trials'] for r in run))
            rows.append((animal, key, 'early step', total >= settings.early_step_min_trials,
                         f'{total} trials over {len(run)} session(s) (needs {settings.early_step_min_trials})'))
        elif last['group'] == 'training':
            k = settings.sessions_in_a_row
            fails = []
            for r in run[:k]:
                bad = []
                if r['correct'] < settings.move_on_min_correct_trials:
                    bad.append(f"{int(r['correct'])} correct")
                if not r['accuracy'] >= settings.min_accuracy:
                    bad.append(f"accuracy {r['accuracy']:.2f}")
                if not r['hard_accuracy'] > settings.min_hard_accuracy:
                    bad.append(f"hard {r['hard_accuracy']:.2f}")
                if not abs(r['accuracy_b_minus_a']) <= settings.max_accuracy_difference:
                    bad.append(f"right − left {r['accuracy_right_minus_left']:+.2f}")
                if bad:
                    fails.append(f"{r['date']}: " + ', '.join(bad))
            ready = len(run) >= k and not fails
            detail = (f'criteria met on the last {k} sessions' if ready else
                      '; '.join(fails) if fails else f'{len(run)} of {k} sessions on {key} so far')
            rows.append((animal, key, 'training', ready, detail))
        else:
            rows.append((animal, key, 'experiment', None, 'no criterion: experiments'))
    return pd.DataFrame(rows, columns=cols)
