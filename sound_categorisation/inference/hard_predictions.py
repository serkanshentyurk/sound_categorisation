"""
Which model, fitted on Uniform sessions only, predicts how the criterion moves on the Hard sessions: across the
blocks of days of the pre-opto mice and the daily alternation of the opto mice (thesis analysis 3).

Each mouse is measured twice over the same sessions: as it is, and as copies whose choices come from a fitted model.
The sessions are the Uniform sessions its fits saw, then straight into its Hard sessions, from its first switch into
a Hard distribution to its last Hard session (``hard_span``). What lies between them is left out of both: SS08's and
SS13's short Uniform/Hard-B toggles, the opto mice's Uniform masking and laser sessions. So the first switch starts
from the last fitted Uniform session, for the mouse as for its copies.

For each parameter draw (the first ``DRAWS`` samples of the saved posterior, ``blocked_update.posterior_draws``) the
model plays every session of the span in one continuous run (``continuous_run``): one burn-in, then the mouse's own
stimuli, aborts, no-response trials and laser flags, learning from the feedback, so what it learnt on one day
carries into the next. On laser trials it learns as on any other (``laser_learning=True``) or not at all (False); a
mouse without laser trials in its span is run with learning on only. The models have no overnight change, so the
sessions run back to back. The choices go into copies of the real sessions (``with_choices``), which keep their
dates, types and distributions.

The mouse and every copy then go through the switch report's own computation (``reports.switches``), so data and
model are measured the same way. Which sessions enter is fixed to the mouse's: the Uniform reference (the shape of
the Uniform phase and the sigma behind the normative criterion) is one fit on the fitted Uniform sessions, for the
mouse and its copies alike, where the switch report uses the last five expert-Uniform sessions. Observed values can
therefore differ a little from the switch report's.

Readouts (``measure``):
    blocked design, per switch    pse_before          the previous phase's pooled 4-parameter PSE (before the first
                                                      switch: the fitted Uniform sessions)
                                  pse_block           the block's pooled PSE
                                  block_convergence   the block's share of the normative shift (empty when either
                                                      PSE lies outside ±1)
                                  hard_accuracy       accuracy on the hard stimuli of the block
    blocked design, per mouse     hard_a_minus_hard_b the mean block PSE of the Hard-A blocks minus that of the
                                                      Hard-B ones (blocks with a PSE inside ±1)
    daily design, per day         convergence_100_300 the switch report's day measure (kept for the figures)
    daily design, per mouse and kind of day ('first day', 'laser day', 'masking day')
                                  convergence_100_300 the days of one kind combined, weighted by span (the per-mouse
                                                      number of analysis 6)

``score``: for every scored readout (``SCORED``), mouse, model and laser variant, where the observed value falls
among the predictions (``percentile``: the share of predictions below it, ties counted half) and its CRPS
(``crps``, in the readout's units, lower is better). ``totals`` sums the CRPS per model over the mice, the mice the
bias check keeps and flags apart. The criterion within each switch (the pinned criterion in running 50-trial
windows; for a block every fifth window, one per 50 trials) is kept for the figures, as the median and 10-90 % range
over the draws.

    design, sessions, table = hard_span(animal, fitted)
    mouse = run_animal(animal, sessions)
    observed, trajectory = measure(mouse, design, n_uniform=len(fitted))
    predicted, bands = animal_predictions(mouse, design, len(fitted), 'BE', draws)
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from behav_utils.data import AnimalData, block_after, select_sessions, session_from_arrays
from behav_utils.models import Design, simulate

from sound_categorisation.behaviour.adaptation import (
    DAY_MIN_TRIALS,
    HARD,
    SWITCH_MIN_BLOCK,
    design_switches,
    hard_design,
    phase_shape,
)
from sound_categorisation.inference.blocked_update import _draw_seed, session_design
from sound_categorisation.models import make_model
from sound_categorisation.reports.switches import SwitchesResult, animal_switch_result, switch_tables
from sound_categorisation.settings import BASE_SEED, SBI_BURN_IN

__all__ = ['MODELS', 'DRAWS', 'BAND', 'SCORED', 'KEY', 'READOUT_COLUMNS', 'PREDICTED_COLUMNS', 'TRAJECTORY_COLUMNS',
           'SCORE_COLUMNS', 'TOTAL_COLUMNS', 'SESSION_COLUMNS', 'fitted_sessions', 'hard_span', 'run_animal',
           'with_choices', 'continuous_run', 'measure', 'animal_predictions', 'prediction_table', 'crps',
           'percentile', 'score', 'totals', 'observed_bands', 'sessions_table', 'laser_variants']

MODELS = ('BE', 'SC')
DRAWS = 200
BAND = (0.10, 0.90)              # the range of the predictions drawn and tabled (10-90 %)
TRAJECTORY_EVERY = 5             # keep every fifth running window (one per 50 trials) of the criterion curve
CURVE_METHOD = 'pinned_running'

# the readouts scored, by design and level; the others are kept for the figures
SCORED = {('blocked', 'switch'): ('pse_before', 'pse_block', 'block_convergence', 'hard_accuracy'),
          ('blocked', 'mouse'): ('hard_a_minus_hard_b',),
          ('daily', 'kind'): ('convergence_100_300',)}

KEY = ['animal', 'design', 'level', 'switch_idx', 'direction', 'kind', 'readout']
READOUT_COLUMNS = KEY + ['value']
PREDICTED_COLUMNS = ['model', 'laser_learning', 'draw'] + READOUT_COLUMNS
_TRAJ_KEY = ['animal', 'design', 'switch_idx', 'direction', 'kind', 'trial']
TRAJECTORY_COLUMNS = _TRAJ_KEY + ['source', 'laser_learning', 'n', 'median', 'lo', 'hi']
SCORE_COLUMNS = KEY + ['genotype', 'group', 'model', 'laser_learning', 'observed', 'median', 'lo', 'hi', 'n_draws',
                       'n_valid', 'percentile', 'crps']
TOTAL_COLUMNS = ['design', 'level', 'kind', 'readout', 'group', 'model', 'laser_learning', 'n', 'crps_sum',
                 'crps_mean']
SESSION_COLUMNS = ['animal', 'session_id', 'date', 'distribution', 'session_type', 'n_trials', 'n_laser', 'role']


# ── which sessions ──────────────────────────────────────────────────────────

def fitted_sessions(fit_run, animal, distribution: str = 'uniform') -> list:
    """The sessions an animal's fits saw: ``<fit run>/sessions/<distribution>.csv`` (written by every
    model-identification runner), matched to the animal's sessions by date. A run without that table falls back to
    the ``model_id_<distribution>`` preset, which is the rule the runners use, and says so.

    Raises:
        ValueError: a listed session is not among the animal's, or one date matches several sessions.
    """
    path = Path(fit_run) / 'sessions' / f'{distribution}.csv'
    aid = animal.animal_id
    if not path.exists():
        from sound_categorisation.data.cohort import ensure_presets
        ensure_presets()
        print(f'{aid}: {path} not found; the fitted sessions are reselected with the model_id_{distribution} preset')
        return list(select_sessions(animal, preset=f'model_id_{distribution}'))
    table = pd.read_csv(path)
    kept = table['kept'].astype(str).str.lower().isin(('true', '1'))
    rows = table[(table['animal_id'] == aid) & kept & table['date'].notna()]
    by_date: Dict[str, list] = {}
    for s in animal.sessions:
        by_date.setdefault(str(s.date)[:10], []).append(s)
    out = []
    for date in rows['date'].astype(str).str[:10]:
        found = by_date.get(date, [])
        if len(found) != 1:
            raise ValueError(f'{aid}: the fitted session of {date} matches {len(found)} sessions in the data')
        out.append(found[0])
    return sorted(out, key=lambda s: s.date)


def _session_row(aid: str, s, role: str) -> dict:
    t = s.trials
    return {'animal': aid, 'session_id': s.session_id, 'date': str(s.date)[:10], 'distribution': s.distribution,
            'session_type': str(getattr(s, 'session_type', '') or ''), 'n_trials': len(t.choice),
            'n_laser': int(np.asarray(t.opto_on, bool).sum()), 'role': role}


def hard_span(animal, fitted: Sequence, *, design: str | None = None, min_block_trials: int = SWITCH_MIN_BLOCK,
              min_day_trials: int = DAY_MIN_TRIALS) -> Tuple[str, list, pd.DataFrame]:
    """``(design, sessions, table)``: the fitted Uniform sessions in date order, then the animal's sessions from its
    first switch into a Hard distribution to its last Hard session, as the switch report orders them
    (``adaptation.design_switches``; ``design`` from ``hard_design`` if None). The table (``SESSION_COLUMNS``) lists
    every session of the animal from the first fitted one to the last Hard one with its ``role``: 'fitted Uniform',
    'Hard phase', or 'left out'; Hard sessions dated before the first fitted one appear as 'Hard before the fits'.

    Raises:
        ValueError: no fitted sessions, no switch into a Hard distribution, or a fitted session dated after the
            start of the Hard phase.
    """
    aid = animal.animal_id
    fitted = sorted(fitted, key=lambda s: s.date)
    if not fitted:
        raise ValueError(f'{aid}: no fitted Uniform sessions')
    design = design or hard_design(animal)
    if design == 'none':
        raise ValueError(f'{aid}: no Hard sessions')
    ordered, switches = design_switches(animal, design, min_block_trials=min_block_trials,
                                        min_day_trials=min_day_trials)
    hard = [sw for sw in switches if sw['to_distribution'] in HARD]
    if not hard:
        raise ValueError(f'{aid}: no switch into a Hard distribution')
    first = block_after(ordered, hard[0])[0]
    if fitted[-1].date >= first.date:
        raise ValueError(f'{aid}: a fitted session ({fitted[-1].date}) is not before the Hard phase ({first.date})')
    last = max(s.date for s in ordered if s.distribution in HARD)
    phase = [s for s in ordered if first.date <= s.date <= last]
    run = list(fitted) + phase
    in_run = {id(s) for s in run}
    rows = [_session_row(aid, s, 'Hard before the fits') for s in animal.sessions
            if s.date < fitted[0].date and s.distribution in HARD]
    for s in sorted(animal.sessions, key=lambda s: s.date):
        if fitted[0].date <= s.date <= last:
            role = ('fitted Uniform' if s.date <= fitted[-1].date else 'Hard phase') if id(s) in in_run else 'left out'
            rows.append(_session_row(aid, s, role))
    return design, run, pd.DataFrame(rows, columns=SESSION_COLUMNS)


def run_animal(animal, sessions: Sequence) -> AnimalData:
    """A new animal holding copies of ``sessions`` (``AnimalData`` renumbers the sessions it holds, so the animal's
    own sessions are never handed to it)."""
    return AnimalData(animal.animal_id, [replace(s) for s in sessions], metadata=dict(animal.metadata),
                      _config=getattr(animal, '_config', None))


# ── the model's sessions ────────────────────────────────────────────────────

def with_choices(session, choices) -> object:
    """A copy of ``session`` with ``choices`` in place of the animal's: scored by the session's task, the
    previous-trial fields rebuilt; its id, date, type, distribution, laser flags and aborts kept."""
    t = session.trials
    scored = session_from_arrays(t.stimulus, np.asarray(choices, dtype=float), session_idx=session.session_idx,
                                 distribution=session.distribution, abort=t.abort, task=session.task,
                                 laser=t.opto_on, session_date=session.date)
    return replace(session, trials=scored.trials)


def continuous_run(agent, sessions: Sequence, rng: np.random.Generator) -> list:
    """``agent`` plays ``sessions`` back to back in one run: one burn-in, its state carried from session to session,
    each session's stimuli, aborts, no-response trials and laser flags (``blocked_update.session_design``). Returns
    the sessions with the agent's choices (``with_choices``).

    Raises:
        ValueError: the sessions are scored by different tasks.
    """
    designs = [session_design(s) for s in sessions]
    task = designs[0].task
    if any(d.task != task for d in designs):
        raise ValueError('continuous_run: the sessions are scored by different tasks')
    joined = Design(task, np.concatenate([d.stimulus for d in designs]),
                    laser=np.concatenate([d.laser for d in designs]),
                    abort=np.concatenate([d.abort for d in designs]),
                    no_response=np.concatenate([d.no_response for d in designs]))
    choice = simulate(agent, joined, rng).trials.choice
    edges = np.cumsum([0] + [d.n_trials for d in designs])
    return [with_choices(s, choice[a:b]) for s, a, b in zip(sessions, edges[:-1], edges[1:])]


# ── the readouts ────────────────────────────────────────────────────────────

def _row(aid, design, level, switch_idx, direction, kind, readout, value) -> dict:
    v = pd.to_numeric(value, errors='coerce')
    return {'animal': aid, 'design': design, 'level': level, 'switch_idx': int(switch_idx), 'direction': direction,
            'kind': kind, 'readout': readout, 'value': float(v) if pd.notna(v) else np.nan}


def measure(animal, design: str, n_uniform: int, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``(readouts, trajectory)`` of a run animal (``run_animal``: the mouse itself or a copy) through the switch
    report's computation (``animal_switch_result``, ``switch_tables``). Its first ``n_uniform`` sessions are the
    fitted Uniform ones, whose pooled laser-off fit is the Uniform reference. ``kw`` goes to ``compute_switches``.
    Columns ``READOUT_COLUMNS``; the trajectory has ``_TRAJ_KEY`` + ``pse``."""
    aid = animal.animal_id
    shape = phase_shape(animal.sessions[:n_uniform], 'non_opto')
    design, res, params, curves, _ = animal_switch_result(animal, design, bias=False, expert_shape=shape, **kw)
    traj_cols = _TRAJ_KEY + ['pse']
    if not res:
        return pd.DataFrame(columns=READOUT_COLUMNS), pd.DataFrame(columns=traj_cols)
    T = switch_tables(SwitchesResult('', {aid: res}, params, curves, pd.DataFrame(), {aid: design}))
    by_idx = {r.switch_idx: r for r in res}
    every = TRAJECTORY_EVERY if design == 'blocked' else 1       # a day is short: every window
    rows, traj = [], []
    for r in res:
        direction = f'{r.from_distribution} → {r.to_distribution}'
        kind = r.transition if design == 'blocked' else r.switch_kind
        c = r.convergence
        c = c[(c['method'] == CURVE_METHOD) & ((c['bin'] - 1) % every == 0)]
        traj.append(pd.DataFrame({'animal': aid, 'design': design, 'switch_idx': int(r.switch_idx),
                                  'direction': direction, 'kind': kind, 'trial': c['trial'].to_numpy(int),
                                  'pse': c['pse'].to_numpy(float)}))
        if design == 'daily':
            rows.append(_row(aid, design, 'switch', r.switch_idx, direction, kind, 'convergence_100_300',
                             r.dynamics.get('convergence_100_300', np.nan)))
    if design == 'blocked' and len(T['pre_post']):
        pp = T['pre_post']
        psy = T['psychometrics']
        accuracy = dict(zip(psy['order'].astype(int), psy['hard_accuracy'])) if len(psy) else {}
        for p in pp.itertuples(index=False):
            r = by_idx[int(p.switch_idx)]
            direction = f'{r.from_distribution} → {r.to_distribution}'
            for readout, value in (('pse_before', p.pse_pre), ('pse_block', p.pse_post),
                                   ('block_convergence', p.block_convergence),
                                   ('hard_accuracy', accuracy.get(int(p.switch_idx) + 1, np.nan))):
                rows.append(_row(aid, design, 'switch', p.switch_idx, direction, r.transition, readout, value))
        block = pd.to_numeric(pp['pse_post'], errors='coerce')
        inside = pp[block.abs() <= 1]
        mean = {d: pd.to_numeric(inside.loc[inside['to_distribution'] == d, 'pse_post']).mean() for d in HARD}
        rows.append(_row(aid, design, 'mouse', -1, '', '', 'hard_a_minus_hard_b', mean['Hard-A'] - mean['Hard-B']))
    if design == 'daily':
        for k in T['animals'].itertuples(index=False):
            rows.append(_row(aid, design, 'kind', -1, '', k.switch_kind, 'convergence_100_300',
                             k.convergence_100_300))
    trajectory = pd.concat(traj, ignore_index=True) if traj else pd.DataFrame(columns=traj_cols)
    return pd.DataFrame(rows, columns=READOUT_COLUMNS), trajectory


def _bands(trajectory: pd.DataFrame, band=BAND) -> pd.DataFrame:
    """Per switch and window: the number of draws with a criterion, its median and ``band`` range."""
    if not len(trajectory):
        return pd.DataFrame(columns=_TRAJ_KEY + ['n', 'median', 'lo', 'hi'])
    g = trajectory.groupby(_TRAJ_KEY, dropna=False)['pse']
    out = g.agg(n='count', median='median').join(g.quantile(band[0]).rename('lo')).join(g.quantile(band[1]).rename('hi'))
    return out.reset_index()


def observed_bands(trajectory: pd.DataFrame) -> pd.DataFrame:
    """The mouse's own criterion trajectory in the layout of the predicted bands (``TRAJECTORY_COLUMNS``)."""
    t = trajectory.rename(columns={'pse': 'median'})
    return t.assign(source='observed', laser_learning=pd.NA, n=1, lo=np.nan, hi=np.nan)[TRAJECTORY_COLUMNS]


# ── predictions ─────────────────────────────────────────────────────────────

def animal_predictions(animal, design: str, n_uniform: int, model: str, draws: Sequence[Dict[str, float]], *,
                       laser_learning: bool = True, seed: int = BASE_SEED, burn_in: int = SBI_BURN_IN,
                       band=BAND, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``(predicted, bands)``: ``model`` at each parameter draw, run continuously through the run animal ``animal``
    (``continuous_run``) and measured like the mouse (``measure``; ``kw`` goes to ``compute_switches``). One random
    stream per animal and draw, the same for both models and both laser variants, so they differ only in the model.
    A draw whose measurement fails is left out with a warning. ``predicted`` has ``PREDICTED_COLUMNS``; ``bands`` the
    criterion trajectory over the draws (``TRAJECTORY_COLUMNS``)."""
    aid, model = animal.animal_id, str(model).upper()
    tag = {'model': model, 'laser_learning': bool(laser_learning)}
    rows, traj = [], []
    for d, params in enumerate(draws):
        agent = make_model(model, params, burn_in=burn_in, laser_blocks_update=not laser_learning)
        copy = run_animal(animal, continuous_run(agent, animal.sessions, np.random.default_rng(_draw_seed(seed, aid, d))))
        try:
            r, t = measure(copy, design, n_uniform, **kw)
        except Exception as exc:                          # one degenerate copy must not end the animal
            warnings.warn(f'{aid} {model} draw {d}: {exc!r}', stacklevel=2)
            continue
        rows.append(r.assign(draw=d, **tag))
        traj.append(t)
    predicted = (pd.concat(rows, ignore_index=True)[PREDICTED_COLUMNS] if rows
                 else pd.DataFrame(columns=PREDICTED_COLUMNS))
    bands = _bands(pd.concat(traj, ignore_index=True) if traj else pd.DataFrame(), band)
    return predicted, bands.assign(source=model, **{'laser_learning': bool(laser_learning)})[TRAJECTORY_COLUMNS]


def _one(job: Tuple, kw: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    animal, design, n_uniform, model, laser_learning, draws = job
    return animal_predictions(animal, design, n_uniform, model, draws, laser_learning=laser_learning, **kw)


def prediction_table(jobs: Sequence[Tuple], n_jobs: int = 1, **kw) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``animal_predictions`` for every ``(run animal, design, n_uniform, model, laser_learning, draws)`` in
    ``jobs``, ``n_jobs`` at once (the numbers do not depend on it). ``kw``: seed, burn_in, band, and what goes to
    ``compute_switches``. Returns ``(predicted, bands)``."""
    if n_jobs > 1 and len(jobs) > 1:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_one)(job, kw) for job in jobs)
    else:
        parts = []
        for job in jobs:
            parts.append(_one(job, kw))
            print(f'  {job[0].animal_id} {job[3]} laser learning {"on" if job[4] else "off"}: {len(job[5])} draws')
    pred = [p for p, _ in parts if len(p)]
    bands = [b for _, b in parts if len(b)]
    return (pd.concat(pred, ignore_index=True) if pred else pd.DataFrame(columns=PREDICTED_COLUMNS),
            pd.concat(bands, ignore_index=True) if bands else pd.DataFrame(columns=TRAJECTORY_COLUMNS))


# ── scores ──────────────────────────────────────────────────────────────────

def crps(samples, observed: float) -> float:
    """The continuous ranked probability score of ``observed`` under the empirical distribution of ``samples``:
    mean |X - y| - mean |X - X'| / 2, in the units of the values (lower is better; a point prediction scores its
    absolute error). NaN without finite samples or observation."""
    x = np.sort(np.asarray(samples, dtype=float))
    x = x[np.isfinite(x)]
    if not x.size or not np.isfinite(observed):
        return np.nan
    n = x.size
    spread = 2.0 * np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2       # mean |X - X'| over all pairs
    return float(np.mean(np.abs(x - observed)) - 0.5 * spread)


def percentile(samples, observed: float) -> float:
    """Where ``observed`` falls among ``samples``: the share below it, ties counted half (0.5 = the median)."""
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size or not np.isfinite(observed):
        return np.nan
    return float((np.sum(x < observed) + 0.5 * np.sum(x == observed)) / x.size)


def _scored(df: pd.DataFrame) -> pd.Series:
    return pd.Series([r in SCORED.get((d, lv), ()) for d, lv, r in zip(df['design'], df['level'], df['readout'])],
                     index=df.index, dtype=bool)


def score(predicted: pd.DataFrame, observed: pd.DataFrame, groups: Dict[str, str] | None = None,
          genotypes: Dict[str, str] | None = None, band=BAND) -> pd.DataFrame:
    """One row per scored readout (``SCORED``), mouse, model and laser variant: the observed value, the median and
    ``band`` range of the predictions, how many draws gave a value, the percentile and the CRPS. ``groups``: animal
    → 'kept' | 'flagged' (the bias check); ``genotypes``: animal → genotype. Columns ``SCORE_COLUMNS``."""
    if not len(predicted):
        return pd.DataFrame(columns=SCORE_COLUMNS)
    pred = predicted[_scored(predicted)]
    obs = {tuple(k): v for k, v in zip(observed[KEY].itertuples(index=False, name=None), observed['value'])}
    rows = []
    for key, g in pred.groupby(KEY + ['model', 'laser_learning'], dropna=False, sort=False):
        k = dict(zip(KEY + ['model', 'laser_learning'], key))
        y = float(obs.get(tuple(k[c] for c in KEY), np.nan))
        x = g['value'].to_numpy(float)
        v = x[np.isfinite(x)]
        rows.append({**k, 'genotype': (genotypes or {}).get(k['animal'], 'unknown'),
                     'group': (groups or {}).get(k['animal'], 'kept'), 'observed': y,
                     'median': float(np.median(v)) if v.size else np.nan,
                     'lo': float(np.quantile(v, band[0])) if v.size else np.nan,
                     'hi': float(np.quantile(v, band[1])) if v.size else np.nan,
                     'n_draws': int(x.size), 'n_valid': int(v.size), 'percentile': percentile(v, y), 'crps': crps(v, y)})
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def totals(scores: pd.DataFrame) -> pd.DataFrame:
    """The CRPS summed per model and laser variant over the mice (and, for blocks, their switches), per readout,
    the mice the bias check keeps and flags apart; daily readouts per kind of day. ``n`` counts the scores that
    entered. Columns ``TOTAL_COLUMNS``."""
    if not len(scores):
        return pd.DataFrame(columns=TOTAL_COLUMNS)
    s = scores.assign(kind=np.where(scores['design'] == 'daily', scores['kind'], ''))
    keys = ['design', 'level', 'kind', 'readout', 'group', 'model', 'laser_learning']
    g = s.groupby(keys, dropna=False)['crps']
    out = pd.concat([g.count().rename('n'), g.sum(min_count=1).rename('crps_sum'), g.mean().rename('crps_mean')],
                    axis=1).reset_index()
    return out[TOTAL_COLUMNS]


def sessions_table(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """The ``hard_span`` tables of the animals run, one after the other."""
    tables = [t for t in tables if len(t)]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=SESSION_COLUMNS)


def laser_variants(sessions: Sequence, requested: Sequence[bool] = (True, False)) -> List[bool]:
    """The laser variants to run for a span: the requested ones when it has laser trials, learning on only
    otherwise (the two would be the same)."""
    has_laser = any(np.asarray(s.trials.opto_on, bool).any() for s in sessions)
    variants = [bool(v) for v in requested]
    return variants if has_laser else [True]
