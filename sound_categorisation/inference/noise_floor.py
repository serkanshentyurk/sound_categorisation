"""
The noise floor of the held-out update-matrix error, and a no-history reference, per animal (thesis analysis 1).

A fit is scored on held-out data: its update matrix, from one simulation on the held-out trials, against the
held-out update matrix (``selection``, ``grid_search``). The floor is the same error between the two halves of the
split the fit is scored on. Both halves are the same animal, so it is the noise of a few hundred trials plus the
drift between days. A fit's matrix comes from one simulation, so it is as noisy as the held-out one, and a correct
model scores about the floor itself: on 44 simulated mice (22 BE, 22 SC; five sessions of 250-500 trials) the true
model, scored this way, averaged 1.03 times the floor (0.76-1.35 for 80 % of them). A model's error over the floor
reads: about 1, as good as a model with fixed parameters can be on these data; well above 1, it misses something.

The no-history reference is the animal's psychometric (4 parameters) fitted on one half and simulated on the other
with no dependence on the previous trial, scored like a fit. Its curve is fitted on choices made while the animal
updated, so it carries the average result of that updating; what it lacks is what the update matrix measures (each
conditional curve minus the main curve), so its expected matrix is zero. A model's error sits between the floor and it.

The splits are the ones the fits are scored on:
    session  the two-fold session split (``selection.fold_sessions``: the first three of five sessions against the
             last two), the pooled network's and the grid search's. Their error is the mean over the two folds;
             so is a score here.
    within   each session's first half of trials against its second half (``selection.session_halves``), the
             single-session network's; per session, then the mean over sessions, as its errors are averaged.

On a synthetic cohort the true model, with its true parameters, is scored the same way (``true_model``): the check
that a correct model scores about the floor.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from behav_utils.data import pool_arrays
from behav_utils.readouts import cumulative_gaussian, fit_psychometric, matrix_error

from sound_categorisation.inference.selection import (
    empirical_target,
    fold_sessions,
    session_halves,
    simulated_target,
    target_from_choices,
)
from sound_categorisation.settings import SBI_BURN_IN

__all__ = ['animal_noise_floor', 'noise_floor_table', 'model_errors', 'floor_ratios', 'true_model_band', 'REPEATS',
           'SPLITS', 'FLOOR_COLUMNS', 'SESSION_COLUMNS']

REPEATS = 16                 # simulations averaged per held-out half for no_history and true_model
SPLITS = ('session', 'within')
FLOOR_COLUMNS = ['animal', 'split', 'floor', 'no_history', 'true_model', 'true_model_name', 'n_parts',
                 'n_trials_a', 'n_trials_b', 'sessions_a', 'sessions_b']
SESSION_COLUMNS = ['animal', 'rep', 'session_idx', 'n_trials_a', 'n_trials_b', 'floor', 'no_history', 'true_model']


def _part(sessions: List, fit_target: str, n_bins: int):
    pooled = pool_arrays(sessions)
    return pooled, empirical_target(pooled, fit_target, n_bins)


def _n_responded(pooled) -> int:
    return int(np.sum(~np.asarray(pooled['no_response'], bool)))


def _mean(values) -> float:
    """Mean of the finite values; NaN (without a warning) when there are none, e.g. halves too short for a matrix."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else np.nan


def _no_history_prob(train, stimuli: np.ndarray) -> np.ndarray | None:
    """P(B) on ``stimuli`` from the psychometric of ``train``'s responded trials alone: no trial history."""
    ok = ~np.asarray(train['no_response'], bool)
    p = fit_psychometric(np.asarray(train['stimuli'])[ok], np.asarray(train['choices'])[ok])
    if not p.get('success', False):
        return None
    return cumulative_gaussian(np.asarray(stimuli, float), p['mu'], p['sigma'], p['lapse_low'], p['lapse_high'])


def _scores(a, b, *, fit_target: str, n_bins: int, repeats: int, seed: int, truth) -> Dict[str, float]:
    """``a``, ``b`` = (pooled, matrix) of the two halves. The floor, and the mean over both directions (fit on one
    half, scored on the other) of the no-history reference and, with ``truth`` = (model, params), the true model,
    each the mean error of ``repeats`` single simulations."""
    out = {'floor': float(matrix_error(a[1], b[1])), 'no_history': np.nan, 'true_model': np.nan}
    nl, tm = [], []
    for k, ((train, _), (test, held_out)) in enumerate(((a, b), (b, a))):
        rng = np.random.default_rng(seed + k)
        prob = _no_history_prob(train, test['stimuli'])
        errs = [] if prob is None else [
            matrix_error(held_out, target_from_choices((rng.random(len(prob)) < prob).astype(float), test,
                                                       fit_target, n_bins))
            for _ in range(repeats)]
        nl.append(_mean(errs))
        if truth is not None:
            model, params = truth
            tm.append(_mean([matrix_error(held_out, simulated_target(model, params, test, fit_target, n_bins,
                                                                     SBI_BURN_IN, seed + 1000 * k + r))
                             for r in range(repeats)]))
    out['no_history'] = float(np.mean(nl))          # NaN if either direction has none, as a fit's fold mean
    if truth is not None:
        out['true_model'] = float(np.mean(tm))
    return out


def animal_noise_floor(sessions: Sequence, *, animal: str = '', truth=None, splits: Sequence[str] = SPLITS,
                       fit_target: str = 'update_matrix', n_bins: int = 8, repeats: int = REPEATS,
                       seed: int = 0) -> tuple:
    """One animal's floor and references on ``splits`` of its ``sessions`` (the fits' sessions, in order).
    ``truth`` = (model, params) adds the true model's score (synthetic animals).

    Returns ``(rows, session_rows)``: one row per split (``FLOOR_COLUMNS``; the within split is the mean over the
    sessions whose halves give a floor, ``n_parts`` of them) and one per session for the within split (``SESSION_COLUMNS``, ``rep`` = the session's position, as
    the single-session conditioning numbers its reps)."""
    sessions = list(sessions)
    name = '' if truth is None else str(getattr(truth[0], 'value', truth[0])).upper()
    rows, session_rows = [], []
    kw = dict(fit_target=fit_target, n_bins=n_bins, repeats=repeats, truth=truth)
    if 'session' in splits and len(sessions) >= 2:
        halves = [test for _, test in fold_sessions(sessions, 2)]
        a, b = (_part(h, fit_target, n_bins) for h in halves)
        rows.append({'animal': animal, 'split': 'session', **_scores(a, b, seed=seed, **kw),
                     'true_model_name': name, 'n_parts': 2, 'n_trials_a': _n_responded(a[0]),
                     'n_trials_b': _n_responded(b[0]),
                     'sessions_a': ','.join(str(s.session_idx) for s in halves[0]),
                     'sessions_b': ','.join(str(s.session_idx) for s in halves[1])})
    if 'within' in splits:
        for si, s in enumerate(sessions):
            halves = session_halves(s, si)
            if halves is None:
                continue
            a, b = (_part([h], fit_target, n_bins) for h in halves)
            session_rows.append({'animal': animal, 'rep': si, 'session_idx': s.session_idx,
                                 'n_trials_a': _n_responded(a[0]), 'n_trials_b': _n_responded(b[0]),
                                 **_scores(a, b, seed=seed + 100 * (si + 1), **kw)})
        d = pd.DataFrame(session_rows, columns=SESSION_COLUMNS)
        d = d[d['floor'].notna()]               # every mean over the same sessions: those with a floor
        if len(d):
            rows.append({'animal': animal, 'split': 'within',
                         **{c: float(d[c].mean()) for c in ('floor', 'no_history', 'true_model')},
                         'true_model_name': name, 'n_parts': len(d), 'n_trials_a': int(d['n_trials_a'].sum()),
                         'n_trials_b': int(d['n_trials_b'].sum()), 'sessions_a': '', 'sessions_b': ''})
    return rows, session_rows


def _one(record, kw) -> tuple:
    truth = None
    if getattr(record, 'true_model', None) in ('BE', 'SC') and getattr(record, 'true_params', None):
        truth = (record.true_model.lower(), dict(record.true_params))
    return animal_noise_floor(record.sessions, animal=record.animal_id, truth=truth, **kw)


def noise_floor_table(records: Sequence, *, n_jobs: int = 1, **kw) -> tuple:
    """``(table, sessions)`` for the ``cohort.select_animals`` records: ``animal_noise_floor`` per animal, the
    true model scored where a record carries one. ``kw`` goes to ``animal_noise_floor``; ``n_jobs`` animals at
    once (joblib)."""
    if n_jobs and n_jobs != 1:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs)(delayed(_one)(r, kw) for r in records)
    else:
        parts = [_one(r, kw) for r in records]
    rows = [r for p in parts for r in p[0]]
    srows = [r for p in parts for r in p[1]]
    return pd.DataFrame(rows, columns=FLOOR_COLUMNS), pd.DataFrame(srows, columns=SESSION_COLUMNS)


def model_errors(comparison: pd.DataFrame) -> pd.DataFrame:
    """Columns animal, model, error from a ``cv_utils.load_cv_results(...).comparison``: each model's mean
    held-out error over the reps."""
    if comparison is None or not len(comparison):
        return pd.DataFrame(columns=['animal', 'model', 'error'])
    long = comparison.melt(id_vars=['animal_id'], value_vars=['be_mean', 'sc_mean'], var_name='model', value_name='error')
    long['model'] = long['model'].map({'be_mean': 'BE', 'sc_mean': 'SC'})
    return long.rename(columns={'animal_id': 'animal'})[['animal', 'model', 'error']]


def floor_ratios(table: pd.DataFrame, errors: pd.DataFrame | None = None, split: str = 'session') -> pd.DataFrame:
    """Per animal on ``split``: the floor, and the no-history reference, the true model (where there is one)
    and each model in ``errors`` (``model_errors``) divided by it. 1 = the floor."""
    t = table[table['split'] == split].set_index('animal')
    out = pd.DataFrame({'floor': t['floor'], 'no_history': t['no_history'] / t['floor']})
    if t['true_model'].notna().any():
        out['true_model'] = t['true_model'] / t['floor']
    if errors is not None and len(errors):
        wide = errors.pivot_table(index='animal', columns='model', values='error')
        for m in wide.columns:
            out[m] = wide[m].reindex(out.index) / out['floor']
    return out.reset_index()


def true_model_band(table: pd.DataFrame, split: str = 'session', q=(0.1, 0.9)) -> tuple | None:
    """The ``q`` quantiles of the true model's error over the floor on ``split`` (a synthetic cohort's table): where
    a correct model's error falls. None without true-model scores."""
    r = floor_ratios(table, split=split)
    if 'true_model' not in r or not r['true_model'].notna().any():
        return None
    lo, hi = r['true_model'].quantile(list(q))
    return float(lo), float(hi)
