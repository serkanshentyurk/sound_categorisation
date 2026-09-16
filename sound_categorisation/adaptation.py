"""
Session trajectory and per-session adaptation, for one animal.

The Hard phase alternates distributions daily (A B A B …, six laser sessions
then six masking sessions), so the unit of adaptation is the session: every
session is one switch from the previous day's distribution. Uniform is blocked
(a masking set and a laser set). Both are described by the same object:

    tr = compute_trajectory(animal, distributions=('Hard-A', 'Hard-B'))
    tr.sessions      one row per session in acquisition order:
                     order, session_id, session_idx, date, distribution, session_type,
                     n_trials, <pooled stats>, pse_fixed (mu with sigma/lapses pinned to the
                     Uniform fit), <pse_dynamics> and the same with `_fixed` (shape pinned),
                     prev_distribution, prev_pse,
                     baseline_pse, normative_pse, convergence_final, delta_from_prev
    tr.curves        rolling PSE within each session (trial = trial within session),
                     with the per-session convergence index

Convergence per session uses the previous session's PSE as 0 and the normative
PSE for the new distribution as 1 (manuscript Fig. 5C); the first session of
the phase uses the animal's expert-Uniform PSE. The normative PSE is computed at
the animal's own psychometric sigma from its expert-Uniform fit (no model fit
needed); pass ``sigma`` to override, e.g. with an SBI posterior later.

``delta_from_prev`` = PSE(this session) − PSE(previous session) is the A–B
amplitude of the alternation, session by session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from behav_utils.analysis.rolling import compute_rolling_stats
from behav_utils.analysis.statistics import compute_stat
from behav_utils.data.arrays import TrialArrays
from behav_utils.data.ops.filtering import filter_trials
from behav_utils.data.ops.selection import select_sessions
from behav_utils.stats import PSE_DYNAMICS, PSYCHOMETRIC, compute_stats
from behav_utils.stats.dynamics import fit_pse_dynamics
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from sound_categorisation.stimuli import compute_normative_pse

__all__ = ['Trajectory', 'compute_trajectory', 'expert_reference', 'TRAJECTORY_STATS',
           'HARD_WINDOW', 'HARD_STEP']

TRAJECTORY_STATS = ['pse', *PSYCHOMETRIC, 'accuracy', 'hard_accuracy', 'side_bias', 'recency', 'win_stay', 'lose_shift']
HARD_WINDOW, HARD_STEP = 100, 20      # rolling PSE on Hard sessions: fewer informative stimuli, wider window
UNIFORM_WINDOW, UNIFORM_STEP = 50, 10
CURVE_COLUMNS = ['order', 'session_id', 'session_idx', 'distribution', 'session_type', 'trial', 'pse', 'convergence']


@dataclass(frozen=True)
class Trajectory:
    animal: str
    distributions: tuple
    sessions: pd.DataFrame
    curves: pd.DataFrame
    baseline_pse: float          # expert-Uniform PSE
    sigma: float                 # psychometric sigma used for the normative PSE

    def __repr__(self) -> str:
        return (f'Trajectory({self.animal!r}, {list(self.distributions)}, n_sessions={len(self.sessions)}, '
                f'sigma={self.sigma:.3f}, baseline_pse={self.baseline_pse:+.3f})')


def expert_reference(animal, *, preset: str = 'expert_uniform', last_n: int = 5,
                     trials: str = 'non_opto') -> tuple:
    """(PSE, sigma, lapse_low, lapse_high) pooled over the animal's last ``last_n`` expert-Uniform sessions."""
    from sound_categorisation.cohort import ensure_presets
    ensure_presets()
    sessions = select_sessions(animal, preset=preset)
    if last_n:
        sessions = sessions[-last_n:]
    phase = filter_trials(sessions, trial_type=trials)
    if not phase:
        return np.nan, np.nan, np.nan, np.nan
    r = compute_stat(phase, ['pse', 'sigma', 'lapse_low', 'lapse_high']).pooled
    return float(r['pse']), float(r['sigma']), float(r['lapse_low']), float(r['lapse_high'])


def pse_fixed(arrays: TrialArrays, sigma: float, lapse_low: float, lapse_high: float) -> float:
    """PSE with sigma and lapses pinned to the animal's Uniform values: a 1-parameter MLE for mu.

    Per-session PSE with all four psychometric parameters free has an SE of
    ~0.1 at mouse-like sigma; pinning the shape leaves only the criterion to
    move, which is the quantity the daily switch is supposed to change.
    """
    v = arrays.valid()
    if v.n_trials < 50 or not (np.isfinite(sigma) and sigma > 0):
        return np.nan
    s, c = v.stimulus, v.choice
    lo, hi = np.clip(lapse_low, 0, 0.45), np.clip(lapse_high, 0, 0.45)

    def nll(mu):
        p = np.clip(lo + (1 - lo - hi) * norm.cdf((s - mu) / sigma), 1e-10, 1 - 1e-10)
        return -np.sum(c * np.log(p) + (1 - c) * np.log(1 - p))
    r = minimize_scalar(nll, bounds=(-1.0, 1.0), method='bounded')
    return float(r.x) if r.success else np.nan


def _distribution(session) -> str | None:
    try:
        return session.distribution
    except Exception:
        return None


def compute_trajectory(
    animal,
    distributions: Sequence[str] = ('Hard-A', 'Hard-B'),
    *,
    stats: Sequence[str] = TRAJECTORY_STATS,
    trials: str = 'all',
    sigma: float | None = None,
    dynamics: bool = True,
    window: int | None = None,
    step: int | None = None,
) -> Trajectory:
    """Per-session statistics and adaptation across all sessions of ``distributions``, in order.

    Args:
        animal:        AnimalData.
        distributions: which distributions' sessions to include (all session types).
        stats:         scalar stats computed per session on ``trials``.
        trials:        'all' (default) or 'non_opto'.
        sigma:         psychometric sigma for the normative PSE; from the expert-Uniform fit if None.
        dynamics:      also fit ``pse_dynamics`` per session (exponential vs step).
        window, step:  rolling-PSE window; defaults depend on whether the phase is Uniform.
    """
    aid = getattr(animal, 'animal_id', '?')
    base_pse, base_sigma, base_lo, base_hi = expert_reference(animal)
    sigma = float(sigma) if sigma is not None else base_sigma

    sessions = []
    for d in distributions:
        sessions += select_sessions(animal, distribution=d)
    sessions = sorted({s.session_idx: s for s in sessions}.values(), key=lambda s: s.session_idx)
    is_uniform = all(d.lower() == 'uniform' for d in distributions)
    window = window or (UNIFORM_WINDOW if is_uniform else HARD_WINDOW)
    step = step or (UNIFORM_STEP if is_uniform else HARD_STEP)

    rows, curves = [], []
    prev_pse, prev_dist = base_pse, 'Uniform'
    for order, s in enumerate(sessions):
        phase = filter_trials([s], trial_type=trials)
        dist = _distribution(s)
        norm = float(compute_normative_pse(dist, sigma)) if (dist and np.isfinite(sigma)) else np.nan
        scale = norm - prev_pse if np.isfinite(norm) and np.isfinite(prev_pse) else np.nan
        row = {'order': order, 'session_id': s.session_id, 'session_idx': s.session_idx,
               'date': getattr(s, 'date', None), 'distribution': dist, 'session_type': s.session_type,
               'n_trials': 0, 'prev_distribution': prev_dist, 'prev_pse': prev_pse,
               'baseline_pse': prev_pse, 'normative_pse': norm}
        if phase:
            arrays = TrialArrays.from_sessions(phase)
            row['n_trials'] = arrays.n_responded
            row.update(compute_stats(arrays, list(stats), strict=False).to_dict())
            row['pse_fixed'] = pse_fixed(arrays, base_sigma, base_lo, base_hi)
            if dynamics:
                row.update(compute_stats(arrays, list(PSE_DYNAMICS), strict=False).to_dict())
                fixed = fit_pse_dynamics(arrays, shape=(base_sigma, base_lo, base_hi))
                row.update({f'{k}_fixed': v for k, v in zip(PSE_DYNAMICS, fixed)})
            rolled = compute_rolling_stats(phase, 'pse', per_session=True, window=window, step=step)
            c = rolled.curve('pse', s.session_id)
            pse_curve = c['value'].to_numpy()
            curves.append(pd.DataFrame({
                'order': order, 'session_id': s.session_id, 'session_idx': s.session_idx,
                'distribution': dist, 'session_type': s.session_type,
                'trial': c['trial'].to_numpy(), 'pse': pse_curve,
                'convergence': (pse_curve - prev_pse) / scale if (np.isfinite(scale) and scale != 0) else np.nan,
            }, columns=CURVE_COLUMNS))
        pse_now = row.get('pse', np.nan)
        end = row.get('pse_final', np.nan)
        end = end if np.isfinite(end) else pse_now
        row['convergence_final'] = (end - prev_pse) / scale if (np.isfinite(scale) and scale != 0 and np.isfinite(end)) else np.nan
        row['delta_from_prev'] = pse_now - prev_pse if (np.isfinite(pse_now) and np.isfinite(prev_pse)) else np.nan
        rows.append(row)
        if np.isfinite(pse_now):
            prev_pse, prev_dist = pse_now, dist

    sess_df = pd.DataFrame(rows)
    curves_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=CURVE_COLUMNS)
    return Trajectory(aid, tuple(distributions), sess_df, curves_df, base_pse, sigma)
