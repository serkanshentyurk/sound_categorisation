"""
Adaptation after a distribution switch, for one animal.

Convergence index (manuscript Fig. 5C, with the expert-Uniform PSE as 0):

    convergence(t) = (PSE(t) − PSE_uniform) / (PSE_normative − PSE_uniform)

0 = the animal's own expert-Uniform PSE, 1 = the normative PSE for the new
distribution (``stimuli.compute_normative_pse`` at the animal's sigma). ``t`` is
trials since the switch, cumulative across the sessions of the block, so a
block of several sessions is one curve on a log-trial axis.

Two descriptions of the same block:

- rolling:   50-trial windows of the lapse-corrected PSE (windows never cross a
             session boundary; the abscissa does), via ``compute_rolling_stats``.
- dynamics:  the ``pse_dynamics`` fit on the whole block (exponential vs step),
             giving tau, endpoint and shape. Order-dependent: session-unit CIs only.

``compute_switch_adaptation`` returns both plus the two anchors. Session type
filtering (opto vs masking) is the caller's choice via ``session_type``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from behav_utils.analysis.rolling import compute_rolling_stats
from behav_utils.analysis.statistics import compute_stat
from behav_utils.data.arrays import TrialArrays
from behav_utils.data.ops.filtering import filter_trials
from behav_utils.data.ops.selection import select_sessions
from behav_utils.stats import PSE_DYNAMICS, compute_stats

from sound_categorisation.stimuli import compute_normative_pse

__all__ = ['SwitchAdaptation', 'compute_switch_adaptation', 'baseline_pse', 'DEFAULT_SIGMA',
           'CRITERION']

DEFAULT_SIGMA = 0.175      # provisional sigma for the normative PSE until the SBI posterior replaces it
CRITERION = 0.8            # convergence level defining trials_to_criterion


@dataclass(frozen=True)
class SwitchAdaptation:
    animal: str
    distribution: str
    session_type: Optional[str]
    baseline_pse: float                 # expert-Uniform PSE (0 on the convergence axis)
    normative_pse: float                # normative PSE for this distribution (1 on the axis)
    curve: pd.DataFrame                 # session, session_idx, trial (since switch), pse, convergence
    dynamics: pd.Series                 # PSE_DYNAMICS + convergence_final, trials_to_criterion
    n_sessions: int
    n_trials: int

    @property
    def scale(self) -> float:
        return self.normative_pse - self.baseline_pse

    def to_rows(self) -> pd.DataFrame:
        """Tidy: one row per dynamics stat."""
        return pd.DataFrame({'animal': self.animal, 'distribution': self.distribution,
                             'session_type': self.session_type or 'all',
                             'stat': self.dynamics.index, 'value': self.dynamics.to_numpy()})

    def __repr__(self) -> str:
        return (f'SwitchAdaptation({self.animal!r}, {self.distribution!r}, type={self.session_type!r}, '
                f'n_sessions={self.n_sessions}, n_trials={self.n_trials}, '
                f'tau={self.dynamics.get("pse_tau", np.nan):.0f})')


def baseline_pse(animal, *, preset: str = 'expert_uniform', last_n: int = 5,
                 trials: str = 'non_opto') -> float:
    """The animal's expert-Uniform PSE: pooled 'pse' over the last ``last_n`` preset sessions."""
    sessions = select_sessions(animal, preset=preset)
    if last_n:
        sessions = sessions[-last_n:]
    phase = filter_trials(sessions, trial_type=trials)
    if not phase:
        return np.nan
    return float(compute_stat(phase, ['pse']).pooled['pse'])


def compute_switch_adaptation(
    animal,
    distribution: str,
    *,
    session_type: Optional[str] = None,
    trials: str = 'all',
    sigma: float = DEFAULT_SIGMA,
    window: int = 50,
    step: int = 10,
    baseline: Optional[float] = None,
    criterion: float = CRITERION,
) -> SwitchAdaptation:
    """Rolling + parametric adaptation for one animal entering ``distribution``.

    Args:
        animal:       AnimalData.
        distribution: the post-switch distribution ('Hard-A' / 'Hard-B').
        session_type: restrict the block to one session type ('opto', 'masking'); None = all.
        trials:       'all' (default — the cumulative effect of laser-on trials is the signal)
                      or 'non_opto'.
        sigma:        perceptual sigma for the normative PSE (provisional constant by default).
        window, step: rolling-window size and stride (trials).
        baseline:     expert-Uniform PSE; computed from the animal if None.
        criterion:    convergence level for ``trials_to_criterion``.
    """
    aid = getattr(animal, 'animal_id', '?')
    base = baseline_pse(animal) if baseline is None else float(baseline)
    norm_pse = float(compute_normative_pse(distribution, sigma))
    scale = norm_pse - base

    kw = {'distribution': distribution}
    if session_type is not None:
        kw['session_type'] = session_type
    block = sorted(select_sessions(animal, exclude_masking=False, **kw), key=lambda s: s.session_idx)
    block = filter_trials(block, trial_type=trials)

    empty = pd.DataFrame(columns=['session', 'session_idx', 'trial', 'pse', 'convergence'])
    names = list(PSE_DYNAMICS) + ['convergence_final', 'trials_to_criterion']
    if not block:
        return SwitchAdaptation(aid, distribution, session_type, base, norm_pse, empty,
                                pd.Series(np.nan, index=names, dtype=float), 0, 0)

    # rolling PSE, per session, then placed on the trials-since-switch axis
    rolled = compute_rolling_stats(block, 'pse', per_session=True, window=window, step=step)
    offset, rows = 0, []
    for s in block:
        sid = getattr(s, 'session_id', '?')
        c = rolled.curve('pse', sid)
        n = rolled.n_trials(sid)
        rows.append(pd.DataFrame({'session': sid, 'session_idx': getattr(s, 'session_idx', None),
                                  'trial': c['trial'].to_numpy() + offset, 'pse': c['value'].to_numpy()}))
        offset += n
    curve = pd.concat(rows, ignore_index=True) if rows else empty
    curve['convergence'] = (curve['pse'] - base) / scale if scale != 0 else np.nan

    # parametric fit on the whole block
    arrays = TrialArrays.from_sessions(block)
    dyn = compute_stats(arrays, list(PSE_DYNAMICS), strict=False)
    dyn['convergence_final'] = (dyn['pse_final'] - base) / scale if scale != 0 else np.nan
    reached = curve.loc[curve['convergence'] >= criterion, 'trial']
    dyn['trials_to_criterion'] = float(reached.iloc[0]) if len(reached) else np.nan

    return SwitchAdaptation(aid, distribution, session_type, base, norm_pse, curve, dyn,
                            len(block), int(arrays.n_responded))
