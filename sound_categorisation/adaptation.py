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
from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.analysis.rolling import compute_rolling_stats
from behav_utils.analysis.statistics import compute_stat
from behav_utils.data.arrays import TrialArrays
from behav_utils.data.ops.filtering import filter_trials
from behav_utils.data.ops.selection import select_sessions
from behav_utils.data.ops.switches import block_after, block_before, find_switches
from behav_utils.stats import PSE_DYNAMICS, PSYCHOMETRIC, compute_stats
from behav_utils.stats.dynamics import fit_pse_dynamics
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from sound_categorisation.stimuli import compute_normative_pse

__all__ = ['Trajectory', 'compute_trajectory', 'expert_reference', 'TRAJECTORY_STATS',
           'HARD_WINDOW', 'HARD_STEP', 'SwitchResult', 'compute_switch', 'compute_switches', 'SWITCH_METHODS',
           'flag_biased_sessions', 'compute_phase_curves', 'animal_switches']

TRAJECTORY_STATS = ['pse', *PSYCHOMETRIC, 'accuracy', 'hard_accuracy', 'side_bias', 'recency', 'win_stay', 'lose_shift']
CRITERION = 0.8            # convergence level defining trials_to_criterion
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


# ═══════════════════════════════════════════════════════════════════════════
# Blocked switches (pre-opto cohorts): adaptation across a whole block
# ═══════════════════════════════════════════════════════════════════════════
#
# Manuscript Fig. 5C recipe (Figures/Figure-5/Panel-C notebook):
#   pre  = last 250 trials before the switch, 4-parameter fit, lapse-corrected PSE
#   post = non-overlapping 50-trial bins across the block's sessions, up to 3000 trials,
#          4-parameter fit per bin, convergence = 1 - (pse_bin - norm)/(pse_pre - norm), clipped [0, 1]
# Reproduced as method='manuscript'. method='pinned' fits only the criterion per bin
# (sigma and lapses held at the pre-switch values) and does not clip; 'pinned_running'
# is the same on running windows (50 trials, step 10). The normative PSE is the
# constant-sigma observer at the animal's pre-switch sigma (the manuscript used a
# per-subject stimulus-dependent-noise normative model whose midpoints we do not have).

SWITCH_PRE_TRIALS = 250
SWITCH_BIN = 50
SWITCH_STEP = 10
SWITCH_MAX_TRIALS = 3000
SWITCH_MIN_BLOCK = 1000
OVERNIGHT_TRIALS = 100
MIN_CONV_SPAN = 0.02       # |pse_pre - normative| below this: convergence undefined
SWITCH_METHODS = ('manuscript', 'pinned', 'pinned_running')
UNIFORM_REF_SESSIONS = 5   # the Uniform phase on the psychometrics pages: last sessions before the first switch


@dataclass(frozen=True)
class SwitchResult:
    animal: str
    switch_idx: int
    from_distribution: str
    to_distribution: str
    transition: str                   # 'first' | 'novel' | 'return'
    n_trials_before: int
    n_trials_after: int
    n_sessions_after: int
    pre: pd.Series                    # mu, sigma, lapse_low, lapse_high, pse  (last SWITCH_PRE_TRIALS trials)
    normative_pse: float
    convergence: pd.DataFrame         # method, bin, trial, pse, convergence, convergence_clipped, n
    dynamics: pd.Series               # PSE_DYNAMICS (pinned shape) on the block + trials_to_criterion, plateau
    sessions: pd.DataFrame            # per session in the block: order_in_block, session_idx, n_trials, pse, pse_fixed, accuracy, ...
    overnight: pd.DataFrame           # per consecutive session pair: pse_end, pse_start_next, delta, session_idx, next_session_idx

    def to_rows(self) -> pd.DataFrame:
        base = {'animal': self.animal, 'switch_idx': self.switch_idx, 'from_distribution': self.from_distribution,
                'to_distribution': self.to_distribution, 'transition': self.transition}
        rows = [{**base, 'stat': f'pre_{k}', 'value': float(v)} for k, v in self.pre.items()]
        rows += [{**base, 'stat': 'normative_pse', 'value': self.normative_pse}]
        rows += [{**base, 'stat': k, 'value': float(v)} for k, v in self.dynamics.items()]
        rows += [{**base, 'stat': 'n_trials_before', 'value': self.n_trials_before},
                 {**base, 'stat': 'n_trials_after', 'value': self.n_trials_after},
                 {**base, 'stat': 'n_sessions_after', 'value': self.n_sessions_after}]
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (f'SwitchResult({self.animal!r} #{self.switch_idx} {self.from_distribution}→{self.to_distribution} '
                f'[{self.transition}], n_after={self.n_trials_after}, tau={self.dynamics.get("pse_tau", np.nan):.0f})')


def _pse_from_params(mu, sigma, lo, hi) -> float:
    """Lapse-corrected PSE (the manuscript's compute_pse); NaN where undefined."""
    denom = 1.0 - lo - hi
    if not (np.isfinite(denom) and denom > 0):
        return np.nan
    arg = (0.5 - lo) / denom
    if not (0.0 < arg < 1.0):
        return np.nan
    return float(norm.ppf(arg, loc=mu, scale=sigma))


def _fit4(stim, choice) -> tuple:
    p = fit_psychometric(stim, choice)
    if not p.get('success', False):
        return (np.nan,) * 5
    return (float(p['mu']), float(p['sigma']), float(p['lapse_low']), float(p['lapse_high']),
            _pse_from_params(p['mu'], p['sigma'], p['lapse_low'], p['lapse_high']))


def _transition(previous_dists: list, to: str, first: bool) -> str:
    if first:
        return 'first'
    return 'return' if to in previous_dists else 'novel'


def compute_switch(
    animal,
    sessions: list,
    switch: dict,
    *,
    transition: str,
    max_trials: int = SWITCH_MAX_TRIALS,
    bin_size: int = SWITCH_BIN,
    step: int = SWITCH_STEP,
    pre_trials: int = SWITCH_PRE_TRIALS,
    criterion: float = CRITERION,
) -> SwitchResult:
    """Adaptation across the whole block after one switch (see module notes for the methods)."""
    aid = getattr(animal, 'animal_id', '?')
    before = filter_trials(block_before(sessions, switch), trial_type='all')
    after = filter_trials(block_after(sessions, switch), trial_type='all')
    to, frm = switch['to_distribution'], switch['from_distribution']

    # ── pre-switch reference ─────────────────────────────────────────────
    A_pre = TrialArrays.from_sessions(before).valid()
    A_pre = A_pre.take(np.arange(A_pre.n_trials)[-pre_trials:])
    mu0, sig0, lo0, hi0, pse0 = _fit4(A_pre.stimulus, A_pre.choice)
    pre = pd.Series({'mu': mu0, 'sigma': sig0, 'lapse_low': lo0, 'lapse_high': hi0, 'pse': pse0}, dtype=float)
    norm_pse = float(compute_normative_pse(to, sig0)) if np.isfinite(sig0) else np.nan

    # ── post-switch block, in order, truncated ───────────────────────────
    A = TrialArrays.from_sessions(after).valid()
    A = A.take(np.arange(min(A.n_trials, max_trials)))
    n = A.n_trials

    def conv(pse):
        # undefined when pre and normative coincide (nothing to converge to): |denominator| < MIN_CONV_SPAN
        if not (np.isfinite(pse0) and np.isfinite(norm_pse) and np.isfinite(pse)):
            return np.nan
        if abs(pse0 - norm_pse) < MIN_CONV_SPAN:
            return np.nan
        return 1.0 - (pse - norm_pse) / (pse0 - norm_pse)

    rows = []
    # manuscript: non-overlapping bins, 4-parameter fit, clipped
    for b, start in enumerate(range(0, n, bin_size), start=1):
        sl = A.take(np.arange(start, min(start + bin_size, n)))
        if sl.n_trials < bin_size // 2:
            continue
        _, _, _, _, pse_b = _fit4(sl.stimulus, sl.choice)
        c = conv(pse_b)
        # the manuscript plots bin q at q*bin_size + 1 (the trial after the bin); x = 1 is its pre-switch anchor
        rows.append({'method': 'manuscript', 'bin': b, 'trial': b * bin_size + 1, 'pse': pse_b, 'convergence': c,
                     'convergence_clipped': float(np.clip(c, 0, 1)) if np.isfinite(c) else np.nan, 'n': sl.n_trials})
        pse_p = pse_fixed(sl, sig0, lo0, hi0)
        c = conv(pse_p)
        rows.append({'method': 'pinned', 'bin': b, 'trial': start + 1, 'pse': pse_p, 'convergence': c,
                     'convergence_clipped': float(np.clip(c, 0, 1)) if np.isfinite(c) else np.nan, 'n': sl.n_trials})
    # pinned, running windows
    for b, start in enumerate(range(0, max(n - bin_size + 1, 0), step), start=1):
        sl = A.take(np.arange(start, start + bin_size))
        pse_p = pse_fixed(sl, sig0, lo0, hi0)
        c = conv(pse_p)
        rows.append({'method': 'pinned_running', 'bin': b, 'trial': start + bin_size // 2, 'pse': pse_p,
                     'convergence': c, 'convergence_clipped': float(np.clip(c, 0, 1)) if np.isfinite(c) else np.nan,
                     'n': bin_size})
    convergence = pd.DataFrame(rows, columns=['method', 'bin', 'trial', 'pse', 'convergence', 'convergence_clipped', 'n'])

    # ── block dynamics (pinned shape) ────────────────────────────────────
    dyn = dict(zip(PSE_DYNAMICS, fit_pse_dynamics(A, shape=(sig0, lo0, hi0))))
    pr = convergence[convergence['method'] == 'pinned_running']
    reached = pr.loc[pr['convergence'] >= criterion, 'trial']
    dyn['trials_to_criterion'] = float(reached.iloc[0]) if len(reached) else np.nan
    # plateau: mean convergence over the last third of the (truncated) block, running windows
    late = pr[pr['trial'] > (2 * n) // 3]['convergence']
    dyn['plateau'] = float(late.mean()) if len(late) else np.nan
    dyn['convergence_final'] = conv(dyn['pse_final']) if np.isfinite(dyn.get('pse_final', np.nan)) else np.nan
    A_full = TrialArrays.from_sessions(after).valid()
    dyn['pse_post'] = pse_fixed(A_full.take(np.arange(A_full.n_trials)[-pre_trials:]), sig0, lo0, hi0)
    dynamics = pd.Series(dyn, dtype=float)

    # ── per session within the block, and overnight pairs ────────────────
    srows, orows = [], []
    prev_end, prev_idx = np.nan, None
    for k, s in enumerate(after):
        a_s = TrialArrays.from_sessions([s]).valid()
        st = compute_stats(a_s, ['pse', 'accuracy', 'hard_accuracy', 'recency', 'win_stay', 'lose_shift', *PSYCHOMETRIC],
                           strict=False)
        srows.append({'order_in_block': k, 'session_idx': s.session_idx, 'n_trials': a_s.n_trials,
                      'pse_fixed': pse_fixed(a_s, sig0, lo0, hi0), **st.to_dict()})
        start_pse = pse_fixed(a_s.take(np.arange(min(OVERNIGHT_TRIALS, a_s.n_trials))), sig0, lo0, hi0)
        end_pse = pse_fixed(a_s.take(np.arange(a_s.n_trials)[-OVERNIGHT_TRIALS:]), sig0, lo0, hi0)
        if prev_idx is not None:
            orows.append({'session_idx': prev_idx, 'next_session_idx': s.session_idx, 'pse_end': prev_end,
                          'pse_start_next': start_pse, 'delta': start_pse - prev_end,
                          'toward_pre': (start_pse - prev_end) * np.sign(pse0 - prev_end) if np.isfinite(pse0) else np.nan})
        prev_end, prev_idx = end_pse, s.session_idx
    sessions_df = pd.DataFrame(srows)
    overnight = pd.DataFrame(orows, columns=['session_idx', 'next_session_idx', 'pse_end', 'pse_start_next', 'delta', 'toward_pre'])

    return SwitchResult(aid, int(switch['switch_idx']), frm, to, transition, int(switch['n_trials_before']),
                        int(switch['n_trials_after']), int(switch['n_sessions_after']), pre, norm_pse,
                        convergence, dynamics, sessions_df, overnight)


def animal_switches(animal, *, stage: str = 'Full_Task_Cont', exclude_types=None,
                    min_block_trials: int = SWITCH_MIN_BLOCK) -> tuple:
    """(ordered regular sessions, qualifying switches) for one animal."""
    from sound_categorisation.cohort import CONTROL_TYPES
    sessions = sorted(select_sessions(animal, stage=stage,
                                      exclude_types=CONTROL_TYPES if exclude_types is None else exclude_types),
                      key=lambda s: s.session_idx)
    return sessions, find_switches(sessions, min_block_trials=min_block_trials)


def compute_switches(animal, *, stage: str = 'Full_Task_Cont', exclude_types=None,
                     min_block_trials: int = SWITCH_MIN_BLOCK, **kw) -> list:
    """All qualifying switches of one animal, in order, with transition labels."""
    sessions, switches = animal_switches(animal, stage=stage, exclude_types=exclude_types,
                                         min_block_trials=min_block_trials)
    out, seen = [], []
    for k, sw in enumerate(switches):
        seen.append(sw['from_distribution'])
        tr = _transition(seen[:-1] + [sw['from_distribution']], sw['to_distribution'], first=(k == 0))
        out.append(compute_switch(animal, sessions, sw, transition=tr, **kw))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Cohort QC: animals that develop a side bias and keep it regardless of the distribution
# ═══════════════════════════════════════════════════════════════════════════
# The rule is about the shape of the per-session psychometric, not about whether the
# animal adapts, so applying it before looking at the A/B effect is not circular.

BIAS_PSE = 0.4          # a session is flagged when |pse_fixed| exceeds this …
BIAS_LAPSE = 0.3        # … or either lapse exceeds this …
BIAS_ACCURACY = 0.6     # … or accuracy is below this
BIAS_MAJORITY = 0.5     # an animal is biased when more than this fraction of its Hard sessions are flagged


def flag_biased_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """Add ``flagged`` (per session) and ``biased_animal`` (per animal) to a sessions table.

    ``sessions`` needs columns animal, to_distribution, pse_fixed, lapse_low, lapse_high, accuracy
    (the switch report's sessions.csv). An animal is ``biased_animal`` when the majority of its
    Hard sessions are flagged AND its flagged PSEs have the same sign on Hard-A and Hard-B blocks —
    a bias that ignores the distribution.
    """
    s = sessions.copy()
    s['flagged'] = ((s['pse_fixed'].abs() > BIAS_PSE) | (s['lapse_low'] > BIAS_LAPSE)
                    | (s['lapse_high'] > BIAS_LAPSE) | (s['accuracy'] < BIAS_ACCURACY)).fillna(False)
    biased = {}
    for aid, d in s.groupby('animal'):
        frac = d['flagged'].mean()
        f = d[d['flagged'] & d['pse_fixed'].notna()]
        signs = f.groupby('to_distribution')['pse_fixed'].apply(lambda v: np.sign(v.median()))
        same_sign = (len(signs) >= 2 and signs.nunique() == 1) or (len(signs) == 1)
        biased[aid] = bool(frac > BIAS_MAJORITY and same_sign)
    s['biased_animal'] = s['animal'].map(biased)
    return s


# ═══════════════════════════════════════════════════════════════════════════
# Psychometrics by phase: Uniform, all Hard-A, all Hard-B, and each block in order
# ═══════════════════════════════════════════════════════════════════════════

def compute_phase_curves(animal, sessions: list, switches: list) -> tuple:
    """(params, curves) tables for the pre-switch Uniform reference (last UNIFORM_REF_SESSIONS sessions), every qualifying Hard block in order,
    and the pooled Hard-A / Hard-B sessions. Fits on pooled trials (thousands per block), no bootstrap."""
    from behav_utils.readouts import compute_psychometric_curve
    aid = getattr(animal, 'animal_id', '?')
    phases = []
    if switches:
        phases.append(('Uniform', 'Uniform', 0, block_before(sessions, switches[0])[-UNIFORM_REF_SESSIONS:]))
        counts = {}
        for sw in switches:
            d = sw['to_distribution']
            counts[d] = counts.get(d, 0) + 1
            phases.append((f"{d} #{counts[d]}", d, sw['switch_idx'] + 1, block_after(sessions, sw)))
        for d in ('Hard-A', 'Hard-B'):
            pooled = [s for sw in switches if sw['to_distribution'] == d for s in block_after(sessions, sw)]
            if pooled:
                phases.append((f'{d} all', d, 90 + (d == 'Hard-B'), pooled))
    prow, crow = [], []
    for label, dist, order, sess in phases:
        clean = filter_trials(sess, trial_type='all')
        if not clean:
            continue
        A = TrialArrays.from_sessions(clean)
        c = compute_psychometric_curve(A, n_bootstrap=0)
        prow.append({'animal': aid, 'phase': label, 'distribution': dist, 'order': order, 'n_sessions': len(clean),
                     'n_trials': A.n_responded, 'mu': c.mu, 'sigma': c.sigma, 'lapse_low': c.lapse_low,
                     'lapse_high': c.lapse_high, 'pse': _pse_from_params(c.mu, c.sigma, c.lapse_low, c.lapse_high),
                     'accuracy': float(compute_stats(A, ['accuracy'])['accuracy']),
                     'hard_accuracy': float(compute_stats(A, ['hard_accuracy'])['hard_accuracy']), 'success': c.success})
        crow.append(pd.DataFrame({'animal': aid, 'phase': label, 'distribution': dist, 'order': order,
                                  'x': c.x, 'y': c.y}))
    params = pd.DataFrame(prow)
    curves = pd.concat(crow, ignore_index=True) if crow else pd.DataFrame(columns=['animal', 'phase', 'distribution', 'order', 'x', 'y'])
    return params, curves
