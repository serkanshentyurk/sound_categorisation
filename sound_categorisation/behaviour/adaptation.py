"""
Adaptation to the stimulus distribution, for one animal: the session trajectory of a phase
(``compute_trajectory``, the opto battery), the switches between blocks or days (``compute_switches``,
the switch report), and the cohort QC (``BiasRule``, ``bias_sessions``).

How fast the criterion moves is measured one way everywhere (manuscript Fig. 5C, made steadier):

    criterion    the 50 % point of the psychometric (the lapse-corrected PSE) with sigma and lapses fixed,
                 so only the criterion is fitted (``pinned_pse``). The shape is the phase's own: one
                 4-parameter fit on the pooled trials of the phase (``phase_shape``), where a phase is a
                 block of days, or in the daily alternation every day of one distribution and session type
                 (``phase_shapes``); the Uniform phase takes the expert-Uniform fit. Slope and lapses do
                 change between phases in some mice, and one shape for all of them shows that change as a
                 criterion shift. A fit that ends at ±1, the edge of the stimulus range, is not a
                 measurement (the mouse chose mostly one side, so its 50 % point is off the scale) and
                 counts as missing.
    curve        that criterion in running 50-trial windows, every 10 trials (``criterion_curve``)
    before       the criterion over the last 250 trials before the switch (the previous day's end, in a
                 daily alternation), with the previous phase's shape; the expert-Uniform PSE before the
                 first session of a trajectory
    convergence  (criterion − before) / (normative − before): 0 = where it was, 1 = the normative criterion
                 of the new distribution at the animal's expert-Uniform sigma; not clipped (``convergence``)
    one number   ``convergence_100_300``: the mean over windows centred on trials 100-300 after the switch,
                 inside the first day, so the slow drift across days hardly enters. It is the result for
                 the daily alternation (five or six days per mouse and kind); a block of days is judged by
                 its whole-block fit instead (the switch report's ``block_convergence``).

A switch is either a block of days (the pre-opto cohorts: Uniform → Hard-B → Hard-A → …) or one day of
the daily alternation (the opto Hard phase: A B A B …, six laser days then six masking days);
``hard_design`` tells the two apart by how many days in a row share a distribution.

    tr = compute_trajectory(animal, distributions=('Hard-A', 'Hard-B'))
    tr.sessions      one row per session in acquisition order: order, session_id, session_idx, date,
                     distribution, session_type, n_trials, <pooled stats>, pse_fixed (the criterion
                     parameter with the expert-Uniform shape: the bias check's), pse_pinned (the criterion,
                     with the phase's shape), shape_sigma, shape_lapse_low, shape_lapse_high (that shape),
                     pse_last250, <pse_dynamics> and the same with `_fixed` (the phase's shape),
                     prev_distribution, prev_pse, baseline_pse (the criterion before the session),
                     normative_pse, convergence_100_300, convergence_final, delta_from_prev
    tr.curves        the criterion within each session in running windows, with its convergence
    sw = compute_switches(animal)          # one SwitchResult per switch into a Hard distribution

Pass ``sigma`` to compute the normative criterion at another sigma (e.g. an SBI posterior); the shapes the
criterion is fitted with do not change. ``delta_from_prev`` = PSE(this session) − PSE(previous session),
both free fits on the whole session: the A–B amplitude of the alternation. The bias check (``BiasRule``)
keeps the expert-Uniform shape: it is a coarse screen (|criterion| > 0.4), and a fixed reference keeps the
mice it excludes independent of how each phase's shape is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from behav_utils.analysis import compute_phase_stats
from behav_utils.data import (
    TrialArrays,
    block_after,
    block_before,
    filter_trials,
    find_switches,
    select_sessions,
)
from behav_utils.readouts import fit_psychometric
from behav_utils.stats import PSE_DYNAMICS, PSYCHOMETRIC, compute_stats, fit_pse_dynamics
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from sound_categorisation.data.stimuli import compute_normative_pse

__all__ = ['Trajectory', 'compute_trajectory', 'expert_reference', 'TRAJECTORY_STATS', 'pinned_pse',
           'phase_shape', 'phase_shapes', 'measured_trials', 'block_convergence', 'MIN_CONV_SPAN',
           'criterion_curve', 'convergence', 'mean_convergence', 'pool_convergence', 'CURVE_WINDOW', 'CURVE_STEP', 'EARLY_TRIALS',
           'SwitchResult', 'compute_switch', 'compute_switches', 'SWITCH_METHODS', 'hard_design',
           'flag_biased_sessions', 'BiasRule', 'bias_sessions', 'BIAS_COLUMNS', 'compute_phase_curves', 'animal_switches',
           'switch_min_trials', 'design_switches', 'HARD', 'CONVERGENCE_COLUMNS']

HARD = ('Hard-A', 'Hard-B')
TRAJECTORY_STATS = ['pse', *PSYCHOMETRIC, 'accuracy', 'hard_accuracy', 'side_bias', 'recency', 'win_stay', 'lose_shift']
CRITERION = 0.8            # convergence level defining trials_to_criterion
CURVE_WINDOW, CURVE_STEP = 50, 10     # the criterion curve: running 50-trial windows, every 10 trials
PRE_TRIALS = 250           # the criterion before a switch (or at the end of a session): its last 250 trials
EARLY_TRIALS = (100, 300)  # window centres averaged into convergence_100_300
EDGE_TOL = 1e-3            # a fitted criterion this close to ±1 (the stimulus range) is at the edge: not measured
MIN_CONV_SPAN = 0.02       # |before - normative| below this: convergence undefined (nothing to converge to)
CURVE_COLUMNS = ['order', 'session_id', 'session_idx', 'distribution', 'session_type', 'trial', 'pse', 'convergence']


@dataclass(frozen=True)
class Trajectory:
    animal: str
    distributions: tuple
    sessions: pd.DataFrame
    curves: pd.DataFrame
    baseline_pse: float          # expert-Uniform PSE
    sigma: float                 # psychometric sigma used for the normative PSE
    bias_rule: BiasRule = field(default_factory=lambda: BiasRule())   # the QC the report tables apply
    bias: pd.DataFrame | None = None     # bias_sessions(animal, bias_rule) when asked for (compute_trajectory(bias=True))

    def __repr__(self) -> str:
        return (f'Trajectory({self.animal!r}, {list(self.distributions)}, n_sessions={len(self.sessions)}, '
                f'sigma={self.sigma:.3f}, baseline_pse={self.baseline_pse:+.3f})')


def expert_reference(animal, *, preset: str = 'expert_uniform', last_n: int = 5,
                     trials: str = 'non_opto') -> tuple:
    """(PSE, sigma, lapse_low, lapse_high) pooled over the animal's last ``last_n`` expert-Uniform sessions."""
    from sound_categorisation.data.cohort import ensure_presets
    ensure_presets()
    sessions = select_sessions(animal, preset=preset)
    if last_n:
        sessions = sessions[-last_n:]
    phase = filter_trials(sessions, trial_type=trials)
    if not phase:
        return np.nan, np.nan, np.nan, np.nan
    r = compute_phase_stats(phase, ['pse', 'sigma', 'lapse_low', 'lapse_high']).pooled
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


def _pse_from_params(mu, sigma, lo, hi) -> float:
    """Lapse-corrected PSE (the manuscript's compute_pse); NaN where undefined."""
    denom = 1.0 - lo - hi
    if not (np.isfinite(denom) and denom > 0):
        return np.nan
    arg = (0.5 - lo) / denom
    if not (0.0 < arg < 1.0):
        return np.nan
    return float(norm.ppf(arg, loc=mu, scale=sigma))


def pinned_pse(arrays: TrialArrays, shape) -> float:
    """The criterion: the 50 % point of the psychometric (lapse-corrected PSE) with sigma and lapses fixed at
    ``shape`` = (sigma, lapse_low, lapse_high), so only the criterion is fitted. The reports pass the shape of
    the phase the trials belong to (``phase_shape``).

    Missing (NaN) with fewer than 50 trials, without a usable shape, or when the fit ends at ±1, the edge
    of the stimulus range: the mouse chose mostly one side, its 50 % point is off the scale, and there is
    nothing to measure. ``pse_fixed`` returns the fitted location ``mu``; with unequal lapses the 50 % point
    sits a constant away from it, and the normative criterion and the criterion before a switch are 50 %
    points, so every curve and convergence uses this.
    """
    sigma, lo, hi = shape
    mu = pse_fixed(arrays, sigma, lo, hi)
    if not np.isfinite(mu) or abs(mu) >= 1.0 - EDGE_TOL:
        return np.nan
    return _pse_from_params(mu, sigma, float(np.clip(lo, 0, 0.45)), float(np.clip(hi, 0, 0.45)))


def criterion_curve(arrays: TrialArrays, shape, *, window: int = CURVE_WINDOW, step: int = CURVE_STEP) -> pd.DataFrame:
    """``pinned_pse`` in running windows of ``window`` trials every ``step`` trials: columns ``trial`` (the
    window's centre, counted in valid trials from the first) and ``pse``. Shorter than a window: no rows."""
    v = arrays.valid()
    rows = [(start + window // 2, pinned_pse(v.take(np.arange(start, start + window)), shape))
            for start in range(0, max(v.n_trials - window + 1, 0), step)]
    return pd.DataFrame(rows, columns=['trial', 'pse'])


def convergence(pse, before: float, normative: float):
    """(pse − before) / (normative − before): 0 = still at the criterion before the switch, 1 = at the
    normative criterion. Not clipped. NaN where either reference is missing or they are closer than
    ``MIN_CONV_SPAN`` (nothing to converge to). ``pse`` is a number or an array."""
    p = np.asarray(pse, float)
    if not (np.isfinite(before) and np.isfinite(normative)) or abs(normative - before) < MIN_CONV_SPAN:
        out = np.full(p.shape, np.nan)
    else:
        out = (p - before) / (normative - before)
    return float(out) if out.ndim == 0 else out


def mean_convergence(curve: pd.DataFrame, trials=EARLY_TRIALS) -> float:
    """Mean ``convergence`` of the windows centred on trials ``trials[0]``..``trials[1]`` (missing windows left
    out); NaN when none is measured."""
    if not len(curve):
        return np.nan
    v = curve.loc[curve['trial'].between(*trials), 'convergence'].to_numpy(float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else np.nan


def pool_convergence(values, spans) -> float:
    """Several switches' convergence as one number: each weighted by its span |normative − before|, which is
    the distance moved over the distance to go, summed over the switches (each signed towards its own
    normative criterion). A switch with almost nothing to converge to then counts for almost nothing; the
    plain mean of the ratios is pulled around by such switches (in simulation it reads a half shift as
    0.42 and no shift as −0.13, where this gives 0.49 and 0.01). NaN when nothing is measured."""
    v, w = np.asarray(values, float), np.asarray(spans, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan


def block_convergence(delta, normative_from, normative_to):
    """How far a whole block moved, as a share of how far the normative criterion moved: ``delta`` (the block's
    pooled PSE − the previous phase's pooled PSE, both 4-parameter fits) / (``normative_to`` − ``normative_from``).
    1 = moved with the distribution by the normative amount, 0 = did not move, < 0 = moved the other way. The
    denominator is fixed by the design (about 0.1 for Uniform → Hard, 0.2 between the Hard distributions), so
    unlike ``convergence`` it cannot blow up when the criterion already sat near the new normative one. NaN where
    the normative criterion moves less than ``MIN_CONV_SPAN``. Numbers or arrays."""
    d, f, t = (np.asarray(x, float) for x in (delta, normative_from, normative_to))
    shift = t - f
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(np.abs(shift) >= MIN_CONV_SPAN, d / shift, np.nan)
    return float(out) if out.ndim == 0 else out


def _distribution(session) -> str | None:
    try:
        return session.distribution
    except Exception:
        return None


def _session_type(session) -> str:
    return str(getattr(session, 'session_type', '') or '')


def measured_trials(session) -> str:
    """The trials a session's criterion is measured on: laser-off trials on a laser day, so the laser's effect on
    the trial it is on stays out ('non_opto'), all trials on any other day ('all'). One rule for the switch report
    and the trajectory."""
    return 'non_opto' if _is_laser(_session_type(session)) else 'all'


def phase_shape(sessions, trials: str = 'all') -> tuple:
    """(sigma, lapse_low, lapse_high) of one 4-parameter fit on the pooled ``trials`` of ``sessions`` (thousands of
    trials for a block, hundreds per day pooled over the days of a daily phase): the shape a phase's criterion is
    fitted with (module notes). NaN without trials."""
    phase = filter_trials(list(sessions), trial_type=trials)
    if not phase:
        return (np.nan,) * 3
    r = compute_phase_stats(phase, ['sigma', 'lapse_low', 'lapse_high']).pooled
    return float(r['sigma']), float(r['lapse_low']), float(r['lapse_high'])


def phase_shapes(sessions, design: str, *, trials: str = 'all', expert=(np.nan,) * 3) -> dict:
    """session_idx → the shape its criterion is fitted with, for ``sessions`` in order. A Hard session takes the
    ``phase_shape`` of its phase: its session type within its block (``design='blocked'``; a block is a run of
    consecutive sessions on one distribution), or its distribution and session type over all the sessions
    (``'daily'``: the alternation's days of one kind pooled). Any other session (Uniform) takes ``expert``, the
    expert-Uniform shape. ``trials='auto'`` fits each phase on the trials its sessions are measured on
    (``measured_trials``: laser-off on laser days)."""
    groups, out, run, prev = {}, {}, 0, None
    for s in sessions:
        d = _distribution(s)
        if d != prev:
            run += 1
        prev = d
        if d not in HARD:
            out[s.session_idx] = tuple(expert)
            continue
        groups.setdefault((d if design == 'daily' else run, _session_type(s)), []).append(s)
    for members in groups.values():              # one session type per group, so one trial rule
        shape = phase_shape(members, measured_trials(members[0]) if trials == 'auto' else trials)
        out.update({s.session_idx: shape for s in members})
    return out


def compute_trajectory(
    animal,
    distributions: Sequence[str] = ('Hard-A', 'Hard-B'),
    *,
    stats: Sequence[str] = TRAJECTORY_STATS,
    trials: str = 'auto',
    sigma: float | None = None,
    dynamics: bool = True,
    curves: bool = True,
    window: int = CURVE_WINDOW,
    step: int = CURVE_STEP,
    baseline_preset: str = 'expert_uniform',
    baseline_last_n: int = 5,
    baseline_trials: str = 'non_opto',
    bias_rule: BiasRule | None = None,
    bias: bool = False,
) -> Trajectory:
    """Per-session statistics and adaptation across all sessions of ``distributions``, in order.

    Each session is a switch from the one before: its criterion curve starts from the criterion over the
    previous session's last 250 trials (the expert-Uniform PSE before the first session), and its
    convergence runs from there to the normative criterion of its own distribution (module notes). Each
    session's criterion is fitted with its phase's shape (``phase_shapes``; the design from ``hard_design``).

    Args:
        animal:        AnimalData.
        distributions: which distributions' sessions to include (all session types).
        stats:         scalar stats computed per session on ``trials``.
        trials:        'auto' (default): each session on the trials it is measured on (``measured_trials``:
                       laser-off trials on a laser day, all trials otherwise), as the switch report; or 'all'
                       or 'non_opto' for every session.
        sigma:         psychometric sigma for the normative PSE; from the expert-Uniform fit if None.
        dynamics:      also fit ``pse_dynamics`` per session (exponential vs step).
        curves:        compute the criterion curve within each session (and ``convergence_100_300``).
        window, step:  the curve's running windows (default 50 trials, every 10).
        baseline_preset, baseline_last_n, baseline_trials:
                       the expert-Uniform reference (``expert_reference``): the last ``baseline_last_n``
                       sessions of the preset, on these trials; it gives the criterion before the first
                       session, the sigma of the normative criterion, the shape of the Uniform phase and the
                       bias check's ``pse_fixed``. With the default preset, opto-cohort references come mostly
                       from laser sessions (masking is excluded); see the report README.
        bias_rule:     the cohort QC carried with the result (``BiasRule()`` if None).
        bias:          also attach ``bias_sessions(animal, bias_rule)`` (``Trajectory.bias``), the one bias
                       check every report uses.
    """
    aid = getattr(animal, 'animal_id', '?')
    base_pse, base_sigma, base_lo, base_hi = expert_reference(animal, preset=baseline_preset,
                                                              last_n=baseline_last_n, trials=baseline_trials)
    expert = (base_sigma, base_lo, base_hi)
    sigma = float(sigma) if sigma is not None else base_sigma
    rule = bias_rule if bias_rule is not None else BiasRule()

    sessions = []
    for d in distributions:
        sessions += select_sessions(animal, distribution=d)
    sessions = sorted({s.session_idx: s for s in sessions}.values(), key=lambda s: s.session_idx)
    design = hard_design(animal) if any(d in HARD for d in distributions) else 'none'
    shapes = phase_shapes(sessions, design, trials=trials, expert=expert)

    rows, curve_frames = [], []
    prev_pse, prev_dist = base_pse, 'Uniform'
    before = base_pse                      # the criterion just before the next session
    for order, s in enumerate(sessions):
        phase = filter_trials([s], trial_type=measured_trials(s) if trials == 'auto' else trials)
        dist = _distribution(s)
        shape = shapes.get(s.session_idx, expert)
        norm = float(compute_normative_pse(dist, sigma)) if (dist and np.isfinite(sigma)) else np.nan
        row = {'order': order, 'session_id': s.session_id, 'session_idx': s.session_idx,
               'date': getattr(s, 'date', None), 'distribution': dist, 'session_type': s.session_type,
               'n_trials': 0, 'prev_distribution': prev_dist, 'prev_pse': prev_pse,
               'baseline_pse': before, 'normative_pse': norm, 'convergence_100_300': np.nan,
               'shape_sigma': shape[0], 'shape_lapse_low': shape[1], 'shape_lapse_high': shape[2]}
        end = np.nan
        if phase:
            arrays = TrialArrays.from_sessions(phase)
            row['n_trials'] = arrays.n_responded
            row.update(compute_stats(arrays, list(stats), strict=False).to_dict())
            row['pse_fixed'] = pse_fixed(arrays, *expert)         # the bias check's: expert-Uniform shape
            row['pse_pinned'] = pinned_pse(arrays, shape)
            valid = arrays.valid()
            end = pinned_pse(valid.take(np.arange(valid.n_trials)[-PRE_TRIALS:]), shape)
            if dynamics:
                row.update(compute_stats(arrays, list(PSE_DYNAMICS), strict=False).to_dict())
                fixed = fit_pse_dynamics(arrays, shape=shape)
                row.update({f'{k}_fixed': v for k, v in zip(PSE_DYNAMICS, fixed)})
            if curves:
                c = criterion_curve(arrays, shape, window=window, step=step)
                c['convergence'] = convergence(c['pse'].to_numpy(), before, norm)
                row['convergence_100_300'] = mean_convergence(c)
                curve_frames.append(pd.DataFrame({
                    'order': order, 'session_id': s.session_id, 'session_idx': s.session_idx,
                    'distribution': dist, 'session_type': s.session_type,
                    'trial': c['trial'].to_numpy(), 'pse': c['pse'].to_numpy(),
                    'convergence': c['convergence'].to_numpy()}, columns=CURVE_COLUMNS))
        row['pse_last250'] = end
        row['convergence_final'] = convergence(end, before, norm)
        row['span'] = abs(norm - before) if np.isfinite(norm - before) else np.nan
        pse_now = row.get('pse', np.nan)
        row['delta_from_prev'] = pse_now - prev_pse if (np.isfinite(pse_now) and np.isfinite(prev_pse)) else np.nan
        rows.append(row)
        if np.isfinite(pse_now):
            prev_pse, prev_dist = pse_now, dist
        before = end                       # unmeasured at the end (at the edge): the next session has no start

    sess_df = pd.DataFrame(rows)
    curves_df = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame(columns=CURVE_COLUMNS)
    return Trajectory(aid, tuple(distributions), sess_df, curves_df, base_pse, sigma, rule,
                      bias_sessions(animal, rule) if bias else None)


# ═══════════════════════════════════════════════════════════════════════════
# Switches: a block of days (pre-opto cohorts) or one day of the daily alternation (opto Hard phase)
# ═══════════════════════════════════════════════════════════════════════════
#
# Manuscript Fig. 5C recipe (Figures/Figure-5/Panel-C notebook):
#   pre  = last 250 trials before the switch, 4-parameter fit, lapse-corrected PSE
#   post = non-overlapping 50-trial bins across the block's sessions, up to 3000 trials,
#          4-parameter fit per bin, convergence = 1 - (pse_bin - norm)/(pse_pre - norm), clipped [0, 1]
# Reproduced as method='manuscript', for comparison. method='pinned' is the module's criterion
# (``pinned_pse``: the 50 % point with the shape fixed at the phase's fit, edge = missing) per bin, unclipped,
# from the same criterion over the last 250 trials before the switch (with the previous phase's shape);
# 'pinned_running' is the curve (``criterion_curve``: running 50-trial windows, step 10), the one the reports
# and the thesis use. The normative PSE is the constant-sigma observer at the animal's expert-Uniform sigma,
# for every method (the manuscript used a per-subject stimulus-dependent-noise normative model fitted on
# Uniform data, whose midpoints we do not have).

SWITCH_PRE_TRIALS = PRE_TRIALS
SWITCH_BIN = 50
SWITCH_STEP = 10
SWITCH_MAX_TRIALS = 3000
SWITCH_MIN_BLOCK = 1000    # blocked design: the fewest trials a run of days needs to count as a block
DAY_MIN_TRIALS = 100       # daily design: the fewest trials a day needs to count as a switch
BLOCKED_RUN = 3            # hard_design: 'blocked' when the median run of days on one distribution is at least this
OVERNIGHT_TRIALS = 100
SWITCH_METHODS = ('manuscript', 'pinned', 'pinned_running')
CONVERGENCE_COLUMNS = ['method', 'bin', 'trial', 'pse', 'convergence', 'convergence_clipped', 'span', 'n']
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
    pre: pd.Series                    # last SWITCH_PRE_TRIALS trials: mu, sigma, lapse_low, lapse_high, pse (4-parameter), pse_pinned; shape_sigma, shape_lapse_low, shape_lapse_high (the previous phase's shape)
    normative_pse: float
    convergence: pd.DataFrame         # CONVERGENCE_COLUMNS: method, bin, trial, pse, convergence, convergence_clipped, span, n
    dynamics: pd.Series               # PSE_DYNAMICS (the phase's shape) on the block + trials_to_criterion, plateau, convergence_100_300, pse_post, convergence_final, shape_sigma, shape_lapse_low, shape_lapse_high
    sessions: pd.DataFrame            # per session in the block: order_in_block, session_idx, session_type, n_trials, pse, pse_pinned, accuracy, ...
    overnight: pd.DataFrame           # per consecutive session pair: pse_end, pse_start_next, delta, toward_pre, session_idx, next_session_idx
    design: str = 'blocked'           # 'blocked' (a block of days) | 'daily' (one day of the alternation)
    switch_kind: str = ''             # blocked: the transition; daily: 'first day', 'laser day', '<type> day' or 'mixed day'
    trials: str = 'all'               # the trials measured: 'all', or 'non_opto' (laser-off) for a laser day
    session_type: str = ''            # the first session after the switch
    prev_session_type: str = ''       # the last session before it
    normative_from: float = np.nan    # the normative criterion of the distribution before the switch

    @property
    def labels(self) -> dict:
        return {'animal': self.animal, 'switch_idx': self.switch_idx, 'from_distribution': self.from_distribution,
                'to_distribution': self.to_distribution, 'transition': self.transition, 'design': self.design,
                'switch_kind': self.switch_kind, 'trials': self.trials, 'session_type': self.session_type,
                'prev_session_type': self.prev_session_type}

    def to_rows(self) -> pd.DataFrame:
        base = self.labels
        rows = [{**base, 'stat': f'pre_{k}', 'value': float(v)} for k, v in self.pre.items()]
        rows += [{**base, 'stat': 'normative_pse', 'value': self.normative_pse},
                 {**base, 'stat': 'normative_from', 'value': self.normative_from}]
        rows += [{**base, 'stat': k, 'value': float(v)} for k, v in self.dynamics.items()]
        rows += [{**base, 'stat': 'n_trials_before', 'value': self.n_trials_before},
                 {**base, 'stat': 'n_trials_after', 'value': self.n_trials_after},
                 {**base, 'stat': 'n_sessions_after', 'value': self.n_sessions_after}]
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (f'SwitchResult({self.animal!r} #{self.switch_idx} {self.from_distribution}→{self.to_distribution} '
                f'[{self.switch_kind or self.transition}], n_after={self.n_trials_after}, '
                f'convergence_100_300={self.dynamics.get("convergence_100_300", np.nan):.2f})')


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


def _is_laser(session_type: str) -> bool:
    from sound_categorisation.data.cohort import LASER_TYPES
    return session_type in LASER_TYPES


def _switch_kind(design: str, transition: str, session_type: str, prev_session_type: str) -> str:
    """Blocked: the transition (first / novel / return). Daily: 'first day' (from the Uniform phase),
    'laser day' (a laser day after a laser day), '<type> day' (e.g. 'masking day': the same no-laser type
    on both days) or 'mixed day' (a day after a day of the other kind, e.g. the first masking day)."""
    if design != 'daily':
        return transition
    if transition == 'first':
        return 'first day'
    laser, prev_laser = _is_laser(session_type), _is_laser(prev_session_type)
    if laser and prev_laser:
        return 'laser day'
    if not laser and not prev_laser and session_type == prev_session_type:
        return f'{session_type} day'
    return 'mixed day'


def _exclude_types(design: str, exclude_types=None) -> tuple:
    """The session types a design leaves out: the controls (``CONTROL_TYPES``), except that the daily
    alternation keeps its masking days, which are its no-laser days."""
    from sound_categorisation.data.cohort import CONTROL_TYPES
    if exclude_types is not None:
        return tuple(exclude_types)
    return tuple(CONTROL_TYPES) if design == 'blocked' else tuple(t for t in CONTROL_TYPES if t != 'masking')


def hard_design(animal, *, stage: str = 'Full_Task_Cont', exclude_types=None) -> str:
    """How the animal's Hard sessions were scheduled: 'blocked' when the median run of consecutive Hard
    sessions on one distribution is at least ``BLOCKED_RUN`` (the pre-opto cohorts: runs of many days),
    'daily' otherwise (the opto Hard phase, A B A B …; one repeated day does not make it blocked), 'none'
    without Hard sessions. Masking days count; other distributions between Hard sessions are ignored."""
    sessions = sorted(select_sessions(animal, stage=stage, exclude_types=_exclude_types('daily', exclude_types)),
                      key=lambda s: s.session_idx)
    dists = [d for d in (_distribution(s) for s in sessions) if d in HARD]
    if not dists:
        return 'none'
    runs, start = [], 0
    for i in range(1, len(dists) + 1):
        if i == len(dists) or dists[i] != dists[i - 1]:
            runs.append(i - start)
            start = i
    return 'blocked' if np.median(runs) >= BLOCKED_RUN else 'daily'


def _phase_sessions(sessions: list, switch: dict, design: str, side: str) -> list | None:
    """The sessions of the phase on ``side`` ('before' | 'after') of ``switch``, whose pooled fit gives the shape
    the criterion there is fitted with: the block itself (blocked), or every day in ``sessions`` with that day's
    distribution and session type (daily). None for a phase that is not Hard: the Uniform phase takes the
    expert-Uniform shape."""
    block = block_before(sessions, switch) if side == 'before' else block_after(sessions, switch)
    if not block:
        return None
    day = block[-1] if side == 'before' else block[0]
    dist = _distribution(day)
    if dist not in HARD:
        return None
    if design != 'daily':
        return block
    return [s for s in sessions if _distribution(s) == dist and _session_type(s) == _session_type(day)]


def compute_switch(
    animal,
    sessions: list,
    switch: dict,
    *,
    transition: str,
    design: str = 'blocked',
    expert_shape: tuple | None = None,
    sigma: float | None = None,
    trials: str = 'all',
    max_trials: int = SWITCH_MAX_TRIALS,
    bin_size: int = SWITCH_BIN,
    step: int = SWITCH_STEP,
    pre_trials: int = SWITCH_PRE_TRIALS,
    criterion: float = CRITERION,
) -> SwitchResult | None:
    """Adaptation after one switch (module notes): across the whole block of days (``design='blocked'``)
    or across the one day that follows it (``design='daily'``).

    The criterion after the switch is fitted with the new phase's shape and the criterion before it with the
    previous phase's (``phase_shape`` on ``trials``: the whole block, or the design's days of that distribution
    and session type pooled); a Uniform phase takes ``expert_shape`` = (sigma, lapse_low, lapse_high), from
    ``expert_reference`` if None. ``sigma`` (default the expert-Uniform one) gives the normative criterion.
    ``trials`` ('all' | 'non_opto') selects the trials on both sides of the switch. None when either side has
    no trials.
    """
    aid = getattr(animal, 'animal_id', '?')
    expert = tuple(expert_shape) if expert_shape is not None else tuple(expert_reference(animal)[1:])
    sigma = expert[0] if sigma is None else float(sigma)

    def shape_of(side):
        members = _phase_sessions(sessions, switch, design, side)
        return expert if members is None else phase_shape(members, trials)
    shape_before, shape = shape_of('before'), shape_of('after')
    before_sessions, after_sessions = block_before(sessions, switch), block_after(sessions, switch)
    if design == 'daily':
        after_sessions = after_sessions[:1]          # one day: the next day is the next switch
    before = filter_trials(before_sessions, trial_type=trials)
    after = filter_trials(after_sessions, trial_type=trials)
    if not before or not after:
        return None
    to, frm = switch['to_distribution'], switch['from_distribution']
    session_type = str(getattr(after_sessions[0], 'session_type', '') or '')
    prev_session_type = str(getattr(before_sessions[-1], 'session_type', '') or '')

    # ── before the switch: the manuscript's 4-parameter fit, and the criterion (previous phase's shape) ──
    A_pre = TrialArrays.from_sessions(before).valid()
    A_pre = A_pre.take(np.arange(A_pre.n_trials)[-pre_trials:])
    mu0, sig0, lo0, hi0, pse0 = _fit4(A_pre.stimulus, A_pre.choice)
    pre_pinned = pinned_pse(A_pre, shape_before)
    pre = pd.Series({'mu': mu0, 'sigma': sig0, 'lapse_low': lo0, 'lapse_high': hi0, 'pse': pse0,
                     'pse_pinned': pre_pinned, 'shape_sigma': shape_before[0], 'shape_lapse_low': shape_before[1],
                     'shape_lapse_high': shape_before[2]}, dtype=float)
    norm_pse = float(compute_normative_pse(to, sigma)) if np.isfinite(sigma) else np.nan
    norm_from = float(compute_normative_pse(frm, sigma)) if np.isfinite(sigma) else np.nan

    # ── after the switch, in order, truncated ─────────────────────────────
    A = TrialArrays.from_sessions(after).valid()
    A = A.take(np.arange(min(A.n_trials, max_trials)))
    n = A.n_trials

    def clipped(c):
        return float(np.clip(c, 0, 1)) if np.isfinite(c) else np.nan

    span_manuscript = abs(norm_pse - pse0) if np.isfinite(norm_pse - pse0) else np.nan
    span = abs(norm_pse - pre_pinned) if np.isfinite(norm_pse - pre_pinned) else np.nan
    rows = []
    for b, start in enumerate(range(0, n, bin_size), start=1):
        sl = A.take(np.arange(start, min(start + bin_size, n)))
        if sl.n_trials < bin_size // 2:
            continue
        # manuscript: 4-parameter fit per bin against its own 4-parameter pre-switch PSE, clipped; it plots
        # bin q at q*bin_size + 1 (the trial after the bin), x = 1 being its pre-switch anchor
        pse_b = _fit4(sl.stimulus, sl.choice)[4]
        c = convergence(pse_b, pse0, norm_pse)
        rows.append({'method': 'manuscript', 'bin': b, 'trial': b * bin_size + 1, 'pse': pse_b, 'convergence': c,
                     'convergence_clipped': clipped(c), 'span': span_manuscript, 'n': sl.n_trials})
        pse_p = pinned_pse(sl, shape)
        c = convergence(pse_p, pre_pinned, norm_pse)
        rows.append({'method': 'pinned', 'bin': b, 'trial': start + 1, 'pse': pse_p, 'convergence': c,
                     'convergence_clipped': clipped(c), 'span': span, 'n': sl.n_trials})
    run = criterion_curve(A, shape, window=bin_size, step=step)
    run_conv = convergence(run['pse'].to_numpy(), pre_pinned, norm_pse)
    for b, (t, p, c) in enumerate(zip(run['trial'], run['pse'], run_conv), start=1):
        rows.append({'method': 'pinned_running', 'bin': b, 'trial': int(t), 'pse': p, 'convergence': c,
                     'convergence_clipped': clipped(c), 'span': span, 'n': bin_size})
    conv_df = pd.DataFrame(rows, columns=CONVERGENCE_COLUMNS)

    # ── the block (or day) as a whole, the phase's shape ─────────────────
    dyn = dict(zip(PSE_DYNAMICS, fit_pse_dynamics(A, shape=shape)))
    pr = conv_df[conv_df['method'] == 'pinned_running']
    reached = pr.loc[pr['convergence'] >= criterion, 'trial']
    dyn['trials_to_criterion'] = float(reached.iloc[0]) if len(reached) else np.nan
    late = pr.loc[pr['trial'] > (2 * n) // 3, 'convergence'].dropna()       # plateau: last third, running windows
    dyn['plateau'] = float(late.mean()) if len(late) else np.nan
    dyn['convergence_100_300'] = mean_convergence(pr)
    dyn['span'] = span                     # |normative - before|: the weight when switches are averaged
    A_full = TrialArrays.from_sessions(after).valid()
    dyn['pse_post'] = pinned_pse(A_full.take(np.arange(A_full.n_trials)[-pre_trials:]), shape)
    dyn['convergence_final'] = convergence(dyn['pse_post'], pre_pinned, norm_pse)
    dyn.update(shape_sigma=shape[0], shape_lapse_low=shape[1], shape_lapse_high=shape[2])
    dynamics = pd.Series(dyn, dtype=float)

    # ── per session within the block, and overnight pairs ────────────────
    # overnight: the criterion over the last 100 trials of a session against the first 100 of the next;
    # `toward_pre` signs the change by the direction of the switch (> 0 = back towards the previous
    # distribution's normative criterion), so noise in either window cannot set its sign
    direction = np.sign(norm_from - norm_pse) if np.isfinite(norm_from - norm_pse) else np.nan
    srows, orows = [], []
    prev_end, prev_idx = np.nan, None
    for k, s in enumerate(after):
        a_s = TrialArrays.from_sessions([s]).valid()
        st = compute_stats(a_s, ['pse', 'accuracy', 'hard_accuracy', 'recency', 'win_stay', 'lose_shift', *PSYCHOMETRIC],
                           strict=False)
        srows.append({'order_in_block': k, 'session_idx': s.session_idx, 'session_type': _session_type(s),
                      'n_trials': a_s.n_trials, 'pse_pinned': pinned_pse(a_s, shape), **st.to_dict()})
        start_pse = pinned_pse(a_s.take(np.arange(min(OVERNIGHT_TRIALS, a_s.n_trials))), shape)
        end_pse = pinned_pse(a_s.take(np.arange(a_s.n_trials)[-OVERNIGHT_TRIALS:]), shape)
        if prev_idx is not None:
            delta = start_pse - prev_end
            orows.append({'session_idx': prev_idx, 'next_session_idx': s.session_idx, 'pse_end': prev_end,
                          'pse_start_next': start_pse, 'delta': delta, 'toward_pre': delta * direction})
        prev_end, prev_idx = end_pse, s.session_idx
    sessions_df = pd.DataFrame(srows)
    overnight = pd.DataFrame(orows, columns=['session_idx', 'next_session_idx', 'pse_end', 'pse_start_next', 'delta', 'toward_pre'])

    return SwitchResult(aid, int(switch['switch_idx']), frm, to, transition, int(switch['n_trials_before']),
                        int(sum(len(s.trials.choice) for s in after_sessions)), len(after_sessions), pre, norm_pse,
                        conv_df, dynamics, sessions_df, overnight, design=design,
                        switch_kind=_switch_kind(design, transition, session_type, prev_session_type),
                        trials=trials, session_type=session_type, prev_session_type=prev_session_type,
                        normative_from=norm_from)


def animal_switches(animal, *, stage: str = 'Full_Task_Cont', exclude_types=None,
                    min_block_trials: int = SWITCH_MIN_BLOCK) -> tuple:
    """(ordered regular sessions, qualifying switches) for one animal."""
    from sound_categorisation.data.cohort import CONTROL_TYPES
    sessions = sorted(select_sessions(animal, stage=stage,
                                      exclude_types=CONTROL_TYPES if exclude_types is None else exclude_types),
                      key=lambda s: s.session_idx)
    return sessions, find_switches(sessions, min_block_trials=min_block_trials)


def switch_min_trials(design: str, min_block_trials: int = SWITCH_MIN_BLOCK, min_day_trials: int = DAY_MIN_TRIALS) -> int:
    """The fewest trials a run needs to count, by design: a block of days, or one day."""
    return min_block_trials if design == 'blocked' else min_day_trials


def design_switches(animal, design: str, *, stage: str = 'Full_Task_Cont', exclude_types=None,
                    min_block_trials: int = SWITCH_MIN_BLOCK, min_day_trials: int = DAY_MIN_TRIALS) -> tuple:
    """(ordered sessions, every switch) for one animal under ``design``: blocks of at least ``min_block_trials``
    trials of the regular sessions, or each day of at least ``min_day_trials`` with the masking days kept
    (they are the alternation's no-laser days). Keep the switches whose ``to_distribution`` is in ``HARD``
    for the Hard phase; the transition labels need them all."""
    return animal_switches(animal, stage=stage, exclude_types=_exclude_types(design, exclude_types),
                           min_block_trials=switch_min_trials(design, min_block_trials, min_day_trials))


def compute_switches(animal, *, stage: str = 'Full_Task_Cont', exclude_types=None, design: str | None = None,
                     min_block_trials: int = SWITCH_MIN_BLOCK, min_day_trials: int = DAY_MIN_TRIALS, **kw) -> list:
    """Every switch of one animal into a Hard distribution, in order, with transition labels.

    ``design`` ('blocked' | 'daily'; ``hard_design`` if None) decides what a switch is: the start of a block
    of at least ``min_block_trials`` trials, or each day of at least ``min_day_trials``. A laser day is measured
    on laser-off trials (``measured_trials``; its own and the day before's), every other switch on all trials,
    as in the trajectory. Every criterion is fitted with the shape of its phase (``compute_switch``).
    ``kw`` goes to ``compute_switch``.
    """
    design = design or hard_design(animal, stage=stage, exclude_types=exclude_types)
    if design == 'none':
        return []
    sessions, switches = design_switches(animal, design, stage=stage, exclude_types=exclude_types,
                                         min_block_trials=min_block_trials, min_day_trials=min_day_trials)
    if kw.get('expert_shape') is None:
        kw['expert_shape'] = tuple(expert_reference(animal)[1:])
    out, seen = [], []
    for k, sw in enumerate(switches):
        seen.append(sw['from_distribution'])
        if sw['to_distribution'] not in HARD:
            continue
        tr = _transition(seen[:-1] + [sw['from_distribution']], sw['to_distribution'], first=(k == 0))
        day = block_after(sessions, sw)[0]
        r = compute_switch(animal, sessions, sw, transition=tr, design=design, trials=measured_trials(day), **kw)
        if r is not None:
            out.append(r)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Cohort QC: animals that develop a side bias and keep it regardless of the distribution
# ═══════════════════════════════════════════════════════════════════════════
# The rule is about the shape of the per-session psychometric, not about whether the
# animal adapts, so applying it before looking at the A/B effect is not circular.

@dataclass(frozen=True)
class BiasRule:
    """The cohort QC: drop animals that just stick to one side, unaffected by the distribution or the stimulus.

    A Hard session is flagged when |pse_fixed| > ``pse``, either lapse > ``lapse``, or accuracy <
    ``accuracy``. An animal is biased when more than ``majority`` of its Hard sessions are flagged and
    its flagged sessions lean the same way (sign of the median ``pse_fixed``) on Hard-A and on Hard-B.
    ``both_distributions=False`` restores the rule up to 0.7, where flags confined to one distribution
    also counted as the same lean — that is a lean that may follow the distribution, not a side bias.
    """
    pse: float = 0.4
    lapse: float = 0.3
    accuracy: float = 0.6
    majority: float = 0.5
    both_distributions: bool = True


def session_bias_checks(sessions: pd.DataFrame, rule: BiasRule | None = None,
                        pse_column: str = 'pse_fixed') -> pd.DataFrame:
    """The per-session part of ``BiasRule``: which of its three tests each session fails.

    Returns a frame on ``sessions``' index with booleans ``pse_out`` (|PSE| > ``rule.pse``),
    ``lapse_out`` (either lapse > ``rule.lapse``), ``accuracy_out`` (accuracy < ``rule.accuracy``) and
    ``flagged`` (any of them). A missing value fails nothing. ``pse_column`` names the PSE to test:
    ``pse_fixed`` in the switch report, the free-fit ``pse`` in the daily tracker.
    """
    r = rule if rule is not None else BiasRule()
    out = pd.DataFrame(index=sessions.index)
    out['pse_out'] = (sessions[pse_column].abs() > r.pse).fillna(False).astype(bool)
    out['lapse_out'] = ((sessions['lapse_low'] > r.lapse)
                        | (sessions['lapse_high'] > r.lapse)).fillna(False).astype(bool)
    out['accuracy_out'] = (sessions['accuracy'] < r.accuracy).fillna(False).astype(bool)
    out['flagged'] = out['pse_out'] | out['lapse_out'] | out['accuracy_out']
    return out


def flag_biased_sessions(sessions: pd.DataFrame, rule: BiasRule | None = None) -> pd.DataFrame:
    """Add ``flagged`` (per session) and ``biased_animal`` (per animal) to a sessions table.

    ``sessions`` needs columns animal, to_distribution, pse_fixed, lapse_low, lapse_high, accuracy
    (as ``bias_sessions`` builds them). ``rule`` defaults to ``BiasRule()``.
    """
    r = rule if rule is not None else BiasRule()
    s = sessions.copy()
    s['flagged'] = session_bias_checks(s, r)['flagged']
    biased = {}
    for aid, d in s.groupby('animal'):
        frac = d['flagged'].mean()
        f = d[d['flagged'] & d['pse_fixed'].notna()]
        signs = f.groupby('to_distribution')['pse_fixed'].apply(lambda v: np.sign(v.median()))
        same_sign = len(signs) >= 1 and signs.nunique() == 1 and (len(signs) >= 2 or not r.both_distributions)
        biased[aid] = bool(frac > r.majority and same_sign)
    s['biased_animal'] = s['animal'].map(biased)
    return s


BIAS_COLUMNS = ['animal', 'order', 'session_idx', 'session_id', 'session_type', 'to_distribution', 'n_trials',
                'pse_fixed', 'lapse_low', 'lapse_high', 'accuracy', 'flagged', 'biased_animal']


def bias_sessions(animal, rule: BiasRule | None = None) -> pd.DataFrame:
    """The one bias check, which every report uses (switch report, opto battery): each Hard session of the
    animal, in order, on laser-off trials, with its criterion (``pse_fixed``, shape fixed at the animal's
    expert-Uniform fit), its lapses (free fit) and accuracy, ``flagged`` by ``rule`` (``BiasRule()`` if
    None), and ``biased_animal``. Columns ``BIAS_COLUMNS``; no Hard sessions → no rows."""
    tr = compute_trajectory(animal, HARD, stats=('lapse_low', 'lapse_high', 'accuracy'), trials='non_opto',
                            dynamics=False, curves=False, bias_rule=rule)
    s = tr.sessions
    if not len(s):
        return pd.DataFrame(columns=BIAS_COLUMNS)
    s = s.assign(animal=tr.animal).rename(columns={'distribution': 'to_distribution'})
    for col in ('pse_fixed', 'lapse_low', 'lapse_high', 'accuracy'):
        if col not in s:
            s[col] = np.nan
    return flag_biased_sessions(s, rule)[BIAS_COLUMNS].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Psychometrics by phase: Uniform, all Hard-A, all Hard-B, and each block in order
# ═══════════════════════════════════════════════════════════════════════════

def compute_phase_curves(animal, sessions: list, switches: list, *, per_block: bool = True,
                         trials: str = 'all') -> tuple:
    """(params, curves) tables for the pre-switch Uniform reference (last UNIFORM_REF_SESSIONS sessions), every
    qualifying Hard block in order (``per_block``; off for the daily alternation, where a block is one day),
    and the pooled Hard-A / Hard-B sessions, on ``trials``. Fits on pooled trials (thousands per phase), no
    bootstrap."""
    from behav_utils.readouts import compute_psychometric_curve
    aid = getattr(animal, 'animal_id', '?')
    phases = []
    if switches:
        phases.append(('Uniform', 'Uniform', 0, block_before(sessions, switches[0])[-UNIFORM_REF_SESSIONS:]))
        counts = {}
        for sw in switches if per_block else ():
            d = sw['to_distribution']
            counts[d] = counts.get(d, 0) + 1
            phases.append((f"{d} #{counts[d]}", d, sw['switch_idx'] + 1, block_after(sessions, sw)))
        for d in HARD:
            pooled = [s for sw in switches if sw['to_distribution'] == d for s in block_after(sessions, sw)]
            if pooled:
                phases.append((f'{d} all', d, 90 + (d == 'Hard-B'), pooled))
    prow, crow = [], []
    for label, dist, order, sess in phases:
        clean = filter_trials(sess, trial_type=trials)
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
