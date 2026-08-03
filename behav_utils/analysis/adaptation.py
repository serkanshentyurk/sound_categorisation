"""
Adaptation analysis — the normative reference and the noise-parameter resolver.

Two pieces here, both feeding the convergence analysis (see compute_adaptation,
built on top):

  compute_normative_pse — the ideal-observer PSE for a distribution, given a
                          perceptual-noise σ. Pure 2AFC theory, no project state.
  resolve_sigma         — picks that σ from an animal's data, which IS a
                          project-specific choice (which sessions, which trials).

The split is deliberate. The normative PSE depends only on σ (the encoding
noise, ≈ the psychometric slope), which is the d′-related channel your opto
artifact SPARES — so the reference is robust to a criterion/PSE shift in the
source sessions. What σ is estimated *from*, however, is a judgement that
depends on which sessions you trust, so it is an explicit argument rather than
baked in: expert-uniform by default, switchable to masking sessions or an SBI
posterior without touching anything downstream.
"""

from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, fsolve
from scipy.stats import norm

if TYPE_CHECKING:
    from behav_utils.data.structures import AnimalData

__all__ = ['compute_normative_pse', 'resolve_sigma']


# ── Hard-distribution density constant ───────────────────────────────────────
# The "hard" half tilts an exponential density toward the boundary; λ solves
# λ + exp(-λ) = 2 so the tilted and flat halves integrate to the same mass.
def _solve_lambda() -> float:
    """Solve λ + exp(-λ) = 2. Result ≈ 1.841."""
    root = fsolve(lambda x: x + np.exp(-x) - 2, 1.0)
    return float(root[0])


_LAMBDA = _solve_lambda()


# ── Normative PSE (ported verbatim from the legacy stimulus_distributions) ───

def compute_normative_pse(
    distribution: str,
    sigma_percep: float,
    boundary: float = 0.0,
) -> float:
    """Optimal PSE for an ideal Bayesian observer.

    The observer perceives x ~ N(s, σ²) given true stimulus s, with balanced
    priors P(A) = P(B) = 0.5. The optimal boundary is the x at which
    p(x|A) = p(x|B), where p(x|c) = ∫ N(x; s, σ²) · p(s|c) ds.

    Uniform: PSE = boundary (by symmetry). Hard-A / Hard-B: solved numerically
    via brentq.

    Args:
        distribution: 'uniform' | 'hard_a' | 'hard_b' (case/hyphen tolerant).
        sigma_percep: Perceptual noise σ (> 0). This is the σ of the encoding
                      model (x ~ N(s, σ²)), NOT a fitted psychometric slope,
                      though psychometric σ on expert uniform sessions is an
                      acceptable approximation when lapses are low. From SBI,
                      use the BE/SC posterior of sigma_percep.
        boundary:     Category boundary (default 0).

    Returns:
        Optimal PSE (mu) in stimulus units, or NaN if the root solve fails.

    Raises:
        ValueError: on non-positive σ or an unknown distribution.
    """
    key = distribution.lower().replace('-', '_').replace(' ', '_')

    if sigma_percep <= 0:
        raise ValueError(f"sigma_percep must be positive, got {sigma_percep}")

    if key == 'uniform':
        return float(boundary)

    if key not in ('hard_a', 'hard_b'):
        raise ValueError(
            f"Unknown distribution '{distribution}'. Available: uniform, hard_a, hard_b")

    sigma = sigma_percep

    def p_x_given_a(x: float) -> float:
        if key == 'hard_a':
            integrand = lambda s: (
                norm.pdf(x, s + boundary, sigma)
                * (_LAMBDA * np.exp(_LAMBDA * s) + np.exp(-_LAMBDA))
            )
        else:
            integrand = lambda s: norm.pdf(x, s + boundary, sigma)
        return quad(integrand, -1, 0, limit=100)[0]

    def p_x_given_b(x: float) -> float:
        if key == 'hard_b':
            integrand = lambda s: (
                norm.pdf(x, s + boundary, sigma)
                * (_LAMBDA * np.exp(-_LAMBDA * s) + np.exp(-_LAMBDA))
            )
        else:
            integrand = lambda s: norm.pdf(x, s + boundary, sigma)
        return quad(integrand, 0, 1, limit=100)[0]

    def difference(x: float) -> float:
        return p_x_given_b(x) - p_x_given_a(x)

    for span in (0.5, 1.0):
        try:
            return float(brentq(difference, boundary - span, boundary + span, xtol=1e-5))
        except ValueError:
            continue
    return float('nan')


# ── σ resolver — the project-specific choice of where σ comes from ───────────

def resolve_sigma(
    animal: 'AnimalData',
    source: str = 'expert_uniform',
    trials: str = 'non_opto',
    sigma_value: Optional[float] = None,
) -> float:
    """Estimate the perceptual-noise σ for one animal.

    σ feeds ``compute_normative_pse`` as the denominator reference for
    convergence. Which sessions and trials to fit it from is a judgement — this
    is the single place that judgement lives, so switching it never touches the
    adaptation pipeline.

    Args:
        animal: AnimalData.
        source: where to estimate σ from —
            'expert_uniform'  — last-half uniform sessions, accuracy-gated
                                (the ``expert_uniform`` preset). Default: least
                                entangled with the manipulation, since it is the
                                animal's own expert slope.
            'uniform_masking' — uniform masking sessions. Use only if the data
                                argues for it; these are the sessions most
                                suspected of carrying the WT artifact, though σ
                                (a slope) should be largely spared by a
                                criterion-channel artifact.
            'fixed'           — use ``sigma_value`` directly (e.g. an SBI
                                posterior σ). Bypasses the data entirely.
        trials: 'non_opto' (drop laser/masked trials — artifact-free, the safe
                default) or 'all' (every trial). Ignored for 'fixed'.
        sigma_value: required when ``source='fixed'``.

    Returns:
        σ as a float (the fitted psychometric sigma, or the fixed value).

    Raises:
        ValueError: on unknown source/trials, missing sigma_value for 'fixed',
            or no usable sessions.
    """
    if source == 'fixed':
        if sigma_value is None or sigma_value <= 0:
            raise ValueError("resolve_sigma: source='fixed' needs a positive sigma_value")
        return float(sigma_value)

    if source not in ('expert_uniform', 'uniform_masking'):
        raise ValueError(
            f"resolve_sigma: unknown source {source!r}; "
            f"expected 'expert_uniform', 'uniform_masking' or 'fixed'")
    if trials not in ('all', 'non_opto'):
        raise ValueError(f"resolve_sigma: trials must be 'all' or 'non_opto', got {trials!r}")

    from behav_utils.data.ops.selection import select_sessions
    from behav_utils.data.ops.filtering import filter_trials
    from behav_utils.analysis.statistics import compute_stat

    if source == 'expert_uniform':
        sessions = select_sessions(animal, 'expert_uniform')
    else:
        sessions = select_sessions(animal, distribution='Uniform', session_type='masking')

    if not sessions:
        raise ValueError(
            f"resolve_sigma: no sessions for source={source!r} on {getattr(animal, 'animal_id', '?')}")

    condition = filter_trials(sessions, trial_type=('non_opto' if trials == 'non_opto' else 'all'))
    sigma = compute_stat(condition, ['psychometric'], mode='pooled')['pooled'].get('sigma', np.nan)

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(
            f"resolve_sigma: psychometric fit gave σ={sigma} for source={source!r} "
            f"on {getattr(animal, 'animal_id', '?')} — too few/degenerate trials?")
    return float(sigma)


_SBI_SIGMA_PLACEHOLDER = 0.175  # midpoint of the plausible range [0.05, 0.30]


def resolve_sigma_sbi(animal: Optional['AnimalData'] = None, *,
                      model: str = 'consensus') -> float:
    """Perceptual σ for the normative target — PLACEHOLDER pending SBI.

    The normative PSE needs σ_percep = the encoding noise x ~ N(s, σ²): the
    *same* additive Gaussian the BE/SC generative models call ``sigma_noise``.
    Once the SBI networks are trained this must return the **per-animal**
    posterior point estimate of ``sigma_noise`` (from the BE or SC network per
    ``model``), fit on data that exclude the block being scored.

    Until then it ignores ``animal`` and returns the constant
    ``_SBI_SIGMA_PLACEHOLDER`` (0.175, the midpoint of [0.05, 0.30]). The
    normative PSE is nearly flat across that range (Hard-A ≈ +0.05…+0.07), so
    this is adequate for wiring and plots but is NOT a real per-animal estimate.
    Replace before drawing any inference from the target.
    """
    import warnings
    warnings.warn(
        "resolve_sigma_sbi: PLACEHOLDER σ=%.3f (SBI not wired) — replace with "
        "the per-animal SBI sigma_noise posterior." % _SBI_SIGMA_PLACEHOLDER,
        RuntimeWarning, stacklevel=2)
    return float(_SBI_SIGMA_PLACEHOLDER)


# ═════════════════════════════════════════════════════════════════════════════
# Convergence over trials — the adaptation time-course (manuscript Fig. 5)
# ═════════════════════════════════════════════════════════════════════════════
#
# After a distribution switch the animal re-adapts. We track the PSE in a
# rolling window over post-switch trials and express it as convergence toward
# the normative optimum:
#
#     Convergence(t) = [PSE(t) - PSE_pre_switch] / [PSE_normative - PSE_pre_switch]
#
# 0 = pre-switch behaviour, 1 = normative optimum. The denominator uses
# compute_normative_pse (σ-based, robust to a criterion artifact); the
# numerator baseline PSE_pre_switch is the animal's own PSE just before the
# switch. When ``normalise=False`` only the numerator is returned (raw PSE
# shift in stimulus units) — always valid, the fallback when no clean σ exists.
#
# The curve collapses to three per-animal scalars for the across-animal fold:
# plateau (where it settles), trials_to_plateau (how fast), auc (a blend).

from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402


def detect_shifts(animal: 'AnimalData') -> List[Dict]:
    """Find distribution-shift boundaries in chronological session order.

    A shift is where consecutive (non-masking) sessions differ in
    ``session.distribution``. One distribution per session is assumed;
    within-session shifts are not detected.

    Args:
        animal: AnimalData to scan.

    Returns:
        One dict per shift, chronological: ``shift_idx``, ``session_idx``
        (post-shift), ``trial_index_in_animal`` (cumulative trials at the
        switch), ``from_distribution``, ``to_distribution``.
    """
    shifts: List[Dict] = []
    sessions = sorted(animal.get_sessions(), key=lambda s: s.session_idx)
    sessions = [s for s in sessions if not s.masking]

    cumulative_trials = 0
    prev_dist = None
    for sess in sessions:
        if prev_dist is not None and sess.distribution != prev_dist:
            shifts.append({
                'shift_idx':             len(shifts),
                'session_idx':           sess.session_idx,
                'trial_index_in_animal': cumulative_trials,
                'from_distribution':     prev_dist,
                'to_distribution':       sess.distribution,
            })
        cumulative_trials += sess.n_trials
        prev_dist = sess.distribution
    return shifts


def _windowed_pse(
    stimuli: np.ndarray,
    choices: np.ndarray,
    window: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """PSE in a rolling window over ordered trials.

    Returns (centres, pses): the trial index at each window centre and the
    fitted mu there. Windows that fail to fit come back NaN. Only full windows
    are used, so the curve stops one window short of the block end rather than
    ending on a partial, noisier window.

    Retained as a utility; the standardised adaptation path below now rolls the
    lapse-corrected 'pse' stat through compute_rolling_stats instead of raw mu.
    """
    from behav_utils.analysis.psychometry import fit_psychometric
    from behav_utils.analysis.rolling import _iter_windows

    centres, pses = [], []
    for centre, sl in _iter_windows(int(stimuli.size), window, step):
        fit = fit_psychometric(stimuli[sl], choices[sl])
        mu = fit.get('mu', np.nan) if isinstance(fit, dict) else np.nan
        centres.append(centre)
        pses.append(mu if mu is not None else np.nan)
    return np.asarray(centres, dtype=float), np.asarray(pses, dtype=float)


def _collapse_curve(
    centres: np.ndarray,
    values: np.ndarray,
    plateau_frac: float = 0.5,
    plateau_target: float = 0.632,
) -> Dict[str, float]:
    """Collapse a trajectory to three scalars.

    plateau:           mean of the last ``plateau_frac`` of finite points —
                       where behaviour settles. Averaged over the tail, so it is
                       the stable summary.
    trials_to_plateau: first window centre whose value reaches
                       ``plateau_target`` × plateau (default 1 - 1/e of the
                       plateau, a time-constant-like crossing). NaN if never
                       reached. Noisier than plateau — read with care.
    auc:               mean of all finite points — a blend of level and speed.
    """
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return {'plateau': np.nan, 'trials_to_plateau': np.nan, 'auc': np.nan}

    v = values[finite]
    c = centres[finite]

    tail_n = max(1, int(np.ceil(plateau_frac * v.size)))
    plateau = float(np.mean(v[-tail_n:]))
    auc = float(np.mean(v))

    trials_to_plateau = np.nan
    if np.isfinite(plateau) and plateau != 0:
        crossing = plateau_target * plateau
        # first point at or beyond the crossing (sign-aware for negative plateaus)
        reached = (v >= crossing) if plateau > 0 else (v <= crossing)
        if reached.any():
            trials_to_plateau = float(c[np.argmax(reached)])

    return {'plateau': plateau, 'trials_to_plateau': trials_to_plateau, 'auc': auc}


def _switch_index_map(animal: 'AnimalData', distribution: str) -> Dict[str, int]:
    """{session_id: switch_index} — which run of ``distribution`` each session
    belongs to (0 = first appearance). A run of consecutive sessions in
    ``distribution`` shares one index; an ABAB chain gives 0, 1, 2, … so early-
    vs-late switches stay recoverable."""
    all_sessions = sorted(animal.get_sessions(), key=lambda s: s.session_idx)
    run, prev, switch_of = -1, None, {}
    for s in all_sessions:
        if s.distribution == distribution and prev != distribution:
            run += 1
        if s.distribution == distribution:
            switch_of[s.session_id] = run
        prev = s.distribution
    return switch_of


def _collapse_rows(centres: np.ndarray, values: Dict[str, np.ndarray],
                   stat_names: Sequence[str]) -> List[Dict]:
    """Collapse each stat's trajectory to plateau / trials_to_plateau / auc,
    as tidy rows ``{'stat': '<name>_<metric>', 'value': v}`` for the group fold."""
    c = np.asarray(centres, dtype=float)
    rows = []
    for s in stat_names:
        sc = _collapse_curve(c, np.asarray(values[s], dtype=float))
        for metric in ('plateau', 'trials_to_plateau', 'auc'):
            rows.append({'stat': f'{s}_{metric}', 'value': sc[metric]})
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# Adaptation time-course — standardised displacement from expert-uniform
# ═════════════════════════════════════════════════════════════════════════════
#
# After a distribution switch the animal re-adapts. We roll each requested stat
# over the post-switch block (compute_rolling_stats) and express it as a
# displacement from the animal's OWN expert-uniform behaviour:
#
#     value(t) - baseline[stat]          (baseline = pooled expert-uniform)
#
# 0 is expert-uniform; there is NO ratio and NO clip, so wrong-way moves and
# overshoot past the normative target stay visible. For 'pse' the normative
# optimum is returned as a reference offset (normative_pse - baseline_pse); the
# perceptual sigma comes from ``sigma_source`` (currently the SBI placeholder).


def compute_adaptation(
    animal: 'AnimalData',
    distribution: str,
    *,
    stat_names: Sequence[str] = ('pse',),
    standardised: bool = True,
    mode: str = 'pooled',
    window: int = 50,
    step: Optional[int] = None,
    baseline_preset: str = 'expert_uniform',
    baseline_last_n: int = 5,
    baseline_trials: str = 'non_opto',
    sigma_source: str = 'sbi',
    trials: str = 'all',
) -> Dict:
    """Standardised adaptation trajectories for one animal at ``distribution``.

    Rolls ``stat_names`` over the post-switch block and (when ``standardised``)
    subtracts the animal's own expert-uniform baseline, so 0 is expert-uniform
    behaviour. No ratio, no clip. The trajectory rides on ``compute_rolling_stats``
    (windows never cross a session boundary); the PSE stat is the lapse-corrected
    'pse' (raw, so early shallow windows survive).

    Args:
        animal:          AnimalData.
        distribution:    the post-switch distribution, e.g. 'Hard-A'.
        stat_names:      scalar stats to trace, e.g. ('pse',) or
                         ('pse', 'accuracy', 'side_bias').
        standardised:    subtract the expert-uniform baseline per stat (default);
                         False returns the raw rolled values.
        mode:            'pooled' (whole block → one trajectory) or 'per_session'
                         (one trajectory per session, boundary-safe).
        window, step:    rolling-window size / stride; ``step=None`` → ``window``
                         (non-overlapping bins — independent points).
        baseline_preset: select_sessions preset defining the expert-uniform
                         baseline sessions (default 'expert_uniform').
        baseline_last_n: use the last N such sessions.
        baseline_trials: 'non_opto' (default, clean) or 'all' for the baseline.
        sigma_source:    'sbi' → ``resolve_sigma_sbi`` (placeholder); otherwise a
                         ``resolve_sigma`` source string. Only used for the 'pse'
                         normative line.
        trials:          'all' (default, keeps laser-on — the cumulative effect on
                         learning is the signal) or 'non_opto' for the trajectory.

    Returns:
        ::

            {'stat_names', 'standardised', 'mode', 'window', 'step',
             'baseline':  {stat: value},         # expert-uniform 0-line, per stat
             'normative': {'pse': offset} | {},  # pse only, baseline-subtracted
             # mode='pooled':
             'curve':    {'trials', 'values': {stat: array}, 'n_trials'},
             # mode='per_session':
             'sessions': [{'session_id','session_idx','switch_index',
                           'session_type','trials','values':{stat: array},
                           'n_trials'}],
             'rows':     [{'stat': '<name>_<metric>', 'value'}...],  # for the fold
             'meta':     {...}}

        ``values`` is always keyed by stat. ``rows`` holds plateau /
        trials_to_plateau / auc per stat (per-session in 'per_session' mode, each
        tagged with session_id / switch_index); stamp animal + group at the call
        site via ``collect_rows``.

    Raises:
        ValueError: empty ``stat_names``; bad ``mode``; or the animal has no
            ``baseline_preset`` sessions (every animal must have an expert
            baseline — an experimental-design error, not a normal empty case).
    """
    from behav_utils.data.ops.selection import select_sessions
    from behav_utils.data.ops.filtering import filter_trials
    from behav_utils.analysis.rolling import compute_rolling_stats
    from behav_utils.analysis.statistics import compute_stat

    stat_names = list(stat_names)
    if not stat_names:
        raise ValueError("compute_adaptation: stat_names is empty.")
    if mode not in ('pooled', 'per_session'):
        raise ValueError(
            f"compute_adaptation: mode must be 'pooled' or 'per_session', "
            f"got {mode!r}.")
    step = window if step is None else step
    aid = getattr(animal, 'animal_id', '?')

    # ── baseline: the animal's expert-uniform behaviour, one value per stat ──
    expert = select_sessions(animal, baseline_preset)
    expert = sorted(expert, key=lambda s: s.session_idx)[-baseline_last_n:]
    if not expert:
        raise ValueError(
            f"compute_adaptation: {aid} has no {baseline_preset!r} sessions — "
            f"every animal needs an expert baseline (check the data / preset). "
            f"This is an experimental-design error, not a normal empty case.")
    expert = filter_trials(expert, trial_type=baseline_trials)
    baseline = dict(compute_stat(expert, stat_names, mode='pooled')['pooled'])

    def _standardise(values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not standardised:
            return {s: np.asarray(v, dtype=float) for s, v in values.items()}
        return {s: np.asarray(v, dtype=float) - baseline.get(s, np.nan)
                for s, v in values.items()}

    # ── normative offset (pse only) ─────────────────────────────────────────
    normative: Dict[str, float] = {}
    if 'pse' in stat_names:
        try:
            if sigma_source == 'sbi':
                sigma = resolve_sigma_sbi(animal)
            else:
                sigma = resolve_sigma(animal, source=sigma_source, trials=baseline_trials)
            pse_norm = float(compute_normative_pse(distribution, sigma))
        except Exception:
            pse_norm = float('nan')
        base_pse = baseline.get('pse', float('nan'))
        normative['pse'] = (pse_norm - base_pse) if standardised else pse_norm

    # ── target block trajectory (pre-filtered → compute_rolling_stats) ──────
    target = select_sessions(animal, distribution=distribution, exclude_masking=False)
    target = sorted(target, key=lambda s: s.session_idx)
    target = filter_trials(target, trial_type=trials)
    rolled = compute_rolling_stats(target, stat_names=stat_names, mode=mode,
                                   window=window, step=step)

    meta = {'distribution': distribution, 'window': window, 'step': step,
            'trials': trials, 'baseline_preset': baseline_preset,
            'baseline_last_n': baseline_last_n, 'baseline_trials': baseline_trials,
            'sigma_source': sigma_source, 'n_target_sessions': len(target)}

    result: Dict = {
        'stat_names': stat_names, 'standardised': standardised, 'mode': mode,
        'window': window, 'step': step,
        'baseline': baseline, 'normative': normative, 'meta': meta,
    }

    if mode == 'pooled':
        curve = rolled['curve']
        vals = _standardise(curve['values'])
        result['curve'] = {'trials': np.asarray(curve['trials'], dtype=float),
                           'values': vals, 'n_trials': curve['n_trials']}
        result['rows'] = _collapse_rows(curve['trials'], vals, stat_names)
        return result

    # per_session
    switch_of = _switch_index_map(animal, distribution)
    entries, rows = [], []
    for e in rolled['sessions']:
        vals = _standardise(e['values'])
        sid = e['session_id']
        si = switch_of.get(sid, np.nan)
        entries.append({
            'session_id': sid, 'session_idx': e['session_idx'],
            'switch_index': si, 'session_type': e['session_type'],
            'trials': np.asarray(e['trials'], dtype=float), 'values': vals,
            'n_trials': e['n_trials'],
        })
        for r in _collapse_rows(e['trials'], vals, stat_names):
            rows.append({'session_id': sid, 'switch_index': si, **r})
    result['sessions'] = entries
    result['rows'] = rows
    return result


def compute_adaptation_per_session(
    animal: 'AnimalData',
    distribution: str,
    *,
    stat_names: Sequence[str] = ('pse',),
    standardised: bool = True,
    window: int = 50,
    step: Optional[int] = None,
    baseline_preset: str = 'expert_uniform',
    baseline_last_n: int = 5,
    baseline_trials: str = 'non_opto',
    sigma_source: str = 'sbi',
    trials: str = 'all',
) -> Dict:
    """``compute_adaptation`` with ``mode='per_session'`` — one trajectory per
    session at ``distribution``, no window crossing a session boundary. The right
    shape when a hard phase is a single session (opto cohort) or when within-
    session re-adaptation matters. See :func:`compute_adaptation`."""
    return compute_adaptation(
        animal, distribution, stat_names=stat_names, standardised=standardised,
        mode='per_session', window=window, step=step,
        baseline_preset=baseline_preset, baseline_last_n=baseline_last_n,
        baseline_trials=baseline_trials, sigma_source=sigma_source, trials=trials)
