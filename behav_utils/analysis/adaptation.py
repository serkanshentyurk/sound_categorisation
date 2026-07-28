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


def _post_switch_arrays(
    animal: 'AnimalData',
    to_distribution: str,
    trials: str = 'non_opto',
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool trials of the post-switch block into (stimuli, choices), in order.

    Trials stay in acquisition order — the windowing depends on it. Aborts and
    non-responses are dropped; the opto/masked trials are dropped when
    ``trials='non_opto'`` so the adaptation curve is artifact-free.
    """
    from behav_utils.data.ops.selection import select_sessions
    from behav_utils.data.ops.filtering import filter_trials
    from behav_utils.analysis.resampling import pool_phase_arrays

    key = to_distribution
    sessions = select_sessions(animal, distribution=key)
    sessions = [s for s in sessions if not s.masking]
    sessions = sorted(sessions, key=lambda s: s.session_idx)
    if not sessions:
        return np.array([]), np.array([])

    condition = filter_trials(sessions, trial_type=('non_opto' if trials == 'non_opto' else 'all'))
    arrays = pool_phase_arrays(condition)
    return arrays['stimuli'], arrays['choices']


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
    """
    from behav_utils.analysis.psychometry import fit_psychometric

    n = stimuli.size
    if n < window:
        return np.array([]), np.array([])

    centres, pses = [], []
    for start in range(0, n - window + 1, step):
        s = stimuli[start:start + window]
        c = choices[start:start + window]
        fit = fit_psychometric(s, c)
        mu = fit.get('mu', np.nan) if isinstance(fit, dict) else np.nan
        centres.append(start + window / 2.0)
        pses.append(mu if mu is not None else np.nan)
    return np.asarray(centres, dtype=float), np.asarray(pses, dtype=float)


def _convergence_from_pse(
    pses: np.ndarray,
    pse_pre_switch: float,
    pse_normative: Optional[float],
    normalise: bool,
) -> np.ndarray:
    """Map a PSE curve to convergence (or the raw numerator if not normalising)."""
    numerator = pses - pse_pre_switch
    if not normalise:
        return numerator
    denom = pse_normative - pse_pre_switch
    if denom is None or not np.isfinite(denom) or abs(denom) < 1e-9:
        return np.full_like(pses, np.nan)
    return numerator / denom


def _collapse_curve(
    centres: np.ndarray,
    values: np.ndarray,
    plateau_frac: float = 0.5,
    plateau_target: float = 0.632,
) -> Dict[str, float]:
    """Collapse a convergence curve to three scalars.

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




def _pre_switch_baseline(animal, to_distribution, window, trials):
    """Fit PSE on the last window of the block PRECEDING the switch into
    ``to_distribution``. Returns NaN if there is no such block."""
    from behav_utils.analysis.psychometry import fit_psychometric
    from behav_utils.analysis.resampling import pool_phase_arrays
    from behav_utils.data.ops.selection import select_sessions
    from behav_utils.data.ops.filtering import filter_trials

    shifts = detect_shifts(animal)
    match = [s for s in shifts if s['to_distribution'] == to_distribution]
    if not match:
        return float('nan')
    from_dist = match[0]['from_distribution']

    sessions = [s for s in select_sessions(animal, distribution=from_dist) if not s.masking]
    if not sessions:
        return float('nan')
    sessions = sorted(sessions, key=lambda s: s.session_idx)
    condition = filter_trials(sessions, trial_type=('non_opto' if trials == 'non_opto' else 'all'))
    arrays = pool_phase_arrays(condition)
    stim, ch = arrays['stimuli'], arrays['choices']
    if stim.size < window:
        return float('nan')
    fit = fit_psychometric(stim[-window:], ch[-window:])
    return float(fit.get('mu', float('nan'))) if isinstance(fit, dict) else float('nan')


def compute_adaptation(
    animal: 'AnimalData',
    to_distribution: str,
    sigma_percep: Optional[float] = None,
    normalise: bool = True,
    window: int = 50,
    step: int = 10,
    trials: str = 'non_opto',
    baseline: str = 'first_window',
    n_bootstrap: int = 0,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict:
    """Convergence over post-switch trials, with three collapsed scalars.

    Args:
        animal:          AnimalData.
        to_distribution: the post-switch distribution, e.g. 'Hard-A'. The
                         adaptation is measured over this block.
        sigma_percep:    perceptual σ for the normative denominator. Required
                         when ``normalise=True`` — obtain it via
                         :func:`resolve_sigma`. Ignored when not normalising.
        normalise:       True gives convergence in [0, 1] units (needs σ); False
                         gives the raw PSE shift (PSE(t) − PSE_pre_switch), which
                         needs no σ and is the always-valid fallback.
        window:          rolling-window size in trials (manuscript: 50).
        step:            step between window centres (manuscript figures dense;
                         10 is a good default). Overlapping windows smooth the
                         curve but do NOT add independent information — see the
                         uncertainty note.
        trials:          'non_opto' (artifact-free, default) or 'all'.
        baseline:        how PSE_pre_switch (the convergence origin) is set —
                         'first_window' uses the first window of the post-switch
                         block (curve starts at 0 by construction; needs no
                         cross-block bookkeeping, but "0" means start-of-block
                         not uniform behaviour). 'pre_switch' fits the last
                         window of the preceding block instead — closer to the
                         manuscript's PSE_pre_switch, at the cost of assuming a
                         clean preceding block. Default 'first_window'.
        n_bootstrap:     if > 0, resample TRIALS this many times and recompute
                         the whole curve per draw, to get a CI band and an
                         honest interval on the scalars. The uncertainty comes
                         from trial resampling, never from the overlapping
                         window points (those are ~80% shared and would give a
                         spuriously tight interval).
        ci:              band mass.
        seed:            RNG seed.

    Returns:
        ::

            {
              'curve':   {'trials', 'values', 'ci_lo', 'ci_hi',
                          'pse_pre_switch', 'pse_normative'},
              'scalars': [ {stat: 'plateau',           value, ci_lo, ci_hi},
                           {stat: 'trials_to_plateau', value, ci_lo, ci_hi},
                           {stat: 'auc',               value, ci_lo, ci_hi} ],
              'meta':    {to_distribution, normalise, window, step, trials,
                          sigma_percep, n_trials, n_windows},
            }

        ``scalars`` is the shared per-animal row shape (add animal/genotype at
        the call site) — so the across-animal fold is a concat + rank_test, and
        ``plateau`` arrives shaped exactly like any other stat.

    Raises:
        ValueError: if ``normalise=True`` without ``sigma_percep``, or the
            post-switch block has fewer than ``window`` trials.
    """
    if normalise and sigma_percep is None:
        raise ValueError(
            "compute_adaptation: normalise=True needs sigma_percep — call "
            "resolve_sigma(animal, ...). Or set normalise=False for the raw "
            "PSE shift, which needs no σ.")

    stimuli, choices = _post_switch_arrays(animal, to_distribution, trials=trials)
    if stimuli.size < window:
        raise ValueError(
            f"compute_adaptation: post-switch block for {to_distribution!r} has "
            f"{stimuli.size} trials, need at least window={window}.")

    # Baseline PSE (the convergence origin). Two definitions, both fit on a
    # single window so they are directly comparable to the rolling PSE:
    #   'first_window' — first window of THIS block; curve starts at 0.
    #   'pre_switch'   — last window of the PRECEDING block (manuscript-style).
    from behav_utils.analysis.psychometry import fit_psychometric

    if baseline == 'pre_switch':
        pse_pre_switch = _pre_switch_baseline(animal, to_distribution, window, trials)
        if not np.isfinite(pse_pre_switch):
            # no usable preceding block — fall back to first-window
            fit = fit_psychometric(stimuli[:window], choices[:window])
            pse_pre_switch = float(fit.get('mu', 0.0)) if isinstance(fit, dict) else 0.0
    elif baseline == 'first_window':
        fit = fit_psychometric(stimuli[:window], choices[:window])
        pse_pre_switch = float(fit.get('mu', 0.0)) if isinstance(fit, dict) else 0.0
    else:
        raise ValueError(
            f"compute_adaptation: baseline must be 'first_window' or 'pre_switch', got {baseline!r}")

    pse_normative = (compute_normative_pse(to_distribution, sigma_percep)
                     if normalise else None)

    centres, pses = _windowed_pse(stimuli, choices, window, step)
    values = _convergence_from_pse(pses, pse_pre_switch, pse_normative, normalise)
    scalars = _collapse_curve(centres, values)

    # ── uncertainty: resample TRIALS, recompute the whole curve ───────────
    ci_lo = ci_hi = None
    scalar_ci = {k: (np.nan, np.nan) for k in scalars}
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        n = stimuli.size
        curve_draws = np.full((n_bootstrap, centres.size), np.nan)
        scalar_draws = {k: [] for k in scalars}
        for b in range(n_bootstrap):
            idx = np.sort(rng.integers(0, n, size=n))   # resample trials, keep order
            bs, bc = stimuli[idx], choices[idx]
            _, bp = _windowed_pse(bs, bc, window, step)
            # baseline recomputed per draw for a coherent numerator
            bf = fit_psychometric(bs[:window], bc[:window])
            bpre = float(bf.get('mu', 0.0)) if isinstance(bf, dict) else 0.0
            bv = _convergence_from_pse(bp, bpre, pse_normative, normalise)
            if bv.size == curve_draws.shape[1]:
                curve_draws[b] = bv
            bsc = _collapse_curve(centres, bv)
            for k in scalars:
                scalar_draws[k].append(bsc[k])
        tail = (1 - ci) / 2 * 100
        ci_lo = np.nanpercentile(curve_draws, tail, axis=0)
        ci_hi = np.nanpercentile(curve_draws, 100 - tail, axis=0)
        for k in scalars:
            arr = np.asarray(scalar_draws[k], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size >= 10:
                scalar_ci[k] = (float(np.percentile(arr, tail)),
                                float(np.percentile(arr, 100 - tail)))

    return {
        'curve': {
            'trials': centres,
            'values': values,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'pse_pre_switch': pse_pre_switch,
            'pse_normative': pse_normative,
        },
        'scalars': [
            {'stat': k, 'value': scalars[k],
             'ci_lo': scalar_ci[k][0], 'ci_hi': scalar_ci[k][1]}
            for k in ('plateau', 'trials_to_plateau', 'auc')
        ],
        'meta': {
            'to_distribution': to_distribution, 'normalise': normalise,
            'window': window, 'step': step, 'trials': trials,
            'baseline': baseline,
            'sigma_percep': sigma_percep, 'n_trials': int(stimuli.size),
            'n_windows': int(centres.size),
        },
    }
