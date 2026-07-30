"""
Trial-level resampling for phase comparisons.

A *phase* is a pooled trial set — whatever ``filter_trials`` returned, treated as
one mega-session. These are the engines that turn a phase into a distribution of
statistic values, so uncertainty can be attached to any contrast built from them.

    load_experiment → select_sessions → filter_trials → phase
                                                          │
                          bootstrap_phase_stats ──────────┤
                          permute_phase_difference ───────┘
                                     ↓
                          summarise_draw_distribution

Two engines, two jobs — they are not interchangeable:

**Bootstrap** resamples trials with labels fixed. It answers "how precisely is
this quantity pinned down", and is valid wherever the trials were sampled, which
is everywhere. Use it for confidence intervals, and for any contrast between
phases that were not randomised against each other (masking sessions vs opto
sessions were recorded on different days, so no shuffle mimics the design).

**Permutation** shuffles the condition label to build a null. It answers "could
the label be irrelevant", and is only meaningful when the label was assigned in a
way the shuffle reproduces. Opto was randomised per trial by the rig, so
permuting opto/non-opto *within a phase* is well grounded. Permuting a
session-level property across trials is not.

Composition rule: bootstrap each condition once, keep the draws, and build every
contrast by subtracting draw arrays elementwise. Differences of differences then
work directly, and a shared reference cancels exactly — ``(A - C) - (B - C)``
uses the *same* ``C`` draws, so its variance drops out instead of being
double-counted.

Public API:
    bootstrap_phase_stats       — one phase → {stat: draws}
    permute_phase_difference    — two phases → {stat: draws of Δ under the null}
    summarise_draw_distribution — draws → CI and two-sided p
    pool_phase_arrays           — phase → flat arrays (shared by both engines)
    compute_stats_from_arrays   — flat arrays → {stat: value}
"""

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    'bootstrap_phase_stats',
    'permute_phase_difference',
    'summarise_draw_distribution',
    'pool_phase_arrays',
    'compute_stats_from_arrays',
]


# ─────────────────────────────────────────────────────────────────────────────
# Phase → arrays → statistic
# ─────────────────────────────────────────────────────────────────────────────

def pool_phase_arrays(phase) -> Dict[str, np.ndarray]:
    """Pool a phase into flat trial arrays, dropping non-responses.

    ``prev_*`` are frozen on the raw session by the loader and carried through
    ``pool_arrays``, so history stats stay correct on an interleaved subset: the
    previous trial is the true predecessor in the original sequence, not the
    previous row of the filtered subset.

    Args:
        phase: list of SessionData (the output of ``filter_trials``).

    Returns:
        Dict with ``stimuli``, ``choices``, ``categories`` and the ``prev_*``
        arrays, all the same length.
    """
    from behav_utils.data.ops.filtering import pool_arrays

    pooled = pool_arrays(phase)
    choices = np.asarray(pooled['choices'], dtype=float)
    keep = ~np.asarray(pooled['no_response'], dtype=bool) & ~np.isnan(choices)

    arrays = {
        'stimuli': np.asarray(pooled['stimuli'], dtype=float)[keep],
        'choices': choices[keep],
        'categories': np.asarray(pooled['categories'], dtype=float)[keep],
    }
    rt = pooled.get('reaction_times')
    if rt is not None:
        arrays['reaction_times'] = np.asarray(rt, dtype=float)[keep]
    for key in ('prev_choices', 'prev_stimuli', 'prev_categories'):
        value = pooled.get(key)
        arrays[key] = np.asarray(value, dtype=float)[keep] if value is not None else None
    return arrays


def compute_stats_from_arrays(
    arrays: Mapping[str, np.ndarray],
    stat_families: Sequence[str],
    strict: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Compute statistics from pooled arrays, flattened to scalar names.

    Args:
        arrays:        output of :func:`pool_phase_arrays`.
        stat_families: family-level names ('psychometric', 'win_stay', ...) —
                       what ``fit_summary_stats`` accepts.
        strict:        True (observed values) lets a genuine error propagate.
                       False (resampling loops) returns ``{}`` on failure so the
                       iteration can be counted as discarded.
        rng:           Generator forwarded to ``fit_summary_stats`` for stochastic
                       stats (e.g. ``reaction_time_jitter``). Pass a seeded one
                       for a reproducible observed value; ignored by every
                       deterministic stat.

    Returns:
        ``{scalar_name: value}`` with multi-value families expanded, so keys
        match ``get_stat_names_expanded`` — 'psychometric' becomes 'mu',
        'sigma', 'lapse_low', 'lapse_high'.
    """
    from behav_utils.analysis.summary_stats import fit_summary_stats

    if not stat_families:
        return {}
    try:
        raw = fit_summary_stats(
            arrays['choices'], arrays['stimuli'], arrays['categories'],
            prev_choices=arrays.get('prev_choices'),
            prev_stimuli=arrays.get('prev_stimuli'),
            prev_categories=arrays.get('prev_categories'),
            stat_names=list(stat_families), return_dict=True,
            reaction_time=arrays.get('reaction_times'), rng=rng,
        )
    except Exception:
        if strict:
            raise
        return {}

    flat: Dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if np.isscalar(sub_value):
                    flat[sub_key] = float(sub_value)
        elif np.isscalar(value):
            flat[key] = float(value)
    return flat


def _index_arrays(arrays: Mapping[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    """Index every array in a pooled dict with the same trial index."""
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in arrays.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Engines
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_phase_stats(
    phase,
    stat_families: Sequence[str],
    n_draws: int = 1000,
    n_trials: Optional[int] = None,
    seed: int = 0,
    unit: str = 'trials',
) -> Dict[str, np.ndarray]:
    """Resample one phase's trials (or sessions) and recompute its statistics.

    Delegates to :func:`behav_utils.analysis.downsample.resample_stat_vectors`,
    the library's single resample-and-recompute engine, so the frozen lag-1
    ``prev_*`` pairing survives every draw (a repeated trial carries its own
    predecessor).

    ``unit='trials'`` (default) is a per-trial bootstrap, stratified by stimulus
    bin, and answers "how precisely is this phase's stat pinned by its trials".
    ``unit='sessions'`` resamples whole sessions with replacement instead: it
    treats the session as the independent unit, so the interval reflects
    session-to-session scatter (the right unit when the phase spans sessions that
    were not randomised per trial — e.g. an opto phase vs a masking phase). A
    matched ``n_trials`` only applies to the trial bootstrap; it is ignored for
    sessions, whose natural count is the number of sessions.

    Note the trial draw is **stratified by stimulus bin**, not a plain bootstrap:
    the stimulus composition is held roughly fixed across draws. That is the
    right choice — the stimulus set is fixed by the experimental design, so the
    interval is conditional on it — but it makes intervals slightly narrower than
    an unconditional bootstrap would give. The session draw is unstratified
    (whole sessions are taken intact).

    Args:
        phase:         list of SessionData from ``filter_trials``.
        stat_families: family-level stat names. For ``unit='trials'`` these must
                       be trial-exchangeable; ``unit='sessions'`` preserves order
                       and accepts any stat.
        n_draws:       number of resamples.
        n_trials:      trial bootstrap only — draw this many trials instead of the
                       natural count (a matched n for equal-precision contrasts).
        seed:          RNG seed.
        unit:          'trials' (default) or 'sessions'.

    Returns:
        ``{scalar_name: ndarray of shape (n_draws,)}``. Failed draws are NaN.

    Raises:
        ValueError: if ``unit='trials'`` and any requested stat is not
            trial-exchangeable.
    """
    from behav_utils.analysis.downsample import resample_stat_vectors
    from behav_utils.analysis.summary_stats import get_stat_names_expanded

    families = list(stat_families)
    if not families:
        return {}
    draw_n = None if unit == 'sessions' else n_trials
    matrix = resample_stat_vectors(
        phase, families, n=draw_n, n_repeats=n_draws,
        with_replacement=True, unit=unit, seed=seed,
    )
    names = get_stat_names_expanded(families)
    return {name: matrix[:, i] for i, name in enumerate(names)}


def permute_phase_difference(
    phase_a,
    phase_b,
    stat_families: Sequence[str],
    n_draws: int = 1000,
    n_trials: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Null distribution of ``stat(a) - stat(b)`` from shuffling the labels.

    Pools both phases, reassigns the phase label at random keeping the group
    sizes fixed, and recomputes the difference — the null being "which phase a
    trial belongs to carries no information".

    Only use this where the label really was randomised per trial. Opto vs
    non-opto within a phase qualifies (the rig interleaved it). Comparing phases
    that differ by session type does not: those trials were collected on
    different days, so a shuffle would treat non-exchangeable trials as
    exchangeable and absorb every between-day difference into the null. Use
    :func:`bootstrap_phase_stats` and an interval there instead.

    Args:
        phase_a, phase_b: the two phases; the difference is ``a - b``.
        stat_families:    family-level stat names.
        n_draws:          number of shuffles. The smallest reportable p is
                          ``1 / (n_draws + 1)``.
        n_trials:         draw this many per side instead of the natural counts.
        seed:             RNG seed.

    Returns:
        ``{scalar_name: ndarray}`` of differences under the null. Length may be
        below ``n_draws`` where a shuffle produced an unfittable split; those
        iterations are dropped rather than counted as zero.
    """
    families = list(stat_families)
    if not families:
        return {}

    arrays_a = pool_phase_arrays(phase_a)
    arrays_b = pool_phase_arrays(phase_b)
    n_a = arrays_a['choices'].size
    n_b = arrays_b['choices'].size

    combined = {}
    for key, value in arrays_a.items():
        other = arrays_b.get(key)
        combined[key] = (np.concatenate([value, other])
                         if isinstance(value, np.ndarray) and isinstance(other, np.ndarray)
                         else None)

    take_a = n_trials if n_trials is not None else n_a
    take_b = n_trials if n_trials is not None else n_b
    total = n_a + n_b

    rng = np.random.default_rng(seed)
    accumulated: Dict[str, list] = {}
    for _ in range(n_draws):
        shuffled = rng.permutation(total)
        stats_a = compute_stats_from_arrays(
            _index_arrays(combined, shuffled[:take_a]), families, strict=False)
        stats_b = compute_stats_from_arrays(
            _index_arrays(combined, shuffled[take_a:take_a + take_b]), families, strict=False)
        if not stats_a or not stats_b:
            continue
        for name in stats_a:
            value_a, value_b = stats_a.get(name, np.nan), stats_b.get(name, np.nan)
            if np.isfinite(value_a) and np.isfinite(value_b):
                accumulated.setdefault(name, []).append(value_a - value_b)

    return {name: np.asarray(values, dtype=float)
            for name, values in accumulated.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Summarising a draw distribution
# ─────────────────────────────────────────────────────────────────────────────

def summarise_draw_distribution(
    draws: np.ndarray,
    observed: Optional[float] = None,
    ci: float = 0.95,
    null_value: float = 0.0,
) -> Dict[str, float]:
    """Percentile interval and two-sided p from a distribution of draws.

    Works for either engine, but the p means different things:

    * bootstrap draws (of a difference) — ``p`` is the proportion of draws on
      the far side of ``null_value``, doubled. Report ``observed`` as the
      estimate and the interval as its uncertainty.
    * permutation draws — pass ``observed`` (the unshuffled difference) and
      ``p`` becomes the proper permutation p-value, ``(1 + #{|draw| >=
      |observed|}) / (n + 1)``. The ``+1`` keeps it from ever being exactly
      zero; the floor is ``1 / (n + 1)``.

    Do not compare two intervals by eye to judge a difference. Two 95%
    intervals can overlap while the difference is significant — for independent
    estimates with equal standard error they only stop overlapping at 3.9 SE,
    while the difference is significant from 2.8 SE. Build the difference
    distribution and summarise that instead.

    Args:
        draws:      1-D array of resampled values. NaNs are dropped.
        observed:   the unshuffled statistic — required for a permutation p.
        ci:         interval mass, e.g. 0.95.
        null_value: value corresponding to "no effect".

    Returns:
        ``{'ci_lo', 'ci_hi', 'p_two_sided', 'n_draws', 'median'}``, all NaN if
        there are fewer than 10 usable draws.
    """
    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    empty = {'ci_lo': np.nan, 'ci_hi': np.nan, 'p_two_sided': np.nan,
             'n_draws': int(values.size), 'median': np.nan}
    if values.size < 10:
        return empty

    tail = (1.0 - ci) / 2.0 * 100.0
    ci_lo = float(np.percentile(values, tail))
    ci_hi = float(np.percentile(values, 100.0 - tail))

    if observed is not None and np.isfinite(observed):
        # Permutation: how often does the null reach the observed magnitude?
        p = float((1 + np.sum(np.abs(values - null_value) >= abs(observed - null_value)))
                  / (values.size + 1))
    else:
        # Bootstrap: how much of the distribution sits across the null? The
        # (1 + count) / (n + 1) form matches the permutation branch and keeps a
        # finite set of draws from ever reporting p = 0 — the floor is
        # 2 / (n_draws + 1), which is a property of how many draws were taken,
        # not evidence. Raise n_draws to lower it.
        below = 1 + int(np.sum(values <= null_value))
        above = 1 + int(np.sum(values >= null_value))
        p = float(min(1.0, 2.0 * min(below, above) / (values.size + 1)))

    return {'ci_lo': ci_lo, 'ci_hi': ci_hi, 'p_two_sided': p,
            'n_draws': int(values.size), 'median': float(np.median(values))}
