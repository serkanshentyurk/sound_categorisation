"""
Rolling summary statistics over ordered trials.

    compute_rolling_stats(sessions, stat_names=[...], mode='per_session'|'pooled')

The general windowed-stat computer: it slides a fixed window along trials in
acquisition order and fits summary statistics in each window. This is the
generic 2-AFC tool — continuous stimulus, binary choice, optional opto — and it
knows nothing about distribution names, session types, or normative models.

Filtering (including opto / non-opto selection) is a prior step: pass sessions
that have already been through ``filter_trials``, per the pipeline

    load -> select_sessions -> filter_trials -> compute_rolling_stats -> plot

so this function never re-filters. The session_type label rides along on each
per-session entry as neutral metadata; assigning it meaning (opto vs masking,
genotype, …) is the caller's job in the plot.

μ is fitted through the summary-stat registry (``fit_summary_stats``), so
windows the registry judges unreliable — the curve too flat (σ above threshold)
or the PSE run to the stimulus edge (|μ| > 0.99) — come back NaN. If you want
the *raw* PSE trajectory that keeps those early, still-shallow windows (the
adaptation curve), use ``compute_adaptation`` / ``compute_adaptation_per_session``
instead; those deliberately fit raw μ.

Windowing is full-windows-only: a window is emitted only when ``window`` trials
are available, so a curve stops one window short of the block end rather than
ending on a shorter, noisier window. A session with fewer than ``window`` but at
least ``min_short`` trials contributes a single whole-session point (matching
``compute_adaptation_per_session``); a shorter session contributes an empty
curve, so it still appears (annotate the empty panel) rather than vanishing.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np


def _iter_windows(n: int, window: int, step: int) -> List[Tuple[float, slice]]:
    """Full windows over ``n`` ordered trials: ``[(centre, slice), ...]``.

    Only complete ``window``-length windows are returned (starts run
    ``0, step, 2*step, …`` up to ``n - window``), each labelled by its centre
    index ``start + window/2``. Empty when ``n < window``.

    This is the shared window-index arithmetic; callers decide what to compute
    in each slice. ``adaptation._windowed_pse`` uses it for raw PSE; the rolling
    tools below use it for registry stats.
    """
    return [(start + window / 2.0, slice(start, start + window))
            for start in range(0, n - window + 1, step)]


def _family_of(stat: str) -> Optional[str]:
    """Registry family that emits scalar ``stat``, or ``None`` if ``stat`` is
    not a scalar summary stat (unknown name, or an array-valued family name).

    e.g. ``'mu' -> 'psychometric'``, ``'accuracy' -> 'accuracy'``,
    ``'binned_acc_0' -> 'binned_accuracy'``, ``'nonsense' -> None``,
    ``'binned_accuracy' -> None`` (the family name returns an array of bins, not
    a single scalar, so it cannot be a rolling curve).
    """
    from behav_utils.analysis.summary_stats import (
        SUMMARY_REGISTRY, get_stat_names_expanded)
    for family in SUMMARY_REGISTRY:
        try:
            if stat in get_stat_names_expanded([family]):
                return family
        except Exception:
            continue
    return None


def _validate_stats(stat_names) -> List[str]:
    """Normalise to a list and reject anything that is not a scalar stat."""
    if isinstance(stat_names, str):
        stat_names = [stat_names]
    stat_names = list(stat_names)
    if not stat_names:
        raise ValueError("compute_rolling_stats: stat_names is empty.")
    bad = [s for s in stat_names if _family_of(s) is None]
    if bad:
        raise ValueError(
            f"compute_rolling_stats: {bad} are not scalar summary stats; a "
            f"rolling curve needs one value per window. Use e.g. 'mu', "
            f"'accuracy', 'side_bias', 'hard_accuracy', 'win_stay'.")
    return stat_names


def _roll_arrays(arrays: Dict[str, np.ndarray], stat_names: List[str],
                 window: int, step: int, min_short: int
                 ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Rolling ``stat_names`` over one ordered block of trials.

    Returns ``(centres, {stat: values})``. One ``fit_summary_stats`` call per
    window covers every requested family; a value is NaN where its window fails
    to fit (including the registry's own reliability guard). Full windows only;
    a block of ``min_short..window-1`` trials yields one whole-block window
    centred at ``n/2``; a block shorter than ``min_short`` yields empty arrays.
    """
    from behav_utils.analysis.summary_stats import fit_summary_stats
    from behav_utils.data.structures import _flatten_stats_dict

    ch = np.asarray(arrays['choices'])
    st = np.asarray(arrays['stimuli'])
    ca = np.asarray(arrays['categories'])
    pc, ps, pca = (arrays.get('prev_choices'), arrays.get('prev_stimuli'),
                   arrays.get('prev_categories'))
    n = int(ch.size)

    if n >= window:
        windows = _iter_windows(n, window, step)
    elif n >= min_short:
        windows = [(n / 2.0, slice(0, n))]
    else:
        windows = []

    # De-duplicated families → one fit per window regardless of how many stats.
    families = list(dict.fromkeys(_family_of(s) for s in stat_names))
    centres: List[float] = []
    cols: Dict[str, List[float]] = {s: [] for s in stat_names}
    for centre, sl in windows:
        try:
            d = fit_summary_stats(
                ch[sl], st[sl], ca[sl],
                prev_choices=None if pc is None else np.asarray(pc)[sl],
                prev_stimuli=None if ps is None else np.asarray(ps)[sl],
                prev_categories=None if pca is None else np.asarray(pca)[sl],
                stat_names=families, return_dict=True)
            # 'psychometric' returns nested {'psychometric': {'mu': …}}; flatten
            # so leaf stats (mu, sigma, lapse_*) resolve by name.
            d = _flatten_stats_dict(d)
        except Exception:
            d = {}
        centres.append(centre)
        for s in stat_names:
            try:
                cols[s].append(float(d.get(s, np.nan)))
            except (TypeError, ValueError):
                # e.g. an array slipped through — treat as a failed window.
                cols[s].append(np.nan)
    return (np.asarray(centres, float),
            {s: np.asarray(v, float) for s, v in cols.items()})


def compute_rolling_stats(
    sessions,
    *,
    stat_names,
    mode: str = 'per_session',
    window: int = 50,
    step: int = 10,
    min_short: int = 10,
) -> Dict:
    """Rolling summary statistics over pre-filtered sessions.

    Slides a ``window``-trial window (stride ``step``) along trials in
    acquisition order and fits ``stat_names`` in each window. See the module
    docstring for the filtering contract, the full-windows-only rule, and why μ
    is guarded here (use ``compute_adaptation*`` for raw μ).

    Args:
        sessions:   Pre-filtered ``SessionData`` (or anything exposing
                    ``get_arrays()`` plus ``session_id`` / ``session_idx`` /
                    ``session_type``). ``filter_trials`` must have run already —
                    this function does not filter.
        stat_names: Scalar stat name(s); str or list. Each must resolve to a
                    single value per window (``'mu'``, ``'accuracy'``,
                    ``'side_bias'``, ``'hard_accuracy'``, ``'win_stay'``, …).
                    Array-valued families (e.g. ``'binned_accuracy'``) are
                    rejected — a rolling curve needs one number per window.
        mode:       ``'per_session'`` windows within each session, no window
                    crossing a boundary → one trajectory per session.
                    ``'pooled'`` concatenates all sessions in order and windows
                    across the whole run (windows may cross boundaries) → one
                    trajectory.
        window:     Window size in trials.
        step:       Stride between window centres. Overlapping windows smooth
                    the curve but do NOT add independent information.
        min_short:  A session with ``min_short..window-1`` trials still gets one
                    whole-session point (centred at ``n/2``); below this it gets
                    an empty curve.

    Returns:
        ``per_session``::

            {'stat_names', 'mode': 'per_session', 'window', 'step',
             'sessions': [ {'session_id', 'session_idx', 'session_type',
                            'distribution', 'trials': centres,
                            'values': {stat: array}, 'n_trials'}, ... ],
             'meta': {...}}

        ``pooled``::

            {'stat_names', 'mode': 'pooled', 'window', 'step',
             'curve': {'trials': centres, 'values': {stat: array}, 'n_trials'},
             'meta': {...}}

        ``values`` is always a dict keyed by stat (a length-``len(trials)`` array
        each), so single- and multi-stat calls have the same shape.

    Raises:
        ValueError: ``stat_names`` empty or containing a non-scalar / unknown
            stat, or ``mode`` not one of ``'per_session'`` / ``'pooled'``.

    Never raises on per-session *data* failures: a session whose arrays cannot
    be read emits an empty curve and a ``RuntimeWarning``, so a cohort loop
    survives one bad animal.
    """
    stat_names = _validate_stats(stat_names)
    if mode not in ('per_session', 'pooled'):
        raise ValueError(
            f"compute_rolling_stats: mode must be 'per_session' or 'pooled', "
            f"got {mode!r}.")

    sessions = list(sessions)
    meta = {'window': window, 'step': step, 'min_short': min_short,
            'n_sessions': len(sessions)}

    if mode == 'pooled':
        from behav_utils.data.ops.filtering import pool_arrays
        if not sessions:
            centres = np.array([], float)
            cols = {s: np.array([], float) for s in stat_names}
            n = 0
        else:
            arrays = pool_arrays(sessions)
            centres, cols = _roll_arrays(arrays, stat_names, window, step, min_short)
            n = int(np.asarray(arrays['choices']).size)
        return {
            'stat_names': stat_names, 'mode': 'pooled',
            'window': window, 'step': step,
            'curve': {'trials': centres, 'values': cols, 'n_trials': n},
            'meta': meta,
        }

    # per_session
    entries = []
    for s in sessions:
        sid = getattr(s, 'session_id', '?')
        try:
            dist = s.distribution
        except Exception:
            dist = None
        entry = {
            'session_id': sid,
            'session_idx': getattr(s, 'session_idx', None),
            'session_type': getattr(s, 'session_type', ''),
            'distribution': dist,
            'trials': np.array([], float),
            'values': {sn: np.array([], float) for sn in stat_names},
            'n_trials': 0,
        }
        try:
            arrays = s.get_arrays()
            centres, cols = _roll_arrays(arrays, stat_names, window, step, min_short)
            entry['trials'] = centres
            entry['values'] = cols
            entry['n_trials'] = int(np.asarray(arrays['choices']).size)
        except Exception as exc:
            warnings.warn(
                f"compute_rolling_stats: session {sid!r} failed ({exc}); "
                f"emitting empty curve.", RuntimeWarning)
        entries.append(entry)

    return {
        'stat_names': stat_names, 'mode': 'per_session',
        'window': window, 'step': step,
        'sessions': entries,
        'meta': meta,
    }
