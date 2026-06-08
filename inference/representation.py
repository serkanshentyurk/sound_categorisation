"""Summary-stat vector builder shared by the SBI simulator (training) and the
amortised network (conditioning on real data).

``to_stat_vector`` turns a list of SessionData into the fixed-length observation
``x`` the SBI network sees. Because the simulator and the conditioning step both
call this one function, the train and test representations cannot diverge (same
pooling, same NaN handling, same ordering) -- this is the guarantee that closes
the seam-bridging / NaN-mismatch class of bug.

Two modes:
    'pooled'  : pool all sessions (block-aware) -> one stat vector, shape (D,).
    'moments' : per-session stat vectors (N x D) -> 4 moments per feature,
                shape (4D,). Requires N >= 4 sessions.

Composes existing primitives only (``pool_arrays``, ``fit_summary_stats``); no
stat or pooling logic is reimplemented here.
"""

from typing import List, Optional, Sequence

import numpy as np
from scipy.stats import skew, kurtosis

from behav_utils.data.ops.filtering import pool_arrays
from behav_utils.analysis.summary_stats import fit_summary_stats


def to_stat_vector(
    sessions: List,
    mode: str = 'pooled',
    stat_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Build the SBI observation vector from sessions.

    Args:
        sessions: List[SessionData], already filtered (aborts dropped upstream
            via filter_trials).
        mode: 'pooled' or 'moments'.
        stat_names: Stats to compute. Defaults to SBI_STATS.

    Returns:
        'pooled'  -> shape (D,).
        'moments' -> shape (4*D,)  [mean, var, skew, kurtosis stacked].

    Raises:
        ValueError: unknown mode, or mode='moments' with fewer than 4 sessions.
    """
    if stat_names is None:
        from inference.constants import SBI_STATS
        stat_names = list(SBI_STATS)
    stat_names = list(stat_names)

    if mode == 'pooled':
        pooled = pool_arrays(sessions)
        return fit_summary_stats(
            pooled['choices'], pooled['stimuli'], pooled['categories'],
            prev_choices=pooled.get('prev_choice'),
            prev_stimuli=pooled.get('prev_stimulus'),
            prev_categories=pooled.get('prev_category'),
            stat_names=stat_names, return_dict=False,
        )

    if mode == 'moments':
        n = len(sessions)
        if n < 4:
            raise ValueError(
                f"mode='moments' needs >= 4 sessions to form 4 moments; got {n}")
        rows = []
        for s in sessions:
            a = s.get_arrays()
            rows.append(fit_summary_stats(
                a['choices'], a['stimuli'], a['categories'],
                prev_choices=a.get('prev_choice'),
                prev_stimuli=a.get('prev_stimulus'),
                prev_categories=a.get('prev_category'),
                stat_names=stat_names, return_dict=False,
            ))
        X = np.vstack(rows)                       # (N, D)
        return _nan_moments(X)

    raise ValueError(f"Unknown mode: {mode!r} (expected 'pooled' or 'moments')")


def _nan_moments(X: np.ndarray) -> np.ndarray:
    """Per-feature (column) moments across sessions, NaN-aware.

    Returns (4*D,) = [mean(D), var(D), skew(D), kurtosis(D)], computed ignoring
    NaNs. Raises if any feature has < 4 finite values across sessions (a moment
    would be undefined or silently degenerate -- fail loudly rather than impute).
    """
    X = np.asarray(X, dtype=float)
    finite = np.sum(np.isfinite(X), axis=0)       # per-column count
    bad = np.where(finite < 4)[0]
    if bad.size:
        raise ValueError(
            f"Columns {bad.tolist()} have < 4 finite values across sessions; "
            f"cannot form moments. Choose stats that stay finite at this "
            f"session length, or use mode='pooled'.")
    mean = np.nanmean(X, axis=0)
    var = np.nanvar(X, axis=0)
    sk = np.asarray(skew(X, axis=0, nan_policy='omit'), dtype=float)
    ku = np.asarray(kurtosis(X, axis=0, nan_policy='omit'), dtype=float)
    return np.concatenate([mean, var, sk, ku])
