"""
compute_stat — the single entry point for scalar statistics on a phase.

    load_experiment → select_sessions → filter_trials → phase
                                                          ↓
                                     compute_stat(phase, stat_names, mode=...)

One door for "give me statistic x for these sessions", at either granularity.
The return shape is the same regardless of ``mode`` — only how much of it is
populated changes — so nothing downstream has to branch on the shape:

    result = compute_stat(phase, ['accuracy', 'side_bias', 'mu'])
    result['pooled']['side_bias']            # trial-weighted pooled estimate
    result['sessions']                       # None (mode='pooled')

    result = compute_stat(phase, ['accuracy', 'side_bias'], mode='per_session')
    result['pooled']['side_bias']            # STILL the pooled estimate
    result['sessions']                       # tidy DataFrame, one row per session×stat

The key invariant: ``pooled`` is ALWAYS the genuine pooled fit — one fit on all
trials, trial-weighted — never the mean of the per-session values. Those two
differ, sometimes a lot: a short, atypical session gets equal weight in a
session-mean but only its trial-share of weight when pooled. ``mode`` decides
whether the per-session breakdown is also computed; it never changes what
``pooled`` means.

Scope: this is the door for scalar statistics — the numbers you test and carry
to the across-animal comparison. The psychometric CURVE and the update MATRIX
are objects that render rather than tabulate, so they keep their own doors
(``compute_psychometric``, ``compute_um``). The psychometric's *parameters*
(mu, sigma, lapse_low, lapse_high) are scalars and do come back here, via the
``'psychometric'`` family.

For a session-level spread (e.g. an error bar on the eyeball plot), derive it
from ``sessions`` — ``result['sessions'].groupby('stat')['value'].agg(['mean',
'std'])`` — where it is clearly labelled as session-to-session variability, not
confused with the pooled estimate's sampling uncertainty.

Public API:
    compute_stat — scalar statistics on a phase, pooled and/or per session
"""

from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import numpy as np

__all__ = ['compute_stat']


def compute_stat(
    phase,
    stat_names: Optional[Sequence[str]] = None,
    mode: str = 'pooled',
    animal_id: Optional[str] = None,
) -> Dict:
    """Scalar statistics for a phase, pooled and (optionally) per session.

    Args:
        phase:      list of SessionData — the output of ``filter_trials``
                    (a condition), or of ``select_sessions`` (unfiltered).
        stat_names: family-level names ('accuracy', 'side_bias', 'win_stay',
                    'psychometric', ...). ``'psychometric'`` expands to mu,
                    sigma, lapse_low, lapse_high. Defaults to a standard set.
                    Must be scalar-valued families; the curve and matrix
                    objects are not returned here (see module docstring).
        mode:       'pooled' computes only the pooled fit and leaves
                    ``sessions`` None. 'per_session' additionally computes one
                    fit per session and fills ``sessions``. ``pooled`` is the
                    genuine pooled fit in both cases.
        animal_id:  stamped into the per-session frame; read from the first
                    session if not given.

    Returns:
        ::

            {
                'pooled':   {stat: value},         # always the pooled fit
                'sessions': DataFrame | None,      # per-session rows, or None
                'meta':     {'mode', 'n_sessions', 'n_trials', 'stat_names'},
            }

        The per-session frame has columns ``animal, session, stat, value,
        n_trials`` — the same tidy shape the group fold and the eyeball plotter
        consume, so a per-animal row for the across-animal test is a filtered
        slice of it (no conversion).

    Raises:
        ValueError: if ``mode`` is not 'pooled' or 'per_session'.
    """
    if mode not in ('pooled', 'per_session'):
        raise ValueError(f"compute_stat: mode must be 'pooled' or 'per_session', got {mode!r}")

    from behav_utils.analysis.summary_stats import get_stat_names_expanded

    if stat_names is None:
        stat_names = ['accuracy', 'side_bias', 'recency', 'win_stay', 'lose_shift']

    # The caller may pass either family names ('psychometric') or expanded
    # scalar names ('mu', 'sigma'). The engine only accepts families, so map
    # any expanded name back to the family that produces it.
    families = _resolve_families(list(stat_names))
    expanded = list(get_stat_names_expanded(families))

    if animal_id is None:
        animal_id = _infer_animal_id(phase)

    # ── pooled: always the genuine pooled fit ─────────────────────────────
    pooled = _fit_flat(phase, families)

    n_trials = _count_trials(phase)
    meta = {
        'mode': mode,
        'n_sessions': int(len(phase)),
        'n_trials': int(n_trials),
        'stat_names': expanded,
    }

    # ── per session: only when asked ──────────────────────────────────────
    sessions_frame = None
    if mode == 'per_session':
        sessions_frame = _per_session_frame(phase, families, expanded, animal_id)

    return {'pooled': pooled, 'sessions': sessions_frame, 'meta': meta}


# ── helpers ─────────────────────────────────────────────────────────────────

def _resolve_families(names: List[str]) -> List[str]:
    """Map any names (family or expanded scalar) to the families to compute.

    ``get_stat_names_expanded(['psychometric'])`` → ['mu','sigma',...], so we
    invert: build family→children once, and for each requested name keep it if
    it is already a family, else swap in the family whose expansion contains it.
    Order preserved, duplicates dropped.
    """
    from behav_utils.analysis.summary_stats import (
        list_available_stats, get_stat_names_expanded,
    )

    families = list_available_stats()
    child_to_family = {}
    for fam in families:
        for child in get_stat_names_expanded([fam]):
            child_to_family.setdefault(child, fam)

    resolved: List[str] = []
    for name in names:
        fam = name if name in families else child_to_family.get(name)
        if fam is None:
            raise ValueError(
                f"compute_stat: unknown stat {name!r}. "
                f"Families: {families}"
            )
        if fam not in resolved:
            resolved.append(fam)
    return resolved


def _infer_animal_id(phase) -> str:
    for session in phase:
        meta = getattr(session, 'filter_info', None) or {}
        selection = meta.get('selection', {}) if isinstance(meta, dict) else {}
        if selection.get('animal_id'):
            return selection['animal_id']
        aid = getattr(session, 'animal_id', None)
        if aid:
            return aid
    return 'unknown'


def _count_trials(phase) -> int:
    total = 0
    for session in phase:
        stimulus = getattr(session.trials, 'stimulus', None)
        if stimulus is not None:
            total += len(stimulus)
    return total


def _fit_flat(sessions, families: List[str]) -> Dict[str, float]:
    """Pooled fit → flat {scalar_name: value}, reusing the resampling helper."""
    from behav_utils.analysis.resampling import pool_phase_arrays, compute_stats_from_arrays

    if not sessions:
        return {}
    arrays = pool_phase_arrays(sessions)
    return compute_stats_from_arrays(arrays, families, strict=True)


def _per_session_frame(sessions, families, expanded, animal_id):
    """One fit per session → tidy long DataFrame."""
    import pandas as pd

    rows = []
    for session in sessions:
        values = _fit_flat([session], families)
        stimulus = getattr(session.trials, 'stimulus', None)
        n_trials = len(stimulus) if stimulus is not None else 0
        session_id = getattr(session, 'session_idx', getattr(session, 'session_id', None))
        for stat in expanded:
            rows.append({
                'animal': animal_id,
                'session': session_id,
                'stat': stat,
                'value': values.get(stat, np.nan),
                'n_trials': n_trials,
            })
    return pd.DataFrame(rows, columns=['animal', 'session', 'stat', 'value', 'n_trials'])
