"""Distribution switches in an ordered session list."""

from __future__ import annotations

from typing import Dict, List, Sequence


def find_switches(sessions: Sequence) -> List[Dict]:
    """Boundaries where consecutive sessions differ in ``session.distribution``.

    Pass an already-selected, chronologically ordered list (filter out the
    session types you don't want first). One distribution per session is
    assumed. Returns one dict per switch: ``switch_idx``, ``session_idx`` (the
    post-switch session), ``order`` (position in the input), ``trial_index``
    (cumulative trials at the boundary), ``from_distribution``, ``to_distribution``.
    """
    out: List[Dict] = []
    cumulative = 0
    prev = None
    for k, s in enumerate(sessions):
        dist = getattr(s, 'distribution', None)
        if prev is not None and dist != prev:
            out.append({'switch_idx': len(out), 'session_idx': getattr(s, 'session_idx', k), 'order': k,
                        'trial_index': cumulative, 'from_distribution': prev, 'to_distribution': dist})
        cumulative += len(s.trials.choice)
        prev = dist
    return out
