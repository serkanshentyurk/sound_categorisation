"""
analysis/adaptation.py — Distribution-shift boundary detection.

One primitive — find when an animal's stimulus distribution changes.
The pre vs post comparison is composed in the notebook using
behav_utils.analysis.comparison.compute_comparison.

Example workflow:

    from analysis.adaptation import detect_shifts
    from behav_utils.analysis.comparison import compute_comparison
    from behav_utils.plotting.comparison import plot_comparison

    shifts = detect_shifts(animal)
    for shift in shifts:
        pre  = [s for s in animal.sessions
                if s.session_idx <  shift['session_idx']
                and s.distribution == shift['from_distribution']]
        post = [s for s in animal.sessions
                if s.session_idx >= shift['session_idx']
                and s.distribution == shift['to_distribution']]
        result = compute_comparison(pre, post,
                                     label_a='pre', label_b='post',
                                     n_bootstrap=1000)
        plot_comparison(result)
"""

from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from behav_utils.data.structures import AnimalData


def detect_shifts(animal: 'AnimalData') -> List[Dict]:
    """
    Find distribution-shift boundaries in chronological session order.

    A shift is detected when consecutive sessions have different
    ``session.distribution`` values. Within-session shifts are not
    detected — the experiment is assumed to keep one distribution per
    session. Masking sessions are skipped (their distribution attribute
    may or may not be meaningful depending on what was loaded).

    Args:
        animal: AnimalData with sessions to scan.

    Returns:
        List of dicts, one per shift, ordered chronologically:
            'shift_idx':              int (0-indexed)
            'session_idx':            int (the post-shift session)
            'trial_index_in_animal':  int (cumulative valid trials at shift)
            'from_distribution':      str
            'to_distribution':        str
    """
    shifts: List[Dict] = []
    sessions = sorted(animal.sessions, key=lambda s: s.session_idx)
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
