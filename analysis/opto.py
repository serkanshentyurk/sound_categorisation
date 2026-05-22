"""
analysis/opto.py — Phase assignment for the opto experiment.

For pairwise comparison of two groups (opto on/off, het/wt, pre/post,
masking/baseline, etc.), use behav_utils.analysis.comparison.compute_comparison
which handles pooling internally:

    from behav_utils.data.filtering import filter_session, opto_mask
    from behav_utils.analysis.comparison import compute_comparison
    from behav_utils.plotting.comparison import plot_comparison
    from analysis.opto import assign_opto_phases

    phases = assign_opto_phases(animal.sessions)

    # Within-session opto comparison on the shift phase:
    sessions = phases['shift_with_opto']
    on  = [filter_session(s, opto_mask(s.trials, 0))         for s in sessions]
    off = [filter_session(s, opto_mask(s.trials, 'control')) for s in sessions]

    result = compute_comparison(
        on, off,
        label_a='opto_on', label_b='opto_off',
        n_bootstrap=1000, n_permutations=1000,
    )
    plot_comparison(result)
"""

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData


# Fraction of opto-marked trials above which a non-masking session counts
# as a real opto session (used for chronological pre/post distinction).
_REAL_OPTO_FRACTION = 0.05


def assign_opto_phases(
    sessions: List['SessionData'],
) -> Dict[str, List['SessionData']]:
    """
    Partition one animal's sessions into named phases.

    Phases:
        masking              — session.masking is True (sham blue-light control)
        expert_uniform_pre   — uniform, no opto, before any real opto session
        expert_uniform_opto  — uniform, has real opto
        washout              — uniform, no opto, after first real opto session
        shift_with_opto      — hard_a/hard_b, has real opto
        shift_no_opto        — hard_a/hard_b, no opto

    Args:
        sessions: All sessions for one animal (any order).

    Returns:
        Dict {phase: [SessionData, ...]}. All six keys always present;
        empty lists indicate phases the animal hasn't reached.
    """
    out = {
        'masking':             [],
        'expert_uniform_pre':  [],
        'expert_uniform_opto': [],
        'washout':             [],
        'shift_with_opto':     [],
        'shift_no_opto':       [],
    }

    sorted_sess = sorted(sessions, key=lambda s: s.session_idx)
    first_opto_idx = _find_first_real_opto_idx(sorted_sess)

    for sess in sorted_sess:
        if sess.masking:
            out['masking'].append(sess)
            continue

        has_opto = _has_real_opto(sess)
        dist = sess.distribution

        if dist == 'uniform':
            if has_opto:
                out['expert_uniform_opto'].append(sess)
            elif first_opto_idx is None or sess.session_idx < first_opto_idx:
                out['expert_uniform_pre'].append(sess)
            else:
                out['washout'].append(sess)
        elif dist in ('hard_a', 'hard_b'):
            if has_opto:
                out['shift_with_opto'].append(sess)
            else:
                out['shift_no_opto'].append(sess)

    return out


def _has_real_opto(sess: 'SessionData') -> bool:
    """True if this is a non-masking session with opto-marked trials."""
    if sess.masking:
        return False
    if sess.trials.opto_on is None or len(sess.trials.opto_on) == 0:
        return False
    return float(sess.trials.opto_on.mean()) > _REAL_OPTO_FRACTION


def _find_first_real_opto_idx(sorted_sessions: List['SessionData']) -> Optional[int]:
    """Lowest session_idx with real opto. None if no such session."""
    for s in sorted_sessions:
        if _has_real_opto(s):
            return s.session_idx
    return None
