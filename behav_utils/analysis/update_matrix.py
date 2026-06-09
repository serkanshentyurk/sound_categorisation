"""
Update Matrix Computation

Computes serial dependence (update) matrices from behavioural data.

Two levels:
    fit_update_matrix()  — raw arrays (no data class dependency)
    compute_um()             — List[SessionData], NO filtering

Data must be pre-filtered via filter_trials / session.filter before calling
session-level functions.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Literal, TYPE_CHECKING

from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.data.ops.filtering import pool_arrays

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData


def fit_update_matrix(
    stimuli: np.ndarray,
    choices: np.ndarray,
    categories: np.ndarray,
    n_bins: int = 8,
    trial_filter: Literal['all', 'post_correct'] = 'post_correct',
    no_response: Optional[np.ndarray] = None,
    not_blockstart: Optional[np.ndarray] = None,
    prev_stimuli: Optional[np.ndarray] = None,
    prev_choices: Optional[np.ndarray] = None,
    prev_categories: Optional[np.ndarray] = None,
    prev_has_prev: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute update matrix from raw behavioural arrays.

    The update matrix captures serial dependence: how does the previous
    trial's stimulus shift the current psychometric curve?

    Args:
        stimuli: Stimulus values for each trial.
        choices: Binary choices (0=A, 1=B).
        categories: True categories (0=A, 1=B).
        n_bins: Number of bins for stimulus discretisation.
        trial_filter: 'post_correct' (only after correct) or 'all'.
        no_response: Bool array (True = no response). Inferred from NaN if None.
        not_blockstart: Bool array (True = not start of block). Auto if None.
        prev_stimuli, prev_choices, prev_categories, prev_has_prev: Frozen,
            abort-aware lag-1 arrays aligned to every trial. If prev_stimuli is
            given, the previous trial is taken from these (NOT from array
            adjacency), so the matrix is correct on a non-consecutive subset
            (e.g. opto-only or post-opto trials). If None, the previous trial is
            the immediately preceding array element via not_blockstart (the
            simulated / SBI path, unchanged).

    Returns:
        update_matrix: (n_bins, n_bins) shift in P(B)
        conditional_matrix: (n_bins, n_bins) conditional P(B) values
        info: Dict with fitting details
    """
    stimuli = np.asarray(stimuli, dtype=np.float64)
    choices = np.asarray(choices, dtype=np.float64)
    categories = np.asarray(categories, dtype=np.float64)
    n_trials = len(stimuli)

    if no_response is None:
        no_response = np.isnan(choices)
    else:
        no_response = np.asarray(no_response, dtype=bool)

    bin_edges = np.linspace(-1, 1, n_bins + 1)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    if prev_stimuli is not None:
        # SESSION PATH: previous trial from the frozen, abort-aware lag-1 view.
        # Current trial = every trial; valid pairs gated by has_prev. Correct on
        # a non-consecutive subset (opto-only / post-opto), where array adjacency
        # would otherwise give the wrong predecessor.
        prev_stimuli = np.asarray(prev_stimuli, dtype=np.float64)
        prev_choices = np.asarray(prev_choices, dtype=np.float64)
        prev_categories = np.asarray(prev_categories, dtype=np.float64)
        has_prev = np.asarray(prev_has_prev, dtype=bool)

        curr_stim = stimuli
        curr_choice = choices
        prev_bin = np.clip(np.digitize(prev_stimuli, bin_edges) - 1, 0, n_bins - 1)
        prev_reward = (prev_choices == prev_categories)   # mirrors rewards, on prev
        curr_responded = ~no_response
        prev_responded = ~np.isnan(prev_choices)

        if trial_filter == 'post_correct':
            base = prev_reward & curr_responded & prev_responded & has_prev
        elif trial_filter == 'all':
            base = curr_responded & prev_responded & has_prev
        else:
            raise ValueError(f"trial_filter must be 'post_correct' or 'all', got '{trial_filter}'")
    else:
        # ADJACENCY PATH: previous trial = the immediately preceding array
        # element (simulated / SBI arrays, which carry no prev_trial view).
        if not_blockstart is None:
            not_blockstart = np.ones(n_trials, dtype=bool)
            if n_trials > 0:
                not_blockstart[0] = False
        else:
            not_blockstart = np.asarray(not_blockstart, dtype=bool)

        rewards = (choices == categories).astype(float)
        rewards[np.isnan(choices)] = np.nan
        bin_indices = np.clip(np.digitize(stimuli, bin_edges) - 1, 0, n_bins - 1)

        curr_stim = stimuli[1:]
        curr_choice = choices[1:]
        prev_bin = bin_indices[:-1]
        curr_responded = ~no_response[1:]
        prev_responded = ~no_response[:-1]
        is_not_blockstart = not_blockstart[1:]

        if trial_filter == 'post_correct':
            prev_correct = rewards[:-1] == 1
            base = prev_correct & curr_responded & prev_responded & is_not_blockstart
        elif trial_filter == 'all':
            base = curr_responded & prev_responded & is_not_blockstart
        else:
            raise ValueError(f"trial_filter must be 'post_correct' or 'all', got '{trial_filter}'")

    total_stimuli = curr_stim[base]
    total_choices = curr_choice[base]
    total_psych = fit_psychometric(total_stimuli, total_choices, midpoints)

    total_curve = total_psych['y_fit'] if total_psych['success'] else np.full(n_bins, np.nan)

    conditional_matrix = np.zeros((n_bins, n_bins))
    update_matrix = np.zeros((n_bins, n_bins))
    bin_counts = np.zeros(n_bins, dtype=int)
    conditional_psychs = []

    for j in range(n_bins):
        prev_in_bin = prev_bin == j
        condition = base & prev_in_bin
        cond_stimuli = curr_stim[condition]
        cond_choices = curr_choice[condition]
        bin_counts[j] = len(cond_stimuli)

        if len(cond_stimuli) < 10:
            conditional_matrix[:, j] = np.nan
            update_matrix[:, j] = np.nan
            conditional_psychs.append(None)
        else:
            cond_psych = fit_psychometric(cond_stimuli, cond_choices, midpoints)
            conditional_psychs.append(cond_psych)
            if cond_psych['success']:
                conditional_matrix[:, j] = cond_psych['y_fit']
                update_matrix[:, j] = cond_psych['y_fit'] - total_curve
            else:
                conditional_matrix[:, j] = np.nan
                update_matrix[:, j] = np.nan

    info = {
        'total_psychometric': total_psych,
        'conditional_psychometrics': conditional_psychs,
        'bin_edges': bin_edges,
        'midpoints': midpoints,
        'bin_counts': bin_counts,
        'total_trials': len(total_stimuli),
        'trial_filter': trial_filter,
        'total_curve': total_curve,
    }
    return update_matrix, conditional_matrix, info


def matrix_error(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """Mean squared error between two matrices, ignoring NaNs."""
    diff = matrix1 - matrix2
    valid = ~np.isnan(diff)
    if np.sum(valid) == 0:
        return np.nan
    return np.mean(diff[valid] ** 2)


# =============================================================================
# SESSION-LEVEL (NO FILTERING — data must be pre-filtered)
# =============================================================================

def compute_um(
    sessions: List['SessionData'],
    mode: Literal['pooled', 'per_session'] = 'pooled',
    n_bins: int = 8,
    trial_filter: Literal['all', 'post_correct'] = 'post_correct',
) -> Dict:
    """
    Compute update matrix from pre-filtered sessions.

    Session-level wrapper around fit_update_matrix(). Two modes, which
    return DIFFERENT shapes:

      'pooled'      : concatenate all sessions, compute one update matrix.
                      Returns {mode, um, conditional_matrix, n_sessions,
                      n_trials, n_bins, info}.
      'per_session' : compute an update matrix per session and return the
                      individual matrices as a LIST, with NO reduction —
                      aggregate (e.g. a nan-aware mean over the list) downstream.
                      Returns {mode, per_session, n_sessions, n_bins}, where each
                      entry has session_id, session_idx, um, conditional_matrix,
                      n_trials, info. Note the UM is data-hungry (9 fits, ≥10
                      pairs/bin), so single-session matrices are often sparse.

    Args:
        sessions: Pre-filtered List[SessionData].
        mode: 'pooled' | 'per_session'.
        n_bins: Number of stimulus bins.
        trial_filter: 'post_correct' or 'all'.
    """
    if mode == 'pooled':
        pooled = pool_arrays(sessions)
        if pooled['n_trials'] == 0:
            empty = np.full((n_bins, n_bins), np.nan)
            return {
                'mode': 'pooled', 'um': empty, 'conditional_matrix': empty,
                'n_sessions': 0, 'n_trials': 0, 'n_bins': n_bins, 'info': {},
            }
        um, conditional, info = fit_update_matrix(
            pooled['stimuli'], pooled['choices'], pooled['categories'],
            n_bins=n_bins, trial_filter=trial_filter,
            no_response=pooled['no_response'],
            prev_stimuli=pooled['prev_stimuli'],
            prev_choices=pooled['prev_choices'],
            prev_categories=pooled['prev_categories'],
            prev_has_prev=pooled['prev_has_prev'],
        )
        return {
            'mode': 'pooled', 'um': um, 'conditional_matrix': conditional,
            'n_sessions': pooled['n_sessions'],
            'n_trials': info.get('total_trials', 0),
            'n_bins': n_bins, 'info': info,
        }

    if mode == 'per_session':
        per_session = []
        for sess in sessions:
            a = sess.get_arrays()
            if a['n_trials'] == 0:
                continue
            m, c, inf = fit_update_matrix(
                a['stimuli'], a['choices'], a['categories'],
                n_bins=n_bins, trial_filter=trial_filter,
                no_response=a['no_response'],
                prev_stimuli=a['prev_stimuli'],
                prev_choices=a['prev_choices'],
                prev_categories=a['prev_categories'],
                prev_has_prev=a['prev_has_prev'],
            )
            per_session.append({
                'session_id': getattr(sess, 'session_id', None),
                'session_idx': getattr(sess, 'session_idx', None),
                'um': m,
                'conditional_matrix': c,
                'n_trials': inf.get('total_trials', 0),
                'info': inf,
            })
        return {
            'mode': 'per_session',
            'per_session': per_session,
            'n_sessions': len(per_session),
            'n_bins': n_bins,
        }

    raise ValueError(f"mode must be 'pooled' or 'per_session', got {mode!r}")

# compute_update_matrix calls compute_um, which calls fit_update_matrix, so we only export the latter. 
def compute_update_matrix(*args, **kwargs):
    """DEPRECATED: use compute_um() instead."""
    import warnings
    warnings.warn("compute_update_matrix() is deprecated; use compute_um() instead.", DeprecationWarning, stacklevel=2)
    return compute_um(*args, **kwargs)
