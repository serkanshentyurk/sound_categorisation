"""
Update Matrix Computation

Computes serial dependence (update) matrices from behavioural data.

Two levels:
    compute_update_matrix()               — raw arrays (no data class dependency)
    compute_update_matrix_from_sessions() — List[SessionData], NO filtering

Data must be pre-filtered via filter_trials / session.filter before calling
session-level functions.
"""

import numpy as np
import warnings
from typing import Optional, Dict, List, Tuple, Literal, TYPE_CHECKING

from behav_utils.analysis.psychometry import fit_psychometric

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData


def compute_update_matrix(
    stimuli: np.ndarray,
    choices: np.ndarray,
    categories: np.ndarray,
    n_bins: int = 8,
    trial_filter: Literal['all', 'post_correct'] = 'post_correct',
    no_response: Optional[np.ndarray] = None,
    not_blockstart: Optional[np.ndarray] = None,
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

    if not_blockstart is None:
        not_blockstart = np.ones(n_trials, dtype=bool)
        if n_trials > 0:
            not_blockstart[0] = False
    else:
        not_blockstart = np.asarray(not_blockstart, dtype=bool)

    rewards = (choices == categories).astype(float)
    rewards[np.isnan(choices)] = np.nan

    bin_edges = np.linspace(-1, 1, n_bins + 1)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.clip(np.digitize(stimuli, bin_edges) - 1, 0, n_bins - 1)

    curr_responded = ~no_response[1:]
    prev_responded = ~no_response[:-1]
    is_not_blockstart = not_blockstart[1:]

    if trial_filter == 'post_correct':
        prev_correct = rewards[:-1] == 1
        base_condition = prev_correct & curr_responded & prev_responded & is_not_blockstart
    elif trial_filter == 'all':
        base_condition = curr_responded & prev_responded & is_not_blockstart
    else:
        raise ValueError(f"trial_filter must be 'post_correct' or 'all', got '{trial_filter}'")

    total_stimuli = stimuli[1:][base_condition]
    total_choices = choices[1:][base_condition]
    total_psych = fit_psychometric(total_stimuli, total_choices, midpoints)

    total_curve = total_psych['y_fit'] if total_psych['success'] else np.full(n_bins, np.nan)

    conditional_matrix = np.zeros((n_bins, n_bins))
    update_matrix = np.zeros((n_bins, n_bins))
    bin_counts = np.zeros(n_bins, dtype=int)
    conditional_psychs = []

    for j in range(n_bins):
        prev_in_bin = bin_indices[:-1] == j
        condition = base_condition & prev_in_bin
        cond_stimuli = stimuli[1:][condition]
        cond_choices = choices[1:][condition]
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

def _sessions_to_pooled_arrays(
    sessions: List['SessionData'],
) -> Optional[Dict[str, np.ndarray]]:
    """
    Concatenate trials from multiple sessions into flat arrays.

    No filtering. Data must be pre-filtered via filter_trials.
    Session boundaries are marked as block starts.
    """
    all_stim, all_choice, all_cat = [], [], []
    all_no_resp, all_nbs = [], []

    for sess in sessions:
        arrays = sess.get_arrays()
        n = arrays['n_trials']
        if n == 0:
            continue

        all_stim.append(arrays['stimuli'])
        all_choice.append(arrays['choices'])
        all_cat.append(arrays['categories'])
        all_no_resp.append(arrays['no_response'])

        nbs = np.ones(n, dtype=bool)
        nbs[0] = False  # block boundary
        all_nbs.append(nbs)

    if not all_stim:
        return None

    return {
        'stimuli': np.concatenate(all_stim),
        'choices': np.concatenate(all_choice),
        'categories': np.concatenate(all_cat),
        'no_response': np.concatenate(all_no_resp),
        'not_blockstart': np.concatenate(all_nbs),
        'n_sessions': len(all_stim),
        'n_trials_pooled': sum(len(s) for s in all_stim),
    }


def compute_update_matrix_from_sessions(
    sessions: List['SessionData'],
    method: Literal['pool', 'average'] = 'pool',
    n_bins: int = 8,
    trial_filter: Literal['all', 'post_correct'] = 'post_correct',
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute update matrix from a list of sessions.

    No filtering. Data must be pre-filtered via filter_trials / session.filter.

    Two methods:
        'pool':    Concatenate all trials, compute one UM. Default.
        'average': Compute UM per session, then nanmean.

    Args:
        sessions: List of (pre-filtered) SessionData.
        method: 'pool' or 'average'.
        n_bins: Number of stimulus bins.
        trial_filter: 'post_correct' or 'all'.

    Returns:
        (update_matrix, conditional_matrix, info)
    """
    empty = np.full((n_bins, n_bins), np.nan)

    if not sessions:
        return empty, empty, {'method': method, 'n_sessions': 0}

    if method == 'pool':
        pooled = _sessions_to_pooled_arrays(sessions)
        if pooled is None:
            return empty, empty, {'method': 'pool', 'n_sessions': 0}

        um, cm, info = compute_update_matrix(
            pooled['stimuli'], pooled['choices'], pooled['categories'],
            n_bins=n_bins, trial_filter=trial_filter,
            no_response=pooled['no_response'],
            not_blockstart=pooled['not_blockstart'],
        )
        info['method'] = 'pool'
        info['n_sessions'] = pooled['n_sessions']
        info['n_trials_pooled'] = pooled['n_trials_pooled']
        return um, cm, info

    elif method == 'average':
        ums, cms, n_trials_list = [], [], []

        for sess in sessions:
            arrays = sess.get_arrays()
            n = arrays['n_trials']
            if n < 20:
                continue

            nbs = np.ones(n, dtype=bool)
            nbs[0] = False

            um_s, cm_s, _ = compute_update_matrix(
                arrays['stimuli'], arrays['choices'], arrays['categories'],
                n_bins=n_bins, trial_filter=trial_filter,
                no_response=arrays['no_response'],
                not_blockstart=nbs,
            )
            ums.append(um_s)
            cms.append(cm_s)
            n_trials_list.append(n)

        if not ums:
            return empty, empty, {'method': 'average', 'n_sessions': 0}

        um_stack = np.stack(ums)
        cm_stack = np.stack(cms)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            um_avg = np.nanmean(um_stack, axis=0)
            cm_avg = np.nanmean(cm_stack, axis=0)

        info = {
            'method': 'average',
            'n_sessions': len(ums),
            'n_trials_per_session': n_trials_list,
            'um_stack': um_stack,
            'um_sem': np.nanstd(um_stack, axis=0) / np.sqrt(len(ums)),
        }
        return um_avg, cm_avg, info

    else:
        raise ValueError(f"method must be 'pool' or 'average', got '{method}'")
