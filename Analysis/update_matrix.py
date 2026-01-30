from Helpers.psychometry import fit_psychometric

import numpy as np
from typing import Optional, Dict, List, Tuple, Union, Literal
import warnings

def _select_trials_post_correct(stimuli: np.ndarray, choices: np.ndarray,
                                 rewards: np.ndarray, no_response: np.ndarray,
                                 not_blockstart: np.ndarray,
                                 previous_bin: Optional[int] = None,
                                 n_bins: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select trials following correct responses, optionally filtered by previous stimulus bin.
    
    Args:
        stimuli: Stimulus values
        choices: Binary choices (0 = A, 1 = B)
        rewards: Binary rewards (1 = correct)
        no_response: Boolean array (True = no response)
        not_blockstart: Boolean array (True = not start of block)
        previous_bin: If provided, only select trials where previous stimulus was in this bin
        n_bins: Number of bins for stimulus discretisation
    
    Returns:
        selected_stimuli, selected_choices
    """
    stimuli = np.asarray(stimuli)
    choices = np.asarray(choices)
    rewards = np.asarray(rewards)
    no_response = np.asarray(no_response)
    not_blockstart = np.asarray(not_blockstart)
    
    # Create bins
    bin_edges = np.linspace(-1, 1, n_bins + 1)
    bin_indices = np.digitize(stimuli, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Base conditions (for trial t, looking at t-1)
    prev_correct = rewards[:-1] == 1
    curr_responded = ~no_response[1:]
    prev_responded = ~no_response[:-1]
    not_block_start = not_blockstart[1:] == True
    
    condition = prev_correct & curr_responded & prev_responded & not_block_start
    
    # Optional: filter by previous stimulus bin
    if previous_bin is not None:
        prev_in_bin = bin_indices[:-1] == previous_bin
        condition = condition & prev_in_bin
    
    selected_stimuli = stimuli[1:][condition]
    selected_choices = choices[1:][condition]
    
    return selected_stimuli, selected_choices


def _select_trials_all(stimuli: np.ndarray, choices: np.ndarray,
                       no_response: np.ndarray, not_blockstart: np.ndarray,
                       previous_bin: Optional[int] = None,
                       n_bins: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select all valid trials (with response), optionally filtered by previous stimulus bin.
    
    Args:
        stimuli: Stimulus values
        choices: Binary choices (0 = A, 1 = B)
        no_response: Boolean array (True = no response)
        not_blockstart: Boolean array (True = not start of block)
        previous_bin: If provided, only select trials where previous stimulus was in this bin
        n_bins: Number of bins for stimulus discretisation
    
    Returns:
        selected_stimuli, selected_choices
    """
    stimuli = np.asarray(stimuli)
    choices = np.asarray(choices)
    no_response = np.asarray(no_response)
    not_blockstart = np.asarray(not_blockstart)
    
    # Create bins
    bin_edges = np.linspace(-1, 1, n_bins + 1)
    bin_indices = np.digitize(stimuli, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Base conditions
    curr_responded = ~no_response[1:]
    prev_responded = ~no_response[:-1]
    not_block_start = not_blockstart[1:] == True
    
    condition = curr_responded & prev_responded & not_block_start
    
    # Optional: filter by previous stimulus bin
    if previous_bin is not None:
        prev_in_bin = bin_indices[:-1] == previous_bin
        condition = condition & prev_in_bin
    
    selected_stimuli = stimuli[1:][condition]
    selected_choices = choices[1:][condition]
    
    return selected_stimuli, selected_choices


def compute_update_matrix(stimuli: np.ndarray, choices: np.ndarray,
                          rewards: np.ndarray, no_response: np.ndarray,
                          not_blockstart: np.ndarray,
                          n_bins: int = 8,
                          trial_filter: Literal['post_correct', 'all'] = 'post_correct'
                          ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute update matrix and conditional psychometric matrix.
    
    The update matrix captures serial dependence: how does the previous trial's
    stimulus location shift the current psychometric curve?
    
    Args:
        stimuli: Stimulus values for each trial
        choices: Binary choices (0 = A, 1 = B)
        rewards: Binary rewards (1 = correct, 0 = incorrect)
        no_response: Boolean array (True = no response on this trial)
        not_blockstart: Boolean array (True = not the start of a block/session)
        n_bins: Number of bins for stimulus discretisation (default: 8)
        trial_filter: 'post_correct' or 'all'
            - 'post_correct': Only trials following correct responses (lab default)
            - 'all': All valid trials with responses
    
    Returns:
        update_matrix: (n_bins, n_bins) array where entry [i, j] is the difference
                       between conditional P(B) and total P(B) for stimulus bin i
                       given previous stimulus was in bin j
        conditional_matrix: (n_bins, n_bins) array of conditional P(B) values
        info: Dict with fitting details (total psychometric, per-bin counts, etc.)
    """
    stimuli = np.asarray(stimuli, dtype=np.float64)
    choices = np.asarray(choices, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    no_response = np.asarray(no_response, dtype=bool)
    not_blockstart = np.asarray(not_blockstart, dtype=bool)
    
    # Select trial filter function
    if trial_filter == 'post_correct':
        select_fn = lambda prev_bin: _select_trials_post_correct(
            stimuli, choices, rewards, no_response, not_blockstart, prev_bin, n_bins
        )
        select_fn_total = lambda: _select_trials_post_correct(
            stimuli, choices, rewards, no_response, not_blockstart, None, n_bins
        )
    elif trial_filter == 'all':
        select_fn = lambda prev_bin: _select_trials_all(
            stimuli, choices, no_response, not_blockstart, prev_bin, n_bins
        )
        select_fn_total = lambda: _select_trials_all(
            stimuli, choices, no_response, not_blockstart, None, n_bins
        )
    else:
        raise ValueError(f"trial_filter must be 'post_correct' or 'all', got '{trial_filter}'")
    
    # Bin midpoints for evaluation
    bin_edges = np.linspace(-1, 1, n_bins + 1)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Fit total psychometric (all selected trials)
    s_total, c_total = select_fn_total()
    total_psych = fit_psychometric(s_total, c_total, midpoints)
    
    if total_psych['success']:
        total_curve = total_psych['y_fit']
    else:
        total_curve = np.full(n_bins, np.nan)
        warnings.warn("Total psychometric fit failed")
    
    # Fit conditional psychometrics for each previous-stimulus bin
    conditional_matrix = np.zeros((n_bins, n_bins))
    update_matrix = np.zeros((n_bins, n_bins))
    bin_counts = np.zeros(n_bins, dtype=int)
    conditional_psychs = []
    
    for j in range(n_bins):
        s_cond, c_cond = select_fn(j)
        bin_counts[j] = len(s_cond)
        
        if len(s_cond) < 10:
            conditional_matrix[:, j] = np.nan
            update_matrix[:, j] = np.nan
            conditional_psychs.append(None)
        else:
            cond_psych = fit_psychometric(s_cond, c_cond, midpoints)
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
        'total_trials': len(s_total),
        'trial_filter': trial_filter
    }
    
    return update_matrix, conditional_matrix, info


def matrix_error(model_matrix: np.ndarray, data_matrix: np.ndarray) -> float:
    """
    Compute mean squared error between model and data matrices, ignoring NaNs.
    
    Args:
        model_matrix: Model-predicted matrix
        data_matrix: Data matrix
    
    Returns:
        Mean squared error (normalised by number of valid entries)
    """
    squared_diff = (model_matrix - data_matrix) ** 2
    
    # Count non-NaN entries per column
    non_nan_per_col = np.sum(~np.isnan(squared_diff), axis=0)
    n_valid_cols = np.sum(non_nan_per_col > 0)
    
    if n_valid_cols == 0:
        return np.nan
    
    # Sum squared differences (ignoring NaN)
    col_sums = np.nansum(squared_diff, axis=0)
    total_error = np.sum(col_sums) / n_valid_cols
    
    return total_error

