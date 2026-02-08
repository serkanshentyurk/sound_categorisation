"""
Update Matrix Computation

Computes serial dependence (update) matrices from behavioural data.
Matches the methodology of the old repo.
"""

from Helpers.psychometry import fit_psychometric

import numpy as np
from typing import Optional, Dict, Tuple, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from Models.BE_core import ModelTrace


def compute_update_matrix(
    stimuli: np.ndarray,
    choices: np.ndarray,
    categories: np.ndarray,
    n_bins: int = 8,
    trial_filter: Literal['all', 'post_correct'] = 'post_correct',
    no_response: Optional[np.ndarray] = None,
    not_blockstart: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute update matrix from behavioural data.
    
    The update matrix captures serial dependence: how does the previous trial's
    stimulus shift the current psychometric curve?
    
    Args:
        stimuli: Stimulus values for each trial
        choices: Binary choices (0 = A, 1 = B)
        categories: True categories (0 = A, 1 = B)
        n_bins: Number of bins for stimulus discretisation (default: 8)
        trial_filter: 'post_correct' (only after correct trials) or 'all'
        no_response: Optional boolean array (True = no response). 
                     If None, inferred from np.isnan(choices)
        not_blockstart: Optional boolean array (True = not start of block).
                        If None, inferred as [False, True, True, ...]
    
    Returns:
        update_matrix: (n_bins, n_bins) array where entry [i, j] is the shift
                       in P(B) at current stimulus bin i given previous stimulus
                       was in bin j, relative to the overall psychometric
        conditional_matrix: (n_bins, n_bins) array of conditional P(B) values
        info: Dict with fitting details
    """
    # Convert to arrays
    stimuli = np.asarray(stimuli, dtype=np.float64)
    choices = np.asarray(choices, dtype=np.float64)
    categories = np.asarray(categories, dtype=np.float64)
    n_trials = len(stimuli)
    
    # Infer no_response if not provided
    if no_response is None:
        no_response = np.isnan(choices)
    else:
        no_response = np.asarray(no_response, dtype=bool)
    
    # Infer not_blockstart if not provided (first trial is block start)
    if not_blockstart is None:
        not_blockstart = np.ones(n_trials, dtype=bool)
        if n_trials > 0:
            not_blockstart[0] = False
    else:
        not_blockstart = np.asarray(not_blockstart, dtype=bool)
    
    # Compute rewards
    rewards = (choices == categories).astype(float)
    rewards[np.isnan(choices)] = np.nan
    
    # Create bins
    bin_edges = np.linspace(-1, 1, n_bins + 1)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(stimuli, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Build selection mask for valid trials
    # Base conditions: current trial responded, previous trial responded, not block start
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
    
    # Get total (overall) psychometric curve
    total_stimuli = stimuli[1:][base_condition]
    total_choices = choices[1:][base_condition]
    
    total_psych = fit_psychometric(total_stimuli, total_choices, midpoints)
    
    if total_psych['success']:
        total_curve = total_psych['y_fit']
    else:
        total_curve = np.full(n_bins, np.nan)
    
    # Compute conditional psychometrics for each previous-stimulus bin
    conditional_matrix = np.zeros((n_bins, n_bins))
    update_matrix = np.zeros((n_bins, n_bins))
    bin_counts = np.zeros(n_bins, dtype=int)
    conditional_psychs = []
    
    for j in range(n_bins):
        # Trials where previous stimulus was in bin j
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
        'total_curve': total_curve
    }
    
    return update_matrix, conditional_matrix, info


def compute_update_matrix_from_model_trace(
    trace: 'ModelTrace',
    n_bins: int = 8,
    trial_filter: Literal['all', 'post_correct'] = 'post_correct'
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute update matrix from a ModelTrace object.
    
    Convenience wrapper that extracts fields from ModelTrace.
    
    Args:
        trace: ModelTrace object from simulation
        n_bins: Number of bins for stimulus discretisation
        trial_filter: 'post_correct' or 'all'
    
    Returns:
        update_matrix, conditional_matrix, info (same as compute_update_matrix)
    """
    return compute_update_matrix(
        stimuli=trace.stimuli,
        choices=trace.choices,
        categories=trace.categories,
        n_bins=n_bins,
        trial_filter=trial_filter,
        no_response=trace.no_response,
        not_blockstart=trace.not_blockstart
    )


# Backward compat alias
compute_update_matrix_from_history = compute_update_matrix_from_model_trace


def matrix_error(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """
    Compute mean squared error between two matrices, ignoring NaNs.
    
    Args:
        matrix1: First matrix
        matrix2: Second matrix
    
    Returns:
        Mean squared error (ignoring NaN entries)
    """
    diff = matrix1 - matrix2
    valid = ~np.isnan(diff)
    
    if np.sum(valid) == 0:
        return np.nan
    
    return np.mean(diff[valid] ** 2)
