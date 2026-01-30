"""
Summary statistics for Simulation-Based Inference.

Modular design with registry pattern for easy extension.
Each stat function has signature: (choices, stimuli, categories) -> scalar or dict
Handles both single-session (n_trials,) and multi-session (n_trials, n_sessions) arrays.
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Union
from functools import wraps

from Helpers.psychometry import fit_psychometric


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_N_BINS = 8  # Default number of bins for binned statistics


# =============================================================================
# REGISTRY
# =============================================================================

SUMMARY_REGISTRY: Dict[str, Callable] = {}


def register_stat(name: str):
    """Decorator to register a summary statistic function."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        SUMMARY_REGISTRY[name] = wrapper
        return wrapper
    return decorator


def list_available_stats() -> List[str]:
    """List all registered summary statistics."""
    return list(SUMMARY_REGISTRY.keys())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _ensure_1d(arr: np.ndarray) -> np.ndarray:
    """Flatten array to 1D if needed."""
    arr = np.asarray(arr)
    if arr.ndim > 1:
        return arr.flatten()
    return arr


def _is_multisession(arr: np.ndarray) -> bool:
    """Check if array is multi-session (2D with n_sessions > 1)."""
    return arr.ndim == 2 and arr.shape[1] > 1


def _apply_per_session(func: Callable, choices: np.ndarray, stimuli: np.ndarray, 
                       categories: np.ndarray) -> np.ndarray:
    """Apply a single-session function to each session of multi-session data."""
    if not _is_multisession(choices):
        return func(_ensure_1d(choices), _ensure_1d(stimuli), _ensure_1d(categories))
    
    n_sessions = choices.shape[1]
    results = []
    for i in range(n_sessions):
        result = func(choices[:, i], stimuli[:, i], categories[:, i])
        results.append(result)
    
    return np.array(results)


# =============================================================================
# CORE SUMMARY STATISTICS
# =============================================================================

@register_stat('accuracy')
def compute_accuracy(choices: np.ndarray, stimuli: np.ndarray, 
                     categories: np.ndarray) -> Union[float, np.ndarray]:
    """
    Compute accuracy (proportion correct).
    
    Returns:
        float for single-session, array of shape (n_sessions,) for multi-session
    """
    choices = np.asarray(choices)
    categories = np.asarray(categories)
    
    if _is_multisession(choices):
        return np.mean(choices == categories, axis=0)
    else:
        return float(np.mean(_ensure_1d(choices) == _ensure_1d(categories)))


@register_stat('psychometric')
def compute_psychometric_params(choices: np.ndarray, stimuli: np.ndarray,
                                 categories: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    """
    Fit psychometric curve and return parameters.
    
    Returns:
        Dict with 'pse', 'slope', 'lapse_low', 'lapse_high'
        Values are floats for single-session, arrays for multi-session
    """
    def _fit_single(c, s, cat):
        psych = fit_psychometric(s, c)
        if psych.get('success', False):
            return {
                'pse': psych['mu'],
                'slope': psych['sigma'],
                'lapse_low': psych['lapse_low'],
                'lapse_high': psych['lapse_high']
            }
        else:
            return {
                'pse': np.nan,
                'slope': np.nan,
                'lapse_low': np.nan,
                'lapse_high': np.nan
            }
    
    choices = np.asarray(choices)
    stimuli = np.asarray(stimuli)
    
    if _is_multisession(choices):
        n_sessions = choices.shape[1]
        results = {k: [] for k in ['pse', 'slope', 'lapse_low', 'lapse_high']}
        
        for i in range(n_sessions):
            single_result = _fit_single(choices[:, i], stimuli[:, i], categories[:, i])
            for k, v in single_result.items():
                results[k].append(v)
        
        return {k: np.array(v) for k, v in results.items()}
    else:
        return _fit_single(_ensure_1d(choices), _ensure_1d(stimuli), _ensure_1d(categories))


@register_stat('recency')
def compute_recency_index(choices: np.ndarray, stimuli: np.ndarray,
                          categories: np.ndarray) -> Union[float, np.ndarray]:
    """
    Compute recency index: effect of previous trial category on current choice.
    
    Measures: P(choose B | prev_category=B) - P(choose B | prev_category=A)
    
    High recency = recent trials strongly influence choice (high learning rate)
    Low recency = stable behaviour (low learning rate)
    
    Returns:
        float for single-session, array for multi-session
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c)
        cat = _ensure_1d(cat)
        
        if len(c) < 10:
            return np.nan
        
        prev_cat = np.roll(cat, 1)
        # Exclude first trial (no valid previous)
        valid = np.arange(1, len(c))
        
        c_valid = c[valid]
        prev_cat_valid = prev_cat[valid]
        
        # P(choose B | prev was B) - P(choose B | prev was A)
        prev_was_b = prev_cat_valid == 1
        prev_was_a = prev_cat_valid == 0
        
        if np.sum(prev_was_b) == 0 or np.sum(prev_was_a) == 0:
            return np.nan
        
        p_b_after_b = np.mean(c_valid[prev_was_b])
        p_b_after_a = np.mean(c_valid[prev_was_a])
        
        return p_b_after_b - p_b_after_a
    
    choices = np.asarray(choices)
    
    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('win_stay')
def compute_win_stay_index(choices: np.ndarray, stimuli: np.ndarray,
                           categories: np.ndarray) -> Union[float, np.ndarray]:
    """
    Compute win-stay index: tendency to repeat choice after reward.
    
    Measures: P(repeat | rewarded) - P(repeat | unrewarded)
    
    Returns:
        float for single-session, array for multi-session
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c)
        cat = _ensure_1d(cat)
        
        if len(c) < 10:
            return np.nan
        
        rewards = (c == cat).astype(int)
        prev_choice = np.roll(c, 1)
        prev_reward = np.roll(rewards, 1)
        
        # Exclude first trial
        valid = np.arange(1, len(c))
        c_valid = c[valid]
        prev_choice_valid = prev_choice[valid]
        prev_reward_valid = prev_reward[valid]
        
        repeat = (c_valid == prev_choice_valid)
        
        won = prev_reward_valid == 1
        lost = prev_reward_valid == 0
        
        if np.sum(won) == 0 or np.sum(lost) == 0:
            return np.nan
        
        p_stay_after_win = np.mean(repeat[won])
        p_stay_after_loss = np.mean(repeat[lost])
        
        return p_stay_after_win - p_stay_after_loss
    
    choices = np.asarray(choices)
    
    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('lose_shift')
def compute_lose_shift_index(choices: np.ndarray, stimuli: np.ndarray,
                             categories: np.ndarray) -> Union[float, np.ndarray]:
    """
    Compute lose-shift index: tendency to switch choice after no reward.
    
    Measures: P(switch | unrewarded)
    
    Returns:
        float for single-session, array for multi-session
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c)
        cat = _ensure_1d(cat)
        
        if len(c) < 10:
            return np.nan
        
        rewards = (c == cat).astype(int)
        prev_choice = np.roll(c, 1)
        prev_reward = np.roll(rewards, 1)
        
        # Exclude first trial
        valid = np.arange(1, len(c))
        c_valid = c[valid]
        prev_choice_valid = prev_choice[valid]
        prev_reward_valid = prev_reward[valid]
        
        switch = (c_valid != prev_choice_valid)
        lost = prev_reward_valid == 0
        
        if np.sum(lost) == 0:
            return np.nan
        
        return float(np.mean(switch[lost]))
    
    choices = np.asarray(choices)
    
    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('choice_autocorr')
def compute_choice_autocorrelation(choices: np.ndarray, stimuli: np.ndarray,
                                    categories: np.ndarray, lag: int = 1) -> Union[float, np.ndarray]:
    """
    Compute choice autocorrelation at given lag.
    
    Measures: correlation between choice_t and choice_{t-lag}
    
    Returns:
        float for single-session, array for multi-session
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        
        if len(c) < lag + 10:
            return np.nan
        
        c_current = c[lag:]
        c_lagged = c[:-lag]
        
        # Pearson correlation
        if np.std(c_current) == 0 or np.std(c_lagged) == 0:
            return np.nan
        
        return float(np.corrcoef(c_current, c_lagged)[0, 1])
    
    choices = np.asarray(choices)
    
    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('bias')
def compute_bias(choices: np.ndarray, stimuli: np.ndarray,
                 categories: np.ndarray) -> Union[float, np.ndarray]:
    """
    Compute choice bias: overall tendency to choose B.
    
    Measures: P(choose B) - 0.5
    
    Returns:
        float for single-session, array for multi-session
    """
    choices = np.asarray(choices)
    
    if _is_multisession(choices):
        return np.mean(choices, axis=0) - 0.5
    else:
        return float(np.mean(_ensure_1d(choices)) - 0.5)


@register_stat('stimulus_sensitivity')
def compute_stimulus_sensitivity(choices: np.ndarray, stimuli: np.ndarray,
                                  categories: np.ndarray) -> Union[float, np.ndarray]:
    """
    Compute stimulus sensitivity: correlation between stimulus and choice.
    
    High sensitivity = choices driven by stimulus (BE-like)
    Low sensitivity = choices independent of stimulus (heuristic-like)
    
    Returns:
        float for single-session, array for multi-session
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c).astype(float)
        s = _ensure_1d(s).astype(float)
        
        if len(c) < 10:
            return np.nan
        
        if np.std(c) == 0 or np.std(s) == 0:
            return np.nan
        
        return float(np.corrcoef(c, s)[0, 1])
    
    choices = np.asarray(choices)
    stimuli = np.asarray(stimuli)
    
    if _is_multisession(choices):
        return _apply_per_session(_compute_single, choices, stimuli, categories)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('binned_accuracy')
def compute_binned_accuracy(choices: np.ndarray, stimuli: np.ndarray,
                            categories: np.ndarray, n_bins: int = 8) -> Union[np.ndarray, np.ndarray]:
    """
    Compute accuracy binned by stimulus value.
    
    Returns:
        Array of shape (n_bins,) for single-session
        Array of shape (n_bins, n_sessions) for multi-session
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c)
        s = _ensure_1d(s)
        cat = _ensure_1d(cat)
        
        bin_edges = np.linspace(-1, 1, n_bins + 1)
        bin_indices = np.digitize(s, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        binned_acc = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_indices == b
            if np.sum(mask) > 0:
                binned_acc[b] = np.mean(c[mask] == cat[mask])
            else:
                binned_acc[b] = np.nan
        
        return binned_acc
    
    choices = np.asarray(choices)
    stimuli = np.asarray(stimuli)
    categories = np.asarray(categories)
    
    if _is_multisession(choices):
        n_sessions = choices.shape[1]
        results = []
        for i in range(n_sessions):
            results.append(_compute_single(choices[:, i], stimuli[:, i], categories[:, i]))
        return np.array(results).T  # (n_bins, n_sessions)
    else:
        return _compute_single(choices, stimuli, categories)


@register_stat('binned_choice_prob')
def compute_binned_choice_probability(choices: np.ndarray, stimuli: np.ndarray,
                                       categories: np.ndarray, n_bins: int = 8) -> Union[np.ndarray, np.ndarray]:
    """
    Compute P(choose B) binned by stimulus value.
    
    This is essentially the empirical psychometric curve.
    
    Returns:
        Array of shape (n_bins,) for single-session
        Array of shape (n_bins, n_sessions) for multi-session
    """
    def _compute_single(c, s, cat):
        c = _ensure_1d(c)
        s = _ensure_1d(s)
        
        bin_edges = np.linspace(-1, 1, n_bins + 1)
        bin_indices = np.digitize(s, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        binned_prob = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_indices == b
            if np.sum(mask) > 0:
                binned_prob[b] = np.mean(c[mask])
            else:
                binned_prob[b] = np.nan
        
        return binned_prob
    
    choices = np.asarray(choices)
    stimuli = np.asarray(stimuli)
    
    if _is_multisession(choices):
        n_sessions = choices.shape[1]
        results = []
        for i in range(n_sessions):
            results.append(_compute_single(choices[:, i], stimuli[:, i], categories[:, i]))
        return np.array(results).T  # (n_bins, n_sessions)
    else:
        return _compute_single(choices, stimuli, categories)


# =============================================================================
# MAIN INTERFACE
# =============================================================================

# Default statistics for SBI
DEFAULT_STATS = ['accuracy', 'psychometric', 'recency', 'win_stay', 'stimulus_sensitivity']


def compute_summary_stats(
    choices: np.ndarray,
    stimuli: np.ndarray,
    categories: np.ndarray,
    stat_names: Optional[List[str]] = None,
    return_dict: bool = False
) -> Union[np.ndarray, Dict]:
    """
    Compute summary statistics for SBI.
    
    Args:
        choices: Binary choices, shape (n_trials,) or (n_trials, n_sessions)
        stimuli: Stimulus values, same shape as choices
        categories: True categories, same shape as choices
        stat_names: List of stat names to compute. If None, uses DEFAULT_STATS
        return_dict: If True, return dict; if False, return flattened array for SBI
    
    Returns:
        If return_dict=True: Dict mapping stat names to values
        If return_dict=False: 1D array of all stats concatenated (suitable for SBI)
    """
    if stat_names is None:
        stat_names = DEFAULT_STATS
    
    # Validate stat names
    for name in stat_names:
        if name not in SUMMARY_REGISTRY:
            raise ValueError(f"Unknown stat: '{name}'. Available: {list_available_stats()}")
    
    results = {}
    for name in stat_names:
        func = SUMMARY_REGISTRY[name]
        results[name] = func(choices, stimuli, categories)
    
    if return_dict:
        return results
    
    # Flatten to 1D array for SBI
    return flatten_stats(results)


def flatten_stats(stats_dict: Dict) -> np.ndarray:
    """
    Flatten stats dict to 1D array for SBI.
    
    Handles scalars, arrays, and nested dicts (like psychometric params).
    """
    flat = []
    
    for name, value in stats_dict.items():
        if isinstance(value, dict):
            # Nested dict (e.g., psychometric params)
            for k, v in value.items():
                v = np.atleast_1d(v)
                flat.extend(v.flatten())
        else:
            value = np.atleast_1d(value)
            flat.extend(value.flatten())
    
    return np.array(flat, dtype=np.float64)


def get_stat_names_expanded(stat_names: Optional[List[str]] = None) -> List[str]:
    """
    Get expanded list of stat names (for labelling flattened array).
    
    Handles nested stats like 'psychometric' which expands to 4 params.
    """
    if stat_names is None:
        stat_names = DEFAULT_STATS
    
    expanded = []
    for name in stat_names:
        if name == 'psychometric':
            expanded.extend(['pse', 'slope', 'lapse_low', 'lapse_high'])
        elif name == 'binned_accuracy':
            expanded.extend([f'binned_acc_{i}' for i in range(8)])  # default n_bins=8
        elif name == 'binned_choice_prob':
            expanded.extend([f'binned_prob_{i}' for i in range(8)])
        else:
            expanded.append(name)
    
    return expanded


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_stats_for_sbi(
    choices: np.ndarray,
    stimuli: np.ndarray,
    categories: np.ndarray,
    stat_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Convenience function for SBI simulator.
    
    Returns flattened 1D array of summary statistics.
    """
    return compute_summary_stats(choices, stimuli, categories, stat_names, return_dict=False)


def describe_stats(stat_names: Optional[List[str]] = None) -> None:
    """Print descriptions of summary statistics."""
    if stat_names is None:
        stat_names = list_available_stats()
    
    print("Summary Statistics")
    print("=" * 60)
    
    for name in stat_names:
        if name in SUMMARY_REGISTRY:
            func = SUMMARY_REGISTRY[name]
            doc = func.__doc__ or "No description"
            # Get first line of docstring
            first_line = doc.strip().split('\n')[0]
            print(f"\n{name}:")
            print(f"  {first_line}")


# =============================================================================
# CUSTOM STAT EXAMPLE
# =============================================================================

def add_custom_stat(name: str, func: Callable) -> None:
    """
    Add a custom summary statistic.
    
    The function must have signature:
        func(choices, stimuli, categories) -> scalar, array, or dict
    
    Example:
        def my_stat(choices, stimuli, categories):
            return np.mean(choices) * np.std(stimuli)
        
        add_custom_stat('my_stat', my_stat)
    """
    SUMMARY_REGISTRY[name] = func
    print(f"Registered custom stat: '{name}'")
