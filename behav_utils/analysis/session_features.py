"""
Session Feature Matrix Builder

Computes a (session x feature) DataFrame from AnimalData, suitable for
SLDS/HMM epoch categorisation and general behavioural analysis.

Combines:
    - Core summary statistics from summary_stats (choice-based)
    - RT-based features
    - Session-to-session deltas
    - Session metadata
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, List, Callable, Dict, Union, TYPE_CHECKING

from behav_utils.analysis.summary_stats import (
    SUMMARY_REGISTRY,
    FEATURE_MATRIX_STATS,
    compute_summary_stats,
)

if TYPE_CHECKING:
    from behav_utils.data.structures import AnimalData, SessionData, TrialData



# =============================================================================
# SINGLE SESSION FEATURES
# =============================================================================

def compute_session_features(
    session: 'SessionData',
    rt_extractor: Callable = default_rt_extractor,
    stat_names: Optional[List[str]] = None,
    hard_threshold: float = 0.3,
    fast_threshold: float = 50.0,
) -> Dict[str, float]:
    """
    Compute all features for a single session.

    No filtering. Data must be pre-filtered via session.filter().

    Returns dict of feature_name -> value (all scalars).
    """
    if stat_names is None:
        stat_names = FEATURE_MATRIX_STATS

    # Metadata
    features = {
        'animal_id': session.metadata.animal_id,
        'session_id': session.session_id,
        'session_idx': session.session_idx,
        'date': session.date,
        'stage': session.stage,
        'distribution': session.distribution,
    }

    # Get arrays (no filtering — session should be pre-filtered)
    arrays = session.get_arrays()

    stimuli = arrays['stimuli']
    categories = arrays['categories']
    choices = arrays['choices']
    no_response = arrays['no_response']

    features['n_trials_total'] = session.trials.n_trials
    features['n_trials_valid'] = int((~no_response).sum())
    features['n_trials_abort'] = int(session.trials.abort.sum())
    features['abort_rate'] = float(session.trials.abort.mean())

    if features['n_trials_valid'] < 10:
        warnings.warn(
            f"Session {session.session_id}: only {features['n_trials_valid']} valid trials."
        )

    # Core summary stats
    stats_dict = compute_summary_stats(
        choices, stimuli, categories,
        stat_names=stat_names,
        return_dict=True,
    )

    for stat_name, value in stats_dict.items():
        if isinstance(value, dict):
            for k, v in value.items():
                features[k] = float(v) if not isinstance(v, (str, type(None))) else v
        elif isinstance(value, np.ndarray):
            for i, v in enumerate(value):
                features[f'{stat_name}_{i}'] = float(v)
        else:
            features[stat_name] = float(value) if not isinstance(value, (str, type(None))) else value

    # RT features
    rt_raw = rt_extractor(session.trials)
    trial_indices = arrays['trial_indices']
    rt_filtered = rt_raw[trial_indices]

    rt_features = compute_rt_features(
        rt_filtered, stimuli, categories, choices,
        hard_threshold=hard_threshold,
        fast_threshold=fast_threshold,
    )
    features.update(rt_features)

    return features



# =============================================================================
# DELTA FEATURES
# =============================================================================

DELTA_FEATURES = [
    'pse', 'slope', 'accuracy', 'side_bias', 'recency',
    'choice_entropy', 'rt_median',
]



# =============================================================================
# METADATA & FEATURE UTILITIES
# =============================================================================

METADATA_COLUMNS = [
    'animal_id', 'session_id', 'session_idx', 'date', 'stage', 'distribution',
    'n_trials_total', 'n_trials_valid', 'n_trials_abort', 'abort_rate',
]