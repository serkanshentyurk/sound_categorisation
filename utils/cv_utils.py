"""
Grid-Search Cross-Validation Utilities

Shared helpers for the BE/SC model comparison pipeline:
- Data format conversion (SessionData → legacy flat DataFrame)
- Loading and aggregating cluster CV results
- ANOVA comparison
- Best-fit parameter extraction and model simulation

Session selection is handled by behav_utils.data.selection — this module
only handles the CV-specific operations.

"""

import numpy as np
import pandas as pd
import pickle
import os
import glob
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, TYPE_CHECKING
from scipy.stats import f_oneway, uniform as sp_uniform

if TYPE_CHECKING:
    from behav_utils.data.structures import AnimalData, SessionData

# Re-export for convenience — callers can import from here or from selection
from behav_utils.data.selection import select_sessions


# =============================================================================
# DATA FORMAT CONVERSION
# =============================================================================

def sessions_to_old_df(
    sessions: List['SessionData'],
    animal_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert behav_utils SessionData objects into the flat DataFrame
    format expected by the legacy Fitter code.

    Each session becomes a 'block'. Aborts and opto trials are excluded.
    No-response trials are kept but flagged.

    Output columns:
        stim_relative, choice, correct, No_response, block,
        Trial, is_not_start_of_block, [Participant_ID]
    """
    all_rows = []

    for block_id, session in enumerate(sessions):
        # No filtering — sessions should be pre-filtered
        arrays = session.get_arrays()
        stim = arrays['stimuli']
        choice = arrays['choices']
        n = arrays['n_trials']
        if n == 0:
            continue

        no_response = arrays['no_response']
        correct = arrays['categories'] == choice  # recompute from arrays
        choice_clean = np.where(no_response, 0, choice).astype(int)
        correct_clean = np.where(no_response, 0, correct).astype(int)

        df_block = pd.DataFrame({
            'stim_relative': stim,
            'choice': choice_clean,
            'correct': correct_clean,
            'No_response': no_response,
            'block': block_id,
            'Trial': np.arange(1, n + 1),
        })
        all_rows.append(df_block)

    if len(all_rows) == 0:
        raise ValueError("No valid trials found")

    df = pd.concat(all_rows, ignore_index=True)
    df['is_not_start_of_block'] = df['block'].eq(df['block'].shift())

    if animal_id is not None:
        df['Participant_ID'] = animal_id

    return df


# =============================================================================
# MODEL SIMULATION 
# =============================================================================

import numpy as np
from typing import List, Tuple

from behav_utils.analysis.update_matrix import compute_update_matrix


def compute_empirical_um(df) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical update matrix from a flat DataFrame.

    Uses behav_utils.analysis.update_matrix.compute_update_matrix instead
    of legacy.fitter.post_correct_update_matrix.

    Args:
        df: Flat DataFrame with columns 'stim_relative', 'choice',
            'No_response', 'is_not_start_of_block'.

    Returns:
        (update_matrix, conditional_matrix)
    """
    s = df['stim_relative'].values
    ch = df['choice'].values
    cat = np.where(s > 0, 1, 0)
    no_resp = df['No_response'].values.astype(bool)
    nbs = df['is_not_start_of_block'].values.astype(bool)

    um, cm, _ = compute_update_matrix(
        s, ch, cat, n_bins=8,
        trial_filter='post_correct',
        no_response=no_resp,
        not_blockstart=nbs,
    )
    return um, cm


def simulate_model_um(
    df,
    model_name: str,
    params: List[float],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a model with given params on the animal's stimulus sequence,
    then compute the update matrix.

    Uses models/BE_core.py and models/SC_core.py instead of legacy.

    Args:
        df: Flat DataFrame with stimulus sequence.
        model_name: 'BE' or 'SC'.
        params: [sigma_noise, A_repulsion, x_axis_val, y_axis_val].
        seed: Random seed.

    Returns:
        (model_update_matrix, model_conditional_matrix)
    """
    from models.BE_core import BEModel, BEParams
    from models.SC_core import SCModel, SCParams
    from models.perception import perceive_stimulus

    s = df['stim_relative'].values
    no_response = df['No_response'].values.astype(bool)
    not_blockstart = df['is_not_start_of_block'].values.astype(bool)
    categories = np.where(s > 0, 1, 0)

    sigma_noise, A_repulsion, x_val, y_val = params
    
    rng = np.random.default_rng(seed)
    
    if model_name == 'BE':
        be_params = BEParams(
            sigma_percep=sigma_noise,
            A_repulsion=A_repulsion,
            eta_learning=y_val,
            eta_relax=x_val,
        )
        choices, _, _, _ = BEModel.simulate_session(
            stimuli=s, categories=categories,
            params=be_params,
            initial_state=None,
            rng=rng,
            no_response=no_response,
            not_blockstart=not_blockstart,
        )

    elif model_name == 'SC':
        sc_params = SCParams(
            sigma_percep=sigma_noise,
            A_repulsion=A_repulsion,
            gamma=y_val,
            sigma_update=x_val,
        )
        choices, _, _, _ = SCModel.simulate_session(
            stimuli=s, categories=categories,
            params=sc_params,
            initial_state=None,
            rng=rng,
            no_response=no_response,
            not_blockstart=not_blockstart,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Compute UM from simulated choices
    um, cm, _ = compute_update_matrix(
        s, choices, categories, n_bins=8,
        trial_filter='post_correct',
        no_response=no_response,
        not_blockstart=not_blockstart,
    )
    return um, cm


def params_to_str(params) -> str:
    """Format params (dict or dataclass) as compact string."""
    if params is None:
        return ''
    if hasattr(params, '__dict__'):
        d = {k: v for k, v in vars(params).items()
             if not str(k).startswith('_') and isinstance(v, (int, float))}
    elif isinstance(params, dict):
        d = {k: v for k, v in params.items() if isinstance(v, (int, float))}
    else:
        return str(params)
    return ', '.join(f'{k}={v:.3f}' for k, v in d.items())


def compute_gs_seed_errors(gs_data: dict):
    """
    Extract per-seed errors and best params from a raw GS pickle.

    Shared utility — used by notebooks for both real and synthetic data.

    Args:
        gs_data: Dict loaded from a grid-search results pickle.
                 Expected keys: 'results' (list of per-seed dicts).

    Returns:
        (errors, best_params) where errors is a list of floats and
        best_params is the params dict from the lowest-error seed,
        or None if no valid seeds.
    """
    results = gs_data.get('results', [])
    errors = [r['avg_test_error'] for r in results
              if not np.isnan(r.get('avg_test_error', np.nan))]
    valid = [r for r in results
             if not np.isnan(r.get('avg_test_error', np.nan))
             and r.get('best_params_single')]
    best_params = (min(valid, key=lambda r: r['avg_test_error'])['best_params_single']
                   if valid else None)
    return errors, best_params


def compute_cv_dataframes(
    animal_id: str,
    be_errors,
    sc_errors,
):
    """
    Build (long_df, comparison_df) from raw per-seed error arrays.

    Used by notebooks to feed into plotting.cv.plot_cv_comparison().
    Uses Wilcoxon signed-rank test on paired seeds.

    Args:
        animal_id: Animal identifier.
        be_errors: List of per-seed BE test errors.
        sc_errors: List of per-seed SC test errors.

    Returns:
        (long_df, comparison_df) or (None, None) if either list is empty.
    """
    from scipy.stats import wilcoxon

    if not be_errors or not sc_errors:
        return None, None

    rows = []
    for i, e in enumerate(be_errors):
        if not np.isnan(e):
            rows.append({'animal_id': animal_id, 'model': 'BE',
                         'seed': i, 'avg_test_error': e})
    for i, e in enumerate(sc_errors):
        if not np.isnan(e):
            rows.append({'animal_id': animal_id, 'model': 'SC',
                         'seed': i, 'avg_test_error': e})

    if not rows:
        return None, None

    long_df = pd.DataFrame(rows)
    be_vals = long_df[long_df['model'] == 'BE']['avg_test_error'].values
    sc_vals = long_df[long_df['model'] == 'SC']['avg_test_error'].values

    if len(be_vals) == 0 or len(sc_vals) == 0:
        return None, None

    be_mean, sc_mean = np.mean(be_vals), np.mean(sc_vals)
    winner = 'BE' if be_mean < sc_mean else 'SC'

    n_paired = min(len(be_vals), len(sc_vals))
    try:
        _, p_val = wilcoxon(be_vals[:n_paired], sc_vals[:n_paired])
    except Exception:
        p_val = np.nan

    comparison_df = pd.DataFrame([{
        'animal_id': animal_id, 'winner': winner, 'p_value': p_val,
        'be_mean': be_mean, 'sc_mean': sc_mean,
    }])
    return long_df, comparison_df
