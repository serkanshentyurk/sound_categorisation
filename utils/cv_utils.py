"""
Grid-Search Cross-Validation Utilities

Shared helpers for the BE/SC model comparison pipeline:
- Loading and aggregating cluster CV results into tidy DataFrames
- Compact parameter formatting for plot labels

Session selection is handled by behav_utils.data.selection — this module
only handles the CV-specific operations.

"""

import numpy as np
import pandas as pd
import pickle
from typing import Optional, List, Dict, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from behav_utils.data.structures import AnimalData, SessionData

# Re-export for convenience — callers can import from here or from selection
from behav_utils.data.ops.selection import select_sessions


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
    except ValueError:
        p_val = np.nan

    comparison_df = pd.DataFrame([{
        'animal_id': animal_id, 'winner': winner, 'p_value': p_val,
        'be_mean': be_mean, 'sc_mean': sc_mean,
    }])
    return long_df, comparison_df
