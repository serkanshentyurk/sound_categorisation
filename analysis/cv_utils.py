"""
Grid-Search Cross-Validation Utilities

Shared helpers for the BE/SC model comparison pipeline:
- Data format conversion (SessionData → legacy flat DataFrame)
- Loading and aggregating cluster CV results
- ANOVA comparison
- Best-fit parameter extraction and model simulation

Session selection is handled by behav_utils.data.selection — this module
only handles the CV-specific operations.

Used by:
    cluster/run_cv_single.py
    cluster/gather_cv_results.py
    notebooks/3a_cv_grid_search
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
# LOAD CLUSTER RESULTS
# =============================================================================

def load_cv_pickles(
    results_dir: str,
    pattern: str = 'cv_*_seed*.pkl',
) -> Dict[str, List[Dict]]:
    """
    Load all per-seed CV pickle files from a directory.

    Expects filenames: cv_{ANIMAL}_seed{NNN}.pkl
    Each pickle contains: {'BE': {...}, 'SC': {...}}

    Returns:
        {animal_id: [result_dicts]}
    """
    full_pattern = os.path.join(results_dir, pattern)
    files = sorted(glob.glob(full_pattern))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No pickle files found matching {full_pattern}"
        )

    all_results = {}

    for fpath in files:
        fname = os.path.basename(fpath)
        parts = fname.replace('.pkl', '').split('_')
        seed_idx = next(
            i for i, p in enumerate(parts) if p.startswith('seed')
        )
        animal_id = '_'.join(parts[1:seed_idx])

        with open(fpath, 'rb') as f:
            data = pickle.load(f)

        if animal_id not in all_results:
            all_results[animal_id] = []

        for model_name in ['BE', 'SC']:
            if model_name in data:
                all_results[animal_id].append(data[model_name])

    return all_results


def summarise_loaded_results(all_results: Dict[str, List[Dict]]) -> None:
    """Print a summary of loaded CV results."""
    for aid in sorted(all_results):
        results = all_results[aid]
        n_be = sum(1 for r in results if r['model'] == 'BE')
        n_sc = sum(1 for r in results if r['model'] == 'SC')
        print(f"  {aid}: {n_be} BE seeds, {n_sc} SC seeds")


# =============================================================================
# DATAFRAME CONSTRUCTION
# =============================================================================

def build_long_df(all_results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Build tidy long-form DataFrame of test errors."""
    rows = []
    for aid, results_list in all_results.items():
        for r in results_list:
            rows.append({
                'animal_id': aid,
                'model': r['model'],
                'seed': r['seed'],
                'avg_test_error': r['avg_test_error'],
            })
    return pd.DataFrame(rows)


# =============================================================================
# STATISTICAL COMPARISON
# =============================================================================

def run_anova(
    long_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Per-animal one-way ANOVA comparing BE vs SC test errors.

    Returns:
        DataFrame: animal_id, be_mean, sc_mean, F_stat, p_value, winner
    """
    records = []
    for aid in sorted(long_df['animal_id'].unique()):
        be = long_df.loc[
            (long_df['animal_id'] == aid) & (long_df['model'] == 'BE'),
            'avg_test_error'
        ].dropna().values

        sc = long_df.loc[
            (long_df['animal_id'] == aid) & (long_df['model'] == 'SC'),
            'avg_test_error'
        ].dropna().values

        if len(be) < 2 or len(sc) < 2:
            warnings.warn(f"{aid}: insufficient seeds for ANOVA")
            continue

        F, p = f_oneway(be, sc)
        be_mean = float(np.mean(be))
        sc_mean = float(np.mean(sc))

        winner = 'Inconclusive'
        if p < alpha:
            winner = 'BE' if be_mean < sc_mean else 'SC'

        records.append({
            'animal_id': aid,
            'be_mean': be_mean,
            'sc_mean': sc_mean,
            'F_stat': float(F),
            'p_value': float(p),
            'winner': winner,
        })

    return pd.DataFrame(records)


# =============================================================================
# PARAMETER EXTRACTION
# =============================================================================

def get_best_seed_params(
    animal_results: List[Dict],
    model_name: str,
) -> Tuple[Optional[Any], Optional[int]]:
    """
    Find the seed with lowest avg test error for a given model,
    return its best parameters.

    Returns:
        (params, seed) where params is either:
        - dict (new format): {'sigma_percep': ..., 'A_repulsion': ..., ...}
        - list (old format): [sigma_noise, A_repulsion, x_val, y_val]
        - None if no valid results

    Use format_params() to normalise either format to a named dict.
    """
    model_runs = [r for r in animal_results if r['model'] == model_name]
    valid = [r for r in model_runs if not np.isnan(r['avg_test_error'])]
    if len(valid) == 0:
        return None, None

    best_run = min(valid, key=lambda r: r['avg_test_error'])

    # New format first
    if 'best_params_single' in best_run and best_run['best_params_single'] is not None:
        return best_run['best_params_single'], best_run['seed']

    # Old format fallback
    params = best_run['best_params']
    if params is not None and len(params) > 0:
        return params[0], best_run['seed']

    return None, None


def format_params(model_name: str, params) -> Dict[str, float]:
    """
    Convert params to named dict. Handles both formats:
    - Old (list): [sigma_noise, A_repulsion, x_val, y_val]
    - New (dict): {'sigma_percep': ..., 'A_repulsion': ..., ...}
    """
    if isinstance(params, dict):
        # New format — already named. Normalise sigma_percep → sigma_noise
        # for display consistency (both names are valid).
        out = dict(params)
        if 'sigma_percep' in out and 'sigma_noise' not in out:
            out['sigma_noise'] = out.pop('sigma_percep')
        return out

    # Old format: positional list
    sigma_noise, A_repulsion, x_val, y_val = params
    base = {'sigma_noise': sigma_noise, 'A_repulsion': A_repulsion}
    if model_name == 'BE':
        base['eta_relax'] = x_val
        base['eta_learning'] = y_val
    elif model_name == 'SC':
        base['sigma_update'] = x_val
        base['gamma'] = y_val
    return base


def extract_param_df(all_results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Extract best parameters from each seed/fold into a tidy DataFrame.

    Handles both old format (list) and new format (dict) for best_params.
    """
    rows = []
    for aid, results_list in all_results.items():
        for r in results_list:
            if r['best_params'] is None:
                continue
            for fold_idx, params in enumerate(r['best_params']):
                if params is None:
                    continue
                named = format_params(r['model'], params)
                row = {
                    'animal_id': aid,
                    'model': r['model'],
                    'seed': r['seed'],
                    'fold': fold_idx,
                    **named,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def build_summary_table(
    all_results: Dict[str, List[Dict]],
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per animal: winner, p-value, mean errors,
    and best-fit parameters for the winning model.
    """
    rows = []
    for _, row in comparison_df.iterrows():
        aid = row['animal_id']
        winner = row['winner']

        entry = {
            'animal_id': aid,
            'winner': winner,
            'p_value': row['p_value'],
            'be_mean_error': row['be_mean'],
            'sc_mean_error': row['sc_mean'],
        }

        if winner in ('BE', 'SC'):
            animal_results = all_results.get(aid, [])
            params, seed = get_best_seed_params(animal_results, winner)
            if params is not None:
                named = format_params(winner, params)
                entry['best_seed'] = seed
                entry.update({f'best_{k}': v for k, v in named.items()})

        rows.append(entry)

    return pd.DataFrame(rows)


# =============================================================================
# MODEL SIMULATION (using legacy code)
# =============================================================================

def compute_empirical_um(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical update matrix from a flat DataFrame.

    Returns (update_matrix, conditional_matrix) with rows reversed
    to match legacy plotting convention.
    """
    from legacy.fitter import post_correct_update_matrix

    s = df['stim_relative'].values
    chooseB = df['choice'].values
    reward = df['correct'].values
    no_resp = df['No_response'].values
    not_block = df['is_not_start_of_block'].values

    um, cm = post_correct_update_matrix(s, chooseB, reward, no_resp, not_block)
    return um[::-1], cm[::-1]


def simulate_model_um(
    df: pd.DataFrame,
    model_name: str,
    params: List[float],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate model with given params on the animal's stimulus sequence,
    then compute update matrix via legacy code.

    Args:
        df: Flat DataFrame (from sessions_to_old_df)
        model_name: 'BE' or 'SC'
        params: [sigma_noise, A_repulsion, x_axis_val, y_axis_val]
        seed: Random seed used during CV

    Returns:
        (model_update_matrix, model_conditional_matrix)
    """
    from legacy.fitter import post_correct_update_matrix
    from legacy.be import BE_model, Noise_generator, Delta_repulsion
    from legacy.sc import SC_model

    s = df['stim_relative'].values
    no_response = df['No_response'].values.astype(bool)
    not_blockstart = df['is_not_start_of_block'].values.astype(bool)
    categories = np.where(s > 0, 1, 0)

    sigma_noise, A_repulsion, x_val, y_val = params

    # Stimulus space (matching legacy convention)
    max_range = 1 + 6 * sigma_noise + 2 * A_repulsion * (1 + 6 * sigma_noise)
    min_range = -max_range
    num_points = round((max_range - min_range) * 1000)
    x = np.linspace(min_range, max_range, num_points)

    # Noisy perception
    s_tilde = s + Noise_generator(len(s), seed, sigma_noise)
    s_hat = Delta_repulsion(A_repulsion, s_tilde)

    if model_name == 'BE':
        y = sp_uniform.pdf(x, loc=min_range, scale=max_range - min_range)
        model_um, model_cm = BE_model(
            post_correct_update_matrix,
            x, y, s, s_hat, categories,
            sigma_noise, A_repulsion,
            y_val,   # eta_learning
            x_val,   # eta_relax
            no_response, not_blockstart, seed, 'simulated',
        )
    elif model_name == 'SC':
        model_um, model_cm = SC_model(
            post_correct_update_matrix,
            x, None, s, s_hat, categories,
            sigma_noise, A_repulsion,
            y_val,   # gamma
            x_val,   # sigma_update
            no_response, not_blockstart, seed, 'simulated',
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model_um, model_cm


def compute_matrix_error(model_um: np.ndarray, emp_um: np.ndarray) -> float:
    """MSE between model and empirical update matrices (NaN-safe)."""
    from legacy.fitter import matrix_error
    return float(matrix_error(model_um, emp_um))
