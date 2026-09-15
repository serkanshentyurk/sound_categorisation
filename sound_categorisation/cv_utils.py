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
from pathlib import Path
from collections import namedtuple
from typing import Optional, List, Dict, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from behav_utils.data.structures import AnimalData, SessionData

# Re-export for convenience — callers can import from here or from selection
from behav_utils.data.ops.selection import select_sessions

# Tidy bundle returned by load_cv_results. Cross-method: the same loader serves
# grid-search and SBI validation, since both write the schema via save_cv_result.
CVResults = namedtuple('CVResults', ['long', 'comparison', 'recovery'])


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


def compute_seed_errors(result_data: dict):
    """
    Extract per-rep errors and best params from a CV results pickle.

    Cross-method: reads the neutral schema written by save_cv_result, so the
    same helper serves grid-search and SBI results. (The grid-search compute
    layer still returns 'avg_test_error'/'best_params_single'; save_cv_result
    maps those to the neutral 'test_error'/'best_params' on write.)

    Args:
        result_data: Dict loaded from a CV results pickle. Expected key
            'results' — a list of per-rep dicts, each with 'test_error' and
            'best_params'.

    Returns:
        (errors, best_params) where errors is a list of floats and best_params
        is the params dict from the lowest-error rep, or None if none valid.
    """
    results = result_data.get('results', [])
    errors = [r['test_error'] for r in results
              if not np.isnan(r.get('test_error', np.nan))]
    valid = [r for r in results
             if not np.isnan(r.get('test_error', np.nan))
             and r.get('best_params')]
    best_params = (min(valid, key=lambda r: r['test_error'])['best_params']
                   if valid else None)
    return errors, best_params


def compare_models(
    animal_id: str,
    be_errors,
    sc_errors,
):
    """Within-method BE-vs-SC comparison for one animal (method-agnostic).

    Operates purely on stored error arrays, so it serves grid_search and every
    SBI rep identically; the file I/O lives in load_cv_results, the cross-method
    consensus in analysis.consensus. The rep axis is whatever produced the
    arrays -- GS seeds, SBI posterior-resampling repeats, or (single rep)
    sessions; it is labelled 'seed' in long_df only for plot_cv_comparison
    compatibility. Winner is the lower mean; p_value is a paired Wilcoxon over
    the rep axis (NaN when too few pairs or all-equal).

    Args:
        animal_id: Animal identifier.
        be_errors: Per-rep BE held-out errors.
        sc_errors: Per-rep SC held-out errors.

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


# Back-compat alias (notebooks import compute_cv_dataframes).
compute_cv_dataframes = compare_models


# =============================================================================
# CROSS-METHOD SCHEMA: WRITE + LOAD
# =============================================================================

def save_cv_result(
    path,
    animal_id: str,
    model: str,
    results: list,
    fit_target: str,
    true_model: Optional[str] = None,
    true_params: Optional[dict] = None,
    metadata: Optional[dict] = None,
):
    """
    Write one (animal, model) CV result in the neutral cross-method schema.

    Single definition of the on-disk shape, shared by grid-search and SBI and
    by quick (notebook) and full (cluster) runs, so every writer produces an
    identical structure that load_cv_results can read.

    Args:
        path: Output .pkl path (parent dirs created if needed).
        animal_id: Animal identifier.
        model: Fitted model, 'BE' or 'SC'.
        results: List of per-rep dicts, each {'rep', 'test_error', 'best_params'}.
        fit_target: 'update_matrix' or 'conditional_psych'.
        true_model: Ground-truth generating model for synthetic data, else None.
        true_params: Ground-truth params dict for synthetic data, else None.
        metadata: Optional run metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'animal_id': animal_id,
        'model': model,
        'true_model': true_model,
        'true_params': true_params,
        'fit_target': fit_target,
        'results': results,
        'metadata': metadata or {},
    }
    with open(path, 'wb') as f:
        pickle.dump(payload, f)


def load_cv_results(results_dir) -> CVResults:
    """
    Load a directory of CV result pickles into tidy DataFrames.

    The single loader for both grid-search and SBI validation — they share the
    schema written by save_cv_result. Globs the directory's top-level '*.pkl'
    (per-seed partials live in a partials/ subdir and are skipped), pairs the
    BE and SC fit for each animal, and returns three frames:

      - long:       per-rep errors, columns animal_id, model, seed, avg_test_error
                    (feeds plotting.cv.plot_cv_comparison).
      - comparison: one row per animal — winner, p_value, be_mean, sc_mean, plus
                    true_model and a `correct` flag (winner == true_model).
      - recovery:   true-model params vs recovered, columns animal_id, true_model,
                    correct, param, true_value, recovered_value. Recovery uses the
                    fit of the *true* model (decoupled from identification).

    Args:
        results_dir: Directory containing {animal}_{model}.pkl files (any
            partials/ subdir is skipped).

    Returns:
        CVResults(long, comparison, recovery). Frames are empty if nothing loads
        or no animal has both a BE and an SC fit.
    """
    results_dir = Path(results_dir)

    by_animal: Dict[str, Dict[str, dict]] = {}
    for pkl in sorted(results_dir.glob('*.pkl')):
        with open(pkl, 'rb') as f:
            d = pickle.load(f)
        by_animal.setdefault(d['animal_id'], {})[d['model']] = d

    long_parts, comp_parts, recov_rows = [], [], []
    for aid, models in by_animal.items():
        be, sc = models.get('BE'), models.get('SC')
        if be is None or sc is None:
            continue

        be_errors, be_best = compute_seed_errors(be)
        sc_errors, sc_best = compute_seed_errors(sc)
        long_df, comp_df = compare_models(aid, be_errors, sc_errors)
        if long_df is None:
            continue

        # Ground truth is identical across an animal's BE/SC pickles; take either.
        true_model = be.get('true_model')
        true_params = be.get('true_params')

        comp_df = comp_df.copy()
        comp_df['true_model'] = true_model
        comp_df['correct'] = comp_df['winner'] == true_model
        long_parts.append(long_df)
        comp_parts.append(comp_df)

        # Recovery: true-model fit vs its true params (independent of the ID call).
        if true_model in ('BE', 'SC') and true_params:
            best = be_best if true_model == 'BE' else sc_best
            if best:
                correct = bool(comp_df['correct'].iloc[0])
                for param, true_value in true_params.items():
                    if param in best:
                        recov_rows.append({
                            'animal_id': aid,
                            'true_model': true_model,
                            'correct': correct,
                            'param': param,
                            'true_value': float(true_value),
                            'recovered_value': float(best[param]),
                        })

    long_all = (pd.concat(long_parts, ignore_index=True)
                if long_parts else pd.DataFrame())
    comp_all = (pd.concat(comp_parts, ignore_index=True)
                if comp_parts else pd.DataFrame())
    recov_all = pd.DataFrame(recov_rows) if recov_rows else pd.DataFrame()
    return CVResults(long_all, comp_all, recov_all)
