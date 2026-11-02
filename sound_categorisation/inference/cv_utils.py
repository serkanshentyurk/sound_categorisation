"""
Grid-Search Cross-Validation Utilities

Shared helpers for the BE/SC model comparison pipeline:
- Loading and aggregating cluster CV results into tidy DataFrames
- Compact parameter formatting for plot labels

Session selection is handled by behav_utils.data.selection — this module
only handles the CV-specific operations.

"""

import pickle
from collections import namedtuple
from collections.abc import Mapping
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

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
        NaN errors are dropped, so the list's positions are not reps: compare
        models with ``errors_by_rep``, never by pairing these lists.
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


def errors_by_rep(result_data: dict) -> Dict:
    """``{rep: test_error}`` from a CV results pickle: every entry, NaN errors kept.

    ``rep`` is the entry's ``'rep'`` (the seed for grid search, the repeat index for SBI), else its
    position. This is what ``compare_models`` pairs BE and SC on.
    """
    return {r.get('rep', i): float(r.get('test_error', np.nan))
            for i, r in enumerate(result_data.get('results', []))}


def _as_rep_dict(errors) -> Dict:
    if isinstance(errors, Mapping):
        return {k: float(v) for k, v in errors.items()}
    return {i: float(e) for i, e in enumerate(errors)}


def compare_models(
    animal_id: str,
    be_errors,
    sc_errors,
):
    """Within-method BE-vs-SC comparison for one animal (method-agnostic).

    Operates purely on stored error arrays, so it serves grid_search and every SBI rep
    identically; the file I/O lives in load_cv_results, the cross-method consensus in
    inference.consensus.

    ``be_errors`` / ``sc_errors`` are the held-out errors per rep: ``{rep: error}`` (what
    load_cv_results passes; the rep is the grid-search seed or the SBI repeat), or sequences
    whose position is the rep. The models are compared on the reps where both have a finite
    error; a NaN or a missing rep on either side drops that pair and never shifts the others.
    (Up to 0.7.0 the NaNs were dropped from each list separately and the rest paired by
    position, so one failed seed misaligned every later pair.) Winner = the lower mean over
    those pairs; p_value = paired Wilcoxon over them (NaN when it cannot be computed, e.g.
    all differences zero); n_pairs = how many pairs that was.

    The call follows the manuscript's model selection (Methods, "Model selection"): the
    cross-validation is repeated over seeds and the lower-error model is called when the
    difference is significant (p < 0.05). The manuscript used an ANOVA over 64 seeds; here it
    is a paired Wilcoxon over the seeds (grid search) or the posterior-resampling repeats
    (SBI). Seeds and repeats vary the simulation noise and posterior samples, not the data
    split, so p describes how stable the call is under that noise.

    Args:
        animal_id: Animal identifier.
        be_errors: BE held-out errors per rep (mapping or sequence).
        sc_errors: SC held-out errors per rep (mapping or sequence).

    Returns:
        (long_df, comparison_df), or (None, None) when no rep has a finite error for both.
        long_df holds the paired reps only (column 'seed' = rep, for plot_cv_comparison).
    """
    from scipy.stats import wilcoxon

    be, sc = _as_rep_dict(be_errors), _as_rep_dict(sc_errors)
    reps = [r for r in be if r in sc and np.isfinite(be[r]) and np.isfinite(sc[r])]
    if not reps:
        return None, None
    try:
        reps = sorted(reps)
    except TypeError:
        pass
    be_vals = np.array([be[r] for r in reps])
    sc_vals = np.array([sc[r] for r in reps])

    long_df = pd.DataFrame(
        [{'animal_id': animal_id, 'model': 'BE', 'seed': r, 'avg_test_error': v} for r, v in zip(reps, be_vals)]
        + [{'animal_id': animal_id, 'model': 'SC', 'seed': r, 'avg_test_error': v} for r, v in zip(reps, sc_vals)])

    be_mean, sc_mean = float(np.mean(be_vals)), float(np.mean(sc_vals))
    winner = 'BE' if be_mean < sc_mean else 'SC'
    try:
        p_val = float(wilcoxon(be_vals, sc_vals).pvalue)
    except ValueError:
        p_val = np.nan

    comparison_df = pd.DataFrame([{
        'animal_id': animal_id, 'winner': winner, 'p_value': p_val,
        'be_mean': be_mean, 'sc_mean': sc_mean, 'n_pairs': len(reps),
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
    true_model: str | None = None,
    true_params: dict | None = None,
    metadata: dict | None = None,
    posterior: dict | None = None,
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
        posterior: SBI only -- ``selection.condition_posterior`` output (saved
            samples, param_names, point_estimate, prior_bounds); None for grid search.
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
        'posterior': posterior,
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
      - comparison: one row per animal — winner, p_value, be_mean, sc_mean, n_pairs, plus
                    true_model and a `correct` flag (winner == true_model). BE and SC are
                    paired by rep (see compare_models).
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

        _, be_best = compute_seed_errors(be)
        _, sc_best = compute_seed_errors(sc)
        long_df, comp_df = compare_models(aid, errors_by_rep(be), errors_by_rep(sc))
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
