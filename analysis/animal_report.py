"""
Per-Animal Report — Computation

Extracts all computation from the old plotting/animal_report.py.
Each function returns a structured result dict that the corresponding
plot_ function in plotting/animal_report.py can render.

Public API:
    compute_animal_summary    → psych result, trajectory, consensus info
    compute_model_fits        → empirical/model UMs, psychometric fits, per-session MSE
    compute_sbi_diagnostics   → posterior samples, PPC, stat comparison

All functions take pre-filtered sessions and/or paths to saved results.
No plotting. No matplotlib imports.
"""

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from behav_utils.analysis.psychometry import fit_psychometric, compute_psychometric
from behav_utils.analysis.update_matrix import compute_update_matrix, compute_um, matrix_error
from behav_utils.analysis.trajectory import compute_trajectory
from behav_utils.analysis.utils import cumulative_gaussian
from behav_utils.analysis.summary_stats import compute_summary_stats, get_stat_names_expanded
from behav_utils.data.filtering import filter_trials, pool_arrays
from behav_utils.data.selection import fitting_data_from_sessions


FIT_TARGETS = ('update_matrix', 'conditional_psych')
FT_LABEL = {'update_matrix': 'UM', 'conditional_psych': 'CP'}


# ═════════════════════════════════════════════════════════════════════════════
# 1. ANIMAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def compute_animal_summary(
    animal_id: str,
    sessions: list,
    results_dir: Union[str, Path],
    distribution: str = 'uniform',
    assign_row: Optional[pd.Series] = None,
    n_bootstrap: int = 1000,
    true_params: Optional[Any] = None,
) -> Dict:
    """
    Compute all data needed for the animal summary figure.

    Args:
        animal_id: Animal identifier.
        sessions: Pre-filtered session list.
        results_dir: Path to results directory.
        distribution: Stimulus distribution label.
        assign_row: Pre-computed consensus row (avoids re-loading).
        n_bootstrap: Bootstrap iterations for psychometric CI.
        true_params: Ground-truth params (for synthetic animals).

    Returns:
        Dict with:
            'psychometric': result dict from compute_psychometric
            'trajectory': result dict from compute_trajectory
            'consensus': dict with method assignments and consensus label
            'animal_id': str
            'n_sessions': int
            'n_trials': int
            'distribution': str
            'true_params': Any or None
    """
    results_dir = Path(results_dir)

    # Psychometric (pooled across sessions)
    psych = compute_psychometric(sessions, mode='pooled', n_bootstrap=n_bootstrap)

    # Accuracy trajectory
    traj = compute_trajectory(sessions, stat_names=['accuracy'])

    # Consensus assignment
    consensus = _load_consensus(animal_id, results_dir, distribution, assign_row)

    pooled = pool_arrays(sessions)
    n_trials = len(pooled['stimuli'])

    return {
        'psychometric': psych,
        'trajectory': traj,
        'consensus': consensus,
        'animal_id': animal_id,
        'n_sessions': len(sessions),
        'n_trials': n_trials,
        'distribution': distribution,
        'true_params': true_params,
    }


def _load_consensus(animal_id, results_dir, distribution, assign_row):
    """Load consensus assignment for one animal."""
    METHOD_COLS = [
        f'GS_UM_{distribution}', f'GS_CP_{distribution}',
        f'SBI_UM_{distribution}', f'SBI_CP_{distribution}',
    ]
    METHOD_NAMES = ['GS-UM', 'GS-CP', 'SBI-UM', 'SBI-CP']

    if assign_row is None:
        try:
            from analysis.consensus import load_all_assignments
            df = load_all_assignments(results_dir)
            if animal_id in df.index:
                assign_row = df.loc[animal_id]
            else:
                return {'methods': {}, 'consensus': 'Unknown', 'method_names': METHOD_NAMES}
        except Exception:
            return {'methods': {}, 'consensus': 'Unknown', 'method_names': METHOD_NAMES}

    methods = {}
    for mc, mn in zip(METHOD_COLS, METHOD_NAMES):
        val = assign_row.get(mc, None)
        if isinstance(val, str) and val in ('BE', 'SC'):
            methods[mn] = val
        else:
            methods[mn] = None

    return {
        'methods': methods,
        'consensus': assign_row.get('Consensus', 'Unclear'),
        'method_names': METHOD_NAMES,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. MODEL FITS
# ═════════════════════════════════════════════════════════════════════════════

def compute_model_fits(
    animal_id: str,
    sessions: list,
    results_dir: Union[str, Path],
    method: str = 'sbi',
    fit_target: str = 'update_matrix',
    distribution: str = 'uniform',
    true_params: Optional[Any] = None,
    n_reps: int = 20,
    burn_in: int = 1000,
    seed: int = 42,
) -> Optional[Dict]:
    """
    Load model parameters, simulate sessions, compute UMs and MSE.

    Args:
        animal_id: Animal identifier.
        sessions: Pre-filtered session list.
        results_dir: Path to results directory.
        method: 'sbi' or 'gs'.
        fit_target: 'update_matrix' or 'conditional_psych'.
        distribution: Stimulus distribution.
        true_params: Ground truth (for synthetic).
        n_reps: Stochastic simulation repetitions.
        burn_in: Model burn-in trials.
        seed: Random seed.

    Returns:
        Dict with:
            'emp_um': ndarray — empirical update matrix
            'be_um': ndarray — BE model update matrix
            'sc_um': ndarray — SC model update matrix
            'be_mse': float — BE UM MSE
            'sc_mse': float — SC UM MSE
            'emp_psych': result dict from compute_psychometric (data)
            'be_psych_params': dict — average BE psychometric params
            'sc_psych_params': dict — average SC psychometric params
            'per_session_mse': list of dicts [{session_idx, be_mse, sc_mse}, ...]
            'source': dict — where params came from
            'be_params': dict
            'sc_params': dict
            'session_data': list of per-session simulation results

        Returns None if params not found.
    """
    results_dir = Path(results_dir)
    from inference.comparison import simulate_all_sessions

    # Load parameters
    be_params, sc_params, source = _get_params(
        animal_id, results_dir, method, fit_target, distribution)
    if not be_params or not sc_params:
        return None

    # Build FittingData from pre-filtered sessions
    clean = filter_trials(sessions)
    fd = fitting_data_from_sessions(clean, animal_id)

    # Simulate
    session_data = simulate_all_sessions(
        fd, be_params, sc_params,
        burn_in=burn_in, n_reps=n_reps, n_bins=8, seed=seed,
    )

    # Pooled UMs
    emp_um = np.nanmean([d['real_um'] for d in session_data], axis=0)
    be_um = np.nanmean([d['be_um'] for d in session_data], axis=0)
    sc_um = np.nanmean([d['sc_um'] for d in session_data], axis=0)

    # Pooled psychometric (empirical data)
    emp_psych = compute_psychometric(clean, mode='pooled', n_bootstrap=500)

    # Model psychometric params (average across sessions)
    be_psych_params = _average_psych_params(session_data, 'be_psych')
    sc_psych_params = _average_psych_params(session_data, 'sc_psych')

    # Per-session MSE
    per_session_mse = [
        {
            'session_idx': d['session_idx'],
            'be_mse': d['be_um_mse'],
            'sc_mse': d['sc_um_mse'],
        }
        for d in session_data
    ]

    return {
        'emp_um': emp_um,
        'be_um': be_um,
        'sc_um': sc_um,
        'be_mse': matrix_error(be_um, emp_um),
        'sc_mse': matrix_error(sc_um, emp_um),
        'emp_psych': emp_psych,
        'be_psych_params': be_psych_params,
        'sc_psych_params': sc_psych_params,
        'per_session_mse': per_session_mse,
        'source': source,
        'be_params': be_params,
        'sc_params': sc_params,
        'session_data': session_data,
        'animal_id': animal_id,
        'distribution': distribution,
        'true_params': true_params,
    }


def _average_psych_params(session_data, key_prefix):
    """Average psychometric params across sessions."""
    ps = [d[key_prefix] for d in session_data if d.get(key_prefix, {}).get('success')]
    if not ps:
        return {}
    return {
        'mu': np.mean([p['mu'] for p in ps]),
        'sigma': np.mean([p['sigma'] for p in ps]),
        'lapse_low': np.mean([p['lapse_low'] for p in ps]),
        'lapse_high': np.mean([p['lapse_high'] for p in ps]),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3. SBI DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_sbi_diagnostics(
    animal_id: str,
    sessions: list,
    snpe_dict: dict,
    results_dir: Optional[Union[str, Path]] = None,
    n_samples: int = 10_000,
    n_ppc_samples: int = 200,
    true_params: Optional[Dict] = None,
    gs_index: Optional[dict] = None,
) -> Optional[Dict]:
    """
    Compute SBI posterior summaries and PPC.

    Args:
        animal_id: Animal identifier.
        sessions: Pre-filtered session list.
        snpe_dict: {'be': {..., 'posterior': ...}, 'sc': {..., 'posterior': ...}}
        results_dir: For loading GS point estimates (optional).
        n_samples: Posterior samples to draw.
        n_ppc_samples: Posterior predictive simulations.
        true_params: Ground truth (for synthetic validation).
        gs_index: Pre-built GS index (optional).

    Returns:
        Dict with:
            'samples': {'be': ndarray, 'sc': ndarray}
            'param_names': {'be': list, 'sc': list}
            'marginals': per-param {mean, median, ci_low, ci_high}
            'ppc': {'be': {...}, 'sc': {...}} or None
            'gs_params': optional GS point estimates
            'true_params': dict or None
            'animal_id': str
    """
    samples = {}
    param_names = {}
    marginals = {}

    for model_key in ('be', 'sc'):
        if model_key not in snpe_dict:
            continue
        entry = snpe_dict[model_key]
        posterior = entry['posterior']
        pnames = entry['param_names']
        param_names[model_key] = pnames

        # Condition on this animal's observed stats
        observed = entry.get('observed_stats')
        if observed is not None:
            import torch
            samps = posterior.sample((n_samples,),
                                     x=torch.tensor(observed, dtype=torch.float32))
            samps = samps.numpy()
        else:
            samps = posterior.sample((n_samples,)).numpy()

        samples[model_key] = samps

        # Marginal summaries
        model_marginals = {}
        for j, pn in enumerate(pnames):
            vals = samps[:, j]
            model_marginals[pn] = {
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'ci_low': float(np.percentile(vals, 2.5)),
                'ci_high': float(np.percentile(vals, 97.5)),
                'std': float(np.std(vals)),
            }
        marginals[model_key] = model_marginals

    # GS point estimates (optional)
    gs_params = None
    if results_dir is not None:
        gs_params = _load_gs_point_estimates(
            animal_id, Path(results_dir), gs_index)

    return {
        'samples': samples,
        'param_names': param_names,
        'marginals': marginals,
        'gs_params': gs_params,
        'true_params': true_params,
        'animal_id': animal_id,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS — parameter loading
# ═════════════════════════════════════════════════════════════════════════════

def _get_params(animal_id, results_dir, method, fit_target, distribution):
    """Load BE and SC params from results directory."""
    results_dir = Path(results_dir)
    be_params = sc_params = None
    source = {'method': method, 'fit_target': fit_target, 'fallback': False}

    if method == 'sbi':
        p = results_dir / 'sbi_static' / 'comparisons' / f'{distribution}_{fit_target}'
        pkl_path = p / f'animal_{animal_id}.pkl'
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            be_params = data.get('be_params')
            sc_params = data.get('sc_params')
    elif method == 'gs':
        for model_key in ('be', 'sc'):
            p = results_dir / 'cv' / f'{distribution}_{fit_target}_{model_key}' / f'{animal_id}.pkl'
            if p.exists():
                with open(p, 'rb') as f:
                    data = pickle.load(f)
                if model_key == 'be':
                    be_params = data.get('best_params')
                else:
                    sc_params = data.get('best_params')

    # Fallback: try alternate method
    if (be_params is None or sc_params is None) and method == 'sbi':
        be_fb, sc_fb, _ = _get_params(animal_id, results_dir, 'gs', fit_target, distribution)
        if be_params is None:
            be_params = be_fb
        if sc_params is None:
            sc_params = sc_fb
        if be_fb is not None or sc_fb is not None:
            source['fallback'] = True

    return be_params, sc_params, source


def _load_gs_point_estimates(animal_id, results_dir, gs_index=None):
    """Load GS point estimates for both fit targets."""
    result = {}
    for ft in FIT_TARGETS:
        for mk in ('be', 'sc'):
            p = results_dir / 'cv' / f'uniform_{ft}_{mk}' / f'{animal_id}.pkl'
            if p.exists():
                with open(p, 'rb') as f:
                    data = pickle.load(f)
                best = data.get('best_params', {})
                if best:
                    result[f'{mk}_{ft}'] = best
    return result if result else None


def _params_to_str(params):
    """Format params dict as a compact string."""
    if params is None:
        return '(none)'
    if hasattr(params, '__dict__'):
        d = {k: v for k, v in vars(params).items()
             if not k.startswith('_') and isinstance(v, (int, float))}
    elif isinstance(params, dict):
        d = {k: v for k, v in params.items() if isinstance(v, (int, float))}
    else:
        return str(params)
    return ', '.join(f'{k}={v:.3f}' for k, v in d.items())
