"""
Synthetic Validation Report — Computation

Extracted from plotting/validation_report.py. Each function returns
a structured dict for the corresponding plot_ function.

Pure computation functions (build_confusion_matrix, build_gs_index,
extract_gs_recovery, extract_sbi_recovery) stay here as they were
always analysis, not plotting.

Public API:
    compute_synth_summary       → psych + um for one synthetic animal
    compute_synth_model_fits    → simulate models, compute UMs, MSE
    compute_synth_sbi_diagnostics → posterior samples + marginals
    build_confusion_matrix      → 2x2 confusion matrix from df
    build_gs_index              → nested index from raw GS pickles
    extract_gs_recovery         → true vs recovered params from GS
    extract_sbi_recovery        → true vs recovered params from SBI
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from behav_utils.analysis.psychometry import compute_psychometric
from behav_utils.analysis.update_matrix import compute_um, compute_update_matrix, matrix_error
from behav_utils.analysis.utils import cumulative_gaussian
from behav_utils.analysis.summary_stats import compute_summary_stats
from behav_utils.data.filtering import filter_trials, pool_arrays
from behav_utils.data.selection import fitting_data_from_sessions

FT_LABEL = {'update_matrix': 'UM', 'conditional_psych': 'CP'}


# ═════════════════════════════════════════════════════════════════════════════
# 1. SYNTH SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def compute_synth_summary(
    sa: dict,
    n_bootstrap: int = 1000,
) -> Dict:
    """
    Compute psychometric + UM for a synthetic animal.

    Args:
        sa: Synthetic animal dict with 'sessions', 'animal_id',
            'true_model', 'true_params'.
        n_bootstrap: Bootstrap iterations for psychometric CI.

    Returns:
        Dict with:
            'psychometric': result from compute_psychometric
            'um': result from compute_um
            'animal_id', 'true_model', 'true_params', 'n_sessions', 'n_trials'
    """
    sessions = sa['sessions']
    clean = filter_trials(sessions)

    psych = compute_psychometric(clean, mode='pooled', n_bootstrap=n_bootstrap)
    um = compute_um(clean)

    pooled = pool_arrays(clean)
    n_trials = len(pooled['stimuli'])

    return {
        'psychometric': psych,
        'um': um,
        'animal_id': sa['animal_id'],
        'true_model': sa['true_model'],
        'true_params': sa.get('true_params'),
        'n_sessions': len(sessions),
        'n_trials': n_trials,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. SYNTH MODEL FITS
# ═════════════════════════════════════════════════════════════════════════════

def compute_synth_model_fits(
    sa: dict,
    gs_index: dict,
    fit_target: str = 'update_matrix',
    n_reps: int = 20,
    burn_in: int = 1000,
    seed: int = 42,
) -> Optional[Dict]:
    """
    Simulate BE/SC with GS-recovered params, compute UMs and MSE.

    Same pattern as compute_model_fits but for synthetic animals
    where we have ground truth and the GS index structure.

    Returns:
        Dict with emp_um, be_um, sc_um, be_mse, sc_mse,
        emp_psych, be_psych_params, sc_psych_params,
        per_session_mse, session_data, true_model, true_params.
        Returns None if params not found.
    """
    from inference.comparison import simulate_all_sessions

    aid = sa['animal_id']
    mt = sa['true_model']
    sessions = sa['sessions']
    ft_short = FT_LABEL.get(fit_target, fit_target)

    # Get params from GS index
    animal_entry = gs_index.get(fit_target, {}).get(aid, {})
    be_data = animal_entry.get('BE')
    sc_data = animal_entry.get('SC')
    if be_data is None or sc_data is None:
        return None

    be_params = be_data.get('best_params')
    sc_params = sc_data.get('best_params')
    if not be_params or not sc_params:
        return None

    # Build FittingData
    clean = filter_trials(sessions)
    fd = fitting_data_from_sessions(clean, aid)

    # Simulate
    session_data = simulate_all_sessions(
        fd, be_params, sc_params,
        burn_in=burn_in, n_reps=n_reps, n_bins=8, seed=seed,
    )

    # Pooled results
    emp_um = np.nanmean([d['real_um'] for d in session_data], axis=0)
    be_um = np.nanmean([d['be_um'] for d in session_data], axis=0)
    sc_um = np.nanmean([d['sc_um'] for d in session_data], axis=0)

    emp_psych = compute_psychometric(clean, mode='pooled', n_bootstrap=500)

    be_psych_params = _avg_psych(session_data, 'be_psych')
    sc_psych_params = _avg_psych(session_data, 'sc_psych')

    per_session_mse = [
        {'session_idx': d['session_idx'], 'be_mse': d['be_um_mse'], 'sc_mse': d['sc_um_mse']}
        for d in session_data
    ]

    return {
        'emp_um': emp_um, 'be_um': be_um, 'sc_um': sc_um,
        'be_mse': matrix_error(be_um, emp_um),
        'sc_mse': matrix_error(sc_um, emp_um),
        'emp_psych': emp_psych,
        'be_psych_params': be_psych_params,
        'sc_psych_params': sc_psych_params,
        'per_session_mse': per_session_mse,
        'session_data': session_data,
        'animal_id': aid, 'true_model': mt, 'true_params': sa.get('true_params'),
        'fit_target': fit_target, 'be_params': be_params, 'sc_params': sc_params,
    }


def _avg_psych(session_data, key):
    ps = [d[key] for d in session_data if d.get(key, {}).get('success')]
    if not ps:
        return {}
    return {k: np.mean([p[k] for p in ps]) for k in ('mu', 'sigma', 'lapse_low', 'lapse_high')}


# ═════════════════════════════════════════════════════════════════════════════
# 3. PURE COMPUTATION UTILITIES (were already analysis, just here now)
# ═════════════════════════════════════════════════════════════════════════════

def build_gs_index(synth_gs_raw: dict) -> dict:
    """Build nested index: {fit_target: {animal_id: {'BE': data, 'SC': data}}}."""
    index = {}
    for (cohort, ft), pickles in synth_gs_raw.items():
        if ft not in index:
            index[ft] = {}
        for r in pickles:
            aid = r['animal_id']
            if aid not in index[ft]:
                index[ft][aid] = {}
            index[ft][aid][r['model']] = r
    return index


def build_confusion_matrix(
    df, true_col='true_model', pred_col='winner', labels=None,
) -> Optional[np.ndarray]:
    """Build 2x2 confusion matrix. Rows=true, columns=predicted."""
    labels = labels or ['BE', 'SC']
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for _, row in df.iterrows():
        t, p = row.get(true_col), row.get(pred_col)
        if t in labels and p in labels:
            cm[labels.index(t), labels.index(p)] += 1
    return cm if cm.sum() > 0 else None


def extract_gs_recovery(synth_gs_raw, fit_target='update_matrix'):
    """Extract true vs recovered params from GS results."""
    recovery = {'BE': {}, 'SC': {}}
    key = next((k for k in synth_gs_raw if k[1] == fit_target), None)
    if key is None:
        return recovery
    for r in synth_gs_raw[key]:
        mt = r.get('true_model', r.get('model'))
        best = r.get('best_params', {})
        true = r.get('true_params', {})
        if not best or not true:
            continue
        for pn in best:
            if pn not in true:
                continue
            if pn not in recovery[mt]:
                recovery[mt][pn] = {'true': [], 'recovered': []}
            recovery[mt][pn]['true'].append(true[pn])
            recovery[mt][pn]['recovered'].append(best[pn])
    return recovery


def extract_sbi_recovery(cohort_data, snpe_dict, n_samples=5000):
    """Extract true vs recovered (posterior median) from SBI."""
    recovery = {'BE': {}, 'SC': {}}
    for sa in cohort_data:
        mt = sa['true_model']
        mk = mt.lower()
        if mk not in snpe_dict:
            continue
        true = sa.get('true_params', {})
        entry = snpe_dict[mk]
        pnames = entry['param_names']
        observed = entry.get('observed_stats')
        if observed is None:
            continue
        try:
            import torch
            samps = entry['posterior'].sample(
                (n_samples,), x=torch.tensor(observed, dtype=torch.float32)).numpy()
        except Exception:
            continue
        medians = np.median(samps, axis=0)
        for j, pn in enumerate(pnames):
            if pn not in true:
                continue
            if pn not in recovery[mt]:
                recovery[mt][pn] = {'true': [], 'recovered': []}
            recovery[mt][pn]['true'].append(true[pn])
            recovery[mt][pn]['recovered'].append(float(medians[j]))
    return recovery


def _params_to_str(params):
    if params is None:
        return '(none)'
    d = {k: v for k, v in (vars(params) if hasattr(params, '__dict__') else params).items()
         if not str(k).startswith('_') and isinstance(v, (int, float))}
    return ', '.join(f'{k}={v:.3f}' for k, v in d.items())
