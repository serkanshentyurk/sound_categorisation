"""
Model Simulation Utilities

Session-by-session simulation of BE and SC models for visual comparison
and timing estimation. Extracted from inference/comparison.py to keep
that module focused on model comparison only.

Public API:
    simulate_all_sessions     — Simulate both models on every session
    simulate_example_session  — Simulate both models on one session
    estimate_timing           — Benchmark simulation speed
    print_timing_report       — Formatted timing output

Usage:
    from inference.simulation import simulate_all_sessions

    session_data = simulate_all_sessions(
        fitting_data, be_params, sc_params,
        burn_in=1000, n_reps=20,
    )
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional

from behav_utils.analysis.update_matrix import compute_update_matrix, matrix_error
from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.data.structures import FittingData
from behav_utils.data.synthetic import sample_stimuli


# =============================================================================
# SESSION-BY-SESSION SIMULATION
# =============================================================================

def simulate_all_sessions(
    fitting_data: FittingData,
    be_params: Dict[str, float],
    sc_params: Dict[str, float],
    burn_in: int = 1000,
    n_reps: int = 20,
    n_bins: int = 8,
    min_valid_trials: int = 30,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Simulate BE and SC on every session for visual comparison.

    Returns list of dicts, one per session, containing:
        stimuli, categories, choices (real),
        session_id, session_idx, accuracy, n_trials,
        real_um, be_um, sc_um,
        real_psych, be_psych, sc_psych,
        be_um_mse, sc_um_mse
    """
    from models.BE_core import BEParams, BEModel
    from models.SC_core import SCParams, SCModel

    be_p = BEParams.from_dict(be_params)
    sc_p = SCParams.from_dict(sc_params)

    results = []

    for i in range(fitting_data.n_sessions):
        v = ~fitting_data.no_response[i]
        stim = fitting_data.stimuli[i][v]
        cat = fitting_data.categories[i][v]
        ch = fitting_data.choices[i][v]

        if len(stim) < min_valid_trials:
            continue

        acc = float(np.mean(ch == cat))
        real_um, _, _ = compute_update_matrix(stim, ch, cat, n_bins=n_bins)
        real_psych = fit_psychometric(stim, ch)

        # BE simulation
        be_ums, be_psychs = _simulate_model_reps(
            'BE', be_p, stim, cat, burn_in, n_reps, n_bins, seed,
            BEModel, None,
        )
        be_mean_um = (np.nanmean(be_ums, axis=0) if be_ums
                      else np.full((n_bins, n_bins), np.nan))
        be_psych = _fit_mean_psychometric(be_psychs, stim)

        # SC simulation
        sc_ums, sc_psychs = _simulate_model_reps(
            'SC', sc_p, stim, cat, burn_in, n_reps, n_bins, seed,
            None, SCModel,
        )
        sc_mean_um = (np.nanmean(sc_ums, axis=0) if sc_ums
                      else np.full((n_bins, n_bins), np.nan))
        sc_psych = _fit_mean_psychometric(sc_psychs, stim)

        results.append({
            'stimuli': stim, 'categories': cat, 'choices': ch,
            'session_id': fitting_data.session_ids[i],
            'session_idx': int(fitting_data.session_indices[i]),
            'accuracy': acc, 'n_trials': len(stim),
            'real_um': real_um, 'be_um': be_mean_um, 'sc_um': sc_mean_um,
            'real_psych': real_psych, 'be_psych': be_psych,
            'sc_psych': sc_psych,
            'be_um_mse': matrix_error(be_mean_um, real_um),
            'sc_um_mse': matrix_error(sc_mean_um, real_um),
        })

    return results


# =============================================================================
# SINGLE-SESSION EXAMPLE SIMULATION
# =============================================================================

def simulate_example_session(
    animal: Any, session_idx: int,
    be_params: Dict, sc_params: Dict,
    stage: str = 'Full_Task_Cont', distribution: str = 'Uniform',
    burn_in: int = 1000, n_reps: int = 20, seed: int = 42,
) -> Dict[str, Any]:
    """Simulate BE and SC on one real session for visualisation."""
    from models.BE_core import BEParams, BEModel
    from models.SC_core import SCParams, SCModel
    from behav_utils import select_sessions
    from behav_utils.data.filtering import filter_session

    sessions = select_sessions(animal, stage=stage, distribution=distribution)
    sess = sessions[session_idx]
    clean = filter_session(sess)
    arrays = clean.get_arrays()
    valid = ~arrays['no_response']
    stim = arrays['stimuli'][valid]
    cat = arrays['categories'][valid]
    ch = arrays['choices'][valid]

    be_p = BEParams(**be_params)
    be_state = BEModel.create_initial_state(params=be_p, burn_in=burn_in, seed=seed)
    _, be_pB, _, _ = BEModel.simulate_session(
        be_p, be_state, stim, cat,
        np.random.default_rng(seed), return_history=False,
    )

    sc_p = SCParams(**sc_params)
    sc_state = SCModel.create_initial_state(params=sc_p, burn_in=burn_in, seed=seed)
    _, sc_pB, _, _ = SCModel.simulate_session(
        sc_p, sc_state, stim, cat,
        np.random.default_rng(seed), return_history=False,
    )

    be_all, sc_all = [], []
    for r in range(n_reps):
        rng_r = np.random.default_rng(seed + r + 1)
        s1 = BEModel.create_initial_state(params=be_p, burn_in=burn_in, seed=seed)
        c1, _, _, _ = BEModel.simulate_session(
            be_p, s1, stim, cat, rng_r, return_history=False,
        )
        be_all.append(c1)
        s2 = SCModel.create_initial_state(params=sc_p, burn_in=burn_in, seed=seed)
        c2, _, _, _ = SCModel.simulate_session(
            sc_p, s2, stim, cat, rng_r, return_history=False,
        )
        sc_all.append(c2)

    return {
        'stimuli': stim, 'categories': cat, 'choices': ch,
        'be_pB': be_pB, 'sc_pB': sc_pB,
        'be_choices_all': be_all, 'sc_choices_all': sc_all,
        'session_info': {
            'session_id': sess.session_id, 'n_trials': len(stim),
            'accuracy': float(np.mean(ch == cat)),
        },
    }


# =============================================================================
# TIMING UTILITIES
# =============================================================================

def estimate_timing(
    stat_names: List[str],
    n_trials: int = 2500,
    burn_in: int = 1000,
    n_sbi_sims: int = 50_000,
    n_test: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Estimate per-simulation cost for BE and SC.

    Runs n_test forward simulations with each model and reports
    timing + NaN rate + projected total training time.
    """
    from inference.simulator import (
        create_be_simulator, create_sc_simulator,
    )

    stim, cat = sample_stimuli(n_trials, 'uniform', np.random.default_rng(seed))
    results = {}

    for model_type in ['be', 'sc']:
        creator = create_be_simulator if model_type == 'be' else create_sc_simulator
        sim = creator(stim, cat, stat_names=stat_names, burn_in=burn_in)

        times = []
        nan_count = 0
        for i in range(n_test):
            theta = sim.sample_prior(seed=seed + i)
            t0 = time.time()
            stats = sim(theta, seed=seed + i)
            times.append(time.time() - t0)
            if np.any(np.isnan(stats)):
                nan_count += 1

        ms_per_sim = np.mean(times) * 1000
        total_min = np.mean(times) * n_sbi_sims / 60
        n_stat_dims = len(stats)

        results[model_type] = {
            'ms_per_sim': ms_per_sim,
            'total_minutes': total_min,
            'total_hours': total_min / 60,
            'nan_rate': nan_count / n_test,
            'stat_dims': n_stat_dims,
            'theta_dims': sim.n_free_params,
        }

    return results


def print_timing_report(
    timing: Dict[str, Any],
    n_sbi_sims: int,
    n_animals: int = 1,
    label: str = '',
):
    """Print a formatted timing report."""
    print(f"\n{'=' * 60}")
    if label:
        print(f"  Timing estimate: {label}")
    print(f"  {n_sbi_sims:,} simulations")
    print(f"{'=' * 60}")
    print(f"  {'Model':<6s} {'ms/sim':>8s} {'Total':>10s} {'NaN%':>6s} "
          f"{'θ dims':>7s} {'Stat dims':>10s}")
    print(f"  {'-' * 50}")

    for mt in ['be', 'sc']:
        t = timing[mt]
        total_str = (f"{t['total_hours']:.1f}h" if t['total_hours'] >= 1
                     else f"{t['total_minutes']:.0f}min")
        print(f"  {mt.upper():<6s} {t['ms_per_sim']:8.0f} {total_str:>10s} "
              f"{t['nan_rate']:5.0%} {t['theta_dims']:>7d} {t['stat_dims']:>10d}")

    if n_animals > 1:
        be_h = timing['be']['total_hours']
        sc_h = timing['sc']['total_hours']
        total = (be_h + sc_h) * n_animals
        print(f"\n  {n_animals} animals × 2 models = ~{total:.0f} hours total")


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _simulate_model_reps(
    model_name, params, stim, cat, burn_in, n_reps, n_bins, seed,
    BEModel_cls, SCModel_cls,
):
    """Helper: run n_reps simulations, collect UMs and choice arrays."""
    ums, all_choices = [], []

    for r in range(n_reps):
        rng_r = np.random.default_rng(seed + r + 1)

        if model_name == 'BE':
            state = BEModel_cls.create_initial_state(
                params=params, burn_in=burn_in, seed=seed,
            )
            c, _, _, _ = BEModel_cls.simulate_session(
                params, state, stim, cat, rng_r, return_history=False,
            )
        else:
            state = SCModel_cls.create_initial_state(
                params=params, burn_in=burn_in, seed=seed,
            )
            c, _, _, _ = SCModel_cls.simulate_session(
                params, state, stim, cat, rng_r, return_history=False,
            )

        vv = ~np.isnan(c)
        if vv.sum() > 50:
            um, _, _ = compute_update_matrix(stim[vv], c[vv], cat[vv], n_bins)
            ums.append(um)
        all_choices.append(c)

    return ums, all_choices


def _fit_mean_psychometric(all_choices, stim):
    """Fit psychometric to mean choice probabilities across reps."""
    if not all_choices:
        return {'success': False}
    mean_choices = np.nanmean(all_choices, axis=0)
    return fit_psychometric(stim, mean_choices)
