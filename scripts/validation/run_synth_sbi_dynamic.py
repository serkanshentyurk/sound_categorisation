#!/usr/bin/env python3
"""
Dynamic SBI validation on one synthetic learning-trajectory animal.

Generates a synthetic animal with known per-session parameters,
runs dynamic SBI with RandomWalk linking for BOTH models (BE and SC),
computes posterior predictive checks, and saves everything needed
for trajectory recovery plots and model comparison violin plots.

Usage:
    # Ultra-fast local test (verifies code runs, ~1 min)
    python scripts/validation/run_synth_sbi_dynamic.py \
        --model be --animal-index 0 --ultra-fast

    # Smoke test (~5 min)
    python scripts/validation/run_synth_sbi_dynamic.py \
        --model be --animal-index 0 --smoke-test

    # Full run (cluster, ~30 min)
    python scripts/validation/run_synth_sbi_dynamic.py \
        --model be --animal-index 0

Output:
    results/validation/synth_sbi_dynamic/{model}_{animal_index:02d}.pkl
"""

import argparse
import pickle
import sys
import time
import warnings
import numpy as np
from pathlib import Path
from dataclasses import asdict

# ── Path setup ───────────────────────────────────────────────────────────────
# Script lives in scripts/validation/, repo root is two levels up.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.config import (
    DYNAMIC_SBI_N_SIMULATIONS, DYNAMIC_SBI_SIGMA_DRIFT,
    DYNAMIC_SBI_VARYING_PARAMS, DYNAMIC_SBI_BOUNDS,
    SYNTH_DYNAMIC_VARYING_PARAMS,
    SBI_BURN_IN, BASE_SEED,
    VALIDATION_DIR,
    SMOKE_DYNAMIC_SBI_N_SIMULATIONS,
    build_metadata, ensure_dirs,
)
from analysis.validation import (
    _sample_params, _make_simulator, generate_session_with_distribution,
)


# ─── Synthetic animal generation with stored per-session params ──────────────

def _compute_session_params(model_type, expert_params, session_idx, n_sessions):
    """
    Compute parameters for one session along the learning trajectory.

    Matches make_learning_cohort logic exactly:
        BE: eta_learning follows hump (0.02 -> peak -> expert)
        SC: gamma follows monotonic increase (0.5 -> expert)
    Perceptual params constant throughout.
    """
    frac = session_idx / max(n_sessions - 1, 1)

    if model_type == 'BE':
        from models.BE_core import BEParams
        peak_eta = min(expert_params.eta_learning * 2.5, 0.9)
        if frac < 0.4:
            eta = 0.02 + (peak_eta - 0.02) * (frac / 0.4)
        else:
            eta = peak_eta + (expert_params.eta_learning - peak_eta) * (
                (frac - 0.4) / 0.6
            )
        return BEParams(
            sigma_percep=expert_params.sigma_percep,
            A_repulsion=expert_params.A_repulsion,
            eta_learning=eta,
            eta_relax=expert_params.eta_relax,
        )
    else:
        from models.SC_core import SCParams
        gamma = 0.5 + (expert_params.gamma - 0.5) * frac
        gamma = min(gamma, 1.0)
        return SCParams(
            sigma_percep=expert_params.sigma_percep,
            A_repulsion=expert_params.A_repulsion,
            gamma=gamma,
            sigma_update=expert_params.sigma_update,
        )


def generate_synthetic_dynamic_animal(
    model_type, animal_index, n_sessions=20,
    trials_per_session=350, burn_in=1000, seed=42,
):
    """
    Generate one synthetic animal with known per-session parameters.

    Uses _sample_params from analysis.validation (same prior as all
    other synthetic cohorts).
    """
    from behav_utils.data.structures import AnimalData

    rng = np.random.default_rng(seed + animal_index * 100)
    expert_params = _sample_params(model_type, rng)

    sessions = []
    session_params_list = []
    sess_rng = np.random.default_rng(seed + animal_index * 100)

    for s in range(n_sessions):
        sp = _compute_session_params(model_type, expert_params, s, n_sessions)
        session_params_list.append(sp)

        sim = _make_simulator(
            model_type, sp, burn_in, seed + animal_index * 100 + s)
        sess = generate_session_with_distribution(
            session_idx=s, n_trials=trials_per_session,
            distribution='uniform',
            animal_id=f'{model_type}_dyn_{animal_index:02d}',
            simulator=sim, stage='Full_Task_Cont', rng=sess_rng,
        )
        sessions.append(sess)

    aid = f'{model_type}_dyn_{animal_index:02d}'
    animal = AnimalData(animal_id=aid, sessions=sessions)

    return {
        'animal_id': aid,
        'true_model': model_type,
        'expert_params': expert_params,
        'session_params': session_params_list,
        'session_params_dicts': [asdict(sp) for sp in session_params_list],
        'sessions': sessions,
        'animal': animal,
    }


# ─── Dynamic SBI fitting ────────────────────────────────────────────────────

def run_dynamic_sbi(
    sessions, animal_id, model_type, varying_params,
    sigma_drift_dict, n_simulations, burn_in, seed,
    bounds_override=None,
    n_trajectory_samples=500, n_ppc_sims=200,
):
    """
    Run dynamic SBI with RandomWalk linking for one model type.

    Args:
        sigma_drift_dict: {param_name: float} — per-parameter drift.
        bounds_override: {param_name: (lo, hi)} — overrides model defaults.
            Use DYNAMIC_SBI_BOUNDS for learning-trajectory fitting.

    Returns dict with trajectories, PPC, session params, and timing.
    """
    from behav_utils.data.selection import fitting_data_from_sessions
    from behav_utils.data.filtering import filter_trials
    from inference.fitting import SBIFitter
    from inference.types import ConstantSpec, RandomWalkSpec

    sessions = filter_trials(sessions)
    fd = fitting_data_from_sessions(sessions, animal_id)

    if model_type.upper() == 'BE':
        from models.BE_core import BEParams
        bounds = BEParams.get_bounds()
        all_params = BEParams.get_param_names()
    else:
        from models.SC_core import SCParams
        bounds = SCParams.get_bounds()
        all_params = SCParams.get_param_names()

    # Override bounds if provided (e.g. wider bounds for dynamic fitting)
    if bounds_override is not None:
        for pname, b in bounds_override.items():
            if pname in bounds:
                bounds[pname] = b

    param_links = {}
    for pname in all_params:
        if pname in varying_params:
            sd = sigma_drift_dict.get(pname, 0.05)
            param_links[pname] = RandomWalkSpec(
                bounds=bounds[pname],
                sigma_drift=sd,
            )
        else:
            param_links[pname] = ConstantSpec(bounds=bounds[pname])

    fitter = SBIFitter(
        fitting_data=fd,
        model_type=model_type.lower(),
        param_links=param_links,
        burn_in=burn_in,
    )

    t0 = time.time()
    result = fitter.train(n_simulations=n_simulations, seed=seed)
    training_time = time.time() - t0

    trajectories = fitter.extract_trajectories(
        result, n_samples=n_trajectory_samples)
    session_params = fitter.extract_session_params(
        result, n_samples=n_trajectory_samples)

    # Posterior predictive check
    ppc = fitter.posterior_predictive_check(
        result, n_simulations=n_ppc_sims, seed=seed)

    return {
        'model_type': model_type.upper(),
        'trajectories': trajectories,
        'session_params': session_params,
        'ppc': ppc,
        'training_time': training_time,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Dynamic SBI validation on synthetic learning trajectory')
    parser.add_argument('--model', required=True, choices=['be', 'sc'],
                        help='True model type for synthetic animal')
    parser.add_argument('--animal-index', required=True, type=int,
                        help='Index within model type (0-based)')
    parser.add_argument('--n-sessions', type=int, default=20)
    parser.add_argument('--trials-per-session', type=int, default=350)
    parser.add_argument('--n-simulations', type=int, default=None)
    parser.add_argument('--seed', type=int, default=BASE_SEED)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--smoke-test', action='store_true',
                        help='Reduced sims (~5 min)')
    parser.add_argument('--ultra-fast', action='store_true',
                        help='Minimal run to verify code works (~1 min)')
    args = parser.parse_args()

    # ── Resolve settings by mode ─────────────────────────────────────────────
    if args.ultra_fast:
        n_sims = args.n_simulations or 200
        n_sessions = min(args.n_sessions, 5)
        trials_per_session = min(args.trials_per_session, 100)
        n_trajectory_samples = 50
        n_ppc_sims = 20
        mode_label = 'ULTRA-FAST'
    elif args.smoke_test:
        n_sims = args.n_simulations or SMOKE_DYNAMIC_SBI_N_SIMULATIONS
        n_sessions = args.n_sessions
        trials_per_session = args.trials_per_session
        n_trajectory_samples = 100
        n_ppc_sims = 50
        mode_label = 'SMOKE TEST'
    else:
        n_sims = args.n_simulations or DYNAMIC_SBI_N_SIMULATIONS
        n_sessions = args.n_sessions
        trials_per_session = args.trials_per_session
        n_trajectory_samples = 500
        n_ppc_sims = 200
        mode_label = 'FULL'

    model_key = args.model.upper()

    # Use SYNTH varying params (only actually-varying params) for validation.
    # For the WRONG model fit, also use synth params so both models get a
    # fair comparison (only fitting what the synthetic data actually varies).
    be_varying = list(SYNTH_DYNAMIC_VARYING_PARAMS['BE'])
    sc_varying = list(SYNTH_DYNAMIC_VARYING_PARAMS['SC'])
    true_varying = be_varying if model_key == 'BE' else sc_varying

    # Per-parameter sigma_drift
    be_drift = DYNAMIC_SBI_SIGMA_DRIFT['BE']
    sc_drift = DYNAMIC_SBI_SIGMA_DRIFT['SC']
    true_drift = be_drift if model_key == 'BE' else sc_drift

    # Dynamic bounds (wider than static to cover learning phase)
    be_bounds = DYNAMIC_SBI_BOUNDS['BE']
    sc_bounds = DYNAMIC_SBI_BOUNDS['SC']
    true_bounds = be_bounds if model_key == 'BE' else sc_bounds

    output_dir = Path(args.output_dir) if args.output_dir else (
        VALIDATION_DIR / 'synth_sbi_dynamic'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{model_key.lower()}_{args.animal_index:02d}.pkl'

    print(f'=== Dynamic SBI Validation ({mode_label}): '
          f'{model_key} #{args.animal_index} ===')
    print(f'  Sessions:           {n_sessions}')
    print(f'  Trials/session:     {trials_per_session}')
    print(f'  Simulations:        {n_sims:,}')
    print(f'  Trajectory samples: {n_trajectory_samples}')
    print(f'  PPC sims:           {n_ppc_sims}')
    print(f'  Varying params:     {true_varying}')
    print(f'  Sigma drift:        {true_drift}')
    print(f'  Dynamic bounds:     {true_bounds}')

    # ── Generate synthetic animal ────────────────────────────────────────────
    print('\n── Generating synthetic animal ──')
    synth = generate_synthetic_dynamic_animal(
        model_type=model_key,
        animal_index=args.animal_index,
        n_sessions=n_sessions,
        trials_per_session=trials_per_session,
        burn_in=SBI_BURN_IN,
        seed=args.seed,
    )
    print(f'  Animal: {synth["animal_id"]}')
    print(f'  Expert params: {synth["expert_params"]}')

    for pname in true_varying:
        vals = [d[pname] for d in synth['session_params_dicts']]
        print(f'  {pname}: {vals[0]:.3f} -> {vals[len(vals)//2]:.3f} -> {vals[-1]:.3f}')

    # ── Run dynamic SBI: TRUE model ─────────────────────────────────────────
    print(f'\n── Fitting TRUE model ({model_key}) ──')
    true_fit = run_dynamic_sbi(
        sessions=synth['sessions'],
        animal_id=synth['animal_id'],
        model_type=model_key,
        varying_params=true_varying,
        sigma_drift_dict=true_drift,
        n_simulations=n_sims,
        burn_in=SBI_BURN_IN,
        seed=args.seed,
        bounds_override=true_bounds,
        n_trajectory_samples=n_trajectory_samples,
        n_ppc_sims=n_ppc_sims,
    )
    print(f'  Training time: {true_fit["training_time"] / 60:.1f} min')

    # ── Run dynamic SBI: WRONG model ────────────────────────────────────────
    wrong_key = 'SC' if model_key == 'BE' else 'BE'
    wrong_varying = sc_varying if model_key == 'BE' else be_varying
    wrong_drift = sc_drift if model_key == 'BE' else be_drift
    wrong_bounds = sc_bounds if model_key == 'BE' else be_bounds

    print(f'\n── Fitting WRONG model ({wrong_key}) ──')
    wrong_fit = run_dynamic_sbi(
        sessions=synth['sessions'],
        animal_id=synth['animal_id'],
        model_type=wrong_key,
        varying_params=wrong_varying,
        sigma_drift_dict=wrong_drift,
        n_simulations=n_sims,
        burn_in=SBI_BURN_IN,
        seed=args.seed + 999,
        bounds_override=wrong_bounds,
        n_trajectory_samples=n_trajectory_samples,
        n_ppc_sims=n_ppc_sims,
    )
    print(f'  Training time: {wrong_fit["training_time"] / 60:.1f} min')

    # ── Compare PPC ──────────────────────────────────────────────────────────
    true_ppc_mse = np.mean(
        (true_fit['ppc']['simulated'].mean(axis=0)
         - true_fit['ppc']['observed']) ** 2
    )
    wrong_ppc_mse = np.mean(
        (wrong_fit['ppc']['simulated'].mean(axis=0)
         - wrong_fit['ppc']['observed']) ** 2
    )
    ppc_winner = model_key if true_ppc_mse <= wrong_ppc_mse else wrong_key
    correct = ppc_winner == model_key

    print(f'\n── Comparison ──')
    print(f'  PPC MSE ({model_key}): {true_ppc_mse:.6f}')
    print(f'  PPC MSE ({wrong_key}): {wrong_ppc_mse:.6f}')
    print(f'  Winner: {ppc_winner} (correct: {"✓" if correct else "✗"})')

    total_time = true_fit['training_time'] + wrong_fit['training_time']
    print(f'\n  Total time: {total_time / 60:.1f} min')

    # ── Save ─────────────────────────────────────────────────────────────────
    save_data = {
        # Ground truth
        'animal_id': synth['animal_id'],
        'true_model': model_key,
        'expert_params': synth['expert_params'],
        'session_params_dicts': synth['session_params_dicts'],
        'n_sessions': n_sessions,
        'trials_per_session': trials_per_session,

        # True model fit (trajectories + PPC for violin plots)
        'true_fit': {
            'model_type': true_fit['model_type'],
            'varying_params': true_varying,
            'trajectories': true_fit['trajectories'],
            'session_params': true_fit['session_params'],
            'ppc_observed': true_fit['ppc']['observed'],
            'ppc_simulated': true_fit['ppc']['simulated'],
            'ppc_p_values': true_fit['ppc']['p_values'],
            'ppc_stat_names': true_fit['ppc']['stat_names'],
            'training_time': true_fit['training_time'],
        },

        # Wrong model fit (for model comparison violin plots)
        'wrong_fit': {
            'model_type': wrong_fit['model_type'],
            'varying_params': wrong_varying,
            'trajectories': wrong_fit['trajectories'],
            'session_params': wrong_fit['session_params'],
            'ppc_observed': wrong_fit['ppc']['observed'],
            'ppc_simulated': wrong_fit['ppc']['simulated'],
            'ppc_p_values': wrong_fit['ppc']['p_values'],
            'ppc_stat_names': wrong_fit['ppc']['stat_names'],
            'training_time': wrong_fit['training_time'],
        },

        # Comparison summary
        'ppc_winner': ppc_winner,
        'correct': correct,
        'true_ppc_mse': float(true_ppc_mse),
        'wrong_ppc_mse': float(wrong_ppc_mse),

        # Config
        'sigma_drift': {
            model_key: true_drift,
            wrong_key: wrong_drift,
        },
        'dynamic_bounds': {
            model_key: true_bounds,
            wrong_key: wrong_bounds,
        },
        'n_simulations': n_sims,
        'n_trajectory_samples': n_trajectory_samples,
        'n_ppc_sims': n_ppc_sims,

        'metadata': build_metadata(
            'validation/run_synth_sbi_dynamic.py', vars(args)),
    }
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f'\n  Saved to {output_path}')


if __name__ == '__main__':
    main()
