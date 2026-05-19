#!/usr/bin/env python3
"""
Simulation-Based Inference (unified script)

Three modes:
    train    — train amortised SNPE on curriculum-matched generic data
    static   — load amortised network, condition on animal, held-out CV
    dynamic  — per-animal SBIFitter with temporal parameter links

CLI usage:
    python scripts/sbi.py train --model BE --curriculum uniform:15
    python scripts/sbi.py train --model SC --curriculum uniform:10,hard_a:5

    python scripts/sbi.py static --animal SS01 --model BE \
        --fit-target update_matrix

    python scripts/sbi.py dynamic --animal SS01 --model BE \
        --link randomwalk --distribution uniform

Importable:
    from scripts.sbi import run_train, run_sbi_static, run_sbi_dynamic

Output schema: see grid_search.py docstring (shared schema).
"""

import argparse
import pickle
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# CURRICULUM PARSING
# =============================================================================

def parse_curriculum(s: str) -> List[Tuple[str, int]]:
    """
    Parse curriculum string: 'uniform:15' or 'uniform:10,hard_a:5,hard_b:5'.

    Returns list of (distribution, n_sessions) tuples.
    """
    parts = s.split(',')
    curriculum = []
    for part in parts:
        dist, n = part.strip().split(':')
        curriculum.append((dist.strip(), int(n.strip())))
    return curriculum


# =============================================================================
# TRAIN (amortised)
# =============================================================================

def run_train(
    model_type: str,
    curriculum: List[Tuple[str, int]],
    n_simulations: Optional[int] = None,
    trials_per_session: int = 350,
    burn_in: Optional[int] = None,
    stat_names: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    smoke_test: bool = False,
    seed: int = 42,
) -> Path:
    """
    Train an amortised SNPE network for one model type.

    Returns path to saved network.
    """
    from scripts.config import (
        SBI_N_SIMULATIONS, SBI_BURN_IN, SNPE_DIR,
        SMOKE_SBI_N_SIMULATIONS,
    )
    from inference.amortised import AmortisedSBI

    n_simulations = n_simulations or (
        SMOKE_SBI_N_SIMULATIONS if smoke_test else SBI_N_SIMULATIONS)
    burn_in = burn_in or SBI_BURN_IN

    if output_path is None:
        SNPE_DIR.mkdir(parents=True, exist_ok=True)
        curr_str = '_'.join(f'{d}_{n}' for d, n in curriculum)
        output_path = SNPE_DIR / f'{curr_str}_{model_type.lower()}.pkl'

    print(f'Training AmortisedSBI: {model_type} / {curriculum}')
    print(f'  Simulations: {n_simulations:,}')

    trainer = AmortisedSBI(
        model_type=model_type,
        curriculum=curriculum,
        trials_per_session=trials_per_session,
        burn_in=burn_in,
        stat_names=stat_names,
    )

    trainer.train(n_simulations=n_simulations, seed=seed)
    trainer.save(output_path)

    print(f'  Saved to {output_path}')
    return output_path


# =============================================================================
# STATIC INFERENCE (amortised)
# =============================================================================

def run_sbi_static(
    sessions: list,
    model_type: str,
    fit_target: str = 'update_matrix',
    snpe_path: Optional[Path] = None,
    animal_id: str = 'unknown',
    distribution: str = 'uniform',
    n_folds: int = 2,
    n_posterior_samples: Optional[int] = None,
    n_stochastic_reps: Optional[int] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """
    Static SBI: load amortised network, condition on animal, held-out CV.

    Returns standardised result dict (same schema as grid_search.py).
    """
    from scripts.config import (
        SBI_N_POSTERIOR_SAMPLES, SBI_N_STOCHASTIC_REPS, SNPE_DIR,
    )
    from inference.amortised import AmortisedSBI

    n_posterior_samples = n_posterior_samples or SBI_N_POSTERIOR_SAMPLES
    n_stochastic_reps = n_stochastic_reps or SBI_N_STOCHASTIC_REPS

    if snpe_path is None:
        snpe_path = _find_snpe(SNPE_DIR, model_type, distribution)

    loaded = AmortisedSBI.load(snpe_path)

    result = loaded.fit(
        sessions=sessions,
        animal_id=animal_id,
        distribution=distribution,
        fit_target=fit_target,
        n_folds=n_folds,
        n_posterior_samples=n_posterior_samples,
        n_stochastic_reps=n_stochastic_reps,
    )
    return result


def _find_snpe(snpe_dir: Path, model_type: str, distribution: str) -> Path:
    """Auto-discover SNPE file matching model + distribution."""
    model_lower = model_type.lower()
    candidates = [
        snpe_dir / f'{distribution}_{model_lower}.pkl',
        snpe_dir / f'{distribution}_15_{model_lower}.pkl',
    ]
    if snpe_dir.exists():
        candidates.extend(
            snpe_dir.glob(f'*{distribution}*{model_lower}*.pkl'))

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f'No SNPE network found for {model_type}/{distribution} in {snpe_dir}. '
        f'Run: python scripts/sbi.py train --model {model_type} '
        f'--curriculum {distribution}:N')


# =============================================================================
# DYNAMIC INFERENCE (per-animal SBIFitter)
# =============================================================================

def run_sbi_dynamic(
    sessions: list,
    model_type: str,
    link_type: str = 'randomwalk',
    fit_target: str = 'update_matrix',
    animal_id: str = 'unknown',
    distribution: str = 'uniform',
    n_simulations: Optional[int] = None,
    varying_params: Optional[List[str]] = None,
    sigma_drift: Optional[float] = None,
    burn_in: Optional[int] = None,
    n_bins: Optional[int] = None,
    n_trajectory_samples: int = 500,
    smoke_test: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Dynamic SBI: per-animal training with temporal parameter links.

    Uses SBIFitter from inference/fitting.py. Produces PPC errors
    (NOT true cross-validated — cv_type='ppc').

    Returns standardised result dict with trajectories.
    """
    from scripts.config import (
        DYNAMIC_SBI_N_SIMULATIONS, DYNAMIC_SBI_SIGMA_DRIFT,
        DYNAMIC_SBI_VARYING_PARAMS, SBI_BURN_IN, GS_N_BINS,
        SMOKE_DYNAMIC_SBI_N_SIMULATIONS,
    )
    from inference.fitting import SBIFitter
    from inference.types import ConstantSpec, RandomWalkSpec, GPSpec
    from behav_utils.data.selection import fitting_data_from_sessions
    from behav_utils.analysis.update_matrix import (
        compute_update_matrix, matrix_error,
    )
    from behav_utils.data.filtering import pool_arrays

    n_simulations = n_simulations or (
        SMOKE_DYNAMIC_SBI_N_SIMULATIONS if smoke_test
        else DYNAMIC_SBI_N_SIMULATIONS)
    burn_in = burn_in or SBI_BURN_IN
    n_bins = n_bins or GS_N_BINS

    model_key = model_type.upper()
    model_lower = model_type.lower()
    varying_params = varying_params or list(
        DYNAMIC_SBI_VARYING_PARAMS.get(model_key, ()))

    # Get param bounds and names
    if model_lower == 'be':
        from models.BE_core import BEParams
        bounds = BEParams.get_bounds()
        all_params = BEParams.get_param_names()
    else:
        from models.SC_core import SCParams
        bounds = SCParams.get_bounds()
        all_params = SCParams.get_param_names()

    # Build link specs
    param_links = {}
    for pname in all_params:
        if pname in varying_params:
            sd = sigma_drift or DYNAMIC_SBI_SIGMA_DRIFT.get(pname, 0.02)
            if link_type == 'constant':
                param_links[pname] = ConstantSpec(bounds=bounds[pname])
            elif link_type == 'randomwalk':
                param_links[pname] = RandomWalkSpec(
                    bounds=bounds[pname], sigma_drift=sd)
            elif link_type == 'gp':
                param_links[pname] = GPSpec(bounds=bounds[pname])
            else:
                raise ValueError(f'Unknown link type: {link_type}')
        else:
            param_links[pname] = ConstantSpec(bounds=bounds[pname])

    # Build fitting data and fitter
    fd = fitting_data_from_sessions(sessions, animal_id)

    fitter = SBIFitter(
        fitting_data=fd,
        model_type=model_lower,
        param_links=param_links,
        burn_in=burn_in,
    )

    # Train
    sbi_result = fitter.train(n_simulations=n_simulations, seed=seed)

    # Extract trajectories
    trajectories = fitter.extract_trajectories(
        sbi_result, n_samples=n_trajectory_samples)

    # PPC errors (per-session)
    cv_errors = _compute_ppc_errors(
        sessions, trajectories,
        model_lower, fit_target, burn_in, n_bins,
    )

    # Point estimate: trajectory medians
    best_params = {}
    for pn in trajectories:
        traj = trajectories[pn]
        if isinstance(traj['median'], (float, int)):
            best_params[pn] = traj['median']
        else:
            best_params[pn] = [float(v) for v in traj['median']]

    return {
        'method': 'sbi_dynamic',
        'cv_type': 'ppc',
        'model_type': model_key,
        'fit_target': fit_target,
        'animal_id': animal_id,
        'distribution': distribution,
        'link_type': link_type,

        'cv_errors': cv_errors,
        'mean_error': float(np.mean(cv_errors)) if cv_errors else np.nan,
        'std_error': float(np.std(cv_errors)) if cv_errors else np.nan,

        'best_params': best_params,
        'posterior_samples': None,
        'param_names': list(all_params),
        'trajectories': trajectories,

        'all_results': None,

        'n_sessions': len(sessions),
        'n_trials': sum(len(s.trials.stimulus) for s in sessions),
        'metadata': None,
    }


def _compute_ppc_errors(
    sessions, trajectories,
    model_type, fit_target, burn_in, n_bins,
) -> List[float]:
    """Per-session PPC: simulate from trajectory median, compare to empirical."""
    from inference.amortised import simulate_choices_from_params
    from behav_utils.analysis.update_matrix import (
        compute_update_matrix, matrix_error,
    )
    from behav_utils.data.filtering import pool_arrays

    errors = []
    for s_idx, sess in enumerate(sessions):
        pooled = pool_arrays([sess])
        stim = pooled['stimuli']
        ch = pooled['choices']
        cat = pooled['categories']
        valid = ~pooled.get('no_response', np.isnan(ch))

        if valid.sum() < 30:
            continue

        emp_um, emp_cm, _ = compute_update_matrix(
            stim, ch, cat, n_bins=n_bins, trial_filter='post_correct')
        emp_target = emp_um if fit_target == 'update_matrix' else emp_cm

        # Extract median params at this session
        sess_params = {}
        for pn, traj in trajectories.items():
            if isinstance(traj['median'], (float, int)):
                sess_params[pn] = traj['median']
            else:
                sess_params[pn] = float(traj['median'][s_idx])

        sim_ch = simulate_choices_from_params(
            model_type, sess_params,
            stim[valid], cat[valid],
            burn_in=burn_in, seed=s_idx,
        )

        sim_um, sim_cm, _ = compute_update_matrix(
            stim[valid], sim_ch, cat[valid],
            n_bins=n_bins, trial_filter='post_correct')
        sim_target = sim_um if fit_target == 'update_matrix' else sim_cm

        errors.append(float(matrix_error(emp_target, sim_target)))

    return errors


# =============================================================================
# CLI
# =============================================================================

def main():
    from scripts.config import (
        SNPE_DIR, SBI_STATIC_DIR, SBI_DYNAMIC_DIR, FIT_TARGETS,
        build_metadata, ensure_dirs, load_animal_data,
    )

    parser = argparse.ArgumentParser(description='SBI (unified)')
    sub = parser.add_subparsers(dest='mode', required=True)

    # -- train -----------------------------------------------------------------
    p_train = sub.add_parser('train', help='Train amortised SNPE')
    p_train.add_argument('--model', required=True, choices=['BE', 'SC'])
    p_train.add_argument('--curriculum', required=True,
                         help='e.g. uniform:15 or uniform:10,hard_a:5')
    p_train.add_argument('--n-sims', type=int, default=None)
    p_train.add_argument('--output', type=str, default=None)
    p_train.add_argument('--seed', type=int, default=42)
    p_train.add_argument('--smoke-test', action='store_true')

    # -- static ----------------------------------------------------------------
    p_static = sub.add_parser('static', help='Static SBI (amortised)')
    p_static.add_argument('--animal', required=True)
    p_static.add_argument('--model', required=True, choices=['BE', 'SC'])
    p_static.add_argument('--fit-target', default='update_matrix',
                          choices=list(FIT_TARGETS))
    p_static.add_argument('--distribution', default='uniform')
    p_static.add_argument('--snpe-path', type=str, default=None)
    p_static.add_argument('--output-dir', type=str, default=None)
    p_static.add_argument('--config', type=str, default=None)
    p_static.add_argument('--smoke-test', action='store_true')

    # -- dynamic ---------------------------------------------------------------
    p_dyn = sub.add_parser('dynamic', help='Dynamic SBI (per-animal)')
    p_dyn.add_argument('--animal', required=True)
    p_dyn.add_argument('--model', required=True, choices=['BE', 'SC'])
    p_dyn.add_argument('--link', default='randomwalk',
                       choices=['constant', 'randomwalk', 'gp'])
    p_dyn.add_argument('--fit-target', default='update_matrix',
                       choices=list(FIT_TARGETS))
    p_dyn.add_argument('--distribution', default='uniform')
    p_dyn.add_argument('--output-dir', type=str, default=None)
    p_dyn.add_argument('--config', type=str, default=None)
    p_dyn.add_argument('--seed', type=int, default=42)
    p_dyn.add_argument('--smoke-test', action='store_true')

    args = parser.parse_args()
    ensure_dirs()

    # == TRAIN =================================================================
    if args.mode == 'train':
        curriculum = parse_curriculum(args.curriculum)
        out = Path(args.output) if args.output else None

        t0 = time.time()
        run_train(
            model_type=args.model,
            curriculum=curriculum,
            n_simulations=args.n_sims,
            output_path=out,
            smoke_test=args.smoke_test,
            seed=args.seed,
        )
        print(f'  {time.time() - t0:.1f}s')

    # == STATIC ================================================================
    elif args.mode == 'static':
        from behav_utils.data.selection import select_sessions
        from behav_utils.data.filtering import filter_trials

        animal = load_animal_data(args.animal, config_path=args.config)
        sessions = select_sessions(animal, f'expert_{args.distribution}')
        clean = filter_trials(sessions)

        if not clean:
            print(f'No expert {args.distribution} sessions. Skipping.')
            sys.exit(0)

        snpe = Path(args.snpe_path) if args.snpe_path else None
        out_dir = Path(args.output_dir) if args.output_dir else (
            SBI_STATIC_DIR / f'{args.distribution}_{args.fit_target}')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{args.animal}_{args.model}.pkl'

        t0 = time.time()
        result = run_sbi_static(
            sessions=clean,
            model_type=args.model,
            fit_target=args.fit_target,
            snpe_path=snpe,
            animal_id=args.animal,
            distribution=args.distribution,
            smoke_test=args.smoke_test,
        )
        result['metadata'] = build_metadata('sbi.py static', vars(args))

        with open(out_path, 'wb') as f:
            pickle.dump(result, f)
        print(f'  {args.animal}/{args.model}: '
              f'err={result["mean_error"]:.6f} '
              f'({time.time() - t0:.1f}s) -> {out_path}')

    # == DYNAMIC ===============================================================
    elif args.mode == 'dynamic':
        from behav_utils.data.selection import select_sessions
        from behav_utils.data.filtering import filter_trials

        animal = load_animal_data(args.animal, config_path=args.config)
        sessions = select_sessions(animal, f'expert_{args.distribution}')
        clean = filter_trials(sessions)

        if not clean:
            print(f'No expert {args.distribution} sessions. Skipping.')
            sys.exit(0)

        out_dir = Path(args.output_dir) if args.output_dir else (
            SBI_DYNAMIC_DIR / f'{args.distribution}_{args.link}')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{args.animal}_{args.model}.pkl'

        t0 = time.time()
        result = run_sbi_dynamic(
            sessions=clean,
            model_type=args.model,
            link_type=args.link,
            fit_target=args.fit_target,
            animal_id=args.animal,
            distribution=args.distribution,
            smoke_test=args.smoke_test,
            seed=args.seed,
        )
        result['metadata'] = build_metadata('sbi.py dynamic', vars(args))

        with open(out_path, 'wb') as f:
            pickle.dump(result, f)
        print(f'  {args.animal}/{args.model}/{args.link}: '
              f'err={result["mean_error"]:.6f} '
              f'({time.time() - t0:.1f}s) -> {out_path}')


if __name__ == '__main__':
    main()
