#!/usr/bin/env python3
"""
Per-animal dynamic SBI with RandomWalk-linked parameters.

Trains one SNPE for a single animal with RandomWalkSpec on learning
parameters and ConstantSpec on perceptual parameters.

Usage:
    python scripts/run_sbi_dynamic_randomwalk.py \
        --animal SS01 --model be --fit-target update_matrix
    python scripts/run_sbi_dynamic_randomwalk.py \
        --animal SS01 --model be --fit-target update_matrix --smoke-test

Output:
    results/sbi_dynamic/{distribution}_{fit_target}/{animal}_{model}.pkl
"""

import argparse
import pickle
import sys
import time
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.config import (
    DYNAMIC_SBI_N_SIMULATIONS, DYNAMIC_SBI_SIGMA_DRIFT,
    DYNAMIC_SBI_VARYING_PARAMS, SBI_BURN_IN, SBI_STATS,
    SBI_DYNAMIC_DIR, BASE_SEED, FIT_TARGETS, STAGE,
    SMOKE_DYNAMIC_SBI_N_SIMULATIONS,
    build_metadata, ensure_dirs,
    load_animal_data,
)


def main():
    parser = argparse.ArgumentParser(
        description='Dynamic SBI with RandomWalk link')
    parser.add_argument('--animal', required=True, help='Animal ID')
    parser.add_argument('--model', required=True, choices=['be', 'sc'],
                        help='Model type (lowercase)')
    parser.add_argument('--fit-target', required=True, choices=list(FIT_TARGETS),
                        help='Matrix for model comparison scoring')
    parser.add_argument('--distribution', default='uniform',
                        choices=['uniform', 'hard_a', 'hard_b'])
    parser.add_argument('--n-simulations', type=int, default=None)
    parser.add_argument('--sigma-drift', type=float,
                        default=DYNAMIC_SBI_SIGMA_DRIFT)
    parser.add_argument('--varying-params', type=str, default=None,
                        help='Comma-separated param names (default: model-specific)')
    parser.add_argument('--seed', type=int, default=BASE_SEED)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to project config YAML (default: auto-detect)')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    n_sims = args.n_simulations or (
        SMOKE_DYNAMIC_SBI_N_SIMULATIONS if args.smoke_test
        else DYNAMIC_SBI_N_SIMULATIONS
    )
    model_key = args.model.upper()
    varying = (
        args.varying_params.split(',') if args.varying_params
        else list(DYNAMIC_SBI_VARYING_PARAMS[model_key])
    )

    output_dir = Path(args.output_dir) if args.output_dir else (
        SBI_DYNAMIC_DIR / f'{args.distribution}_{args.fit_target}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{args.animal}_{model_key}.pkl'

    print(f'=== Dynamic SBI (RandomWalk): {args.animal} / {model_key} ===')
    print(f'  Varying params: {varying}')
    print(f'  sigma_drift:    {args.sigma_drift}')
    print(f'  Simulations:    {n_sims:,}')
    print(f'  Fit target:     {args.fit_target}')
    if args.smoke_test:
        print('  ** SMOKE TEST MODE **')

    # Load real animal data
    from behav_utils.data.selection import select_sessions, fitting_data_from_sessions
    from inference.fitting import SBIFitter
    from inference.types import ConstantSpec, RandomWalkSpec

    animal = load_animal_data(args.animal, config_path=args.config)
    sessions = select_sessions(animal, f'expert_{args.distribution}')

    if not sessions:
        print(f'  No expert {args.distribution} sessions for {args.animal}. Skipping.')
        save_data = {
            'animal': args.animal, 'model': model_key,
            'n_sessions': 0, 'fit_target': args.fit_target,
            'metadata': build_metadata('run_sbi_dynamic_randomwalk.py', vars(args)),
        }
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        return

    fd = fitting_data_from_sessions(sessions, args.animal)
    print(f'  Sessions: {len(sessions)}')

    # Build param links
    if args.model == 'be':
        from models.BE_core import BEParams
        bounds = BEParams.get_bounds()
        all_params = BEParams.get_param_names()
    else:
        from models.SC_core import SCParams
        bounds = SCParams.get_bounds()
        all_params = SCParams.get_param_names()

    param_links = {}
    for pname in all_params:
        if pname in varying:
            param_links[pname] = RandomWalkSpec(
                bounds=bounds[pname],
                sigma_drift=args.sigma_drift,
            )
        else:
            param_links[pname] = ConstantSpec(bounds=bounds[pname])

    print(f'  Links: {", ".join(f"{k}={type(v).__name__}" for k, v in param_links.items())}')

    # Train
    t0 = time.time()
    fitter = SBIFitter(
        fitting_data=fd,
        model_type=args.model,
        param_links=param_links,
        burn_in=SBI_BURN_IN,
    )
    result = fitter.train(n_simulations=n_sims, seed=args.seed)
    trajectories = fitter.extract_trajectories(result, n_samples=500)
    elapsed = time.time() - t0

    print(f'  Training time: {elapsed / 60:.1f} min')

    # Save
    save_data = {
        'animal': args.animal,
        'model': model_key,
        'distribution': args.distribution,
        'fit_target': args.fit_target,
        'n_sessions': len(sessions),
        'varying_params': varying,
        'sigma_drift': args.sigma_drift,
        'n_simulations': n_sims,
        'trajectories': trajectories,
        'training_time': elapsed,
        'metadata': build_metadata('run_sbi_dynamic_randomwalk.py', vars(args)),
    }
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f'  Saved to {output_path}')


if __name__ == '__main__':
    main()
