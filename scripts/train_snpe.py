#!/usr/bin/env python3
"""
Train one amortised SNPE network.

Usage:
    python scripts/train_snpe.py --model be --distribution uniform
    python scripts/train_snpe.py --model sc --distribution uniform --smoke-test

Output:
    results/snpe/{distribution}_{model}.pkl
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

# Resolve repo root
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.config import (
    SBI_N_SIMULATIONS, SBI_N_GENERIC_TRIALS, SBI_BURN_IN, SBI_STATS,
    SNPE_DIR, BASE_SEED,
    SMOKE_SBI_N_SIMULATIONS, SMOKE_SBI_N_GENERIC_TRIALS,
    build_metadata, ensure_dirs,
)


def main():
    parser = argparse.ArgumentParser(description='Train amortised SNPE')
    parser.add_argument('--model', required=True, choices=['be', 'sc'],
                        help='Model type')
    parser.add_argument('--distribution', default='uniform',
                        choices=['uniform', 'hard_a', 'hard_b'],
                        help='Stimulus distribution for training simulations')
    parser.add_argument('--n-simulations', type=int, default=None,
                        help=f'Number of simulations (default: {SBI_N_SIMULATIONS})')
    parser.add_argument('--n-trials', type=int, default=None,
                        help=f'Trials per simulation (default: {SBI_N_GENERIC_TRIALS})')
    parser.add_argument('--seed', type=int, default=BASE_SEED)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run with tiny settings for pipeline validation')
    args = parser.parse_args()

    # Apply smoke-test overrides
    n_sims = args.n_simulations or (
        SMOKE_SBI_N_SIMULATIONS if args.smoke_test else SBI_N_SIMULATIONS
    )
    n_trials = args.n_trials or (
        SMOKE_SBI_N_GENERIC_TRIALS if args.smoke_test else SBI_N_GENERIC_TRIALS
    )

    output_dir = Path(args.output_dir) if args.output_dir else SNPE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{args.distribution}_{args.model}.pkl'

    print(f'=== Training SNPE: {args.model.upper()} on {args.distribution} ===')
    print(f'  Simulations: {n_sims:,}')
    print(f'  Trials/sim:  {n_trials}')
    print(f'  Burn-in:     {SBI_BURN_IN}')
    print(f'  Stats:       {SBI_STATS}')
    print(f'  Seed:        {args.seed}')
    print(f'  Output:      {output_path}')
    if args.smoke_test:
        print('  ** SMOKE TEST MODE **')

    from inference.comparison import train_amortised_snpe

    t0 = time.time()
    snpe_result = train_amortised_snpe(
        model_type=args.model,
        stat_names=SBI_STATS,
        n_simulations=n_sims,
        n_trials=n_trials,
        burn_in=SBI_BURN_IN,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    # Attach metadata
    meta = build_metadata('train_snpe.py', vars(args))
    meta['n_simulations_actual'] = n_sims
    meta['n_trials_actual'] = n_trials
    meta['training_time_seconds'] = elapsed

    # Save: posterior + metadata only (recreate simulator/prior on load)
    save_data = {
        'posterior': snpe_result['posterior'],
        'param_names': snpe_result['param_names'],
        'model_type': snpe_result['model_type'],
        'stat_names': snpe_result['stat_names'],
        'burn_in': snpe_result['burn_in'],
        'n_valid': snpe_result['n_valid'],
        'training_time': snpe_result['training_time'],
        'distribution': args.distribution,
        'metadata': meta,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)

    print(f'\nSaved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)')
    print(f'Total time: {elapsed / 60:.1f} min')


if __name__ == '__main__':
    main()
