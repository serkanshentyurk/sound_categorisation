#!/usr/bin/env python3
"""
Gather dynamic SBI (RandomWalk) results into a summary.

Reads per-animal pickles from results/sbi_dynamic/{distribution}_{fit_target}/
and produces a summary pickle + CSV.

Usage:
    python scripts/gather_dynamic_sbi.py --distribution uniform --fit-target update_matrix
    python scripts/gather_dynamic_sbi.py --all

Output:
    results/sbi_dynamic/{distribution}_{fit_target}/summary.pkl
    results/sbi_dynamic/{distribution}_{fit_target}/summary.csv
"""

import argparse
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.config import SBI_DYNAMIC_DIR, FIT_TARGETS, build_metadata


def gather_one(results_dir: Path, label: str = '') -> pd.DataFrame:
    """Gather all per-animal dynamic SBI pickles in a directory."""
    pkl_files = sorted(results_dir.glob('*.pkl'))
    # Exclude summary.pkl if it exists
    pkl_files = [p for p in pkl_files if p.name != 'summary.pkl']
    if not pkl_files:
        print(f'  No results in {results_dir}')
        return pd.DataFrame()

    rows = []
    trajectories = {}

    for path in pkl_files:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if data.get('n_sessions', 0) == 0:
            continue

        aid = data['animal']
        model = data['model']

        traj = data.get('trajectories', {})
        if traj:
            trajectories[f'{aid}_{model}'] = traj

        # Extract summary stats from trajectories
        row = {
            'animal': aid,
            'model': model,
            'distribution': data.get('distribution', 'uniform'),
            'fit_target': data.get('fit_target', 'update_matrix'),
            'n_sessions': data['n_sessions'],
            'varying_params': data.get('varying_params', []),
            'sigma_drift': data.get('sigma_drift', np.nan),
            'training_time_s': data.get('training_time', np.nan),
        }

        # Extract trajectory summaries for varying params
        for pname in data.get('varying_params', []):
            if pname in traj:
                medians = traj[pname].get('median', [])
                if len(medians) > 0:
                    row[f'{pname}_start'] = medians[0]
                    row[f'{pname}_end'] = medians[-1]
                    row[f'{pname}_range'] = max(medians) - min(medians)
                    row[f'{pname}_trend'] = medians[-1] - medians[0]

        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    print(f'  {label}: {len(df)} results ({df["animal"].nunique()} animals)')
    return df, trajectories


def main():
    parser = argparse.ArgumentParser(description='Gather dynamic SBI results')
    parser.add_argument('--distribution', default=None,
                        choices=['uniform', 'hard_a', 'hard_b'])
    parser.add_argument('--fit-target', default=None, choices=list(FIT_TARGETS))
    parser.add_argument('--all', action='store_true',
                        help='Gather all available distribution × fit_target combos')
    args = parser.parse_args()

    meta = build_metadata('gather_dynamic_sbi.py', vars(args))

    if args.all:
        combos = []
        for d in ['uniform', 'hard_a', 'hard_b']:
            for ft in FIT_TARGETS:
                results_dir = SBI_DYNAMIC_DIR / f'{d}_{ft}'
                if results_dir.exists():
                    combos.append((d, ft, results_dir))
    elif args.distribution and args.fit_target:
        results_dir = SBI_DYNAMIC_DIR / f'{args.distribution}_{args.fit_target}'
        combos = [(args.distribution, args.fit_target, results_dir)]
    else:
        print('Specify --distribution + --fit-target, or --all')
        sys.exit(1)

    for dist, ft, results_dir in combos:
        label = f'{dist}/{ft}'
        if not results_dir.exists():
            print(f'  {label}: directory not found')
            continue

        result = gather_one(results_dir, label)
        if isinstance(result, tuple):
            df, trajectories = result
        else:
            df = result
            trajectories = {}

        if len(df) > 0:
            summary_path = results_dir / 'summary.pkl'
            with open(summary_path, 'wb') as f:
                pickle.dump({
                    'df': df,
                    'trajectories': trajectories,
                    'metadata': meta,
                }, f)
            df.to_csv(results_dir / 'summary.csv', index=False)
            print(f'    Saved {summary_path}')

            # Print trajectory summaries
            for _, row in df.iterrows():
                varying = row.get('varying_params', [])
                parts = []
                for p in varying:
                    trend_col = f'{p}_trend'
                    if trend_col in row and pd.notna(row[trend_col]):
                        parts.append(f'{p}: {row[trend_col]:+.4f}')
                if parts:
                    print(f'    {row["animal"]}/{row["model"]}: {", ".join(parts)}')

    print('\nDone.')


if __name__ == '__main__':
    main()
