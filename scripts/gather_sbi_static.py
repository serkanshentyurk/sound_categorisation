#!/usr/bin/env python3
"""
Aggregate static SBI comparison results into a single summary.

Reads per-animal pickles from results/sbi_static/comparisons/{distribution}_{fit_target}/
and produces a summary pickle + CSV.

Usage:
    python scripts/gather_sbi_static.py --distribution uniform --fit-target update_matrix
    python scripts/gather_sbi_static.py --all

Output:
    results/sbi_static/comparisons/{distribution}_{fit_target}/summary.pkl
    results/sbi_static/comparisons/{distribution}_{fit_target}/summary.csv

Also gathers posterior-only data from results/sbi_static/{distribution}/:
    results/sbi_static/{distribution}/summary.pkl
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

from scripts.config import SBI_STATIC_DIR, FIT_TARGETS, DISTRIBUTIONS, build_metadata


def gather_comparisons(comp_dir: Path, label: str = '') -> pd.DataFrame:
    """Gather per-animal SBI comparison pickles into a DataFrame."""
    pkl_files = sorted(comp_dir.glob('animal_*.pkl'))
    if not pkl_files:
        print(f'  No results in {comp_dir}')
        return pd.DataFrame()

    rows = []
    for path in pkl_files:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        rows.append({
            'animal': data.get('animal_id', path.stem.replace('animal_', '')),
            'distribution': data.get('distribution', 'uniform'),
            'fit_target': data.get('method', data.get('fit_target', 'update_matrix')),
            'winner': data.get('winner'),
            'p': data.get('p', data.get('p_value', np.nan)),
            'be_mean': data.get('be_mean', np.nan),
            'sc_mean': data.get('sc_mean', np.nan),
            'be_std': data.get('be_std', np.nan),
            'sc_std': data.get('sc_std', np.nan),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        print(f'  {label}: {len(df)} animals')
        if 'winner' in df.columns:
            vc = df['winner'].value_counts()
            for m, c in vc.items():
                print(f'    {m}: {c}')
    return df


def gather_posteriors(posterior_dir: Path, label: str = '') -> pd.DataFrame:
    """Gather posterior-only pickles (median params, no CV)."""
    pkl_files = sorted(posterior_dir.glob('animal_*.pkl'))
    if not pkl_files:
        print(f'  No posterior data in {posterior_dir}')
        return pd.DataFrame()

    rows = []
    for path in pkl_files:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        row = {
            'animal': data.get('animal_id', path.stem.replace('animal_', '')),
            'distribution': data.get('distribution', 'uniform'),
            'n_sessions': data.get('n_sessions', 0),
        }
        # Flatten BE and SC median params
        for model_key, prefix in [('be_params', 'be_'), ('sc_params', 'sc_')]:
            params = data.get(model_key, {})
            if isinstance(params, dict):
                for pn, pv in params.items():
                    row[f'{prefix}{pn}'] = pv
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        print(f'  {label} posteriors: {len(df)} animals')
    return df


def main():
    parser = argparse.ArgumentParser(description='Gather static SBI results')
    parser.add_argument('--distribution', default=None,
                        choices=list(DISTRIBUTIONS))
    parser.add_argument('--fit-target', default=None, choices=list(FIT_TARGETS))
    parser.add_argument('--all', action='store_true',
                        help='Gather all available distribution × fit_target combos')
    args = parser.parse_args()

    meta = build_metadata('gather_sbi_static.py', vars(args))

    # Determine what to gather
    if args.all:
        combos = []
        for d in DISTRIBUTIONS:
            for ft in FIT_TARGETS:
                comp_dir = SBI_STATIC_DIR / 'comparisons' / f'{d}_{ft}'
                if comp_dir.exists():
                    combos.append((d, ft))
    elif args.distribution and args.fit_target:
        combos = [(args.distribution, args.fit_target)]
    elif args.distribution:
        combos = [(args.distribution, ft) for ft in FIT_TARGETS]
    else:
        print('Specify --distribution + --fit-target, or --all')
        sys.exit(1)

    # Gather comparisons
    print('=== SBI Comparisons ===')
    for dist, ft in combos:
        comp_dir = SBI_STATIC_DIR / 'comparisons' / f'{dist}_{ft}'
        if not comp_dir.exists():
            print(f'  {dist}/{ft}: directory not found')
            continue

        df = gather_comparisons(comp_dir, label=f'{dist}/{ft}')
        if len(df) > 0:
            summary_path = comp_dir / 'summary.pkl'
            with open(summary_path, 'wb') as f:
                pickle.dump({'df': df, 'metadata': meta}, f)
            df.to_csv(comp_dir / 'summary.csv', index=False)
            print(f'    Saved {summary_path}')

    # Gather posterior-only data
    print('\n=== SBI Posteriors ===')
    distributions_seen = {d for d, _ in combos}
    for dist in distributions_seen:
        posterior_dir = SBI_STATIC_DIR / dist
        if not posterior_dir.exists():
            continue

        df = gather_posteriors(posterior_dir, label=dist)
        if len(df) > 0:
            summary_path = posterior_dir / 'summary.pkl'
            with open(summary_path, 'wb') as f:
                pickle.dump({'df': df, 'metadata': meta}, f)
            df.to_csv(posterior_dir / 'summary.csv', index=False)
            print(f'    Saved {summary_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
