#!/usr/bin/env python3
"""
Aggregate cluster GS results into a single DataFrame.

Reads per-animal pickles from results/cv/{distribution}_{fit_target}/
and produces a summary CSV and pickle.

Usage:
    python scripts/gather_cv_results.py --distribution uniform --fit-target update_matrix
    python scripts/gather_cv_results.py --all

Output:
    results/cv/{distribution}_{fit_target}/summary.pkl
    results/cv/{distribution}_{fit_target}/summary.csv
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

from scripts.config import CV_DIR, FIT_TARGETS, VALIDATION_DIR, build_metadata


def gather_one(results_dir: Path, label: str = '') -> tuple:
    """
    Gather all per-animal GS pickles in a directory.

    Returns
    -------
    (pivot_df, detail_df)
        pivot_df: one row per animal with BE/SC mean errors and winner.
        detail_df: one row per (animal × seed) with individual errors
            and best params — needed for violin plots and parameter recovery.
    """
    pkl_files = sorted(results_dir.glob('cv_*.pkl'))
    if not pkl_files:
        print(f'  No results in {results_dir}')
        return pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    detail_rows = []

    for path in pkl_files:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if data.get('n_sessions', 0) == 0:
            continue

        animal = data['animal']
        model = data['model']
        fit_target = data.get('fit_target', 'update_matrix')
        distribution = data.get('distribution', 'uniform')

        summary_rows.append({
            'animal': animal,
            'model': model,
            'fit_target': fit_target,
            'distribution': distribution,
            'n_sessions': data['n_sessions'],
            'n_seeds': data.get('n_seeds', 0),
            'mean_error': data.get('mean_error', np.nan),
        })

        # Per-seed detail
        for r in data.get('results', []):
            err = r.get('avg_test_error', np.nan)
            if np.isnan(err):
                continue
            detail_row = {
                'animal': animal,
                'model': model,
                'fit_target': fit_target,
                'distribution': distribution,
                'seed': r.get('seed', np.nan),
                'avg_test_error': err,
            }
            # Flatten best params
            bp = r.get('best_params', {})
            if isinstance(bp, dict):
                for pn, pv in bp.items():
                    detail_row[f'param_{pn}'] = pv
            detail_rows.append(detail_row)

    df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    if len(df) == 0:
        return df, detail_df

    # Pivot: one row per animal, columns for BE and SC errors
    pivot = df.pivot_table(
        index=['animal', 'fit_target', 'distribution', 'n_sessions'],
        columns='model',
        values='mean_error',
    ).reset_index()
    pivot.columns.name = None

    if 'BE' in pivot.columns and 'SC' in pivot.columns:
        pivot['winner'] = np.where(pivot['BE'] < pivot['SC'], 'BE', 'SC')
    else:
        pivot['winner'] = np.nan

    print(f'  {label}: {len(pivot)} animals')
    if 'winner' in pivot.columns:
        vc = pivot['winner'].value_counts()
        for m, c in vc.items():
            print(f'    {m}: {c}')

    return pivot, detail_df


def main():
    parser = argparse.ArgumentParser(description='Gather GS CV results')
    parser.add_argument('--distribution', default=None,
                        choices=['uniform', 'hard_a', 'hard_b'])
    parser.add_argument('--fit-target', default=None, choices=list(FIT_TARGETS))
    parser.add_argument('--all', action='store_true',
                        help='Gather all available distribution × fit_target combos')
    parser.add_argument('--include-validation', action='store_true',
                        help='Also gather synthetic validation results')
    args = parser.parse_args()

    meta = build_metadata('gather_cv_results.py', vars(args))

    if args.all:
        combos = []
        for d in ['uniform', 'hard_a', 'hard_b']:
            for ft in FIT_TARGETS:
                results_dir = CV_DIR / f'{d}_{ft}'
                if results_dir.exists():
                    combos.append((d, ft, results_dir))
    elif args.distribution and args.fit_target:
        results_dir = CV_DIR / f'{args.distribution}_{args.fit_target}'
        combos = [(args.distribution, args.fit_target, results_dir)]
    else:
        print('Specify --distribution + --fit-target, or --all')
        sys.exit(1)

    all_dfs = []
    for dist, ft, results_dir in combos:
        label = f'{dist}/{ft}'
        if not results_dir.exists():
            print(f'  {label}: directory not found')
            continue
        df, detail_df = gather_one(results_dir, label)
        if len(df) > 0:
            summary_path = results_dir / 'summary.pkl'
            with open(summary_path, 'wb') as f:
                pickle.dump({
                    'df': df,
                    'detail_df': detail_df,
                    'metadata': meta,
                }, f)
            df.to_csv(results_dir / 'summary.csv', index=False)
            if len(detail_df) > 0:
                detail_df.to_csv(results_dir / 'detail.csv', index=False)
            print(f'    Saved {summary_path}')
            all_dfs.append(df)

    # Synthetic validation
    if args.include_validation:
        print('\n=== Synthetic Validation ===')
        for cohort in ['static_uniform', 'learning_uniform']:
            for ft in FIT_TARGETS:
                synth_dir = VALIDATION_DIR / 'synth_gs' / f'{cohort}_{ft}'
                if not synth_dir.exists():
                    continue
                pkl_files = sorted(synth_dir.glob('synth_*.pkl'))
                if not pkl_files:
                    continue

                rows = []
                for path in pkl_files:
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                    rows.append({
                        'animal_id': data['animal_id'],
                        'true_model': data['true_model'],
                        'model': data['model'],
                        'fit_target': data['fit_target'],
                        'cohort': data['cohort'],
                        'mean_error': data['mean_error'],
                    })

                df = pd.DataFrame(rows)
                if len(df) == 0:
                    continue

                pivot = df.pivot_table(
                    index=['animal_id', 'true_model', 'fit_target', 'cohort'],
                    columns='model', values='mean_error',
                ).reset_index()
                pivot.columns.name = None

                if 'BE' in pivot.columns and 'SC' in pivot.columns:
                    pivot['winner'] = np.where(pivot['BE'] < pivot['SC'], 'BE', 'SC')
                    pivot['correct'] = pivot['winner'] == pivot['true_model']
                    acc = pivot['correct'].mean()
                    print(f'  {cohort}/{ft}: '
                          f'{pivot["correct"].sum()}/{len(pivot)} correct '
                          f'({acc:.0%})')

                summary_path = synth_dir / 'summary.pkl'
                with open(summary_path, 'wb') as f:
                    pickle.dump({'df': pivot, 'metadata': meta}, f)
                pivot.to_csv(synth_dir / 'summary.csv', index=False)

    print('\nDone.')


if __name__ == '__main__':
    main()
