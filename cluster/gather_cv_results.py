#!/usr/bin/env python3
"""
gather_cv_results.py — Merge per-seed CV pickles into DataFrames + plots.

Run after all SLURM array jobs complete:
    python gather_cv_results.py --results-dir ./results/cv --output-dir ./results/cv
"""

import argparse
import os
import pickle

from analysis.cv_utils import (
    load_cv_pickles, summarise_loaded_results,
    build_long_df, run_anova, build_summary_table,
)
from plotting.cv import plot_cv_comparison, plot_winner_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', required=True)
    p.add_argument('--output-dir', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Loading results...")
    all_results = load_cv_pickles(args.results_dir)
    summarise_loaded_results(all_results)

    print("\nBuilding long DataFrame...")
    long_df = build_long_df(all_results)
    long_df.to_csv(os.path.join(output_dir, 'cv_test_errors_long.csv'), index=False)

    print("\nRunning ANOVA...")
    comparison_df = run_anova(long_df)
    comparison_df.to_csv(os.path.join(output_dir, 'cv_comparison_anova.csv'), index=False)
    for _, row in comparison_df.iterrows():
        print(f"  {row['animal_id']}: BE={row['be_mean']:.5f}  "
              f"SC={row['sc_mean']:.5f}  p={row['p_value']:.2e}  → {row['winner']}")

    print("\nBuilding summary table...")
    summary_df = build_summary_table(all_results, comparison_df)
    summary_df.to_csv(os.path.join(output_dir, 'cv_summary.csv'), index=False)

    print("\nPlotting...")
    for aid in sorted(long_df['animal_id'].unique()):
        plot_cv_comparison(long_df, comparison_df, aid, output_dir=output_dir)
    plot_winner_summary(comparison_df, output_dir=output_dir)

    with open(os.path.join(output_dir, 'cv_all_results_merged.pkl'), 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\nDone. Outputs in {output_dir}/")


if __name__ == '__main__':
    main()
