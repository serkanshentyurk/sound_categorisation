#!/usr/bin/env python3
"""
Gather dynamic SBI validation results and compute recovery metrics.

Usage:
    python scripts/validation/gather_synth_sbi_dynamic.py
    python scripts/validation/gather_synth_sbi_dynamic.py --results-dir path/to/dir

Output:
    results/validation/synth_sbi_dynamic/summary.pkl
    Prints per-parameter recovery correlations, calibration, and
    model comparison accuracy.
"""

import argparse
import pickle
import sys
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.config import VALIDATION_DIR


def load_results(results_dir):
    """Load all dynamic validation result pickles."""
    results = []
    for p in sorted(results_dir.glob('*.pkl')):
        if p.name == 'summary.pkl':
            continue
        with open(p, 'rb') as f:
            data = pickle.load(f)
        # Verify it has the expected structure
        if 'true_fit' not in data:
            print(f'  Skipping {p.name}: missing true_fit (old format?)')
            continue
        results.append(data)
    return results


def compute_trajectory_recovery(result):
    """
    Per-parameter recovery metrics for one animal.

    Compares true per-session params against recovered trajectories
    from the true model fit.
    """
    true_fit = result['true_fit']
    trajectories = true_fit['trajectories']
    true_dicts = result['session_params_dicts']
    varying = true_fit['varying_params']
    n_sessions = result['n_sessions']

    metrics = {}
    for pname in varying:
        true_vals = np.array([d[pname] for d in true_dicts])

        traj = trajectories.get(pname, None)
        if traj is None:
            metrics[pname] = {'error': f'{pname} not in trajectories'}
            continue

        # extract_trajectories returns dict with median, ci_low, ci_high
        if not isinstance(traj, dict):
            metrics[pname] = {'error': f'unexpected type: {type(traj)}'}
            continue

        recovered_median = np.asarray(traj.get('median', []))
        recovered_ci_low = np.asarray(traj.get('ci_low', []))
        recovered_ci_high = np.asarray(traj.get('ci_high', []))

        if len(recovered_median) != n_sessions:
            metrics[pname] = {
                'error': f'length mismatch: {len(recovered_median)} vs {n_sessions}'
            }
            continue

        # Recovery quality
        mae = float(np.mean(np.abs(true_vals - recovered_median)))
        rmse = float(np.sqrt(np.mean((true_vals - recovered_median) ** 2)))

        # Correlation (does the trajectory shape match?)
        if np.std(true_vals) > 1e-8 and np.std(recovered_median) > 1e-8:
            r_pearson, p_pearson = pearsonr(true_vals, recovered_median)
            r_spearman, p_spearman = spearmanr(true_vals, recovered_median)
        else:
            r_pearson, p_pearson = np.nan, np.nan
            r_spearman, p_spearman = np.nan, np.nan

        # Calibration: fraction of true values within 95% CI
        within_ci = float(np.mean(
            (true_vals >= recovered_ci_low) & (true_vals <= recovered_ci_high)
        ))

        # Normalised error
        param_range = np.ptp(true_vals) if np.ptp(true_vals) > 1e-8 else 1.0
        nrmse = rmse / param_range

        metrics[pname] = {
            'true_trajectory': true_vals,
            'recovered_median': recovered_median,
            'recovered_ci_low': recovered_ci_low,
            'recovered_ci_high': recovered_ci_high,
            'mae': mae,
            'rmse': rmse,
            'nrmse': nrmse,
            'r_pearson': float(r_pearson),
            'p_pearson': float(p_pearson),
            'r_spearman': float(r_spearman),
            'p_spearman': float(p_spearman),
            'ci_coverage': within_ci,
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Gather dynamic SBI validation results')
    parser.add_argument('--results-dir', type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else (
        VALIDATION_DIR / 'synth_sbi_dynamic'
    )

    results = load_results(results_dir)
    if not results:
        print(f'No results found in {results_dir}')
        return

    print(f'Loaded {len(results)} results from {results_dir}\n')

    # ── Model comparison accuracy ────────────────────────────────────────────
    n_correct = sum(1 for r in results if r['correct'])
    print(f'Model comparison: {n_correct}/{len(results)} correct '
          f'({100 * n_correct / len(results):.0f}%)\n')

    # ── Trajectory recovery by model type ────────────────────────────────────
    all_metrics = []
    for res in results:
        m = compute_trajectory_recovery(res)
        m['_animal_id'] = res['animal_id']
        m['_model'] = res['true_model']
        m['_correct'] = res['correct']
        m['_true_ppc_mse'] = res['true_ppc_mse']
        m['_wrong_ppc_mse'] = res['wrong_ppc_mse']
        m['_training_time'] = (
            res['true_fit']['training_time']
            + res['wrong_fit']['training_time']
        )
        all_metrics.append(m)

    for model_type in ['BE', 'SC']:
        model_results = [m for m in all_metrics if m['_model'] == model_type]
        if not model_results:
            continue

        n_model_correct = sum(1 for m in model_results if m['_correct'])
        print(f'=== {model_type} ({len(model_results)} animals, '
              f'{n_model_correct} correct) ===')

        param_names = [k for k in model_results[0]
                       if not k.startswith('_') and isinstance(model_results[0][k], dict)
                       and 'error' not in model_results[0][k]]

        for pname in param_names:
            param_metrics = [
                m[pname] for m in model_results
                if isinstance(m.get(pname), dict) and 'error' not in m[pname]
            ]
            if not param_metrics:
                print(f'  {pname}: no valid results')
                continue

            rs = [m['r_pearson'] for m in param_metrics]
            maes = [m['mae'] for m in param_metrics]
            coverages = [m['ci_coverage'] for m in param_metrics]

            print(f'  {pname}:')
            print(f'    Pearson r:    {np.mean(rs):.3f} +/- {np.std(rs):.3f}'
                  f'  (range: {np.min(rs):.3f} - {np.max(rs):.3f})')
            print(f'    MAE:          {np.mean(maes):.4f} +/- {np.std(maes):.4f}')
            print(f'    95% CI cov:   {np.mean(coverages):.2f} +/- {np.std(coverages):.2f}'
                  f'  (ideal: 0.95)')

        # PPC MSE comparison
        true_mses = [m['_true_ppc_mse'] for m in model_results]
        wrong_mses = [m['_wrong_ppc_mse'] for m in model_results]
        print(f'  PPC MSE (true model):  {np.mean(true_mses):.6f} '
              f'+/- {np.std(true_mses):.6f}')
        print(f'  PPC MSE (wrong model): {np.mean(wrong_mses):.6f} '
              f'+/- {np.std(wrong_mses):.6f}')
        print()

    # ── Timing ───────────────────────────────────────────────────────────────
    times = [m['_training_time'] for m in all_metrics]
    if times:
        print(f'Total time per animal: '
              f'{np.mean(times)/60:.1f} +/- {np.std(times)/60:.1f} min')

    # ── Save summary ─────────────────────────────────────────────────────────
    summary_path = results_dir / 'summary.pkl'
    with open(summary_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'metrics': all_metrics,
            'n_correct': n_correct,
            'n_total': len(results),
        }, f)
    print(f'\nSaved summary to {summary_path}')


if __name__ == '__main__':
    main()
