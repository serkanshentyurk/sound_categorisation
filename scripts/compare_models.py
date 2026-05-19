#!/usr/bin/env python3
"""
Model Comparison

Loads BE and SC result files from grid_search.py or sbi.py,
runs a statistical test on their CV error distributions,
and outputs the winner.

Agnostic to method — works with any result dict that has 'cv_errors'.

CLI usage:
    python scripts/compare_models.py \
        --be-path results/cv/uniform_update_matrix/SS01_BE.pkl \
        --sc-path results/cv/uniform_update_matrix/SS01_SC.pkl

    python scripts/compare_models.py \
        --results-dir results/cv/uniform_update_matrix \
        --animal SS01

Importable:
    from scripts.compare_models import compare_results

    comparison = compare_results(be_result, sc_result)
    # comparison['winner'], comparison['p_value'], ...

Output schema:
    {
        'animal_id': str,
        'method': str,              # 'grid_search' | 'sbi_static' | ...
        'fit_target': str,
        'distribution': str,

        'winner': 'BE' | 'SC',
        'p_value': float,           # Wilcoxon signed-rank p
        'significant': bool,        # p < alpha

        'be_mean_error': float,
        'sc_mean_error': float,
        'be_std_error': float,
        'sc_std_error': float,

        'effect_size': float,       # (sc_mean - be_mean) / pooled_std
        'n_cv': int,                # number of paired observations

        'metadata': dict,
    }
"""

import argparse
import pickle
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# CORE FUNCTION (importable)
# =============================================================================

def compare_results(
    be_result: Dict[str, Any],
    sc_result: Dict[str, Any],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compare BE and SC results via Wilcoxon signed-rank test on CV errors.

    Args:
        be_result: Result dict from grid_search.py or sbi.py with model_type='BE'.
        sc_result: Result dict from grid_search.py or sbi.py with model_type='SC'.
        alpha: Significance threshold.

    Returns:
        Comparison dict (see module docstring).
    """
    from scipy.stats import wilcoxon # type: ignore[import]

    # Warn if comparing different CV types
    be_cv_type = be_result.get('cv_type', 'unknown')
    sc_cv_type = sc_result.get('cv_type', 'unknown')
    if be_cv_type != sc_cv_type:
        import warnings
        warnings.warn(
            f'Comparing {be_cv_type} errors (BE) with {sc_cv_type} errors (SC) '
            f'— interpret with caution')

    be_errors = np.array(be_result['cv_errors'])
    sc_errors = np.array(sc_result['cv_errors'])

    # Truncate to same length if needed
    n = min(len(be_errors), len(sc_errors))
    be_errors = be_errors[:n]
    sc_errors = sc_errors[:n]

    be_mean = float(np.mean(be_errors))
    sc_mean = float(np.mean(sc_errors))
    be_std = float(np.std(be_errors))
    sc_std = float(np.std(sc_errors))

    winner = 'BE' if be_mean < sc_mean else 'SC'

    # Wilcoxon signed-rank test (paired)
    if n >= 5:
        try:
            _, p_value = wilcoxon(be_errors, sc_errors)
            p_value = float(p_value)
        except ValueError:
            # All differences are zero
            p_value = 1.0
    else:
        p_value = np.nan

    # Effect size (Cohen's d, pooled)
    pooled_std = np.sqrt((be_std**2 + sc_std**2) / 2)
    if pooled_std > 0:
        effect_size = float((sc_mean - be_mean) / pooled_std)
    else:
        effect_size = 0.0

    # Infer shared fields from either result
    method = be_result.get('method', sc_result.get('method', 'unknown'))
    fit_target = be_result.get('fit_target', sc_result.get('fit_target', ''))
    distribution = be_result.get('distribution',
                                 sc_result.get('distribution', ''))
    animal_id = be_result.get('animal_id', sc_result.get('animal_id', ''))

    return {
        'animal_id': animal_id,
        'method': method,
        'cv_type': be_cv_type if be_cv_type == sc_cv_type else 'mixed',
        'fit_target': fit_target,
        'distribution': distribution,

        'winner': winner,
        'p_value': p_value,
        'significant': bool(p_value < alpha) if not np.isnan(p_value) else False,

        'be_mean_error': be_mean,
        'sc_mean_error': sc_mean,
        'be_std_error': be_std,
        'sc_std_error': sc_std,

        'effect_size': effect_size,
        'n_cv': n,

        'metadata': None,
    }


def load_and_compare(
    be_path: Path,
    sc_path: Path,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Load two result pickles and compare them."""
    with open(be_path, 'rb') as f:
        be_result = pickle.load(f)
    with open(sc_path, 'rb') as f:
        sc_result = pickle.load(f)

    return compare_results(be_result, sc_result, alpha=alpha)


# =============================================================================
# CLI
# =============================================================================

def main():
    from scripts.config import build_metadata

    parser = argparse.ArgumentParser(
        description='Compare BE vs SC results')

    # Option 1: explicit paths
    parser.add_argument('--be-path', type=str, default=None)
    parser.add_argument('--sc-path', type=str, default=None)

    # Option 2: directory + animal (auto-discovers BE/SC files)
    parser.add_argument('--results-dir', type=str, default=None)
    parser.add_argument('--animal', type=str, default=None)

    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--output', type=str, default=None,
                        help='Save comparison result to this path')
    args = parser.parse_args()

    # Resolve paths
    if args.be_path and args.sc_path:
        be_path = Path(args.be_path)
        sc_path = Path(args.sc_path)
    elif args.results_dir and args.animal:
        d = Path(args.results_dir)
        be_path = d / f'{args.animal}_BE.pkl'
        sc_path = d / f'{args.animal}_SC.pkl'
    else:
        parser.error('Provide --be-path + --sc-path, or --results-dir + --animal')

    if not be_path.exists():
        print(f'BE result not found: {be_path}')
        sys.exit(1)
    if not sc_path.exists():
        print(f'SC result not found: {sc_path}')
        sys.exit(1)

    comparison = load_and_compare(be_path, sc_path, alpha=args.alpha)
    comparison['metadata'] = build_metadata('compare_models.py', vars(args))

    sig = '✓' if comparison['significant'] else '✗'
    print(f'{comparison["animal_id"]}: '
          f'{comparison["winner"]} wins  '
          f'(p={comparison["p_value"]:.4f} {sig}, '
          f'd={comparison["effect_size"]:.2f}, '
          f'BE={comparison["be_mean_error"]:.6f}, '
          f'SC={comparison["sc_mean_error"]:.6f})')

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'wb') as f:
            pickle.dump(comparison, f)
        print(f'Saved to {out}')


if __name__ == '__main__':
    main()
