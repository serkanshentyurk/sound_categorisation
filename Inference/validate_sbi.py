"""
SBI Validation Script for BE Model

This script demonstrates the full workflow:
1. Generate synthetic data from known parameters
2. Train SBI (NPE/NLE/NRE)
3. Run diagnostics (SBC, parameter recovery, posterior predictive)
4. Visualise results

Usage:
    python validate_sbi.py
    
    # Or with options:
    python validate_sbi.py --method NLE --n_simulations 30000
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description='Validate SBI for BE model')
    parser.add_argument('--method', type=str, default='NPE', 
                       choices=['NPE', 'NLE', 'NRE'],
                       help='SBI method to use')
    parser.add_argument('--n_simulations', type=int, default=50000,
                       help='Number of training simulations')
    parser.add_argument('--n_trials', type=int, default=500,
                       help='Number of trials per session')
    parser.add_argument('--n_sbc', type=int, default=500,
                       help='Number of SBC iterations')
    parser.add_argument('--n_recovery', type=int, default=200,
                       help='Number of parameter recovery tests')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--burn_in', type=int, default=100,
                       help='Burn-in trials for BE model')
    parser.add_argument('--save_dir', type=str, default='sbi_validation_results',
                       help='Directory to save results')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced samples')
    
    args = parser.parse_args()
    
    # Reduce samples for quick mode
    if args.quick:
        args.n_simulations = 10000
        args.n_sbc = 50
        args.n_recovery = 30
    
    print("="*70)
    print("SBI Validation for BE Model")
    print("="*70)
    print(f"Method: {args.method}")
    print(f"Simulations: {args.n_simulations}")
    print(f"Trials per session: {args.n_trials}")
    print(f"Burn-in: {args.burn_in}")
    print(f"Seed: {args.seed}")
    print("="*70)
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # =========================================================================
    # Step 1: Setup
    # =========================================================================
    print("\n[1/5] Setting up simulator and prior...")
    
    # Import modules
    try:
        from Inference.simulator import create_be_simulator
        from Inference.priors import create_prior, DEFAULT_BE_BOUNDS
        from Analysis.summary_stats import compute_summary_stats, get_stat_names_expanded, DEFAULT_STATS
        from Inference.sbi_wrapper import train_sbi, sample_posterior
        from Inference.diagnostics import (
            run_sbc, parameter_recovery, posterior_predictive_check,
            plot_sbc_ranks, plot_recovery_scatter, plot_posterior_predictive
        )
        from Helpers.utils import generate_stimuli
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nTrying alternative imports...")
        # For standalone testing
        from Inference import (
            create_be_simulator, create_prior, DEFAULT_BE_BOUNDS,
            compute_summary_stats, train_sbi, sample_posterior,
            run_sbc, parameter_recovery, posterior_predictive_check,
            plot_sbc_ranks, plot_recovery_scatter, plot_posterior_predictive
        )
        from Helpers.utils import generate_stimuli
    
    # Generate stimuli
    stimuli, categories, _ = generate_stimuli(
        n_trials=args.n_trials,
        seed=args.seed
    )
    
    # Create simulator
    simulator = create_be_simulator(
        stimuli=stimuli,
        categories=categories,
        burn_in=args.burn_in,
        seed=args.seed
    )
    
    # Create prior
    prior = create_prior()
    
    print(f"  Simulator: {simulator.n_free_params} free parameters")
    print(f"  Prior bounds: {DEFAULT_BE_BOUNDS}")
    print(f"  Summary stats: {DEFAULT_STATS}")
    
    # =========================================================================
    # Step 2: Generate "observed" data from known parameters
    # =========================================================================
    print("\n[2/5] Generating synthetic observed data...")
    
    # True parameters for testing
    true_params = {
        'sigma_percep': 0.15,
        'A_repulsion': 0.1,
        'eta_learning': 0.35,
        'eta_relax': 0.12
    }
    theta_true = np.array([
        true_params['sigma_percep'],
        true_params['A_repulsion'],
        true_params['eta_learning'],
        true_params['eta_relax']
    ])
    
    print(f"  True parameters: {true_params}")
    
    # Simulate observed data
    observed_stats = simulator(theta_true, seed=args.seed + 1000)
    observed_stats_tensor = torch.tensor(observed_stats, dtype=torch.float32)
    
    print(f"  Observed stats shape: {observed_stats.shape}")
    print(f"  Observed stats: {observed_stats[:4]}... (first 4)")
    
    # =========================================================================
    # Step 3: Train SBI
    # =========================================================================
    print(f"\n[3/5] Training {args.method}...")
    
    result = train_sbi(
        simulator=simulator,
        prior=prior,
        observed_stats=observed_stats_tensor,
        method=args.method,
        n_simulations=args.n_simulations,
        n_rounds=1,
        seed=args.seed,
        show_progress=True,
        param_names=list(true_params.keys())
    )
    
    print(f"  Training time: {result.training_time:.1f}s")
    
    # Quick posterior check
    samples = result.sample(5000, x=observed_stats_tensor)
    posterior_mean = samples.mean(dim=0).numpy()
    posterior_std = samples.std(dim=0).numpy()
    
    print("\n  Posterior summary (vs true):")
    for i, (name, true_val) in enumerate(true_params.items()):
        print(f"    {name}: {posterior_mean[i]:.3f} ± {posterior_std[i]:.3f} (true: {true_val})")
    
    # =========================================================================
    # Step 4: Run diagnostics
    # =========================================================================
    print(f"\n[4/5] Running diagnostics...")
    
    # --- SBC ---
    print("\n  Running Simulation-Based Calibration...")
    sbc_result = run_sbc(
        simulator=simulator,
        prior=prior,
        posterior=result.posterior,
        n_sbc=args.n_sbc,
        n_posterior_samples=500,
        param_names=list(true_params.keys()),
        seed=args.seed,
        show_progress=True
    )
    print(sbc_result.summary())
    
    # Plot SBC
    fig_sbc = plot_sbc_ranks(sbc_result, title=f'SBC Ranks ({args.method})')
    fig_sbc.savefig(os.path.join(args.save_dir, f'sbc_ranks_{args.method}.png'), 
                    dpi=150, bbox_inches='tight')
    print(f"  Saved: sbc_ranks_{args.method}.png")
    
    # --- Parameter Recovery ---
    print("\n  Running parameter recovery...")
    recovery_result = parameter_recovery(
        simulator=simulator,
        prior=prior,
        posterior=result.posterior,
        n_tests=args.n_recovery,
        n_posterior_samples=1000,
        param_names=list(true_params.keys()),
        seed=args.seed,
        show_progress=True
    )
    print(recovery_result.summary())
    
    # Plot recovery
    fig_recovery = plot_recovery_scatter(recovery_result, 
                                         title=f'Parameter Recovery ({args.method})')
    fig_recovery.savefig(os.path.join(args.save_dir, f'recovery_{args.method}.png'),
                         dpi=150, bbox_inches='tight')
    print(f"  Saved: recovery_{args.method}.png")
    
    # --- Posterior Predictive ---
    print("\n  Running posterior predictive check...")
    stat_names = get_stat_names_expanded(DEFAULT_STATS)
    ppc_result = posterior_predictive_check(
        simulator=simulator,
        posterior=result.posterior,
        observed_stats=observed_stats,
        n_samples=500,
        stat_names=stat_names,
        seed=args.seed,
        show_progress=True
    )
    print(ppc_result.summary())
    
    # Plot PPC
    fig_ppc = plot_posterior_predictive(ppc_result, 
                                        title=f'Posterior Predictive ({args.method})')
    fig_ppc.savefig(os.path.join(args.save_dir, f'ppc_{args.method}.png'),
                    dpi=150, bbox_inches='tight')
    print(f"  Saved: ppc_{args.method}.png")
    
    # =========================================================================
    # Step 5: Summary and posterior visualization
    # =========================================================================
    print(f"\n[5/5] Generating summary plots...")
    
    # Posterior corner plot (pairwise marginals)
    samples_np = samples.numpy()
    n_params = samples_np.shape[1]
    param_names = list(true_params.keys())
    
    fig, axes = plt.subplots(n_params, n_params, figsize=(10, 10))
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                ax.hist(samples_np[:, i], bins=30, density=True, 
                       color='steelblue', alpha=0.7)
                ax.axvline(theta_true[i], color='red', linewidth=2, 
                          label='True' if i == 0 else None)
                ax.axvline(posterior_mean[i], color='orange', linewidth=2, 
                          linestyle='--', label='Mean' if i == 0 else None)
                ax.set_xlabel(param_names[i])
            elif i > j:
                # Lower triangle: scatter
                ax.scatter(samples_np[:, j], samples_np[:, i], 
                          alpha=0.1, s=1, color='steelblue')
                ax.axvline(theta_true[j], color='red', linewidth=1, alpha=0.7)
                ax.axhline(theta_true[i], color='red', linewidth=1, alpha=0.7)
                ax.scatter([theta_true[j]], [theta_true[i]], 
                          color='red', s=50, zorder=10)
                ax.set_xlabel(param_names[j])
                ax.set_ylabel(param_names[i])
            else:
                # Upper triangle: hide
                ax.set_visible(False)
    
    axes[0, 0].legend(loc='upper right', fontsize=8)
    fig.suptitle(f'Posterior Samples ({args.method})', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_dir, f'posterior_{args.method}.png'),
                dpi=150, bbox_inches='tight')
    print(f"  Saved: posterior_{args.method}.png")
    
    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.save_dir}/")
    print("\nKey findings:")
    
    # Check calibration
    calibrated = sbc_result.is_calibrated(alpha=0.05)
    n_calibrated = sum(calibrated.values())
    print(f"  SBC calibration: {n_calibrated}/{len(calibrated)} parameters pass uniformity test")
    
    # Check recovery
    good_recovery = sum(1 for r in recovery_result.correlations.values() if r > 0.7)
    print(f"  Parameter recovery: {good_recovery}/{len(recovery_result.correlations)} params with r > 0.7")
    
    # Check coverage
    expected_coverage = 0.95
    coverage_ok = sum(1 for c in recovery_result.coverages.values() 
                     if abs(c - expected_coverage) < 0.1)
    print(f"  Coverage calibration: {coverage_ok}/{len(recovery_result.coverages)} params within 10% of 95%")
    
    print("\nDone!")
    plt.close('all')
    
    return result, sbc_result, recovery_result, ppc_result


if __name__ == '__main__':
    main()
