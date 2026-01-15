"""
Parameter recovery analysis for the Boundary Estimation model.

This module provides functions to test how well model parameters can be 
recovered under various conditions (e.g., different burn-in assumptions).

Analysis Functions:
    - burn_in_recovery_analysis: Test parameter recovery across burn-in conditions
    - burn_in_recovery_summary_stats: Generate summary statistics table
    - fit_and_evaluate: Convenience function for fitting and full diagnostics

Plotting Functions (wrappers using Plotting modules):
    - plot_psychometric_by_burn_in: Visualise psychometric curves across burn-in
    - plot_belief_after_burn_in: Visualise belief distributions across burn-in

Note: Core plotting functions are in the Plotting module. The functions here
are specific wrappers that include model simulation for burn-in analysis.
"""

from Models.BE_model import BoundaryEstimationModel

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple

from Helpers.psychometry import fit_psychometric, compute_psychometric_gof, compute_psych_error
from Helpers.utils import generate_stimuli
from Plotting.psychometric import plot_psychometric
from Plotting.belief import plot_belief_distributions, plot_belief_uncertainty
from Plotting.recovery import plot_burn_in_recovery, plot_burn_in_param_distributions


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def burn_in_recovery_analysis(
    true_params: Dict[str, float],
    burn_in_values: List[int],
    n_trials: int = 300,
    n_replicates: int = 20,
    fitter_burn_in: int = 0,
    validation: Optional[str] = 'holdout',
    validation_config: Optional[Dict] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Test how burn-in affects parameter and behavioural recovery.
    
    Simulates data from a model with known parameters and varying burn-in,
    then fits assuming a fixed (typically naive) burn-in. This tests robustness
    of single-session fitting when the animal's prior experience is unknown.
    
    NOTE: Even burn_in=0 assumes the animal has the BE task schema (knows there's
    a boundary to estimate). This does not model truly naive animals who haven't
    yet learned the task structure.
    
    Args:
        true_params: Dict of true parameter values (fixed across all conditions)
        burn_in_values: List of burn-in values to test, e.g., [0, 100, 500, 1000, 5000]
        n_trials: Number of trials per simulated session
        n_replicates: Number of replicates per burn-in condition
        fitter_burn_in: Burn-in assumed by fitter (default 0 = naive assumption)
        validation: Validation method for fitting ('holdout', 'cv', or None)
        validation_config: Validation configuration
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with:
            'param_recovery': {burn_in: {param_name: {...}}}
            'psych_recovery': {burn_in: {mu, sigma, curve_mae, ...}}
            'fit_quality': {burn_in: {train_nll, test_nll}}
            'config': {experiment configuration}
    """
    # rng = np.random.default_rng(seed)
    
    param_names = BoundaryEstimationModel.get_param_names()
    
    # Storage
    results = {
        'param_recovery': {},
        'psych_recovery': {},
        'fit_quality': {},
        'config': {
            'true_params': true_params,
            'burn_in_values': burn_in_values,
            'n_trials': n_trials,
            'n_replicates': n_replicates,
            'fitter_burn_in': fitter_burn_in,
            'validation': validation,
            'seed': seed
        }
    }
    
    # Common stimuli for all (can vary per replicate if desired)
    x_eval = np.linspace(-1, 1, 100)
    
    for burn_in in burn_in_values:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing burn_in = {burn_in}")
            print(f"{'='*60}")
        
        # Initialise storage for this burn-in
        param_storage = {name: {'true': [], 'fitted': [], 'error': []} 
                        for name in param_names}
        psych_storage = {
            'mu': {'true': [], 'fitted': [], 'error': []},
            'sigma': {'true': [], 'fitted': [], 'error': []},
            'lapse_low': {'true': [], 'fitted': [], 'error': []},
            'lapse_high': {'true': [], 'fitted': [], 'error': []},
            'curve_mae': [],
            'curve_max_diff': []
        }
        fit_storage = {'train_nll': [], 'test_nll': []}
        
        for rep in range(n_replicates):
            if verbose:
                print(f"  Replicate {rep+1}/{n_replicates}...", end=' ')
            
            rep_seed = seed + burn_in * 10000 + rep * 100
            
            # Generate stimuli for this replicate
            stimuli, categories, rep_rng = generate_stimuli(n_trials = n_trials,
                                            x_min = -1,
                                            x_max = 1,
                                            seed = rep_seed)
            
            # --- True model ---
            true_model = BoundaryEstimationModel(**true_params)
            true_model.reset_belief(burn_in=burn_in, burn_in_seed=rep_seed)
            
            # Simulate choices
            sim_rng = np.random.default_rng(rep_seed + 1)
            true_choices, _ = true_model.simulate_session(stimuli, categories, rng=sim_rng)
            
            # Fit psychometric to true model's choices
            valid_mask = ~np.isnan(true_choices)
            true_psych = fit_psychometric(stimuli[valid_mask], true_choices[valid_mask], x_eval)
            
            # --- Fit model (assuming fitter_burn_in) ---
            try:
                fitted_model, fit_results = BoundaryEstimationModel.fit(
                    stimuli, categories, true_choices,
                    burn_in=fitter_burn_in,
                    burn_in_seed=rep_seed,  # Same seed for reproducibility
                    validation=validation,
                    validation_config=validation_config,
                    n_restarts=5,
                    seed=rep_seed + 2
                )
                
                # Store parameter results
                for name in param_names:
                    param_storage[name]['true'].append(true_params[name])
                    param_storage[name]['fitted'].append(fit_results['params'][name])
                    param_storage[name]['error'].append(
                        fit_results['params'][name] - true_params[name]
                    )
                
                # Store fit quality
                fit_storage['train_nll'].append(fit_results.get('train_nll_per_trial', np.nan))
                if 'test_nll_per_trial' in fit_results:
                    fit_storage['test_nll'].append(fit_results['test_nll_per_trial'])
                
                # --- Simulate from fitted model ---
                fitted_model.reset_belief(burn_in=fitter_burn_in, burn_in_seed=rep_seed)
                fitted_rng = np.random.default_rng(rep_seed + 3)
                fitted_choices, _ = fitted_model.simulate_session(stimuli, categories, rng=fitted_rng)
                
                # Fit psychometric to fitted model's choices
                valid_fitted = ~np.isnan(fitted_choices)
                fitted_psych = fit_psychometric(stimuli[valid_fitted], fitted_choices[valid_fitted], x_eval)
                
                # Compute psychometric errors
                psych_errors = compute_psych_error(true_psych, fitted_psych)
                
                for key in ['mu', 'sigma', 'lapse_low', 'lapse_high']:
                    psych_storage[key]['true'].append(psych_errors[f'{key}_true'])
                    psych_storage[key]['fitted'].append(psych_errors[f'{key}_fitted'])
                    psych_storage[key]['error'].append(psych_errors[key])
                
                psych_storage['curve_mae'].append(psych_errors.get('curve_mae', np.nan))
                psych_storage['curve_max_diff'].append(psych_errors.get('curve_max_diff', np.nan))
                
                if verbose:
                    print("OK")
                    
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")
                
                # Fill with NaN
                for name in param_names:
                    param_storage[name]['true'].append(true_params[name])
                    param_storage[name]['fitted'].append(np.nan)
                    param_storage[name]['error'].append(np.nan)
                
                for key in ['mu', 'sigma', 'lapse_low', 'lapse_high']:
                    psych_storage[key]['true'].append(np.nan)
                    psych_storage[key]['fitted'].append(np.nan)
                    psych_storage[key]['error'].append(np.nan)
                
                psych_storage['curve_mae'].append(np.nan)
                psych_storage['curve_max_diff'].append(np.nan)
                fit_storage['train_nll'].append(np.nan)
        
        # Convert to arrays and compute summary statistics
        for name in param_names:
            for key in ['true', 'fitted', 'error']:
                param_storage[name][key] = np.array(param_storage[name][key])
            
            # Summary stats
            errors = param_storage[name]['error']
            param_storage[name]['mean_error'] = np.nanmean(errors)
            param_storage[name]['std_error'] = np.nanstd(errors)
            param_storage[name]['abs_mean_error'] = np.nanmean(np.abs(errors))
            
            # Fitted std across replicates
            fitted = param_storage[name]['fitted']
            if np.sum(~np.isnan(fitted)) >= 2:
                param_storage[name]['fitted_std'] = np.nanstd(fitted)
            else:
                param_storage[name]['fitted_std'] = np.nan
        
        for key in ['mu', 'sigma', 'lapse_low', 'lapse_high']:
            for subkey in ['true', 'fitted', 'error']:
                psych_storage[key][subkey] = np.array(psych_storage[key][subkey])
            psych_storage[key]['mean_error'] = np.nanmean(psych_storage[key]['error'])
            psych_storage[key]['std_error'] = np.nanstd(psych_storage[key]['error'])
        
        psych_storage['curve_mae'] = np.array(psych_storage['curve_mae'])
        psych_storage['curve_max_diff'] = np.array(psych_storage['curve_max_diff'])
        psych_storage['mean_curve_mae'] = np.nanmean(psych_storage['curve_mae'])
        
        for key in ['train_nll', 'test_nll']:
            fit_storage[key] = np.array(fit_storage[key]) if fit_storage[key] else np.array([])
        
        # Store
        results['param_recovery'][burn_in] = param_storage
        results['psych_recovery'][burn_in] = psych_storage
        results['fit_quality'][burn_in] = fit_storage
    
    return results


def burn_in_recovery_summary_stats(results: Dict) -> pd.DataFrame:
    """
    Generate summary statistics table for burn-in recovery analysis.
    
    Args:
        results: Output from burn_in_recovery_analysis()
    
    Returns:
        DataFrame with summary statistics for each burn-in condition, including:
        - BE model parameter recovery (bias, std for each)
        - Psychometric parameter recovery (μ, σ, λ_low, λ_high)
        - Curve error (MAE between true and fitted psychometric curves)
        - Fit quality (NLL per trial)
    """
    burn_in_values = results['config']['burn_in_values']
    true_params = results['config']['true_params']
    param_names = list(true_params.keys())
    
    rows = []
    
    for burn_in in burn_in_values:
        row = {'burn_in': burn_in}
        
        # BE model parameter recovery
        for name in param_names:
            data = results['param_recovery'][burn_in][name]
            row[f'{name}_true'] = true_params[name]
            row[f'{name}_mean'] = np.nanmean(data['fitted'])
            row[f'{name}_std'] = np.nanstd(data['fitted'])
            row[f'{name}_bias'] = data['mean_error']
            row[f'{name}_abs_error'] = data['abs_mean_error']
        
        # Psychometric parameter recovery (all 4 params)
        for key in ['mu', 'sigma', 'lapse_low', 'lapse_high']:
            if key in results['psych_recovery'][burn_in]:
                data = results['psych_recovery'][burn_in][key]
                row[f'psych_{key}_bias'] = data['mean_error']
                row[f'psych_{key}_std'] = data['std_error']
        
        # Curve error: MAE between true and fitted model psychometric curves
        row['psych_curve_mae'] = results['psych_recovery'][burn_in]['mean_curve_mae']
        
        # Fit quality: NLL per trial (how well BE model predicts choices)
        row['train_nll_mean'] = np.nanmean(results['fit_quality'][burn_in]['train_nll'])
        test_nll = results['fit_quality'][burn_in]['test_nll']
        row['test_nll_mean'] = np.nanmean(test_nll) if len(test_nll) > 0 else np.nan
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def fit_and_evaluate(
    stimuli: np.ndarray, 
    categories: np.ndarray,
    observed_choices: np.ndarray, 
    rewards: np.ndarray,
    no_response: np.ndarray, 
    not_blockstart: np.ndarray,
    burn_in: int = 1000,
    validation: str = 'holdout',
    n_simulations: int = 10,
    seed: int = 42
) -> Dict:
    """
    Convenience function to fit model and compute all diagnostics.
    
    Args:
        stimuli, categories, observed_choices, rewards: Trial data
        no_response, not_blockstart: Trial masks
        burn_in: Burn-in trials for expert initialisation
        validation: Validation method
        n_simulations: Simulations for diagnostic comparisons
        seed: Random seed
    
    Returns:
        Dict with fitted model, results, and all diagnostics
    """
    # Fit
    model, results = BoundaryEstimationModel.fit(
        stimuli, categories, observed_choices,
        burn_in=burn_in,
        validation=validation,
        seed=seed
    )
    
    # Psychometric comparison
    psych_comparison = model.compare_to_data(
        stimuli, categories, observed_choices,
        n_simulations=n_simulations,
        seed=seed
    )
    
    # Serial dependence comparison
    serial_comparison = model.compare_serial_dependence(
        stimuli, categories, observed_choices, rewards,
        no_response, not_blockstart,
        n_simulations=n_simulations,
        seed=seed
    )
    
    return {
        'model': model,
        'fit_results': results,
        'psychometric_comparison': psych_comparison,
        'serial_dependence_comparison': serial_comparison
    }


# =============================================================================
# BURN-IN SPECIFIC PLOTTING WRAPPERS
# =============================================================================

def plot_psychometric_by_burn_in(
    params: Dict[str, float],
    burn_in_values: List[int],
    n_trials: int = 300,
    n_bins: int = 8,
    seed: int = 42,
    figsize: Tuple[int, int] = (12, 4),
    show_gof: bool = True,
    show_params: bool = True,
    show_lapse: bool = False,
    n_bootstrap: int = 0,
    show_ci: bool = True
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Visualise psychometric curves for different burn-in levels.
    
    Shows how behaviour changes from naive (burn_in=0) to expert (burn_in=1000+).
    For each burn-in, simulates choices and plots:
    - Binned proportion choosing B (dots with error bars)
    - Fitted cumulative Gaussian (line)
    - Goodness-of-fit metrics and parameters
    
    Uses Plotting.psychometric.plot_psychometric for core plotting.
    
    Args:
        params: Model parameters
        burn_in_values: List of burn-in values to compare
        n_trials: Number of trials to simulate
        n_bins: Number of bins for plotting
        seed: Random seed
        figsize: Figure size
        show_gof: Whether to show GOF metrics (Acc, R², RMSE)
        show_params: Whether to show μ, σ parameters
        show_lapse: Whether to show lapse parameters (λ_low, λ_high)
        n_bootstrap: Number of bootstrap samples for CIs (0 = no bootstrap)
        show_ci: Whether to show CI band on curve
    
    Returns:
        fig: Matplotlib figure
        gof_df: DataFrame with all psychometric parameters and GOF metrics
    """
    n_conditions = len(burn_in_values)
    fig, axes = plt.subplots(1, n_conditions, figsize=figsize, sharey=True)
    
    if n_conditions == 1:
        axes = [axes]
    
    # Common stimulus set
    stimuli, categories, rng = generate_stimuli(n_trials = n_trials,
                                                x_min = -1,
                                                x_max = 1,
                                                seed = seed)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_conditions))
    
    # Storage for GOF metrics
    gof_records = []
    
    for i, (burn_in, ax, color) in enumerate(zip(burn_in_values, axes, colors)):
        # Create and simulate
        model = BoundaryEstimationModel(**params)
        model.reset_belief(burn_in=burn_in, burn_in_seed=seed + burn_in)
        
        sim_rng = np.random.default_rng(seed + i * 1000)
        choices, _ = model.simulate_session(stimuli, categories, rng=sim_rng)
        
        # Use core plotting function with all options
        _, info = plot_psychometric(
            stimuli, choices, ax=ax,
            n_bins=n_bins,
            show_fit=True,
            show_gof=show_gof,
            show_params=show_params,
            show_lapse=show_lapse,
            n_bootstrap=n_bootstrap,
            show_ci=show_ci,
            color=color,
            title=f'burn_in = {burn_in}',
            gof_position='upper left',
            seed=seed + i
        )
        
        # Store all metrics
        gof = info['gof'].copy()
        gof['burn_in'] = burn_in
        gof['mu'] = info['psych_params'].get('mu', np.nan)
        gof['sigma'] = info['psych_params'].get('sigma', np.nan)
        gof['lapse_low'] = info['psych_params'].get('lapse_low', np.nan)
        gof['lapse_high'] = info['psych_params'].get('lapse_high', np.nan)
        gof['accuracy'] = np.mean(choices == categories)
        
        # Add CIs if available
        if n_bootstrap > 0:
            for param in ['mu', 'sigma', 'lapse_low', 'lapse_high']:
                ci_key = f'{param}_ci'
                if ci_key in info['psych_params']:
                    ci = info['psych_params'][ci_key]
                    gof[f'{param}_ci_low'] = ci[0]
                    gof[f'{param}_ci_high'] = ci[1]
        
        gof_records.append(gof)
        
        # Only show y-label on first plot
        if i > 0:
            ax.set_ylabel('')
    
    plt.tight_layout()
    
    gof_df = pd.DataFrame(gof_records)
    # Reorder columns
    col_order = ['burn_in', 'accuracy', 'mu', 'sigma', 'lapse_low', 'lapse_high',
                 'r_squared', 'rmse', 'mae', 'deviance_explained', 
                 'log_likelihood', 'aic', 'bic', 'n_trials']
    # Add CI columns if present
    for param in ['mu', 'sigma', 'lapse_low', 'lapse_high']:
        col_order.extend([f'{param}_ci_low', f'{param}_ci_high'])
    gof_df = gof_df[[c for c in col_order if c in gof_df.columns]]
    
    return fig, gof_df


def plot_belief_after_burn_in(
    params: Dict[str, float],
    burn_in_values: List[int],
    seed: int = 42,
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """
    Visualise the boundary belief distribution after different burn-in amounts.
    
    Shows how the belief sharpens from uniform (burn_in=0) to peaked (burn_in=1000+).
    Uses Plotting.belief functions for core plotting.
    
    Args:
        params: Model parameters
        burn_in_values: List of burn-in values to compare
        seed: Random seed
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Collect beliefs and compute uncertainties
    beliefs = []
    uncertainties = []
    x = None
    
    for burn_in in burn_in_values:
        model = BoundaryEstimationModel(**params)
        model.reset_belief(burn_in=burn_in, burn_in_seed=seed + burn_in)
        
        beliefs.append(model.boundary_belief.copy())
        if x is None:
            x = model.x.copy()
        
        # Compute uncertainty (belief std)
        belief_norm = model.boundary_belief / np.sum(model.boundary_belief)
        belief_mean = np.sum(x * belief_norm)
        belief_var = np.sum((x - belief_mean)**2 * belief_norm)
        uncertainties.append(np.sqrt(belief_var))
    
    # Left: Belief distributions using core function
    labels = [str(b) for b in burn_in_values]
    plot_belief_distributions(
        x, beliefs, labels,
        ax=axes[0],
        true_boundary=0.0,
        title='Boundary belief after burn-in',
        legend_title='Burn-in'
    )
    
    # Right: Uncertainty vs burn-in using core function
    plot_belief_uncertainty(
        burn_in_values, uncertainties,
        ax=axes[1],
        title='Belief uncertainty vs burn-in',
        show_uniform_reference=True
    )
    
    plt.tight_layout()
    return fig


# =============================================================================
# RE-EXPORTS FOR CONVENIENCE
# =============================================================================

# Re-export plotting functions from Plotting module for backwards compatibility
__all__ = [
    # Analysis
    'burn_in_recovery_analysis',
    'burn_in_recovery_summary_stats',
    'fit_and_evaluate',
    # Burn-in specific plotting
    'plot_psychometric_by_burn_in',
    'plot_belief_after_burn_in',
    # Re-exported from Plotting
    'plot_burn_in_recovery',
    'plot_burn_in_param_distributions',
]
