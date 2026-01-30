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
from Models.agent import MixedAgent

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from Helpers.psychometry import fit_psychometric, compute_psychometric_gof, compute_psych_error
from Helpers.utils import generate_stimuli
from Plotting.psychometric import plot_psychometric
from Plotting.belief import plot_belief_distributions, plot_belief_uncertainty
from Plotting.recovery import (
    plot_burn_in_recovery, 
    plot_burn_in_param_distributions,
    plot_mixed_agent_recovery,
    plot_mixed_agent_param_distributions
)


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
        - Psychometric parameter recovery (Î¼, Ïƒ, Î»_low, Î»_high)
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
# MIXED AGENT RECOVERY ANALYSIS
# =============================================================================

def mixed_agent_recovery_analysis(
    true_be_params: Dict[str, float],
    alpha_values: List[float],
    heuristic_params: Optional[Dict[str, float]] = None,
    n_trials: int = 300,
    n_replicates: int = 10,
    agent_burn_in: int = 1000,
    fitter_burn_in: int = 0,
    validation: Optional[str] = 'holdout',
    validation_config: Optional[Dict] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Test how BE weight (α) affects parameter recovery when fitting BE model to MixedAgent data.
    
    This analysis addresses the question: Can we detect when an animal transitions from
    heuristic-dominated to BE-dominated behaviour? As α increases, we expect:
    - Better recovery of true BE parameters
    - Better psychometric fit quality (R², lower NLL)
    - Psychometric curves that look more 'sensible' (sigmoidal)
    
    For each α value:
    1. Generate data from MixedAgent(α, true_be_params, heuristic_params)
    2. Fit BE model to that data (assuming fitter_burn_in)
    3. Compare recovered params to true BE params
    4. Track behavioural/psychometric metrics
    
    Args:
        true_be_params: Dict of true BE parameter values used by MixedAgent
            Required keys: 'sigma_percep', 'A_repulsion', 'eta_learning', 'eta_relax'
        alpha_values: List of α values to test, e.g., [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            α=0 means pure heuristic, α=1 means pure BE
        heuristic_params: Dict of heuristic parameters for MixedAgent
            Keys: 'bias', 'p_winstay', 'p_loseshift', 'w_bias', 'w_winstay', 
                  'w_loseshift', 'w_random'
            If None, uses defaults (mild win-stay/lose-shift bias)
        n_trials: Number of trials per simulated session
        n_replicates: Number of replicates per α condition
        agent_burn_in: Burn-in for MixedAgent's internal BE model (experience level)
        fitter_burn_in: Burn-in assumed by BE fitter (default 0 = naive assumption)
        validation: Validation method for fitting ('holdout', 'cv', or None)
        validation_config: Validation configuration
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with:
            'param_recovery': {alpha: {param_name: {true, fitted, error, ...}}}
            'psych_metrics': {alpha: {accuracy, mu, sigma, r_squared, ...}}
            'fit_quality': {alpha: {train_nll, test_nll}}
            'behaviour': {alpha: {choices, p_B, ...}} (per replicate, for detailed analysis)
            'config': {experiment configuration}
    """
    # Set default heuristic params if not provided
    if heuristic_params is None:
        heuristic_params = {
            'bias': 0.05,           # Slight B bias
            'p_winstay': 0.6,       # Moderate win-stay
            'p_loseshift': 0.4,     # Moderate lose-shift
            'w_bias': 1.0,
            'w_winstay': 1.0,
            'w_loseshift': 1.0,
            'w_random': 0.5
        }
    
    param_names = BoundaryEstimationModel.get_param_names()
    
    # Storage
    results = {
        'param_recovery': {},
        'psych_metrics': {},
        'fit_quality': {},
        'behaviour': {},
        'config': {
            'true_be_params': true_be_params,
            'heuristic_params': heuristic_params,
            'alpha_values': alpha_values,
            'n_trials': n_trials,
            'n_replicates': n_replicates,
            'agent_burn_in': agent_burn_in,
            'fitter_burn_in': fitter_burn_in,
            'validation': validation,
            'seed': seed
        }
    }
    
    x_eval = np.linspace(-1, 1, 100)
    
    for alpha in alpha_values:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing α = {alpha:.2f}")
            print(f"{'='*60}")
        
        # Initialise storage for this α
        param_storage = {name: {'true': [], 'fitted': [], 'error': []} 
                        for name in param_names}
        psych_storage = {
            'accuracy': [],
            'mu': [],
            'sigma': [],
            'lapse_low': [],
            'lapse_high': [],
            'r_squared': [],
            'rmse': [],
            'deviance_explained': []
        }
        fit_storage = {'train_nll': [], 'test_nll': []}
        behaviour_storage = {
            'choices': [],
            'stimuli': [],
            'categories': [],
            'p_be': [],
            'p_heuristic': [],
            'p_mixed': []
        }
        
        for rep in range(n_replicates):
            if verbose:
                print(f"  Replicate {rep+1}/{n_replicates}...", end=' ')
            
            rep_seed = seed + int(alpha * 10000) + rep * 100
            
            # Generate stimuli for this replicate
            stimuli, categories, rep_rng = generate_stimuli(
                n_trials=n_trials,
                x_min=-1,
                x_max=1,
                seed=rep_seed
            )
            
            # --- Create MixedAgent and simulate ---
            agent = MixedAgent(
                # BE params
                sigma_percep=true_be_params['sigma_percep'],
                A_repulsion=true_be_params['A_repulsion'],
                eta_learning=true_be_params['eta_learning'],
                eta_relax=true_be_params['eta_relax'],
                # Mixture weight
                alpha=alpha,
                # Heuristic params
                **heuristic_params,
                # Initialisation
                burn_in=agent_burn_in,
                burn_in_seed=rep_seed
            )
            
            # Simulate with detailed output
            sim_rng = np.random.default_rng(rep_seed + 1)
            df_detailed = agent.simulate_session_detailed(stimuli, categories, rng=sim_rng)
            
            choices = df_detailed['choice'].values
            
            # Store behaviour for later analysis
            behaviour_storage['choices'].append(choices.copy())
            behaviour_storage['stimuli'].append(stimuli.copy())
            behaviour_storage['categories'].append(categories.copy())
            behaviour_storage['p_be'].append(df_detailed['p_be'].values.copy())
            behaviour_storage['p_heuristic'].append(df_detailed['p_heuristic'].values.copy())
            behaviour_storage['p_mixed'].append(df_detailed['p_mixed'].values.copy())
            
            # --- Compute psychometric metrics on MixedAgent's choices ---
            valid_mask = ~np.isnan(choices)
            if np.sum(valid_mask) > 10:
                psych_fit = fit_psychometric(stimuli[valid_mask], choices[valid_mask], x_eval)
                gof = compute_psychometric_gof(stimuli[valid_mask], choices[valid_mask], psych_fit)
                
                psych_storage['accuracy'].append(np.mean(choices[valid_mask] == categories[valid_mask]))
                psych_storage['mu'].append(psych_fit['mu'])
                psych_storage['sigma'].append(psych_fit['sigma'])
                psych_storage['lapse_low'].append(psych_fit.get('lapse_low', 0))
                psych_storage['lapse_high'].append(psych_fit.get('lapse_high', 0))
                psych_storage['r_squared'].append(gof.get('r_squared', np.nan))
                psych_storage['rmse'].append(gof.get('rmse', np.nan))
                psych_storage['deviance_explained'].append(gof.get('deviance_explained', np.nan))
            else:
                for key in psych_storage:
                    psych_storage[key].append(np.nan)
            
            # --- Fit BE model to MixedAgent's choices ---
            try:
                fitted_model, fit_results = BoundaryEstimationModel.fit(
                    stimuli, categories, choices,
                    burn_in=fitter_burn_in,
                    burn_in_seed=rep_seed,
                    validation=validation,
                    validation_config=validation_config,
                    n_restarts=5,
                    seed=rep_seed + 2
                )
                
                # Store parameter results
                for name in param_names:
                    param_storage[name]['true'].append(true_be_params[name])
                    param_storage[name]['fitted'].append(fit_results['params'][name])
                    param_storage[name]['error'].append(
                        fit_results['params'][name] - true_be_params[name]
                    )
                
                # Store fit quality
                fit_storage['train_nll'].append(fit_results.get('train_nll_per_trial', np.nan))
                if 'test_nll_per_trial' in fit_results:
                    fit_storage['test_nll'].append(fit_results['test_nll_per_trial'])
                
                if verbose:
                    print("OK")
                    
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")
                
                # Fill with NaN
                for name in param_names:
                    param_storage[name]['true'].append(true_be_params[name])
                    param_storage[name]['fitted'].append(np.nan)
                    param_storage[name]['error'].append(np.nan)
                
                fit_storage['train_nll'].append(np.nan)
        
        # Convert to arrays and compute summary statistics
        for name in param_names:
            for key in ['true', 'fitted', 'error']:
                param_storage[name][key] = np.array(param_storage[name][key])
            
            # Summary stats
            errors = param_storage[name]['error']
            fitted = param_storage[name]['fitted']
            true_vals = param_storage[name]['true']
            
            param_storage[name]['mean_error'] = np.nanmean(errors)
            param_storage[name]['std_error'] = np.nanstd(errors)
            param_storage[name]['abs_mean_error'] = np.nanmean(np.abs(errors))
            param_storage[name]['fitted_mean'] = np.nanmean(fitted)
            param_storage[name]['fitted_std'] = np.nanstd(fitted)
            
            # Correlation (if enough valid values)
            valid = ~np.isnan(fitted) & ~np.isnan(true_vals)
            if np.sum(valid) >= 3:
                # For recovery across replicates with same true params, 
                # correlation isn't meaningful. Instead track recovery accuracy.
                param_storage[name]['recovery_fraction'] = np.sum(valid) / len(fitted)
            else:
                param_storage[name]['recovery_fraction'] = 0.0
        
        # Convert psychometric storage to arrays and compute summary
        for key in psych_storage:
            psych_storage[key] = np.array(psych_storage[key])
        
        psych_storage['accuracy_mean'] = np.nanmean(psych_storage['accuracy'])
        psych_storage['accuracy_std'] = np.nanstd(psych_storage['accuracy'])
        psych_storage['mu_mean'] = np.nanmean(psych_storage['mu'])
        psych_storage['mu_std'] = np.nanstd(psych_storage['mu'])
        psych_storage['sigma_mean'] = np.nanmean(psych_storage['sigma'])
        psych_storage['sigma_std'] = np.nanstd(psych_storage['sigma'])
        psych_storage['r_squared_mean'] = np.nanmean(psych_storage['r_squared'])
        psych_storage['r_squared_std'] = np.nanstd(psych_storage['r_squared'])
        
        # Fit quality
        for key in ['train_nll', 'test_nll']:
            fit_storage[key] = np.array(fit_storage[key]) if fit_storage[key] else np.array([])
        fit_storage['train_nll_mean'] = np.nanmean(fit_storage['train_nll'])
        fit_storage['train_nll_std'] = np.nanstd(fit_storage['train_nll'])
        if len(fit_storage['test_nll']) > 0:
            fit_storage['test_nll_mean'] = np.nanmean(fit_storage['test_nll'])
            fit_storage['test_nll_std'] = np.nanstd(fit_storage['test_nll'])
        
        # Store
        results['param_recovery'][alpha] = param_storage
        results['psych_metrics'][alpha] = psych_storage
        results['fit_quality'][alpha] = fit_storage
        results['behaviour'][alpha] = behaviour_storage
    
    return results


def mixed_agent_recovery_summary_stats(results: Dict) -> pd.DataFrame:
    """
    Generate summary statistics table for mixed agent recovery analysis.
    
    Args:
        results: Output from mixed_agent_recovery_analysis()
    
    Returns:
        DataFrame with summary statistics for each α condition, including:
        - BE model parameter recovery (bias, std for each)
        - Psychometric metrics (accuracy, μ, σ, R²)
        - Fit quality (NLL per trial)
    """
    alpha_values = results['config']['alpha_values']
    true_be_params = results['config']['true_be_params']
    param_names = list(true_be_params.keys())
    
    rows = []
    
    for alpha in alpha_values:
        row = {'alpha': alpha}
        
        # BE model parameter recovery
        for name in param_names:
            data = results['param_recovery'][alpha][name]
            row[f'{name}_true'] = true_be_params[name]
            row[f'{name}_fitted'] = data['fitted_mean']
            row[f'{name}_fitted_std'] = data['fitted_std']
            row[f'{name}_bias'] = data['mean_error']
            row[f'{name}_abs_error'] = data['abs_mean_error']
        
        # Psychometric metrics
        psych = results['psych_metrics'][alpha]
        row['accuracy_mean'] = psych['accuracy_mean']
        row['accuracy_std'] = psych['accuracy_std']
        row['mu_mean'] = psych['mu_mean']
        row['mu_std'] = psych['mu_std']
        row['sigma_mean'] = psych['sigma_mean']
        row['sigma_std'] = psych['sigma_std']
        row['r_squared_mean'] = psych['r_squared_mean']
        row['r_squared_std'] = psych['r_squared_std']
        
        # Fit quality
        fit = results['fit_quality'][alpha]
        row['train_nll_mean'] = fit.get('train_nll_mean', np.nan)
        row['train_nll_std'] = fit.get('train_nll_std', np.nan)
        row['test_nll_mean'] = fit.get('test_nll_mean', np.nan)
        row['test_nll_std'] = fit.get('test_nll_std', np.nan)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def mixed_agent_parameter_sweep(
    base_be_params: Dict[str, float],
    sweep_param: str,
    sweep_values: List[float],
    alpha_values: List[float] = [0.0, 0.5, 1.0],
    heuristic_params: Optional[Dict[str, float]] = None,
    n_trials: int = 300,
    n_replicates: int = 10,
    agent_burn_in: int = 1000,
    fitter_burn_in: int = 0,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Sweep a single parameter while testing recovery across α values.
    
    This allows testing how recovery depends on the true parameter value.
    For example: Is sigma_percep recovery better when sigma_percep is larger?
    
    Args:
        base_be_params: Base BE parameters (one will be overwritten by sweep)
        sweep_param: Name of parameter to sweep ('sigma_percep', 'A_repulsion', 
                     'eta_learning', 'eta_relax', or heuristic params)
        sweep_values: Values to test for the swept parameter
        alpha_values: α values to test for each sweep value
        heuristic_params: Heuristic parameters (or None for defaults)
        n_trials: Trials per session
        n_replicates: Replicates per condition
        agent_burn_in: MixedAgent burn-in
        fitter_burn_in: Fitter burn-in assumption
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with results organised by {sweep_value: {alpha: recovery_data}}
    """
    be_param_names = BoundaryEstimationModel.get_param_names()
    is_be_param = sweep_param in be_param_names
    
    if heuristic_params is None:
        heuristic_params = {
            'bias': 0.05,
            'p_winstay': 0.6,
            'p_loseshift': 0.4,
            'w_bias': 1.0,
            'w_winstay': 1.0,
            'w_loseshift': 1.0,
            'w_random': 0.5
        }
    
    results = {
        'sweep_results': {},
        'config': {
            'base_be_params': base_be_params,
            'sweep_param': sweep_param,
            'sweep_values': sweep_values,
            'alpha_values': alpha_values,
            'heuristic_params': heuristic_params,
            'n_trials': n_trials,
            'n_replicates': n_replicates,
            'agent_burn_in': agent_burn_in,
            'fitter_burn_in': fitter_burn_in,
            'seed': seed
        }
    }
    
    for i, sweep_val in enumerate(sweep_values):
        if verbose:
            print(f"\n{'#'*70}")
            print(f"Sweep {i+1}/{len(sweep_values)}: {sweep_param} = {sweep_val}")
            print(f"{'#'*70}")
        
        # Create params for this sweep value
        if is_be_param:
            current_be_params = base_be_params.copy()
            current_be_params[sweep_param] = sweep_val
            current_heuristic_params = heuristic_params.copy()
        else:
            current_be_params = base_be_params.copy()
            current_heuristic_params = heuristic_params.copy()
            current_heuristic_params[sweep_param] = sweep_val
        
        # Run recovery analysis for this sweep value
        sweep_seed = seed + i * 100000
        sweep_result = mixed_agent_recovery_analysis(
            true_be_params=current_be_params,
            alpha_values=alpha_values,
            heuristic_params=current_heuristic_params,
            n_trials=n_trials,
            n_replicates=n_replicates,
            agent_burn_in=agent_burn_in,
            fitter_burn_in=fitter_burn_in,
            validation='holdout',
            seed=sweep_seed,
            verbose=verbose
        )
        
        results['sweep_results'][sweep_val] = sweep_result
    
    return results


def mixed_agent_sweep_summary(results: Dict) -> pd.DataFrame:
    """
    Generate summary table for parameter sweep analysis.
    
    Args:
        results: Output from mixed_agent_parameter_sweep()
    
    Returns:
        DataFrame with rows for each (sweep_value, alpha) combination
    """
    sweep_param = results['config']['sweep_param']
    sweep_values = results['config']['sweep_values']
    alpha_values = results['config']['alpha_values']
    
    rows = []
    
    for sweep_val in sweep_values:
        sweep_result = results['sweep_results'][sweep_val]
        
        for alpha in alpha_values:
            row = {
                sweep_param: sweep_val,
                'alpha': alpha
            }
            
            # Add recovery stats for each BE param
            for param_name in BoundaryEstimationModel.get_param_names():
                data = sweep_result['param_recovery'][alpha][param_name]
                row[f'{param_name}_bias'] = data['mean_error']
                row[f'{param_name}_abs_error'] = data['abs_mean_error']
            
            # Add psychometric metrics
            psych = sweep_result['psych_metrics'][alpha]
            row['accuracy'] = psych['accuracy_mean']
            row['r_squared'] = psych['r_squared_mean']
            row['sigma_psych'] = psych['sigma_mean']
            
            # Add fit quality
            row['train_nll'] = sweep_result['fit_quality'][alpha].get('train_nll_mean', np.nan)
            
            rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# BE PARAMETER EFFECTS ON BEHAVIOUR
# =============================================================================

def be_param_behaviour_sweep(
    base_params: Dict[str, float],
    sweep_param: str,
    sweep_values: List[float],
    burn_in_values: List[int] = [0, 500, 2000],
    n_trials: int = 300,
    n_replicates: int = 10,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Analyse how a BE parameter affects behaviour generation.
    
    For each sweep value × burn_in combination, simulates sessions and computes:
    - Psychometric parameters (μ, σ, lapses)
    - Accuracy and performance vs chance
    - R², deviance explained
    - Psychometric curves for plotting
    - Confidence intervals via bootstrap across replicates
    
    Args:
        base_params: Base BE parameters (one will be swept)
        sweep_param: Parameter to sweep ('sigma_percep', 'A_repulsion', 
                     'eta_learning', 'eta_relax')
        sweep_values: Values to test
        burn_in_values: Burn-in levels to test
        n_trials: Trials per session
        n_replicates: Replicates per condition
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with:
            'behaviour': {sweep_val: {burn_in: {metrics, psych_curves, ...}}}
            'config': experiment configuration
    """
    param_names = BoundaryEstimationModel.get_param_names()
    if sweep_param not in param_names:
        raise ValueError(f"sweep_param must be one of {param_names}")
    
    x_eval = np.linspace(-1, 1, 100)
    
    results = {
        'behaviour': {},
        'x_eval': x_eval,  # Store for plotting
        'config': {
            'base_params': base_params,
            'sweep_param': sweep_param,
            'sweep_values': sweep_values,
            'burn_in_values': burn_in_values,
            'n_trials': n_trials,
            'n_replicates': n_replicates,
            'seed': seed
        }
    }
    
    for sweep_val in sweep_values:
        if verbose:
            print(f"\n{sweep_param} = {sweep_val}")
        
        results['behaviour'][sweep_val] = {}
        
        # Create params for this sweep value
        current_params = base_params.copy()
        current_params[sweep_param] = sweep_val
        
        for burn_in in burn_in_values:
            if verbose:
                print(f"  burn_in = {burn_in}...", end=' ')
            
            # Storage for this condition
            metrics = {
                'accuracy': [],
                'mu': [],
                'sigma': [],
                'lapse_low': [],
                'lapse_high': [],
                'r_squared': [],
                'deviance_explained': [],
                'rmse': [],
                'choices': [],  # Store for detailed analysis
                'stimuli': [],  # Store stimuli for plotting
                'psych_curves': [],  # Store fitted curves for plotting
            }
            
            for rep in range(n_replicates):
                rep_seed = seed + int(sweep_val * 1000) + burn_in + rep
                
                # Generate stimuli
                stimuli, categories, rng = generate_stimuli(
                    n_trials=n_trials, seed=rep_seed
                )
                
                # Create and simulate
                model = BoundaryEstimationModel(**current_params)
                model.reset_belief(burn_in=burn_in, burn_in_seed=rep_seed)
                
                sim_rng = np.random.default_rng(rep_seed + 1)
                choices, _ = model.simulate_session(stimuli, categories, rng=sim_rng)
                
                # Compute metrics
                valid = ~np.isnan(choices)
                acc = np.mean(choices[valid] == categories[valid])
                metrics['accuracy'].append(acc)
                metrics['choices'].append(choices.copy())
                metrics['stimuli'].append(stimuli.copy())
                
                # Fit psychometric
                psych = fit_psychometric(stimuli[valid], choices[valid], x_eval)
                
                if psych.get('success', False):
                    metrics['mu'].append(psych['mu'])
                    metrics['sigma'].append(psych['sigma'])
                    metrics['lapse_low'].append(psych['lapse_low'])
                    metrics['lapse_high'].append(psych['lapse_high'])
                    metrics['psych_curves'].append(psych['y_fit'])
                    
                    # GOF metrics
                    gof = compute_psychometric_gof(stimuli[valid], choices[valid], psych)
                    metrics['r_squared'].append(gof['r_squared'])
                    metrics['deviance_explained'].append(gof['deviance_explained'])
                    metrics['rmse'].append(gof['rmse'])
                else:
                    for key in ['mu', 'sigma', 'lapse_low', 'lapse_high', 
                               'r_squared', 'deviance_explained', 'rmse']:
                        metrics[key].append(np.nan)
                    metrics['psych_curves'].append(None)
            
            # Convert to arrays and compute summary stats
            for key in ['accuracy', 'mu', 'sigma', 'lapse_low', 'lapse_high',
                       'r_squared', 'deviance_explained', 'rmse']:
                metrics[key] = np.array(metrics[key])
            
            # Compute mean psychometric curve
            valid_curves = [c for c in metrics['psych_curves'] if c is not None]
            if valid_curves:
                metrics['psych_curve_mean'] = np.nanmean(valid_curves, axis=0)
                metrics['psych_curve_std'] = np.nanstd(valid_curves, axis=0)
            else:
                metrics['psych_curve_mean'] = None
                metrics['psych_curve_std'] = None
            
            # Summary statistics
            metrics['accuracy_mean'] = np.nanmean(metrics['accuracy'])
            metrics['accuracy_std'] = np.nanstd(metrics['accuracy'])
            metrics['accuracy_ci'] = _bootstrap_ci(metrics['accuracy'])
            
            # Test vs chance (50%)
            metrics['accuracy_vs_chance_t'], metrics['accuracy_vs_chance_p'] = \
                _one_sample_ttest(metrics['accuracy'], 0.5)
            
            for key in ['mu', 'sigma', 'lapse_low', 'lapse_high', 
                       'r_squared', 'deviance_explained', 'rmse']:
                metrics[f'{key}_mean'] = np.nanmean(metrics[key])
                metrics[f'{key}_std'] = np.nanstd(metrics[key])
                metrics[f'{key}_ci'] = _bootstrap_ci(metrics[key])
            
            results['behaviour'][sweep_val][burn_in] = metrics
            
            if verbose:
                print(f"Acc={metrics['accuracy_mean']:.3f}±{metrics['accuracy_std']:.3f}, "
                      f"R²={metrics['r_squared_mean']:.3f}")
    
    return results


def be_param_recovery_sweep(
    base_params: Dict[str, float],
    sweep_param: str,
    sweep_values: List[float],
    burn_in_values: List[int] = [0, 500, 2000],
    n_trials: int = 300,
    n_replicates: int = 10,
    fitter_burn_in: int = 0,
    validation: Optional[str] = 'holdout',
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Analyse how true BE parameter values affect parameter recovery.
    
    For each sweep value × burn_in, simulates from BE model and fits BE model back.
    Tests whether some true parameter values are easier to recover than others.
    
    Args:
        base_params: Base BE parameters
        sweep_param: Parameter to sweep
        sweep_values: Values to test
        burn_in_values: Burn-in levels for data generation
        n_trials: Trials per session
        n_replicates: Replicates per condition
        fitter_burn_in: Burn-in assumed by fitter
        validation: Validation method
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with recovery results organised by sweep_val and burn_in
    """
    param_names = BoundaryEstimationModel.get_param_names()
    
    results = {
        'recovery': {},
        'config': {
            'base_params': base_params,
            'sweep_param': sweep_param,
            'sweep_values': sweep_values,
            'burn_in_values': burn_in_values,
            'n_trials': n_trials,
            'n_replicates': n_replicates,
            'fitter_burn_in': fitter_burn_in,
            'seed': seed
        }
    }
    
    for sweep_val in sweep_values:
        if verbose:
            print(f"\n{'='*50}")
            print(f"{sweep_param} = {sweep_val}")
            print(f"{'='*50}")
        
        results['recovery'][sweep_val] = {}
        
        # Create params for this sweep value
        current_params = base_params.copy()
        current_params[sweep_param] = sweep_val
        
        for burn_in in burn_in_values:
            if verbose:
                print(f"  burn_in = {burn_in}:")
            
            # Storage
            param_storage = {name: {'true': [], 'fitted': [], 'error': []} 
                           for name in param_names}
            fit_storage = {'train_nll': [], 'test_nll': []}
            
            for rep in range(n_replicates):
                if verbose:
                    print(f"    Rep {rep+1}/{n_replicates}...", end=' ')
                
                rep_seed = seed + int(sweep_val * 1000) + burn_in * 10 + rep
                
                # Generate and simulate
                stimuli, categories, rng = generate_stimuli(
                    n_trials=n_trials, seed=rep_seed
                )
                
                model = BoundaryEstimationModel(**current_params)
                model.reset_belief(burn_in=burn_in, burn_in_seed=rep_seed)
                
                sim_rng = np.random.default_rng(rep_seed + 1)
                choices, _ = model.simulate_session(stimuli, categories, rng=sim_rng)
                
                # Fit
                try:
                    fitted_model, fit_results = BoundaryEstimationModel.fit(
                        stimuli, categories, choices,
                        burn_in=fitter_burn_in,
                        burn_in_seed=rep_seed,
                        validation=validation,
                        n_restarts=5,
                        seed=rep_seed + 2
                    )
                    
                    for name in param_names:
                        param_storage[name]['true'].append(current_params[name])
                        param_storage[name]['fitted'].append(fit_results['params'][name])
                        param_storage[name]['error'].append(
                            fit_results['params'][name] - current_params[name]
                        )
                    
                    fit_storage['train_nll'].append(
                        fit_results.get('train_nll_per_trial', np.nan)
                    )
                    if 'test_nll_per_trial' in fit_results:
                        fit_storage['test_nll'].append(fit_results['test_nll_per_trial'])
                    
                    if verbose:
                        print("OK")
                        
                except Exception as e:
                    if verbose:
                        print(f"FAILED: {e}")
                    for name in param_names:
                        param_storage[name]['true'].append(current_params[name])
                        param_storage[name]['fitted'].append(np.nan)
                        param_storage[name]['error'].append(np.nan)
                    fit_storage['train_nll'].append(np.nan)
            
            # Compute summary stats
            for name in param_names:
                for key in ['true', 'fitted', 'error']:
                    param_storage[name][key] = np.array(param_storage[name][key])
                
                errors = param_storage[name]['error']
                param_storage[name]['bias'] = np.nanmean(errors)
                param_storage[name]['abs_error'] = np.nanmean(np.abs(errors))
                param_storage[name]['std'] = np.nanstd(errors)
                param_storage[name]['bias_ci'] = _bootstrap_ci(errors)
            
            for key in ['train_nll', 'test_nll']:
                fit_storage[key] = np.array(fit_storage[key]) if fit_storage[key] else np.array([])
            
            results['recovery'][sweep_val][burn_in] = {
                'params': param_storage,
                'fit': fit_storage,
                'true_params': current_params.copy()
            }
    
    return results


def be_param_sweep_summary(results: Dict, analysis_type: str = 'behaviour') -> pd.DataFrame:
    """
    Generate summary DataFrame from parameter sweep results.
    
    Args:
        results: Output from be_param_behaviour_sweep or be_param_recovery_sweep
        analysis_type: 'behaviour' or 'recovery'
    
    Returns:
        DataFrame with one row per (sweep_value, burn_in) combination
    """
    sweep_param = results['config']['sweep_param']
    sweep_values = results['config']['sweep_values']
    burn_in_values = results['config']['burn_in_values']
    
    rows = []
    
    if analysis_type == 'behaviour':
        for sweep_val in sweep_values:
            for burn_in in burn_in_values:
                metrics = results['behaviour'][sweep_val][burn_in]
                row = {
                    sweep_param: sweep_val,
                    'burn_in': burn_in,
                    'accuracy_mean': metrics['accuracy_mean'],
                    'accuracy_std': metrics['accuracy_std'],
                    'accuracy_p_vs_chance': metrics['accuracy_vs_chance_p'],
                    'mu_mean': metrics['mu_mean'],
                    'mu_std': metrics['mu_std'],
                    'sigma_mean': metrics['sigma_mean'],
                    'sigma_std': metrics['sigma_std'],
                    'r_squared_mean': metrics['r_squared_mean'],
                    'r_squared_std': metrics['r_squared_std'],
                    'deviance_explained_mean': metrics['deviance_explained_mean'],
                }
                rows.append(row)
    
    elif analysis_type == 'recovery':
        param_names = BoundaryEstimationModel.get_param_names()
        for sweep_val in sweep_values:
            for burn_in in burn_in_values:
                data = results['recovery'][sweep_val][burn_in]
                row = {
                    sweep_param: sweep_val,
                    'burn_in': burn_in,
                }
                for name in param_names:
                    row[f'{name}_true'] = data['true_params'][name]
                    row[f'{name}_bias'] = data['params'][name]['bias']
                    row[f'{name}_abs_error'] = data['params'][name]['abs_error']
                    row[f'{name}_std'] = data['params'][name]['std']
                
                row['train_nll_mean'] = np.nanmean(data['fit']['train_nll'])
                rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# MIXED AGENT: BE PARAM EFFECTS
# =============================================================================

def mixed_agent_be_param_sweep(
    base_be_params: Dict[str, float],
    sweep_param: str,
    sweep_values: List[float],
    alpha_values: List[float] = [0.0, 0.5, 1.0],
    heuristic_params: Optional[Dict[str, float]] = None,
    burn_in_values: List[int] = [500, 2000],
    n_trials: int = 300,
    n_replicates: int = 10,
    fitter_burn_in: int = 0,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Sweep a BE parameter and examine effects on MixedAgent behaviour AND recovery.
    
    Combines behaviour analysis and recovery analysis in one sweep.
    
    Args:
        base_be_params: Base BE parameters
        sweep_param: BE parameter to sweep
        sweep_values: Values to test
        alpha_values: α values to test
        heuristic_params: Heuristic parameters (or None for defaults)
        burn_in_values: Burn-in values for MixedAgent
        n_trials: Trials per session
        n_replicates: Replicates per condition
        fitter_burn_in: Fitter burn-in assumption
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with behaviour and recovery results
    """
    if heuristic_params is None:
        heuristic_params = {
            'bias': 0.05, 'p_winstay': 0.6, 'p_loseshift': 0.4,
            'w_bias': 1.0, 'w_winstay': 1.0, 'w_loseshift': 1.0, 'w_random': 0.5
        }
    
    param_names = BoundaryEstimationModel.get_param_names()
    x_eval = np.linspace(-1, 1, 100)
    
    results = {
        'data': {},  # {sweep_val: {burn_in: {alpha: {metrics}}}}
        'config': {
            'base_be_params': base_be_params,
            'sweep_param': sweep_param,
            'sweep_values': sweep_values,
            'alpha_values': alpha_values,
            'heuristic_params': heuristic_params,
            'burn_in_values': burn_in_values,
            'n_trials': n_trials,
            'n_replicates': n_replicates,
            'fitter_burn_in': fitter_burn_in,
            'seed': seed
        }
    }
    
    for sweep_val in sweep_values:
        if verbose:
            print(f"\n{'#'*60}")
            print(f"{sweep_param} = {sweep_val}")
            print(f"{'#'*60}")
        
        results['data'][sweep_val] = {}
        
        current_be_params = base_be_params.copy()
        current_be_params[sweep_param] = sweep_val
        
        for burn_in in burn_in_values:
            if verbose:
                print(f"\n  burn_in = {burn_in}")
            
            results['data'][sweep_val][burn_in] = {}
            
            for alpha in alpha_values:
                if verbose:
                    print(f"    α = {alpha:.1f}...", end=' ')
                
                # Storage
                metrics = {
                    'accuracy': [], 'mu': [], 'sigma': [],
                    'r_squared': [], 'deviance_explained': [],
                }
                recovery = {name: {'fitted': [], 'error': []} for name in param_names}
                fit_quality = {'train_nll': [], 'test_nll': []}
                
                for rep in range(n_replicates):
                    rep_seed = (seed + int(sweep_val * 1000) + 
                               burn_in * 10 + int(alpha * 100) + rep)
                    
                    # Generate stimuli
                    stimuli, categories, rng = generate_stimuli(
                        n_trials=n_trials, seed=rep_seed
                    )
                    
                    # Create MixedAgent and simulate
                    agent = MixedAgent(
                        **current_be_params,
                        alpha=alpha,
                        **heuristic_params,
                        burn_in=burn_in,
                        burn_in_seed=rep_seed
                    )
                    
                    sim_rng = np.random.default_rng(rep_seed + 1)
                    choices, _ = agent.simulate_session(stimuli, categories, rng=sim_rng)
                    
                    # Behaviour metrics
                    valid = ~np.isnan(choices)
                    metrics['accuracy'].append(np.mean(choices[valid] == categories[valid]))
                    
                    psych = fit_psychometric(stimuli[valid], choices[valid], x_eval)
                    if psych.get('success', False):
                        metrics['mu'].append(psych['mu'])
                        metrics['sigma'].append(psych['sigma'])
                        gof = compute_psychometric_gof(stimuli[valid], choices[valid], psych)
                        metrics['r_squared'].append(gof['r_squared'])
                        metrics['deviance_explained'].append(gof['deviance_explained'])
                    else:
                        for k in ['mu', 'sigma', 'r_squared', 'deviance_explained']:
                            metrics[k].append(np.nan)
                    
                    # Recovery (fit BE model)
                    try:
                        fitted_model, fit_results = BoundaryEstimationModel.fit(
                            stimuli, categories, choices,
                            burn_in=fitter_burn_in,
                            validation='holdout',
                            n_restarts=5,
                            seed=rep_seed + 2
                        )
                        
                        for name in param_names:
                            fitted_val = fit_results['params'][name]
                            recovery[name]['fitted'].append(fitted_val)
                            recovery[name]['error'].append(
                                fitted_val - current_be_params[name]
                            )
                        
                        fit_quality['train_nll'].append(
                            fit_results.get('train_nll_per_trial', np.nan)
                        )
                        if 'test_nll_per_trial' in fit_results:
                            fit_quality['test_nll'].append(
                                fit_results['test_nll_per_trial']
                            )
                    except:
                        for name in param_names:
                            recovery[name]['fitted'].append(np.nan)
                            recovery[name]['error'].append(np.nan)
                        fit_quality['train_nll'].append(np.nan)
                
                # Summarise - fix: iterate over copy of keys to avoid RuntimeError
                metric_keys = list(metrics.keys())
                for k in metric_keys:
                    metrics[k] = np.array(metrics[k])
                    metrics[f'{k}_mean'] = np.nanmean(metrics[k])
                    metrics[f'{k}_std'] = np.nanstd(metrics[k])
                
                metrics['accuracy_vs_chance_p'] = _one_sample_ttest(
                    metrics['accuracy'], 0.5
                )[1]
                
                for name in param_names:
                    for k in ['fitted', 'error']:
                        recovery[name][k] = np.array(recovery[name][k])
                    recovery[name]['bias'] = np.nanmean(recovery[name]['error'])
                    recovery[name]['abs_error'] = np.nanmean(np.abs(recovery[name]['error']))
                
                # fix: iterate over copy of keys
                fit_keys = list(fit_quality.keys())
                for k in fit_keys:
                    fit_quality[k] = np.array(fit_quality[k]) if fit_quality[k] else np.array([])
                fit_quality['train_nll_mean'] = np.nanmean(fit_quality['train_nll'])
                
                results['data'][sweep_val][burn_in][alpha] = {
                    'behaviour': metrics,
                    'recovery': recovery,
                    'fit_quality': fit_quality,
                    'true_params': current_be_params.copy()
                }
                
                if verbose:
                    print(f"Acc={metrics['accuracy_mean']:.3f}, "
                          f"R²={metrics['r_squared_mean']:.3f}")
    
    return results


def mixed_agent_be_param_sweep_summary(results: Dict) -> pd.DataFrame:
    """
    Generate summary DataFrame from mixed agent BE param sweep.
    
    Args:
        results: Output from mixed_agent_be_param_sweep
    
    Returns:
        DataFrame with comprehensive summary
    """
    sweep_param = results['config']['sweep_param']
    sweep_values = results['config']['sweep_values']
    burn_in_values = results['config']['burn_in_values']
    alpha_values = results['config']['alpha_values']
    param_names = BoundaryEstimationModel.get_param_names()
    
    rows = []
    
    for sweep_val in sweep_values:
        for burn_in in burn_in_values:
            for alpha in alpha_values:
                data = results['data'][sweep_val][burn_in][alpha]
                
                row = {
                    sweep_param: sweep_val,
                    'burn_in': burn_in,
                    'alpha': alpha,
                    # Behaviour
                    'accuracy': data['behaviour']['accuracy_mean'],
                    'accuracy_std': data['behaviour']['accuracy_std'],
                    'accuracy_p': data['behaviour']['accuracy_vs_chance_p'],
                    'r_squared': data['behaviour']['r_squared_mean'],
                    'sigma_psych': data['behaviour']['sigma_mean'],
                    # Recovery
                    'train_nll': data['fit_quality']['train_nll_mean'],
                }
                
                for name in param_names:
                    row[f'{name}_bias'] = data['recovery'][name]['bias']
                    row[f'{name}_abs_err'] = data['recovery'][name]['abs_error']
                
                rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, 
                  ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    data = data[~np.isnan(data)]
    if len(data) < 3:
        return (np.nan, np.nan)
    
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    
    alpha = (1 - ci) / 2
    return (np.percentile(boot_means, alpha * 100), 
            np.percentile(boot_means, (1 - alpha) * 100))


def _one_sample_ttest(data: np.ndarray, null_value: float = 0.0
                     ) -> Tuple[float, float]:
    """One-sample t-test against a null value."""
    data = data[~np.isnan(data)]
    if len(data) < 3:
        return (np.nan, np.nan)
    
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(data, null_value)
    return (t_stat, p_value)


# =============================================================================
# JOINT PARAMETER EFFECTS (2D SWEEPS)
# =============================================================================

def be_param_joint_sweep(
    base_params: Dict[str, float],
    param1: str,
    param1_values: List[float],
    param2: str,
    param2_values: List[float],
    burn_in: int = 1000,
    n_trials: int = 300,
    n_replicates: int = 5,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Analyse joint effects of two BE parameters on behaviour.
    
    Creates a 2D grid of param1 × param2 combinations and measures
    behaviour at each point. Useful for detecting interactions.
    
    Args:
        base_params: Base BE parameters
        param1: First parameter to sweep
        param1_values: Values for param1
        param2: Second parameter to sweep
        param2_values: Values for param2
        burn_in: Burn-in level (fixed)
        n_trials: Trials per session
        n_replicates: Replicates per condition
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with:
            'grid': {(p1_val, p2_val): {metrics}}
            'matrices': {metric_name: 2D array}
            'config': experiment configuration
    """
    param_names = BoundaryEstimationModel.get_param_names()
    if param1 not in param_names or param2 not in param_names:
        raise ValueError(f"Parameters must be from {param_names}")
    
    x_eval = np.linspace(-1, 1, 100)
    
    results = {
        'grid': {},
        'matrices': {},
        'x_eval': x_eval,
        'config': {
            'base_params': base_params,
            'param1': param1,
            'param1_values': param1_values,
            'param2': param2,
            'param2_values': param2_values,
            'burn_in': burn_in,
            'n_trials': n_trials,
            'n_replicates': n_replicates,
            'seed': seed
        }
    }
    
    # Initialise matrices
    n1, n2 = len(param1_values), len(param2_values)
    for metric in ['accuracy', 'r_squared', 'sigma', 'mu', 'deviance_explained']:
        results['matrices'][metric] = np.zeros((n1, n2))
        results['matrices'][f'{metric}_std'] = np.zeros((n1, n2))
    
    total = n1 * n2
    count = 0
    
    for i, p1_val in enumerate(param1_values):
        for j, p2_val in enumerate(param2_values):
            count += 1
            if verbose:
                print(f"\r  [{count}/{total}] {param1}={p1_val:.3f}, {param2}={p2_val:.3f}", end='')
            
            # Create params
            current_params = base_params.copy()
            current_params[param1] = p1_val
            current_params[param2] = p2_val
            
            # Storage
            metrics = {
                'accuracy': [], 'r_squared': [], 'sigma': [], 
                'mu': [], 'deviance_explained': [], 'psych_curves': []
            }
            
            for rep in range(n_replicates):
                rep_seed = seed + i * 1000 + j * 100 + rep
                
                stimuli, categories, rng = generate_stimuli(
                    n_trials=n_trials, seed=rep_seed
                )
                
                model = BoundaryEstimationModel(**current_params)
                model.reset_belief(burn_in=burn_in, burn_in_seed=rep_seed)
                
                sim_rng = np.random.default_rng(rep_seed + 1)
                choices, _ = model.simulate_session(stimuli, categories, rng=sim_rng)
                
                valid = ~np.isnan(choices)
                metrics['accuracy'].append(np.mean(choices[valid] == categories[valid]))
                
                psych = fit_psychometric(stimuli[valid], choices[valid], x_eval)
                
                if psych.get('success', False):
                    metrics['sigma'].append(psych['sigma'])
                    metrics['mu'].append(psych['mu'])
                    metrics['psych_curves'].append(psych['y_fit'])
                    
                    gof = compute_psychometric_gof(stimuli[valid], choices[valid], psych)
                    metrics['r_squared'].append(gof['r_squared'])
                    metrics['deviance_explained'].append(gof['deviance_explained'])
                else:
                    for k in ['sigma', 'mu', 'r_squared', 'deviance_explained']:
                        metrics[k].append(np.nan)
                    metrics['psych_curves'].append(None)
            
            # Summarise
            for k in ['accuracy', 'r_squared', 'sigma', 'mu', 'deviance_explained']:
                metrics[k] = np.array(metrics[k])
                metrics[f'{k}_mean'] = np.nanmean(metrics[k])
                metrics[f'{k}_std'] = np.nanstd(metrics[k])
                
                results['matrices'][k][i, j] = metrics[f'{k}_mean']
                results['matrices'][f'{k}_std'][i, j] = metrics[f'{k}_std']
            
            # Mean curve
            valid_curves = [c for c in metrics['psych_curves'] if c is not None]
            if valid_curves:
                metrics['psych_curve_mean'] = np.nanmean(valid_curves, axis=0)
            else:
                metrics['psych_curve_mean'] = None
            
            metrics['accuracy_vs_chance_p'] = _one_sample_ttest(metrics['accuracy'], 0.5)[1]
            
            results['grid'][(p1_val, p2_val)] = metrics
    
    if verbose:
        print("\nDone!")
    
    return results


def be_param_joint_sweep_summary(results: Dict) -> pd.DataFrame:
    """
    Generate summary DataFrame from joint parameter sweep.
    
    Args:
        results: Output from be_param_joint_sweep
    
    Returns:
        DataFrame with one row per (param1, param2) combination
    """
    param1 = results['config']['param1']
    param2 = results['config']['param2']
    param1_values = results['config']['param1_values']
    param2_values = results['config']['param2_values']
    
    rows = []
    for p1_val in param1_values:
        for p2_val in param2_values:
            metrics = results['grid'][(p1_val, p2_val)]
            rows.append({
                param1: p1_val,
                param2: p2_val,
                'accuracy': metrics['accuracy_mean'],
                'accuracy_std': metrics['accuracy_std'],
                'accuracy_p': metrics['accuracy_vs_chance_p'],
                'r_squared': metrics['r_squared_mean'],
                'sigma': metrics['sigma_mean'],
                'mu': metrics['mu_mean'],
            })
    
    return pd.DataFrame(rows)


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
        show_gof: Whether to show GOF metrics (Acc, RÂ², RMSE)
        show_params: Whether to show Î¼, Ïƒ parameters
        show_lapse: Whether to show lapse parameters (Î»_low, Î»_high)
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
    # Burn-in analysis
    'burn_in_recovery_analysis',
    'burn_in_recovery_summary_stats',
    'fit_and_evaluate',
    # Mixed agent analysis
    'mixed_agent_recovery_analysis',
    'mixed_agent_recovery_summary_stats',
    'mixed_agent_parameter_sweep',
    'mixed_agent_sweep_summary',
    # BE parameter effects on behaviour
    'be_param_behaviour_sweep',
    'be_param_recovery_sweep',
    'be_param_sweep_summary',
    # Joint parameter effects
    'be_param_joint_sweep',
    'be_param_joint_sweep_summary',
    # Mixed agent BE param sweep
    'mixed_agent_be_param_sweep',
    'mixed_agent_be_param_sweep_summary',
    # Sobol sensitivity analysis
    'be_behaviour_simulator',
    'be_recovery_simulator',
    'sobol_be_behaviour',
    'sobol_be_recovery',
    # Burn-in specific plotting
    'plot_psychometric_by_burn_in',
    'plot_belief_after_burn_in',
    # Re-exported from Plotting
    'plot_burn_in_recovery',
    'plot_burn_in_param_distributions',
    'plot_mixed_agent_recovery',
    'plot_mixed_agent_param_distributions',
]


# =============================================================================
# SOBOL SENSITIVITY ANALYSIS - BE MODEL SIMULATORS
# =============================================================================

def be_behaviour_simulator(
    params: Dict[str, float],
    seed: int = 42,
    n_trials: int = 300
) -> Dict[str, float]:
    """
    Simulator function for BE model behaviour analysis.
    
    Takes BE parameters, simulates a session, returns behaviour metrics.
    Designed to work with sobol_analysis.run_sobol_analysis().
    
    Args:
        params: Dict with keys: sigma_percep, A_repulsion, eta_learning, eta_relax,
                and optionally burn_in
        seed: Random seed for reproducibility
        n_trials: Number of trials to simulate
    
    Returns:
        Dict with behaviour metrics:
            - accuracy: Proportion correct
            - mu: Psychometric PSE
            - sigma: Psychometric slope
            - lapse_low: Lower lapse rate
            - lapse_high: Upper lapse rate
            - r_squared: Psychometric R²
            - deviance_explained: Psychometric deviance explained
    """
    # Extract parameters
    be_params = {
        'sigma_percep': params['sigma_percep'],
        'A_repulsion': params['A_repulsion'],
        'eta_learning': params['eta_learning'],
        'eta_relax': params['eta_relax']
    }
    burn_in = int(params.get('burn_in', 0))
    
    # Generate stimuli
    stimuli, categories, rng = generate_stimuli(n_trials=n_trials, seed=seed)
    
    # Create and simulate
    model = BoundaryEstimationModel(**be_params)
    model.reset_belief(burn_in=burn_in, burn_in_seed=seed)
    
    sim_rng = np.random.default_rng(seed + 1)
    choices, _ = model.simulate_session(stimuli, categories, rng=sim_rng)
    
    # Compute outputs
    valid = ~np.isnan(choices)
    accuracy = np.mean(choices[valid] == categories[valid])
    
    # Fit psychometric
    x_eval = np.linspace(-1, 1, 100)
    psych = fit_psychometric(stimuli[valid], choices[valid], x_eval)
    
    if psych.get('success', False):
        gof = compute_psychometric_gof(stimuli[valid], choices[valid], psych)
        
        return {
            'accuracy': accuracy,
            'mu': psych['mu'],
            'sigma': psych['sigma'],
            'lapse_low': psych['lapse_low'],
            'lapse_high': psych['lapse_high'],
            'r_squared': gof['r_squared'],
            'deviance_explained': gof['deviance_explained'],
        }
    else:
        return {
            'accuracy': accuracy,
            'mu': np.nan,
            'sigma': np.nan,
            'lapse_low': np.nan,
            'lapse_high': np.nan,
            'r_squared': np.nan,
            'deviance_explained': np.nan,
        }


def be_recovery_simulator(
    params: Dict[str, float],
    seed: int = 42,
    n_trials: int = 300,
    fitter_burn_in: int = 0,
    validation: str = 'holdout'
) -> Dict[str, float]:
    """
    Simulator function for BE model parameter recovery analysis.
    
    Takes BE parameters, simulates a session, fits BE model back,
    returns recovery metrics.
    
    Args:
        params: Dict with BE parameters and optionally burn_in
        seed: Random seed
        n_trials: Number of trials
        fitter_burn_in: Burn-in assumed by fitter
        validation: Validation method ('holdout', 'kfold', None)
    
    Returns:
        Dict with recovery metrics:
            - {param}_bias: Fitted - true for each BE param
            - {param}_abs_error: Absolute error
            - psych_mu_error: Error in recovered psychometric μ
            - psych_sigma_error: Error in recovered psychometric σ
            - psych_curve_mae: MAE between true and recovered curves
            - train_nll: Training NLL per trial
    """
    # Extract true parameters
    true_params = {
        'sigma_percep': params['sigma_percep'],
        'A_repulsion': params['A_repulsion'],
        'eta_learning': params['eta_learning'],
        'eta_relax': params['eta_relax']
    }
    burn_in = int(params.get('burn_in', 0))
    
    # Generate and simulate
    stimuli, categories, rng = generate_stimuli(n_trials=n_trials, seed=seed)
    
    model = BoundaryEstimationModel(**true_params)
    model.reset_belief(burn_in=burn_in, burn_in_seed=seed)
    
    sim_rng = np.random.default_rng(seed + 1)
    choices, _ = model.simulate_session(stimuli, categories, rng=sim_rng)
    
    # True psychometric
    valid = ~np.isnan(choices)
    x_eval = np.linspace(-1, 1, 100)
    psych_true = fit_psychometric(stimuli[valid], choices[valid], x_eval)
    
    # Fit BE model
    try:
        fitted_model, fit_results = BoundaryEstimationModel.fit(
            stimuli, categories, choices,
            burn_in=fitter_burn_in,
            burn_in_seed=seed,
            validation=validation,
            n_restarts=5,
            seed=seed + 2
        )
        
        # Parameter recovery metrics
        outputs = {}
        for name in true_params:
            fitted_val = fit_results['params'][name]
            true_val = true_params[name]
            outputs[f'{name}_bias'] = fitted_val - true_val
            outputs[f'{name}_abs_error'] = abs(fitted_val - true_val)
        
        outputs['train_nll'] = fit_results.get('train_nll_per_trial', np.nan)
        
        # Simulate from fitted model to get recovered psychometric
        fitted_model.reset_belief(burn_in=fitter_burn_in, burn_in_seed=seed)
        fitted_choices, _ = fitted_model.simulate_session(stimuli, categories, 
                                                          rng=np.random.default_rng(seed + 3))
        
        valid_fitted = ~np.isnan(fitted_choices)
        psych_fitted = fit_psychometric(stimuli[valid_fitted], fitted_choices[valid_fitted], x_eval)
        
        # Psychometric curve comparison
        if psych_true.get('success', False) and psych_fitted.get('success', False):
            outputs['psych_mu_error'] = psych_fitted['mu'] - psych_true['mu']
            outputs['psych_sigma_error'] = psych_fitted['sigma'] - psych_true['sigma']
            outputs['psych_lapse_low_error'] = psych_fitted['lapse_low'] - psych_true['lapse_low']
            outputs['psych_lapse_high_error'] = psych_fitted['lapse_high'] - psych_true['lapse_high']
            outputs['psych_curve_mae'] = np.mean(np.abs(psych_fitted['y_fit'] - psych_true['y_fit']))
        else:
            outputs['psych_mu_error'] = np.nan
            outputs['psych_sigma_error'] = np.nan
            outputs['psych_lapse_low_error'] = np.nan
            outputs['psych_lapse_high_error'] = np.nan
            outputs['psych_curve_mae'] = np.nan
        
        return outputs
        
    except Exception as e:
        # Return NaN on failure
        outputs = {}
        for name in true_params:
            outputs[f'{name}_bias'] = np.nan
            outputs[f'{name}_abs_error'] = np.nan
        outputs['train_nll'] = np.nan
        outputs['psych_mu_error'] = np.nan
        outputs['psych_sigma_error'] = np.nan
        outputs['psych_lapse_low_error'] = np.nan
        outputs['psych_lapse_high_error'] = np.nan
        outputs['psych_curve_mae'] = np.nan
        return outputs


def sobol_be_behaviour(
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    burn_in_values: List[int] = [0, 500, 1000, 2000],
    n_sobol: int = 256,
    n_replicates: int = 5,
    n_trials: int = 300,
    seed: int = 42,
    verbose: bool = True
) -> 'SobolResults':
    """
    Run Sobol sensitivity analysis on BE model behaviour.
    
    Analyses how BE parameters and burn_in affect behaviour metrics.
    
    Args:
        param_ranges: {param_name: (min, max)} for BE params.
                     If None, uses sensible defaults.
        burn_in_values: Discrete burn_in values to test
        n_sobol: Sobol sequence base size
        n_replicates: Replicates per parameter combination
        n_trials: Trials per session
        seed: Random seed
        verbose: Print progress
    
    Returns:
        SobolResults with sensitivity indices for:
            - accuracy, mu, sigma, lapse_low, lapse_high, r_squared, deviance_explained
    
    Example:
        results = sobol_be_behaviour(n_sobol=128, n_replicates=3)
        print(results.sensitivity['accuracy'])
        
        # Most influential parameter for accuracy
        print(results.most_influential('accuracy'))
    """
    # Import here to avoid circular import
    try:
        from Analysis.sobol_analysis import run_sobol_analysis
    except ImportError:
        from Analysis.sobol_analysis import run_sobol_analysis
    
    # Default ranges
    if param_ranges is None:
        param_ranges = {
            'sigma_percep': (0.05, 0.5),
            'A_repulsion': (0.0, 0.3),
            'eta_learning': (0.1, 0.6),
            'eta_relax': (0.05, 0.3),
        }
    
    # Create simulator with fixed n_trials
    def simulator(params, seed):
        return be_behaviour_simulator(params, seed=seed, n_trials=n_trials)
    
    # Output names
    output_names = ['accuracy', 'mu', 'sigma', 'lapse_low', 'lapse_high', 
                   'r_squared', 'deviance_explained']
    
    # Run analysis
    results = run_sobol_analysis(
        simulator=simulator,
        param_ranges=param_ranges,
        output_names=output_names,
        n_sobol=n_sobol,
        n_replicates=n_replicates,
        discrete_params={'burn_in': burn_in_values},
        seed=seed,
        verbose=verbose
    )
    
    return results


def sobol_be_recovery(
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    burn_in_values: List[int] = [0, 500, 1000, 2000],
    fitter_burn_in: int = 0,
    n_sobol: int = 128,
    n_replicates: int = 5,
    n_trials: int = 300,
    seed: int = 42,
    verbose: bool = True
) -> 'SobolResults':
    """
    Run Sobol sensitivity analysis on BE model parameter recovery.
    
    Analyses how true BE parameters and burn_in affect recovery accuracy.
    
    Args:
        param_ranges: {param_name: (min, max)} for BE params
        burn_in_values: Discrete burn_in values to test
        fitter_burn_in: Burn-in assumed by the fitter
        n_sobol: Sobol sequence base size (use fewer than behaviour - fitting is slow)
        n_replicates: Replicates per parameter combination
        n_trials: Trials per session
        seed: Random seed
        verbose: Print progress
    
    Returns:
        SobolResults with sensitivity indices for recovery metrics
    
    Example:
        results = sobol_be_recovery(n_sobol=64, n_replicates=3)
        print(results.sensitivity['sigma_percep_bias'])
    """
    try:
        from Analysis.sobol_analysis import run_sobol_analysis
    except ImportError:
        from Analysis.sobol_analysis import run_sobol_analysis
    
    # Default ranges
    if param_ranges is None:
        param_ranges = {
            'sigma_percep': (0.05, 0.5),
            'A_repulsion': (0.0, 0.3),
            'eta_learning': (0.1, 0.6),
            'eta_relax': (0.05, 0.3),
        }
    
    # Create simulator with fixed settings
    def simulator(params, seed):
        return be_recovery_simulator(
            params, seed=seed, n_trials=n_trials, 
            fitter_burn_in=fitter_burn_in
        )
    
    # Output names
    output_names = [
        'sigma_percep_bias', 'A_repulsion_bias', 'eta_learning_bias', 'eta_relax_bias',
        'sigma_percep_abs_error', 'A_repulsion_abs_error', 'eta_learning_abs_error', 'eta_relax_abs_error',
        'psych_mu_error', 'psych_sigma_error', 'psych_lapse_low_error', 'psych_lapse_high_error',
        'psych_curve_mae', 'train_nll'
    ]
    
    # Run analysis
    results = run_sobol_analysis(
        simulator=simulator,
        param_ranges=param_ranges,
        output_names=output_names,
        n_sobol=n_sobol,
        n_replicates=n_replicates,
        discrete_params={'burn_in': burn_in_values},
        seed=seed,
        verbose=verbose
    )
    
    return results
