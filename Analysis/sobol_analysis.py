"""
Sobol Sensitivity Analysis Module

Model-agnostic Sobol sensitivity analysis for understanding parameter effects.
Designed to work with any simulator function that takes params → returns metrics.

Usage:
    from Analysis.sobol_analysis import run_sobol_analysis, SobolResults
    
    results = run_sobol_analysis(
        simulator=my_simulator_function,
        param_ranges={'param1': (0, 1), 'param2': (0, 10)},
        output_names=['accuracy', 'sigma'],
        n_sobol=256,
        n_replicates=5
    )
    
    print(results.sensitivity['accuracy'])  # S1, ST indices

Dependencies:
    pip install SALib
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    warnings.warn("SALib not installed. Install with: pip install SALib\n"
                 "Using fallback random sampling (less efficient).")


# =============================================================================
# FALLBACK IMPLEMENTATIONS (when SALib not available)
# =============================================================================

def _fallback_sample(problem: Dict, n: int, seed: int = 42) -> np.ndarray:
    """
    Fallback sampling using Latin Hypercube when SALib unavailable.
    
    Less efficient than Saltelli scheme but still space-filling.
    """
    rng = np.random.default_rng(seed)
    d = problem['num_vars']
    bounds = np.array(problem['bounds'])
    
    # Latin Hypercube Sampling
    samples = np.zeros((n, d))
    for i in range(d):
        lo, hi = bounds[i]
        # Create stratified samples
        cuts = np.linspace(0, 1, n + 1)
        uniform_samples = rng.uniform(cuts[:-1], cuts[1:])
        rng.shuffle(uniform_samples)
        samples[:, i] = lo + uniform_samples * (hi - lo)
    
    return samples


def _fallback_analyze(
    problem: Dict,
    Y: np.ndarray,
    samples: np.ndarray
) -> Dict:
    """
    Fallback sensitivity analysis using correlation-based measures.
    
    Not true Sobol indices but provides rough importance ranking.
    """
    d = problem['num_vars']
    n = len(Y)
    
    # Simple correlation-based importance
    s1 = np.zeros(d)
    s1_conf = np.zeros(d)
    
    for i in range(d):
        # Pearson correlation squared ≈ proportion of variance explained
        valid = ~np.isnan(Y)
        if valid.sum() > 10:
            corr = np.corrcoef(samples[valid, i], Y[valid])[0, 1]
            s1[i] = corr ** 2
            # Bootstrap confidence
            boot_corrs = []
            rng = np.random.default_rng(42)
            for _ in range(100):
                idx = rng.choice(valid.sum(), size=valid.sum(), replace=True)
                bc = np.corrcoef(samples[valid][idx, i], Y[valid][idx])[0, 1]
                boot_corrs.append(bc ** 2)
            s1_conf[i] = np.std(boot_corrs) * 1.96
    
    # Normalise to sum to 1 (roughly)
    total = s1.sum()
    if total > 0:
        s1 = s1 / total
        s1_conf = s1_conf / total
    
    return {
        'S1': s1,
        'S1_conf': s1_conf,
        'ST': s1 * 1.1,  # Rough approximation (no interaction info)
        'ST_conf': s1_conf * 1.1,
    }


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SobolResults:
    """
    Container for Sobol sensitivity analysis results.
    
    Attributes:
        sensitivity: Dict[output_name, DataFrame] with S1, ST, S1_conf, ST_conf
        interactions: Dict[output_name, DataFrame] with S2 (second-order indices)
        raw_samples: Array of parameter samples (N × d)
        raw_outputs: Dict[output_name, Array] of outputs for each sample
        problem: SALib problem definition
        config: Analysis configuration
    """
    sensitivity: Dict[str, pd.DataFrame]
    interactions: Dict[str, pd.DataFrame]
    raw_samples: np.ndarray
    raw_outputs: Dict[str, np.ndarray]
    problem: Dict
    config: Dict
    
    def summary(self, output: Optional[str] = None) -> pd.DataFrame:
        """
        Get summary of sensitivity indices.
        
        Args:
            output: Specific output to summarise (None = all outputs)
        
        Returns:
            DataFrame with S1, ST for each parameter
        """
        if output is not None:
            return self.sensitivity[output]
        
        # Combine all outputs
        dfs = []
        for out_name, df in self.sensitivity.items():
            df = df.copy()
            df['output'] = out_name
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    
    def most_influential(self, output: str, metric: str = 'ST') -> List[str]:
        """
        Get parameters ranked by influence.
        
        Args:
            output: Which output to rank by
            metric: 'S1' (first-order) or 'ST' (total-order)
        
        Returns:
            List of parameter names, most influential first
        """
        df = self.sensitivity[output]
        return df.sort_values(metric, ascending=False)['parameter'].tolist()
    
    def has_interactions(self, output: str, threshold: float = 0.1) -> bool:
        """
        Check if there are significant interactions.
        
        Interactions exist when sum(S1) < 1 or when ST - S1 is large.
        
        Args:
            output: Which output to check
            threshold: Gap threshold for ST - S1
        
        Returns:
            True if interactions are significant
        """
        df = self.sensitivity[output]
        s1_sum = df['S1'].sum()
        max_gap = (df['ST'] - df['S1']).max()
        
        return s1_sum < 0.9 or max_gap > threshold


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def create_problem_definition(
    param_ranges: Dict[str, Tuple[float, float]],
    discrete_params: Optional[Dict[str, List[float]]] = None
) -> Dict:
    """
    Create SALib problem definition.
    
    Args:
        param_ranges: {param_name: (min, max)} for continuous params
        discrete_params: {param_name: [values]} for discrete params
                        These are treated as continuous but discretized later
    
    Returns:
        SALib problem dict with 'num_vars', 'names', 'bounds'
    """
    names = []
    bounds = []
    
    # Add continuous params
    for name, (lo, hi) in param_ranges.items():
        names.append(name)
        bounds.append([lo, hi])
    
    # Add discrete params (as continuous range, discretized later)
    if discrete_params:
        for name, values in discrete_params.items():
            names.append(name)
            bounds.append([min(values), max(values)])
    
    return {
        'num_vars': len(names),
        'names': names,
        'bounds': bounds
    }


def sobol_sample(
    problem: Dict,
    n: int = 256,
    seed: int = 42
) -> np.ndarray:
    """
    Generate Sobol samples using Saltelli's scheme.
    
    Args:
        problem: SALib problem definition
        n: Base sample size (total samples = n × (2d + 2) with SALib,
           or just n with fallback)
        seed: Random seed for reproducibility
    
    Returns:
        Array of shape (N, d) with parameter samples
    """
    if SALIB_AVAILABLE:
        # SALib's saltelli.sample
        samples = saltelli.sample(problem, n, calc_second_order=True)
    else:
        # Fallback: simple LHS with more samples to compensate
        d = problem['num_vars']
        n_total = n * (2 * d + 2)  # Match Saltelli sample count
        samples = _fallback_sample(problem, n_total, seed)
    
    return samples


def discretize_samples(
    samples: np.ndarray,
    problem: Dict,
    discrete_params: Dict[str, List[float]]
) -> np.ndarray:
    """
    Discretize specified parameters to nearest allowed value.
    
    Args:
        samples: Array of continuous samples
        problem: SALib problem definition
        discrete_params: {param_name: [allowed_values]}
    
    Returns:
        Samples with discrete params rounded to nearest allowed value
    """
    samples = samples.copy()
    names = problem['names']
    
    for param_name, allowed_values in discrete_params.items():
        if param_name in names:
            idx = names.index(param_name)
            allowed = np.array(sorted(allowed_values))
            
            # Find nearest allowed value for each sample
            for i in range(len(samples)):
                val = samples[i, idx]
                nearest_idx = np.argmin(np.abs(allowed - val))
                samples[i, idx] = allowed[nearest_idx]
    
    return samples


def _run_single_simulation(
    args: Tuple[int, Dict, Callable, int, int]
) -> Tuple[int, List[Dict]]:
    """Helper for parallel simulation."""
    sample_idx, params, simulator, n_replicates, base_seed = args
    
    replicate_outputs = []
    for rep in range(n_replicates):
        seed = base_seed + sample_idx * 1000 + rep
        try:
            output = simulator(params, seed=seed)
            replicate_outputs.append(output)
        except Exception as e:
            # Return NaN dict on failure
            replicate_outputs.append(None)
    
    return sample_idx, replicate_outputs


def run_sobol_analysis(
    simulator: Callable[[Dict, int], Dict],
    param_ranges: Dict[str, Tuple[float, float]],
    output_names: List[str],
    n_sobol: int = 256,
    n_replicates: int = 5,
    discrete_params: Optional[Dict[str, List[float]]] = None,
    seed: int = 42,
    n_jobs: int = 1,
    verbose: bool = True
) -> SobolResults:
    """
    Run Sobol sensitivity analysis with any simulator.
    
    This is the main entry point for Sobol analysis. It:
    1. Generates Sobol samples (Saltelli scheme)
    2. Runs simulator for each sample × replicate
    3. Computes first-order (S1) and total-order (ST) indices
    4. Optionally computes second-order (S2) interaction indices
    
    Args:
        simulator: Function with signature (params: Dict, seed: int) -> Dict
                  Takes parameter dict and seed, returns output metrics dict
        param_ranges: {param_name: (min, max)} for each parameter
        output_names: List of output names to analyse (must be keys in simulator output)
        n_sobol: Sobol sequence base size. Total samples = n × (2d + 2)
                 where d = number of parameters
        n_replicates: Number of replicates per parameter combination
                      (averaged before Sobol analysis)
        discrete_params: {param_name: [allowed_values]} for discrete parameters
                        These are sampled continuously then rounded to nearest value
        seed: Random seed
        n_jobs: Number of parallel workers (1 = sequential)
        verbose: Print progress
    
    Returns:
        SobolResults object with sensitivity indices and raw data
    
    Example:
        def my_simulator(params, seed):
            # ... run simulation ...
            return {'accuracy': 0.75, 'sigma': 0.3}
        
        results = run_sobol_analysis(
            simulator=my_simulator,
            param_ranges={'param1': (0, 1), 'param2': (0, 10)},
            output_names=['accuracy', 'sigma'],
            n_sobol=256
        )
    """
    # Create problem definition
    problem = create_problem_definition(param_ranges, discrete_params)
    d = problem['num_vars']
    
    if verbose:
        if not SALIB_AVAILABLE:
            print("WARNING: SALib not available, using fallback correlation-based analysis")
            print("         Install SALib for proper Sobol indices: pip install SALib\n")
        print(f"Sobol Analysis Setup:")
        print(f"  Parameters: {problem['names']}")
        n_total = n_sobol * (2*d + 2) if SALIB_AVAILABLE else n_sobol * (2*d + 2)
        print(f"  n_sobol={n_sobol} → {n_total} samples")
        print(f"  n_replicates={n_replicates}")
        print(f"  Total simulations: {n_total * n_replicates}")
    
    # Generate samples
    samples = sobol_sample(problem, n=n_sobol, seed=seed)
    
    # Discretize if needed
    if discrete_params:
        samples = discretize_samples(samples, problem, discrete_params)
    
    n_samples = len(samples)
    
    # Storage for outputs
    outputs_storage = {name: [] for name in output_names}
    
    if verbose:
        print(f"\nRunning {n_samples} parameter combinations...")
    
    # Run simulations
    if n_jobs == 1:
        # Sequential
        for i, sample in enumerate(samples):
            if verbose and (i + 1) % 100 == 0:
                print(f"  {i+1}/{n_samples} ({100*(i+1)/n_samples:.0f}%)")
            
            # Convert to param dict
            params = {name: sample[j] for j, name in enumerate(problem['names'])}
            
            # Run replicates
            replicate_values = {name: [] for name in output_names}
            
            for rep in range(n_replicates):
                rep_seed = seed + i * 1000 + rep
                try:
                    result = simulator(params, seed=rep_seed)
                    for name in output_names:
                        replicate_values[name].append(result.get(name, np.nan))
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Sample {i} rep {rep} failed: {e}")
                    for name in output_names:
                        replicate_values[name].append(np.nan)
            
            # Average over replicates
            for name in output_names:
                outputs_storage[name].append(np.nanmean(replicate_values[name]))
    
    else:
        # Parallel (simplified - full parallel would need more work)
        raise NotImplementedError("Parallel execution not yet implemented. Use n_jobs=1")
    
    # Convert to arrays
    for name in output_names:
        outputs_storage[name] = np.array(outputs_storage[name])
    
    if verbose:
        print("\nComputing Sobol indices...")
    
    # Compute Sobol indices for each output
    sensitivity_results = {}
    interaction_results = {}
    
    for name in output_names:
        Y = outputs_storage[name]
        
        # Check for NaN issues
        nan_frac = np.isnan(Y).mean()
        if nan_frac > 0.1:
            warnings.warn(f"Output '{name}' has {nan_frac*100:.1f}% NaN values. "
                         f"Sobol analysis may be unreliable.")
        
        # Replace NaN with mean (crude but allows analysis to proceed)
        Y_clean = Y.copy()
        Y_clean[np.isnan(Y_clean)] = np.nanmean(Y_clean)
        
        # Sobol analysis
        try:
            if SALIB_AVAILABLE:
                # Full SALib Sobol analysis
                Si = sobol.analyze(problem, Y_clean, calc_second_order=True, 
                                  print_to_console=False)
                
                # First-order and total-order
                sensitivity_results[name] = pd.DataFrame({
                    'parameter': problem['names'],
                    'S1': Si['S1'],
                    'S1_conf': Si['S1_conf'],
                    'ST': Si['ST'],
                    'ST_conf': Si['ST_conf'],
                })
                
                # Second-order interactions
                if 'S2' in Si:
                    s2_data = []
                    for i in range(d):
                        for j in range(i+1, d):
                            s2_data.append({
                                'param1': problem['names'][i],
                                'param2': problem['names'][j],
                                'S2': Si['S2'][i, j],
                                'S2_conf': Si['S2_conf'][i, j]
                            })
                    interaction_results[name] = pd.DataFrame(s2_data)
                else:
                    interaction_results[name] = pd.DataFrame()
            else:
                # Fallback correlation-based analysis
                Si = _fallback_analyze(problem, Y_clean, samples)
                
                sensitivity_results[name] = pd.DataFrame({
                    'parameter': problem['names'],
                    'S1': Si['S1'],
                    'S1_conf': Si['S1_conf'],
                    'ST': Si['ST'],
                    'ST_conf': Si['ST_conf'],
                })
                
                # No second-order with fallback
                interaction_results[name] = pd.DataFrame()
                
        except Exception as e:
            warnings.warn(f"Sobol analysis failed for '{name}': {e}")
            sensitivity_results[name] = pd.DataFrame({
                'parameter': problem['names'],
                'S1': [np.nan] * d,
                'S1_conf': [np.nan] * d,
                'ST': [np.nan] * d,
                'ST_conf': [np.nan] * d,
            })
            interaction_results[name] = pd.DataFrame()
    
    if verbose:
        print("Done!")
        print("\nSummary (first output):")
        first_output = output_names[0]
        print(sensitivity_results[first_output].to_string(index=False))
    
    # Package results
    config = {
        'param_ranges': param_ranges,
        'discrete_params': discrete_params,
        'output_names': output_names,
        'n_sobol': n_sobol,
        'n_replicates': n_replicates,
        'n_samples': n_samples,
        'seed': seed
    }
    
    return SobolResults(
        sensitivity=sensitivity_results,
        interactions=interaction_results,
        raw_samples=samples,
        raw_outputs=outputs_storage,
        problem=problem,
        config=config
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_sensitivity_check(
    simulator: Callable[[Dict, int], Dict],
    param_ranges: Dict[str, Tuple[float, float]],
    output_name: str,
    n_sobol: int = 64,
    n_replicates: int = 3,
    discrete_params: Optional[Dict[str, List[float]]] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Quick sensitivity check with fewer samples.
    
    Useful for initial exploration before full analysis.
    
    Args:
        simulator: Simulator function
        param_ranges: Parameter ranges
        output_name: Single output to analyse
        n_sobol: Smaller sample size (default 64)
        n_replicates: Fewer replicates (default 3)
        discrete_params: Discrete parameters
        seed: Random seed
    
    Returns:
        DataFrame with S1, ST for the output
    """
    results = run_sobol_analysis(
        simulator=simulator,
        param_ranges=param_ranges,
        output_names=[output_name],
        n_sobol=n_sobol,
        n_replicates=n_replicates,
        discrete_params=discrete_params,
        seed=seed,
        verbose=False
    )
    
    return results.sensitivity[output_name]


def compare_outputs_sensitivity(results: SobolResults) -> pd.DataFrame:
    """
    Compare sensitivity across multiple outputs.
    
    Creates a wide table showing which parameters matter for which outputs.
    
    Args:
        results: SobolResults from run_sobol_analysis
    
    Returns:
        DataFrame with parameters as rows, outputs as columns, ST as values
    """
    param_names = results.problem['names']
    output_names = results.config['output_names']
    
    data = {name: [] for name in output_names}
    data['parameter'] = param_names
    
    for out_name in output_names:
        df = results.sensitivity[out_name]
        for param in param_names:
            st = df[df['parameter'] == param]['ST'].values[0]
            data[out_name].append(st)
    
    result_df = pd.DataFrame(data)
    result_df = result_df.set_index('parameter')
    
    return result_df


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SobolResults',
    'create_problem_definition',
    'sobol_sample',
    'discretize_samples',
    'run_sobol_analysis',
    'quick_sensitivity_check',
    'compare_outputs_sensitivity',
    'SALIB_AVAILABLE',
]
