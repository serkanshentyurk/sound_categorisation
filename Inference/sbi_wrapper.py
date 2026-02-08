"""
SBI Wrapper for BE and MixedAgent Models.

High-level interface for running simulation-based inference:
- Training neural density estimators (NPE, NLE, NRE)
- Posterior sampling
- Multi-round (sequential) inference

Usage:
    from Inference.sbi_wrapper import train_sbi, sample_posterior
    from Inference.simulator import create_be_simulator
    from Inference.priors import create_prior
    
    # Setup
    simulator = create_be_simulator(stimuli, categories, burn_in=100)
    prior = create_prior()
    
    # Get observed summary stats
    observed_stats = compute_summary_stats(observed_choices, stimuli, categories)
    
    # Train
    result = train_sbi(
        simulator=simulator,
        prior=prior,
        observed_stats=observed_stats,
        method='NPE',
        n_simulations=50000
    )
    
    # Sample from posterior
    samples = sample_posterior(result.posterior, observed_stats, n_samples=10000)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
import warnings
import time


# =============================================================================
# RESULT CONTAINERS
# =============================================================================

@dataclass
class SBIResult:
    """
    Container for SBI training results.
    
    Attributes:
        posterior: Trained posterior object (can sample and evaluate)
        inference: The sbi inference object
        density_estimator: The trained neural network
        method: SBI method used ('NPE', 'NLE', 'NRE')
        n_simulations: Total simulations used
        n_rounds: Number of sequential rounds
        training_time: Time taken to train (seconds)
        theta_train: Training parameters (if stored)
        x_train: Training summary stats (if stored)
        prior: Prior used
        observed_stats: Observed data summary stats (if sequential)
    """
    posterior: Any
    inference: Any
    density_estimator: Any
    method: str
    n_simulations: int
    n_rounds: int
    training_time: float
    theta_train: Optional[torch.Tensor] = None
    x_train: Optional[torch.Tensor] = None
    prior: Any = None
    observed_stats: Optional[torch.Tensor] = None
    param_names: Optional[List[str]] = None
    
    def sample(self, n_samples: int, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convenience method to sample from posterior.
        
        Args:
            n_samples: Number of samples
            x: Observed stats (uses stored if None)
        
        Returns:
            Samples tensor of shape (n_samples, n_params)
        """
        if x is None:
            x = self.observed_stats
        if x is None:
            raise ValueError("Must provide observed_stats (x)")
        
        return self.posterior.sample((n_samples,), x=x)
    
    def log_prob(self, theta: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluate posterior log probability.
        
        Args:
            theta: Parameter values, shape (n, n_params)
            x: Observed stats (uses stored if None)
        
        Returns:
            Log probabilities, shape (n,)
        """
        if x is None:
            x = self.observed_stats
        if x is None:
            raise ValueError("Must provide observed_stats (x)")
        
        return self.posterior.log_prob(theta, x=x)
    
    def map_estimate(self, x: Optional[torch.Tensor] = None, 
                     n_samples: int = 10000) -> torch.Tensor:
        """
        Get MAP (maximum a posteriori) estimate.
        
        Approximates MAP by finding sample with highest posterior density.
        
        Args:
            x: Observed stats
            n_samples: Number of samples to draw for approximation
        
        Returns:
            MAP estimate, shape (n_params,)
        """
        samples = self.sample(n_samples, x=x)
        log_probs = self.log_prob(samples, x=x)
        map_idx = log_probs.argmax()
        return samples[map_idx]
    
    def posterior_mean(self, x: Optional[torch.Tensor] = None,
                       n_samples: int = 10000) -> torch.Tensor:
        """
        Get posterior mean estimate.
        
        Args:
            x: Observed stats
            n_samples: Number of samples
        
        Returns:
            Mean estimate, shape (n_params,)
        """
        samples = self.sample(n_samples, x=x)
        return samples.mean(dim=0)
    
    def credible_interval(self, x: Optional[torch.Tensor] = None,
                          level: float = 0.95,
                          n_samples: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get credible interval.
        
        Args:
            x: Observed stats
            level: Credible level (e.g., 0.95 for 95% CI)
            n_samples: Number of samples
        
        Returns:
            (lower, upper) tensors of shape (n_params,)
        """
        samples = self.sample(n_samples, x=x)
        alpha = (1 - level) / 2
        lower = torch.quantile(samples, alpha, dim=0)
        upper = torch.quantile(samples, 1 - alpha, dim=0)
        return lower, upper
    
    def summary(self, x: Optional[torch.Tensor] = None,
                n_samples: int = 10000) -> Dict[str, Any]:
        """
        Get posterior summary statistics.
        
        Returns dict with mean, std, median, and 95% CI for each parameter.
        """
        samples = self.sample(n_samples, x=x)
        lower, upper = self.credible_interval(x=x, n_samples=n_samples)
        
        summary = {}
        param_names = self.param_names or [f'param_{i}' for i in range(samples.shape[1])]
        
        for i, name in enumerate(param_names):
            summary[name] = {
                'mean': float(samples[:, i].mean()),
                'std': float(samples[:, i].std()),
                'median': float(samples[:, i].median()),
                'ci_lower': float(lower[i]),
                'ci_upper': float(upper[i]),
            }
        
        return summary


# =============================================================================
# SIMULATOR WRAPPER
# =============================================================================

def _wrap_simulator_for_sbi(simulator: Callable, seed_offset: int = 0) -> Callable:
    """
    Wrap a numpy-based simulator for sbi (handles torch tensors).
    
    Args:
        simulator: Function that takes numpy array and returns numpy array
        seed_offset: Offset added to sample index for seeding
    
    Returns:
        Function compatible with sbi
    """
    def wrapped(theta: torch.Tensor) -> torch.Tensor:
        # Convert to numpy
        if hasattr(theta, 'numpy'):
            theta_np = theta.numpy()
        else:
            theta_np = np.asarray(theta)
        
        # Handle batched input
        if theta_np.ndim == 1:
            result = simulator(theta_np)
            return torch.tensor(result, dtype=torch.float32)
        else:
            results = []
            for i in range(len(theta_np)):
                # Use index as seed variation
                if hasattr(simulator, 'simulate'):
                    result = simulator.simulate(theta_np[i], seed=seed_offset + i)
                else:
                    result = simulator(theta_np[i])
                results.append(result)
            return torch.tensor(np.stack(results), dtype=torch.float32)
    
    return wrapped


def _get_sbi_prior(prior):
    """
    Convert prior to sbi-compatible format.
    
    Args:
        prior: UniformPrior, MultiSessionPrior, or sbi prior
    
    Returns:
        sbi-compatible prior
    """
    # If already sbi prior, return as-is
    if hasattr(prior, 'sample') and hasattr(prior, 'log_prob'):
        # Check if it's our custom prior
        if hasattr(prior, 'to_sbi_prior'):
            return prior.to_sbi_prior()
        # Check if needs wrapping for sbi
        try:
            from sbi.utils import BoxUniform
            if isinstance(prior, BoxUniform):
                return prior
        except ImportError:
            pass
        return prior
    
    raise ValueError(f"Unknown prior type: {type(prior)}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_sbi(
    simulator: Callable,
    prior: Any,
    observed_stats: Optional[Union[np.ndarray, torch.Tensor]] = None,
    method: str = 'NPE',
    n_simulations: int = 50000,
    n_rounds: int = 1,
    training_batch_size: int = 256,
    learning_rate: float = 5e-4,
    hidden_features: int = 50,
    num_transforms: int = 5,
    stop_after_epochs: int = 20,
    validation_fraction: float = 0.1,
    density_estimator: Optional[str] = None,
    device: str = 'cpu',
    seed: Optional[int] = None,
    store_simulations: bool = False,
    show_progress: bool = True,
    param_names: Optional[List[str]] = None,
    **kwargs
) -> SBIResult:
    """
    Train SBI model and return posterior.
    
    Args:
        simulator: Callable that takes parameter array and returns summary stats.
                  Can be a Simulator object or any callable.
        prior: Prior distribution (UniformPrior, MultiSessionPrior, or sbi prior)
        observed_stats: Observed data summary statistics (required for n_rounds > 1)
        method: SBI method - 'NPE', 'NLE', or 'NRE'
        n_simulations: Total number of simulations (split across rounds)
        n_rounds: Number of sequential inference rounds (1 = amortized)
        training_batch_size: Batch size for training
        learning_rate: Learning rate for neural network
        hidden_features: Hidden layer size
        num_transforms: Number of transforms (for flow-based methods)
        stop_after_epochs: Early stopping patience
        validation_fraction: Fraction of data for validation
        density_estimator: Type of density estimator (None = auto select)
                          NPE: 'maf', 'nsf', 'mdn'
                          NLE: 'maf', 'nsf', 'mdn'
                          NRE: 'resnet', 'mlp'
        device: 'cpu' or 'cuda'
        seed: Random seed
        store_simulations: If True, store training data in result
        show_progress: Show progress bar
        param_names: Parameter names for labelling
        **kwargs: Additional kwargs passed to density estimator
    
    Returns:
        SBIResult containing trained posterior and metadata
    
    Example:
        result = train_sbi(
            simulator=my_simulator,
            prior=my_prior,
            observed_stats=obs_stats,
            method='NPE',
            n_simulations=50000,
            n_rounds=1
        )
        samples = result.sample(10000)
    """
    try:
        import sbi
        from sbi.inference import SNPE, SNLE, SNRE
        from sbi.utils import BoxUniform
        try:
            from sbi.utils.user_input_checks import process_prior
        except ImportError:
            from sbi.utils import process_prior
    except ImportError:
        raise ImportError("sbi package required. Install with: pip install sbi")
    
    # Set seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    start_time = time.time()
    
    # Convert observed_stats to tensor
    if observed_stats is not None:
        if isinstance(observed_stats, np.ndarray):
            observed_stats = torch.tensor(observed_stats, dtype=torch.float32)
        observed_stats = observed_stats.to(device)
    
    # Setup prior — must be a PyTorch Distribution or wrapped via process_prior
    if hasattr(prior, 'low') and hasattr(prior, 'high'):
        # Our custom UniformPrior — convert to BoxUniform directly
        sbi_prior = BoxUniform(
            low=prior.low.to(device),
            high=prior.high.to(device)
        )
        if param_names is None and hasattr(prior, 'param_names'):
            param_names = prior.param_names
    elif hasattr(prior, 'sample') and hasattr(prior, 'log_prob'):
        # Custom prior with sample/log_prob (e.g. MultiSessionPrior)
        # Wrap via process_prior so SBI accepts it
        result = process_prior(prior)
        # process_prior returns (prior, n_params, numpy_flag) or just prior
        sbi_prior = result[0] if isinstance(result, tuple) else result
        if param_names is None and hasattr(prior, 'param_names'):
            param_names = prior.param_names
    else:
        sbi_prior = prior
    
    # Wrap simulator
    sim_wrapped = _wrap_simulator_for_sbi(simulator, seed_offset=seed or 0)
    
    # Select inference class
    method = method.upper()
    if method == 'NPE':
        InferenceClass = SNPE
        default_estimator = 'maf'
    elif method == 'NLE':
        InferenceClass = SNLE
        default_estimator = 'maf'
    elif method == 'NRE':
        InferenceClass = SNRE
        default_estimator = 'resnet'
    else:
        raise ValueError(f"Unknown method: {method}. Use 'NPE', 'NLE', or 'NRE'")
    
    if density_estimator is None:
        density_estimator = default_estimator
    
    # Build density estimator config
    if method in ['NPE', 'NLE']:
        if density_estimator in ['maf', 'nsf']:
            estimator_kwargs = {
                'hidden_features': hidden_features,
                'num_transforms': num_transforms,
                **kwargs
            }
        else:
            estimator_kwargs = {'hidden_features': hidden_features, **kwargs}
    else:  # NRE
        estimator_kwargs = {'hidden_features': hidden_features, **kwargs}
    
    # Initialize inference
    inference = InferenceClass(
        prior=sbi_prior,
        density_estimator=density_estimator,
        device=device,
        show_progress_bars=show_progress
    )
    
    # Simulation and training
    all_theta = []
    all_x = []
    sims_per_round = n_simulations // n_rounds
    
    proposal = sbi_prior
    
    for round_idx in range(n_rounds):
        if show_progress:
            print(f"Round {round_idx + 1}/{n_rounds}: Simulating {sims_per_round} samples...")
        
        # Sample from proposal
        theta = proposal.sample((sims_per_round,))
        
        # Simulate
        x = sim_wrapped(theta)
        
        # Handle NaN/inf in simulations
        valid_mask = torch.isfinite(x).all(dim=-1)
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum().item()
            warnings.warn(f"Removed {n_invalid} simulations with NaN/inf values")
            theta = theta[valid_mask]
            x = x[valid_mask]
        
        all_theta.append(theta)
        all_x.append(x)
        
        # Append simulations
        inference.append_simulations(theta, x, proposal=proposal)
        
        # Train
        if show_progress:
            print(f"Round {round_idx + 1}/{n_rounds}: Training...")
        
        density_est = inference.train(
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            stop_after_epochs=stop_after_epochs,
            validation_fraction=validation_fraction,
            show_train_summary=show_progress
        )
        
        # Build posterior for next round
        if round_idx < n_rounds - 1:
            if observed_stats is None:
                raise ValueError("observed_stats required for multi-round inference")
            posterior = inference.build_posterior(density_est)
            proposal = posterior.set_default_x(observed_stats)
    
    # Build final posterior
    posterior = inference.build_posterior(density_est)
    
    training_time = time.time() - start_time
    
    # Store results
    theta_train = torch.cat(all_theta, dim=0) if store_simulations else None
    x_train = torch.cat(all_x, dim=0) if store_simulations else None
    
    return SBIResult(
        posterior=posterior,
        inference=inference,
        density_estimator=density_est,
        method=method,
        n_simulations=n_simulations,
        n_rounds=n_rounds,
        training_time=training_time,
        theta_train=theta_train,
        x_train=x_train,
        prior=prior,
        observed_stats=observed_stats,
        param_names=param_names
    )


# =============================================================================
# POSTERIOR SAMPLING
# =============================================================================

def sample_posterior(
    posterior: Any,
    observed_stats: Union[np.ndarray, torch.Tensor],
    n_samples: int = 10000,
    method: str = 'direct',
    thin: int = 1,
    warmup: int = 200,
    num_chains: int = 1,
    show_progress: bool = True
) -> torch.Tensor:
    """
    Sample from trained posterior.
    
    Args:
        posterior: Trained posterior (from SBIResult or sbi directly)
        observed_stats: Observed data summary statistics
        n_samples: Number of samples to draw
        method: Sampling method
                'direct': Direct sampling (for NPE)
                'rejection': Rejection sampling
                'mcmc': MCMC sampling (slower but can be more accurate)
                'vi': Variational inference
        thin: Thinning factor for MCMC
        warmup: Warmup samples for MCMC
        num_chains: Number of MCMC chains
        show_progress: Show progress bar
    
    Returns:
        Samples tensor of shape (n_samples, n_params)
    """
    # Handle SBIResult
    if isinstance(posterior, SBIResult):
        posterior = posterior.posterior
    
    # Convert observed_stats
    if isinstance(observed_stats, np.ndarray):
        observed_stats = torch.tensor(observed_stats, dtype=torch.float32)
    
    if method == 'direct':
        samples = posterior.sample((n_samples,), x=observed_stats, 
                                   show_progress_bars=show_progress)
    
    elif method == 'rejection':
        samples = posterior.sample(
            (n_samples,), x=observed_stats,
            sample_with='rejection',
            show_progress_bars=show_progress
        )
    
    elif method == 'mcmc':
        samples = posterior.sample(
            (n_samples,), x=observed_stats,
            sample_with='mcmc',
            mcmc_method='slice_np',
            thin=thin,
            warmup_steps=warmup,
            num_chains=num_chains,
            show_progress_bars=show_progress
        )
    
    elif method == 'vi':
        samples = posterior.sample(
            (n_samples,), x=observed_stats,
            sample_with='vi',
            show_progress_bars=show_progress
        )
    
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    return samples


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_posterior(
    simulator: Callable,
    prior: Any,
    observed_stats: Union[np.ndarray, torch.Tensor],
    n_simulations: int = 20000,
    method: str = 'NPE',
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, SBIResult]:
    """
    Quick posterior estimation with sensible defaults.
    
    For rapid prototyping and testing.
    
    Args:
        simulator: Simulator callable
        prior: Prior distribution
        observed_stats: Observed summary statistics
        n_simulations: Number of simulations
        method: SBI method
        seed: Random seed
    
    Returns:
        (samples, result) where samples is (10000, n_params) tensor
    """
    result = train_sbi(
        simulator=simulator,
        prior=prior,
        observed_stats=observed_stats,
        method=method,
        n_simulations=n_simulations,
        n_rounds=1,
        seed=seed,
        show_progress=True
    )
    
    samples = sample_posterior(result.posterior, observed_stats, n_samples=10000)
    
    return samples, result


def compare_methods(
    simulator: Callable,
    prior: Any,
    observed_stats: Union[np.ndarray, torch.Tensor],
    methods: List[str] = ['NPE', 'NLE', 'NRE'],
    n_simulations: int = 30000,
    seed: Optional[int] = None
) -> Dict[str, SBIResult]:
    """
    Compare different SBI methods on the same problem.
    
    Args:
        simulator: Simulator callable
        prior: Prior distribution  
        observed_stats: Observed summary statistics
        methods: List of methods to compare
        n_simulations: Number of simulations per method
        seed: Random seed
    
    Returns:
        Dict mapping method names to SBIResult objects
    """
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Training {method}...")
        print(f"{'='*60}")
        
        method_seed = seed + hash(method) % 10000 if seed else None
        
        results[method] = train_sbi(
            simulator=simulator,
            prior=prior,
            observed_stats=observed_stats,
            method=method,
            n_simulations=n_simulations,
            seed=method_seed,
            show_progress=True
        )
        
        print(f"{method} training time: {results[method].training_time:.1f}s")
    
    return results


# =============================================================================
# MULTI-SESSION INFERENCE
# =============================================================================

def train_multisession_sbi(
    simulator: Callable,
    prior: Any,  # MultiSessionPrior
    observed_stats: Union[np.ndarray, torch.Tensor],
    method: str = 'NPE',
    n_simulations: int = 50000,
    **kwargs
) -> SBIResult:
    """
    Train SBI for multi-session inference.
    
    This is a thin wrapper around train_sbi that handles MultiSessionPrior
    and provides appropriate defaults.
    
    Args:
        simulator: Multi-session simulator (returns concatenated summary stats)
        prior: MultiSessionPrior instance
        observed_stats: Concatenated observed summary stats from all sessions
        method: SBI method
        n_simulations: Number of simulations
        **kwargs: Additional arguments to train_sbi
    
    Returns:
        SBIResult with posterior over all parameters
    """
    # For multi-session, we may need more hidden features
    kwargs.setdefault('hidden_features', 100)
    kwargs.setdefault('num_transforms', 8)
    
    return train_sbi(
        simulator=simulator,
        prior=prior,
        observed_stats=observed_stats,
        method=method,
        n_simulations=n_simulations,
        param_names=prior.param_names if hasattr(prior, 'param_names') else None,
        **kwargs
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SBIResult',
    'train_sbi',
    'sample_posterior',
    'quick_posterior',
    'compare_methods',
    'train_multisession_sbi',
]
