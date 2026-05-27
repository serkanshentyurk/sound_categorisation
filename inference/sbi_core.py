"""
SBI core training and result container.

Low-level SBI training functions and the SBIResult dataclass that holds
trained posteriors. Used by both SBIFitter (high-level per-animal API)
and train_per_animal_snpe (script-style entry point).

Public API:
    SBIResult       — Container for trained posteriors
    train_sbi       — Core SBI training loop (NPE/NLE/NRE)
    sample_posterior — Posterior sampling with multiple methods
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field

# Lazy torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# RESULT CONTAINER
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
        param_names: Parameter names for labelling
    """
    posterior: Any
    inference: Any
    density_estimator: Any
    method: str
    n_simulations: int
    n_rounds: int
    training_time: float
    theta_train: Optional[Any] = None  # torch.Tensor if stored
    x_train: Optional[Any] = None
    prior: Any = None
    observed_stats: Optional[Any] = None
    param_names: Optional[List[str]] = None

    def sample(self, n_samples: int, x: Optional[Any] = None) -> Any:
        """Sample from posterior. Returns tensor (n_samples, n_params)."""
        if x is None:
            x = self.observed_stats
        if x is None:
            raise ValueError("Must provide observed_stats (x)")
        return self.posterior.sample((n_samples,), x=x)

    def log_prob(self, theta: Any, x: Optional[Any] = None) -> Any:
        """Evaluate posterior log probability."""
        if x is None:
            x = self.observed_stats
        if x is None:
            raise ValueError("Must provide observed_stats (x)")
        return self.posterior.log_prob(theta, x=x)

    def map_estimate(self, x: Optional[Any] = None, n_samples: int = 10000) -> Any:
        """Approximate MAP by finding highest-density sample."""
        samples = self.sample(n_samples, x=x)
        log_probs = self.log_prob(samples, x=x)
        return samples[log_probs.argmax()]

    def posterior_mean(self, x: Optional[Any] = None, n_samples: int = 10000) -> Any:
        """Posterior mean estimate."""
        return self.sample(n_samples, x=x).mean(dim=0)

    def credible_interval(self, x: Optional[Any] = None,
                          level: float = 0.95, n_samples: int = 10000) -> Tuple:
        """Get (lower, upper) credible interval tensors."""
        import torch
        samples = self.sample(n_samples, x=x)
        alpha = (1 - level) / 2
        return (torch.quantile(samples, alpha, dim=0),
                torch.quantile(samples, 1 - alpha, dim=0))

    def summary(self, x: Optional[Any] = None, n_samples: int = 10000) -> Dict[str, Any]:
        """Posterior summary: mean, std, median, 95% CI per parameter."""
        import torch
        samples = self.sample(n_samples, x=x)
        lower, upper = self.credible_interval(x=x, n_samples=n_samples)
        names = self.param_names or [f'param_{i}' for i in range(samples.shape[1])]
        return {
            name: {
                'mean': float(samples[:, i].mean()),
                'std': float(samples[:, i].std()),
                'median': float(samples[:, i].median()),
                'ci_lower': float(lower[i]),
                'ci_upper': float(upper[i]),
            }
            for i, name in enumerate(names)
        }


# =============================================================================
# SIMULATOR WRAPPING
# =============================================================================

def _wrap_simulator_for_sbi(simulator: Callable, seed_offset: int = 0) -> Callable:
    """Wrap a numpy simulator for sbi (handles torch tensors)."""
    import torch

    def wrapped(theta):
        theta_np = theta.numpy() if hasattr(theta, 'numpy') else np.asarray(theta)
        if theta_np.ndim == 1:
            result = simulator(theta_np)
            return torch.tensor(result, dtype=torch.float32)
        else:
            results = []
            for i in range(len(theta_np)):
                if hasattr(simulator, 'simulate'):
                    result = simulator.simulate(theta_np[i], seed=seed_offset + i)
                else:
                    result = simulator(theta_np[i])
                results.append(result)
            return torch.tensor(np.stack(results), dtype=torch.float32)

    return wrapped

# =============================================================================
# CORE SBI TRAINING 
# =============================================================================

def train_sbi(
    simulator: Callable,
    prior: Any,
    observed_stats: Optional[Union[np.ndarray, Any]] = None,
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
    **kwargs,
) -> SBIResult:
    """
    Train SBI model and return posterior.

    Args:
        simulator: Callable: parameter array → summary stats array
        prior: Prior distribution (UniformPrior, MultiSessionPrior, or sbi prior)
        observed_stats: Observed summary statistics (required for n_rounds > 1)
        method: 'NPE', 'NLE', or 'NRE'
        n_simulations: Total number of simulations
        n_rounds: Sequential inference rounds (1 = amortised)
        training_batch_size: Batch size
        learning_rate: Learning rate
        hidden_features: Hidden layer size
        num_transforms: Normalising flow transforms
        stop_after_epochs: Early stopping patience
        validation_fraction: Validation fraction
        density_estimator: 'maf', 'nsf', 'mdn', 'resnet', 'mlp' (None = auto)
        device: 'cpu' or 'cuda'
        seed: Random seed
        store_simulations: Store training theta/x in result
        show_progress: Show progress bars
        param_names: Parameter names for labelling
        **kwargs: Passed to density estimator

    Returns:
        SBIResult with trained posterior
    """
    import torch
    from sbi.inference import SNPE, SNLE, SNRE
    from sbi.utils import BoxUniform
    try:
        from sbi.utils.user_input_checks import process_prior
    except ImportError:
        from sbi.utils import process_prior

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    start_time = time.time()

    # Convert observed_stats
    if observed_stats is not None:
        if isinstance(observed_stats, np.ndarray):
            observed_stats = torch.tensor(observed_stats, dtype=torch.float32)
        observed_stats = observed_stats.to(device)

    # Setup prior
    if hasattr(prior, 'low') and hasattr(prior, 'high'):
        sbi_prior = BoxUniform(low=prior.low.to(device), high=prior.high.to(device))
        if param_names is None and hasattr(prior, 'param_names'):
            param_names = prior.param_names
    elif hasattr(prior, 'sample') and hasattr(prior, 'log_prob'):
        result = process_prior(prior)
        sbi_prior = result[0] if isinstance(result, tuple) else result
        if param_names is None and hasattr(prior, 'param_names'):
            param_names = prior.param_names
    else:
        sbi_prior = prior

    sim_wrapped = _wrap_simulator_for_sbi(simulator, seed_offset=seed or 0)

    method = method.upper()
    InferenceClass = {'NPE': SNPE, 'NLE': SNLE, 'NRE': SNRE}.get(method)
    if InferenceClass is None:
        raise ValueError(f"Unknown method: {method}. Use 'NPE', 'NLE', or 'NRE'")

    default_est = {'NPE': 'maf', 'NLE': 'maf', 'NRE': 'resnet'}[method]
    if density_estimator is None:
        density_estimator = default_est

    if method in ['NPE', 'NLE'] and density_estimator in ['maf', 'nsf']:
        est_kwargs = {'hidden_features': hidden_features,
                      'num_transforms': num_transforms, **kwargs}
    else:
        est_kwargs = {'hidden_features': hidden_features, **kwargs}

    inference = InferenceClass(
        prior=sbi_prior, density_estimator=density_estimator,
        device=device, show_progress_bars=show_progress,
    )

    all_theta, all_x = [], []
    sims_per_round = n_simulations // n_rounds
    proposal = sbi_prior

    for round_idx in range(n_rounds):
        if show_progress:
            print(f"Round {round_idx + 1}/{n_rounds}: Simulating {sims_per_round}...")

        theta = proposal.sample((sims_per_round,))
        x = sim_wrapped(theta)

        valid_mask = torch.isfinite(x).all(dim=-1)
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum().item()
            warnings.warn(f"Removed {n_invalid} simulations with NaN/inf")
            theta, x = theta[valid_mask], x[valid_mask]

        all_theta.append(theta)
        all_x.append(x)
        inference.append_simulations(theta, x, proposal=proposal)

        if show_progress:
            print(f"Round {round_idx + 1}/{n_rounds}: Training...")

        density_est = inference.train(
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            stop_after_epochs=stop_after_epochs,
            validation_fraction=validation_fraction,
            show_train_summary=show_progress,
        )

        if round_idx < n_rounds - 1:
            if observed_stats is None:
                raise ValueError("observed_stats required for multi-round inference")
            posterior = inference.build_posterior(density_est)
            proposal = posterior.set_default_x(observed_stats)

    posterior = inference.build_posterior(density_est)
    training_time = time.time() - start_time

    return SBIResult(
        posterior=posterior, inference=inference,
        density_estimator=density_est, method=method,
        n_simulations=n_simulations, n_rounds=n_rounds,
        training_time=training_time,
        theta_train=torch.cat(all_theta) if store_simulations else None,
        x_train=torch.cat(all_x) if store_simulations else None,
        prior=prior, observed_stats=observed_stats,
        param_names=param_names,
    )


# =============================================================================
# POSTERIOR SAMPLING 
# =============================================================================

def sample_posterior(
    posterior: Any,
    observed_stats: Union[np.ndarray, Any],
    n_samples: int = 10000,
    method: str = 'direct',
    thin: int = 1,
    warmup: int = 200,
    num_chains: int = 1,
    show_progress: bool = True,
) -> Any:
    """
    Sample from trained posterior.

    Args:
        posterior: Trained posterior (SBIResult or sbi posterior)
        observed_stats: Observed summary statistics
        n_samples: Number of samples
        method: 'direct', 'rejection', 'mcmc', 'vi'

    Returns:
        Samples tensor (n_samples, n_params)
    """
    import torch

    if isinstance(posterior, SBIResult):
        posterior = posterior.posterior
    if isinstance(observed_stats, np.ndarray):
        observed_stats = torch.tensor(observed_stats, dtype=torch.float32)

    sample_kwargs = {'show_progress_bars': show_progress}
    if method == 'direct':
        return posterior.sample((n_samples,), x=observed_stats, **sample_kwargs)
    elif method == 'rejection':
        return posterior.sample((n_samples,), x=observed_stats,
                                sample_with='rejection', **sample_kwargs)
    elif method == 'mcmc':
        return posterior.sample((n_samples,), x=observed_stats,
                                sample_with='mcmc', mcmc_method='slice_np',
                                thin=thin, warmup_steps=warmup,
                                num_chains=num_chains, **sample_kwargs)
    elif method == 'vi':
        return posterior.sample((n_samples,), x=observed_stats,
                                sample_with='vi', **sample_kwargs)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


# =============================================================================
# PRIOR BUILDER 
# =============================================================================
