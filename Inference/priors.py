"""
Prior Definitions for SBI.

Provides flexible prior structures for:
- Single-session inference (uniform priors)
- Multi-session inference with various linking functions

Linking Functions:
    - 'independent': Each session has independent prior
    - 'gp': Gaussian Process prior over sessions
    - 'random_walk': Random walk with drift parameter
    - 'hierarchical': Hierarchical prior with group-level parameters
    - Custom callable

Usage:
    # Single session
    prior = create_prior(param_bounds)
    
    # Multi-session with GP-linked eta_learning
    prior = create_multisession_prior(
        param_bounds=param_bounds,
        n_sessions=10,
        varying_params=['eta_learning'],
        linking_fn='gp',
        linking_config={'lengthscale': 5.0, 'amplitude': 0.1}
    )

Note on naming:
    This module defines torch implementation classes (IndependentLink,
    RandomWalkLink, GaussianProcessLink, HierarchicalLink) that share
    names with the specification dataclasses in inference/types.py
    (IndependentSpec, RandomWalkSpec, GPSpec, HierarchicalSpec).
    
    The specs are lightweight, frozen descriptions of desired behaviour.
    The classes here are the torch implementations with sample()/log_prob().
    
    Users interact with the *Spec types from types.py.
    The build_prior() function in fitting.py bridges specs → implementations.
"""

import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution, Uniform, Normal, MultivariateNormal
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from inference.types import ModelType, ParamConfig

# =============================================================================
# PARAMETER BOUNDS
# =============================================================================

from inference.types import get_default_param_configs, ModelType
_be_configs = get_default_param_configs(ModelType.BE)
DEFAULT_BE_BOUNDS = {name: cfg.bounds for name, cfg in _be_configs.items()}


# =============================================================================
# BASE PRIOR CLASS
# =============================================================================

class BasePrior(ABC):
    """
    Abstract base class for SBI-compatible priors.
    
    Must implement:
        - sample(n): Draw n samples, returns tensor of shape (n, dim)
        - log_prob(theta): Compute log probability, returns tensor of shape (n,)
    """
    
    @abstractmethod
    def sample(self, sample_shape: Tuple[int, ...] = (1,)) -> torch.Tensor:
        """Draw samples from prior."""
        pass
    
    @abstractmethod
    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute log probability of theta."""
        pass
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of parameter space."""
        pass
    
    @property
    @abstractmethod
    def param_names(self) -> List[str]:
        """Names of parameters in order."""
        pass


# =============================================================================
# SINGLE-SESSION UNIFORM PRIOR
# =============================================================================

class UniformPrior(BasePrior):
    """
    Uniform (box) prior for single-session inference.
    
    Compatible with sbi's BoxUniform interface.
    
    Args:
        bounds: Dict mapping param names to (low, high) bounds
        param_order: Optional list specifying parameter order
                    (default: sorted keys)
    
    Example:
        prior = UniformPrior({
            'sigma_percep': (0.05, 0.5),
            'eta_learning': (0.05, 0.9),
        })
        samples = prior.sample((1000,))
    """
    
    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        param_order: Optional[List[str]] = None
    ):
        self.bounds = bounds
        self._param_order = param_order or sorted(bounds.keys())
        
        # Validate param_order
        if set(self._param_order) != set(bounds.keys()):
            raise ValueError("param_order must contain exactly the keys in bounds")
        
        # Build tensors
        self._low = torch.tensor(
            [bounds[p][0] for p in self._param_order],
            dtype=torch.float32
        )
        self._high = torch.tensor(
            [bounds[p][1] for p in self._param_order],
            dtype=torch.float32
        )
        
        # Uniform distribution
        self._dist = Uniform(self._low, self._high)
    
    def sample(self, sample_shape: Tuple[int, ...] = (1,)) -> torch.Tensor:
        """Draw samples from uniform prior."""
        # Handle tuple or int
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        return self._dist.sample(sample_shape)
    
    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute log probability (constant within bounds, -inf outside)."""
        # Sum log probs across dimensions
        return self._dist.log_prob(theta).sum(dim=-1)
    
    @property
    def dim(self) -> int:
        return len(self._param_order)
    
    @property
    def param_names(self) -> List[str]:
        return self._param_order.copy()
    
    @property
    def low(self) -> torch.Tensor:
        return self._low
    
    @property
    def high(self) -> torch.Tensor:
        return self._high
    
    def get_bounds_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) bounds as numpy arrays."""
        return self._low.numpy(), self._high.numpy()
    
    def to_sbi_prior(self):
        """
        Convert to sbi BoxUniform prior.
        
        Returns:
            sbi.utils.BoxUniform instance
        """
        try:
            from sbi.utils import BoxUniform
        except ImportError:
            raise ImportError("sbi package required. Install with: pip install sbi")
        
        return BoxUniform(low=self._low, high=self._high)


# =============================================================================
# LINKING FUNCTIONS FOR MULTI-SESSION
# =============================================================================



@dataclass
class LinkingConfig:
    """Configuration for linking function."""
    link_type: str  # 'independent', 'gp', 'random_walk', 'hierarchical', 'custom'
    params: Dict[str, Any] = field(default_factory=dict)
    custom_fn: Optional[Callable] = None


class IndependentLink:
    """
    Independent prior for each session (no temporal structure).
    
    Each session's parameter is drawn independently from the base prior.
    """
    
    def __init__(self, base_bounds: Tuple[float, float], n_sessions: int):
        self.base_bounds = base_bounds
        self.n_sessions = n_sessions
        self.low = base_bounds[0]
        self.high = base_bounds[1]
    
    def sample(self, n_samples: int, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample (n_samples, n_sessions) tensor."""
        return torch.rand(n_samples, self.n_sessions, generator=rng) * (self.high - self.low) + self.low
    
    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Log probability of theta (shape: batch x n_sessions).
        
        Returns log prob summed across sessions.
        """
        # Uniform: log(1/(high-low)) = -log(high-low) for each session
        in_bounds = (theta >= self.low) & (theta <= self.high)
        log_p = torch.where(
            in_bounds,
            torch.tensor(-np.log(self.high - self.low)),
            torch.tensor(float('-inf'))
        )
        return log_p.sum(dim=-1)
    
    @property
    def dim(self) -> int:
        return self.n_sessions
    
    @property
    def hyperparameter_names(self) -> List[str]:
        return []


class RandomWalkLink:
    """
    Random walk prior across sessions.
    
    Î¸_0 ~ Uniform(low, high)
    Î¸_{t+1} | Î¸_t ~ N(Î¸_t, Ïƒ_driftÂ²)  truncated to [low, high]
    
    Args:
        base_bounds: (low, high) for the parameter
        n_sessions: Number of sessions
        sigma_drift: Standard deviation of random walk steps
        infer_drift: If True, sigma_drift becomes an inferred hyperparameter
    """
    
    def __init__(
        self,
        base_bounds: Tuple[float, float],
        n_sessions: int,
        sigma_drift: float = 0.05,
        infer_drift: bool = False,
        drift_bounds: Tuple[float, float] = (0.01, 0.3)
    ):
        self.base_bounds = base_bounds
        self.n_sessions = n_sessions
        self.sigma_drift = sigma_drift
        self.infer_drift = infer_drift
        self.drift_bounds = drift_bounds
        self.low = base_bounds[0]
        self.high = base_bounds[1]
    
    def sample(self, n_samples: int, sigma_drift: Optional[torch.Tensor] = None,
               rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Sample trajectories.
        
        Args:
            n_samples: Number of trajectories to sample
            sigma_drift: If provided, use this drift (shape: n_samples,)
            rng: Random generator
        
        Returns:
            Tensor of shape (n_samples, n_sessions)
        """
        if sigma_drift is None:
            sigma_drift = torch.full((n_samples,), self.sigma_drift)
        
        trajectories = torch.zeros(n_samples, self.n_sessions)
        
        # Initial value from uniform
        trajectories[:, 0] = torch.rand(n_samples, generator=rng) * (self.high - self.low) + self.low
        
        # Random walk
        for t in range(1, self.n_sessions):
            noise = torch.randn(n_samples, generator=rng) * sigma_drift
            trajectories[:, t] = trajectories[:, t-1] + noise
            # Reflect at boundaries
            trajectories[:, t] = torch.clamp(trajectories[:, t], self.low, self.high)
        
        return trajectories
    
    def log_prob(self, theta: torch.Tensor, sigma_drift: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute log probability of trajectory.
        
        This is approximate (ignores truncation correction) but works for SBI.
        """
        if sigma_drift is None:
            sigma_drift = torch.full((theta.shape[0],), self.sigma_drift)
        
        # Check bounds
        in_bounds = (theta >= self.low) & (theta <= self.high)
        if not in_bounds.all():
            return torch.full((theta.shape[0],), float('-inf'))
        
        # Initial: uniform
        log_p = -np.log(self.high - self.low)
        
        # Transitions: Gaussian (ignoring truncation)
        for t in range(1, self.n_sessions):
            diff = theta[:, t] - theta[:, t-1]
            log_p = log_p - 0.5 * (diff / sigma_drift) ** 2 - torch.log(sigma_drift) - 0.5 * np.log(2 * np.pi)
        
        return log_p
    
    @property
    def dim(self) -> int:
        return self.n_sessions + (1 if self.infer_drift else 0)
    
    @property
    def hyperparameter_names(self) -> List[str]:
        return ['sigma_drift'] if self.infer_drift else []

class GaussianProcessLink:
    """
    Gaussian Process prior over sessions.
    
    Provides smooth trajectories with controllable correlation structure.
    
    Î¸ ~ GP(mean_fn, kernel)
    
    Default kernel: RBF (squared exponential)
    k(t, t') = amplitudeÂ² * exp(-0.5 * (t-t')Â² / lengthscaleÂ²)
    
    Args:
        base_bounds: (low, high) for the parameter
        n_sessions: Number of sessions
        lengthscale: GP lengthscale (in session units)
        amplitude: GP amplitude (std of function values)
        mean: Mean function value
        infer_hyperparams: If True, lengthscale/amplitude become inferred
    """
    
    def __init__(
        self,
        base_bounds: Tuple[float, float],
        n_sessions: int,
        lengthscale: float = 5.0,
        amplitude: float = 0.1,
        mean: Optional[float] = None,
        infer_hyperparams: bool = False,
        lengthscale_bounds: Tuple[float, float] = (1.0, 20.0),
        amplitude_bounds: Tuple[float, float] = (0.01, 0.3)
    ):
        self.base_bounds = base_bounds
        self.n_sessions = n_sessions
        self.lengthscale = lengthscale
        self.amplitude = amplitude
        self.mean = mean if mean is not None else (base_bounds[0] + base_bounds[1]) / 2
        self.infer_hyperparams = infer_hyperparams
        self.lengthscale_bounds = lengthscale_bounds
        self.amplitude_bounds = amplitude_bounds
        self.low = base_bounds[0]
        self.high = base_bounds[1]
        
        # Session indices
        self.t = torch.arange(n_sessions, dtype=torch.float32)
    
    def _build_covariance(self, lengthscale: float, amplitude: float) -> torch.Tensor:
        """Build RBF covariance matrix."""
        t1 = self.t.unsqueeze(0)  # (1, n_sessions)
        t2 = self.t.unsqueeze(1)  # (n_sessions, 1)
        sq_dist = (t1 - t2) ** 2
        K = amplitude ** 2 * torch.exp(-0.5 * sq_dist / lengthscale ** 2)
        # Add jitter for numerical stability
        K = K + 1e-6 * torch.eye(self.n_sessions)
        return K
    
    def sample(self, n_samples: int, lengthscale: Optional[torch.Tensor] = None,
               amplitude: Optional[torch.Tensor] = None,
               rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Sample GP trajectories.
        
        Returns:
            Tensor of shape (n_samples, n_sessions)
        """
        ls = lengthscale if lengthscale is not None else self.lengthscale
        amp = amplitude if amplitude is not None else self.amplitude
        
        # Handle scalar vs tensor hyperparams
        if isinstance(ls, (int, float)):
            ls = float(ls)
            amp = float(amp)
            K = self._build_covariance(ls, amp)
            mean_vec = torch.full((self.n_sessions,), self.mean)
            mvn = MultivariateNormal(mean_vec, K)
            samples = mvn.sample((n_samples,))
        else:
            # Different hyperparams per sample - sample one at a time
            samples = torch.zeros(n_samples, self.n_sessions)
            for i in range(n_samples):
                K = self._build_covariance(float(ls[i]), float(amp[i]))
                mean_vec = torch.full((self.n_sessions,), self.mean)
                mvn = MultivariateNormal(mean_vec, K)
                samples[i] = mvn.sample()
        
        # Clamp to bounds
        samples = torch.clamp(samples, self.low, self.high)
        
        return samples
    
    def log_prob(self, theta: torch.Tensor, lengthscale: Optional[torch.Tensor] = None,
                 amplitude: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute log probability under GP prior.
        
        Note: This ignores the truncation to bounds (approximate).
        """
        ls = lengthscale if lengthscale is not None else self.lengthscale
        amp = amplitude if amplitude is not None else self.amplitude
        
        # Check bounds
        in_bounds = (theta >= self.low) & (theta <= self.high)
        if not in_bounds.all():
            return torch.full((theta.shape[0],), float('-inf'))
        
        if isinstance(ls, (int, float)):
            K = self._build_covariance(float(ls), float(amp))
            mean_vec = torch.full((self.n_sessions,), self.mean)
            mvn = MultivariateNormal(mean_vec, K)
            return mvn.log_prob(theta)
        else:
            # Different hyperparams per sample
            log_probs = torch.zeros(theta.shape[0])
            for i in range(theta.shape[0]):
                K = self._build_covariance(float(ls[i]), float(amp[i]))
                mean_vec = torch.full((self.n_sessions,), self.mean)
                mvn = MultivariateNormal(mean_vec, K)
                log_probs[i] = mvn.log_prob(theta[i])
            return log_probs
    
    @property
    def dim(self) -> int:
        return self.n_sessions + (2 if self.infer_hyperparams else 0)
    
    @property
    def hyperparameter_names(self) -> List[str]:
        return ['lengthscale', 'amplitude'] if self.infer_hyperparams else []


class HierarchicalLink:
    """
    Hierarchical prior across sessions.
    
    Î¸_session ~ N(Î¼_group, Ïƒ_groupÂ²)  truncated to [low, high]
    
    Useful when sessions are exchangeable (no temporal order).
    
    Args:
        base_bounds: (low, high) for the parameter
        n_sessions: Number of sessions
        group_mean: Mean of group-level distribution
        group_std: Std of group-level distribution
        infer_hyperparams: If True, mean/std become inferred
    """
    
    def __init__(
        self,
        base_bounds: Tuple[float, float],
        n_sessions: int,
        group_mean: Optional[float] = None,
        group_std: float = 0.1,
        infer_hyperparams: bool = False,
        mean_bounds: Optional[Tuple[float, float]] = None,
        std_bounds: Tuple[float, float] = (0.01, 0.3)
    ):
        self.base_bounds = base_bounds
        self.n_sessions = n_sessions
        self.group_mean = group_mean if group_mean is not None else (base_bounds[0] + base_bounds[1]) / 2
        self.group_std = group_std
        self.infer_hyperparams = infer_hyperparams
        self.mean_bounds = mean_bounds or base_bounds
        self.std_bounds = std_bounds
        self.low = base_bounds[0]
        self.high = base_bounds[1]
    
    def sample(self, n_samples: int, group_mean: Optional[torch.Tensor] = None,
               group_std: Optional[torch.Tensor] = None,
               rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample from hierarchical prior."""
        mean = group_mean if group_mean is not None else self.group_mean
        std = group_std if group_std is not None else self.group_std
        
        if isinstance(mean, (int, float)):
            samples = torch.randn(n_samples, self.n_sessions, generator=rng) * std + mean
        else:
            # Different hyperparams per sample
            samples = torch.randn(n_samples, self.n_sessions, generator=rng)
            samples = samples * std.unsqueeze(1) + mean.unsqueeze(1)
        
        # Clamp to bounds
        samples = torch.clamp(samples, self.low, self.high)
        
        return samples
    
    def log_prob(self, theta: torch.Tensor, group_mean: Optional[torch.Tensor] = None,
                 group_std: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute log probability (approximate, ignores truncation)."""
        mean = group_mean if group_mean is not None else self.group_mean
        std = group_std if group_std is not None else self.group_std
        
        # Check bounds
        in_bounds = (theta >= self.low) & (theta <= self.high)
        if not in_bounds.all():
            return torch.full((theta.shape[0],), float('-inf'))
        
        if isinstance(mean, (int, float)):
            dist = Normal(mean, std)
            return dist.log_prob(theta).sum(dim=-1)
        else:
            log_probs = torch.zeros(theta.shape[0])
            for i in range(theta.shape[0]):
                dist = Normal(mean[i], std[i])
                log_probs[i] = dist.log_prob(theta[i]).sum()
            return log_probs
    
    @property
    def dim(self) -> int:
        return self.n_sessions + (2 if self.infer_hyperparams else 0)
    
    @property
    def hyperparameter_names(self) -> List[str]:
        return ['group_mean', 'group_std'] if self.infer_hyperparams else []


# =============================================================================
# MULTI-SESSION PRIOR
# =============================================================================

class MultiSessionPrior(BasePrior):
    """
    Prior for multi-session inference with flexible linking.
    
    Combines:
    - Shared parameters: Same value across all sessions
    - Varying parameters: Different value per session, linked by specified function
    
    Args:
        param_bounds: Dict of all parameter bounds
        n_sessions: Number of sessions
        varying_params: List of parameters that vary across sessions
        linking_configs: Dict mapping varying param names to LinkingConfig
                        (if not specified, uses 'independent' linking)
        param_order: Optional parameter ordering
    
    Example:
        prior = MultiSessionPrior(
            param_bounds=DEFAULT_BE_BOUNDS,
            n_sessions=10,
            varying_params=['eta_learning'],
            linking_configs={
                'eta_learning': LinkingConfig(
                    link_type='gp',
                    params={'lengthscale': 5.0, 'amplitude': 0.1}
                )
            }
        )
    """
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_sessions: int,
        varying_params: Optional[List[str]] = None,
        linking_configs: Optional[Dict[str, LinkingConfig]] = None,
        param_order: Optional[List[str]] = None
    ):
        self.param_bounds = param_bounds
        self.n_sessions = n_sessions
        self.varying_params = varying_params or []
        self.linking_configs = linking_configs or {}
        
        # Determine parameter order
        all_params = param_order or sorted(param_bounds.keys())
        self._shared_params = [p for p in all_params if p not in self.varying_params]
        self._varying_params = [p for p in all_params if p in self.varying_params]
        
        # Build linking functions
        self._linkers: Dict[str, Any] = {}
        for param in self._varying_params:
            config = self.linking_configs.get(param, LinkingConfig(link_type='independent'))
            self._linkers[param] = self._create_linker(param, config)
        
        # Compute total dimension
        self._dim = len(self._shared_params)  # Shared params
        for param in self._varying_params:
            self._dim += self._linkers[param].dim
        
        # Build param names list
        self._param_names = []
        for p in self._shared_params:
            self._param_names.append(p)
        for p in self._varying_params:
            linker = self._linkers[p]
            for s in range(self.n_sessions):
                self._param_names.append(f"{p}_{s}")
            for hp in linker.hyperparameter_names:
                self._param_names.append(f"{p}_{hp}")
    
    def _create_linker(self, param: str, config: LinkingConfig):
        """Create appropriate linker based on config."""
        bounds = self.param_bounds[param]
        
        if config.link_type == 'independent':
            return IndependentLink(bounds, self.n_sessions)
        
        elif config.link_type == 'random_walk':
            return RandomWalkLink(
                bounds, self.n_sessions,
                sigma_drift=config.params.get('sigma_drift', 0.05),
                infer_drift=config.params.get('infer_drift', False),
                drift_bounds=config.params.get('drift_bounds', (0.01, 0.3))
            )
        
        elif config.link_type == 'gp':
            return GaussianProcessLink(
                bounds, self.n_sessions,
                lengthscale=config.params.get('lengthscale', 5.0),
                amplitude=config.params.get('amplitude', 0.1),
                mean=config.params.get('mean', None),
                infer_hyperparams=config.params.get('infer_hyperparams', False),
                lengthscale_bounds=config.params.get('lengthscale_bounds', (1.0, 20.0)),
                amplitude_bounds=config.params.get('amplitude_bounds', (0.01, 0.3))
            )
        
        elif config.link_type == 'hierarchical':
            return HierarchicalLink(
                bounds, self.n_sessions,
                group_mean=config.params.get('group_mean', None),
                group_std=config.params.get('group_std', 0.1),
                infer_hyperparams=config.params.get('infer_hyperparams', False),
                mean_bounds=config.params.get('mean_bounds', None),
                std_bounds=config.params.get('std_bounds', (0.01, 0.3))
            )
        
        elif config.link_type == 'custom':
            if config.custom_fn is None:
                raise ValueError("custom_fn required for 'custom' link_type")
            return config.custom_fn(bounds, self.n_sessions, **config.params)
        
        else:
            raise ValueError(f"Unknown link_type: {config.link_type}")
    
    def sample(self, sample_shape: Tuple[int, ...] = (1,)) -> torch.Tensor:
        """Sample from multi-session prior."""
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        # SBI calls sample(torch.Size()) to check dtype — treat empty shape as 1
        n_samples = sample_shape[0] if len(sample_shape) > 0 else 1
        
        samples = torch.zeros(n_samples, self._dim)
        idx = 0
        
        # Shared parameters: uniform
        for p in self._shared_params:
            low, high = self.param_bounds[p]
            samples[:, idx] = torch.rand(n_samples) * (high - low) + low
            idx += 1
        
        # Varying parameters: use linker
        for p in self._varying_params:
            linker = self._linkers[p]
            n_vals = linker.dim
            # TODO: Handle hyperparameters properly
            trajectories = linker.sample(n_samples)
            samples[:, idx:idx + self.n_sessions] = trajectories
            idx += self.n_sessions
            # Hyperparameters if inferred
            for hp in linker.hyperparameter_names:
                if hp == 'sigma_drift':
                    low, high = linker.drift_bounds
                elif hp == 'lengthscale':
                    low, high = linker.lengthscale_bounds
                elif hp == 'amplitude':
                    low, high = linker.amplitude_bounds
                elif hp in ['group_mean']:
                    low, high = linker.mean_bounds
                elif hp in ['group_std']:
                    low, high = linker.std_bounds
                else:
                    low, high = 0, 1
                samples[:, idx] = torch.rand(n_samples) * (high - low) + low
                idx += 1
        
        # If sample_shape was empty, return 1D (single sample)
        if len(sample_shape) == 0:
            return samples.squeeze(0)
        return samples
    
    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute log probability."""
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        
        n_samples = theta.shape[0]
        log_p = torch.zeros(n_samples)
        idx = 0
        
        # Shared parameters: uniform log prob
        for p in self._shared_params:
            low, high = self.param_bounds[p]
            in_bounds = (theta[:, idx] >= low) & (theta[:, idx] <= high)
            log_p = log_p + torch.where(
                in_bounds,
                torch.tensor(-np.log(high - low)),
                torch.tensor(float('-inf'))
            )
            idx += 1
        
        # Varying parameters: use linker
        for p in self._varying_params:
            linker = self._linkers[p]
            trajectories = theta[:, idx:idx + self.n_sessions]
            idx += self.n_sessions
            
            # Get hyperparameters if inferred
            hyperparam_kwargs = {}
            for hp in linker.hyperparameter_names:
                hyperparam_kwargs[hp] = theta[:, idx]
                idx += 1
            
            log_p = log_p + linker.log_prob(trajectories, **hyperparam_kwargs)
        
        return log_p
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def param_names(self) -> List[str]:
        return self._param_names.copy()
    
    @property
    def shared_params(self) -> List[str]:
        return self._shared_params.copy()
    
    def theta_to_dict(self, theta: torch.Tensor) -> Dict[str, Any]:
        """
        Convert flat theta array to dictionary.
        
        Returns dict with:
            - Shared params as scalars
            - Varying params as arrays of length n_sessions
            - Hyperparameters as scalars (if inferred)
        """
        if theta.dim() > 1:
            theta = theta.squeeze()
        
        result = {}
        idx = 0
        
        for p in self._shared_params:
            result[p] = float(theta[idx])
            idx += 1
        
        for p in self._varying_params:
            result[p] = theta[idx:idx + self.n_sessions].numpy()
            idx += self.n_sessions
            linker = self._linkers[p]
            for hp in linker.hyperparameter_names:
                result[f"{p}_{hp}"] = float(theta[idx])
                idx += 1
        
        return result
    
    def dict_to_theta(self, d: Dict[str, Any]) -> torch.Tensor:
        """Convert dictionary to flat theta array."""
        theta = []
        
        for p in self._shared_params:
            theta.append(d[p])
        
        for p in self._varying_params:
            vals = d[p]
            if isinstance(vals, np.ndarray):
                theta.extend(vals.tolist())
            else:
                theta.extend(list(vals))
            linker = self._linkers[p]
            for hp in linker.hyperparameter_names:
                theta.append(d[f"{p}_{hp}"])
        
        return torch.tensor(theta, dtype=torch.float32)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_prior(
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    model_type: str = 'be'
) -> UniformPrior:
    """
    Create single-session uniform prior.
    
    Args:
        param_bounds: Parameter bounds (if None, uses defaults for model_type)
        model_type: 'be' or 'sc'
    
    Returns:
        UniformPrior instance
    """
    if param_bounds is None:
        if model_type == 'be':
            param_bounds = DEFAULT_BE_BOUNDS
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    return UniformPrior(param_bounds)


def create_multisession_prior(
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    n_sessions: int = 10,
    varying_params: Optional[List[str]] = None,
    linking_fn: Union[str, Dict[str, str]] = 'independent',
    linking_config: Optional[Dict[str, Any]] = None,
    model_type: str = 'be'
) -> MultiSessionPrior:
    """
    Create multi-session prior with specified linking.
    
    Args:
        param_bounds: Parameter bounds (if None, uses defaults)
        n_sessions: Number of sessions
        varying_params: Parameters that vary across sessions
        linking_fn: Linking function type ('independent', 'gp', 'random_walk', 
                   'hierarchical') or dict mapping param names to types
        linking_config: Configuration for linking (passed to all varying params,
                       or dict mapping param names to configs)
        model_type: 'be' or 'sc'
    
    Returns:
        MultiSessionPrior instance
    
    Example:
        # GP-linked eta with independent sigma
        prior = create_multisession_prior(
            n_sessions=10,
            varying_params=['eta_learning', 'sigma_percep'],
            linking_fn={'eta_learning': 'gp', 'sigma_percep': 'independent'},
            linking_config={'eta_learning': {'lengthscale': 5.0}}
        )
    """
    if param_bounds is None:
        if model_type == 'be':
            param_bounds = DEFAULT_BE_BOUNDS
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    if varying_params is None:
        varying_params = ['eta_learning']
    
    # Build linking configs
    linking_configs = {}
    
    for param in varying_params:
        # Determine link type
        if isinstance(linking_fn, str):
            link_type = linking_fn
        elif isinstance(linking_fn, dict):
            link_type = linking_fn.get(param, 'independent')
        else:
            link_type = 'independent'
        
        # Determine config params
        if linking_config is None:
            params = {}
        elif isinstance(linking_config, dict):
            if param in linking_config and isinstance(linking_config[param], dict):
                params = linking_config[param]
            else:
                params = linking_config
        else:
            params = {}
        
        linking_configs[param] = LinkingConfig(link_type=link_type, params=params)
    
    return MultiSessionPrior(
        param_bounds=param_bounds,
        n_sessions=n_sessions,
        varying_params=varying_params,
        linking_configs=linking_configs
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Bounds
    'DEFAULT_BE_BOUNDS',
    # Base class
    'BasePrior',
    # Single-session
    'UniformPrior',
    # Linking functions
    'LinkingConfig',
    'IndependentLink',
    'RandomWalkLink',
    'GaussianProcessLink',
    'HierarchicalLink',
    # Multi-session
    'MultiSessionPrior',
    # Convenience
    'create_prior',
    'create_multisession_prior',
]
