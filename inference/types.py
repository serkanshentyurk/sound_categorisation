"""
Shared Types for Inference

Single source of truth for:
- ModelType enum
- ParamConfig (parameter bounds and metadata)
- Link specifications (ConstantSpec, GPSpec, RandomWalkSpec, etc.)
- ThetaLayout (flat theta ↔ per-session params mapping)

These are SPECIFICATION objects — lightweight, frozen dataclasses that
describe how parameters behave. They do NOT contain torch code or do
sampling. The actual prior implementations that consume these specs live
in inference/priors.py.

Consumers:
    inference/priors.py          builds torch priors from specs
    inference/fitting.py         user-facing API accepts specs; uses ThetaLayout
    inference/simulator.py       ModelType enum, ParamConfig
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Union


# =============================================================================
# MODEL TYPE
# =============================================================================

class ModelType(Enum):
    """Supported model types."""
    BE = "be"
    SC = "sc"


# =============================================================================
# PARAMETER CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ParamConfig:
    """
    Configuration for a single model parameter.

    Stores bounds, default value, and display name.
    Used by simulator and prior construction.
    """
    name: str
    bounds: Tuple[float, float]
    default: Optional[float] = None

    def sample_uniform(self, rng: np.random.Generator) -> float:
        """Sample from uniform prior within bounds."""
        return rng.uniform(self.bounds[0], self.bounds[1])

    def clip(self, value: float) -> float:
        """Clip value to bounds."""
        return float(np.clip(value, self.bounds[0], self.bounds[1]))


# =============================================================================
# LINK SPECIFICATIONS
# =============================================================================
# These describe HOW a parameter varies across sessions.
# Frozen dataclasses — immutable, hashable, lightweight.
# Named *Spec to distinguish from the torch implementations in priors.py.

@dataclass(frozen=True)
class ConstantSpec:
    """
    Parameter is constant across all sessions.

    Prior: Uniform(bounds[0], bounds[1]).
    Contributes 1 dimension to theta.
    """
    bounds: Tuple[float, float]


@dataclass(frozen=True)
class GPSpec:
    """
    Parameter varies smoothly across sessions via Gaussian Process prior.

    Prior: GP(mean, RBF kernel) truncated to bounds.
    Contributes n_sessions dimensions to theta.

    Args:
        bounds: (low, high) hard bounds
        lengthscale: GP lengthscale in session-index units.
                     ~5 means parameters correlated over ~5 sessions.
        amplitude: GP amplitude (std of function values).
        mean: Mean of GP. If None, uses midpoint of bounds.
    """
    bounds: Tuple[float, float]
    lengthscale: float = 5.0
    amplitude: float = 0.1
    mean: Optional[float] = None


@dataclass(frozen=True)
class RandomWalkSpec:
    """
    Parameter drifts across sessions via random walk.

    Prior: theta_0 ~ Uniform, theta_{t+1} ~ N(theta_t, sigma_drift^2)
    Contributes n_sessions dimensions to theta.
    """
    bounds: Tuple[float, float]
    sigma_drift: float = 0.05


@dataclass(frozen=True)
class IndependentSpec:
    """
    Parameter varies independently per session (no temporal structure).

    Prior: Uniform(bounds[0], bounds[1]) per session, independently.
    Contributes n_sessions dimensions to theta.
    """
    bounds: Tuple[float, float]


@dataclass(frozen=True)
class HierarchicalSpec:
    """
    Parameter drawn from shared group distribution per session.

    The group mean and std are themselves parameters (hyperpriors).
    Contributes n_sessions + 2 dimensions to theta.

    Args:
        bounds: (low, high) hard bounds for per-session values
        group_mean_bounds: bounds for the group mean hyperparameter
        group_std_bounds: bounds for the group std hyperparameter
    """
    bounds: Tuple[float, float]
    group_mean_bounds: Optional[Tuple[float, float]] = None
    group_std_bounds: Tuple[float, float] = (0.01, 0.3)

    def __post_init__(self):
        if self.group_mean_bounds is None:
            object.__setattr__(self, 'group_mean_bounds', self.bounds)


# Type alias for any link spec
LinkSpec = Union[ConstantSpec, GPSpec, RandomWalkSpec, IndependentSpec, HierarchicalSpec]



# =============================================================================
# THETA LAYOUT
# =============================================================================

@dataclass
class ThetaLayout:
    """
    Describes how a flat theta vector maps to per-session parameters.

    Constructed from param_links and n_sessions. Handles packing/unpacking
    between flat theta (for SBI) and per-session parameter dicts (for simulator).

    This is the SINGLE canonical mapping used by both simulator.py and
    fitting.py — no duplication.
    """
    param_names: List[str]           # Canonical order of model params
    n_sessions: int
    links: Dict[str, LinkSpec]       # param_name -> link spec
    model_type: str = 'be'           # 'be' or 'sc'

    # Computed layout
    slices: Dict[str, slice] = field(init=False)
    total_dim: int = field(init=False)
    varying_params: List[str] = field(init=False)
    constant_params: List[str] = field(init=False)

    def __post_init__(self):
        self.varying_params = []
        self.constant_params = []
        self.slices = {}

        idx = 0
        for name in self.param_names:
            link = self.links[name]
            if isinstance(link, ConstantSpec):
                self.slices[name] = slice(idx, idx + 1)
                self.constant_params.append(name)
                idx += 1
            else:
                # All other links produce n_sessions values
                self.slices[name] = slice(idx, idx + self.n_sessions)
                self.varying_params.append(name)
                idx += self.n_sessions

        self.total_dim = idx

    @property
    def _PARAM_CLAMP(self) -> Dict[str, Tuple]:
        return PARAM_CLAMP.get(self.model_type, PARAM_CLAMP['be'])

    def theta_to_session_params(self, theta: np.ndarray) -> List[Dict[str, float]]:
        """
        Convert flat theta to list of per-session parameter dicts.

        Clamps values to valid ranges (posterior samples can slightly
        exceed prior bounds).

        Args:
            theta: 1D array of shape (total_dim,)

        Returns:
            List of n_sessions dicts, each with all param names as keys
        """
        session_params = []
        for s in range(self.n_sessions):
            params = {}
            for name in self.param_names:
                sl = self.slices[name]
                values = theta[sl]
                if isinstance(self.links[name], ConstantSpec):
                    val = float(values[0])
                else:
                    val = float(values[s])
                # Clamp to valid model bounds
                lo, hi = self._PARAM_CLAMP.get(name, (None, None))
                if lo is not None:
                    val = max(val, lo)
                if hi is not None:
                    val = min(val, hi)
                params[name] = val
            session_params.append(params)
        return session_params

    def theta_to_trajectories(self, theta: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert flat theta to parameter trajectories.

        Args:
            theta: Shape (total_dim,) or (n_samples, total_dim)

        Returns:
            Dict mapping param names to arrays:
                constant params: shape (n_samples,) or scalar
                varying params: shape (n_samples, n_sessions) or (n_sessions,)
        """
        single = theta.ndim == 1
        if single:
            theta = theta[np.newaxis, :]

        result = {}
        for name in self.param_names:
            sl = self.slices[name]
            values = theta[:, sl]
            if isinstance(self.links[name], ConstantSpec):
                result[name] = values[:, 0]
            else:
                result[name] = values  # (n_samples, n_sessions)

        if single:
            result = {k: v[0] if v.ndim == 1 else v[0] for k, v in result.items()}

        return result

    def get_expanded_names(self) -> List[str]:
        """Get flat list of names matching theta dimensions."""
        names = []
        for name in self.param_names:
            if isinstance(self.links[name], ConstantSpec):
                names.append(name)
            else:
                for s in range(self.n_sessions):
                    names.append(f"{name}_{s}")
        return names


# =============================================================================
# DEFAULT PARAMETER CONFIGURATIONS
# =============================================================================

def get_default_param_configs(model_type: ModelType) -> Dict[str, ParamConfig]:
    """Get default ParamConfig for each parameter of a model."""
    if model_type == ModelType.BE:
        from models.BE_core import BEParams
        bounds = BEParams.get_bounds()
        return {
            name: ParamConfig(name, bounds=bounds[name])
            for name in BEParams.get_param_names()
        }
    elif model_type == ModelType.SC:
        from models.SC_core import SCParams
        bounds = SCParams.get_bounds()
        return {
            name: ParamConfig(name, bounds=bounds[name])
            for name in SCParams.get_param_names()
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_default_links(
    model_type: ModelType,
    varying_params: Optional[List[str]] = None,
    link_type: str = 'gp',
) -> Dict[str, LinkSpec]:
    """
    Build default link specs for a model.

    Constant links for all params except those in varying_params,
    which get the specified link type.

    Args:
        model_type: BE or SC
        varying_params: Params that vary across sessions (default: model-specific)
        link_type: 'gp', 'random_walk', 'independent', 'hierarchical'
    """
    configs = get_default_param_configs(model_type)

    if varying_params is None:
        if model_type == ModelType.BE:
            varying_params = ['eta_learning', 'eta_relax']
        elif model_type == ModelType.SC:
            varying_params = ['gamma', 'sigma_update']

    link_classes = {
        'gp': GPSpec,
        'random_walk': RandomWalkSpec,
        'independent': IndependentSpec,
        'hierarchical': HierarchicalSpec,
    }

    if link_type not in link_classes:
        raise ValueError(f"Unknown link_type: {link_type}")

    LinkClass = link_classes[link_type]
    links = {}
    for name, cfg in configs.items():
        if name in varying_params:
            links[name] = LinkClass(bounds=cfg.bounds)
        else:
            links[name] = ConstantSpec(bounds=cfg.bounds)

    return links


# =============================================================================
# PARAMETER CLAMPING BOUNDS (for posterior samples that exceed priors)
# =============================================================================

PARAM_CLAMP = {
    'be': {
        'sigma_percep': (1e-6, None),
        'A_repulsion': (0.0, None),
        'eta_learning': (1e-6, 1.0),
        'eta_relax': (0.0, 1.0 - 1e-6),
    },
    'sc': {
        'sigma_percep': (1e-6, None),
        'A_repulsion': (0.0, None),
        'gamma': (1e-6, 1.0),
        'sigma_update': (1e-6, None),
    },
}
