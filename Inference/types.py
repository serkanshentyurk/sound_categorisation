"""
Shared Types for Inference

Single source of truth for:
- ModelType enum
- ParamConfig (parameter bounds and metadata)
- Link specifications (ConstantLink, GPLink, RandomWalkLink, etc.)

These are SPECIFICATION objects — lightweight, frozen dataclasses that
describe how parameters behave. They do NOT contain torch code or do
sampling. The actual prior implementations that use these specs live
in inference/priors.py.

Consumers:
    inference/priors.py          builds torch priors from link specs
    inference/theta_layout.py    maps flat theta → per-session params
    inference/fitter.py          user-facing API accepts link specs
    inference/simulator.py       ModelType enum, ParamConfig
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional


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

@dataclass(frozen=True)
class ConstantLink:
    """
    Parameter is constant across all sessions.

    Prior: Uniform(bounds[0], bounds[1]).
    Contributes 1 dimension to theta.
    """
    bounds: Tuple[float, float]


@dataclass(frozen=True)
class GPLink:
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
class RandomWalkLink:
    """
    Parameter drifts across sessions via random walk.

    Prior: theta_0 ~ Uniform, theta_{t+1} ~ N(theta_t, sigma_drift^2)
    Contributes n_sessions dimensions to theta.
    """
    bounds: Tuple[float, float]
    sigma_drift: float = 0.05


@dataclass(frozen=True)
class IndependentLink:
    """
    Parameter varies independently per session (no temporal structure).

    Prior: Uniform(bounds[0], bounds[1]) per session, independently.
    Contributes n_sessions dimensions to theta.
    """
    bounds: Tuple[float, float]


@dataclass(frozen=True)
class HierarchicalLink:
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
LinkSpec = (ConstantLink | GPLink | RandomWalkLink | IndependentLink | HierarchicalLink)


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

    links = {}
    for name, cfg in configs.items():
        if name in varying_params:
            if link_type == 'gp':
                links[name] = GPLink(bounds=cfg.bounds)
            elif link_type == 'random_walk':
                links[name] = RandomWalkLink(bounds=cfg.bounds)
            elif link_type == 'independent':
                links[name] = IndependentLink(bounds=cfg.bounds)
            elif link_type == 'hierarchical':
                links[name] = HierarchicalLink(bounds=cfg.bounds)
            else:
                raise ValueError(f"Unknown link_type: {link_type}")
        else:
            links[name] = ConstantLink(bounds=cfg.bounds)

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
