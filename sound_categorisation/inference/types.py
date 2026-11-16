"""
Shared Types for Inference

Single source of truth for:
- ModelType enum
- ParamConfig (parameter bounds and metadata)

These are lightweight specification objects. They contain no torch code and
do no sampling.

Consumers:
    inference/simulator.py   ModelType enum, ParamConfig, get_default_param_configs
    inference/comparison.py  ModelType, ParamConfig
    inference/amortised.py   ModelType, ParamConfig, get_default_param_configs
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

import numpy as np

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
    default: float | None = None

    def sample_uniform(self, rng: np.random.Generator) -> float:
        """Sample from uniform prior within bounds."""
        return rng.uniform(self.bounds[0], self.bounds[1])

    def clip(self, value: float) -> float:
        """Clip value to bounds."""
        return float(np.clip(value, self.bounds[0], self.bounds[1]))


# =============================================================================
# DEFAULT PARAMETER CONFIGURATIONS
# =============================================================================

def get_default_param_configs(model_type: ModelType, prior: str | None = None) -> Dict[str, ParamConfig]:
    """ParamConfig for each parameter of a model, in theta order, with the bounds of a prior
    variant (``models.priors``; None = the one in use, ``settings.SBI_PRIOR``)."""
    from sound_categorisation.models.priors import prior_bounds
    if not isinstance(model_type, ModelType):
        raise ValueError(f"Unknown model type: {model_type}")
    return {name: ParamConfig(name, bounds=bounds)
            for name, bounds in prior_bounds(model_type.value, prior).items()}
