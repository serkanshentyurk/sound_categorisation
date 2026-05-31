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

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


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
