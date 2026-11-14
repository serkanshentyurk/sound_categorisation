"""
models — BE and SC computational models for inference.

Usage:
    from sound_categorisation.models import BEParams, BEState, BEModel
    from sound_categorisation.models import SCParams, SCState, SCModel

As behav_utils models (``behav_utils.models.simulate`` and the synthetic generators run them):
    from sound_categorisation.models import BoundaryEstimation, StatisticalCategorisation, make_model
"""
from sound_categorisation.models.be_core import BEModel, BEParams, BEState, ModelTrace
from sound_categorisation.models.sc_core import SCModel, SCParams, SCState
from sound_categorisation.models.simulate import (
    BoundaryEstimation,
    StatisticalCategorisation,
    make_model,
    simulate_choices,
)

__all__ = [
    'BEParams', 'BEState', 'BEModel',
    'SCParams', 'SCState', 'SCModel',
    'ModelTrace',
    'BoundaryEstimation', 'StatisticalCategorisation', 'make_model', 'simulate_choices',
]
