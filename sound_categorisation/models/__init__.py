"""
models — BE and SC computational models for inference.

Usage:
    from sound_categorisation.models import BEParams, BEState, BEModel
    from sound_categorisation.models import SCParams, SCState, SCModel
"""
from sound_categorisation.models.be_core import BEModel, BEParams, BEState, ModelTrace
from sound_categorisation.models.sc_core import SCModel, SCParams, SCState

__all__ = [
    'BEParams', 'BEState', 'BEModel',
    'SCParams', 'SCState', 'SCModel',
    'ModelTrace',
]
