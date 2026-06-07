"""
behav_utils.data.ops — Data operations (reshaping the data).

Operations that take SessionData (or raw arrays) and return a reorganised
view: session selection, trial filtering, pooling, downsampling. The data
classes themselves live in behav_utils.data.structures.
"""

from behav_utils.data.ops.selection import (
    select_sessions,
    SessionFilter,
    fitting_data_from_sessions,
    register_preset,
    list_presets,
    register_presets_from_config,
)
from behav_utils.data.ops.filtering import (
    filter_trials,
    pool_arrays,
)

__all__ = [
    'select_sessions', 'SessionFilter', 'fitting_data_from_sessions',
    'register_preset', 'list_presets', 'register_presets_from_config',
    'filter_trials', 'pool_arrays',
]