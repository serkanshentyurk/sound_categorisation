"""inference -- static simulation-based inference for the BE and SC models.

Flow:
    build_simulator(model, dist_schedule, N, T, burn_in, mode, stat_names)
        -> (sim_fn, prior, param_names)            simulator.py
    AmortisedSBI(...).train(n_sims) / .save / .load / .condition(sessions)
        -> posterior, point estimates              amortised.py   (train once,
                                                                    condition many)
    condition_sbi(sessions, net, model, ...)
        -> per-rep held-out MSE results            selection.py   (held-out UM/CP
                                                                    CV; the BE-vs-SC
                                                                    call is
                                                                    cv_utils.compare_models)

The observation x is built by to_stat_vector (representation.py) for BOTH
simulation (training) and conditioning, so train and test representations match.

Module structure:
    types.py          -- ModelType, ParamConfig, get_default_param_configs
    constants.py      -- SBI_STATS
    representation.py  -- to_stat_vector (pooled / moments), the shared x-builder
    simulator.py      -- build_simulator, theta_to_params, get_param_names,
                          get_bounds_arrays, wrap_for_sbi
    amortised.py      -- AmortisedSBI (SNPE engine)
    selection.py      -- condition_sbi (held-out CV; comparison in cv_utils)
"""

from sound_categorisation.inference.amortised import AmortisedSBI
from sound_categorisation.inference.constants import SBI_STATS
from sound_categorisation.inference.representation import to_stat_vector
from sound_categorisation.inference.selection import condition_sbi
from sound_categorisation.inference.simulator import (
    build_simulator,
    get_bounds_arrays,
    get_param_names,
    theta_to_params,
    wrap_for_sbi,
)
from sound_categorisation.inference.types import (
    ModelType,
    ParamConfig,
    get_default_param_configs,
)

__all__ = [
    'ModelType', 'ParamConfig', 'get_default_param_configs',
    'SBI_STATS',
    'to_stat_vector',
    'build_simulator', 'theta_to_params',
    'get_param_names', 'get_bounds_arrays', 'wrap_for_sbi',
    'AmortisedSBI',
    'condition_sbi',
]
