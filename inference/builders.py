"""
SBI building blocks: prior and simulator construction.

Shared helpers used by SBIFitter and train_per_animal_snpe.

Public API:
    build_prior            — Prior construction from ThetaLayout
    build_simulator        — Simulator construction from ThetaLayout + data
    compute_observed_stats — Observed stats from FittingData
    DEFAULT_SUMMARY_STATS  — Standard stat list
    DEFAULT_BE_PARAM_LINKS — Standard BE link specs
    DEFAULT_SC_PARAM_LINKS — Standard SC link specs
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Callable, Union, Any

# Lazy torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from inference.types import (
    ConstantSpec, GPSpec, RandomWalkSpec, IndependentSpec, HierarchicalSpec,
    ThetaLayout, PARAM_CLAMP, LinkSpec,
)


def build_prior(layout: ThetaLayout) -> Any:
    """
    Build an SBI-compatible prior from ThetaLayout.

    Uses link specifications to construct appropriate marginal
    distributions, then combines them.

    Returns an object with .sample() and .log_prob() methods.
    """
    from inference.priors import MultiSessionPrior, LinkingConfig, UniformPrior

    if not layout.varying_params:
        bounds = {name: layout.links[name].bounds for name in layout.param_names}
        return UniformPrior(bounds, param_order=layout.param_names)

    param_bounds = {name: layout.links[name].bounds for name in layout.param_names}

    linking_configs = {}
    for name in layout.varying_params:
        link = layout.links[name]
        if isinstance(link, GPSpec):
            linking_configs[name] = LinkingConfig(
                link_type='gp',
                params={'lengthscale': link.lengthscale,
                        'amplitude': link.amplitude, 'mean': link.mean},
            )
        elif isinstance(link, RandomWalkSpec):
            linking_configs[name] = LinkingConfig(
                link_type='random_walk',
                params={'sigma_drift': link.sigma_drift},
            )
        elif isinstance(link, IndependentSpec):
            linking_configs[name] = LinkingConfig(link_type='independent')
        elif isinstance(link, HierarchicalSpec):
            linking_configs[name] = LinkingConfig(
                link_type='hierarchical',
                params={'group_mean': link.group_mean, 'group_std': link.group_std},
            )
        else:
            raise ValueError(f"Unknown link type for {name}: {type(link)}")

    return MultiSessionPrior(
        param_bounds=param_bounds, n_sessions=layout.n_sessions,
        varying_params=layout.varying_params,
        linking_configs=linking_configs, param_order=layout.param_names,
    )


# =============================================================================
# SIMULATOR BUILDER 
# =============================================================================

def build_simulator(
    layout: ThetaLayout,
    stimuli_per_session: List[np.ndarray],
    categories_per_session: List[np.ndarray],
    no_response_per_session: List[np.ndarray],
    not_blockstart_per_session: List[np.ndarray],
    summary_stat_names: List[str],
    burn_in: int = 0,
    burn_in_seed: int = 42,
    model_type: str = 'be',
) -> Callable:
    """
    Build simulator function: theta → summary_stats.

    The returned function unpacks theta into per-session parameter dicts,
    simulates each session with real stimuli/categories, chains belief
    state, computes summary statistics, and returns a flat 1D array.
    """
    from behav_utils.analysis.summary_stats import compute_summary_stats

    n_sessions = layout.n_sessions

    if model_type == 'be':
        from models.BE_core import BEParams, BEState, BEModel

        def simulate(theta: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
            if seed is None:
                seed = np.random.randint(0, 2**31)
            rng = np.random.default_rng(seed)
            session_params = layout.theta_to_session_params(theta)

            be_params_0 = BEParams(**session_params[0])
            state = (BEModel.run_burn_in(be_params_0, BEState.initial_uniform(),
                                          burn_in, burn_in_seed)
                     if burn_in > 0 else BEState.initial_uniform())

            all_choices = []
            for s in range(n_sessions):
                be_params = BEParams(**session_params[s])
                choices, _, state, _ = BEModel.simulate_session(
                    params=be_params, initial_state=state,
                    stimuli=stimuli_per_session[s],
                    categories=categories_per_session[s], rng=rng,
                    no_response=no_response_per_session[s],
                    not_blockstart=not_blockstart_per_session[s],
                    return_history=False,
                )
                all_choices.append(choices)

            all_stats = []
            for s in range(n_sessions):
                stats = compute_summary_stats(
                    choices=all_choices[s], stimuli=stimuli_per_session[s],
                    categories=categories_per_session[s],
                    stat_names=summary_stat_names, return_dict=False,
                )
                all_stats.append(stats)
            return np.concatenate(all_stats)

    elif model_type == 'sc':
        from models.SC_core import SCParams, SCState, SCModel

        def simulate(theta: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
            if seed is None:
                seed = np.random.randint(0, 2**31)
            rng = np.random.default_rng(seed)
            session_params = layout.theta_to_session_params(theta)

            sc_params_0 = SCParams(**session_params[0])
            state = (SCModel.create_initial_state(params=sc_params_0,
                                                   burn_in=burn_in,
                                                   seed=burn_in_seed)
                     if burn_in > 0 else SCState.initial_default())

            all_choices = []
            for s in range(n_sessions):
                sc_params = SCParams(**session_params[s])
                choices, _, state, _ = SCModel.simulate_session(
                    params=sc_params, initial_state=state,
                    stimuli=stimuli_per_session[s],
                    categories=categories_per_session[s], rng=rng,
                    no_response=no_response_per_session[s],
                    not_blockstart=not_blockstart_per_session[s],
                    return_history=False,
                )
                all_choices.append(choices)

            all_stats = []
            for s in range(n_sessions):
                stats = compute_summary_stats(
                    choices=all_choices[s], stimuli=stimuli_per_session[s],
                    categories=categories_per_session[s],
                    stat_names=summary_stat_names, return_dict=False,
                )
                all_stats.append(stats)
            return np.concatenate(all_stats)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return simulate


# =============================================================================
# OBSERVED STATS 
# =============================================================================

def compute_observed_stats(
    fitting_data: Any,
    summary_stat_names: List[str],
) -> np.ndarray:
    """Compute summary statistics from observed (real) FittingData."""
    from behav_utils.analysis.summary_stats import compute_summary_stats

    all_stats = []
    for s in range(fitting_data.n_sessions):
        sa = fitting_data.get_session(s)
        stats = compute_summary_stats(
            choices=sa['choices'], stimuli=sa['stimuli'],
            categories=sa['categories'],
            stat_names=summary_stat_names, return_dict=False,
        )
        all_stats.append(stats)
    return np.concatenate(all_stats)


# =============================================================================
# DEFAULT LINK SPECS
# =============================================================================

DEFAULT_SUMMARY_STATS = [
    'accuracy', 'psychometric', 'recency', 'win_stay', 'stimulus_sensitivity',
]

DEFAULT_BE_PARAM_LINKS = {
    'sigma_percep': ConstantSpec(bounds=(0.05, 0.5)),
    'A_repulsion': ConstantSpec(bounds=(0.0, 0.5)),
    'eta_learning': ConstantSpec(bounds=(0.05, 0.9)),
    'eta_relax': ConstantSpec(bounds=(0.01, 0.4)),
}

DEFAULT_SC_PARAM_LINKS = {
    'sigma_percep': ConstantSpec(bounds=(0.05, 0.5)),
    'A_repulsion': ConstantSpec(bounds=(0.0, 0.5)),
    'gamma': ConstantSpec(bounds=(0.1, 1.0)),
    'sigma_update': ConstantSpec(bounds=(0.1, 1.0)),
}
