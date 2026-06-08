"""Static simulation-based-inference simulator for the BE / SC models.

ONE simulator. Given a model, a distribution schedule, and the number/length of
sessions, ``build_simulator`` returns:

    sim_fn(theta[, seed]) -> np.ndarray   the summary-stat observation x
    prior                                  an sbi BoxUniform over the model's
                                           parameter bounds (None if sbi/torch
                                           are unavailable)
    param_names                            order of the theta entries

Static = one parameter vector per simulated animal, shared across its N
sessions (no per-session parameter drift). The observation is built by
``to_stat_vector`` -- the SAME function used to condition on real data -- so the
train and test representations cannot diverge (same pooling, same NaN handling).

Reuses:
    utils.stimulus_distributions.sample_distribution   stimulus sampling
                                                        (uniform / hard_a / hard_b)
    models.simulate.simulate_choices                   the single params->choices core
    behav_utils.data.synthetic.session_from_arrays     sim arrays -> SessionData
    inference.representation.to_stat_vector            pooled / moments x-builder
    inference.types.get_default_param_configs          theta order + bounds
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from inference.types import ModelType, get_default_param_configs
from inference.representation import to_stat_vector
from models.simulate import simulate_choices
from behav_utils.data.synthetic import session_from_arrays


# =============================================================================
# PARAMETER LAYOUT (theta <-> params), driven by get_default_param_configs
# =============================================================================

def _as_model(model) -> ModelType:
    """Coerce a model identifier to ModelType."""
    if isinstance(model, ModelType):
        return model
    return ModelType(str(getattr(model, 'value', model)).lower())


def get_param_names(model) -> List[str]:
    """Ordered free-parameter names for a model (the theta layout)."""
    return list(get_default_param_configs(_as_model(model)).keys())


def get_bounds_arrays(model) -> Tuple[np.ndarray, np.ndarray]:
    """(lower, upper) bound arrays in param_names order."""
    cfg = get_default_param_configs(_as_model(model))
    names = list(cfg.keys())
    lower = np.array([cfg[n].bounds[0] for n in names], dtype=float)
    upper = np.array([cfg[n].bounds[1] for n in names], dtype=float)
    return lower, upper


def theta_to_params(theta: np.ndarray, model) -> Dict[str, float]:
    """Map a parameter vector to a clipped {name: value} dict."""
    cfg = get_default_param_configs(_as_model(model))
    names = list(cfg.keys())
    theta = np.asarray(theta, dtype=float).ravel()
    if theta.shape[0] != len(names):
        raise ValueError(
            f"theta has {theta.shape[0]} entries; model {_as_model(model)} "
            f"expects {len(names)} ({names})")
    return {n: cfg[n].clip(theta[i]) for i, n in enumerate(names)}


# =============================================================================
# DISTRIBUTION SCHEDULE
# =============================================================================

def _expand_schedule(dist_schedule: Union[str, Sequence[str]], N: int) -> List[str]:
    """Resolve the schedule to a per-session list of length N."""
    if isinstance(dist_schedule, str):
        return [dist_schedule] * N
    sched = list(dist_schedule)
    if len(sched) == 1:
        return sched * N
    if len(sched) != N:
        raise ValueError(
            f"dist_schedule has {len(sched)} entries but N={N}; pass one "
            f"distribution name, or exactly N.")
    return sched


# =============================================================================
# SIMULATOR
# =============================================================================

def build_simulator(
    model,
    dist_schedule: Union[str, Sequence[str]] = 'uniform',
    N: int = 1,
    T: int = 350,
    burn_in: int = 1000,
    mode: str = 'pooled',
    stat_names: Optional[Sequence[str]] = None,
) -> Tuple[Callable, object, List[str]]:
    """Build a static SBI simulator.

    Args:
        model: ModelType or 'be'/'sc'.
        dist_schedule: One distribution name applied to every session, or a list
            of exactly N names ('uniform' | 'hard_a' | 'hard_b', case-insensitive).
        N: Sessions per simulated animal. mode='moments' requires N >= 4.
        T: Trials per session (fixed; never a network input).
        burn_in: Model burn-in per session.
        mode: 'pooled' or 'moments' (forwarded to to_stat_vector).
        stat_names: Summary stats (defaults to SBI_STATS via to_stat_vector).

    Returns:
        (sim_fn, prior, param_names) where
            sim_fn(theta, seed=None) -> np.ndarray  observation x;
            prior is an sbi BoxUniform (None if sbi/torch unavailable).
    """
    model = _as_model(model)
    names = get_param_names(model)
    sched = _expand_schedule(dist_schedule, N)
    if mode == 'moments' and N < 4:
        raise ValueError(f"mode='moments' needs N>=4 sessions; got N={N}")
    if stat_names is not None:
        stat_names = list(stat_names)

    def sim_fn(theta, seed: Optional[int] = None) -> np.ndarray:
        from utils.stimulus_distributions import sample_distribution
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        params = theta_to_params(theta, model)
        rng = np.random.default_rng(seed)

        sessions = []
        for i, dist in enumerate(sched):
            stim, cat = sample_distribution(T, dist, rng=rng)
            sess_seed = int(rng.integers(0, 2**31 - 1))   # decorrelated per session
            ch = simulate_choices(
                model, params, stim, cat, burn_in=burn_in, seed=sess_seed)
            sessions.append(session_from_arrays(
                stim, ch, cat, session_idx=i, distribution=dist))

        return to_stat_vector(sessions, mode=mode, stat_names=stat_names)

    prior = _build_prior(model)
    return sim_fn, prior, names


# =============================================================================
# PRIOR + SBI WRAPPER (torch / sbi)
# =============================================================================

def _build_prior(model):
    """BoxUniform over the model's parameter bounds. None if sbi/torch absent."""
    try:
        from sbi.utils import BoxUniform
        import torch
    except ImportError:
        return None
    lower, upper = get_bounds_arrays(model)
    return BoxUniform(
        low=torch.tensor(lower, dtype=torch.float32),
        high=torch.tensor(upper, dtype=torch.float32),
    )


def wrap_for_sbi(sim_fn: Callable, base_seed: int = 0) -> Callable:
    """Wrap a sim_fn (theta->x, numpy) for the sbi package.

    The returned function accepts a torch tensor of shape (D_theta,) or
    (batch, D_theta) and returns a torch tensor of stacked observation vectors.
    Each batch row gets a distinct seed so the simulations are independent.
    """
    import torch

    def sbi_simulator(theta):
        if hasattr(theta, 'detach'):
            theta_np = theta.detach().cpu().numpy()
        else:
            theta_np = np.asarray(theta)

        if theta_np.ndim == 1:
            x = sim_fn(theta_np, seed=base_seed)
            return torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float32)

        xs = [sim_fn(theta_np[i], seed=base_seed + i) for i in range(len(theta_np))]
        return torch.as_tensor(np.stack(xs).astype(float), dtype=torch.float32)

    return sbi_simulator
