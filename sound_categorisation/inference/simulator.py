"""Static simulation-based-inference simulator for the BE / SC models.

ONE simulator. Given a model, a distribution schedule, and the number/length of
sessions, ``build_simulator`` returns:

    sim_fn(theta[, seed]) -> np.ndarray   the summary-stat observation x (None when the
                                           accuracy rule is on and a session falls below it)
    prior                                  an sbi BoxUniform over the model's parameter
                                           bounds in flat space (None if sbi/torch are
                                           unavailable)
    param_names                            order of the theta entries

Static = one parameter vector per simulated animal, shared across its N
sessions (no per-session parameter drift). The observation is built by
``to_stat_vector`` -- the SAME function used to condition on real data -- so the
train and test representations cannot diverge (same pooling, same NaN handling).

The parameter ranges are a prior variant of ``models.priors`` (``prior=None``: the one in use,
``settings.SBI_PRIOR``). ``theta`` is always in the parameters' own units here; the logarithm a
log-uniform parameter is trained in stays inside ``AmortisedSBI``.

Reuses:
    data.stimuli.sample_distribution                    stimulus sampling
                                                        (uniform / hard_a / hard_b)
    models.simulate.make_model                         BE / SC as behav_utils models
    behav_utils.models.simulate                        a model on a Design -> SessionData
    inference.representation.to_stat_vector            pooled / moments x-builder
    inference.types.get_default_param_configs          theta order + bounds
"""

from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
from behav_utils.models import Design, simulate
from behav_utils.tasks import TwoAFC

from sound_categorisation.inference.representation import to_stat_vector
from sound_categorisation.inference.types import ModelType, get_default_param_configs
from sound_categorisation.models.simulate import make_model

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


def get_bounds_arrays(model, prior: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """(lower, upper) bound arrays in param_names order, in the parameters' own units, under a
    prior variant (None = the one in use)."""
    cfg = get_default_param_configs(_as_model(model), prior)
    names = list(cfg.keys())
    lower = np.array([cfg[n].bounds[0] for n in names], dtype=float)
    upper = np.array([cfg[n].bounds[1] for n in names], dtype=float)
    return lower, upper


def theta_to_params(theta: np.ndarray, model, prior: str | None = None) -> Dict[str, float]:
    """Map a parameter vector to a {name: value} dict, clipped to a prior variant's bounds
    (None = the one in use)."""
    cfg = get_default_param_configs(_as_model(model), prior)
    names = list(cfg.keys())
    theta = np.asarray(theta, dtype=float).ravel()
    if theta.shape[0] != len(names):
        raise ValueError(
            f"theta has {theta.shape[0]} entries; model {_as_model(model)} "
            f"expects {len(names)} ({names})")
    return {n: cfg[n].clip(theta[i]) for i, n in enumerate(names)}


def accuracy_rule(rep: str, distribution: str) -> float | None:
    """The minimum session accuracy a representation is trained with on a distribution: the
    session filter's (config.yaml, ``model_id_<distribution>``) when
    ``settings.SBI_REPRESENTATIONS[rep]['accuracy_rule']`` is set, else None (no rule)."""
    from sound_categorisation.settings import SBI_REPRESENTATIONS
    if not SBI_REPRESENTATIONS[rep].get('accuracy_rule'):
        return None
    from sound_categorisation.data.cohort import preset_min_accuracy
    return preset_min_accuracy(distribution)


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

def draw_T(T, rng: np.random.Generator) -> int:
    """Trials for one session: ``T`` fixed (int) or drawn uniformly from ``(lo, hi)``."""
    if isinstance(T, (tuple, list)):
        return int(rng.integers(int(T[0]), int(T[1]) + 1))
    return int(T)


def session_accuracy(session) -> float:
    """A session's accuracy as the session filter measures it."""
    return float(session.summary().accuracy)


def simulate_sessions(model, params: Dict[str, float], schedule: Sequence[str], T,
                      burn_in: int, rng: np.random.Generator,
                      min_accuracy: float | None = None) -> List | None:
    """One simulated animal: a SessionData per entry of ``schedule``, each ``draw_T(T, rng)``
    trials from ``params``, run through behav_utils' ``simulate`` (each session's seed drawn from
    ``rng``, so sessions are decorrelated). The one place the training simulator, the synthetic
    cohort and the SBC calibration draw their sessions from. ``T`` is an int or a ``(lo, hi)``
    range drawn per session.

    With ``min_accuracy`` (the accuracy rule) the animal is given up at the first session below it
    and None is returned: the remaining sessions are not simulated. An animal that passes is the
    same, session for session, as without the rule.
    """
    from sound_categorisation.data.stimuli import sample_distribution
    agent, task = make_model(model, params, burn_in=burn_in), TwoAFC()
    sessions = []
    for i, dist in enumerate(schedule):
        stim, _ = sample_distribution(draw_T(T, rng), dist, rng=rng)
        sessions.append(simulate(agent, Design(task, stim, label=dist), rng, session_idx=i))
        if min_accuracy is not None and session_accuracy(sessions[-1]) < min_accuracy:
            return None
    return sessions


def build_simulator(
    model,
    dist_schedule: Union[str, Sequence[str]] = 'uniform',
    N: int = 1,
    T=350,
    burn_in: int = 1000,
    mode: str = 'pooled',
    stat_names: Sequence[str] | None = None,
    design_features: Sequence[str] = (),
    prior: str | None = None,
    min_accuracy: float | None = None,
) -> Tuple[Callable, object, List[str]]:
    """Build a static SBI simulator.

    Args:
        model: ModelType or 'be'/'sc'.
        dist_schedule: One distribution name applied to every session, or a list
            of exactly N names ('uniform' | 'hard_a' | 'hard_b', case-insensitive).
        N: Sessions per simulated animal. An int (fixed) or a ``(lo, hi)`` tuple
            (drawn uniformly over {lo..hi} per simulation, single-distribution
            schedule only). mode='moments' requires min N >= 2.
        T: Trials per session: an int (fixed) or a ``(lo, hi)`` range drawn
            uniformly per session.
        burn_in: Model burn-in per session.
        mode: 'pooled' or 'moments' (forwarded to to_stat_vector).
        stat_names: Summary stats (defaults to SBI_STATS via to_stat_vector).
        design_features: Appended to x by to_stat_vector ('n_sessions',
            'n_valid_trials'), so the network sees how much data it was given.
        prior: The prior variant (``models.priors``) whose bounds theta is clipped to and the
            returned prior covers; None = the one in use.
        min_accuracy: The accuracy rule: sim_fn returns None for a draw with a session below it.

    Returns:
        (sim_fn, prior, param_names) where
            sim_fn(theta, seed=None) -> np.ndarray  observation x (theta in the parameters' own
            units), or None for a draw the accuracy rule rejects;
            prior is an sbi BoxUniform over the flat-space bounds (None if sbi/torch unavailable).
    """
    from sound_categorisation.models.priors import resolve
    model = _as_model(model)
    prior = resolve(prior)                # fixed now: the simulator does not follow a later setting
    names = get_param_names(model)
    variable_N = isinstance(N, (tuple, list))
    if variable_N:
        n_lo, n_hi = int(N[0]), int(N[1])
        if not isinstance(dist_schedule, str):
            raise ValueError(
                "variable N=(lo,hi) requires a single-distribution schedule string")
        n_min = n_lo
    else:
        sched = _expand_schedule(dist_schedule, N)
        n_min = int(N)
    if mode == 'moments' and n_min < 2:
        raise ValueError(f"mode='moments' needs >= 2 sessions; got n_min={n_min}")
    if stat_names is not None:
        stat_names = list(stat_names)
    design_features = tuple(design_features or ())

    def sim_fn(theta, seed: int | None = None) -> np.ndarray:
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        if variable_N:
            n_sess = int(rng.integers(n_lo, n_hi + 1))    # uniform over {n_lo..n_hi}
            this_sched = [dist_schedule] * n_sess
        else:
            this_sched = sched
        sessions = simulate_sessions(model, theta_to_params(theta, model, prior), this_sched, T,
                                     burn_in, rng, min_accuracy=min_accuracy)
        if sessions is None:
            return None
        return to_stat_vector(sessions, mode=mode, stat_names=stat_names,
                              design_features=design_features)

    return sim_fn, _build_prior(model, prior), names


# =============================================================================
# PRIOR + SBI WRAPPER (torch / sbi)
# =============================================================================

def _build_prior(model, prior: str | None = None):
    """BoxUniform over a prior variant's bounds in flat space (``models.priors.flat_bounds``):
    the space the networks are trained in. None if sbi/torch absent."""
    try:
        import torch
        from sbi.utils import BoxUniform
    except ImportError:
        return None
    from sound_categorisation.models.priors import flat_bounds
    lower, upper = flat_bounds(_as_model(model).value, prior)
    return BoxUniform(
        low=torch.tensor(lower, dtype=torch.float32),
        high=torch.tensor(upper, dtype=torch.float32),
    )


def wrap_for_sbi(sim_fn: Callable, base_seed: int = 0, n_jobs: int = -1) -> Callable:
    """Wrap a sim_fn (theta->x, numpy) for the sbi package (theta in the parameters' own units,
    no accuracy rule). ``AmortisedSBI.train`` no longer uses it: it simulates its own training set.

    The returned function accepts a torch tensor of shape (D_theta,) or
    (batch, D_theta) and returns a torch tensor of stacked observation vectors.
    Each batch row gets its own seed (``base_seed + i``), so the batch is simulated
    in parallel (joblib, ``n_jobs``; -1 = every core the job has) with results that
    do not depend on ``n_jobs``. Batches under 64 rows run in-process.
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

        n = len(theta_np)
        if n < 64 or n_jobs == 1:
            xs = [sim_fn(theta_np[i], seed=base_seed + i) for i in range(n)]
        else:
            from joblib import Parallel, delayed
            xs = Parallel(n_jobs=n_jobs, batch_size=64)(
                delayed(sim_fn)(theta_np[i], seed=base_seed + i) for i in range(n))
        return torch.as_tensor(np.stack(xs).astype(float), dtype=torch.float32)

    return sbi_simulator
