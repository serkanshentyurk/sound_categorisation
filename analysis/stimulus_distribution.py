"""
Extended Stimulus Sampling

Adds Hard-A and Hard-B asymmetric distributions to behav_utils,
porting from legacy/sampling.py with proper RNG handling.

Distribution definitions (matching the experiment):
    Uniform:  50% Uniform[-1,0] + 50% Uniform[0,1]
    Hard-A:   50% Uniform[0,1]  + 50% HardA[-1,0]  (exponential tilt toward 0 from negative side)
    Hard-B:   50% Uniform[-1,0] + 50% HardB[0,1]   (exponential tilt toward 0 from positive side)

    Config mapping: Asym_Right → Hard-A, Asym_Left → Hard-B

    The "hard" half uses a tilted density:
        HardA on [-1,0]: f(x) = λ·exp(λx) + exp(-λ)
        HardB on [0,1]:  f(x) = λ·exp(-λx) + exp(-λ)
    where λ ≈ 1.841 solves λ + exp(-λ) = 2 (ensures integral = 1).

    In both cases, density is highest near the boundary (x=0) and lowest
    at the extremes — making near-boundary trials overrepresented.

Integration:
    Call register_distributions() once at startup to make 'hard_a' and
    'hard_b' available in behav_utils.data.synthetic.sample_stimuli().

    Or use sample_hard_a() / sample_hard_b() directly.

Usage:
    from analysis.stimulus_distributions import sample_hard_a, sample_hard_b

    rng = np.random.default_rng(42)
    stim_a, cat_a = sample_hard_a(300, rng=rng)
    stim_b, cat_b = sample_hard_b(300, rng=rng)
"""

import numpy as np
from scipy.optimize import fsolve
from typing import Optional, Tuple


# =============================================================================
# λ CONSTANT
# =============================================================================

def _solve_lambda() -> float:
    """Solve λ + exp(-λ) = 2. Result ≈ 1.841."""
    root = fsolve(lambda x: x + np.exp(-x) - 2, 1.0)
    return float(root[0])


# Cache the constant — it's invariant.
_LAMBDA = _solve_lambda()


# =============================================================================
# CORE SAMPLERS (one-sided, rejection sampling)
# =============================================================================

def _sample_hard_positive(
    n_samples: int,
    lam: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample from HardB density on [0, 1]:
        f(x) = λ·exp(-λx) + exp(-λ)

    Rejection sampling with proposal Uniform[0,1], envelope M=2.
    """
    samples = np.empty(n_samples)
    count = 0
    while count < n_samples:
        # Batch rejection sampling for efficiency
        batch = max(n_samples - count, 64)
        x_cand = rng.uniform(0, 1, batch)
        u = rng.uniform(0, 1, batch)
        f_x = lam * np.exp(-lam * x_cand) + np.exp(-lam)
        accept = u <= f_x / 2.0
        n_accept = accept.sum()
        if n_accept > 0:
            take = min(n_accept, n_samples - count)
            samples[count:count + take] = x_cand[accept][:take]
            count += take
    return samples


def _sample_hard_negative(
    n_samples: int,
    lam: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample from HardA density on [-1, 0]:
        f(x) = λ·exp(λx) + exp(-λ)

    Rejection sampling with proposal Uniform[-1,0], envelope M=2.
    """
    samples = np.empty(n_samples)
    count = 0
    while count < n_samples:
        batch = max(n_samples - count, 64)
        x_cand = rng.uniform(-1, 0, batch)
        u = rng.uniform(0, 1, batch)
        f_x = lam * np.exp(lam * x_cand) + np.exp(-lam)
        accept = u <= f_x / 2.0
        n_accept = accept.sum()
        if n_accept > 0:
            take = min(n_accept, n_samples - count)
            samples[count:count + take] = x_cand[accept][:take]
            count += take
    return samples


# =============================================================================
# FULL DISTRIBUTION SAMPLERS
# =============================================================================
def sample_hard_a(
    n_trials: int,
    rng: Optional[np.random.Generator] = None,
    boundary: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample from Hard-A distribution (config: Asym_Right).

    50% Uniform[0,1] + 50% HardA[-1,0] (exponential tilt toward boundary).
    More near-boundary trials from the NEGATIVE side.

    Returns:
        (stimuli, categories)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Per-trial coin flip: which half-distribution?
    is_uniform_half = rng.random(n_trials) < 0.5
    n_uniform = int(is_uniform_half.sum())
    n_hard = n_trials - n_uniform

    stimuli = np.empty(n_trials)
    stimuli[is_uniform_half] = rng.uniform(0, 1, n_uniform)
    if n_hard > 0:
        stimuli[~is_uniform_half] = _sample_hard_negative(n_hard, _LAMBDA, rng)

    categories = (stimuli > boundary).astype(int)
    return stimuli, categories


def sample_hard_b(
    n_trials: int,
    rng: Optional[np.random.Generator] = None,
    boundary: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample from Hard-B distribution (config: Asym_Left).

    50% Uniform[-1,0] + 50% HardB[0,1] (exponential tilt toward boundary).
    More near-boundary trials from the POSITIVE side.

    Returns:
        (stimuli, categories)
    """
    if rng is None:
        rng = np.random.default_rng()

    is_uniform_half = rng.random(n_trials) < 0.5
    n_uniform = int(is_uniform_half.sum())
    n_hard = n_trials - n_uniform

    stimuli = np.empty(n_trials)
    stimuli[is_uniform_half] = rng.uniform(-1, 0, n_uniform)
    if n_hard > 0:
        stimuli[~is_uniform_half] = _sample_hard_positive(n_hard, _LAMBDA, rng)

    categories = (stimuli > boundary).astype(int)
    return stimuli, categories

def sample_uniform(
    n_trials: int,
    rng: Optional[np.random.Generator] = None,
    boundary: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample from Uniform distribution (matching legacy Uniform function).

    50% Uniform[-1,0] + 50% Uniform[0,1].
    Equivalent to Uniform[-1,1] but uses the same per-trial coin flip.

    Returns:
        (stimuli, categories)
    """
    if rng is None:
        rng = np.random.default_rng()

    stimuli = rng.uniform(-1, 1, n_trials)
    categories = (stimuli > boundary).astype(int)
    return stimuli, categories


def sample_distribution(
    n_trials: int,
    distribution: str,
    rng: Optional[np.random.Generator] = None,
    boundary: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample stimuli from a named distribution.

    Args:
        n_trials: Number of trials
        distribution: 'Uniform', 'Hard-A', 'Hard-B'
                      (case-insensitive, underscores and hyphens equivalent)
        rng: Random number generator
        boundary: Category boundary

    Returns:
        (stimuli, categories)
    """
    key = distribution.lower().replace('-', '_').replace(' ', '_')

    samplers = {
        'uniform': sample_uniform,
        'hard_a': sample_hard_a,
        'hard_b': sample_hard_b,
    }

    if key not in samplers:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Available: {list(samplers.keys())}"
        )

    return samplers[key](n_trials, rng=rng, boundary=boundary)


# =============================================================================
# LAMBDA VALUE (exposed for testing/documentation)
# =============================================================================

def get_lambda() -> float:
    """Return the λ value used for Hard-A/B distributions (≈ 1.841)."""
    return _LAMBDA
