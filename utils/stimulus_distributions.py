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

    Or use _sample_hard_a() / _sample_hard_b() directly.

Usage:
    from utils.stimulus_distributions import _sample_hard_a, _sample_hard_b

    rng = np.random.default_rng(42)
    stim_a, cat_a = _sample_hard_a(300, rng=rng)
    stim_b, cat_b = _sample_hard_b(300, rng=rng)
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm
from typing import Optional, Tuple, Dict


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
def _sample_hard_a(
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


def _sample_hard_b(
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

def _sample_uniform(
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
        'uniform': _sample_uniform,
        'hard_a': _sample_hard_a,
        'hard_b': _sample_hard_b,
    }

    if key not in samplers:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Available: {list(samplers.keys())}"
        )

    return samplers[key](n_trials, rng=rng, boundary=boundary)


def compute_distribution_density(
    distribution: str,
    s: np.ndarray,
    boundary: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Evaluate per-category density p(s|c=A) and p(s|c=B) at stimulus s.

    For the three named distributions (uniform, hard_a, hard_b), returns
    the normalised density on the support of each category. Values
    outside the support are 0.

    Args:
        distribution: 'uniform' | 'hard_a' | 'hard_b' (case-insensitive,
                      hyphens / underscores equivalent).
        s: Stimulus values. Scalar or array.
        boundary: Category boundary (default 0). A is s<boundary,
                  B is s>=boundary.

    Returns:
        {
            's':         input stimulus values (same shape as input),
            'density_a': p(s|c=A) at each s,
            'density_b': p(s|c=B) at each s,
        }
    """
    key = distribution.lower().replace('-', '_').replace(' ', '_')
    s_arr = np.asarray(s, dtype=float)
    s_shift = s_arr - boundary

    p_a = np.zeros_like(s_shift)
    p_b = np.zeros_like(s_shift)

    on_a = (s_shift >= -1) & (s_shift < 0)
    on_b = (s_shift >= 0) & (s_shift <= 1)

    if key == 'uniform':
        p_a[on_a] = 1.0
        p_b[on_b] = 1.0
    elif key == 'hard_a':
        s_a = s_shift[on_a]
        p_a[on_a] = _LAMBDA * np.exp(_LAMBDA * s_a) + np.exp(-_LAMBDA)
        p_b[on_b] = 1.0
    elif key == 'hard_b':
        p_a[on_a] = 1.0
        s_b = s_shift[on_b]
        p_b[on_b] = _LAMBDA * np.exp(-_LAMBDA * s_b) + np.exp(-_LAMBDA)
    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Available: uniform, hard_a, hard_b"
        )

    return {'s': s_arr, 'density_a': p_a, 'density_b': p_b}


def compute_normative_pse(
    distribution: str,
    sigma_percep: float,
    boundary: float = 0.0,
) -> float:
    """
    Optimal PSE for an ideal Bayesian observer.

    The observer perceives x ~ N(s, σ²) given true stimulus s, with
    balanced priors P(A) = P(B) = 0.5. The optimal boundary is the
    value of x at which p(x|A) = p(x|B), where

        p(x|c) = ∫ N(x; s, σ²) · p(s|c) ds

    Uniform distribution: PSE = boundary (by symmetry).
    Hard-A / Hard-B: solved numerically via brentq.

    Args:
        distribution: 'uniform' | 'hard_a' | 'hard_b'.
        sigma_percep: Perceptual noise standard deviation (> 0). This is
                      the σ of the encoding model (x ~ N(s, σ²)), NOT
                      the slope of a fitted psychometric curve. The
                      psychometric σ is influenced by lapses and any
                      decision noise on top; they are related but not
                      identical. From SBI: use the posterior of the BE
                      or SC model's sigma_percep parameter. From data:
                      psychometric σ on expert uniform sessions is an
                      acceptable approximation when lapse rates are low.
        boundary: Category boundary (default 0).

    Returns:
        Optimal PSE (mu) in stimulus units.
    """
    key = distribution.lower().replace('-', '_').replace(' ', '_')

    if sigma_percep <= 0:
        raise ValueError(f"sigma_percep must be positive, got {sigma_percep}")

    if key == 'uniform':
        return float(boundary)

    if key not in ('hard_a', 'hard_b'):
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Available: uniform, hard_a, hard_b"
        )

    sigma = sigma_percep

    def p_x_given_a(x: float) -> float:
        if key == 'hard_a':
            integrand = lambda s: (
                norm.pdf(x, s + boundary, sigma)
                * (_LAMBDA * np.exp(_LAMBDA * s) + np.exp(-_LAMBDA))
            )
        else:
            integrand = lambda s: norm.pdf(x, s + boundary, sigma)
        return quad(integrand, -1, 0, limit=100)[0]

    def p_x_given_b(x: float) -> float:
        if key == 'hard_b':
            integrand = lambda s: (
                norm.pdf(x, s + boundary, sigma)
                * (_LAMBDA * np.exp(-_LAMBDA * s) + np.exp(-_LAMBDA))
            )
        else:
            integrand = lambda s: norm.pdf(x, s + boundary, sigma)
        return quad(integrand, 0, 1, limit=100)[0]

    def difference(x: float) -> float:
        return p_x_given_b(x) - p_x_given_a(x)

    try:
        pse = brentq(difference, boundary - 0.5, boundary + 0.5, xtol=1e-5)
    except ValueError:
        try:
            pse = brentq(difference, boundary - 1.0, boundary + 1.0, xtol=1e-5)
        except ValueError:
            return float('nan')

    return float(pse)
