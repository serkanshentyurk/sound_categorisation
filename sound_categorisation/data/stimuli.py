"""
Stimulus distributions: Uniform, Hard-A and Hard-B as the rig draws them.

    Uniform:  Uniform[-1, 1]
    Hard-A:   50% Uniform[0, 1]  + 50% hard side on [-1, 0]   (rig: Asym_Right)
    Hard-B:   50% Uniform[-1, 0] + 50% hard side on [0, 1]    (rig: Asym_Left)

The hard side is a truncated exponential in the distance d from the boundary (d in [0, 1]):

    f(d) = r · exp(-r·d) / (1 - exp(-r)),   r = HARD_EXP_RATE = 2.15329

which is what the rig does: ``Sound_Cat_V2.bonsai`` draws an exponential with rate
``Distribution_Exp_Rate`` and redraws until the value is <= 1. Every Hard session in the
snapshot records 2.15329; it puts the mean distance from the boundary at 1/3. (Measured on the
snapshot after the stable start, the first trials of a session, which play easy sounds: 0.328-0.334.)

Up to 0.7 this module used the legacy density f(d) = λ·exp(-λd) + exp(-λ), λ ≈ 1.841 (mean
distance 0.378), which is not what this rig plays; normative PSEs computed with it were about
25 % too small.

Sampling is exact (inverse CDF), so it needs no rejection loop. Every function takes ``rate`` if
a different rig setting has to be modelled.

Usage:
    from sound_categorisation.data.stimuli import sample_distribution, compute_normative_pse

    stim, cat = sample_distribution(300, 'hard_a', rng=np.random.default_rng(42))
    compute_normative_pse('hard_a', sigma_percep=0.2)    # ≈ +0.078
"""

from typing import Dict, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

# The rig's Distribution_Exp_Rate (Sound_Cat_V2.bonsai); the same in every Hard session recorded.
HARD_EXP_RATE = 2.15329


def hard_side_density(d, rate: float = HARD_EXP_RATE):
    """Density of the distance ``d`` from the boundary on the hard side (zero outside [0, 1])."""
    d = np.asarray(d, dtype=float)
    out = rate * np.exp(-rate * d) / (1.0 - np.exp(-rate))
    return np.where((d >= 0) & (d <= 1), out, 0.0)


def _sample_hard_distance(n: int, rng: np.random.Generator, rate: float = HARD_EXP_RATE) -> np.ndarray:
    """``n`` distances from the hard-side density, by inverting its CDF (1 - exp(-r·d)) / (1 - exp(-r))."""
    u = rng.uniform(0.0, 1.0, n)
    return -np.log1p(-u * (1.0 - np.exp(-rate))) / rate


def _sample_hard(n_trials: int, rng: np.random.Generator | None, boundary: float, hard_sign: int,
                 rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Half the trials (by a per-trial coin flip) from the hard side, ``hard_sign`` -1 (A) or +1 (B);
    the other half uniform on the other side."""
    rng = np.random.default_rng() if rng is None else rng
    is_uniform_half = rng.random(n_trials) < 0.5
    n_uniform = int(is_uniform_half.sum())
    stimuli = np.empty(n_trials)
    stimuli[is_uniform_half] = -hard_sign * rng.uniform(0, 1, n_uniform)
    stimuli[~is_uniform_half] = hard_sign * _sample_hard_distance(n_trials - n_uniform, rng, rate)
    stimuli = stimuli + boundary
    return stimuli, (stimuli > boundary).astype(int)


def _sample_hard_a(n_trials: int, rng: np.random.Generator | None = None, boundary: float = 0.0,
                   rate: float = HARD_EXP_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """Hard-A (rig: Asym_Right): the A side (below the boundary) is the hard side. Returns (stimuli, categories)."""
    return _sample_hard(n_trials, rng, boundary, -1, rate)


def _sample_hard_b(n_trials: int, rng: np.random.Generator | None = None, boundary: float = 0.0,
                   rate: float = HARD_EXP_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """Hard-B (rig: Asym_Left): the B side (above the boundary) is the hard side. Returns (stimuli, categories)."""
    return _sample_hard(n_trials, rng, boundary, +1, rate)


def _sample_uniform(n_trials: int, rng: np.random.Generator | None = None, boundary: float = 0.0,
                    rate: float = HARD_EXP_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform[-1, 1] around the boundary (``rate`` unused). Returns (stimuli, categories)."""
    rng = np.random.default_rng() if rng is None else rng
    stimuli = rng.uniform(-1, 1, n_trials) + boundary
    return stimuli, (stimuli > boundary).astype(int)


def sample_distribution(
    n_trials: int,
    distribution: str,
    rng: np.random.Generator | None = None,
    boundary: float = 0.0,
    rate: float = HARD_EXP_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample stimuli from a named distribution.

    Args:
        n_trials: Number of trials
        distribution: 'Uniform', 'Hard-A', 'Hard-B'
                      (case-insensitive, underscores and hyphens equivalent)
        rng: Random number generator
        boundary: Category boundary
        rate: exponential rate of the hard side (default: the rig's)

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
    return samplers[key](n_trials, rng=rng, boundary=boundary, rate=rate)


def compute_distribution_density(
    distribution: str,
    s: np.ndarray,
    boundary: float = 0.0,
    rate: float = HARD_EXP_RATE,
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
        rate: exponential rate of the hard side (default: the rig's).

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
        p_a[on_a] = hard_side_density(-s_shift[on_a], rate)
        p_b[on_b] = 1.0
    elif key == 'hard_b':
        p_a[on_a] = 1.0
        p_b[on_b] = hard_side_density(s_shift[on_b], rate)
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
    rate: float = HARD_EXP_RATE,
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
        rate: exponential rate of the hard side (default: the rig's).

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
        def integrand(s):
            if key == 'hard_a':
                return norm.pdf(x, s + boundary, sigma) * float(hard_side_density(-s, rate))
            return norm.pdf(x, s + boundary, sigma)
        return quad(integrand, -1, 0, limit=100)[0]

    def p_x_given_b(x: float) -> float:
        def integrand(s):
            if key == 'hard_b':
                return norm.pdf(x, s + boundary, sigma) * float(hard_side_density(s, rate))
            return norm.pdf(x, s + boundary, sigma)
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
