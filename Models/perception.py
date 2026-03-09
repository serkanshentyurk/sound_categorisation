"""
Shared Perception Pipeline

Perceptual processing shared by all categorisation models (BE, SC).
Implements stimulus perception with noise and serial dependence.

Both the BE and SC models share the same front-end:
    true stimulus → perceptual noise → serial repulsion → perceived stimulus

This module extracts that shared computation so changes to perception
(e.g., adding attraction, distance-dependent noise, or multi-modal
perception) propagate to all models automatically.

Usage:
    from Models.perception import perceive_stimulus, stimulus_space_bounds

    s_hat = perceive_stimulus(s_t, sigma_percep, A_repulsion, s_hat_prev, rng)
    x_min, x_max, n_points = stimulus_space_bounds(sigma_percep, A_repulsion)
"""

import numpy as np
from typing import Optional, Tuple


# =============================================================================
# STIMULUS PERCEPTION
# =============================================================================

def perceive_stimulus(
    s_t: float,
    sigma_percep: float,
    A_repulsion: float,
    s_hat_prev: Optional[float],
    rng: np.random.Generator,
) -> float:
    """
    Apply perceptual noise and serial dependence (repulsion).

    Pipeline:
        1. Add Gaussian noise: s_tilde = s_t + N(0, sigma_percep)
        2. Apply repulsion from previous perceived stimulus:
           s_hat = s_tilde + A * (s_tilde - s_hat_prev) * exp(-|s_tilde - s_hat_prev|)

    The repulsion function pushes the current percept away from the previous
    one, with magnitude decaying exponentially with distance. This produces
    an attractive–repulsive profile matching psychophysical serial dependence
    data (Fischer & Whitney 2014).

    Args:
        s_t: True stimulus value
        sigma_percep: Perceptual noise standard deviation (>0)
        A_repulsion: Repulsion strength (>=0; 0 = no serial dependence)
        s_hat_prev: Previous perceived stimulus (None for first trial)
        rng: NumPy random number generator

    Returns:
        s_hat: Perceived stimulus value
    """
    # Perceptual noise
    noise = rng.normal(0, sigma_percep)
    s_tilde = s_t + noise

    # Repulsion from previous trial
    if s_hat_prev is not None:
        diff = s_tilde - s_hat_prev
        repulsion = A_repulsion * diff * np.exp(-np.abs(diff))
        s_hat = s_tilde + repulsion
    else:
        s_hat = s_tilde

    return s_hat


# =============================================================================
# STIMULUS SPACE BOUNDS
# =============================================================================

def stimulus_space_bounds(
    sigma_percep: float,
    A_repulsion: float,
    stim_half_range: float = 1.0,
    n_sigma: float = 6.0,
) -> Tuple[float, float, int]:
    """
    Compute stimulus space bounds for the discretisation grid.

    The grid must extend beyond the nominal stimulus range [-1, 1] to
    accommodate perceptual noise and repulsion. Without this, s_hat values
    near the edges would be clipped to the grid boundary, distorting CDF
    computations and belief updates.

    Convention (matching original BE code):
        max_range = stim_range + n_sigma * sigma + 2 * A * (stim_range + n_sigma * sigma)
        n_points  = round((x_max - x_min) * 1000)

    Args:
        sigma_percep: Perceptual noise standard deviation
        A_repulsion: Repulsion strength
        stim_half_range: Half-width of nominal stimulus range (default 1.0)
        n_sigma: Number of sigma to extend beyond nominal range (default 6)

    Returns:
        x_min, x_max, n_points
    """
    extension = n_sigma * sigma_percep
    half = stim_half_range + extension + 2 * A_repulsion * (stim_half_range + extension)
    x_min, x_max = -half, half
    n_points = round((x_max - x_min) * 1000)
    return x_min, x_max, n_points


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'perceive_stimulus',
    'stimulus_space_bounds',
]
