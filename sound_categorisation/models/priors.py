"""Prior variants: the parameter ranges BE and SC are simulated over.

    manuscript   the four model parameters over the ranges the first networks were trained on
                 (``BEParams.get_bounds`` / ``SCParams.get_bounds``)
    learning     the same four parameters with the learning parameters over wider ranges
                 (``WIDE_BOUNDS``)

``settings.SBI_PRIOR`` names the variant the model fits use now (networks, grid, synthetic cohorts,
SBC draws); every function here takes ``variant=None`` to mean that one. A saved network records the
variant it was trained on. docs/prior_variants.md has why the wider ranges are what they are, what
they gain and what they cost.

A variant is an ordered ``{parameter: (low, high)}`` per model; every parameter is drawn uniformly
between its bounds, except those in ``LOG_UNIFORM`` under ``learning`` (drawn uniformly in the
logarithm, so that a range spanning orders of magnitude is covered at its low end too).

*Flat space* is where a variant's prior is uniform: the parameter itself, or its natural logarithm
for a log-uniform one (``to_flat`` / ``from_flat`` / ``flat_bounds``). The networks are trained in
flat space, and recovery is measured there, since "relative to the prior's width" only means
something where the prior is flat. Everything a caller sees is in the parameters' own units.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from sound_categorisation.models.be_core import BEParams
from sound_categorisation.models.sc_core import SCParams

PRIOR_VARIANTS = ('manuscript', 'learning')

# The wider learning ranges, chosen from a scan of the simulator (docs/prior_variants.md).
# BE: lower learning and relaxation rates, drawn uniformly in the logarithm (slow learning with
# little relaxation is where BE is most accurate and least history-dependent).
# SC: lower retention (faster updating), the one direction in which SC's statistics change.
WIDE_BOUNDS = {
    'be': {'eta_learning': (0.005, 0.9), 'eta_relax': (0.001, 0.4)},
    'sc': {'gamma': (0.30, 0.999)},
}
LOG_UNIFORM = {'be': ('eta_learning', 'eta_relax'), 'sc': ()}

_PARAMS = {'be': BEParams, 'sc': SCParams}


def _key(model) -> str:
    key = str(getattr(model, 'value', model)).lower()
    if key not in _PARAMS:
        raise ValueError(f"Unknown model: {model!r} (expected 'be' or 'sc')")
    return key


def current() -> str:
    """The variant the model fits use now (``settings.SBI_PRIOR``)."""
    from sound_categorisation.settings import SBI_PRIOR
    return SBI_PRIOR


def resolve(variant: str | None) -> str:
    """A variant name, with None meaning the one in use; an unknown name is an error."""
    variant = current() if variant is None else variant
    if variant not in PRIOR_VARIANTS:
        raise ValueError(f'Unknown prior variant: {variant!r} (expected one of {PRIOR_VARIANTS})')
    return variant


def prior_bounds(model, variant: str | None = None) -> Dict[str, Tuple[float, float]]:
    """``{parameter: (low, high)}`` of a model under a variant, in the order the draws are made."""
    key, variant = _key(model), resolve(variant)
    manuscript = _PARAMS[key].get_bounds()
    bounds = {name: tuple(manuscript[name]) for name in _PARAMS[key].get_param_names()}
    if variant == 'learning':
        bounds.update(WIDE_BOUNDS[key])
    return bounds


def log_uniform_params(model, variant: str | None = None) -> Tuple[str, ...]:
    """The parameters of a model drawn uniformly in the logarithm under a variant."""
    key = _key(model)
    return tuple(LOG_UNIFORM[key]) if resolve(variant) == 'learning' else ()


def bounds_arrays(model, variant: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """``(low, high)`` arrays in parameter order, in the parameters' own units."""
    bounds = prior_bounds(model, variant)
    return (np.array([lo for lo, _ in bounds.values()], dtype=float),
            np.array([hi for _, hi in bounds.values()], dtype=float))


def _logged(model, variant: str | None) -> np.ndarray:
    logs = log_uniform_params(model, variant)
    return np.array([name in logs for name in prior_bounds(model, variant)])


def to_flat(model, variant: str | None, thetas) -> np.ndarray:
    """Parameters (last axis in parameter order) into flat space: log-uniform ones are logged."""
    out = np.array(thetas, dtype=float)
    logged = _logged(model, variant)
    out[..., logged] = np.log(out[..., logged])
    return out


def from_flat(model, variant: str | None, thetas) -> np.ndarray:
    """Flat space back into the parameters' own units (the inverse of ``to_flat``)."""
    out = np.array(thetas, dtype=float)
    logged = _logged(model, variant)
    out[..., logged] = np.exp(out[..., logged])
    return out


def flat_bounds(model, variant: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """``(low, high)`` arrays in flat space: the box the prior is uniform over."""
    low, high = bounds_arrays(model, variant)
    return to_flat(model, variant, low), to_flat(model, variant, high)


def sample_prior(model, variant: str | None, n: int,
                 rng: np.random.Generator) -> Tuple[List[str], np.ndarray]:
    """``n`` draws of a model's parameters under a variant, in the parameters' own units.

    Returns ``(names, thetas)`` with ``thetas`` of shape ``(n, len(names))``: uniform draws in
    flat space, mapped back. Under ``manuscript`` this is ``rng.uniform(low, high, size=(n, 4))``:
    the prior the first networks were trained on.
    """
    names = list(prior_bounds(model, variant))
    low, high = flat_bounds(model, variant)
    return names, from_flat(model, variant, rng.uniform(low, high, size=(int(n), len(names))))
