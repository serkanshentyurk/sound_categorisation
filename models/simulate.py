"""Single params -> choices simulation core for the BE/SC models.

One place that turns ``(model, parameter dict, stimuli, categories)`` into a
choice array by running the model forward. Both the SBI simulator and
grid-search call this, so the BE/SC branching is not duplicated.

The model identifier is duck-typed: pass a ``ModelType`` (its ``.value`` is
used) or the lowercase string ``'be'`` / ``'sc'``. This module therefore does
NOT import ``inference`` (which would create a cycle, since inference imports
models).
"""

import numpy as np


def _model_key(model) -> str:
    """Normalise a model identifier to 'be'/'sc'. Accepts ModelType or str."""
    key = getattr(model, 'value', model)          # ModelType.BE -> 'be'
    return str(key).lower()


def simulate_choices(
    model,
    params: dict,
    stimuli: np.ndarray,
    categories: np.ndarray,
    burn_in: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Run a model forward and return its choices.

    Args:
        model: ModelType or 'be'/'sc'.
        params: Parameter dict for the model
            (e.g. {'sigma_percep': ..., 'A_repulsion': ..., ...}).
        stimuli: Stimulus values, shape (T,).
        categories: True categories, shape (T,).
        burn_in: Burn-in trials used to initialise the model state.
        seed: RNG seed (used for both state init and the choice draw).

    Returns:
        choices: shape (T,), binary.
    """
    key = _model_key(model)
    rng = np.random.default_rng(seed)

    if key == 'be':
        from models.BE_core import BEParams, BEModel
        p = BEParams(**params)
        state = BEModel.create_initial_state(
            burn_in=burn_in, params=p, seed=seed)
        choices, _, _, _ = BEModel.simulate_session(
            p, state, stimuli, categories, rng, return_history=False)
    elif key == 'sc':
        from models.SC_core import SCParams, SCModel
        p = SCParams(**params)
        state = SCModel.create_initial_state(
            burn_in=burn_in, params=p, seed=seed)
        choices, _, _, _ = SCModel.simulate_session(
            p, state, stimuli, categories, rng, return_history=False)
    else:
        raise ValueError(
            f"Unknown model: {model!r} (expected ModelType or 'be'/'sc')")

    return choices
