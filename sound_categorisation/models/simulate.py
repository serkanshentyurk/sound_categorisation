"""Simulating the BE and SC models.

``simulate_choices`` turns ``(model, parameter dict, stimuli, categories)`` into choices by running a model
forward from a fresh burn-in. The SBI simulator's sessions, the synthetic cohort, the SBC calibration, model
selection and the feature diagnostics all draw their simulations through it.

``BoundaryEstimation`` and ``StatisticalCategorisation`` are the same two models as behav_utils models
(``behav_utils.models.Model``): ``simulate(design, rng)`` runs one on a ``Design``, so behav_utils'
``simulate``, ``simulate_responses`` and synthetic generators (``generate_synthetic_session``,
``generate_synthetic_animal``) work with them. ``make_model`` builds one from a model identifier and a
parameter dict. A run draws one integer seed from ``rng`` and is then ``simulate_choices`` with that seed,
so a model and the function give the same choices.

The model identifier is duck-typed: a ``ModelType`` (its ``.value`` is used) or 'be' / 'sc'. This module
therefore does not import ``inference`` (which imports models).
"""

from typing import ClassVar

import numpy as np
from behav_utils.models import Design, Model, Responses


def _model_key(model) -> str:
    """Normalise a model identifier to 'be'/'sc'. Accepts ModelType or str."""
    key = getattr(model, 'value', model)          # ModelType.BE -> 'be'
    return str(key).lower()


def _run(key: str, params, stimuli: np.ndarray, categories: np.ndarray, burn_in: int, seed: int,
         no_response: np.ndarray | None = None, update_mask: np.ndarray | None = None) -> np.ndarray:
    """One run of BE or SC: the burn-in state and the run's own draws are both seeded by ``seed``."""
    if key == 'be':
        from sound_categorisation.models.be_core import BEModel as core
        from sound_categorisation.models.be_core import BEParams as params_type
    elif key == 'sc':
        from sound_categorisation.models.sc_core import SCModel as core
        from sound_categorisation.models.sc_core import SCParams as params_type
    else:
        raise ValueError(f"Unknown model: {key!r} (expected ModelType or 'be'/'sc')")
    p = params if isinstance(params, params_type) else params_type(**params)
    rng = np.random.default_rng(seed)
    state = core.create_initial_state(burn_in=burn_in, params=p, seed=seed)
    choices, _, _, _ = core.simulate_session(p, state, stimuli, categories, rng, no_response=no_response,
                                             return_history=False, update_mask=update_mask)
    return choices


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
    return _run(_model_key(model), params, stimuli, categories, burn_in, seed)


class _CategorisationModel(Model):
    """BE or SC on a behav_utils ``Design`` (see the module docstring)."""

    requires: ClassVar[frozenset] = frozenset({'binary', 'stimulus_axis', 'scored'})
    key: ClassVar[str] = ''

    def __init__(self, params, burn_in: int = 1000, laser_blocks_update: bool = False):
        """
        Args:
            params: a parameter dict (or BEParams / SCParams).
            burn_in: uniform trials the model has seen before each run (a fresh burn-in per run).
            laser_blocks_update: True: on a laser trial the model chooses as usual but does not learn from
                its feedback (inactivation during the update window). False (default): the laser is ignored.
        """
        self.params = params
        self.burn_in = int(burn_in)
        self.laser_blocks_update = bool(laser_blocks_update)

    def simulate(self, design: Design, rng: np.random.Generator) -> Responses:
        """The model's choices to ``design``. The task's truth is the feedback; aborted and no-response trials
        are skipped (no choice, no learning), and so is the learning on a trial without a correct response."""
        seed = int(rng.integers(0, 2**31 - 1))
        truth = np.asarray(design.truth, dtype=float)
        known = ~np.isnan(truth)
        categories = np.where(known, truth, 0.0).astype(int)
        skipped = design.abort | design.no_response
        blocked = design.laser if self.laser_blocks_update else np.zeros(design.n_trials, dtype=bool)
        learns = known & ~blocked
        choices = _run(self.key, self.params, design.stimulus, categories, self.burn_in, seed,
                       no_response=skipped if skipped.any() else None,
                       update_mask=None if learns.all() else learns)
        return Responses(choice=np.asarray(choices, dtype=float))

    def __repr__(self) -> str:
        laser = ', laser_blocks_update=True' if self.laser_blocks_update else ''
        return f'{type(self).__name__}({self.params!r}, burn_in={self.burn_in}{laser})'


class BoundaryEstimation(_CategorisationModel):
    """The BE model (models/be_core.py) as a behav_utils model."""

    key: ClassVar[str] = 'be'


class StatisticalCategorisation(_CategorisationModel):
    """The SC model (models/sc_core.py) as a behav_utils model."""

    key: ClassVar[str] = 'sc'


def make_model(model, params, burn_in: int = 1000, laser_blocks_update: bool = False) -> _CategorisationModel:
    """BoundaryEstimation or StatisticalCategorisation from a model identifier and its parameters."""
    classes = {'be': BoundaryEstimation, 'sc': StatisticalCategorisation}
    key = _model_key(model)
    if key not in classes:
        raise ValueError(f"Unknown model: {model!r} (expected ModelType or 'be'/'sc')")
    return classes[key](params, burn_in=burn_in, laser_blocks_update=laser_blocks_update)
