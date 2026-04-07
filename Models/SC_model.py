"""
Stimulus-Category Model — High-level Interface

Stateful wrapper around the stateless SCModel core for convenient
interactive use: holds parameters and belief state, provides simulation.

For inference (SBI), use the core directly:
    from models.SC_core import SCParams, SCState, SCModel

For interactive use and exploration:
    from models.SC_model import StimulusCategoryModel

Usage:
    model = StimulusCategoryModel(
        sigma_percep=0.15, A_repulsion=0.1,
        gamma=0.95, sigma_update=0.3,
    )
    model.reset_belief(burn_in=1000)
    choices, p_B = model.simulate_session(stimuli, categories)
"""

import numpy as np
from typing import Optional, Dict, Tuple

from models.SC_core import SCParams, SCState, SCModel
from models.BE_core import ModelTrace


class StimulusCategoryModel:
    """
    Stimulus-Category (SC) model for sound categorisation.

    Stateful wrapper around the functional SCModel core.  Holds parameters
    and belief state internally for convenient interactive use.

    Interface mirrors BoundaryEstimationModel.

    Parameters:
        sigma_percep:  Perceptual noise standard deviation
        A_repulsion:   Serial dependence strength
        gamma:         Retention factor (1 − learning rate)
        sigma_update:  Gaussian bump width for belief updates
    """

    def __init__(
        self,
        sigma_percep: float,
        A_repulsion: float,
        gamma: float,
        sigma_update: float,
        x_min: float = None,
        x_max: float = None,
        n_points: int = None,
    ):
        self._params = SCParams(
            sigma_percep=sigma_percep,
            A_repulsion=A_repulsion,
            gamma=gamma,
            sigma_update=sigma_update,
        )

        if x_min is None or x_max is None or n_points is None:
            _x_min, _x_max, _n_pts = self._params.stimulus_space_bounds()
            x_min = x_min if x_min is not None else _x_min
            x_max = x_max if x_max is not None else _x_max
            n_points = n_points if n_points is not None else _n_pts

        self._x_min = x_min
        self._x_max = x_max
        self._n_points = n_points

        self._state = SCState.initial_default(x_min, x_max, n_points)

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def params(self) -> SCParams:
        return self._params

    @property
    def state(self) -> SCState:
        return self._state

    @property
    def sigma_percep(self) -> float:
        return self._params.sigma_percep

    @property
    def A_repulsion(self) -> float:
        return self._params.A_repulsion

    @property
    def gamma(self) -> float:
        return self._params.gamma

    @property
    def sigma_update(self) -> float:
        return self._params.sigma_update

    @property
    def x(self) -> np.ndarray:
        return self._state.x

    @property
    def x_min(self) -> float:
        return self._x_min

    @property
    def x_max(self) -> float:
        return self._x_max

    @property
    def n_points(self) -> int:
        return self._n_points

    @property
    def A_distribution(self) -> np.ndarray:
        return self._state.A_distribution

    @property
    def B_distribution(self) -> np.ndarray:
        return self._state.B_distribution

    @property
    def s_hat_prev(self) -> Optional[float]:
        return self._state.s_hat_prev

    # -----------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------

    def reset_belief(
        self,
        A_dist: Optional[np.ndarray] = None,
        B_dist: Optional[np.ndarray] = None,
        burn_in: int = 0,
        burn_in_seed: int = 42,
        uniform_init: bool = False,
    ):
        """
        Reset category distributions to defaults, custom, or via burn-in.

        Args:
            A_dist, B_dist: Custom initial distributions.
                            If provided, burn_in is ignored.
            burn_in: Number of simulated trials to run first.
            burn_in_seed: Random seed for burn-in.
            uniform_init: If True (and no custom dists), start from flat.
        """
        if A_dist is not None and B_dist is not None:
            self._state = SCState.from_distributions(
                A_dist, B_dist, self._x_min, self._x_max,
            )
        elif uniform_init:
            self._state = SCState.initial_uniform(
                self._x_min, self._x_max, self._n_points,
            )
            if burn_in > 0:
                self._state = SCModel.run_burn_in(
                    self._params, self._state, burn_in, burn_in_seed,
                )
        elif burn_in > 0:
            initial = SCState.initial_default(
                self._x_min, self._x_max, self._n_points,
            )
            self._state = SCModel.run_burn_in(
                self._params, initial, burn_in, burn_in_seed,
            )
        else:
            self._state = SCState.initial_default(
                self._x_min, self._x_max, self._n_points,
            )

    def get_params(self) -> Dict[str, float]:
        return self._params.to_dict()

    def get_state(self) -> Dict:
        return {
            'params': self._params.to_dict(),
            'A_distribution': self._state.A_distribution.copy(),
            'B_distribution': self._state.B_distribution.copy(),
            's_hat_prev': self._state.s_hat_prev,
            'x': self._state.x.copy(),
            'x_min': self._x_min,
            'x_max': self._x_max,
            'n_points': self._n_points,
        }

    # -----------------------------------------------------------------
    # Simulation
    # -----------------------------------------------------------------

    def simulate_session(
        self,
        stimuli: np.ndarray,
        categories: np.ndarray,
        no_response: Optional[np.ndarray] = None,
        not_blockstart: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        store_history: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate choices for a session.

        Returns:
            choices: Simulated choice sequence (0=A, 1=B, NaN=no response)
            p_B: P(choose B) at each trial
        """
        if rng is None:
            rng = np.random.default_rng()

        choices, p_B, self._state, trace = SCModel.simulate_session(
            self._params, self._state, stimuli, categories, rng,
            no_response, not_blockstart, return_history=store_history,
        )
        if store_history:
            self._trace = trace

        return choices, p_B

    def get_model_trace(self) -> Optional[ModelTrace]:
        """Get ModelTrace from last simulation (if store_history=True)."""
        return getattr(self, '_trace', None)

    # Backward compat alias
    get_trial_history = get_model_trace

    # -----------------------------------------------------------------
    # Class methods
    # -----------------------------------------------------------------

    @classmethod
    def get_bounds(cls) -> Dict[str, Tuple[float, float]]:
        return SCParams.get_bounds()

    @classmethod
    def get_param_names(cls):
        return SCParams.get_param_names()


__all__ = ['StimulusCategoryModel']
