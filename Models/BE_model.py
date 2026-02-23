"""
Boundary Estimation Model - High-level Interface

Stateful wrapper around the stateless BEModel core for convenient
interactive use: holds parameters and belief state, provides simulation.

For inference (SBI), use the core directly:
    from Models.BE_core import BEParams, BEState, BEModel

For interactive use and exploration:
    from Models.BE_model import BoundaryEstimationModel

Usage:
    # Simulation
    model = BoundaryEstimationModel(sigma_percep=0.15, A_repulsion=0.1,
                                    eta_learning=0.35, eta_relax=0.12)
    model.reset_belief(burn_in=1000)
    choices, p_B = model.simulate_session(stimuli, categories)
    
    # Access core components
    params = model.params  # BEParams
    state = model.state    # BEState
    trace = model.get_model_trace()  # ModelTrace (if store_history=True)
"""

import numpy as np
from typing import Optional, Dict, Tuple

from Models.BE_core import BEParams, BEState, BEModel, ModelTrace


# =============================================================================
# BOUNDARY ESTIMATION MODEL CLASS
# =============================================================================

class BoundaryEstimationModel:
    """
    Boundary Estimation (BE) model for sound categorisation.
    
    Stateful wrapper around the functional BEModel core. Holds parameters
    and belief state internally for convenient interactive use.
    
    Parameters:
        sigma_percep: Perceptual noise standard deviation
        A_repulsion: Serial dependence strength (repulsion from previous trial)
        eta_learning: Learning rate for boundary belief updates
        eta_relax: Relaxation rate toward uniform distribution
    
    Attributes:
        params: BEParams object (immutable parameters)
        state: BEState object (mutable belief state)
    """
    
    # =========================================================================
    # INITIALISATION
    # =========================================================================
    
    def __init__(self, sigma_percep: float, A_repulsion: float,
                 eta_learning: float, eta_relax: float,
                 x_min: float = None,
                 x_max: float = None,
                 n_points: int = None):
        """
        Initialise model with parameters.
        
        Args:
            sigma_percep: Perceptual noise standard deviation
            A_repulsion: Strength of serial dependence (repulsion)
            eta_learning: Learning rate for boundary belief updates
            eta_relax: Relaxation rate toward uniform distribution
            x_min, x_max, n_points: Stimulus space bounds and resolution.
                If not provided, computed automatically from params via
                BEParams.stimulus_space_bounds() to match original BE convention.
        """
        self._params = BEParams(
            sigma_percep=sigma_percep,
            A_repulsion=A_repulsion,
            eta_learning=eta_learning,
            eta_relax=eta_relax
        )
        
        # Compute grid from params if not explicitly provided
        if x_min is None or x_max is None or n_points is None:
            _x_min, _x_max, _n_pts = self._params.stimulus_space_bounds()
            x_min    = x_min    if x_min    is not None else _x_min
            x_max    = x_max    if x_max    is not None else _x_max
            n_points = n_points if n_points is not None else _n_pts
        
        self._x_min = x_min
        self._x_max = x_max
        self._n_points = n_points
        
        self._state = BEState.initial_uniform(x_min, x_max, n_points)
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def params(self) -> BEParams:
        """Access underlying BEParams object."""
        return self._params
    
    @property
    def state(self) -> BEState:
        """Access underlying BEState object."""
        return self._state
    
    @property
    def sigma_percep(self) -> float:
        return self._params.sigma_percep
    
    @property
    def A_repulsion(self) -> float:
        return self._params.A_repulsion
    
    @property
    def eta_learning(self) -> float:
        return self._params.eta_learning
    
    @property
    def eta_relax(self) -> float:
        return self._params.eta_relax
    
    @property
    def sigma_boundary(self) -> float:
        return self._params.sigma_boundary
    
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
    def boundary_belief(self) -> np.ndarray:
        return self._state.boundary_belief
    
    @property
    def s_hat_prev(self) -> Optional[float]:
        return self._state.s_hat_prev
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def reset_belief(self, belief: Optional[np.ndarray] = None,
                     burn_in: int = 0, burn_in_seed: int = 42):
        """
        Reset boundary belief to uniform, custom distribution, or via burn-in.
        
        Args:
            belief: Custom initial belief distribution (must match self.x length).
                    If provided, burn_in is ignored.
            burn_in: Number of simulated expert trials to run before session.
                     Allows belief to converge to a sensible boundary estimate.
                     If 0, starts with uniform belief.
            burn_in_seed: Random seed for burn-in simulation
        """
        if belief is not None:
            self._state = BEState.from_belief(
                belief, self._x_min, self._x_max, s_hat_prev=None
            )
        elif burn_in > 0:
            initial = BEState.initial_uniform(self._x_min, self._x_max, self._n_points)
            self._state = BEModel.run_burn_in(
                self._params, initial, burn_in, burn_in_seed
            )
        else:
            self._state = BEState.initial_uniform(
                self._x_min, self._x_max, self._n_points
            )
    
    def get_belief_copy(self) -> np.ndarray:
        """Return copy of current boundary belief distribution."""
        return self._state.boundary_belief.copy()
    
    def set_belief(self, belief: np.ndarray):
        """
        Set boundary belief distribution (for carrying across sessions).
        
        Args:
            belief: New belief distribution (will be normalised)
        """
        self._state = BEState.from_belief(
            belief, self._x_min, self._x_max, self._state.s_hat_prev
        )
    
    def get_params(self) -> Dict[str, float]:
        """Return current parameters as dict."""
        return self._params.to_dict()
    
    def get_state(self) -> Dict:
        """Return complete model state (for checkpointing)."""
        return {
            'params': self._params.to_dict(),
            'boundary_belief': self._state.boundary_belief.copy(),
            's_hat_prev': self._state.s_hat_prev,
            'x': self._state.x.copy(),
            'x_min': self._x_min,
            'x_max': self._x_max,
            'n_points': self._n_points
        }
    
    # =========================================================================
    # SIMULATION
    # =========================================================================
    
    def simulate_session(self, stimuli: np.ndarray, categories: np.ndarray,
                         no_response: Optional[np.ndarray] = None,
                         not_blockstart: Optional[np.ndarray] = None,
                         rng: Optional[np.random.Generator] = None,
                         store_history: bool = False
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate choices for a session.
        
        Args:
            stimuli: Array of stimulus values
            categories: Array of true categories (for feedback)
            no_response: Boolean array (True = skip trial)
            not_blockstart: Boolean array (True = not start of block/session)
            rng: Numpy random generator
            store_history: If True, store ModelTrace for post-hoc analysis
        
        Returns:
            choices: Simulated choice sequence (0 = A, 1 = B, NaN = no response)
            p_B_sequence: P(choose B) at each trial
        """
        if rng is None:
            rng = np.random.default_rng()
        
        choices, p_B, self._state, trace = BEModel.simulate_session(
                    self._params, self._state, stimuli, categories, rng,
                    no_response, not_blockstart, return_history=store_history
                )
        if store_history:
            self._trace = trace
        
        return choices, p_B
    
    def get_model_trace(self) -> Optional[ModelTrace]:
        """
        Get the ModelTrace from the last simulation (if store_history=True).
        
        Returns:
            ModelTrace object or None if no history was stored
        """
        return getattr(self, '_trace', None)
    
    # Backward compat alias
    get_trial_history = get_model_trace
    
    # =========================================================================
    # CLASS METHODS
    # =========================================================================
    
    @classmethod
    def get_bounds(cls) -> Dict[str, Tuple[float, float]]:
        """Parameter bounds."""
        return BEParams.get_bounds()
    
    @classmethod
    def get_param_names(cls):
        """Parameter names in canonical order."""
        return BEParams.get_param_names()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BoundaryEstimationModel',
]
