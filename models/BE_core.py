"""
Stateless Boundary Estimation Model Core

This module provides the core computational components for the BE model
in a stateless, functional style that supports:
- SBI inference (single and multi-session)
- Clean session chaining for longitudinal analysis

Components:
    BEParams: Immutable parameter container
    BEState: Model state (belief distribution, previous stimulus)
    BEModel: Stateless operations (simulate, compute likelihood, etc.)

Usage:
    # Single session simulation
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1, 
                      eta_learning=0.35, eta_relax=0.12)
    state = BEState.initial_uniform()
    choices, p_B, final_state = BEModel.simulate_session(
        params, state, stimuli, categories, rng
    )
    
    # Multi-session with state chaining
    state = BEState.initial_uniform()
    for session_idx, (stim, cat) in enumerate(sessions):
        params = params_per_session[session_idx]
        choices, _, state = BEModel.simulate_session(
            params, state, stim, cat, rng
        )
    
    # Log-likelihood for inference
    ll, trial_lls, _ = BEModel.compute_log_likelihood(
        params, state, stimuli, categories, observed_choices, rng
    )
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Union
from scipy.integrate import trapezoid

from models.perception import perceive_stimulus as _perceive_stimulus
from models.perception import stimulus_space_bounds as _stimulus_space_bounds


# =============================================================================
# TRIAL HISTORY CONTAINER
# =============================================================================

@dataclass
class ModelTrace:
    """
    Record of model computation across trials (model output).
    
    Stores everything the BE model computed for a session, including the
    input arrays it operated on. Used for post-hoc analysis: update matrices,
    belief visualisation, model diagnostics.
    
    This is MODEL OUTPUT â€” created by BEModel.simulate_session or
    BEModel.compute_log_likelihood. For experimental INPUT data, see
    Data.structures.TrialData.
    
    Attributes:
        # Input arrays (what the model received)
        stimuli: (n_trials,) actual stimulus values
        categories: (n_trials,) true categories (0=A, 1=B)
        choices: (n_trials,) simulated or observed choices (0=A, 1=B, NaN=no response)
        no_response: (n_trials,) boolean mask for no-response trials
        not_blockstart: (n_trials,) boolean mask (True = not start of block)
        
        # Model outputs (what the model computed)
        p_B: (n_trials,) model's P(choose B) at actual stimulus
        s_hat: (n_trials,) perceived stimulus (includes noise + repulsion)
        beliefs: (n_trials, n_points) full belief distributions before each trial
        x: (n_points,) discretisation grid for beliefs
    
    Usage:
        # After simulation
        choices, p_B, final_state, trace = BEModel.simulate_session(
            params, state, stimuli, categories, rng, return_history=True
        )
        
        # From experimental TrialData + model outputs
        trace = ModelTrace.from_trial_data(trial_data, p_B, s_hat, beliefs, x)
        
        # Compute update matrix
        update_matrix = compute_model_update_matrix(trace, method='deterministic')
    """
    # Input arrays
    stimuli: np.ndarray
    categories: np.ndarray
    choices: np.ndarray
    no_response: np.ndarray
    not_blockstart: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Model outputs
    p_B: np.ndarray = field(default_factory=lambda: np.array([]))
    s_hat: np.ndarray = field(default_factory=lambda: np.array([]))
    beliefs: np.ndarray = field(default_factory=lambda: np.array([]))  # (n_trials, n_points)
    x: np.ndarray = field(default_factory=lambda: np.array([]))  # (n_points,)
    
    # SC-specific (populated by SCModel, empty for BE)
    beliefs_A: np.ndarray = field(default_factory=lambda: np.array([]))
    beliefs_B: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        """Validate and set defaults."""
        n_trials = len(self.stimuli)
        
        # Default not_blockstart: first trial is block start, rest are not
        if len(self.not_blockstart) == 0:
            self.not_blockstart = np.ones(n_trials, dtype=bool)
            if n_trials > 0:
                self.not_blockstart[0] = False
    
    # =========================================================================
    # FACTORY METHODS
    # =========================================================================
    
    @classmethod
    def from_trial_data(
        cls,
        trial_data: 'Any',
        p_B: np.ndarray,
        s_hat: np.ndarray,
        beliefs: np.ndarray,
        x: np.ndarray,
    ) -> 'ModelTrace':
        """
        Create ModelTrace from a TrialData object and model outputs.
        
        trial_data should be from a pre-filtered session.
        """
        from behav_utils.data.filtering import get_arrays
        arrays = get_arrays(trial_data)
        
        # Add not_blockstart (models need this)
        n = arrays['n_trials']
        not_blockstart = np.ones(n, dtype=bool)
        if n > 0:
            not_blockstart[0] = False
        
        return cls(
            stimuli=arrays['stimuli'],
            categories=arrays['categories'],
            choices=arrays['choices'],
            no_response=arrays['no_response'],
            not_blockstart=not_blockstart,
            p_B=p_B,
            s_hat=s_hat,
            beliefs=beliefs,
            x=x,
        )
    
    @classmethod
    def from_arrays(
        cls,
        stimuli: np.ndarray,
        categories: np.ndarray,
        choices: np.ndarray,
        p_B: np.ndarray,
        s_hat: np.ndarray,
        beliefs: np.ndarray,
        x: np.ndarray,
        no_response: Optional[np.ndarray] = None,
        not_blockstart: Optional[np.ndarray] = None,
    ) -> 'ModelTrace':
        """
        Create ModelTrace from raw arrays (e.g., from simulation).
        
        Convenience constructor for when you have arrays but not a TrialData object.
        """
        n_trials = len(stimuli)
        if no_response is None:
            no_response = np.isnan(choices)
        if not_blockstart is None:
            not_blockstart = np.ones(n_trials, dtype=bool)
            if n_trials > 0:
                not_blockstart[0] = False
        
        return cls(
            stimuli=stimuli,
            categories=categories,
            choices=choices,
            no_response=no_response,
            not_blockstart=not_blockstart,
            p_B=p_B,
            s_hat=s_hat,
            beliefs=beliefs,
            x=x,
        )
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def n_trials(self) -> int:
        return len(self.stimuli)
    
    @property
    def n_points(self) -> int:
        return len(self.x)
    
    @property
    def has_beliefs(self) -> bool:
        """Whether full BE belief distributions are stored."""
        return self.beliefs.ndim == 2 and self.beliefs.shape[0] > 0
    
    @property
    def has_sc_beliefs(self) -> bool:
        """Whether SC category distributions are stored."""
        return self.beliefs_A.ndim == 2 and self.beliefs_A.shape[0] > 0
    
    @property
    def rewards(self) -> np.ndarray:
        """Compute rewards (1 if choice == category, 0 otherwise)."""
        rewards = (self.choices == self.categories).astype(float)
        rewards[np.isnan(self.choices)] = np.nan
        return rewards
    
    @property
    def belief_means(self) -> np.ndarray:
        """Mean of boundary belief at each trial."""
        if not self.has_beliefs:
            return np.full(self.n_trials, np.nan)
        return trapezoid(self.beliefs * self.x[np.newaxis, :], self.x, axis=1)
    
    @property
    def belief_stds(self) -> np.ndarray:
        """Std of boundary belief at each trial."""
        if not self.has_beliefs:
            return np.full(self.n_trials, np.nan)
        means = self.belief_means
        deviations = (self.x[np.newaxis, :] - means[:, np.newaxis]) ** 2
        variances = trapezoid(self.beliefs * deviations, self.x, axis=1)
        return np.sqrt(variances)
    
    # =========================================================================
    # BELIEF QUERIES
    # =========================================================================
    
    def get_p_B_at_stimulus(self, s: float, trial_idx: int) -> float:
        """
        Compute P(choose B | stimulus=s) using belief at given trial.
        
        This is the CDF of the belief distribution at point s.
        """
        belief = self.beliefs[trial_idx]
        j = np.abs(self.x - s).argmin()
        return trapezoid(belief[:j+1], self.x[:j+1])
    
    def get_p_B_at_midpoints(self, midpoints: np.ndarray, trial_idx: int) -> np.ndarray:
        """
        Compute P(choose B) at multiple stimulus values for a given trial.
        
        Uses cumulative trapezoid to compute the CDF once, then interpolates.
        
        Args:
            midpoints: Array of stimulus values to evaluate
            trial_idx: Which trial's belief to use
        
        Returns:
            Array of P(B) values at each midpoint
        """
        from scipy.integrate import cumulative_trapezoid
        
        belief = self.beliefs[trial_idx]
        # Compute full CDF via cumulative integration
        cdf = np.zeros(len(self.x))
        cdf[1:] = cumulative_trapezoid(belief, self.x)
        
        # Interpolate at requested points
        p_B_values = np.interp(midpoints, self.x, cdf)
        return np.clip(p_B_values, 1e-10, 1 - 1e-10)
    
    def copy(self) -> 'ModelTrace':
        """Create independent copy."""
        return ModelTrace(
            stimuli=self.stimuli.copy(),
            categories=self.categories.copy(),
            choices=self.choices.copy(),
            no_response=self.no_response.copy(),
            not_blockstart=self.not_blockstart.copy(),
            p_B=self.p_B.copy(),
            s_hat=self.s_hat.copy(),
            beliefs=self.beliefs.copy(),
            x=self.x.copy(),
            beliefs_A=self.beliefs_A.copy(),
            beliefs_B=self.beliefs_B.copy(),
        )


# =============================================================================
# PARAMETER CONTAINER
# =============================================================================

@dataclass(frozen=True)
class BEParams:
    """
    Immutable container for BE model parameters.
    
    Parameters:
        sigma_percep: Perceptual noise standard deviation
        A_repulsion: Serial dependence strength (repulsion from previous trial)
        eta_learning: Learning rate for boundary belief updates
        eta_relax: Relaxation rate toward uniform distribution
    
    The frozen=True makes this immutable, which is important for:
    - Thread safety in parallel simulations
    - Clarity about what's being sampled in inference
    - Preventing accidental mutation during session chaining
    """
    sigma_percep: float
    A_repulsion: float
    eta_learning: float
    eta_relax: float
    
    @property
    def sigma_boundary(self) -> float:
        """Derived parameter: update precision in sigmoid."""
        return 1.0 / self.sigma_percep
    
    def __post_init__(self):
        """Validate and clamp parameters on creation.
        
        Posterior samples from SBI can slightly exceed bounds,
        so we clamp with a warning rather than raising.
        Uses object.__setattr__ because this is a frozen dataclass.
        """
        import warnings
        
        # sigma_percep: must be > 0
        if self.sigma_percep <= 0:
            clamped = max(self.sigma_percep, 1e-6)
            warnings.warn(
                f"sigma_percep={self.sigma_percep:.6f} clamped to {clamped:.6f}",
                stacklevel=2,
            )
            object.__setattr__(self, 'sigma_percep', clamped)
        
        # A_repulsion: must be >= 0
        if self.A_repulsion < 0:
            clamped = max(self.A_repulsion, 0.0)
            warnings.warn(
                f"A_repulsion={self.A_repulsion:.6f} clamped to {clamped:.6f}",
                stacklevel=2,
            )
            object.__setattr__(self, 'A_repulsion', clamped)
        
        # eta_learning: must be in (0, 1]
        if self.eta_learning <= 0 or self.eta_learning > 1:
            clamped = float(np.clip(self.eta_learning, 1e-6, 1.0))
            warnings.warn(
                f"eta_learning={self.eta_learning:.6f} clamped to {clamped:.6f}",
                stacklevel=2,
            )
            object.__setattr__(self, 'eta_learning', clamped)
        
        # eta_relax: must be in [0, 1)
        if self.eta_relax < 0 or self.eta_relax >= 1:
            clamped = float(np.clip(self.eta_relax, 0.0, 1.0 - 1e-6))
            warnings.warn(
                f"eta_relax={self.eta_relax:.6f} clamped to {clamped:.6f}",
                stacklevel=2,
            )
            object.__setattr__(self, 'eta_relax', clamped)
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'BEParams':
        """Create BEParams from dictionary."""
        return cls(
            sigma_percep=d['sigma_percep'],
            A_repulsion=d['A_repulsion'],
            eta_learning=d['eta_learning'],
            eta_relax=d['eta_relax']
        )
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'BEParams':
        """
        Create BEParams from array.
        
        Order: [sigma_percep, A_repulsion, eta_learning, eta_relax]
        """
        return cls(
            sigma_percep=arr[0],
            A_repulsion=arr[1],
            eta_learning=arr[2],
            eta_relax=arr[3]
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'sigma_percep': self.sigma_percep,
            'A_repulsion': self.A_repulsion,
            'eta_learning': self.eta_learning,
            'eta_relax': self.eta_relax
        }
    
    def to_array(self) -> np.ndarray:
        """
        Convert to array.
        
        Order: [sigma_percep, A_repulsion, eta_learning, eta_relax]
        """
        return np.array([
            self.sigma_percep,
            self.A_repulsion,
            self.eta_learning,
            self.eta_relax
        ])
    
    @classmethod
    def get_bounds(cls) -> Dict[str, Tuple[float, float]]:
        """Parameter bounds for fitting/sampling."""
        return {
            'sigma_percep': (0.05, 0.5),
            'A_repulsion': (0.0, 0.5),
            'eta_learning': (0.05, 0.9),
            'eta_relax': (0.01, 0.4)
        }
    
    @classmethod
    def get_param_names(cls) -> List[str]:
        """Parameter names in canonical order."""
        return ['sigma_percep', 'A_repulsion', 'eta_learning', 'eta_relax']
    
    @classmethod
    def sample_prior(cls, rng: np.random.Generator) -> 'BEParams':
        """Sample from uniform prior over bounds."""
        bounds = cls.get_bounds()
        return cls(
            sigma_percep=rng.uniform(*bounds['sigma_percep']),
            A_repulsion=rng.uniform(*bounds['A_repulsion']),
            eta_learning=rng.uniform(*bounds['eta_learning']),
            eta_relax=rng.uniform(*bounds['eta_relax'])
        )
    
    def stimulus_space_bounds(
        self,
        stim_half_range: float = 1.0,
        n_sigma: float = 6.0,
    ) -> Tuple[float, float, int]:
        """Compute stimulus grid bounds (delegates to shared perception module)."""
        return _stimulus_space_bounds(
            self.sigma_percep, self.A_repulsion, stim_half_range, n_sigma,
        )


# =============================================================================
# STATE CONTAINER
# =============================================================================

@dataclass
class BEState:
    """
    BE model state - carried across trials and sessions.
    
    Attributes:
        boundary_belief: Probability distribution over boundary location
        s_hat_prev: Previous perceived stimulus (for serial dependence)
        x: Discretisation grid points
        x_min: Minimum of stimulus space
        x_max: Maximum of stimulus space
    
    Note: This is mutable (not frozen) because we create new instances
    rather than mutating in place. The copy() method ensures clean
    state separation when needed.
    """
    boundary_belief: np.ndarray
    s_hat_prev: Optional[float]
    x: np.ndarray
    x_min: float
    x_max: float
    relax_target: float = 0.5  # uniform density over NOMINAL stimulus range [-1, 1]

    @classmethod
    def initial_uniform(
        cls,
        x_min: float = -1.0,
        x_max: float = 1.0,
        n_points: int = 500,
        stim_half_range: float = 1.0,
    ) -> 'BEState':
        """
        Create initial state with uniform belief distribution.

        Args:
            x_min, x_max: Grid bounds (should be wider than stim_half_range to
                          accommodate noise and repulsion — see
                          BEParams.stimulus_space_bounds())
            n_points: Grid resolution
            stim_half_range: Half-width of the NOMINAL stimulus range.
                             Relaxation pulls toward a uniform density over
                             [-stim_half_range, stim_half_range], NOT over the
                             extended grid. Default 1.0 matches old code behaviour.
        """
        x = np.linspace(x_min, x_max, n_points)
        belief = np.ones(n_points) / (x_max - x_min)
        belief = belief / trapezoid(belief, x)  # Normalise
        return cls(
            boundary_belief=belief,
            s_hat_prev=None,
            x=x,
            x_min=x_min,
            x_max=x_max,
            relax_target=1.0 / (2.0 * stim_half_range),
        )
    
    @classmethod
    def from_belief(
        cls,
        belief: np.ndarray,
        x_min: float = -1.0,
        x_max: float = 1.0,
        s_hat_prev: Optional[float] = None,
        stim_half_range: float = 1.0,
    ) -> 'BEState':
        """Create state from existing belief distribution."""
        x = np.linspace(x_min, x_max, len(belief))
        belief_norm = belief / trapezoid(belief, x)
        return cls(
            boundary_belief=belief_norm,
            s_hat_prev=s_hat_prev,
            x=x,
            x_min=x_min,
            x_max=x_max,
            relax_target=1.0 / (2.0 * stim_half_range),
        )

    def copy(self) -> 'BEState':
        """Create independent copy of state."""
        return BEState(
            boundary_belief=self.boundary_belief.copy(),
            s_hat_prev=self.s_hat_prev,
            x=self.x,  # Can share - never mutated
            x_min=self.x_min,
            x_max=self.x_max,
            relax_target=self.relax_target,
        )
    
    @property
    def n_points(self) -> int:
        """Number of discretisation points."""
        return len(self.x)
    
    @property
    def belief_mean(self) -> float:
        """Mean of boundary belief distribution."""
        return trapezoid(self.x * self.boundary_belief, self.x)
    
    @property
    def belief_std(self) -> float:
        """Standard deviation of boundary belief distribution."""
        mu = self.belief_mean
        var = trapezoid((self.x - mu)**2 * self.boundary_belief, self.x)
        return np.sqrt(var)
    
    def get_belief_stats(self) -> Tuple[float, float]:
        """Return (mean, std) of boundary belief."""
        return self.belief_mean, self.belief_std


# =============================================================================
# STATELESS MODEL OPERATIONS
# =============================================================================

class BEModel:
    """
    Stateless BE model operations.
    
    All methods are static - they take params and state as arguments
    and return results without side effects. This design enables:
    
    1. Clean separation of concerns (params vs state)
    2. Easy parallelisation (no shared mutable state)
    3. Explicit state flow for multi-session chaining
    4. Simple integration with inference frameworks (MCMC, SBI)
    """
    
    # =========================================================================
    # CORE COMPUTATIONS
    # =========================================================================
    
    @staticmethod
    def perceive_stimulus(s_t: float, params: BEParams,
                          s_hat_prev: Optional[float],
                          rng: np.random.Generator) -> float:
        """Apply perceptual noise and serial dependence."""
        return _perceive_stimulus(
            s_t, params.sigma_percep, params.A_repulsion, s_hat_prev, rng,
        )
    
    @staticmethod
    def get_choice_probability(s_hat: float, state: BEState) -> float:
        """
        Compute P(choose B) given perceived stimulus.
        
        P(choose B) = P(boundary < s_hat) = CDF of boundary belief at s_hat
        
        Args:
            s_hat: Perceived stimulus value
            state: Current model state
        
        Returns:
            P(choose B)
        """
        j = np.abs(state.x - s_hat).argmin()
        p_B = trapezoid(state.boundary_belief[:j+1], state.x[:j+1])
        return np.clip(p_B, 1e-10, 1 - 1e-10)
    
    @staticmethod
    def update_belief(s_hat: float, true_category: int,
                      params: BEParams, state: BEState) -> BEState:
        """
        Update boundary belief based on feedback.
        
        Creates and returns a NEW state - does not mutate input.
        
        Args:
            s_hat: Perceived stimulus value
            true_category: True category (0 = A, 1 = B)
            params: Model parameters
            state: Current model state
        
        Returns:
            New BEState with updated belief
        """
        # C = +1 for category B, -1 for category A
        C = 1 if true_category == 1 else -1
        
        # Learning update (sigmoid)
        delta_learning = 1 / (1 + np.exp(-params.sigma_boundary * C * (state.x - s_hat)))
        y_prime = state.boundary_belief - params.eta_learning * delta_learning
        
        # Relaxation toward uniform density over NOMINAL stimulus range
        delta_relax = y_prime - state.relax_target
        y_double_prime = y_prime - params.eta_relax * delta_relax
        
        # Ensure non-negative
        min_val = np.min(y_double_prime)
        if min_val < 0:
            y_double_prime = y_double_prime + np.abs(min_val)
        
        # Normalise
        new_belief = y_double_prime / trapezoid(y_double_prime, state.x)
        
        return BEState(
            boundary_belief=new_belief,
            s_hat_prev=s_hat,
            x=state.x,
            x_min=state.x_min,
            x_max=state.x_max,
            relax_target=state.relax_target,
        )
        
    @staticmethod
    def _update_belief_inplace(
        s_hat: float, true_category: int,
        params: BEParams,
        belief: np.ndarray, x: np.ndarray,
        x_min: float, x_max: float,
        relax_target: float = 0.5,
    ) -> None:
        """
        Update boundary belief IN-PLACE. For tight simulation loops only.
        
        The public update_belief() returns a new BEState and is used for
        external calls. This avoids allocation overhead in simulate_session
        and compute_log_likelihood.
        """
        C = 1 if true_category == 1 else -1
        sigma_boundary = 1.0 / params.sigma_percep
        
        # Learning update
        delta_learning = 1.0 / (1.0 + np.exp(-sigma_boundary * C * (x - s_hat)))
        belief -= params.eta_learning * delta_learning
        
        # Relaxation toward uniform density over NOMINAL stimulus range
        belief -= params.eta_relax * (belief - relax_target)
        
        # Ensure non-negative
        min_val = belief.min()
        if min_val < 0:
            belief += np.abs(min_val)
        
        # Normalise
        belief /= trapezoid(belief, x)
    
    # =========================================================================
    # SESSION SIMULATION
    # =========================================================================

    @staticmethod
    def simulate_session(
        params: BEParams,
        initial_state: BEState,
        stimuli: np.ndarray,
        categories: np.ndarray,
        rng: np.random.Generator,
        no_response: Optional[np.ndarray] = None,
        not_blockstart: Optional[np.ndarray] = None,
        return_history: bool = False,
        update_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, BEState, Optional['ModelTrace']]:
        """
        Simulate choices for a full session.
        
        Args:
            params: Model parameters
            initial_state: Starting state (belief distribution)
            stimuli: Array of stimulus values
            categories: Array of true categories (for feedback)
            rng: Random number generator
            no_response: Optional boolean array (True = skip trial)
            not_blockstart: Optional boolean array (True = not start of block/session)
                           Used for update matrix computation. Default: first trial is block start.
            return_history: If True, return full ModelTrace for update matrix analysis
            update_mask: Optional boolean array (True = update normally,
                         False = skip belief update for this trial).
                         Simulates opto inactivation during the update window.
                         If None, all trials update normally.
        
        Returns:
            choices: Simulated choices (0 = A, 1 = B, NaN = no response)
            p_B: Choice probabilities at each trial
            final_state: State after session (for chaining)
            history: ModelTrace if return_history=True, else None
        """
        n_trials = len(stimuli)
        choices = np.full(n_trials, np.nan)
        p_B = np.full(n_trials, np.nan)
        s_hat_arr = np.full(n_trials, np.nan)
        
        if no_response is None:
            no_response = np.zeros(n_trials, dtype=bool)
        
        if not_blockstart is None:
            not_blockstart = np.ones(n_trials, dtype=bool)
            if n_trials > 0:
                not_blockstart[0] = False
        
        # Storage for full history if requested
        if return_history:
            beliefs = np.zeros((n_trials, initial_state.n_points))
        
        state = initial_state.copy()
        belief = state.boundary_belief  # Work with the array directly
        x = state.x
        s_hat_prev = state.s_hat_prev
        
        for t in range(n_trials):
            # Store belief BEFORE this trial (for update matrix computation)
            if return_history:
                beliefs[t] = belief.copy()
            
            if no_response[t]:
                continue
            
            # Perceive
            s_hat = BEModel.perceive_stimulus(
                stimuli[t], params, s_hat_prev, rng
            )
            s_hat_arr[t] = s_hat
            
            # Choice probability (inline to avoid method call overhead)
            j = np.abs(x - s_hat).argmin()
            p_B[t] = np.clip(
                trapezoid(belief[:j + 1], x[:j + 1]), 1e-10, 1 - 1e-10
            )
            
            # Make choice
            choices[t] = rng.binomial(1, p_B[t])
            
            # Update belief in-place (skip if opto inactivation)
            if update_mask is None or update_mask[t]:
                BEModel._update_belief_inplace(
                    s_hat, categories[t], params, belief, x,
                    state.x_min, state.x_max,
                    state.relax_target,
                )
            s_hat_prev = s_hat
        
        # Build final state (snapshot current belief)
        final_state = BEState(
            boundary_belief=belief.copy(),
            s_hat_prev=s_hat_prev,
            x=x,
            x_min=state.x_min,
            x_max=state.x_max,
            relax_target=state.relax_target,
        )
        
        # Build history if requested
        if return_history:
            history = ModelTrace(
                stimuli=stimuli.copy(),
                categories=categories.copy(),
                choices=choices.copy(),
                p_B=p_B.copy(),
                s_hat=s_hat_arr.copy(),
                beliefs=beliefs,
                x=x.copy(),
                no_response=no_response.copy(),
                not_blockstart=not_blockstart.copy()
            )
        else:
            history = None
        
        return choices, p_B, final_state, history
    
    
    @staticmethod
    def make_simulator(params: BEParams, burn_in: int = 1000, seed: int = 42):
        """Return a stateful simulator callable for generate_synthetic_animal."""
        state = BEModel.create_initial_state(params=params, burn_in=burn_in, seed=seed)
        
        def simulator(stimuli, categories, rng, **kwargs):
            nonlocal state
            choices, _, state, _ = BEModel.simulate_session(
                params, state, stimuli, categories, rng, return_history=False,
            )
            return choices
        
        return simulator
    
    
    # =========================================================================
    # BURN-IN SIMULATION
    # =========================================================================
    
    @staticmethod
    def run_burn_in(
        params: BEParams,
        initial_state: BEState,
        n_trials: int,
        seed: int = 42
    ) -> BEState:
        """
        Run burn-in simulation to establish experienced belief state.
        
        Simulates trials with uniform stimulus distribution to mimic
        prior experience before the actual session.
        
        Args:
            params: Model parameters
            initial_state: Starting state (typically uniform)
            n_trials: Number of burn-in trials
            seed: Random seed
        
        Returns:
            State after burn-in (belief has converged toward boundary)
        """
        if n_trials == 0:
            return initial_state.copy()
        
        rng = np.random.default_rng(seed)
        
        # Generate burn-in stimuli (uniform)
        stimuli = rng.uniform(initial_state.x_min, initial_state.x_max, n_trials)
        categories = (stimuli > 0).astype(int)  # Boundary at 0
        
        # Simulate (don't need choices, just state evolution)
        _, _, final_state, _ = BEModel.simulate_session(
            params, initial_state, stimuli, categories, rng,
            return_history=False
        )
        
        return final_state
    
    @staticmethod
    def create_initial_state(
        burn_in: int = 0,
        params: Optional[BEParams] = None,
        x_min: float = None,
        x_max: float = None,
        n_points: int = None,
        seed: int = 42
    ) -> BEState:
        """
        Convenience function to create initial state with optional burn-in.
        
        Args:
            burn_in: Number of burn-in trials (0 = naive/uniform)
            params: Model parameters (required if burn_in > 0, and used to
                    compute x_min/x_max/n_points if not explicitly provided)
            x_min, x_max: Stimulus space bounds. If None, computed from params
                          via BEParams.stimulus_space_bounds().
            n_points: Discretisation resolution. If None, computed from params.
            seed: Random seed for burn-in
        
        Returns:
            Initial state (uniform if burn_in=0, experienced otherwise)
        """
        # Compute grid from params if not explicitly provided
        if (x_min is None or x_max is None or n_points is None):
            if params is None:
                raise ValueError("params required to compute grid bounds when x_min/x_max/n_points not provided")
            _x_min, _x_max, _n_pts = params.stimulus_space_bounds()
            x_min    = x_min    if x_min    is not None else _x_min
            x_max    = x_max    if x_max    is not None else _x_max
            n_points = n_points if n_points is not None else _n_pts

        state = BEState.initial_uniform(x_min, x_max, n_points)
        
        if burn_in > 0:
            if params is None:
                raise ValueError("params required for burn_in > 0")
            state = BEModel.run_burn_in(params, state, burn_in, seed)
        
        return state



# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ModelTrace',
    'BEParams',
    'BEState',
    'BEModel',
]