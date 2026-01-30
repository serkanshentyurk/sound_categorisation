"""
Stateless Boundary Estimation Model Core

This module provides the core computational components for the BE model
in a stateless, functional style that supports:
- MCMC inference (single-session)
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
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Union
from scipy.integrate import trapezoid


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
        """Validate parameters on creation."""
        if self.sigma_percep <= 0:
            raise ValueError(f"sigma_percep must be positive, got {self.sigma_percep}")
        if self.A_repulsion < 0:
            raise ValueError(f"A_repulsion must be non-negative, got {self.A_repulsion}")
        if not 0 < self.eta_learning <= 1:
            raise ValueError(f"eta_learning must be in (0, 1], got {self.eta_learning}")
        if not 0 <= self.eta_relax < 1:
            raise ValueError(f"eta_relax must be in [0, 1), got {self.eta_relax}")
    
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
    
    @classmethod
    def initial_uniform(cls, x_min: float = -1.0, x_max: float = 1.0,
                        n_points: int = 500) -> 'BEState':
        """Create initial state with uniform belief distribution."""
        x = np.linspace(x_min, x_max, n_points)
        belief = np.ones(n_points) / (x_max - x_min)
        belief = belief / trapezoid(belief, x)  # Normalise
        return cls(
            boundary_belief=belief,
            s_hat_prev=None,
            x=x,
            x_min=x_min,
            x_max=x_max
        )
    
    @classmethod
    def from_belief(cls, belief: np.ndarray, x_min: float = -1.0,
                    x_max: float = 1.0, s_hat_prev: Optional[float] = None) -> 'BEState':
        """Create state from existing belief distribution."""
        x = np.linspace(x_min, x_max, len(belief))
        # Normalise
        belief_norm = belief / trapezoid(belief, x)
        return cls(
            boundary_belief=belief_norm,
            s_hat_prev=s_hat_prev,
            x=x,
            x_min=x_min,
            x_max=x_max
        )
    
    def copy(self) -> 'BEState':
        """Create independent copy of state."""
        return BEState(
            boundary_belief=self.boundary_belief.copy(),
            s_hat_prev=self.s_hat_prev,
            x=self.x,  # Can share - never mutated
            x_min=self.x_min,
            x_max=self.x_max
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
        """
        Apply perceptual noise and serial dependence.
        
        Args:
            s_t: True stimulus value
            params: Model parameters
            s_hat_prev: Previous perceived stimulus (or None for first trial)
            rng: Random number generator
        
        Returns:
            s_hat: Perceived stimulus value
        """
        # Perceptual noise
        noise = rng.normal(0, params.sigma_percep)
        s_tilde = s_t + noise
        
        # Repulsion from previous trial
        if s_hat_prev is not None:
            diff = s_tilde - s_hat_prev
            repulsion = params.A_repulsion * diff * np.exp(-np.abs(diff))
            s_hat = s_tilde + repulsion
        else:
            s_hat = s_tilde
        
        return s_hat
    
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
        
        # Relaxation toward uniform (0.5)
        delta_relax = y_prime - 0.5
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
            x_max=state.x_max
        )
    
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
        no_response: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, BEState]:
        """
        Simulate choices for a full session.
        
        Args:
            params: Model parameters
            initial_state: Starting state (belief distribution)
            stimuli: Array of stimulus values
            categories: Array of true categories (for feedback)
            rng: Random number generator
            no_response: Optional boolean array (True = skip trial)
        
        Returns:
            choices: Simulated choices (0 = A, 1 = B, NaN = no response)
            p_B: Choice probabilities at each trial
            final_state: State after session (for chaining)
        """
        n_trials = len(stimuli)
        choices = np.full(n_trials, np.nan)
        p_B = np.full(n_trials, np.nan)
        
        if no_response is None:
            no_response = np.zeros(n_trials, dtype=bool)
        
        state = initial_state.copy()
        
        for t in range(n_trials):
            if no_response[t]:
                continue
            
            # Perceive
            s_hat = BEModel.perceive_stimulus(
                stimuli[t], params, state.s_hat_prev, rng
            )
            
            # Choice probability
            p_B[t] = BEModel.get_choice_probability(s_hat, state)
            
            # Make choice
            choices[t] = rng.binomial(1, p_B[t])
            
            # Update belief based on TRUE CATEGORY
            state = BEModel.update_belief(s_hat, categories[t], params, state)
        
        return choices, p_B, state
    
    @staticmethod
    def simulate_session_with_history(
        params: BEParams,
        initial_state: BEState,
        stimuli: np.ndarray,
        categories: np.ndarray,
        rng: np.random.Generator,
        no_response: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, BEState]:
        """
        Simulate session with full belief history for plotting.
        
        Returns:
            choices: Simulated choices
            p_B: Choice probabilities
            belief_mu: Belief mean at each trial
            belief_std: Belief std at each trial
            final_state: State after session
        """
        n_trials = len(stimuli)
        choices = np.full(n_trials, np.nan)
        p_B = np.full(n_trials, np.nan)
        belief_mu = np.full(n_trials, np.nan)
        belief_std = np.full(n_trials, np.nan)
        
        if no_response is None:
            no_response = np.zeros(n_trials, dtype=bool)
        
        state = initial_state.copy()
        
        for t in range(n_trials):
            # Store belief stats before trial
            belief_mu[t], belief_std[t] = state.get_belief_stats()
            
            if no_response[t]:
                continue
            
            s_hat = BEModel.perceive_stimulus(
                stimuli[t], params, state.s_hat_prev, rng
            )
            p_B[t] = BEModel.get_choice_probability(s_hat, state)
            choices[t] = rng.binomial(1, p_B[t])
            state = BEModel.update_belief(s_hat, categories[t], params, state)
        
        return choices, p_B, belief_mu, belief_std, state
    
    # =========================================================================
    # LIKELIHOOD COMPUTATION
    # =========================================================================
    
    @staticmethod
    def compute_log_likelihood(
        params: BEParams,
        initial_state: BEState,
        stimuli: np.ndarray,
        categories: np.ndarray,
        observed_choices: np.ndarray,
        rng: np.random.Generator,
        eval_mask: Optional[np.ndarray] = None,
        no_response: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray, BEState]:
        """
        Compute log-likelihood of observed choices.
        
        Processes ALL trials (to maintain correct belief state) but only
        accumulates likelihood for trials where eval_mask=True.
        
        Args:
            params: Model parameters
            initial_state: Starting state
            stimuli: Stimulus values
            categories: True categories (for belief updates)
            observed_choices: Observed choices to evaluate
            rng: Random number generator (for perceptual noise)
            eval_mask: Boolean array - True = include in LL (default: all)
            no_response: Boolean array - True = skip trial
        
        Returns:
            total_ll: Total log-likelihood
            trial_lls: Per-trial log-likelihoods (NaN for skipped/masked)
            final_state: State after processing
        """
        n_trials = len(stimuli)
        trial_lls = np.full(n_trials, np.nan)
        
        if no_response is None:
            no_response = np.isnan(observed_choices)
        if eval_mask is None:
            eval_mask = np.ones(n_trials, dtype=bool)
        
        state = initial_state.copy()
        
        for t in range(n_trials):
            if no_response[t]:
                continue
            
            # Perceive
            s_hat = BEModel.perceive_stimulus(
                stimuli[t], params, state.s_hat_prev, rng
            )
            
            # Choice probability
            p_B = BEModel.get_choice_probability(s_hat, state)
            
            # Accumulate LL if in eval set
            if eval_mask[t]:
                if observed_choices[t] == 1:
                    trial_lls[t] = np.log(p_B)
                else:
                    trial_lls[t] = np.log(1 - p_B)
            
            # ALWAYS update belief
            state = BEModel.update_belief(s_hat, categories[t], params, state)
        
        total_ll = np.nansum(trial_lls)
        return total_ll, trial_lls, state
    
    @staticmethod
    def compute_log_likelihood_mc(
        params: BEParams,
        initial_state: BEState,
        stimuli: np.ndarray,
        categories: np.ndarray,
        observed_choices: np.ndarray,
        n_mc_samples: int = 10,
        seed: int = 42,
        eval_mask: Optional[np.ndarray] = None,
        no_response: Optional[np.ndarray] = None
    ) -> Tuple[float, float, BEState]:
        """
        Compute log-likelihood with Monte Carlo marginalisation over perceptual noise.
        
        For MCMC inference where we want to marginalise out the stochastic
        perceptual noise rather than condition on a single realisation.
        
        Args:
            params: Model parameters
            initial_state: Starting state
            stimuli: Stimulus values
            categories: True categories
            observed_choices: Observed choices
            n_mc_samples: Number of MC samples for marginalisation
            seed: Base random seed
            eval_mask: Boolean mask for LL evaluation
            no_response: Boolean mask for no-response trials
        
        Returns:
            mean_ll: Mean log-likelihood across MC samples
            std_ll: Standard deviation of log-likelihood
            final_state: State after processing (from last MC sample)
        """
        lls = []
        final_state = None
        
        for mc_idx in range(n_mc_samples):
            rng = np.random.default_rng(seed + mc_idx)
            ll, _, final_state = BEModel.compute_log_likelihood(
                params, initial_state, stimuli, categories,
                observed_choices, rng, eval_mask, no_response
            )
            lls.append(ll)
        
        lls = np.array(lls)
        return np.mean(lls), np.std(lls), final_state
    
    # =========================================================================
    # MULTI-SESSION OPERATIONS
    # =========================================================================
    
    @staticmethod
    def simulate_multisession(
        params_per_session: List[BEParams],
        initial_state: BEState,
        session_data: List[Tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[BEState]]:
        """
        Simulate multiple sessions with state chaining.
        
        Args:
            params_per_session: Parameters for each session
            initial_state: Starting state (before first session)
            session_data: List of (stimuli, categories) for each session
            rng: Random number generator
        
        Returns:
            all_choices: List of choice arrays
            all_p_B: List of probability arrays
            states: List of states after each session
        """
        all_choices = []
        all_p_B = []
        states = []
        
        state = initial_state.copy()
        
        for session_idx, (stimuli, categories) in enumerate(session_data):
            params = params_per_session[session_idx]
            choices, p_B, state = BEModel.simulate_session(
                params, state, stimuli, categories, rng
            )
            all_choices.append(choices)
            all_p_B.append(p_B)
            states.append(state.copy())
        
        return all_choices, all_p_B, states
    
    @staticmethod
    def compute_log_likelihood_multisession(
        params_per_session: List[BEParams],
        initial_state: BEState,
        session_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        rng: np.random.Generator
    ) -> Tuple[float, List[np.ndarray], List[BEState]]:
        """
        Compute log-likelihood across multiple sessions.
        
        Args:
            params_per_session: Parameters for each session
            initial_state: Starting state
            session_data: List of (stimuli, categories, choices) per session
            rng: Random number generator
        
        Returns:
            total_ll: Sum of log-likelihoods across all sessions
            session_lls: List of per-trial LL arrays
            states: List of states after each session
        """
        total_ll = 0.0
        session_lls = []
        states = []
        
        state = initial_state.copy()
        
        for session_idx, (stimuli, categories, choices) in enumerate(session_data):
            params = params_per_session[session_idx]
            ll, trial_lls, state = BEModel.compute_log_likelihood(
                params, state, stimuli, categories, choices, rng
            )
            total_ll += ll
            session_lls.append(trial_lls)
            states.append(state.copy())
        
        return total_ll, session_lls, states
    
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
        _, _, final_state = BEModel.simulate_session(
            params, initial_state, stimuli, categories, rng
        )
        
        return final_state
    
    @staticmethod
    def create_initial_state(
        burn_in: int = 0,
        params: Optional[BEParams] = None,
        x_min: float = -1.0,
        x_max: float = 1.0,
        n_points: int = 500,
        seed: int = 42
    ) -> BEState:
        """
        Convenience function to create initial state with optional burn-in.
        
        Args:
            burn_in: Number of burn-in trials (0 = naive/uniform)
            params: Model parameters (required if burn_in > 0)
            x_min, x_max: Stimulus space bounds
            n_points: Discretisation resolution
            seed: Random seed for burn-in
        
        Returns:
            Initial state (uniform if burn_in=0, experienced otherwise)
        """
        state = BEState.initial_uniform(x_min, x_max, n_points)
        
        if burn_in > 0:
            if params is None:
                raise ValueError("params required for burn_in > 0")
            state = BEModel.run_burn_in(params, state, burn_in, seed)
        
        return state


# =============================================================================
# CONVENIENCE FUNCTIONS FOR SBI
# =============================================================================

def simulate_for_sbi(
    param_array: np.ndarray,
    stimuli: np.ndarray,
    categories: np.ndarray,
    seed: int,
    burn_in: int = 0,
    return_choices: bool = False
) -> np.ndarray:
    """
    Simulator function formatted for SBI packages.
    
    Args:
        param_array: [sigma_percep, A_repulsion, eta_learning, eta_relax]
        stimuli: Stimulus array
        categories: Category array
        seed: Random seed
        burn_in: Burn-in trials
        return_choices: If True, return raw choices; else return summary stats
    
    Returns:
        Summary statistics or raw choices
    """
    params = BEParams.from_array(param_array)
    initial_state = BEModel.create_initial_state(
        burn_in=burn_in, params=params, seed=seed
    )
    rng = np.random.default_rng(seed + 1000)
    
    choices, p_B, _ = BEModel.simulate_session(
        params, initial_state, stimuli, categories, rng
    )
    
    if return_choices:
        return choices
    
    # Default summary statistics
    return compute_summary_stats(choices, stimuli)


def compute_summary_stats(choices: np.ndarray, stimuli: np.ndarray) -> np.ndarray:
    """
    Compute summary statistics for SBI.
    
    Returns vector of statistics that capture behavioural patterns.
    """
    valid = ~np.isnan(choices)
    choices_valid = choices[valid]
    stimuli_valid = stimuli[valid]
    
    # Basic stats
    mean_choice = np.mean(choices_valid)
    accuracy = np.mean(choices_valid == (stimuli_valid > 0).astype(int))
    
    # Binned choice proportions (coarse psychometric)
    n_bins = 5
    bin_edges = np.linspace(-1, 1, n_bins + 1)
    bin_props = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (stimuli_valid >= bin_edges[i]) & (stimuli_valid < bin_edges[i+1])
        if mask.sum() > 0:
            bin_props[i] = np.mean(choices_valid[mask])
    
    # Serial dependence (choice autocorrelation)
    if len(choices_valid) > 1:
        autocorr = np.corrcoef(choices_valid[:-1], choices_valid[1:])[0, 1]
        if np.isnan(autocorr):
            autocorr = 0.0
    else:
        autocorr = 0.0
    
    # Combine
    stats = np.concatenate([
        [mean_choice, accuracy, autocorr],
        bin_props
    ])
    
    return stats


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BEParams',
    'BEState',
    'BEModel',
    'simulate_for_sbi',
    'compute_summary_stats',
]
