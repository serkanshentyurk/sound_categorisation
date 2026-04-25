"""
Stateless Stimulus-Category (SC) Model Core

The SC model maintains two independent probability distributions — one for
each stimulus category — and updates them based on choice-contingent
feedback.  This contrasts with the BE model, which maintains a single
boundary-location distribution.

SC model mechanics:
    Decision:  P(B) = B(s_hat) / (A(s_hat) + B(s_hat))
    Update:    Gaussian-bump mixture directed at the CHOSEN category's
               distribution, with sign depending on feedback correctness.

Components:
    SCParams:  Immutable parameter container
    SCState:   Model state (two category distributions + previous stimulus)
    SCModel:   Stateless operations (simulate, log-likelihood, etc.)

The public interface mirrors BEModel exactly so that the SBI pipeline,
summary statistics, and analysis code can swap between models
transparently.

Usage:
    from models.SC_core import SCParams, SCState, SCModel

    params = SCParams(sigma_percep=0.15, A_repulsion=0.1,
                      gamma=0.95, sigma_update=0.3)
    state = SCState.initial_default()
    choices, p_B, final_state, trace = SCModel.simulate_session(
        params, state, stimuli, categories, rng
    )
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Union
from scipy.integrate import trapezoid
from scipy.stats import norm as sp_norm

from models.perception import perceive_stimulus, stimulus_space_bounds


# =============================================================================
# PARAMETER CONTAINER
# =============================================================================

@dataclass(frozen=True)
class SCParams:
    """
    Immutable container for SC model parameters.

    Parameters:
        sigma_percep:  Perceptual noise standard deviation (shared with BE)
        A_repulsion:   Serial dependence strength (shared with BE)
        gamma:         Retention factor for belief updates, in (0, 1].
                       gamma=1 → no update; gamma=0 → replace with bump.
                       Effective learning rate = (1 − gamma).
        sigma_update:  Standard deviation of the Gaussian bump used for
                       belief updates.  Controls how localised the update
                       is in stimulus space.

    Note on gamma: The old code uses the same gamma for both correct
    (reinforcing) and incorrect (weakening) updates.  A future extension
    might split this into gamma_pos and gamma_neg, but for now we keep
    a single parameter to match the original model and avoid
    identifiability issues.
    """
    sigma_percep: float
    A_repulsion: float
    gamma: float
    sigma_update: float

    def __post_init__(self):
        """Validate and clamp parameters on creation."""
        import warnings

        if self.sigma_percep <= 0:
            clamped = max(self.sigma_percep, 1e-6)
            warnings.warn(
                f"sigma_percep={self.sigma_percep:.6f} clamped to {clamped:.6f}",
                stacklevel=2,
            )
            object.__setattr__(self, 'sigma_percep', clamped)

        if self.A_repulsion < 0:
            clamped = max(self.A_repulsion, 0.0)
            warnings.warn(
                f"A_repulsion={self.A_repulsion:.6f} clamped to {clamped:.6f}",
                stacklevel=2,
            )
            object.__setattr__(self, 'A_repulsion', clamped)

        if self.gamma <= 0 or self.gamma > 1:
            clamped = float(np.clip(self.gamma, 1e-6, 1.0))
            warnings.warn(
                f"gamma={self.gamma:.6f} clamped to {clamped:.6f}",
                stacklevel=2,
            )
            object.__setattr__(self, 'gamma', clamped)

        if self.sigma_update <= 0:
            clamped = max(self.sigma_update, 1e-6)
            warnings.warn(
                f"sigma_update={self.sigma_update:.6f} clamped to {clamped:.6f}",
                stacklevel=2,
            )
            object.__setattr__(self, 'sigma_update', clamped)

    # -----------------------------------------------------------------
    # Serialisation (matching BEParams interface)
    # -----------------------------------------------------------------

    PARAM_NAMES = ['sigma_percep', 'A_repulsion', 'gamma', 'sigma_update']

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'SCParams':
        return cls(**{k: d[k] for k in cls.PARAM_NAMES})

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'SCParams':
        """Order: [sigma_percep, A_repulsion, gamma, sigma_update]"""
        return cls(*arr[:4])

    def to_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in self.PARAM_NAMES}

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in self.PARAM_NAMES])

    @classmethod
    def get_bounds(cls) -> Dict[str, Tuple[float, float]]:
        """Parameter bounds for fitting / prior specification."""
        return {
            'sigma_percep': (0.05, 0.5),
            'A_repulsion':  (0.0, 0.5),
            'gamma':        (0.80, 0.999),
            'sigma_update': (0.05, 1.0),
        }

    @classmethod
    def get_param_names(cls) -> List[str]:
        return list(cls.PARAM_NAMES)

    @classmethod
    def sample_prior(cls, rng: np.random.Generator) -> 'SCParams':
        bounds = cls.get_bounds()
        return cls(**{k: rng.uniform(*v) for k, v in bounds.items()})

    def stimulus_space_bounds(
        self,
        stim_half_range: float = 1.0,
        n_sigma: float = 6.0,
    ) -> Tuple[float, float, int]:
        """Compute stimulus grid bounds (delegates to shared function)."""
        return stimulus_space_bounds(
            self.sigma_percep, self.A_repulsion,
            stim_half_range, n_sigma,
        )


# =============================================================================
# STATE CONTAINER
# =============================================================================

@dataclass
class SCState:
    """
    SC model state — carried across trials and sessions.

    Attributes:
        A_distribution: Category-A belief distribution (integrates to 1)
        B_distribution: Category-B belief distribution (integrates to 1)
        s_hat_prev:     Previous perceived stimulus (for serial dependence)
        x:              Discretisation grid
        x_min, x_max:   Grid bounds
    """
    A_distribution: np.ndarray
    B_distribution: np.ndarray
    s_hat_prev: Optional[float]
    x: np.ndarray
    x_min: float
    x_max: float

    # -----------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------

    @classmethod
    def initial_default(
        cls,
        x_min: float = -1.0,
        x_max: float = 1.0,
        n_points: int = 500,
        A_centre: float = -0.75,
        B_centre: float = 0.75,
        init_scale: float = 0.5,
    ) -> 'SCState':
        """
        Create initial state with Gaussian priors for each category.

        Default centres (−0.75 and +0.75) and scale (0.5) match the
        original SC code.  The distributions are normalised to integrate
        to 1 over the grid.

        Args:
            x_min, x_max: Grid bounds
            n_points: Grid resolution
            A_centre: Centre of initial A distribution
            B_centre: Centre of initial B distribution
            init_scale: Std of initial Gaussians
        """
        x = np.linspace(x_min, x_max, n_points)
        A_dist = sp_norm.pdf(x, loc=A_centre, scale=init_scale)
        B_dist = sp_norm.pdf(x, loc=B_centre, scale=init_scale)
        # Normalise over actual grid (not −∞ to ∞)
        A_dist = A_dist / trapezoid(A_dist, x)
        B_dist = B_dist / trapezoid(B_dist, x)
        return cls(
            A_distribution=A_dist,
            B_distribution=B_dist,
            s_hat_prev=None,
            x=x,
            x_min=x_min,
            x_max=x_max,
        )

    @classmethod
    def initial_uniform(
        cls,
        x_min: float = -1.0,
        x_max: float = 1.0,
        n_points: int = 500,
    ) -> 'SCState':
        """
        Create initial state with flat (uniform) category distributions.

        Suitable for testing whether the model can learn categories from
        scratch without prior knowledge of category centres.
        """
        x = np.linspace(x_min, x_max, n_points)
        uniform = np.ones(n_points) / (x_max - x_min)
        uniform = uniform / trapezoid(uniform, x)
        return cls(
            A_distribution=uniform.copy(),
            B_distribution=uniform.copy(),
            s_hat_prev=None,
            x=x,
            x_min=x_min,
            x_max=x_max,
        )

    @classmethod
    def from_distributions(
        cls,
        A_dist: np.ndarray,
        B_dist: np.ndarray,
        x_min: float = -1.0,
        x_max: float = 1.0,
        s_hat_prev: Optional[float] = None,
    ) -> 'SCState':
        """Create state from existing distributions (normalises)."""
        x = np.linspace(x_min, x_max, len(A_dist))
        A_norm = A_dist / trapezoid(A_dist, x)
        B_norm = B_dist / trapezoid(B_dist, x)
        return cls(
            A_distribution=A_norm,
            B_distribution=B_norm,
            s_hat_prev=s_hat_prev,
            x=x,
            x_min=x_min,
            x_max=x_max,
        )

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def copy(self) -> 'SCState':
        return SCState(
            A_distribution=self.A_distribution.copy(),
            B_distribution=self.B_distribution.copy(),
            s_hat_prev=self.s_hat_prev,
            x=self.x,          # shared — never mutated
            x_min=self.x_min,
            x_max=self.x_max,
        )

    @property
    def n_points(self) -> int:
        return len(self.x)

    @property
    def decision_boundary(self) -> float:
        """
        Estimated decision boundary: stimulus value where P(B) = 0.5.

        Found by locating where B/(A+B) crosses 0.5, i.e. where A = B.
        Returns NaN if distributions don't cross.
        """
        diff = self.B_distribution - self.A_distribution
        crossings = np.where(np.diff(np.sign(diff)))[0]
        if len(crossings) == 0:
            return np.nan
        # Take crossing closest to 0
        best = crossings[np.argmin(np.abs(self.x[crossings]))]
        # Linear interpolation for sub-grid precision
        x0, x1 = self.x[best], self.x[best + 1]
        d0, d1 = diff[best], diff[best + 1]
        if d1 == d0:
            return (x0 + x1) / 2
        return x0 - d0 * (x1 - x0) / (d1 - d0)


# =============================================================================
# STATELESS MODEL OPERATIONS
# =============================================================================

class SCModel:
    """
    Stateless SC model operations.

    All methods are static — they take params and state as arguments
    and return results without side effects.  Interface mirrors BEModel.
    """

    # =================================================================
    # CORE COMPUTATIONS
    # =================================================================

    @staticmethod
    def perceive_stimulus(
        s_t: float, params: SCParams,
        s_hat_prev: Optional[float],
        rng: np.random.Generator,
    ) -> float:
        """Apply perceptual noise and serial dependence."""
        return perceive_stimulus(
            s_t, params.sigma_percep, params.A_repulsion, s_hat_prev, rng,
        )

    @staticmethod
    def get_choice_probability(s_hat: float, state: SCState) -> float:
        """
        Compute P(choose B) given perceived stimulus.

        P(B) = B(s_hat) / (A(s_hat) + B(s_hat))

        Falls back to 0.5 if both densities are ≤0 at s_hat (can happen
        in tails of the grid).
        """
        j = np.abs(state.x - s_hat).argmin()
        a_val = state.A_distribution[j]
        b_val = state.B_distribution[j]
        denom = a_val + b_val
        if denom <= 0:
            return 0.5
        p_B = b_val / denom
        return float(np.clip(p_B, 1e-10, 1 - 1e-10))

    @staticmethod
    def update_belief(
        s_hat: float,
        true_category: int,
        choice: int,
        params: SCParams,
        state: SCState,
    ) -> SCState:
        """
        Update category distributions based on choice and feedback.

        Creates and returns a NEW state — does not mutate input.

        The update rule (from Akrami lab SC model):
            g = N(x; x_closest_to_s_hat, sigma_update)

            Correct choice   → chosen_dist = dist * gamma + g * (1−gamma)
            Incorrect choice → chosen_dist = dist * gamma − g * (1−gamma)
                               (clamped non-negative, then renormalised)

        Only the distribution corresponding to the CHOSEN category is
        updated.  The other distribution is unchanged.

        Args:
            s_hat: Perceived stimulus value
            true_category: True category (0=A, 1=B)
            choice: Model's choice (0=A, 1=B)
            params: Model parameters
            state: Current model state

        Returns:
            New SCState with updated distributions
        """
        j = np.abs(state.x - s_hat).argmin()
        g = sp_norm.pdf(state.x, loc=state.x[j], scale=params.sigma_update)

        A_new = state.A_distribution.copy()
        B_new = state.B_distribution.copy()

        correct = (choice == true_category)

        if choice == 0:
            # Update A distribution
            if correct:
                A_new = A_new * params.gamma + g * (1 - params.gamma)
            else:
                A_new = A_new * params.gamma - g * (1 - params.gamma)
                min_val = A_new.min()
                if min_val < 0:
                    A_new += np.abs(min_val)
            A_new = A_new / trapezoid(A_new, state.x)
        else:
            # Update B distribution
            if correct:
                B_new = B_new * params.gamma + g * (1 - params.gamma)
            else:
                B_new = B_new * params.gamma - g * (1 - params.gamma)
                min_val = B_new.min()
                if min_val < 0:
                    B_new += np.abs(min_val)
            B_new = B_new / trapezoid(B_new, state.x)

        return SCState(
            A_distribution=A_new,
            B_distribution=B_new,
            s_hat_prev=s_hat,
            x=state.x,
            x_min=state.x_min,
            x_max=state.x_max,
        )

    @staticmethod
    def _update_beliefs_inplace(
        s_hat: float,
        true_category: int,
        choice: int,
        params: SCParams,
        A_dist: np.ndarray,
        B_dist: np.ndarray,
        x: np.ndarray,
    ) -> None:
        """
        Update A and B distributions IN-PLACE.  For tight simulation
        loops only — the public update_belief() returns a new SCState.
        """
        j = np.abs(x - s_hat).argmin()
        g = sp_norm.pdf(x, loc=x[j], scale=params.sigma_update)

        correct = (choice == true_category)

        if choice == 0:
            if correct:
                A_dist[:] = A_dist * params.gamma + g * (1 - params.gamma)
            else:
                A_dist[:] = A_dist * params.gamma - g * (1 - params.gamma)
                min_val = A_dist.min()
                if min_val < 0:
                    A_dist[:] += np.abs(min_val)
            A_dist[:] /= trapezoid(A_dist, x)
        else:
            if correct:
                B_dist[:] = B_dist * params.gamma + g * (1 - params.gamma)
            else:
                B_dist[:] = B_dist * params.gamma - g * (1 - params.gamma)
                min_val = B_dist.min()
                if min_val < 0:
                    B_dist[:] += np.abs(min_val)
            B_dist[:] /= trapezoid(B_dist, x)

    # =================================================================
    # SESSION SIMULATION
    # =================================================================

    @staticmethod
    def simulate_session(
        params: SCParams,
        initial_state: SCState,
        stimuli: np.ndarray,
        categories: np.ndarray,
        rng: np.random.Generator,
        no_response: Optional[np.ndarray] = None,
        not_blockstart: Optional[np.ndarray] = None,
        return_history: bool = False,
        update_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, SCState, Optional['ModelTrace']]:
        """
        Simulate choices for a full session.

        Interface is identical to BEModel.simulate_session.

        Args:
            params: SC model parameters
            initial_state: Starting state (category distributions)
            stimuli: Array of stimulus values
            categories: Array of true categories (for feedback)
            rng: Random number generator
            no_response: Optional boolean array (True = skip trial)
            not_blockstart: Optional boolean array (True = not block start)
            return_history: If True, return full ModelTrace
            update_mask: Optional boolean array (True = update normally,
                         False = skip belief update for this trial).
                         Simulates opto inactivation during the update window.
                         If None, all trials update normally.

        Returns:
            choices: Simulated choices (0=A, 1=B, NaN=no response)
            p_B: Choice probabilities at each trial
            final_state: State after session (for chaining)
            history: ModelTrace if return_history, else None
        """
        # Lazy import to avoid circular dependency
        from models.BE_core import ModelTrace

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

        state = initial_state.copy()
        A_dist = state.A_distribution    # work with arrays directly
        B_dist = state.B_distribution
        x = state.x
        s_hat_prev = state.s_hat_prev

        if return_history:
            beliefs_A = np.zeros((n_trials, state.n_points))
            beliefs_B = np.zeros((n_trials, state.n_points))

        for t in range(n_trials):
            if return_history:
                beliefs_A[t] = A_dist.copy()
                beliefs_B[t] = B_dist.copy()

            if no_response[t]:
                continue

            # Perceive
            s_hat = perceive_stimulus(
                stimuli[t], params.sigma_percep, params.A_repulsion,
                s_hat_prev, rng,
            )
            s_hat_arr[t] = s_hat

            # Choice probability
            j = np.abs(x - s_hat).argmin()
            a_val = A_dist[j]
            b_val = B_dist[j]
            denom = a_val + b_val
            p_B_t = float(np.clip(b_val / denom, 1e-10, 1 - 1e-10)) if denom > 0 else 0.5
            p_B[t] = p_B_t

            # Make choice
            choice = rng.binomial(1, p_B_t)
            choices[t] = choice

            # Update beliefs in-place (skip if opto inactivation)
            if update_mask is None or update_mask[t]:
                SCModel._update_beliefs_inplace(
                    s_hat, categories[t], choice, params,
                    A_dist, B_dist, x,
                )
            s_hat_prev = s_hat

        # Build final state
        final_state = SCState(
            A_distribution=A_dist.copy(),
            B_distribution=B_dist.copy(),
            s_hat_prev=s_hat_prev,
            x=x,
            x_min=state.x_min,
            x_max=state.x_max,
        )

        # Build history if requested
        if return_history:
            history = ModelTrace(
                stimuli=stimuli.copy(),
                categories=categories.copy(),
                choices=choices.copy(),
                p_B=p_B.copy(),
                s_hat=s_hat_arr.copy(),
                beliefs=np.array([]),      # Not used for SC
                x=x.copy(),
                no_response=no_response.copy(),
                not_blockstart=not_blockstart.copy(),
                beliefs_A=beliefs_A,
                beliefs_B=beliefs_B,
            )
        else:
            history = None

        return choices, p_B, final_state, history
    
    @staticmethod
    def make_simulator(params: SCParams, burn_in: int = 1000, seed: int = 42):
        """Return a stateful simulator callable for generate_synthetic_animal."""
        state = SCModel.create_initial_state(params=params, burn_in=burn_in, seed=seed)
        
        def simulator(stimuli, categories, rng, **kwargs):
            nonlocal state
            choices, _, state, _ = SCModel.simulate_session(
                params, state, stimuli, categories, rng, return_history=False,
            )
            return choices
        
        return simulator
    # =================================================================
    # LIKELIHOOD COMPUTATION
    # =================================================================

    @staticmethod
    def compute_log_likelihood(
        params: SCParams,
        initial_state: SCState,
        stimuli: np.ndarray,
        categories: np.ndarray,
        observed_choices: np.ndarray,
        rng: np.random.Generator,
        eval_mask: Optional[np.ndarray] = None,
        no_response: Optional[np.ndarray] = None,
        not_blockstart: Optional[np.ndarray] = None,
        return_history: bool = False,
        update_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, SCState, Optional['ModelTrace']]:
        """
        Compute log-likelihood of observed choices under SC model.

        Processes ALL trials (to maintain correct belief state) but only
        accumulates likelihood for trials where eval_mask=True.

        IMPORTANT: The SC update rule is choice-contingent — the
        distribution that gets updated depends on the animal's actual
        choice.  During log-likelihood computation, we therefore use
        the OBSERVED choice for both the likelihood term and the belief
        update.

        Interface is identical to BEModel.compute_log_likelihood.

        Returns:
            total_ll: Total log-likelihood
            trial_lls: Per-trial log-likelihoods (NaN for skipped/masked)
            final_state: State after processing
            history: ModelTrace if return_history, else None
        """
        from models.BE_core import ModelTrace

        n_trials = len(stimuli)
        trial_lls = np.full(n_trials, np.nan)
        p_B_arr = np.full(n_trials, np.nan)
        s_hat_arr = np.full(n_trials, np.nan)

        if no_response is None:
            no_response = np.isnan(observed_choices)
        if eval_mask is None:
            eval_mask = np.ones(n_trials, dtype=bool)
        if not_blockstart is None:
            not_blockstart = np.ones(n_trials, dtype=bool)
            if n_trials > 0:
                not_blockstart[0] = False

        state = initial_state.copy()
        A_dist = state.A_distribution
        B_dist = state.B_distribution
        x = state.x
        s_hat_prev = state.s_hat_prev

        if return_history:
            beliefs_A = np.zeros((n_trials, state.n_points))
            beliefs_B = np.zeros((n_trials, state.n_points))

        for t in range(n_trials):
            if return_history:
                beliefs_A[t] = A_dist.copy()
                beliefs_B[t] = B_dist.copy()

            if no_response[t]:
                continue

            # Perceive
            s_hat = perceive_stimulus(
                stimuli[t], params.sigma_percep, params.A_repulsion,
                s_hat_prev, rng,
            )
            s_hat_arr[t] = s_hat

            # Choice probability
            j = np.abs(x - s_hat).argmin()
            a_val = A_dist[j]
            b_val = B_dist[j]
            denom = a_val + b_val
            p_B_t = float(np.clip(b_val / denom, 1e-10, 1 - 1e-10)) if denom > 0 else 0.5
            p_B_arr[t] = p_B_t

            # Log-likelihood for this trial
            if eval_mask[t]:
                if observed_choices[t] == 1:
                    trial_lls[t] = np.log(p_B_t)
                else:
                    trial_lls[t] = np.log(1 - p_B_t)

            # Update belief using OBSERVED choice (skip if opto inactivation)
            if update_mask is None or update_mask[t]:
                obs_choice = int(observed_choices[t])
                SCModel._update_beliefs_inplace(
                    s_hat, categories[t], obs_choice, params,
                    A_dist, B_dist, x,
                )
            s_hat_prev = s_hat

        total_ll = np.nansum(trial_lls)

        final_state = SCState(
            A_distribution=A_dist.copy(),
            B_distribution=B_dist.copy(),
            s_hat_prev=s_hat_prev,
            x=x,
            x_min=state.x_min,
            x_max=state.x_max,
        )

        if return_history:
            history = ModelTrace(
                stimuli=stimuli.copy(),
                categories=categories.copy(),
                choices=observed_choices.copy(),
                p_B=p_B_arr.copy(),
                s_hat=s_hat_arr.copy(),
                beliefs=np.array([]),
                x=x.copy(),
                no_response=no_response.copy(),
                not_blockstart=not_blockstart.copy(),
                beliefs_A=beliefs_A,
                beliefs_B=beliefs_B,
            )
        else:
            history = None

        return total_ll, trial_lls, final_state, history

    # =================================================================
    # MULTI-SESSION OPERATIONS
    # =================================================================

    @staticmethod
    def simulate_multisession(
        params_per_session: List[SCParams],
        initial_state: SCState,
        session_data: List[Tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
        return_history: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[SCState], Optional[List]]:
        """Simulate multiple sessions with state chaining."""
        all_choices = []
        all_p_B = []
        states = []
        histories = [] if return_history else None

        state = initial_state.copy()
        for session_idx, (stimuli, categories) in enumerate(session_data):
            params = params_per_session[session_idx]
            choices, p_B, state, history = SCModel.simulate_session(
                params, state, stimuli, categories, rng,
                return_history=return_history,
            )
            all_choices.append(choices)
            all_p_B.append(p_B)
            states.append(state.copy())
            if return_history:
                histories.append(history)

        return all_choices, all_p_B, states, histories

    @staticmethod
    def compute_log_likelihood_multisession(
        params_per_session: List[SCParams],
        initial_state: SCState,
        session_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        rng: np.random.Generator,
        return_history: bool = False,
    ) -> Tuple[float, List[np.ndarray], List[SCState], Optional[List]]:
        """Compute log-likelihood across multiple sessions."""
        total_ll = 0.0
        session_lls = []
        states = []
        histories = [] if return_history else None

        state = initial_state.copy()
        for session_idx, (stimuli, categories, choices) in enumerate(session_data):
            params = params_per_session[session_idx]
            ll, trial_lls, state, history = SCModel.compute_log_likelihood(
                params, state, stimuli, categories, choices, rng,
                return_history=return_history,
            )
            total_ll += ll
            session_lls.append(trial_lls)
            states.append(state.copy())
            if return_history:
                histories.append(history)

        return total_ll, session_lls, states, histories

    # =================================================================
    # BURN-IN AND INITIAL STATE
    # =================================================================

    @staticmethod
    def run_burn_in(
        params: SCParams,
        initial_state: SCState,
        n_trials: int,
        seed: int = 42,
    ) -> SCState:
        """
        Run burn-in simulation to establish experienced category beliefs.

        Simulates trials with uniform stimulus distribution.
        """
        if n_trials == 0:
            return initial_state.copy()

        rng = np.random.default_rng(seed)
        stimuli = rng.uniform(initial_state.x_min, initial_state.x_max, n_trials)
        categories = (stimuli > 0).astype(int)

        _, _, final_state, _ = SCModel.simulate_session(
            params, initial_state, stimuli, categories, rng,
            return_history=False,
        )
        return final_state

    @staticmethod
    def create_initial_state(
        burn_in: int = 0,
        params: Optional[SCParams] = None,
        x_min: float = None,
        x_max: float = None,
        n_points: int = None,
        seed: int = 42,
        uniform_init: bool = False,
    ) -> SCState:
        """
        Convenience function to create initial state with optional burn-in.

        Args:
            burn_in: Number of burn-in trials (0 = no burn-in)
            params: Model parameters (required for grid bounds and burn-in)
            x_min, x_max: Grid bounds (computed from params if None)
            n_points: Grid resolution (computed from params if None)
            seed: Random seed for burn-in
            uniform_init: If True, start from flat distributions instead
                          of the default Gaussians at ±0.75.
        """
        if x_min is None or x_max is None or n_points is None:
            if params is None:
                raise ValueError(
                    "params required to compute grid bounds "
                    "when x_min/x_max/n_points not provided"
                )
            _x_min, _x_max, _n_pts = params.stimulus_space_bounds()
            x_min = x_min if x_min is not None else _x_min
            x_max = x_max if x_max is not None else _x_max
            n_points = n_points if n_points is not None else _n_pts

        if uniform_init:
            state = SCState.initial_uniform(x_min, x_max, n_points)
        else:
            state = SCState.initial_default(x_min, x_max, n_points)

        if burn_in > 0:
            if params is None:
                raise ValueError("params required for burn_in > 0")
            state = SCModel.run_burn_in(params, state, burn_in, seed)

        return state


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SCParams',
    'SCState',
    'SCModel',
]
