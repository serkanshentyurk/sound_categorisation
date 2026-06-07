from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import trapezoid

from typing import Optional, Dict, Tuple, List, Union

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
        from behav_utils.data.ops.filtering import get_arrays
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
# EXPORTS
# =============================================================================

__all__ = [
    'ModelTrace',
]