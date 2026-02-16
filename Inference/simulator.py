"""
SBI Simulator for BE and SC(#TODO) models.

Wraps models for use with simulation-based inference.
Supports single-session and multi-session with state chaining.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from enum import Enum
from scipy.integrate import trapezoid

from Analysis.summary_stats import compute_summary_stats, DEFAULT_STATS


# =============================================================================
# MODEL TYPE ENUM
# =============================================================================

class ModelType(Enum):
    """Supported model types."""
    BE = "be"


# =============================================================================
# PARAMETER CONFIGURATION
# =============================================================================

@dataclass
class ParamConfig:
    """Configuration for a single parameter."""
    name: str
    bounds: Tuple[float, float]
    default: Optional[float] = None
    
    def sample_uniform(self, rng: np.random.Generator) -> float:
        """Sample from uniform prior within bounds."""
        return rng.uniform(self.bounds[0], self.bounds[1])
    
    def clip(self, value: float) -> float:
        """Clip value to bounds."""
        return float(np.clip(value, self.bounds[0], self.bounds[1]))

from Models.BE_core import BEParams

_be_bounds = BEParams.get_bounds()

BE_PARAM_CONFIGS = {
    'sigma_percep': ParamConfig('sigma_percep', bounds=_be_bounds['sigma_percep'], default=0.15),
    'A_repulsion': ParamConfig('A_repulsion', bounds=_be_bounds['A_repulsion'], default=0.1),
    'eta_learning': ParamConfig('eta_learning', bounds=_be_bounds['eta_learning'], default=0.35),
    'eta_relax': ParamConfig('eta_relax', bounds=_be_bounds['eta_relax'], default=0.12),
}


def get_default_param_configs(model_type: ModelType) -> Dict[str, ParamConfig]:
    """Get default parameter configs for model type."""
    if model_type == ModelType.BE:
        return {k: ParamConfig(v.name, v.bounds, v.default) 
                for k, v in BE_PARAM_CONFIGS.items()}
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# STATE TRANSITION FUNCTIONS
# =============================================================================

def state_transition_identity(state: Any, **kwargs) -> Any:
    """Carry state forward unchanged between sessions."""
    if hasattr(state, 'copy'):
        return state.copy()
    return state


def state_transition_decay(state: Any, decay_rate: float = 0.1, **kwargs) -> Any:
    """
    Decay belief toward uniform between sessions.
    
    Only works with states that have a 'boundary_belief' attribute.
    """
    if not hasattr(state, 'boundary_belief'):
        return state
    
    uniform = np.ones_like(state.boundary_belief) / len(state.boundary_belief)
    new_belief = (1 - decay_rate) * state.boundary_belief + decay_rate * uniform
    new_belief = new_belief / trapezoid(new_belief, state.x)
    
    # Create new state with updated belief
    new_state = state.copy()
    new_state.boundary_belief = new_belief
    return new_state


def state_transition_reset(state: Any, **kwargs) -> Any:
    """Reset to uniform belief between sessions."""
    if not hasattr(state, 'boundary_belief'):
        return state
    
    new_state = state.copy()
    uniform = np.ones_like(state.boundary_belief)
    new_state.boundary_belief = uniform / trapezoid(uniform, state.x)
    return new_state


# Registry of state transition functions
STATE_TRANSITIONS: Dict[str, Callable] = {
    'identity': state_transition_identity,
    'decay': state_transition_decay,
    'reset': state_transition_reset,
}


def register_state_transition(name: str, func: Callable) -> None:
    """Register a custom state transition function."""
    STATE_TRANSITIONS[name] = func


# =============================================================================
# SIMULATOR CONFIGURATION
# =============================================================================

@dataclass
class SimulatorConfig:
    """Configuration for the SBI simulator."""
    
    # Model type
    model_type: ModelType = ModelType.BE
    
    # Parameter configurations (bounds and defaults)
    param_configs: Dict[str, ParamConfig] = field(default_factory=dict)
    
    # Which parameters to infer (if empty, infer all in param_configs)
    params_to_infer: List[str] = field(default_factory=list)
    
    # Fixed parameter values (not inferred)
    fixed_params: Dict[str, float] = field(default_factory=dict)
    
    # Which parameters vary across sessions (for multi-session)
    varying_params: List[str] = field(default_factory=list)
    
    # Summary statistics to compute
    stat_names: List[str] = field(default_factory=lambda: DEFAULT_STATS.copy())
    
    # State transition between sessions
    state_transition: str = 'identity'
    state_transition_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Burn-in settings
    burn_in: int = 0
    burn_in_seed: Optional[int] = None
    
    def __post_init__(self):
        # Set default param configs if not provided
        if not self.param_configs:
            self.param_configs = get_default_param_configs(self.model_type)
        
        # If params_to_infer not specified, infer all non-fixed params
        if not self.params_to_infer:
            self.params_to_infer = [
                name for name in self.param_configs.keys() 
                if name not in self.fixed_params
            ]
    
    def get_free_param_names(self) -> List[str]:
        """Get names of parameters to be inferred."""
        return [name for name in self.params_to_infer 
                if name not in self.fixed_params]
    
    def get_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get bounds for free parameters."""
        return {name: self.param_configs[name].bounds 
                for name in self.get_free_param_names()}


# =============================================================================
# MAIN SIMULATOR CLASS
# =============================================================================

class Simulator:
    """
    SBI-compatible simulator for BE and SC (TODO) models.
    
    Handles:
    - Single and multi-session simulation
    - Parameter array <-> dict conversion  
    - State chaining between sessions
    - Summary statistic computation
    - Both BE and SC models
    
    Usage (single-session BE):
        config = SimulatorConfig(model_type=ModelType.BE)
        sim = Simulator(config, stimuli, categories)
        summary_stats = sim(theta_array)
        
    Usage (single-session SC):
        #TODO
        
    Usage (multi-session with varying eta):
        config = SimulatorConfig(
            model_type=ModelType.BE,
            varying_params=['eta_learning']
        )
        sim = Simulator(config, stimuli_2d, categories_2d)
        summary_stats = sim(theta_array)
    """
    
    def __init__(
        self,
        config: SimulatorConfig,
        stimuli: np.ndarray,
        categories: np.ndarray,
        seed: Optional[int] = None
    ):
        """
        Initialise simulator.
        
        Args:
            config: Simulator configuration
            stimuli: Shape (n_trials,) or (n_trials, n_sessions)
            categories: Same shape as stimuli
            seed: Base random seed
        """
        self.config = config
        self.seed = seed
        
        # Ensure 2D arrays
        stimuli = np.atleast_1d(stimuli)
        categories = np.atleast_1d(categories)
        
        if stimuli.ndim == 1:
            stimuli = stimuli.reshape(-1, 1)
            categories = categories.reshape(-1, 1)
        
        self.stimuli = stimuli
        self.categories = categories
        self.n_trials, self.n_sessions = self.stimuli.shape
        
        # Compute parameter structure
        self.free_param_names = config.get_free_param_names()
        self.n_free_params = self._compute_n_free_params()
        
        # Get state transition function
        self.state_transition_fn = STATE_TRANSITIONS.get(
            config.state_transition, 
            state_transition_identity
        )
        
        # Try to import models (defer to allow flexibility)
        self._be_model_class = None
        self._import_models()
    
    def _import_models(self):
        """Import model classes."""
        # Try different import paths
        try:
            from Models.BE_model import BoundaryEstimationModel
            self._be_model_class = BoundaryEstimationModel
        except ImportError:
            pass
        

    def _compute_n_free_params(self) -> int:
        """Compute total number of free parameters in theta array."""
        n = 0
        for name in self.free_param_names:
            if name in self.config.varying_params:
                n += self.n_sessions  # One value per session
            else:
                n += 1  # Single value
        return n
    
    def _theta_to_params(self, theta: np.ndarray) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Convert theta array to parameter dicts.
        
        Returns:
            constant_params: Parameters constant across sessions
            varying_params: Parameters that vary (name -> array of values)
        """
        constant_params = dict(self.config.fixed_params)
        varying_params = {}
        
        idx = 0
        for name in self.free_param_names:
            config = self.config.param_configs[name]
            
            if name in self.config.varying_params:
                # Extract session-specific values
                values = theta[idx:idx + self.n_sessions]
                varying_params[name] = np.array([config.clip(v) for v in values])
                idx += self.n_sessions
            else:
                # Constant parameter
                constant_params[name] = config.clip(theta[idx])
                idx += 1
        
        return constant_params, varying_params
    
    def _get_session_params(
        self, 
        constant_params: Dict[str, float],
        varying_params: Dict[str, np.ndarray],
        session_idx: int
    ) -> Dict[str, float]:
        """Get full parameter dict for a specific session."""
        params = constant_params.copy()
        for name, values in varying_params.items():
            params[name] = values[session_idx]
        return params
    
    def _simulate_be_session(
        self,
        params: Dict[str, float],
        stimuli: np.ndarray,
        categories: np.ndarray,
        rng: np.random.Generator,
        initial_belief: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate one session with BE model.
        
        Uses the functional BEModel API from BE_core.py.
        
        Returns:
            choices: Array of choices
            final_belief: Belief distribution at end of session
        """
        # Import BE_core components
        try:
            from Models.BE_core import BEParams, BEState, BEModel
        except ImportError:
            raise ImportError("Models.BE_core not available")
        
        # Create parameter object
        be_params = BEParams(
            sigma_percep=params.get('sigma_percep', 0.15),
            A_repulsion=params.get('A_repulsion', 0.1),
            eta_learning=params.get('eta_learning', 0.35),
            eta_relax=params.get('eta_relax', 0.12),
        )
        
        # Initialise state
        if initial_belief is not None:
            state = BEState.from_belief(initial_belief)
        else:
            # Run burn-in to get initial state
            state = BEModel.create_initial_state(
                params=be_params,
                burn_in=self.config.burn_in,
                seed=self.config.burn_in_seed or rng.integers(0, 2**31)
            )
        
        # Simulate session using functional API
        choices, p_B, final_state, _ = BEModel.simulate_session(
            params=be_params,
            initial_state=state,
            stimuli=stimuli,
            categories=categories,
            rng=rng
        )
        
        return choices.astype(int), final_state.boundary_belief.copy()
    
    
    def simulate(
        self,
        theta: np.ndarray,
        seed: Optional[int] = None,
        return_choices: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate behaviour and compute summary statistics.
        
        Args:
            theta: Parameter array of shape (n_free_params,)
            seed: Random seed
            return_choices: If True, also return raw choices
        
        Returns:
            summary_stats: 1D array of summary statistics
            choices: (optional) Array of shape (n_trials, n_sessions)
        """
        # Set up RNG
        if seed is None:
            seed = self.seed if self.seed is not None else np.random.randint(0, 2**31)
        rng = np.random.default_rng(seed)
        
        # Parse parameters
        constant_params, varying_params = self._theta_to_params(theta)
        
        # Storage
        all_choices = np.zeros((self.n_trials, self.n_sessions), dtype=int)
        current_belief = None
        
        # Simulate each session
        for s in range(self.n_sessions):
            # Get session parameters
            session_params = self._get_session_params(constant_params, varying_params, s)
            
            # Simulate based on model type
            if self.config.model_type == ModelType.BE:
                choices, final_belief = self._simulate_be_session(
                    session_params,
                    self.stimuli[:, s],
                    self.categories[:, s],
                    rng,
                    initial_belief=current_belief
                )
            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")
            
            all_choices[:, s] = choices
            
            # Apply state transition for next session
            if s < self.n_sessions - 1:
                # Wrap belief in simple namespace for transition function
                class BeliefState:
                    def __init__(self, belief):
                        self.boundary_belief = belief
                    def copy(self):
                        return BeliefState(self.boundary_belief.copy())
                
                state = BeliefState(final_belief)
                new_state = self.state_transition_fn(state, **self.config.state_transition_kwargs)
                current_belief = new_state.boundary_belief
        
        # Compute summary statistics
        summary_stats = compute_summary_stats(
            all_choices, self.stimuli, self.categories,
            stat_names=self.config.stat_names,
            return_dict=False
        )
        
        if return_choices:
            return summary_stats, all_choices
        return summary_stats
    
    def __call__(self, theta: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Make simulator callable for SBI."""
        return self.simulate(theta, seed=seed)
    
    def sample_prior(self, n_samples: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample parameters from prior (uniform within bounds).
        
        Args:
            n_samples: Number of samples
            seed: Random seed
        
        Returns:
            Array of shape (n_samples, n_free_params) or (n_free_params,) if n_samples=1
        """
        rng = np.random.default_rng(seed)
        samples = np.zeros((n_samples, self.n_free_params))
        
        for i in range(n_samples):
            theta = []
            for name in self.free_param_names:
                config = self.config.param_configs[name]
                if name in self.config.varying_params:
                    theta.extend([config.sample_uniform(rng) for _ in range(self.n_sessions)])
                else:
                    theta.append(config.sample_uniform(rng))
            samples[i] = theta
        
        return samples if n_samples > 1 else samples[0]
    
    def get_param_names(self) -> List[str]:
        """Get parameter names corresponding to theta array positions."""
        names = []
        for name in self.free_param_names:
            if name in self.config.varying_params:
                names.extend([f"{name}_{s}" for s in range(self.n_sessions)])
            else:
                names.append(name)
        return names
    
    def get_bounds_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get lower and upper bounds as arrays."""
        lower, upper = [], []
        for name in self.free_param_names:
            config = self.config.param_configs[name]
            if name in self.config.varying_params:
                lower.extend([config.bounds[0]] * self.n_sessions)
                upper.extend([config.bounds[1]] * self.n_sessions)
            else:
                lower.append(config.bounds[0])
                upper.append(config.bounds[1])
        return np.array(lower), np.array(upper)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_be_simulator(
    stimuli: np.ndarray,
    categories: np.ndarray,
    fixed_params: Optional[Dict[str, float]] = None,
    varying_params: Optional[List[str]] = None,
    stat_names: Optional[List[str]] = None,
    burn_in: int = 0,
    state_transition: str = 'identity',
    seed: Optional[int] = None
) -> Simulator:
    """
    Create a BE model simulator.
    
    Args:
        stimuli: Shape (n_trials,) or (n_trials, n_sessions)
        categories: Same shape as stimuli
        fixed_params: Parameters to fix (not infer)
        varying_params: Parameters that vary across sessions
        stat_names: Summary statistics to compute
        burn_in: Burn-in trials for initial belief
        state_transition: How to transition state between sessions
        seed: Random seed
    
    Returns:
        Configured Simulator
    """
    config = SimulatorConfig(
        model_type=ModelType.BE,
        fixed_params=fixed_params or {},
        varying_params=varying_params or [],
        stat_names=stat_names or DEFAULT_STATS.copy(),
        burn_in=burn_in,
        state_transition=state_transition,
    )
    return Simulator(config, stimuli, categories, seed=seed)




# =============================================================================
# SBI INTEGRATION HELPERS
# =============================================================================

def get_sbi_prior(simulator: Simulator):
    """
    Create SBI-compatible prior from simulator.
    
    Returns:
        sbi.utils.BoxUniform prior
    """
    try:
        from sbi.utils import BoxUniform
        import torch
    except ImportError:
        raise ImportError("sbi package required. Install with: pip install sbi")
    
    lower, upper = simulator.get_bounds_arrays()
    
    return BoxUniform(
        low=torch.tensor(lower, dtype=torch.float32),
        high=torch.tensor(upper, dtype=torch.float32)
    )


def wrap_for_sbi(simulator: Simulator):
    """
    Wrap simulator for use with sbi package.
    
    Returns a function that:
    - Accepts torch tensors
    - Returns torch tensors
    - Handles batched inputs
    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch required. Install with: pip install torch")
    
    def sbi_simulator(theta):
        # Convert to numpy
        if hasattr(theta, 'numpy'):
            theta_np = theta.numpy()
        else:
            theta_np = np.asarray(theta)
        
        # Handle batched input
        if theta_np.ndim == 1:
            result = simulator(theta_np)
            return torch.tensor(result, dtype=torch.float32)
        else:
            results = []
            for i in range(len(theta_np)):
                # Different seed for each sample
                seed = (simulator.seed or 0) + i
                results.append(simulator(theta_np[i], seed=seed))
            return torch.tensor(np.stack(results), dtype=torch.float32)
    
    return sbi_simulator
