"""
SBI Fitting Interface for BE Model.

Top-level interface that connects experimental data (FittingData) with
simulation-based inference. Handles:
- Building priors from per-parameter link specifications
- Constructing a simulator that uses real stimuli/categories
- Training neural density estimators
- Extracting per-session parameter trajectories from posterior samples

Usage:
    from behav_utils.data.structures import AnimalData
    from Inference.sbi_fitter import SBIFitter, GPLink, ConstantLink
    
    animal = load_animal(...)
    fitting_data = animal.get_fitting_data(stage='Full_Task_Cont')
    
    fitter = SBIFitter(
        fitting_data=fitting_data,
        param_links={
            'sigma_percep': ConstantLink(bounds=(0.05, 0.5)),
            'A_repulsion': ConstantLink(bounds=(0.0, 0.5)),
            'eta_learning': GPLink(bounds=(0.05, 0.9), lengthscale=5.0),
            'eta_relax': GPLink(bounds=(0.01, 0.4), lengthscale=5.0),
        },
    )
    
    result = fitter.train(n_simulations=50_000)
    trajectories = fitter.extract_trajectories(result, n_samples=5000)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import time
import warnings

# torch is imported lazily where needed (prior building, training)
# to allow simulator/layout usage without torch installed


# =============================================================================
# PARAMETER LINK SPECIFICATIONS
# =============================================================================

@dataclass(frozen=True)
class ConstantLink:
    """
    Parameter is constant across all sessions.
    
    Prior: Uniform(bounds[0], bounds[1])
    Contributes 1 dimension to theta.
    """
    bounds: Tuple[float, float]


@dataclass(frozen=True)
class GPLink:
    """
    Parameter varies smoothly across sessions via Gaussian Process prior.
    
    Prior: GP(mean, RBF kernel) truncated to bounds.
    Contributes n_sessions dimensions to theta.
    
    Args:
        bounds: (low, high) hard bounds
        lengthscale: GP lengthscale in session-index units.
                     ~5 means parameters correlated over ~5 sessions.
        amplitude: GP amplitude (std of function values).
                   Controls how much the parameter varies.
        mean: Mean of GP. If None, uses midpoint of bounds.
    """
    bounds: Tuple[float, float]
    lengthscale: float = 5.0
    amplitude: float = 0.1
    mean: Optional[float] = None


@dataclass(frozen=True)
class RandomWalkLink:
    """
    Parameter drifts across sessions via random walk.
    
    Prior: theta_0 ~ Uniform, theta_{t+1} ~ N(theta_t, sigma_drift^2)
    Contributes n_sessions dimensions to theta.
    
    Args:
        bounds: (low, high) hard bounds
        sigma_drift: Step size std per session
    """
    bounds: Tuple[float, float]
    sigma_drift: float = 0.05


@dataclass(frozen=True)
class IndependentLink:
    """
    Parameter varies independently per session (no temporal structure).
    
    Prior: Uniform(bounds[0], bounds[1]) per session, independently.
    Contributes n_sessions dimensions to theta.
    """
    bounds: Tuple[float, float]


@dataclass(frozen=True)
class HierarchicalLink:
    """
    Parameter drawn from shared group distribution per session.
    
    Prior: theta_s ~ N(mu_group, sigma_group^2) truncated to bounds.
    Sessions are exchangeable (no temporal order).
    Contributes n_sessions dimensions to theta.
    
    Args:
        bounds: (low, high) hard bounds
        group_mean: Mean of group distribution. If None, uses midpoint.
        group_std: Std of group distribution.
    """
    bounds: Tuple[float, float]
    group_mean: Optional[float] = None
    group_std: float = 0.1


# Default link specifications for BE model
DEFAULT_BE_PARAM_LINKS = {
    'sigma_percep': ConstantLink(bounds=(0.05, 0.5)),
    'A_repulsion': ConstantLink(bounds=(0.0, 0.5)),
    'eta_learning': GPLink(bounds=(0.05, 0.9), lengthscale=5.0, amplitude=0.1),
    'eta_relax': GPLink(bounds=(0.01, 0.4), lengthscale=5.0, amplitude=0.1),
}

# Default link specifications for SC model
DEFAULT_SC_PARAM_LINKS = {
    'sigma_percep': ConstantLink(bounds=(0.05, 0.5)),
    'A_repulsion': ConstantLink(bounds=(0.0, 0.5)),
    'gamma': GPLink(bounds=(0.1, 1.0), lengthscale=5.0, amplitude=0.1),
    'sigma_update': ConstantLink(bounds=(0.1, 1.0)),
}

# Backwards compatibility alias
DEFAULT_PARAM_LINKS = DEFAULT_BE_PARAM_LINKS


# =============================================================================
# THETA LAYOUT
# =============================================================================

@dataclass
class ThetaLayout:
    """
    Describes how a flat theta vector maps to per-session parameters.
    
    Constructed from param_links and n_sessions. Handles packing/unpacking
    between flat theta (for SBI) and per-session parameter dicts (for simulator).
    """
    param_names: List[str]           # Canonical order of model params
    n_sessions: int
    links: Dict[str, Any]            # param_name -> link spec
    model_type: str = 'be'           # 'be' or 'sc'
    
    # Computed layout
    slices: Dict[str, slice] = field(init=False)
    total_dim: int = field(init=False)
    varying_params: List[str] = field(init=False)
    constant_params: List[str] = field(init=False)
    
    def __post_init__(self):
        self.varying_params = []
        self.constant_params = []
        self.slices = {}
        
        idx = 0
        for name in self.param_names:
            link = self.links[name]
            if isinstance(link, ConstantLink):
                self.slices[name] = slice(idx, idx + 1)
                self.constant_params.append(name)
                idx += 1
            else:
                # All other links produce n_sessions values
                self.slices[name] = slice(idx, idx + self.n_sessions)
                self.varying_params.append(name)
                idx += self.n_sessions
        
        self.total_dim = idx
    
    # Validation bounds (hard constraints from model)
    # Posterior samples can slightly exceed prior bounds.
    # Populated from model-specific defaults via class method.
    _PARAM_CLAMP_BE = {
        'sigma_percep': (1e-6, None),
        'A_repulsion':  (0.0, None),
        'eta_learning': (1e-6, 1.0),
        'eta_relax':    (0.0, 1.0 - 1e-6),
    }
    _PARAM_CLAMP_SC = {
        'sigma_percep': (1e-6, None),
        'A_repulsion':  (0.0, None),
        'gamma':        (1e-6, 1.0),
        'sigma_update': (1e-6, None),
    }
    
    # Set from model_type at construction (defaults to BE for backwards compat)
    model_type: str = 'be'
    
    @property
    def _PARAM_CLAMP(self) -> Dict[str, Tuple]:
        if self.model_type == 'sc':
            return self._PARAM_CLAMP_SC
        return self._PARAM_CLAMP_BE
    
    def theta_to_session_params(self, theta: np.ndarray) -> List[Dict[str, float]]:
        """
        Convert flat theta to list of per-session parameter dicts.
        
        Clamps values to valid BEParams ranges (posterior samples can
        slightly exceed prior bounds).
        
        Args:
            theta: 1D array of shape (total_dim,)
        
        Returns:
            List of n_sessions dicts, each with all param names as keys
        """
        session_params = []
        for s in range(self.n_sessions):
            params = {}
            for name in self.param_names:
                sl = self.slices[name]
                values = theta[sl]
                if isinstance(self.links[name], ConstantLink):
                    val = float(values[0])
                else:
                    val = float(values[s])
                # Clamp to valid model bounds
                lo, hi = self._PARAM_CLAMP.get(name, (None, None))
                if lo is not None:
                    val = max(val, lo)
                if hi is not None:
                    val = min(val, hi)
                params[name] = val
            session_params.append(params)
        return session_params
    
    def theta_to_trajectories(self, theta: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert flat theta to parameter trajectories.
        
        Args:
            theta: Shape (total_dim,) or (n_samples, total_dim)
        
        Returns:
            Dict mapping param names to arrays:
                constant params: shape (n_samples,) or scalar
                varying params: shape (n_samples, n_sessions) or (n_sessions,)
        """
        single = theta.ndim == 1
        if single:
            theta = theta[np.newaxis, :]
        
        result = {}
        for name in self.param_names:
            sl = self.slices[name]
            values = theta[:, sl]
            if isinstance(self.links[name], ConstantLink):
                result[name] = values[:, 0]
            else:
                result[name] = values  # (n_samples, n_sessions)
        
        if single:
            result = {k: v[0] if v.ndim == 1 else v[0] for k, v in result.items()}
        
        return result
    
    def get_expanded_names(self) -> List[str]:
        """Get flat list of names matching theta dimensions."""
        names = []
        for name in self.param_names:
            if isinstance(self.links[name], ConstantLink):
                names.append(name)
            else:
                for s in range(self.n_sessions):
                    names.append(f"{name}_{s}")
        return names


# =============================================================================
# PRIOR BUILDER
# =============================================================================

def build_prior(layout: ThetaLayout) -> Any:
    """
    Build an SBI-compatible prior from ThetaLayout.
    
    Uses the link specifications to construct appropriate marginal
    distributions for each parameter block, then combines them.
    
    Returns an object with .sample() and .log_prob() methods.
    """
    from Inference.priors import (
        MultiSessionPrior, LinkingConfig, UniformPrior
    )
    
    # If all parameters are constant, use simple uniform prior
    if not layout.varying_params:
        bounds = {name: layout.links[name].bounds for name in layout.param_names}
        return UniformPrior(bounds, param_order=layout.param_names)
    
    # Build MultiSessionPrior
    param_bounds = {name: layout.links[name].bounds for name in layout.param_names}
    
    linking_configs = {}
    for name in layout.varying_params:
        link = layout.links[name]
        
        if isinstance(link, GPLink):
            linking_configs[name] = LinkingConfig(
                link_type='gp',
                params={
                    'lengthscale': link.lengthscale,
                    'amplitude': link.amplitude,
                    'mean': link.mean,
                }
            )
        elif isinstance(link, RandomWalkLink):
            linking_configs[name] = LinkingConfig(
                link_type='random_walk',
                params={'sigma_drift': link.sigma_drift}
            )
        elif isinstance(link, IndependentLink):
            linking_configs[name] = LinkingConfig(link_type='independent')
        elif isinstance(link, HierarchicalLink):
            linking_configs[name] = LinkingConfig(
                link_type='hierarchical',
                params={
                    'group_mean': link.group_mean,
                    'group_std': link.group_std,
                }
            )
        else:
            raise ValueError(f"Unknown link type for {name}: {type(link)}")
    
    return MultiSessionPrior(
        param_bounds=param_bounds,
        n_sessions=layout.n_sessions,
        varying_params=layout.varying_params,
        linking_configs=linking_configs,
        param_order=layout.param_names,
    )


# =============================================================================
# SIMULATOR BUILDER
# =============================================================================

def build_simulator(
    layout: ThetaLayout,
    stimuli_per_session: List[np.ndarray],
    categories_per_session: List[np.ndarray],
    no_response_per_session: List[np.ndarray],
    not_blockstart_per_session: List[np.ndarray],
    summary_stat_names: List[str],
    burn_in: int = 0,
    burn_in_seed: int = 42,
    model_type: str = 'be',
) -> callable:
    """
    Build a simulator function: theta -> summary_stats.
    
    The returned function:
    1. Unpacks theta into per-session parameter dicts
    2. Simulates each session with real stimuli/categories
    3. Chains belief state across sessions
    4. Computes summary statistics
    5. Returns flat 1D array for SBI
    
    Args:
        layout: ThetaLayout describing parameter structure
        stimuli_per_session: List of stimulus arrays (one per session)
        categories_per_session: List of category arrays
        no_response_per_session: List of no_response masks
        not_blockstart_per_session: List of not_blockstart masks
        summary_stat_names: Which summary stats to compute
        burn_in: Burn-in trials before first session
        burn_in_seed: Seed for burn-in
        model_type: 'be' or 'sc'
    
    Returns:
        Callable: theta (np.ndarray) -> summary_stats (np.ndarray)
    """
    from behav_utils.analysis.summary_stats import compute_summary_stats, flatten_stats    
    
    n_sessions = layout.n_sessions
    
    if model_type == 'be':
        from Models.BE_core import BEParams, BEState, BEModel

        def simulate(theta: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
            if seed is None:
                seed = np.random.randint(0, 2**31)
            rng = np.random.default_rng(seed)
            
            session_params = layout.theta_to_session_params(theta)
            
            be_params_0 = BEParams(**session_params[0])
            if burn_in > 0:
                state = BEModel.run_burn_in(
                    be_params_0, BEState.initial_uniform(),
                    burn_in, burn_in_seed,
                )
            else:
                state = BEState.initial_uniform()
            
            all_choices = []
            for s in range(n_sessions):
                be_params = BEParams(**session_params[s])
                choices, p_B, state, _ = BEModel.simulate_session(
                    params=be_params, initial_state=state,
                    stimuli=stimuli_per_session[s],
                    categories=categories_per_session[s],
                    rng=rng,
                    no_response=no_response_per_session[s],
                    not_blockstart=not_blockstart_per_session[s],
                    return_history=False,
                )
                all_choices.append(choices)
            
            all_stats = []
            for s in range(n_sessions):
                stats = compute_summary_stats(
                    choices=all_choices[s],
                    stimuli=stimuli_per_session[s],
                    categories=categories_per_session[s],
                    stat_names=summary_stat_names,
                    return_dict=False,
                )
                all_stats.append(stats)
            return np.concatenate(all_stats)

    elif model_type == 'sc':
        from Models.SC_core import SCParams, SCState, SCModel

        def simulate(theta: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
            if seed is None:
                seed = np.random.randint(0, 2**31)
            rng = np.random.default_rng(seed)
            
            session_params = layout.theta_to_session_params(theta)
            
            sc_params_0 = SCParams(**session_params[0])
            if burn_in > 0:
                state = SCModel.create_initial_state(
                    params=sc_params_0, burn_in=burn_in, seed=burn_in_seed,
                )
            else:
                state = SCState.initial_default()
            
            all_choices = []
            for s in range(n_sessions):
                sc_params = SCParams(**session_params[s])
                choices, p_B, state, _ = SCModel.simulate_session(
                    params=sc_params, initial_state=state,
                    stimuli=stimuli_per_session[s],
                    categories=categories_per_session[s],
                    rng=rng,
                    no_response=no_response_per_session[s],
                    not_blockstart=not_blockstart_per_session[s],
                    return_history=False,
                )
                all_choices.append(choices)
            
            all_stats = []
            for s in range(n_sessions):
                stats = compute_summary_stats(
                    choices=all_choices[s],
                    stimuli=stimuli_per_session[s],
                    categories=categories_per_session[s],
                    stat_names=summary_stat_names,
                    return_dict=False,
                )
                all_stats.append(stats)
            return np.concatenate(all_stats)

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Must be 'be' or 'sc'.")
    
    return simulate


# =============================================================================
# OBSERVED STATS COMPUTATION
# =============================================================================

def compute_observed_stats(
    fitting_data: Any,
    summary_stat_names: List[str],
) -> np.ndarray:
    """
    Compute summary statistics from observed (real) data.
    
    Args:
        fitting_data: FittingData object from AnimalData.get_fitting_data()
        summary_stat_names: Which stats to compute
    
    Returns:
        Flat 1D array matching simulator output format
    """
    from behav_utils.analysis.summary_stats import compute_summary_stats
    
    all_stats = []
    for s in range(fitting_data.n_sessions):
        sa = fitting_data.get_session(s)
        stats = compute_summary_stats(
            choices=sa['choices'],
            stimuli=sa['stimuli'],
            categories=sa['categories'],
            stat_names=summary_stat_names,
            return_dict=False,
        )
        all_stats.append(stats)
    
    return np.concatenate(all_stats)


# =============================================================================
# SBI FITTER
# =============================================================================

# Default summary statistics
DEFAULT_SUMMARY_STATS = [
    'accuracy', 'psychometric', 'recency', 'win_stay', 'stimulus_sensitivity'
]


class SBIFitter:
    """
    Top-level SBI fitting interface for BE and SC models.
    
    Takes experimental data (FittingData) and parameter link specifications,
    builds all necessary components, and provides methods to train and
    extract results.
    
    Args:
        fitting_data: FittingData from AnimalData.get_fitting_data()
        model_type: 'be' or 'sc'
        param_links: Dict mapping param names to link specs.
                     If None, uses model-specific defaults.
        summary_stats: List of summary stat names.
                       If None, uses DEFAULT_SUMMARY_STATS.
        burn_in: Burn-in trials before first session
        burn_in_seed: Seed for burn-in simulation
    
    Example (BE):
        fitter = SBIFitter(
            fitting_data=animal.get_fitting_data(stage='Full_Task_Cont'),
            model_type='be',
            param_links={
                'sigma_percep': ConstantLink(bounds=(0.05, 0.5)),
                'A_repulsion': ConstantLink(bounds=(0.0, 0.5)),
                'eta_learning': GPLink(bounds=(0.05, 0.9), lengthscale=5.0),
                'eta_relax': GPLink(bounds=(0.01, 0.4), lengthscale=5.0),
            },
        )
    
    Example (SC):
        fitter = SBIFitter(
            fitting_data=animal.get_fitting_data(stage='Full_Task_Cont'),
            model_type='sc',
            param_links={
                'sigma_percep': ConstantLink(bounds=(0.05, 0.5)),
                'A_repulsion': ConstantLink(bounds=(0.0, 0.5)),
                'gamma': GPLink(bounds=(0.1, 1.0), lengthscale=5.0),
                'sigma_update': ConstantLink(bounds=(0.1, 1.0)),
            },
        )
    """
    
    # Canonical parameter orders per model
    BE_PARAM_ORDER = ['sigma_percep', 'A_repulsion', 'eta_learning', 'eta_relax']
    SC_PARAM_ORDER = ['sigma_percep', 'A_repulsion', 'gamma', 'sigma_update']
    
    _DEFAULT_LINKS = {
        'be': DEFAULT_BE_PARAM_LINKS,
        'sc': DEFAULT_SC_PARAM_LINKS,
    }
    
    def __init__(
        self,
        fitting_data: Any,
        model_type: str = 'be',
        param_links: Optional[Dict[str, Any]] = None,
        summary_stats: Optional[List[str]] = None,
        burn_in: int = 0,
        burn_in_seed: int = 42,
    ):
        self.fitting_data = fitting_data
        self.model_type = model_type.lower()
        self.burn_in = burn_in
        self.burn_in_seed = burn_in_seed
        
        # Resolve model-specific defaults
        if self.model_type == 'be':
            self._param_order = self.BE_PARAM_ORDER
        elif self.model_type == 'sc':
            self._param_order = self.SC_PARAM_ORDER
        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type!r}. Must be 'be' or 'sc'."
            )
        
        self.param_links = param_links or dict(self._DEFAULT_LINKS[self.model_type])
        self.summary_stats = summary_stats or list(DEFAULT_SUMMARY_STATS)
        
        # Validate param_links covers all required params
        for name in self._param_order:
            if name not in self.param_links:
                raise ValueError(
                    f"Missing link spec for '{name}'. "
                    f"All {self.model_type.upper()} params required: {self._param_order}"
                )
        
        # Build layout
        self.layout = ThetaLayout(
            param_names=self._param_order,
            n_sessions=fitting_data.n_sessions,
            links=self.param_links,
            model_type=self.model_type,
        )
        
        # Extract per-session arrays from FittingData
        self._stimuli = fitting_data.stimuli
        self._categories = fitting_data.categories
        self._choices = fitting_data.choices
        self._no_response = fitting_data.no_response
        self._not_blockstart = fitting_data.not_blockstart
        
        # Prior is built lazily on first access (requires torch)
        self._prior = None
        
        # Build simulator (numpy only, no torch needed)
        self.simulator = build_simulator(
            layout=self.layout,
            stimuli_per_session=self._stimuli,
            categories_per_session=self._categories,
            no_response_per_session=self._no_response,
            not_blockstart_per_session=self._not_blockstart,
            summary_stat_names=self.summary_stats,
            burn_in=self.burn_in,
            burn_in_seed=self.burn_in_seed,
            model_type=self.model_type,
        )
        
        # Compute observed stats
        self.observed_stats = compute_observed_stats(
            fitting_data, self.summary_stats
        )
        
        # ── NaN safety net ─────────────────────────────────────────────
        # If any observed stat is NaN (e.g. psychometric fit failure on a
        # session with chance-level performance), mask those dimensions
        # from both the observation vector and the simulator output.
        #
        # This is a FALLBACK — the root cause should be fixed upstream
        # (e.g. psychometric fitter should return finite values for flat
        # curves). The mask is frozen at construction time, so held-out
        # data conditioned on this fitter will also drop these dims even
        # if they are valid there.
        self._valid_dims = np.isfinite(self.observed_stats)
        self._n_masked = int((~self._valid_dims).sum())
        
        if self._n_masked > 0:
            warnings.warn(
                f"NaN MASKING: {self._n_masked}/{len(self.observed_stats)} "
                f"observed stats are NaN/inf and will be masked from the "
                f"observation vector AND simulator output. This means the "
                f"density estimator cannot see these dimensions. Fix the "
                f"upstream cause (e.g. psychometric fitter) to avoid this."
            )
            # Mask observed stats
            self.observed_stats = self.observed_stats[self._valid_dims]
            
            # Wrap simulator to drop the same dims
            _raw_simulator = self.simulator
            _mask = self._valid_dims  # capture in closure
            
            def _masked_simulator(theta, seed=None):
                stats = _raw_simulator(theta, seed=seed)
                return stats[_mask]
            
            self.simulator = _masked_simulator
    
    # =========================================================================
    # PRIOR (lazy, requires torch)
    # =========================================================================
    
    @property
    def prior(self):
        """Build and cache prior on first access (requires torch)."""
        if self._prior is None:
            self._prior = build_prior(self.layout)
        return self._prior
    
    # =========================================================================
    # INFO
    # =========================================================================
    
    @property
    def n_sessions(self) -> int:
        return self.fitting_data.n_sessions
    
    @property
    def theta_dim(self) -> int:
        return self.layout.total_dim
    
    @property
    def n_summary_stats(self) -> int:
        return len(self.observed_stats)
    
    def describe(self) -> str:
        """Print summary of fitter configuration."""
        lines = [
            f"SBIFitter Configuration",
            f"{'=' * 50}",
            f"Model: {self.model_type.upper()}",
            f"Animal: {self.fitting_data.animal_id}",
            f"Sessions: {self.n_sessions}",
            f"Trials per session: {[len(s) for s in self._stimuli]}",
            f"",
            f"Parameter links:",
        ]
        for name in self._param_order:
            link = self.param_links[name]
            link_type = type(link).__name__
            bounds = link.bounds
            extra = ""
            if isinstance(link, GPLink):
                extra = f", ls={link.lengthscale}, amp={link.amplitude}"
            elif isinstance(link, RandomWalkLink):
                extra = f", drift={link.sigma_drift}"
            lines.append(f"  {name}: {link_type}{bounds}{extra}")
        
        lines.extend([
            f"",
            f"Theta dimensionality: {self.theta_dim}",
            f"  Constant params: {self.layout.constant_params}",
            f"  Varying params: {self.layout.varying_params}",
            f"",
            f"Summary stats: {self.summary_stats}",
            f"Stats vector length: {self.n_summary_stats}",
            f"  Per session (raw): {len(self._valid_dims) // self.n_sessions}",
        ])
        if self._n_masked > 0:
            lines.append(
                f"  ⚠ Masked dims: {self._n_masked} "
                f"(NaN in observed stats — fix upstream!)"
            )
        lines.extend([
            f"",
            f"Burn-in: {self.burn_in} trials",
        ])
        return "\n".join(lines)
    
    # =========================================================================
    # SIMULATION TESTING
    # =========================================================================
    
    def test_simulator(self, n_tests: int = 5, seed: int = 42) -> Dict[str, Any]:
        """
        Run a few test simulations to verify everything works.
        
        Returns dict with timing info and example outputs.
        """
        rng = np.random.default_rng(seed)
        
        times = []
        stats_list = []
        nan_counts = []
        
        for i in range(n_tests):
            # Sample theta from uniform within bounds (no torch needed)
            theta = self._sample_theta_numpy(rng)
            
            t0 = time.time()
            stats = self.simulator(theta, seed=seed + i)
            dt = time.time() - t0
            
            times.append(dt)
            stats_list.append(stats)
            nan_counts.append(np.sum(np.isnan(stats)))
        
        mean_time = np.mean(times)
        estimated_total = mean_time * 50_000
        
        return {
            'mean_time_per_sim': mean_time,
            'estimated_time_50k': estimated_total,
            'estimated_time_50k_str': f"{estimated_total / 3600:.1f} hours",
            'stats_dim': len(stats_list[0]),
            'nan_counts': nan_counts,
            'example_theta': self._sample_theta_numpy(rng),
            'example_stats': stats_list[0],
            'observed_stats': self.observed_stats,
        }
    
    def _sample_theta_numpy(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a single theta from uniform within bounds (no torch)."""
        theta = np.zeros(self.layout.total_dim)
        for name in self._param_order:
            link = self.param_links[name]
            sl = self.layout.slices[name]
            n_vals = sl.stop - sl.start
            theta[sl] = rng.uniform(link.bounds[0], link.bounds[1], n_vals)
        return theta
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    def train(
        self,
        n_simulations: int = 50_000,
        method: str = 'NPE',
        n_rounds: int = 1,
        seed: Optional[int] = 42,
        show_progress: bool = True,
        # Neural network config
        hidden_features: int = 50,
        num_transforms: int = 5,
        # Storage
        store_simulations: bool = True,
        **kwargs,
    ) -> Any:
        """
        Train SBI density estimator.
        
        Args:
            n_simulations: Number of training simulations
            method: 'NPE', 'NLE', or 'NRE'
            n_rounds: Number of sequential rounds (1 = amortised)
            seed: Random seed
            show_progress: Show training progress
            hidden_features: Hidden layer size in neural network
            num_transforms: Number of normalising flow transforms
            store_simulations: Store theta/x for diagnostics
            **kwargs: Additional args passed to sbi
        
        Returns:
            SBIResult with trained posterior
        """
        from Inference.sbi_wrapper import train_sbi
        import torch
        
        observed_tensor = torch.tensor(
            self.observed_stats, dtype=torch.float32
        )
        
        result = train_sbi(
            simulator=self.simulator,
            prior=self.prior,
            observed_stats=observed_tensor,
            method=method,
            n_simulations=n_simulations,
            n_rounds=n_rounds,
            seed=seed,
            show_progress=show_progress,
            hidden_features=hidden_features,
            num_transforms=num_transforms,
            store_simulations=store_simulations,
            param_names=self.layout.get_expanded_names(),
            **kwargs,
        )
        
        return result
    
    # =========================================================================
    # POSTERIOR EXTRACTION
    # =========================================================================
    
    def extract_trajectories(
        self,
        result: Any,
        n_samples: int = 5000,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract per-parameter trajectories from posterior samples.
        
        For each parameter, computes mean, median, and credible intervals
        across sessions.
        
        Args:
            result: SBIResult from train()
            n_samples: Number of posterior samples to draw
        
        Returns:
            Dict[param_name] -> {
                'mean': (n_sessions,) or scalar,
                'median': (n_sessions,) or scalar,
                'ci_low': (n_sessions,) or scalar,  (2.5th percentile)
                'ci_high': (n_sessions,) or scalar,  (97.5th percentile)
                'samples': (n_samples, n_sessions) or (n_samples,),
                'session_indices': array of session indices,
            }
        """
        samples = result.sample(n_samples).numpy()
        trajectories = self.layout.theta_to_trajectories(samples)
        
        session_idx = self.fitting_data.time_axis
        
        summaries = {}
        for name in self._param_order:
            vals = trajectories[name]  # (n_samples,) or (n_samples, n_sessions)
            
            if isinstance(self.param_links[name], ConstantLink):
                summaries[name] = {
                    'mean': float(np.mean(vals)),
                    'median': float(np.median(vals)),
                    'ci_low': float(np.percentile(vals, 2.5)),
                    'ci_high': float(np.percentile(vals, 97.5)),
                    'std': float(np.std(vals)),
                    'samples': vals,
                    'session_indices': session_idx,
                    'link_type': 'constant',
                }
            else:
                summaries[name] = {
                    'mean': np.mean(vals, axis=0),
                    'median': np.median(vals, axis=0),
                    'ci_low': np.percentile(vals, 2.5, axis=0),
                    'ci_high': np.percentile(vals, 97.5, axis=0),
                    'std': np.std(vals, axis=0),
                    'samples': vals,
                    'session_indices': session_idx,
                    'link_type': type(self.param_links[name]).__name__,
                }
        
        return summaries
    
    def extract_session_params(
        self,
        result: Any,
        n_samples: int = 5000,
        point_estimate: str = 'median',
    ) -> List[Dict[str, float]]:
        """
        Extract point-estimate parameters for each session.
        
        Useful for running the model with fitted parameters.
        
        Args:
            result: SBIResult from train()
            n_samples: Posterior samples for computing estimate
            point_estimate: 'mean' or 'median'
        
        Returns:
            List of n_sessions dicts, each with all param values
        """
        trajectories = self.extract_trajectories(result, n_samples)
        
        session_params = []
        for s in range(self.n_sessions):
            params = {}
            for name in self._param_order:
                traj = trajectories[name]
                if traj['link_type'] == 'constant':
                    params[name] = traj[point_estimate]
                else:
                    params[name] = float(traj[point_estimate][s])
            session_params.append(params)
        
        return session_params
    
    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================
    
    def posterior_predictive_check(
        self,
        result: Any,
        n_simulations: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Run posterior predictive checks.
        
        Samples parameters from posterior, simulates data, and compares
        summary statistics to observed.
        
        Args:
            result: SBIResult from train()
            n_simulations: Number of predictive simulations
            seed: Random seed
        
        Returns:
            Dict with:
                'observed': observed stats array
                'simulated': (n_simulations, n_stats) array
                'p_values': Bayesian p-value per stat
                'stat_names': expanded stat names
        """
        from behav_utils.analysis.summary_stats import get_stat_names_expanded
        
        samples = result.sample(n_simulations).numpy()
        
        simulated_stats = np.zeros((n_simulations, len(self.observed_stats)))
        valid_count = 0
        for i in range(n_simulations):
            stats_i = self.simulator(samples[i], seed=seed + i)
            if np.all(np.isfinite(stats_i)):
                simulated_stats[valid_count] = stats_i
                valid_count += 1
        
        if valid_count < n_simulations:
            warnings.warn(
                f"PPC: {n_simulations - valid_count}/{n_simulations} "
                f"simulations produced NaN/inf stats"
            )
        simulated_stats = simulated_stats[:valid_count]
        
        # Bayesian p-values: proportion of simulated >= observed
        p_values = np.mean(simulated_stats >= self.observed_stats[np.newaxis, :], axis=0)
        
        # Expanded stat names (one per dimension)
        per_session_names = get_stat_names_expanded(self.summary_stats)
        all_stat_names = []
        for s in range(self.n_sessions):
            for name in per_session_names:
                all_stat_names.append(f"s{s}_{name}")
        
        # Apply masking if active
        if self._n_masked > 0:
            all_stat_names = [
                n for n, v in zip(all_stat_names, self._valid_dims) if v
            ]
        
        return {
            'observed': self.observed_stats,
            'simulated': simulated_stats,
            'p_values': p_values,
            'stat_names': all_stat_names,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_fit(
    fitting_data: Any,
    model_type: str = 'be',
    n_simulations: int = 30_000,
    varying_params: Optional[List[str]] = None,
    method: str = 'NPE',
    seed: int = 42,
) -> Tuple['SBIFitter', Any, Dict]:
    """
    Quick-start fitting with sensible defaults.
    
    For BE: eta_learning and eta_relax vary (GP-linked) by default.
    For SC: gamma varies (GP-linked) by default.
    sigma_percep and A_repulsion are constant for both.
    
    Args:
        fitting_data: FittingData object
        model_type: 'be' or 'sc'
        n_simulations: Training simulations
        varying_params: Which params to GP-link. If None, uses model defaults.
        method: SBI method
        seed: Random seed
    
    Returns:
        (fitter, result, trajectories)
    """
    model_type = model_type.lower()
    default_links = SBIFitter._DEFAULT_LINKS[model_type]
    param_order = SBIFitter.BE_PARAM_ORDER if model_type == 'be' else SBIFitter.SC_PARAM_ORDER
    
    if varying_params is None:
        if model_type == 'be':
            varying_params = ['eta_learning', 'eta_relax']
        else:
            varying_params = ['gamma']
    
    param_links = {}
    for name in param_order:
        bounds = default_links[name].bounds
        if name in varying_params:
            param_links[name] = GPLink(bounds=bounds, lengthscale=5.0, amplitude=0.1)
        else:
            param_links[name] = ConstantLink(bounds=bounds)
    
    fitter = SBIFitter(
        fitting_data=fitting_data,
        model_type=model_type,
        param_links=param_links,
        burn_in=100,
    )
    
    print(fitter.describe())
    print()
    
    # Test simulator
    test = fitter.test_simulator()
    print(f"Simulator test: {test['mean_time_per_sim']:.3f}s/sim, "
          f"estimated {test['estimated_time_50k_str']} for 50k sims")
    print(f"NaN stats in tests: {test['nan_counts']}")
    print()
    
    # Train
    result = fitter.train(n_simulations=n_simulations, method=method, seed=seed)
    
    # Extract
    trajectories = fitter.extract_trajectories(result)
    
    return fitter, result, trajectories


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Link specifications
    'ConstantLink',
    'GPLink',
    'RandomWalkLink',
    'IndependentLink',
    'HierarchicalLink',
    'DEFAULT_PARAM_LINKS',
    'DEFAULT_BE_PARAM_LINKS',
    'DEFAULT_SC_PARAM_LINKS',
    # Layout
    'ThetaLayout',
    # Fitter
    'SBIFitter',
    # Convenience
    'quick_fit',
]
