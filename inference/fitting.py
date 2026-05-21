"""
SBI Fitting and Training

Unified interface for simulation-based inference. Combines training,
posterior sampling, and the high-level SBIFitter API.

Public API:
    SBIResult           — Container for trained posteriors
    train_sbi           — Core SBI training loop (NPE/NLE/NRE)
    sample_posterior     — Posterior sampling with multiple methods
    SBIFitter           — High-level: FittingData → trained posterior
    build_prior          — Prior construction from ThetaLayout
    build_simulator      — Simulator construction from ThetaLayout + data
    compute_observed_stats — Observed stats from FittingData
    quick_fit            — One-call fitting with defaults
    quick_posterior      — One-call training + sampling
    compare_methods      — Compare NPE/NLE/NRE on same problem

Usage (recommended — high-level):
    from inference.fitting import SBIFitter
    from inference.types import ConstantSpec, GPSpec
    from behav_utils import select_sessions, filter_trials, fitting_data_from_sessions

    sessions = select_sessions(animal, preset='expert_uniform')
    clean = filter_trials(sessions)
    fd = fitting_data_from_sessions(clean, animal.animal_id)

    fitter = SBIFitter(
        fitting_data=fd,
        param_links={
            'sigma_percep': ConstantSpec(bounds=(0.05, 0.5)),
            'eta_learning': GPSpec(bounds=(0.05, 0.9), lengthscale=5.0),
            ...
        },
    )
    result = fitter.train(n_simulations=50_000)
    trajectories = fitter.extract_trajectories(result)

Usage (low-level):
    from inference.fitting import train_sbi, sample_posterior
    result = train_sbi(simulator, prior, observed_stats)
    samples = sample_posterior(result, observed_stats)
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field

# Lazy torch import — only needed for training, not simulator/layout
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from inference.types import (
    ConstantSpec, GPSpec, RandomWalkSpec, IndependentSpec, HierarchicalSpec,
    ThetaLayout, PARAM_CLAMP, LinkSpec,
    # Backwards-compat aliases
    ConstantLink, GPLink, RandomWalkLink, IndependentLink, HierarchicalLink,
)


# =============================================================================
# RESULT CONTAINER 
# =============================================================================

@dataclass
class SBIResult:
    """
    Container for SBI training results.

    Attributes:
        posterior: Trained posterior object (can sample and evaluate)
        inference: The sbi inference object
        density_estimator: The trained neural network
        method: SBI method used ('NPE', 'NLE', 'NRE')
        n_simulations: Total simulations used
        n_rounds: Number of sequential rounds
        training_time: Time taken to train (seconds)
        theta_train: Training parameters (if stored)
        x_train: Training summary stats (if stored)
        prior: Prior used
        observed_stats: Observed data summary stats (if sequential)
        param_names: Parameter names for labelling
    """
    posterior: Any
    inference: Any
    density_estimator: Any
    method: str
    n_simulations: int
    n_rounds: int
    training_time: float
    theta_train: Optional[Any] = None  # torch.Tensor if stored
    x_train: Optional[Any] = None
    prior: Any = None
    observed_stats: Optional[Any] = None
    param_names: Optional[List[str]] = None

    def sample(self, n_samples: int, x: Optional[Any] = None) -> Any:
        """Sample from posterior. Returns tensor (n_samples, n_params)."""
        if x is None:
            x = self.observed_stats
        if x is None:
            raise ValueError("Must provide observed_stats (x)")
        return self.posterior.sample((n_samples,), x=x)

    def log_prob(self, theta: Any, x: Optional[Any] = None) -> Any:
        """Evaluate posterior log probability."""
        if x is None:
            x = self.observed_stats
        if x is None:
            raise ValueError("Must provide observed_stats (x)")
        return self.posterior.log_prob(theta, x=x)

    def map_estimate(self, x: Optional[Any] = None, n_samples: int = 10000) -> Any:
        """Approximate MAP by finding highest-density sample."""
        samples = self.sample(n_samples, x=x)
        log_probs = self.log_prob(samples, x=x)
        return samples[log_probs.argmax()]

    def posterior_mean(self, x: Optional[Any] = None, n_samples: int = 10000) -> Any:
        """Posterior mean estimate."""
        return self.sample(n_samples, x=x).mean(dim=0)

    def credible_interval(self, x: Optional[Any] = None,
                          level: float = 0.95, n_samples: int = 10000) -> Tuple:
        """Get (lower, upper) credible interval tensors."""
        import torch
        samples = self.sample(n_samples, x=x)
        alpha = (1 - level) / 2
        return (torch.quantile(samples, alpha, dim=0),
                torch.quantile(samples, 1 - alpha, dim=0))

    def summary(self, x: Optional[Any] = None, n_samples: int = 10000) -> Dict[str, Any]:
        """Posterior summary: mean, std, median, 95% CI per parameter."""
        import torch
        samples = self.sample(n_samples, x=x)
        lower, upper = self.credible_interval(x=x, n_samples=n_samples)
        names = self.param_names or [f'param_{i}' for i in range(samples.shape[1])]
        return {
            name: {
                'mean': float(samples[:, i].mean()),
                'std': float(samples[:, i].std()),
                'median': float(samples[:, i].median()),
                'ci_lower': float(lower[i]),
                'ci_upper': float(upper[i]),
            }
            for i, name in enumerate(names)
        }


# =============================================================================
# SIMULATOR WRAPPING 
# =============================================================================

def _wrap_simulator_for_sbi(simulator: Callable, seed_offset: int = 0) -> Callable:
    """Wrap a numpy simulator for sbi (handles torch tensors)."""
    import torch

    def wrapped(theta):
        theta_np = theta.numpy() if hasattr(theta, 'numpy') else np.asarray(theta)
        if theta_np.ndim == 1:
            result = simulator(theta_np)
            return torch.tensor(result, dtype=torch.float32)
        else:
            results = []
            for i in range(len(theta_np)):
                if hasattr(simulator, 'simulate'):
                    result = simulator.simulate(theta_np[i], seed=seed_offset + i)
                else:
                    result = simulator(theta_np[i])
                results.append(result)
            return torch.tensor(np.stack(results), dtype=torch.float32)

    return wrapped


# =============================================================================
# CORE SBI TRAINING 
# =============================================================================

def train_sbi(
    simulator: Callable,
    prior: Any,
    observed_stats: Optional[Union[np.ndarray, Any]] = None,
    method: str = 'NPE',
    n_simulations: int = 50000,
    n_rounds: int = 1,
    training_batch_size: int = 256,
    learning_rate: float = 5e-4,
    hidden_features: int = 50,
    num_transforms: int = 5,
    stop_after_epochs: int = 20,
    validation_fraction: float = 0.1,
    density_estimator: Optional[str] = None,
    device: str = 'cpu',
    seed: Optional[int] = None,
    store_simulations: bool = False,
    show_progress: bool = True,
    param_names: Optional[List[str]] = None,
    **kwargs,
) -> SBIResult:
    """
    Train SBI model and return posterior.

    Args:
        simulator: Callable: parameter array → summary stats array
        prior: Prior distribution (UniformPrior, MultiSessionPrior, or sbi prior)
        observed_stats: Observed summary statistics (required for n_rounds > 1)
        method: 'NPE', 'NLE', or 'NRE'
        n_simulations: Total number of simulations
        n_rounds: Sequential inference rounds (1 = amortised)
        training_batch_size: Batch size
        learning_rate: Learning rate
        hidden_features: Hidden layer size
        num_transforms: Normalising flow transforms
        stop_after_epochs: Early stopping patience
        validation_fraction: Validation fraction
        density_estimator: 'maf', 'nsf', 'mdn', 'resnet', 'mlp' (None = auto)
        device: 'cpu' or 'cuda'
        seed: Random seed
        store_simulations: Store training theta/x in result
        show_progress: Show progress bars
        param_names: Parameter names for labelling
        **kwargs: Passed to density estimator

    Returns:
        SBIResult with trained posterior
    """
    import torch
    from sbi.inference import SNPE, SNLE, SNRE
    from sbi.utils import BoxUniform
    try:
        from sbi.utils.user_input_checks import process_prior
    except ImportError:
        from sbi.utils import process_prior

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    start_time = time.time()

    # Convert observed_stats
    if observed_stats is not None:
        if isinstance(observed_stats, np.ndarray):
            observed_stats = torch.tensor(observed_stats, dtype=torch.float32)
        observed_stats = observed_stats.to(device)

    # Setup prior
    if hasattr(prior, 'low') and hasattr(prior, 'high'):
        sbi_prior = BoxUniform(low=prior.low.to(device), high=prior.high.to(device))
        if param_names is None and hasattr(prior, 'param_names'):
            param_names = prior.param_names
    elif hasattr(prior, 'sample') and hasattr(prior, 'log_prob'):
        result = process_prior(prior)
        sbi_prior = result[0] if isinstance(result, tuple) else result
        if param_names is None and hasattr(prior, 'param_names'):
            param_names = prior.param_names
    else:
        sbi_prior = prior

    sim_wrapped = _wrap_simulator_for_sbi(simulator, seed_offset=seed or 0)

    method = method.upper()
    InferenceClass = {'NPE': SNPE, 'NLE': SNLE, 'NRE': SNRE}.get(method)
    if InferenceClass is None:
        raise ValueError(f"Unknown method: {method}. Use 'NPE', 'NLE', or 'NRE'")

    default_est = {'NPE': 'maf', 'NLE': 'maf', 'NRE': 'resnet'}[method]
    if density_estimator is None:
        density_estimator = default_est

    if method in ['NPE', 'NLE'] and density_estimator in ['maf', 'nsf']:
        est_kwargs = {'hidden_features': hidden_features,
                      'num_transforms': num_transforms, **kwargs}
    else:
        est_kwargs = {'hidden_features': hidden_features, **kwargs}

    inference = InferenceClass(
        prior=sbi_prior, density_estimator=density_estimator,
        device=device, show_progress_bars=show_progress,
    )

    all_theta, all_x = [], []
    sims_per_round = n_simulations // n_rounds
    proposal = sbi_prior

    for round_idx in range(n_rounds):
        if show_progress:
            print(f"Round {round_idx + 1}/{n_rounds}: Simulating {sims_per_round}...")

        theta = proposal.sample((sims_per_round,))
        x = sim_wrapped(theta)

        valid_mask = torch.isfinite(x).all(dim=-1)
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum().item()
            warnings.warn(f"Removed {n_invalid} simulations with NaN/inf")
            theta, x = theta[valid_mask], x[valid_mask]

        all_theta.append(theta)
        all_x.append(x)
        inference.append_simulations(theta, x, proposal=proposal)

        if show_progress:
            print(f"Round {round_idx + 1}/{n_rounds}: Training...")

        density_est = inference.train(
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            stop_after_epochs=stop_after_epochs,
            validation_fraction=validation_fraction,
            show_train_summary=show_progress,
        )

        if round_idx < n_rounds - 1:
            if observed_stats is None:
                raise ValueError("observed_stats required for multi-round inference")
            posterior = inference.build_posterior(density_est)
            proposal = posterior.set_default_x(observed_stats)

    posterior = inference.build_posterior(density_est)
    training_time = time.time() - start_time

    return SBIResult(
        posterior=posterior, inference=inference,
        density_estimator=density_est, method=method,
        n_simulations=n_simulations, n_rounds=n_rounds,
        training_time=training_time,
        theta_train=torch.cat(all_theta) if store_simulations else None,
        x_train=torch.cat(all_x) if store_simulations else None,
        prior=prior, observed_stats=observed_stats,
        param_names=param_names,
    )


# =============================================================================
# POSTERIOR SAMPLING 
# =============================================================================

def sample_posterior(
    posterior: Any,
    observed_stats: Union[np.ndarray, Any],
    n_samples: int = 10000,
    method: str = 'direct',
    thin: int = 1,
    warmup: int = 200,
    num_chains: int = 1,
    show_progress: bool = True,
) -> Any:
    """
    Sample from trained posterior.

    Args:
        posterior: Trained posterior (SBIResult or sbi posterior)
        observed_stats: Observed summary statistics
        n_samples: Number of samples
        method: 'direct', 'rejection', 'mcmc', 'vi'

    Returns:
        Samples tensor (n_samples, n_params)
    """
    import torch

    if isinstance(posterior, SBIResult):
        posterior = posterior.posterior
    if isinstance(observed_stats, np.ndarray):
        observed_stats = torch.tensor(observed_stats, dtype=torch.float32)

    sample_kwargs = {'show_progress_bars': show_progress}
    if method == 'direct':
        return posterior.sample((n_samples,), x=observed_stats, **sample_kwargs)
    elif method == 'rejection':
        return posterior.sample((n_samples,), x=observed_stats,
                                sample_with='rejection', **sample_kwargs)
    elif method == 'mcmc':
        return posterior.sample((n_samples,), x=observed_stats,
                                sample_with='mcmc', mcmc_method='slice_np',
                                thin=thin, warmup_steps=warmup,
                                num_chains=num_chains, **sample_kwargs)
    elif method == 'vi':
        return posterior.sample((n_samples,), x=observed_stats,
                                sample_with='vi', **sample_kwargs)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def quick_posterior(
    simulator: Callable, prior: Any,
    observed_stats: Union[np.ndarray, Any],
    n_simulations: int = 20000, method: str = 'NPE',
    seed: Optional[int] = None,
) -> Tuple[Any, SBIResult]:
    """Deprecated: use SBIFitter instead."""
    warnings.warn(
        "quick_posterior is deprecated. Use SBIFitter for all fitting.",
        DeprecationWarning, stacklevel=2,
    )


def compare_methods() -> Dict[str, SBIResult]:
    """Deprecated: use SBIFitter instead. """
    warnings.warn(
        "compare_methods is deprecated. Use SBIFitter for all fitting.",
        DeprecationWarning, stacklevel=2,
    )


def train_multisession_sbi() -> SBIResult:
    """Deprecated: use SBIFitter instead."""
    warnings.warn(
        "train_multisession_sbi is deprecated. Use SBIFitter for all fitting.",
        DeprecationWarning, stacklevel=2,
    )


# =============================================================================
# PRIOR BUILDER 
# =============================================================================

def build_prior(layout: ThetaLayout) -> Any:
    """
    Build an SBI-compatible prior from ThetaLayout.

    Uses link specifications to construct appropriate marginal
    distributions, then combines them.

    Returns an object with .sample() and .log_prob() methods.
    """
    from inference.priors import MultiSessionPrior, LinkingConfig, UniformPrior

    if not layout.varying_params:
        bounds = {name: layout.links[name].bounds for name in layout.param_names}
        return UniformPrior(bounds, param_order=layout.param_names)

    param_bounds = {name: layout.links[name].bounds for name in layout.param_names}

    linking_configs = {}
    for name in layout.varying_params:
        link = layout.links[name]
        if isinstance(link, GPSpec):
            linking_configs[name] = LinkingConfig(
                link_type='gp',
                params={'lengthscale': link.lengthscale,
                        'amplitude': link.amplitude, 'mean': link.mean},
            )
        elif isinstance(link, RandomWalkSpec):
            linking_configs[name] = LinkingConfig(
                link_type='random_walk',
                params={'sigma_drift': link.sigma_drift},
            )
        elif isinstance(link, IndependentSpec):
            linking_configs[name] = LinkingConfig(link_type='independent')
        elif isinstance(link, HierarchicalSpec):
            linking_configs[name] = LinkingConfig(
                link_type='hierarchical',
                params={'group_mean': link.group_mean, 'group_std': link.group_std},
            )
        else:
            raise ValueError(f"Unknown link type for {name}: {type(link)}")

    return MultiSessionPrior(
        param_bounds=param_bounds, n_sessions=layout.n_sessions,
        varying_params=layout.varying_params,
        linking_configs=linking_configs, param_order=layout.param_names,
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
) -> Callable:
    """
    Build simulator function: theta → summary_stats.

    The returned function unpacks theta into per-session parameter dicts,
    simulates each session with real stimuli/categories, chains belief
    state, computes summary statistics, and returns a flat 1D array.
    """
    from behav_utils.analysis.summary_stats import compute_summary_stats

    n_sessions = layout.n_sessions

    if model_type == 'be':
        from models.BE_core import BEParams, BEState, BEModel

        def simulate(theta: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
            if seed is None:
                seed = np.random.randint(0, 2**31)
            rng = np.random.default_rng(seed)
            session_params = layout.theta_to_session_params(theta)

            be_params_0 = BEParams(**session_params[0])
            state = (BEModel.run_burn_in(be_params_0, BEState.initial_uniform(),
                                          burn_in, burn_in_seed)
                     if burn_in > 0 else BEState.initial_uniform())

            all_choices = []
            for s in range(n_sessions):
                be_params = BEParams(**session_params[s])
                choices, _, state, _ = BEModel.simulate_session(
                    params=be_params, initial_state=state,
                    stimuli=stimuli_per_session[s],
                    categories=categories_per_session[s], rng=rng,
                    no_response=no_response_per_session[s],
                    not_blockstart=not_blockstart_per_session[s],
                    return_history=False,
                )
                all_choices.append(choices)

            all_stats = []
            for s in range(n_sessions):
                stats = compute_summary_stats(
                    choices=all_choices[s], stimuli=stimuli_per_session[s],
                    categories=categories_per_session[s],
                    stat_names=summary_stat_names, return_dict=False,
                )
                all_stats.append(stats)
            return np.concatenate(all_stats)

    elif model_type == 'sc':
        from models.SC_core import SCParams, SCState, SCModel

        def simulate(theta: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
            if seed is None:
                seed = np.random.randint(0, 2**31)
            rng = np.random.default_rng(seed)
            session_params = layout.theta_to_session_params(theta)

            sc_params_0 = SCParams(**session_params[0])
            state = (SCModel.create_initial_state(params=sc_params_0,
                                                   burn_in=burn_in,
                                                   seed=burn_in_seed)
                     if burn_in > 0 else SCState.initial_default())

            all_choices = []
            for s in range(n_sessions):
                sc_params = SCParams(**session_params[s])
                choices, _, state, _ = SCModel.simulate_session(
                    params=sc_params, initial_state=state,
                    stimuli=stimuli_per_session[s],
                    categories=categories_per_session[s], rng=rng,
                    no_response=no_response_per_session[s],
                    not_blockstart=not_blockstart_per_session[s],
                    return_history=False,
                )
                all_choices.append(choices)

            all_stats = []
            for s in range(n_sessions):
                stats = compute_summary_stats(
                    choices=all_choices[s], stimuli=stimuli_per_session[s],
                    categories=categories_per_session[s],
                    stat_names=summary_stat_names, return_dict=False,
                )
                all_stats.append(stats)
            return np.concatenate(all_stats)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return simulate


# =============================================================================
# OBSERVED STATS 
# =============================================================================

def compute_observed_stats(
    fitting_data: Any,
    summary_stat_names: List[str],
) -> np.ndarray:
    """Compute summary statistics from observed (real) FittingData."""
    from behav_utils.analysis.summary_stats import compute_summary_stats

    all_stats = []
    for s in range(fitting_data.n_sessions):
        sa = fitting_data.get_session(s)
        stats = compute_summary_stats(
            choices=sa['choices'], stimuli=sa['stimuli'],
            categories=sa['categories'],
            stat_names=summary_stat_names, return_dict=False,
        )
        all_stats.append(stats)
    return np.concatenate(all_stats)


# =============================================================================
# DEFAULT LINK SPECS
# =============================================================================

DEFAULT_SUMMARY_STATS = [
    'accuracy', 'psychometric', 'recency', 'win_stay', 'stimulus_sensitivity',
]

DEFAULT_BE_PARAM_LINKS = {
    'sigma_percep': ConstantSpec(bounds=(0.05, 0.5)),
    'A_repulsion': ConstantSpec(bounds=(0.0, 0.5)),
    'eta_learning': ConstantSpec(bounds=(0.05, 0.9)),
    'eta_relax': ConstantSpec(bounds=(0.01, 0.4)),
}

DEFAULT_SC_PARAM_LINKS = {
    'sigma_percep': ConstantSpec(bounds=(0.05, 0.5)),
    'A_repulsion': ConstantSpec(bounds=(0.0, 0.5)),
    'gamma': ConstantSpec(bounds=(0.1, 1.0)),
    'sigma_update': ConstantSpec(bounds=(0.1, 1.0)),
}


# =============================================================================
# SBI FITTER 
# =============================================================================

class SBIFitter:
    """
    High-level SBI fitting interface for BE and SC models.

    Takes FittingData and parameter link specifications, builds all
    components, and provides train/extract methods.

    Args:
        fitting_data: FittingData from AnimalData.get_fitting_data()
        model_type: 'be' or 'sc'
        param_links: Dict mapping param names to link specs.
        summary_stats: List of summary stat names.
        burn_in: Burn-in trials before first session
        burn_in_seed: Seed for burn-in simulation
    """

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
        param_links: Optional[Dict[str, LinkSpec]] = None,
        summary_stats: Optional[List[str]] = None,
        burn_in: int = 0,
        burn_in_seed: int = 42,
    ):
        self.fitting_data = fitting_data
        self.model_type = model_type.lower()
        self.burn_in = burn_in
        self.burn_in_seed = burn_in_seed

        if self.model_type == 'be':
            self._param_order = self.BE_PARAM_ORDER
        elif self.model_type == 'sc':
            self._param_order = self.SC_PARAM_ORDER
        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}")

        self.param_links = param_links or dict(self._DEFAULT_LINKS[self.model_type])
        self.summary_stats = summary_stats or list(DEFAULT_SUMMARY_STATS)

        for name in self._param_order:
            if name not in self.param_links:
                raise ValueError(
                    f"Missing link spec for '{name}'. "
                    f"Required: {self._param_order}"
                )

        self.layout = ThetaLayout(
            param_names=self._param_order,
            n_sessions=fitting_data.n_sessions,
            links=self.param_links,
            model_type=self.model_type,
        )

        self._stimuli = fitting_data.stimuli
        self._categories = fitting_data.categories
        self._choices = fitting_data.choices
        self._no_response = fitting_data.no_response
        self._not_blockstart = fitting_data.not_blockstart

        self._prior = None

        self.simulator = build_simulator(
            layout=self.layout,
            stimuli_per_session=self._stimuli,
            categories_per_session=self._categories,
            no_response_per_session=self._no_response,
            not_blockstart_per_session=self._not_blockstart,
            summary_stat_names=self.summary_stats,
            burn_in=self.burn_in, burn_in_seed=self.burn_in_seed,
            model_type=self.model_type,
        )

        self.observed_stats = compute_observed_stats(
            fitting_data, self.summary_stats,
        )

        # NaN masking
        self._valid_dims = np.isfinite(self.observed_stats)
        self._n_masked = int((~self._valid_dims).sum())

        if self._n_masked > 0:
            warnings.warn(
                f"NaN MASKING: {self._n_masked}/{len(self.observed_stats)} "
                f"observed stats are NaN/inf — masked from training."
            )
            self.observed_stats = self.observed_stats[self._valid_dims]
            _raw_sim = self.simulator
            _mask = self._valid_dims

            def _masked_sim(theta, seed=None):
                return _raw_sim(theta, seed=seed)[_mask]
            self.simulator = _masked_sim

    @property
    def prior(self):
        """Build and cache prior on first access (requires torch)."""
        if self._prior is None:
            self._prior = build_prior(self.layout)
        return self._prior

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
            f"Theta dim: {self.theta_dim}",
            f"  Constant: {self.layout.constant_params}",
            f"  Varying: {self.layout.varying_params}",
            f"Stats: {self.summary_stats} ({self.n_summary_stats} dims)",
            f"Burn-in: {self.burn_in}",
        ]
        if self._n_masked > 0:
            lines.append(f"WARNING: {self._n_masked} masked NaN dims")
        return "\n".join(lines)

    def _sample_theta_numpy(self, rng: np.random.Generator) -> np.ndarray:
        """Sample theta from uniform within bounds (no torch)."""
        theta = np.zeros(self.layout.total_dim)
        for name in self._param_order:
            link = self.param_links[name]
            sl = self.layout.slices[name]
            n_vals = sl.stop - sl.start
            theta[sl] = rng.uniform(link.bounds[0], link.bounds[1], n_vals)
        return theta

    def test_simulator(self, n_tests: int = 5, seed: int = 42) -> Dict[str, Any]:
        """Run test simulations to verify everything works."""
        rng = np.random.default_rng(seed)
        times, stats_list, nan_counts = [], [], []
        for i in range(n_tests):
            theta = self._sample_theta_numpy(rng)
            t0 = time.time()
            stats = self.simulator(theta, seed=seed + i)
            times.append(time.time() - t0)
            stats_list.append(stats)
            nan_counts.append(int(np.sum(np.isnan(stats))))

        mean_t = np.mean(times)
        return {
            'mean_time_per_sim': mean_t,
            'estimated_time_50k': f"{mean_t * 50000 / 3600:.1f} hours",
            'stats_dim': len(stats_list[0]),
            'nan_counts': nan_counts,
        }

    def train(
        self,
        n_simulations: int = 50_000,
        method: str = 'NPE',
        n_rounds: int = 1,
        seed: Optional[int] = 42,
        show_progress: bool = True,
        hidden_features: int = 50,
        num_transforms: int = 5,
        store_simulations: bool = True,
        **kwargs,
    ) -> SBIResult:
        """Train SBI density estimator. Returns SBIResult."""
        import torch

        observed_tensor = torch.tensor(
            self.observed_stats, dtype=torch.float32,
        )

        return train_sbi(
            simulator=self.simulator, prior=self.prior,
            observed_stats=observed_tensor, method=method,
            n_simulations=n_simulations, n_rounds=n_rounds,
            seed=seed, show_progress=show_progress,
            hidden_features=hidden_features,
            num_transforms=num_transforms,
            store_simulations=store_simulations,
            param_names=self.layout.get_expanded_names(),
            **kwargs,
        )

    def extract_trajectories(
        self, result: SBIResult, n_samples: int = 5000,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract per-parameter trajectories from posterior samples.

        Returns:
            Dict[param_name] → {mean, median, ci_low, ci_high, std,
                                 samples, session_indices, link_type}
        """
        samples = result.sample(n_samples).numpy()
        trajectories = self.layout.theta_to_trajectories(samples)
        session_idx = self.fitting_data.time_axis

        summaries = {}
        for name in self._param_order:
            vals = trajectories[name]
            if isinstance(self.param_links[name], ConstantSpec):
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
        self, result: SBIResult, n_samples: int = 5000,
        point_estimate: str = 'median',
    ) -> List[Dict[str, float]]:
        """Extract point-estimate parameters for each session."""
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

    def posterior_predictive_check(
        self, result: SBIResult, n_simulations: int = 200, seed: int = 42,
    ) -> Dict[str, Any]:
        """Run posterior predictive checks."""
        from behav_utils.analysis.summary_stats import get_stat_names_expanded

        samples = result.sample(n_simulations).numpy()
        simulated = np.zeros((n_simulations, len(self.observed_stats)))
        valid = 0
        for i in range(n_simulations):
            s = self.simulator(samples[i], seed=seed + i)
            if np.all(np.isfinite(s)):
                simulated[valid] = s
                valid += 1

        if valid < n_simulations:
            warnings.warn(f"PPC: {n_simulations - valid} sims had NaN/inf")
        simulated = simulated[:valid]

        p_values = np.mean(simulated >= self.observed_stats[np.newaxis, :], axis=0)

        per_session_names = get_stat_names_expanded(self.summary_stats)
        all_names = [f"s{s}_{n}" for s in range(self.n_sessions) for n in per_session_names]
        if self._n_masked > 0:
            all_names = [n for n, v in zip(all_names, self._valid_dims) if v]

        return {
            'observed': self.observed_stats, 'simulated': simulated,
            'p_values': p_values, 'stat_names': all_names,
        }


# =============================================================================
# QUICK FIT 
# =============================================================================

def quick_fit(
    fitting_data: Any, model_type: str = 'be',
    n_simulations: int = 30_000,
    varying_params: Optional[List[str]] = None,
    method: str = 'NPE', seed: int = 42,
    ) -> Tuple['SBIFitter', SBIResult, Dict]:
    """
    Deprecated: use SBIFitter directly.

    Returns (fitter, result, trajectories).
    """
    warnings.warn(
        "quick_fit is deprecated. Use SBIFitter directly with param_links.",
        DeprecationWarning, stacklevel=2,
    )


def train_per_animal_snpe(
    model_type: str,
    fitting_data: 'FittingData',
    stat_names: List[str],
    n_simulations: int = 10_000,
    burn_in: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train SNPE for one animal using its real stimulus sequence.

    This is the per-animal training path (as opposed to amortised training
    which trains once on generic data). Produces a posterior that is
    conditioned on this animal's specific stimulus sequence.

    Args:
        model_type: 'be' or 'sc'.
        fitting_data: FittingData for one animal.
        stat_names: Summary stat names (CAN include update_matrix).
        n_simulations: Number of training simulations.
        burn_in: Burn-in trials for model initialisation.
        seed: Random seed.

    Returns:
        Dict with 'posterior', 'prior', 'simulator', 'sbi_sim',
        'param_names', 'model_type', 'stat_names', 'burn_in',
        'training_time', 'n_valid'.
    """
    import torch
    from sbi.inference import SNPE
    from inference.simulator import (
        create_be_simulator, create_sc_simulator,
        get_sbi_prior, wrap_for_sbi,
    )

    name = model_type.upper()
    aid = fitting_data.animal_id
    pooled = fitting_data.pool()
    stim, cat = pooled['stimuli'], pooled['categories']

    print(f"  Training per-animal SNPE [{name}] for {aid} "
          f"({n_simulations:,} sims, {len(stim)} trials)...")

    creator = create_be_simulator if model_type == 'be' else create_sc_simulator
    sim = creator(stim, cat, stat_names=stat_names, burn_in=burn_in)
    prior = get_sbi_prior(sim)
    sbi_sim = wrap_for_sbi(sim)

    t0 = time.time()
    theta = prior.sample((n_simulations,))
    x = torch.stack([sbi_sim(t) for t in theta])

    valid = ~torch.any(torch.isnan(x), dim=1)
    n_valid = valid.sum().item()
    print(f"    {n_valid}/{n_simulations} valid "
          f"({100 * n_valid / n_simulations:.0f}%)")

    inference_engine = SNPE(prior=prior)
    inference_engine.append_simulations(theta[valid], x[valid])
    posterior = inference_engine.build_posterior(inference_engine.train())

    dt = time.time() - t0
    print(f"    Done in {dt / 60:.1f} min")

    return {
        'posterior': posterior, 'prior': prior,
        'simulator': sim, 'sbi_sim': sbi_sim,
        'param_names': sim.get_param_names(),
        'model_type': model_type, 'stat_names': stat_names,
        'burn_in': burn_in, 'training_time': dt, 'n_valid': n_valid,
    }
    
# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Result container
    'SBIResult',
    # Core training
    'train_sbi', 'sample_posterior', 'train_per_animal_snpe',
    # Building blocks
    'build_prior', 'build_simulator', 'compute_observed_stats',
    # High-level fitter
    'SBIFitter',
    # Defaults
    'DEFAULT_SUMMARY_STATS', 'DEFAULT_BE_PARAM_LINKS', 'DEFAULT_SC_PARAM_LINKS',
    # Deprecated (kept for backwards compatibility)
    'quick_posterior', 'compare_methods', 'train_multisession_sbi', 'quick_fit',
]