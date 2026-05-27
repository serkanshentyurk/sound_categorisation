"""
High-level SBIFitter API.

Per-animal SBI fitting interface that wraps the lower-level builders
and core training functions. Supports time-varying parameters via
ConstantSpec, GPSpec, RandomWalkSpec.

Public API:
    SBIFitter — FittingData → trained posterior, with parameter linking

Usage:
    from inference import SBIFitter
    from inference.types import ConstantSpec, GPSpec

    fitter = SBIFitter(
        fitting_data=fd,
        param_links={
            'sigma_percep': ConstantSpec(bounds=(0.05, 0.5)),
            'eta_learning': GPSpec(bounds=(0.05, 0.9)),
            'eta_relax':    ConstantSpec(bounds=(0.01, 0.4)),
            'A_repulsion':  ConstantSpec(bounds=(0.0, 0.5)),
        },
        model_type='be',
    )
    fitter.train(n_simulations=10000)
    trajectory = fitter.extract_trajectories()
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Callable, Union, Any

# Lazy torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from inference.types import (
    ConstantSpec, GPSpec, RandomWalkSpec,
    ThetaLayout, LinkSpec,
)
from inference.sbi_core import SBIResult, train_sbi, _wrap_simulator_for_sbi
from inference.builders import (
    build_prior, build_simulator, compute_observed_stats,
    DEFAULT_SUMMARY_STATS, DEFAULT_BE_PARAM_LINKS, DEFAULT_SC_PARAM_LINKS,
)


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

