"""Amortised SBI engine for the BE / SC models.

Train once on synthetic curriculum-matched data, condition on many animals.

The observation x is built by ``inference.representation.to_stat_vector`` -- the
SAME function the simulator uses during training -- so the train and test
representations cannot diverge. There is deliberately NO test-time imputation
and NO automatic column dropping: choose ``stat_names`` so the chosen stats stay
finite at the session length you train/condition at (a systematically bad stat
shows up as a low valid-row count at train time, i.e. a loud failure, rather
than being silently dropped).

This module is the engine only. Model selection lives in
``inference.selection`` (held-out UM/CP CV, the same protocol as grid_search).
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from inference.types import ModelType
from inference.simulator import build_simulator, wrap_for_sbi
from inference.representation import to_stat_vector


def _as_model(model) -> ModelType:
    if isinstance(model, ModelType):
        return model
    return ModelType(str(getattr(model, 'value', model)).lower())


class AmortisedSBI:
    """Train once, condition on many animals (static representation).

    Args:
        model: ModelType or 'be'/'sc'.
        dist_schedule: One distribution name applied to every session, or a list
            of exactly N names. Should match the curriculum of the data that
            will be conditioned on.
        N: Sessions per simulated animal (match the typical real N).
        T: Trials per session (fixed).
        burn_in: Model burn-in per session.
        mode: 'pooled' or 'moments' (the representation; must match conditioning).
        stat_names: Summary stats (defaults to SBI_STATS).
    """

    def __init__(
        self,
        model,
        dist_schedule: Union[str, Sequence[str]] = 'uniform',
        N: int = 1,
        T: int = 350,
        burn_in: int = 1000,
        mode: str = 'pooled',
        stat_names: Optional[Sequence[str]] = None,
    ):
        self.model = _as_model(model)
        self.dist_schedule = dist_schedule
        self.N = N
        self.T = T
        self.burn_in = burn_in
        self.mode = mode

        if stat_names is None:
            from inference.constants import SBI_STATS
            stat_names = list(SBI_STATS)
        self.stat_names = list(stat_names)

        self._sim_fn, self._prior, self.param_names = build_simulator(
            self.model, dist_schedule=self.dist_schedule, N=self.N, T=self.T,
            burn_in=self.burn_in, mode=self.mode, stat_names=self.stat_names,
        )

        self._trained_posterior = None
        self._training_metadata = None

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        n_simulations: int = 50_000,
        seed: int = 42,
        show_progress: bool = True,
    ):
        """Train the amortised posterior estimator via SNPE."""
        if not TORCH_AVAILABLE:
            raise ImportError('torch required for SBI training')

        from sbi.inference import SNPE
        from sbi.utils import process_simulator
        import time as _time

        if self._prior is None:
            raise RuntimeError(
                'No prior (sbi/torch unavailable when the simulator was built).')

        if show_progress:
            print(f'Training AmortisedSBI [{self.model.value.upper()}] '
                  f'({n_simulations:,} sims, schedule={self.dist_schedule}, '
                  f'N={self.N}, T={self.T}, mode={self.mode})')

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        prior = self._prior
        sbi_sim = wrap_for_sbi(self._sim_fn, base_seed=(seed or 0))
        sbi_sim = process_simulator(sbi_sim, prior, is_numpy_simulator=False)

        t0 = _time.time()
        theta = prior.sample((n_simulations,))
        if show_progress:
            print(f'  Simulating {n_simulations:,} datasets...')
        x = sbi_sim(theta)

        # Row-filter only: drop simulations with any non-finite stat. No column
        # dropping and no imputation -- a column that is bad for most thetas
        # surfaces here as a low valid count.
        valid = torch.isfinite(x).all(dim=-1)
        n_valid = int(valid.sum().item())
        if show_progress:
            print(f'  {n_valid}/{n_simulations} valid sims '
                  f'({100 * n_valid / max(n_simulations, 1):.0f}%)')
        if n_valid == 0:
            raise RuntimeError(
                'All simulations produced non-finite stats. The chosen '
                'stat_names are not finite at this N/T; revise the stat set '
                "(or use mode='pooled').")
        theta, x = theta[valid], x[valid]

        inference_obj = SNPE(prior=prior)
        inference_obj.append_simulations(theta, x)
        if show_progress:
            print('  Training neural density estimator...')
        density_est = inference_obj.train(show_train_summary=show_progress)
        posterior = inference_obj.build_posterior(density_est)

        dt = _time.time() - t0
        if show_progress:
            print(f'  Done in {dt / 60:.1f} min')

        self._trained_posterior = posterior
        self._training_metadata = {
            'n_simulations': n_simulations,
            'n_valid': n_valid,
            'seed': seed,
            'dist_schedule': self.dist_schedule,
            'N': self.N, 'T': self.T, 'mode': self.mode,
            'training_time': dt,
        }
        return posterior

    # ── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path):
        """Save trained posterior + config. Simulator/prior rebuilt on load."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._trained_posterior is None:
            raise RuntimeError('Nothing to save -- call train() first.')

        save_data = {
            'model': self.model.value,
            'dist_schedule': self.dist_schedule,
            'N': self.N, 'T': self.T, 'burn_in': self.burn_in,
            'mode': self.mode,
            'stat_names': self.stat_names,
            'param_names': list(self.param_names),
            'trained_posterior': self._trained_posterior,
            'training_metadata': self._training_metadata,
            '_version': 4,
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, path) -> 'AmortisedSBI':
        """Load a trained AmortisedSBI. Rebuilds simulator/prior fresh."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls(
            model=data['model'],
            dist_schedule=data['dist_schedule'],
            N=data['N'], T=data['T'], burn_in=data['burn_in'],
            mode=data['mode'], stat_names=data['stat_names'],
        )
        obj._trained_posterior = data['trained_posterior']
        obj._training_metadata = data.get('training_metadata')
        return obj

    # ── Conditioning ─────────────────────────────────────────────────────────

    def condition(self, sessions: List, n_samples: int = 1000) -> Dict[str, Any]:
        """Condition the posterior on the observation built from sessions.

        x = to_stat_vector(sessions, mode, stat_names) -- the SAME builder used
        in training. Returns posterior samples, the point estimate (per-param
        median), and the median theta vector.
        """
        if self._trained_posterior is None:
            raise RuntimeError('No trained posterior -- call train() or load().')
        import torch as _torch

        x = to_stat_vector(sessions, mode=self.mode, stat_names=self.stat_names)
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError(
                'Observation contains non-finite stats; cannot condition. '
                'Revise stat_names for this data, or use a more robust mode.')

        obs = _torch.as_tensor(x, dtype=_torch.float32)
        samples = self._trained_posterior.sample(
            (n_samples,), x=obs, show_progress_bars=False)
        samples_np = samples.numpy()

        theta_median = np.median(samples_np, axis=0)
        point_estimate = {
            pn: float(theta_median[i]) for i, pn in enumerate(self.param_names)
        }
        return {
            'posterior_samples': samples_np,
            'param_names': list(self.param_names),
            'point_estimate': point_estimate,
            'theta_median': theta_median,
        }
