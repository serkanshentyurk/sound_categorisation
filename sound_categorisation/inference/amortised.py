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
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sound_categorisation.inference.representation import to_stat_vector
from sound_categorisation.inference.simulator import build_simulator, wrap_for_sbi
from sound_categorisation.inference.types import ModelType


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
        T: Trials per session: int (fixed) or (lo, hi) range drawn per session.
        burn_in: Model burn-in per session.
        mode: 'pooled' or 'moments' (the representation; must match conditioning).
        stat_names: Summary stats (defaults to SBI_STATS).
    """

    def __init__(
        self,
        model,
        dist_schedule: Union[str, Sequence[str]] = 'uniform',
        N: int = 1,
        T=350,
        burn_in: int = 1000,
        mode: str = 'pooled',
        stat_names: Sequence[str] | None = None,
    ):
        self.model = _as_model(model)
        self.dist_schedule = dist_schedule
        self.N = N
        self.T = T
        self.burn_in = burn_in
        self.mode = mode

        if stat_names is None:
            from sound_categorisation.inference.constants import SBI_STATS
            stat_names = list(SBI_STATS)
        self.stat_names = list(stat_names)

        self._sim_fn, self._prior, self.param_names = build_simulator(
            self.model, dist_schedule=self.dist_schedule, N=self.N, T=self.T,
            burn_in=self.burn_in, mode=self.mode, stat_names=self.stat_names,
        )

        self._trained_posterior = None
        self._training_metadata = None
        self._feature_medians = None

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

        import time as _time

        from sbi.inference import SNPE
        from sbi.utils import process_simulator

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

        # Median-impute non-finite features instead of dropping simulations:
        # learn per-column medians here and store them so condition() imputes
        # identically. A column non-finite in EVERY sim is fatal
        # (compute_feature_medians raises) -- that stat is unusable at this N/T
        # and must leave the stat set.
        from sound_categorisation.inference.representation import compute_feature_medians, impute_with_medians
        x_np = x.detach().cpu().numpy().astype(float)
        try:
            medians = compute_feature_medians(x_np)
        except ValueError as e:
            raise RuntimeError(f'{e} (N={self.N}, T={self.T})') from e
        bad_cells = ~np.isfinite(x_np)
        n_bad_rows = int(bad_cells.any(axis=1).sum())
        n_valid = n_simulations - n_bad_rows           # fully-finite sims (for metadata)
        x_np = impute_with_medians(x_np, medians)
        self._feature_medians = medians
        if show_progress:
            frac = 100 * n_bad_rows / max(n_simulations, 1)
            print(f'  median-imputed {int(bad_cells.sum())} non-finite cells '
                  f'across {n_bad_rows} sims ({frac:.1f}%); kept all {n_simulations:,}')
        x = torch.as_tensor(x_np, dtype=torch.float32)

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
            'feature_medians': self._feature_medians,
            '_version': 5,
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
        obj._feature_medians = data.get('feature_medians')
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

        from sound_categorisation.inference.constants import SBI_STATS
        if list(self.stat_names) != list(SBI_STATS) and not getattr(self, '_warned_stats', False):
            import warnings
            warnings.warn(f'network stat vector {self.stat_names} differs from the current SBI_STATS '
                          f'{list(SBI_STATS)}: this is a stale network (retrain) unless intended.',
                          stacklevel=2)
            self._warned_stats = True

        x = to_stat_vector(sessions, mode=self.mode, stat_names=self.stat_names)
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            if self._feature_medians is None:
                raise ValueError(
                    'Observation contains non-finite stats and no stored medians '
                    'to impute from (train()/load() a median-imputed net first).')
            from sound_categorisation.inference.representation import impute_with_medians
            x = impute_with_medians(x, self._feature_medians)

        obs = _torch.as_tensor(np.array(x, dtype=np.float32))
        samples_np, acceptance = self._sample_within_prior(obs, n_samples)

        theta_median = np.median(samples_np, axis=0)
        point_estimate = {
            pn: float(theta_median[i]) for i, pn in enumerate(self.param_names)
        }
        return {
            'posterior_samples': samples_np,
            'param_names': list(self.param_names),
            'point_estimate': point_estimate,
            'theta_median': theta_median,
            'acceptance_rate': acceptance,
        }

    # Budgeted rejection sampling. sbi's DirectPosterior.sample rejects flow samples that fall
    # outside the prior box and keeps proposing until it has n -- with no budget, so an
    # observation the flow puts ~0 mass on inside the prior (typically data from the OTHER
    # model, which the comparison conditions on deliberately) loops for hours. Here the flow is
    # sampled directly with a fixed proposal budget; what is accepted is resampled up to n, and
    # if nothing is accepted the proposals are clipped to the prior box. 'acceptance_rate' is
    # returned so a result can be flagged; a rate near 0 means "this net does not describe
    # these data", which is itself the answer the model comparison is after.
    LEAK_BUDGET = 400_000
    LEAK_BATCH = 20_000

    def _sample_within_prior(self, obs, n_samples: int):
        import torch as _torch
        from sound_categorisation.inference.simulator import get_bounds_arrays
        lo, hi = get_bounds_arrays(self.model)
        flow = getattr(self._trained_posterior, 'posterior_estimator', None)
        if flow is None:
            # unknown sbi internals: fall back to sbi's own (unbudgeted) sampler
            samples = self._trained_posterior.sample((n_samples,), x=obs, show_progress_bars=False)
            return samples.numpy(), float('nan')
        cond = obs.reshape(1, -1)
        accepted, proposed, last = [], 0, None
        n_acc = 0
        with _torch.no_grad():
            while n_acc < n_samples and proposed < self.LEAK_BUDGET:
                batch = min(self.LEAK_BATCH, self.LEAK_BUDGET - proposed)
                theta = flow.sample((batch,), condition=cond)
                theta = np.asarray(theta.reshape(-1, len(lo)).cpu().numpy(), dtype=float)
                proposed += len(theta)
                last = theta
                ok = np.all((theta >= lo) & (theta <= hi), axis=1)
                if ok.any():
                    accepted.append(theta[ok])
                    n_acc += int(ok.sum())
        rate = n_acc / max(proposed, 1)
        if n_acc >= n_samples:
            return np.vstack(accepted)[:n_samples], rate
        rng = np.random.default_rng(0)
        if n_acc > 0:
            pool = np.vstack(accepted)
            return pool[rng.integers(0, len(pool), size=n_samples)], rate
        clipped = np.clip(last[:n_samples], lo, hi)
        if len(clipped) < n_samples:
            clipped = clipped[rng.integers(0, len(clipped), size=n_samples)]
        return clipped, rate

    # ── Accessors (for validation / plotting) ────────────────────────────────

    @property
    def posterior(self):
        """The trained sbi posterior (None until train()/load())."""
        return self._trained_posterior

    @property
    def simulator(self):
        """The simulator callable theta -> x, matching this net's config."""
        return self._sim_fn

    @property
    def prior(self):
        """The sbi prior (BoxUniform; None if sbi/torch unavailable)."""
        return self._prior
