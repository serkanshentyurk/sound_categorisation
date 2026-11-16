"""Amortised SBI engine for the BE / SC models.

Train once on synthetic curriculum-matched data, condition on many animals.

The observation x is built by ``inference.representation.to_stat_vector`` -- the
SAME function the simulator uses during training -- so the train and test
representations cannot diverge. There is deliberately NO test-time imputation
and NO automatic column dropping: choose ``stat_names`` so the chosen stats stay
finite at the session length you train/condition at (a systematically bad stat
shows up as a low valid-row count at train time, i.e. a loud failure, rather
than being silently dropped).

A network belongs to a prior variant (``models.priors``; the one in use when it is trained, recorded
in its file) and, optionally, to the accuracy rule: only simulations whose every session reaches
``min_accuracy`` are kept, as every selected real session does. Inside, the density estimator works
in flat space (a log-uniform parameter as its logarithm); everything that goes in or comes out of
this class is in the parameters' own units.

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
from sound_categorisation.inference.simulator import build_simulator
from sound_categorisation.inference.types import ModelType
from sound_categorisation.models import priors


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
        design_features: How much data x was built from, appended after the stats
            (defaults to settings.SBI_DESIGN_FEATURES). Networks saved without them load
            with an empty tuple and condition exactly as they were trained.
        prior: The prior variant (``models.priors``); None = the one in use
            (``settings.SBI_PRIOR``). A file saved without one is a ``manuscript`` network.
        min_accuracy: The accuracy rule: train only on simulations whose every session
            reaches it. None = no rule.
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
        design_features: Sequence[str] | None = None,
        prior: str | None = None,
        min_accuracy: float | None = None,
    ):
        self.model = _as_model(model)
        self.prior_name = priors.resolve(prior)
        self.min_accuracy = None if min_accuracy is None else float(min_accuracy)
        self.dist_schedule = dist_schedule
        self.N = N
        self.T = T
        self.burn_in = burn_in
        self.mode = mode

        if stat_names is None:
            from sound_categorisation.inference.constants import SBI_STATS
            stat_names = list(SBI_STATS)
        self.stat_names = list(stat_names)
        if design_features is None:
            from sound_categorisation.settings import SBI_DESIGN_FEATURES
            design_features = SBI_DESIGN_FEATURES
        self.design_features = tuple(design_features)

        self._sim_fn, self._prior, self.param_names = build_simulator(
            self.model, dist_schedule=self.dist_schedule, N=self.N, T=self.T,
            burn_in=self.burn_in, mode=self.mode, stat_names=self.stat_names,
            design_features=self.design_features, prior=self.prior_name,
            min_accuracy=self.min_accuracy,
        )

        self._trained_posterior = None
        self._training_metadata = None
        self._feature_medians = None
        self._prior_bounds = None        # {'low': [...], 'high': [...]} the network was trained on

    # ── The prior this network belongs to ───────────────────────────────────

    def bounds(self):
        """(low, high) arrays of this network's prior, in the parameters' own units."""
        return priors.bounds_arrays(self.model.value, self.prior_name)

    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` parameter draws from this network's prior, ``(n, n_params)``, own units."""
        return priors.sample_prior(self.model.value, self.prior_name, n, rng)[1]

    # ── Training ─────────────────────────────────────────────────────────────

    SIM_CHUNK = 5_000          # most prior draws simulated in one round

    def simulate_training_set(self, n_keep: int, seed: int = 42, n_jobs: int = -1,
                              max_draws: int | None = None, show_progress: bool = True):
        """Draw from the prior and simulate until ``n_keep`` simulations are kept.

        Draw ``i`` (0-based) takes its parameters from one stream (``default_rng(seed)``) and is
        simulated with seed ``seed + i``; the training set is the first ``n_keep`` draws, in draw
        order, that the accuracy rule keeps (all of them without the rule). The result therefore
        does not depend on ``n_jobs`` or on how the draws are split into rounds.

        Returns ``(theta, x, n_drawn, n_passed)``: ``theta`` ``(n_keep, n_params)`` in the
        parameters' own units, ``x`` ``(n_keep, n_features)``, how many draws were simulated and
        how many of those passed (the last round may pass a few more than are kept). Raises if
        ``max_draws`` (default ``settings.SBI_MAX_DRAWS``) is reached first.
        """
        import time as _time
        if max_draws is None:
            from sound_categorisation.settings import SBI_MAX_DRAWS
            max_draws = SBI_MAX_DRAWS
        seed = int(seed or 0)
        rng = np.random.default_rng(seed)
        kept_theta, kept_x, n_kept, n_passed, n_drawn, t0 = [], [], 0, 0, 0, _time.time()
        while n_kept < n_keep:
            if n_drawn >= max_draws:
                raise RuntimeError(
                    f'{self.model.value.upper()}: {n_kept:,} of {n_keep:,} simulations kept after '
                    f'{n_drawn:,} prior draws (accuracy rule >= {self.min_accuracy}); '
                    'raise max_draws / settings.SBI_MAX_DRAWS or lower n_simulations')
            if self.min_accuracy is None:
                want = n_keep - n_kept                       # every draw is kept: none is wasted
            else:                                            # enough to finish at the rate so far
                rate = n_passed / n_drawn if n_drawn else 1.0
                want = max(int(np.ceil(1.1 * (n_keep - n_kept) / max(rate, 0.02))), 64)
            chunk = int(min(want, self.SIM_CHUNK, max_draws - n_drawn))
            thetas = self.sample_prior(chunk, rng)
            if n_jobs == 1:
                xs = [self._sim_fn(thetas[i], seed=seed + n_drawn + i) for i in range(chunk)]
            else:
                from joblib import Parallel, delayed, parallel_config

                # one thread per worker: the job's thread count is meant for the density estimator,
                # and workers that each took all of it would oversubscribe the cores
                with parallel_config(backend='loky', inner_max_num_threads=1):
                    xs = Parallel(n_jobs=n_jobs, batch_size=16)(
                        delayed(self._sim_fn)(thetas[i], seed=seed + n_drawn + i) for i in range(chunk))
            n_drawn += chunk
            for theta, x in zip(thetas, xs):
                if x is None:
                    continue
                n_passed += 1
                if n_kept < n_keep:
                    kept_theta.append(theta)
                    kept_x.append(np.asarray(x, dtype=float))
                    n_kept += 1
            if show_progress:
                print(f'    {n_drawn:,} drawn, {n_passed:,} passed ({100 * n_passed / n_drawn:.1f}%), '
                      f'{n_kept:,} of {n_keep:,} kept, {(_time.time() - t0) / 60:.1f} min', flush=True)
        if n_jobs != 1:
            # hand the cores back before the density estimator trains, and leave no worker behind
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        return np.vstack(kept_theta), np.vstack(kept_x), n_drawn, n_passed

    def train(
        self,
        n_simulations: int = 50_000,
        seed: int = 42,
        show_progress: bool = True,
        n_jobs: int = -1,
        max_draws: int | None = None,
    ):
        """Train the amortised posterior estimator via SNPE on ``n_simulations`` kept simulations
        (``simulate_training_set``)."""
        if not TORCH_AVAILABLE:
            raise ImportError('torch required for SBI training')

        import time as _time

        from sbi.inference import SNPE

        if self._prior is None:
            raise RuntimeError(
                'No prior (sbi/torch unavailable when the simulator was built).')

        if show_progress:
            rule = 'none' if self.min_accuracy is None else f'every session >= {self.min_accuracy:g}'
            print(f'Training AmortisedSBI [{self.model.value.upper()}] '
                  f'({n_simulations:,} sims kept, schedule={self.dist_schedule}, '
                  f'N={self.N}, T={self.T}, mode={self.mode}, '
                  f'design={list(self.design_features)}, prior={self.prior_name}, '
                  f'accuracy rule: {rule})')

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        t0 = _time.time()
        if show_progress:
            print(f'  Simulating until {n_simulations:,} datasets are kept...')
        theta_np, x_np, n_drawn, n_passed = self.simulate_training_set(
            n_simulations, seed=(seed or 0), n_jobs=n_jobs, max_draws=max_draws,
            show_progress=show_progress)
        t_sim = _time.time() - t0
        if show_progress:
            print(f'  Simulated in {t_sim / 60:.1f} min: {n_drawn:,} draws, {n_passed:,} passed '
                  f'({100 * n_passed / n_drawn:.1f}%), {n_simulations:,} kept')

        # Median-impute non-finite features instead of dropping simulations:
        # learn per-column medians here and store them so condition() imputes
        # identically. A column non-finite in EVERY sim is fatal
        # (compute_feature_medians raises) -- that stat is unusable at this N/T
        # and must leave the stat set.
        from sound_categorisation.inference.representation import compute_feature_medians, impute_with_medians
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
        # the density estimator sees flat space, where this network's prior is a uniform box
        theta = torch.as_tensor(priors.to_flat(self.model.value, self.prior_name, theta_np),
                                dtype=torch.float32)

        inference_obj = SNPE(prior=self._prior)
        inference_obj.append_simulations(theta, x)
        if show_progress:
            print('  Training neural density estimator...')
        t1 = _time.time()
        density_est = inference_obj.train(show_train_summary=show_progress)
        posterior = inference_obj.build_posterior(density_est)
        t_fit = _time.time() - t1

        dt = _time.time() - t0
        if show_progress:
            print(f'  Density estimator trained in {t_fit / 60:.1f} min')
            print(f'  Done in {dt / 60:.1f} min')

        lo, hi = self.bounds()
        self._trained_posterior = posterior
        self._prior_bounds = {'low': lo.tolist(), 'high': hi.tolist()}
        self._training_metadata = {
            'n_simulations': n_simulations,
            'n_valid': n_valid,
            'n_drawn': int(n_drawn),
            'pass_rate': float(n_passed / n_drawn),
            'seed': seed,
            'dist_schedule': self.dist_schedule,
            'N': self.N, 'T': self.T, 'mode': self.mode,
            'design_features': list(self.design_features),
            'prior': self.prior_name,
            'min_accuracy': self.min_accuracy,
            'simulation_time': t_sim,
            'fit_time': t_fit,
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
            'design_features': list(self.design_features),
            'param_names': list(self.param_names),
            'prior': self.prior_name,
            'log_uniform': list(priors.log_uniform_params(self.model.value, self.prior_name)),
            'min_accuracy': self.min_accuracy,
            'prior_bounds': self._prior_bounds,
            'trained_posterior': self._trained_posterior,
            'training_metadata': self._training_metadata,
            'feature_medians': self._feature_medians,
            '_version': 7,
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, path) -> 'AmortisedSBI':
        """Load a trained AmortisedSBI. Rebuilds simulator/prior fresh, for the prior variant and
        accuracy rule the file records (a file from before they existed: ``manuscript``, no rule).

        Raises if the saved parameter names differ from the model's current order: the
        network's output columns would otherwise be relabelled silently.
        """
        data = read_network_file(path)
        obj = cls(
            model=data['model'],
            dist_schedule=data['dist_schedule'],
            N=data['N'], T=data['T'], burn_in=data['burn_in'],
            mode=data['mode'], stat_names=data['stat_names'],
            design_features=tuple(data.get('design_features') or ()),
            prior=network_prior(data), min_accuracy=data.get('min_accuracy'),
        )
        saved_names = data.get('param_names')
        if saved_names is not None and list(saved_names) != list(obj.param_names):
            raise ValueError(
                f'{path}: the network was trained with parameters {list(saved_names)}, but '
                f'{obj.model.value.upper()} now has {list(obj.param_names)}; its columns would '
                'be mislabelled. Retrain it.')
        obj._trained_posterior = data['trained_posterior']
        obj._training_metadata = data.get('training_metadata')
        obj._feature_medians = data.get('feature_medians')
        obj._prior_bounds = data.get('prior_bounds')
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

        from sound_categorisation.settings import SBI_DESIGN_FEATURES
        if (tuple(self.design_features) != tuple(SBI_DESIGN_FEATURES)
                and not getattr(self, '_warned_design', False)):
            import warnings
            warnings.warn(f'network design features {list(self.design_features)} differ from the '
                          f'current SBI_DESIGN_FEATURES {list(SBI_DESIGN_FEATURES)}: this network '
                          'predates them (retrain) unless intended.', stacklevel=2)
            self._warned_design = True

        if self.prior_name != priors.current() and not getattr(self, '_warned_prior', False):
            import warnings
            warnings.warn(f"network trained on the '{self.prior_name}' prior; the prior in use is "
                          f"'{priors.current()}': this is an earlier network unless intended.",
                          stacklevel=2)
            self._warned_prior = True

        x = to_stat_vector(sessions, mode=self.mode, stat_names=self.stat_names,
                           design_features=self.design_features)
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
            # fewer than n_samples fell inside the prior within LEAK_BUDGET proposals, so the
            # samples were resampled from what was accepted (or clipped if nothing was)
            'budget_hit': bool(np.isfinite(acceptance)
                               and acceptance * self.LEAK_BUDGET < n_samples),
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
        """``n_samples`` posterior samples inside the prior box, in the parameters' own units, and
        the acceptance rate. The flow samples flat space; the box is this network's own prior."""
        samples, rate = self._sample_flat(obs, n_samples)
        return priors.from_flat(self.model.value, self.prior_name, samples), rate

    def _sample_flat(self, obs, n_samples: int):
        import torch as _torch

        lo, hi = priors.flat_bounds(self.model.value, self.prior_name)
        flow = getattr(self._trained_posterior, 'posterior_estimator', None)
        if flow is None:
            # unknown sbi internals: fall back to sbi's own (unbudgeted) sampler
            samples = self._trained_posterior.sample((n_samples,), x=obs, show_progress_bars=False)
            return np.asarray(samples.numpy(), dtype=float), float('nan')
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
        """The simulator callable theta -> x, matching this net's config (theta in the parameters'
        own units; None for a draw this net's accuracy rule rejects)."""
        return self._sim_fn

    @property
    def prior(self):
        """The sbi prior: a BoxUniform over this net's prior in flat space (None if sbi/torch
        unavailable)."""
        return self._prior


# =============================================================================
# RAW NETWORK FILES (read without building anything; used by load and the network check)
# =============================================================================

def read_network_file(path) -> Dict[str, Any]:
    """The saved dict of a network file, as written by AmortisedSBI.save.

    Unpickling the posterior needs torch/sbi; everything else is plain Python.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def network_prior(data: Dict[str, Any]) -> str:
    """The prior variant a saved network was trained on. Files from before the variants existed
    (mid-November 2026) were trained on the ``manuscript`` ranges."""
    return data.get('prior') or 'manuscript'


def trained_prior_bounds(data: Dict[str, Any]):
    """(low, high) arrays of the prior a saved network was trained on, in the parameters' own
    units, or None.

    Networks saved from Nov 2026 store them; older ones keep them only inside the
    pickled sbi posterior (a BoxUniform), which is read here when available.
    """
    stored = data.get('prior_bounds')
    if stored:
        return np.asarray(stored['low'], dtype=float), np.asarray(stored['high'], dtype=float)
    prior = getattr(data.get('trained_posterior'), 'prior', None)
    for owner in (getattr(prior, 'base_dist', None), prior):
        low, high = getattr(owner, 'low', None), getattr(owner, 'high', None)
        if low is not None and high is not None:
            return _as_np(low), _as_np(high)
    support = getattr(prior, 'support', None)
    low = getattr(support, 'lower_bound', None)
    high = getattr(support, 'upper_bound', None)
    if low is not None and high is not None:
        return _as_np(low), _as_np(high)
    return None


def _as_np(t) -> np.ndarray:
    if hasattr(t, 'detach'):
        t = t.detach().cpu().numpy()
    return np.asarray(t, dtype=float).reshape(-1)

