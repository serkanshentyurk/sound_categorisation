"""
Amortised SBI for Static Model Comparison

Train a neural density estimator once on curriculum-matched generic data,
then condition on each animal's observed summary statistics to get a
posterior over model parameters. Produces true held-out CV errors
comparable to grid-search CV.

Usage:
    from inference.amortised import AmortisedSBI

    # Train (once per model type × curriculum)
    trainer = AmortisedSBI(
        model_type='be',
        curriculum=[('uniform', 15)],
    )
    trainer.train(n_simulations=50_000)
    trainer.save('results/snpe/uniform_be.pkl')

    # Condition (per animal, ~1 sec each)
    loaded = AmortisedSBI.load('results/snpe/uniform_be.pkl')
    result = loaded.fit(
        sessions=clean_sessions,
        animal_id='SS01',
        fit_target='update_matrix',
    )
    # result['cv_errors'] — true held-out, same schema as grid_search.py

Public API:
    AmortisedSBI           — Main class
    build_curriculum_simulator — Curriculum-aware simulator factory
"""

import pickle
import warnings
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import torch #type: ignore
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# CURRICULUM SIMULATOR
# =============================================================================

def build_curriculum_simulator(
    model_type: str,
    curriculum: List[Tuple[str, int]],
    trials_per_session: int = 350,
    burn_in: int = 1000,
    stat_names: Optional[List[str]] = None,
):
    """
    Build a simulator that generates summary stats for a full curriculum.

    The simulator:
        1. Receives a parameter vector θ
        2. Generates stimuli per the curriculum schedule
        3. Simulates the model across all sessions, chaining state
        4. Computes summary stats per session, concatenates into one vector

    Args:
        model_type: 'be' or 'sc'.
        curriculum: List of (distribution, n_sessions) tuples.
            E.g. [('uniform', 15)] or [('uniform', 10), ('hard_a', 5)].
        trials_per_session: Trials per simulated session.
        burn_in: Burn-in trials before the first session.
        stat_names: Summary stats to compute per session.

    Returns:
        (simulator_fn, prior, param_names, n_stats_per_session)
        where simulator_fn(theta, seed) -> stats_vector
    """
    from behav_utils.analysis.summary_stats import (
        compute_summary_stats, get_stat_names_expanded,
    )

    if stat_names is None:
        from scripts.config import SBI_STATS
        stat_names = list(SBI_STATS)

    # Count total sessions
    total_sessions = sum(n_sess for _, n_sess in curriculum)

    # Build distribution schedule
    dist_schedule = []
    for dist_name, n_sess in curriculum:
        dist_schedule.extend([dist_name] * n_sess)

    # Determine stats dimensions
    expanded = get_stat_names_expanded(stat_names)
    n_stats_per_session = len(expanded)

    model_type = model_type.lower()

    if model_type == 'be':
        from models.BE_core import BEParams, BEState, BEModel
        param_names = ['sigma_percep', 'A_repulsion', 'eta_learning', 'eta_relax']
        bounds = BEParams.get_bounds()

        def _make_params(theta):
            return BEParams(
                sigma_percep=float(theta[0]),
                A_repulsion=float(theta[1]),
                eta_learning=float(theta[2]),
                eta_relax=float(theta[3]),
            )

        def _create_state(params, burn_in_n, seed):
            return BEModel.create_initial_state(
                burn_in=burn_in_n, params=params, seed=seed)

        def _simulate_session(params, state, stimuli, categories, rng):
            choices, _, final_state, _ = BEModel.simulate_session(
                params, state, stimuli, categories, rng,
                return_history=False)
            return choices, final_state

    elif model_type == 'sc':
        from models.SC_core import SCParams, SCState, SCModel
        param_names = ['sigma_percep', 'A_repulsion', 'gamma', 'sigma_update']
        bounds = SCParams.get_bounds()

        def _make_params(theta):
            return SCParams(
                sigma_percep=float(theta[0]),
                A_repulsion=float(theta[1]),
                gamma=float(theta[2]),
                sigma_update=float(theta[3]),
            )

        def _create_state(params, burn_in_n, seed):
            return SCModel.create_initial_state(
                burn_in=burn_in_n, params=params, seed=seed)

        def _simulate_session(params, state, stimuli, categories, rng):
            choices, _, final_state, _ = SCModel.simulate_session(
                params, state, stimuli, categories, rng,
                return_history=False)
            return choices, final_state
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # Build prior (uniform box)
    if TORCH_AVAILABLE:
        from sbi.utils import BoxUniform #type: ignore
        lower = torch.tensor([bounds[p][0] for p in param_names]) 
        upper = torch.tensor([bounds[p][1] for p in param_names])
        prior = BoxUniform(low=lower, high=upper)
    else:
        prior = None

    # Stimulus sampler
    def _sample_stimuli(dist_name, n_trials, rng):
        if dist_name in ('hard_a', 'hard_b'):
            from analysis.stimulus_distribution import sample_distribution
            return sample_distribution(n_trials, dist_name, rng=rng)
        else:
            stimuli = rng.uniform(-1, 1, n_trials)
            categories = (stimuli > 0).astype(int)
            return stimuli, categories

    def simulator(theta, seed=None):
        """Simulate full curriculum, return concatenated stats."""
        if seed is None:
            seed = np.random.randint(0, 2**31)
        rng = np.random.default_rng(seed)
        theta_np = np.asarray(theta, dtype=float)

        params = _make_params(theta_np)
        state = _create_state(params, burn_in, seed)

        all_stats = []
        for s_idx, dist_name in enumerate(dist_schedule):
            stimuli, categories = _sample_stimuli(
                dist_name, trials_per_session, rng)

            choices, state = _simulate_session(
                params, state, stimuli, categories, rng)

            stats = compute_summary_stats(
                choices=choices, stimuli=stimuli, categories=categories,
                stat_names=stat_names, return_dict=False,
            )
            all_stats.append(stats)

        return np.concatenate(all_stats)

    return simulator, prior, param_names, n_stats_per_session


# =============================================================================
# OBSERVED STATS FROM REAL SESSIONS
# =============================================================================

def compute_observed_stats_from_sessions(
    sessions: list,
    stat_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute summary stats per session, concatenate into one vector."""
    from behav_utils.analysis.summary_stats import compute_summary_stats
    from behav_utils.data.filtering import pool_arrays

    if stat_names is None:
        from scripts.config import SBI_STATS
        stat_names = list(SBI_STATS)

    all_stats = []
    for sess in sessions:
        # pool_arrays on a single session gives us clean arrays
        pooled = pool_arrays([sess])
        stim = pooled['stimuli']
        ch = pooled['choices']
        cat = pooled['categories']
        valid = ~pooled.get('no_response', np.isnan(ch))

        stats = compute_summary_stats(
            choices=ch[valid],
            stimuli=stim[valid],
            categories=cat[valid],
            stat_names=stat_names,
            return_dict=False,
        )
        all_stats.append(stats)

    return np.concatenate(all_stats)


# =============================================================================
# SIMULATE FROM PARAMS (for PPC / CV evaluation)
# =============================================================================

def simulate_choices_from_params(
    model_type: str,
    params_dict: dict,
    stimuli: np.ndarray,
    categories: np.ndarray,
    burn_in: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate choices from specific parameter values.

    Thin wrapper around BEModel/SCModel.simulate_session.
    """
    rng = np.random.default_rng(seed)

    if model_type.lower() == 'be':
        from models.BE_core import BEParams, BEModel
        params = BEParams(**params_dict)
        state = BEModel.create_initial_state(
            burn_in=burn_in, params=params, seed=seed)
        choices, _, _, _ = BEModel.simulate_session(
            params, state, stimuli, categories, rng,
            return_history=False)
    elif model_type.lower() == 'sc':
        from models.SC_core import SCParams, SCModel
        params = SCParams(**params_dict)
        state = SCModel.create_initial_state(
            burn_in=burn_in, params=params, seed=seed)
        choices, _, _, _ = SCModel.simulate_session(
            params, state, stimuli, categories, rng,
            return_history=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return choices


# =============================================================================
# AMORTISED SBI CLASS
# =============================================================================

class AmortisedSBI:
    """
    Train once on curriculum-matched generic data, condition on many animals.

    Produces true held-out CV errors comparable to grid-search CV.
    Fold splitting is always first-half / second-half (preserving
    temporal order, matching GS convention).
    """

    def __init__(
        self,
        model_type: str,
        curriculum: List[Tuple[str, int]],
        trials_per_session: int = 350,
        burn_in: int = 1000,
        stat_names: Optional[List[str]] = None,
    ):
        self.model_type = model_type.lower()
        self.curriculum = list(curriculum)
        self.trials_per_session = trials_per_session
        self.burn_in = burn_in

        if stat_names is None:
            from scripts.config import SBI_STATS
            stat_names = list(SBI_STATS)
        self.stat_names = stat_names

        self.total_sessions = sum(n for _, n in curriculum)

        (self._simulator, self._prior, self.param_names,
         self._n_stats_per_session) = build_curriculum_simulator(
            model_type=self.model_type,
            curriculum=self.curriculum,
            trials_per_session=self.trials_per_session,
            burn_in=self.burn_in,
            stat_names=self.stat_names,
        )

        self._trained_posterior = None
        self._training_metadata = None

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        n_simulations: int = 50_000,
        seed: int = 42,
        method: str = 'NPE',
        show_progress: bool = True,
    ):
        """
        Train the amortised posterior estimator.

        Uses sbi package's SNPE (Neural Posterior Estimation).
        The simulator is wrapped to handle torch tensors as
        required by the sbi package.
        """
        if not TORCH_AVAILABLE:
            raise ImportError('torch required for SBI training')

        import torch #type: ignore
        from sbi.inference import SNPE #type: ignore
        from sbi.utils import process_simulator #type: ignore

        print(f'Training AmortisedSBI [{self.model_type.upper()}] '
              f'({n_simulations:,} sims, curriculum={self.curriculum})')

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Wrap numpy simulator → torch (sbi convention)
        raw_sim = self._simulator
        sim_seed_counter = [0]

        def _sbi_simulator(theta_batch):
            """Batched: theta_batch is (batch, n_params) tensor."""
            results = []
            for theta in theta_batch:
                sim_seed_counter[0] += 1
                x = raw_sim(theta.numpy(), seed=sim_seed_counter[0])
                results.append(torch.tensor(x, dtype=torch.float32))
            return torch.stack(results)

        prior = self._prior
        simulator_fn, prior_sbi = process_simulator(
            _sbi_simulator, prior, is_numpy_simulator=False)

        # Simulate training data
        import time as _time
        t0 = _time.time()

        theta = prior_sbi.sample((n_simulations,))
        if show_progress:
            print(f'  Simulating {n_simulations:,} datasets...')
        x = simulator_fn(theta)

        # Remove NaN simulations
        valid = torch.isfinite(x).all(dim=-1)
        n_valid = valid.sum().item()
        if show_progress:
            print(f'  {n_valid}/{n_simulations} valid '
                  f'({100 * n_valid / n_simulations:.0f}%)')
        theta, x = theta[valid], x[valid]

        # Train SNPE
        inference_obj = SNPE(prior=prior_sbi)
        inference_obj.append_simulations(theta, x)
        if show_progress:
            print('  Training neural density estimator...')
        density_est = inference_obj.train(show_train_summary=show_progress)
        posterior = inference_obj.build_posterior(density_est)

        dt = _time.time() - t0
        if show_progress:
            print(f'  Done in {dt / 60:.1f} min')

        # Store as SBIResult for compatibility with sample_posterior()
        from inference.fitting import SBIResult
        result = SBIResult(
            posterior=posterior,
            inference=inference_obj,
            density_estimator=density_est,
            method=method,
            n_simulations=n_simulations,
            n_rounds=1,
            training_time=dt,
            theta_train=theta,
            x_train=x,
            prior=prior_sbi,
            observed_stats=None,  # amortised — no default observation
            param_names=list(self.param_names),
        )

        self._trained_posterior = result
        self._training_metadata = {
            'n_simulations': n_simulations,
            'n_valid': n_valid,
            'seed': seed,
            'method': method,
            'curriculum': self.curriculum,
            'training_time': dt,
        }
        return result

    # ── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path):
        """
        Save trained posterior + config.
        
        The simulator and prior are NOT saved (rebuilt on load per
        the pickle fragility rule). Only the trained SBIResult and
        the configuration needed to rebuild are persisted.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._trained_posterior is None:
            raise RuntimeError('Nothing to save — call train() first.')

        save_data = {
            'model_type': self.model_type,
            'curriculum': self.curriculum,
            'trials_per_session': self.trials_per_session,
            'burn_in': self.burn_in,
            'stat_names': self.stat_names,
            'param_names': list(self.param_names),
            'trained_posterior': self._trained_posterior,
            'training_metadata': self._training_metadata,
            '_version': 1,
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, path) -> 'AmortisedSBI':
        """
        Load a trained AmortisedSBI.
        
        Rebuilds the simulator and prior fresh from the saved config
        (not from pickle). Only the trained posterior is loaded from disk.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        obj = cls(
            model_type=data['model_type'],
            curriculum=data['curriculum'],
            trials_per_session=data['trials_per_session'],
            burn_in=data['burn_in'],
            stat_names=data['stat_names'],
        )
        obj._trained_posterior = data['trained_posterior']
        obj._training_metadata = data.get('training_metadata')
        return obj

    # ── Conditioning ─────────────────────────────────────────────────────────

    def condition(
        self,
        sessions: list,
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Condition trained posterior on one animal's observed stats.

        Args:
            sessions: Pre-filtered SessionData list.
            n_samples: Number of posterior samples.

        Returns:
            {
                'posterior_samples': np.ndarray (n_samples, n_params),
                'param_names': list[str],
                'point_estimate': dict (posterior median),
            }
        """
        if self._trained_posterior is None:
            raise RuntimeError('No trained posterior. Call train() or load() first.')

        import torch #type: ignore
        from inference.fitting import sample_posterior

        observed = compute_observed_stats_from_sessions(
            sessions, self.stat_names)
        obs_tensor = torch.tensor(observed, dtype=torch.float32)

        # sample_posterior accepts SBIResult or raw posterior,
        # plus observed_stats tensor. Returns (n_samples, n_params) tensor.
        samples_tensor = sample_posterior(
            self._trained_posterior, obs_tensor,
            n_samples=n_samples, method='direct',
            show_progress=False,
        )
        samples_np = samples_tensor.numpy()

        point_estimate = {
            pn: float(np.median(samples_np[:, i]))
            for i, pn in enumerate(self.param_names)
        }

        return {
            'posterior_samples': samples_np,
            'param_names': list(self.param_names),
            'point_estimate': point_estimate,
        }

    # ── Held-out CV ──────────────────────────────────────────────────────────

    def fit(
        self,
        sessions: list,
        animal_id: str = 'unknown',
        distribution: str = 'uniform',
        fit_target: str = 'update_matrix',
        n_folds: int = 2,
        n_posterior_samples: int = 50,
        n_stochastic_reps: int = 10,
    ) -> Dict[str, Any]:
        """
        Full pipeline: condition + held-out CV errors.

        CV protocol (true held-out, temporal order preserved):
            1. Split sessions into first half / second half
            2. For each fold pair (train, test):
                a. Condition posterior on TRAIN sessions' stats
                b. Draw posterior samples
                c. For each sample × stochastic rep:
                    - Simulate model on TEST sessions' stimuli
                    - Compute target matrix (UM or CP)
                    - MSE against TEST empirical matrix
                d. Mean across samples × reps = one fold error
            3. Collect fold errors as cv_errors

        Returns:
            Standardised result dict (same schema as grid_search.py).
        """
        from behav_utils.analysis.update_matrix import (
            compute_update_matrix, matrix_error,
        )
        from behav_utils.data.filtering import pool_arrays

        if self._trained_posterior is None:
            raise RuntimeError('No trained posterior. Call train() or load() first.')

        n_sess = len(sessions)
        if n_sess < 2:
            raise ValueError(f'Need ≥2 sessions for CV, got {n_sess}')

        # Split: first half / second half (temporal order preserved)
        mid = n_sess // 2
        folds = [
            (sessions[:mid], sessions[mid:]),   # train=first, test=second
            (sessions[mid:], sessions[:mid]),   # train=second, test=first
        ]

        n_bins = 8
        cv_errors = []

        for train_sessions, test_sessions in folds:
            # Condition on train fold
            cond = self.condition(train_sessions, n_samples=n_posterior_samples)
            samples = cond['posterior_samples']

            # Compute empirical target on test fold
            test_pooled = pool_arrays(test_sessions)
            test_stim = test_pooled['stimuli']
            test_ch = test_pooled['choices']
            test_cat = test_pooled['categories']
            test_valid = ~test_pooled.get(
                'no_response', np.isnan(test_ch))

            emp_um, emp_cm, _ = compute_update_matrix(
                test_stim, test_ch, test_cat,
                n_bins=n_bins, trial_filter='post_correct',
            )
            emp_target = emp_um if fit_target == 'update_matrix' else emp_cm

            # Evaluate posterior samples on test fold
            fold_errors = []
            for s_idx in range(min(n_posterior_samples, len(samples))):
                sample_params = {
                    pn: float(samples[s_idx, i])
                    for i, pn in enumerate(self.param_names)
                }

                for rep in range(n_stochastic_reps):
                    seed = s_idx * 1000 + rep
                    sim_choices = simulate_choices_from_params(
                        self.model_type, sample_params,
                        test_stim[test_valid], test_cat[test_valid],
                        burn_in=self.burn_in, seed=seed,
                    )

                    sim_um, sim_cm, _ = compute_update_matrix(
                        test_stim[test_valid], sim_choices,
                        test_cat[test_valid],
                        n_bins=n_bins, trial_filter='post_correct',
                    )
                    sim_target = (sim_um if fit_target == 'update_matrix'
                                  else sim_cm)
                    fold_errors.append(
                        float(matrix_error(emp_target, sim_target)))

            cv_errors.append(float(np.mean(fold_errors)))

        # Full posterior (condition on ALL sessions — for best_params)
        full_cond = self.condition(sessions, n_samples=n_posterior_samples)

        # Count trials
        all_pooled = pool_arrays(sessions)
        no_resp = all_pooled.get('no_response', np.isnan(all_pooled['choices']))
        n_trials = int((~no_resp).sum())

        return {
            'method': 'sbi_static',
            'cv_type': 'held_out',
            'model_type': self.model_type.upper(),
            'fit_target': fit_target,
            'animal_id': animal_id,
            'distribution': distribution,

            'cv_errors': cv_errors,
            'mean_error': float(np.mean(cv_errors)),
            'std_error': float(np.std(cv_errors)),

            'best_params': full_cond['point_estimate'],
            'posterior_samples': full_cond['posterior_samples'],
            'param_names': list(self.param_names),
            'trajectories': None,
            'link_type': None,

            'all_results': None,

            'n_sessions': n_sess,
            'n_trials': n_trials,
            'metadata': None,  # Filled by caller
        }
