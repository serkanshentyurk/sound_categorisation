"""
Amortised SBI for Static Model Comparison

Train a neural density estimator once on curriculum-matched generic data,
then condition on each animal's pooled summary statistics. Produces true
held-out CV errors matching the GS-CV protocol from the manuscript:

    For each of N repeats:
        Split sessions into 2 folds (first half / second half)
        For each fold pair (train, test):
            Pool train fold trials → compute summary stats → condition posterior
            Simulate with posterior median on test fold stimuli → UM MSE
        Average across folds → one error per repeat
    N errors → compare with GS errors via Wilcoxon

Summary stats are always computed on POOLED trials (not per-session),
giving a fixed-length vector (~20 dims) regardless of session count.

Usage:
    from inference.amortised import AmortisedSBI

    trainer = AmortisedSBI('be', curriculum=[('uniform', 15)])
    trainer.train(50_000)
    trainer.save('results/snpe/uniform_be.pkl')

    loaded = AmortisedSBI.load('results/snpe/uniform_be.pkl')
    result = loaded.fit(sessions, animal_id='SS01')
"""

import pickle
import warnings
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# POOLED STATS (used by both simulator and conditioning)
# =============================================================================

def compute_pooled_stats(
    stimuli: np.ndarray,
    choices: np.ndarray,
    categories: np.ndarray,
    stat_names: List[str] = None,
) -> np.ndarray:
    """
    Compute summary stats on pooled trial arrays.

    Returns a flat numpy array of length n_stats.
    NaN values are left in place (handled by caller).
    """
    from behav_utils.analysis.summary_stats import compute_summary_stats

    if stat_names is None:
        from inference.constants import SBI_STATS
        stat_names = list(SBI_STATS)

    valid = np.isfinite(choices)
    return compute_summary_stats(
        choices=choices[valid],
        stimuli=stimuli[valid],
        categories=categories[valid],
        stat_names=stat_names,
        return_dict=False,
    )


def compute_observed_stats_from_sessions(
    sessions: list,
    stat_names: List[str] = None,
) -> np.ndarray:
    """Pool all sessions' trials, compute stats on the pool."""
    from behav_utils.data.filtering import pool_arrays

    pooled = pool_arrays(sessions)
    return compute_pooled_stats(
        pooled['stimuli'], pooled['choices'], pooled['categories'],
        stat_names=stat_names,
    )


# =============================================================================
# SIMULATE CHOICES FROM PARAMS
# =============================================================================

def simulate_choices_from_params(
    model_type: str,
    params_dict: dict,
    stimuli: np.ndarray,
    categories: np.ndarray,
    burn_in: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate choices from specific parameter values."""
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
# CURRICULUM SIMULATOR (pools trials, returns fixed-length stats)
# =============================================================================

def build_curriculum_simulator(
    model_type: str,
    curriculum: List[Tuple[str, int]],
    trials_per_session: int = 350,
    burn_in: int = 1000,
    stat_names: List[str] = None,
):
    """
    Build a simulator that generates POOLED summary stats for a curriculum.

    The simulator:
        1. Receives a parameter vector theta
        2. Generates stimuli per curriculum schedule
        3. Simulates across sessions, chaining state
        4. POOLS all trials into one sequence
        5. Computes stats on the pool → fixed-length vector

    Returns:
        (simulator_fn, prior, param_names)
    """
    from behav_utils.analysis.summary_stats import compute_summary_stats

    if stat_names is None:
        from inference.constants import SBI_STATS
        stat_names = list(SBI_STATS)

    dist_schedule = []
    for dist_name, n_sess in curriculum:
        dist_schedule.extend([dist_name] * n_sess)

    model_type = model_type.lower()

    if model_type == 'be':
        from models.BE_core import BEParams, BEModel
        param_names = BEParams.get_param_names()
        bounds = BEParams.get_bounds()

        def _make_params(theta):
            return BEParams(
                sigma_percep=float(theta[0]),
                A_repulsion=float(theta[1]),
                eta_learning=float(theta[2]),
                eta_relax=float(theta[3]),
            )

        def _create_state(params, bi, sd):
            return BEModel.create_initial_state(
                burn_in=bi, params=params, seed=sd)

        def _simulate_session(params, state, stim, cat, rng):
            ch, _, st, _ = BEModel.simulate_session(
                params, state, stim, cat, rng, return_history=False)
            return ch, st

    elif model_type == 'sc':
        from models.SC_core import SCParams, SCModel
        param_names = SCParams.get_param_names()
        bounds = SCParams.get_bounds()

        def _make_params(theta):
            return SCParams(
                sigma_percep=float(theta[0]),
                A_repulsion=float(theta[1]),
                gamma=float(theta[2]),
                sigma_update=float(theta[3]),
            )

        def _create_state(params, bi, sd):
            return SCModel.create_initial_state(
                burn_in=bi, params=params, seed=sd)

        def _simulate_session(params, state, stim, cat, rng):
            ch, _, st, _ = SCModel.simulate_session(
                params, state, stim, cat, rng, return_history=False)
            return ch, st
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # Prior
    if TORCH_AVAILABLE:
        from sbi.utils import BoxUniform
        lower = torch.tensor([bounds[p][0] for p in param_names])
        upper = torch.tensor([bounds[p][1] for p in param_names])
        prior = BoxUniform(low=lower, high=upper)
    else:
        prior = None

    def _sample_stimuli(dist_name, n_trials, rng):
        if dist_name in ('hard_a', 'hard_b'):
            from utils.stimulus_distributions import sample_distribution
            return sample_distribution(n_trials, dist_name, rng=rng)
        else:
            stim = rng.uniform(-1, 1, n_trials)
            cat = (stim > 0).astype(int)
            return stim, cat

    def simulator(theta, seed=None):
        """Simulate curriculum → pool → stats → fixed-length vector."""
        if seed is None:
            seed = np.random.randint(0, 2**31)
        rng = np.random.default_rng(seed)
        theta_np = np.asarray(theta, dtype=float)

        params = _make_params(theta_np)
        state = _create_state(params, burn_in, seed)

        all_stim, all_ch, all_cat = [], [], []
        for dist_name in dist_schedule:
            stim, cat = _sample_stimuli(dist_name, trials_per_session, rng)
            ch, state = _simulate_session(params, state, stim, cat, rng)
            all_stim.append(stim)
            all_ch.append(ch)
            all_cat.append(cat)

        pooled_stim = np.concatenate(all_stim)
        pooled_ch = np.concatenate(all_ch)
        pooled_cat = np.concatenate(all_cat)

        return compute_pooled_stats(
            pooled_stim, pooled_ch, pooled_cat, stat_names)

    return simulator, prior, param_names


# =============================================================================
# AMORTISED SBI CLASS
# =============================================================================

class AmortisedSBI:
    """
    Train once on curriculum-matched generic data, condition on many animals.

    Stats are always computed on POOLED trials → fixed-length vector.
    CV protocol matches GS-CV exactly: 2-fold session-block split,
    64 repeats, Wilcoxon comparison.
    """

    def __init__(
        self,
        model_type: str,
        curriculum: List[Tuple[str, int]],
        trials_per_session: int = 350,
        burn_in: int = 1000,
        stat_names: List[str] = None,
    ):
        self.model_type = model_type.lower()
        self.curriculum = list(curriculum)
        self.trials_per_session = trials_per_session
        self.burn_in = burn_in

        if stat_names is None:
            from inference.constants import SBI_STATS
            stat_names = list(SBI_STATS)
        self.stat_names = stat_names

        self._simulator, self._prior, self.param_names = \
            build_curriculum_simulator(
                model_type=self.model_type,
                curriculum=self.curriculum,
                trials_per_session=self.trials_per_session,
                burn_in=self.burn_in,
                stat_names=self.stat_names,
            )

        self._trained_posterior = None
        self._training_metadata = None
        self._x_train = None
        self._valid_stat_mask = None  # set during train(), applied at conditioning

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

        import torch
        from sbi.inference import SNPE
        from sbi.utils import process_simulator
        import time as _time

        print(f'Training AmortisedSBI [{self.model_type.upper()}] '
              f'({n_simulations:,} sims, curriculum={self.curriculum})')

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Wrap numpy simulator for sbi
        raw_sim = self._simulator
        sim_counter = [0]

        def _sbi_simulator(theta_batch):
            results = []
            for theta in theta_batch:
                sim_counter[0] += 1
                x = raw_sim(theta.numpy(), seed=sim_counter[0])
                results.append(torch.tensor(x, dtype=torch.float32))
            return torch.stack(results)

        prior = self._prior
        simulator_fn = process_simulator(
            _sbi_simulator, prior, is_numpy_simulator=False)

        t0 = _time.time()
        theta = prior.sample((n_simulations,))
        if show_progress:
            print(f'  Simulating {n_simulations:,} datasets...')
        x = simulator_fn(theta)

        # ── Auto-drop bad stat columns ──────────────────────────────────
        # Before filtering NaN rows, check each stat column.
        # Drop columns that are: >30% NaN OR near-zero variance.
        # The mask is stored and applied at conditioning time.
        from behav_utils.analysis.summary_stats import get_stat_names_expanded
        expanded = get_stat_names_expanded(self.stat_names)

        n_cols = x.shape[1]
        col_nan_rate = (~torch.isfinite(x)).float().mean(dim=0)
        # For std, compute on finite values only
        x_for_std = x.clone()
        x_for_std[~torch.isfinite(x_for_std)] = float('nan')
        col_std = torch.tensor([
            float(np.nanstd(x_for_std[:, i].numpy()))
            for i in range(n_cols)
        ])

        NAN_THRESHOLD = 0.30
        STD_THRESHOLD = 1e-10

        keep_mask = torch.ones(n_cols, dtype=torch.bool)
        dropped = []
        for i in range(n_cols):
            name = expanded[i] if i < len(expanded) else f'stat_{i}'
            nan_r = float(col_nan_rate[i])
            std_v = float(col_std[i])

            if nan_r > NAN_THRESHOLD:
                keep_mask[i] = False
                dropped.append((name, f'{nan_r:.0%} NaN'))
            elif std_v < STD_THRESHOLD:
                keep_mask[i] = False
                dropped.append((name, f'constant (std={std_v:.2e})'))

        self._valid_stat_mask = keep_mask.numpy()

        if dropped and show_progress:
            print(f'  ⚠ Auto-dropping {len(dropped)}/{n_cols} stats:')
            for name, reason in dropped:
                print(f'    {name}: {reason}')
            print(f'  Keeping {keep_mask.sum().item()}/{n_cols} stats')

        # Apply column mask
        x = x[:, keep_mask]

        # Filter NaN rows (on remaining columns)
        valid = torch.isfinite(x).all(dim=-1)
        n_valid = valid.sum().item()
        if show_progress:
            print(f'  {n_valid}/{n_simulations} valid sims '
                  f'({100 * n_valid / n_simulations:.0f}%)')
        theta, x = theta[valid], x[valid]

        # Train SNPE
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
        self._x_train = x  # for NaN imputation
        self._training_metadata = {
            'n_simulations': n_simulations,
            'n_valid': n_valid,
            'seed': seed,
            'curriculum': self.curriculum,
            'training_time': dt,
        }
        return posterior

    # ── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path):
        """Save trained posterior + config. Simulator rebuilt on load."""
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
            'x_train': self._x_train,
            'valid_stat_mask': self._valid_stat_mask,
            'training_metadata': self._training_metadata,
            '_version': 3,
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, path) -> 'AmortisedSBI':
        """Load trained AmortisedSBI. Rebuilds simulator/prior fresh."""
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
        obj._x_train = data.get('x_train')
        obj._valid_stat_mask = data.get('valid_stat_mask')
        obj._training_metadata = data.get('training_metadata')
        return obj

    # ── Conditioning ─────────────────────────────────────────────────────────

    def _impute_nan(self, stats: np.ndarray) -> np.ndarray:
        """Replace NaN/Inf with training data median."""
        bad = ~np.isfinite(stats)
        if not bad.any():
            return stats

        n_bad = bad.sum()
        warnings.warn(
            f'{n_bad}/{len(stats)} observed stats are NaN/Inf — '
            f'imputing from training data median')

        stats = stats.copy()
        if self._x_train is not None:
            x_np = self._x_train.numpy() if hasattr(
                self._x_train, 'numpy') else np.asarray(self._x_train)
            medians = np.nanmedian(x_np, axis=0)
            n = min(len(stats), len(medians))
            for i in range(n):
                if bad[i]:
                    stats[i] = medians[i]
            if len(stats) > n:
                stats[n:][bad[n:]] = 0.0
        else:
            stats[bad] = 0.0
        return stats

    def condition_from_arrays(
        self,
        stimuli: np.ndarray,
        choices: np.ndarray,
        categories: np.ndarray,
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Condition posterior on pooled raw arrays.

        This is the low-level entry point. condition() delegates here
        after pooling SessionData objects.

        Applies the same stat column mask that was determined during
        training (auto-drop of NaN-heavy or constant stats).

        Returns dict with posterior_samples, param_names, point_estimate.
        """
        if self._trained_posterior is None:
            raise RuntimeError('No trained posterior.')

        import torch

        # Compute full stats vector
        observed = compute_pooled_stats(
            stimuli, choices, categories, self.stat_names)

        # Apply column mask (same columns the network was trained on)
        if self._valid_stat_mask is not None:
            observed = observed[self._valid_stat_mask]

        # Impute remaining NaN (x_train is already masked, so indices match)
        observed = self._impute_nan(observed)
        obs_tensor = torch.tensor(observed, dtype=torch.float32)

        samples = self._trained_posterior.sample(
            (n_samples,), x=obs_tensor,
            show_progress_bars=False)
        samples_np = samples.numpy()

        point_estimate = {
            pn: float(np.median(samples_np[:, i]))
            for i, pn in enumerate(self.param_names)
        }

        return {
            'posterior_samples': samples_np,
            'param_names': list(self.param_names),
            'point_estimate': point_estimate,
        }

    def condition(
        self,
        sessions: list,
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Condition posterior on pooled observed stats from sessions.

        Convenience wrapper around condition_from_arrays.
        """
        from behav_utils.data.filtering import pool_arrays

        pooled = pool_arrays(sessions)
        return self.condition_from_arrays(
            pooled['stimuli'], pooled['choices'], pooled['categories'],
            n_samples=n_samples,
        )

    # ── Held-out CV (matching GS-CV protocol) ────────────────────────────────

    def _run_cv(
        self,
        session_arrays: List[Dict[str, np.ndarray]],
        animal_id: str,
        distribution: str,
        fit_target: str,
        n_repeats: int,
        n_posterior_samples: int,
        n_stochastic_reps: int,
        n_bins: int,
        seed: int,
    ) -> Dict[str, Any]:
        """
        Core CV logic on a list of session dicts.

        Each dict must have 'stimuli', 'choices', 'categories' keys.
        Both fit() and fit_from_arrays() delegate here.
        """
        from behav_utils.analysis.update_matrix import (
            compute_update_matrix, matrix_error,
        )

        if self._trained_posterior is None:
            raise RuntimeError('No trained posterior.')

        n_sess = len(session_arrays)
        if n_sess < 2:
            raise ValueError(f'Need ≥2 sessions for CV, got {n_sess}')

        def _pool(sess_list):
            stim = np.concatenate([s['stimuli'] for s in sess_list])
            ch = np.concatenate([s['choices'] for s in sess_list])
            cat = np.concatenate([s['categories'] for s in sess_list])
            return stim, ch, cat

        mid = n_sess // 2
        folds = [
            (session_arrays[:mid], session_arrays[mid:]),
            (session_arrays[mid:], session_arrays[:mid]),
        ]

        cv_errors = []

        for rep in range(n_repeats):
            fold_errors = []

            for train_list, test_list in folds:
                # Condition on pooled train fold
                tr_stim, tr_ch, tr_cat = _pool(train_list)
                cond = self.condition_from_arrays(
                    tr_stim, tr_ch, tr_cat,
                    n_samples=n_posterior_samples)
                median_params = cond['point_estimate']

                # Empirical target on pooled test fold
                te_stim, te_ch, te_cat = _pool(test_list)
                te_valid = np.isfinite(te_ch)

                emp_um, emp_cm, _ = compute_update_matrix(
                    te_stim, te_ch, te_cat,
                    n_bins=n_bins, trial_filter='post_correct',
                )
                emp_target = emp_um if fit_target == 'update_matrix' else emp_cm

                # Simulate with posterior median on test stimuli
                rep_errors = []
                for sr in range(n_stochastic_reps):
                    sim_seed = seed + rep * 10000 + sr
                    sim_ch = simulate_choices_from_params(
                        self.model_type, median_params,
                        te_stim[te_valid], te_cat[te_valid],
                        burn_in=self.burn_in, seed=sim_seed,
                    )

                    sim_um, sim_cm, _ = compute_update_matrix(
                        te_stim[te_valid], sim_ch,
                        te_cat[te_valid],
                        n_bins=n_bins, trial_filter='post_correct',
                    )
                    sim_target = (sim_um if fit_target == 'update_matrix'
                                  else sim_cm)
                    rep_errors.append(
                        float(matrix_error(emp_target, sim_target)))

                fold_errors.append(float(np.mean(rep_errors)))

            cv_errors.append(float(np.mean(fold_errors)))

        # Full posterior (condition on ALL data)
        all_stim, all_ch, all_cat = _pool(session_arrays)
        full_cond = self.condition_from_arrays(
            all_stim, all_ch, all_cat, n_samples=n_posterior_samples)

        n_trials = int(np.isfinite(all_ch).sum())

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
            'metadata': None,
        }

    def fit(
        self,
        sessions: list,
        animal_id: str = 'unknown',
        distribution: str = 'uniform',
        fit_target: str = 'update_matrix',
        n_repeats: int = 64,
        n_posterior_samples: int = 50,
        n_stochastic_reps: int = 10,
        n_bins: int = 8,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Full pipeline on SessionData objects: 2-fold held-out CV.

        See _run_cv for protocol details.
        """
        from behav_utils.data.filtering import pool_arrays

        # Convert SessionData → dicts with arrays
        session_arrays = []
        for sess in sessions:
            p = pool_arrays([sess])
            session_arrays.append({
                'stimuli': p['stimuli'],
                'choices': p['choices'],
                'categories': p['categories'],
            })

        return self._run_cv(
            session_arrays, animal_id, distribution, fit_target,
            n_repeats, n_posterior_samples, n_stochastic_reps,
            n_bins, seed,
        )

    def fit_from_arrays(
        self,
        session_arrays: List[Dict[str, np.ndarray]],
        animal_id: str = 'unknown',
        distribution: str = 'uniform',
        fit_target: str = 'update_matrix',
        n_repeats: int = 64,
        n_posterior_samples: int = 50,
        n_stochastic_reps: int = 10,
        n_bins: int = 8,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Full pipeline on raw arrays: 2-fold held-out CV.

        Args:
            session_arrays: List of dicts, each with keys
                'stimuli', 'choices', 'categories' (1D numpy arrays).

        See _run_cv for protocol details.
        """
        return self._run_cv(
            session_arrays, animal_id, distribution, fit_target,
            n_repeats, n_posterior_samples, n_stochastic_reps,
            n_bins, seed,
        )
