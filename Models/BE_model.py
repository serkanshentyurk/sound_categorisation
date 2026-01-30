"""
Boundary Estimation Model - High-level Interface

This module provides the BoundaryEstimationModel class, a stateful wrapper
around the stateless BEModel core. It maintains backward compatibility with
existing analysis code while delegating to the new functional core.

For inference (MCMC, SBI), use the core directly:
    from Models.BE_core import BEParams, BEState, BEModel

For interactive use, analysis, and plotting:
    from Models.BE_model import BoundaryEstimationModel

Usage:
    # Simulation
    model = BoundaryEstimationModel(sigma_percep=0.15, A_repulsion=0.1,
                                    eta_learning=0.35, eta_relax=0.12)
    model.reset_belief(burn_in=1000)
    choices, p_B = model.simulate_session(stimuli, categories)
    
    # Fitting
    model, results = BoundaryEstimationModel.fit(stimuli, categories, observed_choices)
    
    # Access core components
    params = model.params  # BEParams
    state = model.state    # BEState
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
import warnings

from scipy.optimize import minimize

from Models.BE_core import BEParams, BEState, BEModel
from Helpers.psychometry import fit_psychometric
from Helpers.utils import generate_stimuli


# =============================================================================
# BOUNDARY ESTIMATION MODEL CLASS
# =============================================================================

class BoundaryEstimationModel:
    """
    Boundary Estimation (BE) model for sound categorisation.
    
    This is a stateful wrapper around the functional BEModel core.
    It maintains parameters and state internally for convenient interactive use.
    
    Parameters:
        sigma_percep: Perceptual noise standard deviation
        A_repulsion: Serial dependence strength (repulsion from previous trial)
        eta_learning: Learning rate for boundary belief updates
        eta_relax: Relaxation rate toward uniform distribution
    
    Attributes:
        params: BEParams object (immutable parameters)
        state: BEState object (mutable belief state)
    
    Usage:
        # Simulation
        model = BoundaryEstimationModel(sigma_percep=0.15, A_repulsion=0.1,
                                        eta_learning=0.35, eta_relax=0.12)
        choices, p_B = model.simulate_session(stimuli, categories)
        
        # Fitting
        model, results = BoundaryEstimationModel.fit(stimuli, categories, observed_choices)
        
        # Diagnostics
        comparison = model.compare_serial_dependence(stimuli, categories, 
                                                      observed_choices, rewards, ...)
    """
    
    # =========================================================================
    # INITIALISATION
    # =========================================================================
    
    def __init__(self, sigma_percep: float, A_repulsion: float,
                 eta_learning: float, eta_relax: float,
                 x_min: float = -1, x_max: float = 1, n_points: int = 500):
        """
        Initialise model with parameters.
        
        Args:
            sigma_percep: Perceptual noise standard deviation
            A_repulsion: Strength of serial dependence (repulsion)
            eta_learning: Learning rate for boundary belief updates
            eta_relax: Relaxation rate toward uniform distribution
            x_min, x_max: Stimulus space bounds
            n_points: Discretisation resolution for belief distribution
        """
        # Store as BEParams
        self._params = BEParams(
            sigma_percep=sigma_percep,
            A_repulsion=A_repulsion,
            eta_learning=eta_learning,
            eta_relax=eta_relax
        )
        
        # Grid settings
        self._x_min = x_min
        self._x_max = x_max
        self._n_points = n_points
        
        # Initialise state
        self._state = BEState.initial_uniform(x_min, x_max, n_points)
        
        # Storage for plotting
        self._history = {}
    
    # =========================================================================
    # PROPERTIES - Access to core components
    # =========================================================================
    
    @property
    def params(self) -> BEParams:
        """Access underlying BEParams object."""
        return self._params
    
    @property
    def state(self) -> BEState:
        """Access underlying BEState object."""
        return self._state
    
    @property
    def sigma_percep(self) -> float:
        return self._params.sigma_percep
    
    @property
    def A_repulsion(self) -> float:
        return self._params.A_repulsion
    
    @property
    def eta_learning(self) -> float:
        return self._params.eta_learning
    
    @property
    def eta_relax(self) -> float:
        return self._params.eta_relax
    
    @property
    def sigma_boundary(self) -> float:
        return self._params.sigma_boundary
    
    @property
    def x(self) -> np.ndarray:
        return self._state.x
    
    @property
    def x_min(self) -> float:
        return self._x_min
    
    @property
    def x_max(self) -> float:
        return self._x_max
    
    @property
    def n_points(self) -> int:
        return self._n_points
    
    @property
    def boundary_belief(self) -> np.ndarray:
        return self._state.boundary_belief
    
    @property
    def s_hat_prev(self) -> Optional[float]:
        return self._state.s_hat_prev
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def reset_belief(self, belief: Optional[np.ndarray] = None,
                     burn_in: int = 0, burn_in_seed: int = 42):
        """
        Reset boundary belief to uniform, custom distribution, or via burn-in simulation.
        
        Args:
            belief: Custom initial belief distribution (must match self.x length)
                    If provided, burn_in is ignored.
            burn_in: Number of simulated expert trials to run before actual session.
                     This allows the belief to converge to a sensible boundary estimate.
                     If 0, starts with uniform belief.
            burn_in_seed: Random seed for burn-in simulation
        
        Note:
            Burn-in simulates trials with uniform stimulus distribution and uses
            the model's current parameters. This mimics what an expert animal
            would have experienced before the session starts.
        """
        if belief is not None:
            self._state = BEState.from_belief(
                belief, self._x_min, self._x_max, s_hat_prev=None
            )
        elif burn_in > 0:
            initial = BEState.initial_uniform(self._x_min, self._x_max, self._n_points)
            self._state = BEModel.run_burn_in(
                self._params, initial, burn_in, burn_in_seed
            )
        else:
            self._state = BEState.initial_uniform(
                self._x_min, self._x_max, self._n_points
            )
    
    def get_belief_copy(self) -> np.ndarray:
        """Return copy of current boundary belief distribution."""
        return self._state.boundary_belief.copy()
    
    def set_belief(self, belief: np.ndarray):
        """
        Set boundary belief distribution (for carrying across sessions).
        
        Args:
            belief: New belief distribution (will be normalised)
        """
        self._state = BEState.from_belief(
            belief, self._x_min, self._x_max, self._state.s_hat_prev
        )
    
    def get_params(self) -> Dict[str, float]:
        """Return current parameters as dict."""
        return self._params.to_dict()
    
    def get_state(self) -> Dict:
        """Return complete model state (for checkpointing)."""
        return {
            'params': self._params.to_dict(),
            'boundary_belief': self._state.boundary_belief.copy(),
            's_hat_prev': self._state.s_hat_prev,
            'x': self._state.x.copy(),
            'x_min': self._x_min,
            'x_max': self._x_max,
            'n_points': self._n_points
        }
    
    def _get_belief_stats(self) -> Tuple[float, float]:
        """Compute mean and std of current boundary belief distribution."""
        return self._state.get_belief_stats()
    
    # =========================================================================
    # SIMULATION
    # =========================================================================
    
    def simulate_session(self, stimuli: np.ndarray, categories: np.ndarray,
                         no_response: Optional[np.ndarray] = None,
                         rng: Optional[np.random.Generator] = None,
                         store_history: bool = False
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate choices for a session.
        
        Model generates its own choices and updates based on true category.
        Used for: parameter recovery, model comparison, SBI training.
        
        Args:
            stimuli: Array of stimulus values
            categories: Array of true categories (for feedback)
            no_response: Boolean array (True = skip trial)
            rng: Numpy random generator
            store_history: If True, store belief history for plotting
        
        Returns:
            choices: Simulated choice sequence (0 = A, 1 = B, NaN = no response)
            p_B_sequence: P(choose B) at each trial
        
        Note:
            If store_history=True, also populates self._history for plot_session()
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if store_history:
            choices, p_B, belief_mu, belief_std, self._state = \
                BEModel.simulate_session_with_history(
                    self._params, self._state, stimuli, categories, rng, no_response
                )
            
            self._history = {
                'stimuli': stimuli.copy(),
                'categories': categories.copy(),
                'choices': choices.copy(),
                'p_B': p_B.copy(),
                'belief_mu': belief_mu,
                'belief_std': belief_std
            }
        else:
            choices, p_B, self._state = BEModel.simulate_session(
                self._params, self._state, stimuli, categories, rng, no_response
            )
        
        return choices, p_B
    
    # =========================================================================
    # LIKELIHOOD COMPUTATION
    # =========================================================================
    
    def compute_log_likelihood(self, stimuli: np.ndarray, categories: np.ndarray,
                               observed_choices: np.ndarray,
                               eval_mask: Optional[np.ndarray] = None,
                               no_response: Optional[np.ndarray] = None,
                               seed: int = 42) -> Tuple[float, np.ndarray, int]:
        """
        Compute log-likelihood of observed choices under the model.
        
        Processes ALL trials in order (to maintain correct belief state),
        but only accumulates likelihood for trials where eval_mask=True.
        
        Args:
            stimuli: Array of stimulus values
            categories: Array of true categories
            observed_choices: Animal's actual choices (0 = A, 1 = B)
            eval_mask: Boolean array, True = include this trial in LL.
                       If None, include all valid trials.
            no_response: Boolean array (True = no response). If None, inferred from NaN choices.
            seed: Fixed seed for reproducibility of perceptual noise
        
        Returns:
            total_log_lik: Sum of log-likelihoods for evaluated trials
            trial_log_liks: Array of per-trial log-likelihoods
            n_eval: Number of evaluated trials
        """
        rng = np.random.default_rng(seed)
        
        total_ll, trial_lls, self._state = BEModel.compute_log_likelihood(
            self._params, self._state, stimuli, categories, observed_choices,
            rng, eval_mask, no_response
        )
        
        n_eval = int(np.sum(~np.isnan(trial_lls)))
        return total_ll, trial_lls, n_eval
    
    # =========================================================================
    # CLASS METHODS: PARAMETER INFO
    # =========================================================================
    
    @classmethod
    def get_bounds(cls) -> Dict[str, Tuple[float, float]]:
        """Parameter bounds for fitting."""
        return BEParams.get_bounds()
    
    @classmethod
    def get_param_names(cls) -> List[str]:
        """Parameter names in canonical order."""
        return BEParams.get_param_names()
    
    # =========================================================================
    # FITTING: MAIN ENTRY POINT
    # =========================================================================
    
    @classmethod
    def fit(cls, stimuli: np.ndarray, categories: np.ndarray,
            observed_choices: np.ndarray,
            no_response: Optional[np.ndarray] = None,
            fixed_params: Optional[Dict[str, float]] = None,
            initial_belief: Optional[np.ndarray] = None,
            burn_in: int = 0,
            burn_in_seed: int = 42,
            # --- Validation ---
            validation: Optional[str] = None,
            validation_config: Optional[Dict] = None,
            # --- Optimisation settings ---
            method: str = 'L-BFGS-B',
            n_restarts: int = 5,
            seed: int = 42
            ) -> Tuple['BoundaryEstimationModel', Dict]:
        """
        Fit model to observed data using MLE.
        
        Args:
            stimuli: Array of stimulus values
            categories: Array of true categories (0 or 1)
            observed_choices: Array of animal's choices (0 or 1)
            no_response: Boolean array (True = skip trial). If None, inferred from NaN.
            fixed_params: Dict of parameters to fix, e.g., {'sigma_percep': 0.15}
            initial_belief: Initial boundary belief. If None, uses uniform or burn-in.
            burn_in: Number of burn-in trials to simulate before fitting.
                     Ignored if initial_belief is provided.
            burn_in_seed: Random seed for burn-in simulation
            
            validation: None, 'holdout', or 'cv'
                - None: Fit to all data
                - 'holdout': Temporal split (last X% held out)
                - 'cv': Cross-validation with random blocks
            
            validation_config: Validation-specific settings
                - 'holdout': {'test_fraction': 0.3}
                - 'cv': {'block_size': 50, 'n_folds': 2, 'n_repetitions': 4}
            
            method: Optimisation method for scipy.optimize.minimize
            n_restarts: Number of random restarts
            seed: Random seed for optimisation
        
        Returns:
            best_model: Fitted model instance (belief reset to initial state)
            results: Dict with fitting results
        """
        if validation not in [None, 'holdout', 'cv']:
            raise ValueError(f"validation must be None, 'holdout', or 'cv', got '{validation}'")
        
        # Set default configs
        if validation_config is None:
            if validation == 'holdout':
                validation_config = {'test_fraction': 0.3}
            elif validation == 'cv':
                validation_config = {
                    'block_size': 50,
                    'n_folds': 2,
                    'n_repetitions': 4
                }
        
        # Dispatch to appropriate method
        if validation is None:
            return cls._fit_mle_no_validation(
                stimuli, categories, observed_choices, no_response,
                fixed_params, initial_belief, burn_in, burn_in_seed,
                method, n_restarts, seed
            )
        elif validation == 'holdout':
            return cls._fit_mle_holdout(
                stimuli, categories, observed_choices, no_response,
                fixed_params, initial_belief, burn_in, burn_in_seed,
                validation_config, method, n_restarts, seed
            )
        elif validation == 'cv':
            return cls._fit_mle_cv(
                stimuli, categories, observed_choices, no_response,
                fixed_params, initial_belief, burn_in, burn_in_seed,
                validation_config, method, n_restarts, seed
            )
    
    # =========================================================================
    # FITTING: MLE IMPLEMENTATIONS
    # =========================================================================
    
    @classmethod
    def _fit_mle_no_validation(cls, stimuli: np.ndarray, categories: np.ndarray,
                                observed_choices: np.ndarray,
                                no_response: Optional[np.ndarray],
                                fixed_params: Optional[Dict[str, float]],
                                initial_belief: Optional[np.ndarray],
                                burn_in: int, burn_in_seed: int,
                                method: str, n_restarts: int, seed: int
                                ) -> Tuple['BoundaryEstimationModel', Dict]:
        """MLE fitting without validation (fit on all data)."""
        
        all_bounds = cls.get_bounds()
        all_param_names = cls.get_param_names()
        
        if fixed_params is None:
            fixed_params = {}
        
        if no_response is None:
            no_response = np.isnan(observed_choices)
        
        # Separate free and fixed parameters
        free_param_names = [p for p in all_param_names if p not in fixed_params]
        free_bounds = [all_bounds[p] for p in free_param_names]
        
        rng = np.random.default_rng(seed)
        
        def neg_log_likelihood(free_param_values):
            # Reconstruct full param dict
            params_dict = fixed_params.copy()
            for name, val in zip(free_param_names, free_param_values):
                params_dict[name] = val
            
            # Create params and initial state
            params = BEParams.from_dict(params_dict)
            
            if initial_belief is not None:
                initial_state = BEState.from_belief(initial_belief)
            else:
                initial_state = BEModel.create_initial_state(
                    burn_in=burn_in, params=params, seed=burn_in_seed
                )
            
            # Compute likelihood
            ll_rng = np.random.default_rng(42)  # Fixed for deterministic LL
            ll, _, _ = BEModel.compute_log_likelihood(
                params, initial_state, stimuli, categories, observed_choices,
                ll_rng, no_response=no_response
            )
            
            return -ll
        
        # Multiple restarts
        best_nll = np.inf
        best_free_params = None
        
        for i in range(n_restarts):
            x0 = [rng.uniform(b[0], b[1]) for b in free_bounds]
            
            try:
                result = minimize(
                    neg_log_likelihood,
                    x0=x0,
                    method=method,
                    bounds=free_bounds,
                    options={'maxiter': 1000}
                )
                
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_free_params = result.x
            
            except Exception:
                continue
        
        if best_free_params is None:
            raise RuntimeError("All optimisation attempts failed")
        
        # Reconstruct full params
        best_params = fixed_params.copy()
        for name, val in zip(free_param_names, best_free_params):
            best_params[name] = val
        
        # Create fitted model
        best_model = cls(
            sigma_percep=best_params['sigma_percep'],
            A_repulsion=best_params['A_repulsion'],
            eta_learning=best_params['eta_learning'],
            eta_relax=best_params['eta_relax']
        )
        
        # Set initial belief for model
        if initial_belief is not None:
            best_model.set_belief(initial_belief)
        elif burn_in > 0:
            best_model.reset_belief(burn_in=burn_in, burn_in_seed=burn_in_seed)
        
        # Compute fit statistics
        n_valid = int(np.sum(~no_response))
        n_free_params = len(free_param_names)
        aic = 2 * best_nll + 2 * n_free_params
        bic = 2 * best_nll + n_free_params * np.log(n_valid)
        
        results = {
            'inference': 'mle',
            'validation': None,
            'params': best_params,
            'fixed_params': fixed_params,
            'free_params': dict(zip(free_param_names, best_free_params)),
            'train_nll': best_nll,
            'train_nll_per_trial': best_nll / n_valid,
            'n_train': n_valid,
            'n_free_params': n_free_params,
            'aic': aic,
            'bic': bic,
            'burn_in': burn_in
        }
        
        return best_model, results
    
    @classmethod
    def _fit_mle_holdout(cls, stimuli: np.ndarray, categories: np.ndarray,
                          observed_choices: np.ndarray,
                          no_response: Optional[np.ndarray],
                          fixed_params: Optional[Dict[str, float]],
                          initial_belief: Optional[np.ndarray],
                          burn_in: int, burn_in_seed: int,
                          validation_config: Dict,
                          method: str, n_restarts: int, seed: int
                          ) -> Tuple['BoundaryEstimationModel', Dict]:
        """MLE fitting with temporal holdout validation."""
        
        test_fraction = validation_config.get('test_fraction', 0.3)
        
        n_trials = len(stimuli)
        split_idx = int(n_trials * (1 - test_fraction))
        
        if no_response is None:
            no_response = np.isnan(observed_choices)
        
        # Create masks
        train_mask = np.zeros(n_trials, dtype=bool)
        train_mask[:split_idx] = True
        train_mask = train_mask & ~no_response
        
        test_mask = np.zeros(n_trials, dtype=bool)
        test_mask[split_idx:] = True
        test_mask = test_mask & ~no_response
        
        all_bounds = cls.get_bounds()
        all_param_names = cls.get_param_names()
        
        if fixed_params is None:
            fixed_params = {}
        
        free_param_names = [p for p in all_param_names if p not in fixed_params]
        free_bounds = [all_bounds[p] for p in free_param_names]
        
        rng = np.random.default_rng(seed)
        
        def neg_log_likelihood_train(free_param_values):
            params_dict = fixed_params.copy()
            for name, val in zip(free_param_names, free_param_values):
                params_dict[name] = val
            
            params = BEParams.from_dict(params_dict)
            
            if initial_belief is not None:
                initial_state = BEState.from_belief(initial_belief)
            else:
                initial_state = BEModel.create_initial_state(
                    burn_in=burn_in, params=params, seed=burn_in_seed
                )
            
            ll_rng = np.random.default_rng(42)
            ll, _, _ = BEModel.compute_log_likelihood(
                params, initial_state, stimuli, categories, observed_choices,
                ll_rng, eval_mask=train_mask, no_response=no_response
            )
            
            return -ll
        
        # Optimise
        best_nll = np.inf
        best_free_params = None
        
        for i in range(n_restarts):
            x0 = [rng.uniform(b[0], b[1]) for b in free_bounds]
            
            try:
                result = minimize(
                    neg_log_likelihood_train,
                    x0=x0,
                    method=method,
                    bounds=free_bounds,
                    options={'maxiter': 1000}
                )
                
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_free_params = result.x
            
            except Exception:
                continue
        
        if best_free_params is None:
            raise RuntimeError("All optimisation attempts failed")
        
        # Reconstruct params
        best_params = fixed_params.copy()
        for name, val in zip(free_param_names, best_free_params):
            best_params[name] = val
        
        # Compute test likelihood
        params = BEParams.from_dict(best_params)
        if initial_belief is not None:
            initial_state = BEState.from_belief(initial_belief)
        else:
            initial_state = BEModel.create_initial_state(
                burn_in=burn_in, params=params, seed=burn_in_seed
            )
        
        ll_rng = np.random.default_rng(42)
        test_ll, _, _ = BEModel.compute_log_likelihood(
            params, initial_state, stimuli, categories, observed_choices,
            ll_rng, eval_mask=test_mask, no_response=no_response
        )
        test_nll = -test_ll
        
        # Create model
        best_model = cls(**best_params)
        if initial_belief is not None:
            best_model.set_belief(initial_belief)
        elif burn_in > 0:
            best_model.reset_belief(burn_in=burn_in, burn_in_seed=burn_in_seed)
        
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        n_free_params = len(free_param_names)
        
        results = {
            'inference': 'mle',
            'validation': 'holdout',
            'params': best_params,
            'fixed_params': fixed_params,
            'free_params': dict(zip(free_param_names, best_free_params)),
            'train_nll': best_nll,
            'train_nll_per_trial': best_nll / n_train if n_train > 0 else np.nan,
            'test_nll': test_nll,
            'test_nll_per_trial': test_nll / n_test if n_test > 0 else np.nan,
            'n_train': n_train,
            'n_test': n_test,
            'n_free_params': n_free_params,
            'aic': 2 * best_nll + 2 * n_free_params,
            'bic': 2 * best_nll + n_free_params * np.log(n_train),
            'burn_in': burn_in,
            'test_fraction': test_fraction
        }
        
        return best_model, results
    
    @classmethod
    def _fit_mle_cv(cls, stimuli: np.ndarray, categories: np.ndarray,
                     observed_choices: np.ndarray,
                     no_response: Optional[np.ndarray],
                     fixed_params: Optional[Dict[str, float]],
                     initial_belief: Optional[np.ndarray],
                     burn_in: int, burn_in_seed: int,
                     validation_config: Dict,
                     method: str, n_restarts: int, seed: int
                     ) -> Tuple['BoundaryEstimationModel', Dict]:
        """MLE fitting with cross-validation."""
        
        block_size = validation_config.get('block_size', 50)
        n_folds = validation_config.get('n_folds', 2)
        n_repetitions = validation_config.get('n_repetitions', 4)
        
        n_trials = len(stimuli)
        
        if no_response is None:
            no_response = np.isnan(observed_choices)
        
        all_bounds = cls.get_bounds()
        all_param_names = cls.get_param_names()
        
        if fixed_params is None:
            fixed_params = {}
        
        free_param_names = [p for p in all_param_names if p not in fixed_params]
        free_bounds = [all_bounds[p] for p in free_param_names]
        
        rng = np.random.default_rng(seed)
        
        # Create block indices
        n_blocks = n_trials // block_size
        block_indices = np.arange(n_blocks)
        
        cv_results = []
        
        for rep in range(n_repetitions):
            rng.shuffle(block_indices)
            fold_size = n_blocks // n_folds
            
            for fold in range(n_folds):
                # Create train/test masks
                test_blocks = block_indices[fold * fold_size:(fold + 1) * fold_size]
                
                test_mask = np.zeros(n_trials, dtype=bool)
                for b in test_blocks:
                    test_mask[b * block_size:(b + 1) * block_size] = True
                test_mask = test_mask & ~no_response
                
                train_mask = ~test_mask & ~no_response
                
                # Fit on train
                def neg_ll_train(free_param_values):
                    params_dict = fixed_params.copy()
                    for name, val in zip(free_param_names, free_param_values):
                        params_dict[name] = val
                    
                    params = BEParams.from_dict(params_dict)
                    if initial_belief is not None:
                        initial_state = BEState.from_belief(initial_belief)
                    else:
                        initial_state = BEModel.create_initial_state(
                            burn_in=burn_in, params=params, seed=burn_in_seed
                        )
                    
                    ll_rng = np.random.default_rng(42)
                    ll, _, _ = BEModel.compute_log_likelihood(
                        params, initial_state, stimuli, categories, observed_choices,
                        ll_rng, eval_mask=train_mask, no_response=no_response
                    )
                    return -ll
                
                best_nll = np.inf
                best_free = None
                
                for restart in range(n_restarts):
                    x0 = [rng.uniform(b[0], b[1]) for b in free_bounds]
                    try:
                        result = minimize(neg_ll_train, x0=x0, method=method,
                                         bounds=free_bounds, options={'maxiter': 1000})
                        if result.fun < best_nll:
                            best_nll = result.fun
                            best_free = result.x
                    except Exception:
                        continue
                
                if best_free is not None:
                    # Compute test LL
                    params_dict = fixed_params.copy()
                    for name, val in zip(free_param_names, best_free):
                        params_dict[name] = val
                    
                    params = BEParams.from_dict(params_dict)
                    if initial_belief is not None:
                        initial_state = BEState.from_belief(initial_belief)
                    else:
                        initial_state = BEModel.create_initial_state(
                            burn_in=burn_in, params=params, seed=burn_in_seed
                        )
                    
                    ll_rng = np.random.default_rng(42)
                    test_ll, _, _ = BEModel.compute_log_likelihood(
                        params, initial_state, stimuli, categories, observed_choices,
                        ll_rng, eval_mask=test_mask, no_response=no_response
                    )
                    
                    cv_results.append({
                        'rep': rep,
                        'fold': fold,
                        'train_nll': best_nll,
                        'test_nll': -test_ll,
                        'n_train': int(train_mask.sum()),
                        'n_test': int(test_mask.sum()),
                        'params': params_dict.copy()
                    })
        
        # Fit final model on all data
        best_model, full_results = cls._fit_mle_no_validation(
            stimuli, categories, observed_choices, no_response,
            fixed_params, initial_belief, burn_in, burn_in_seed,
            method, n_restarts, seed
        )
        
        # Aggregate CV results
        cv_df = pd.DataFrame(cv_results)
        
        results = full_results.copy()
        results['validation'] = 'cv'
        results['cv_results'] = cv_df
        results['cv_test_nll_mean'] = cv_df['test_nll'].mean()
        results['cv_test_nll_std'] = cv_df['test_nll'].std()
        results['cv_test_nll_per_trial_mean'] = (cv_df['test_nll'] / cv_df['n_test']).mean()
        results['cv_test_nll_per_trial_std'] = (cv_df['test_nll'] / cv_df['n_test']).std()
        
        return best_model, results
    
    # =========================================================================
    # PARAMETER RECOVERY
    # =========================================================================
    
    @classmethod
    def parameter_recovery_test(cls, n_tests: int = 20,
                                 n_trials: int = 300,
                                 fixed_params: Optional[Dict[str, float]] = None,
                                 burn_in: int = 0,
                                 burn_in_seed: int = 42,
                                 validation: Optional[str] = 'holdout',
                                 validation_config: Optional[Dict] = None,
                                 seed: int = 42,
                                 verbose: bool = True
                                 ) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Test parameter recovery by simulating and fitting.
        
        Args:
            n_tests: Number of recovery tests
            n_trials: Trials per simulated session
            fixed_params: Parameters to fix during fitting
            burn_in: Burn-in trials
            burn_in_seed: Burn-in seed
            validation: Validation method
            validation_config: Validation config
            seed: Random seed
            verbose: Print progress
        
        Returns:
            true_params: {param_name: array of true values}
            recovered_params: {param_name: array of recovered values}
            correlations: {param_name: correlation}
            diagnostics: {train_nlls, test_nlls}
        """
        all_bounds = cls.get_bounds()
        all_param_names = cls.get_param_names()
        
        if fixed_params is None:
            fixed_params = {}
        
        free_param_names = [p for p in all_param_names if p not in fixed_params]
        
        rng = np.random.default_rng(seed)
        
        true_params = {name: [] for name in free_param_names}
        recovered_params = {name: [] for name in free_param_names}
        diagnostics = {'train_nlls': [], 'test_nlls': []}
        
        for i in range(n_tests):
            if verbose:
                print(f"Recovery test {i+1}/{n_tests}...", end=' ')
            
            # Sample true free parameters
            true_free = {}
            for name in free_param_names:
                true_free[name] = rng.uniform(*all_bounds[name])
            
            # Combine with fixed
            all_true = {**fixed_params, **true_free}
            
            # Generate stimuli
            stimuli, categories, _ = generate_stimuli(
                n_trials=n_trials, seed=seed + i * 1000
            )
            
            # Create and simulate
            true_model = cls(**all_true)
            if burn_in > 0:
                true_model.reset_belief(burn_in=burn_in, burn_in_seed=burn_in_seed + i)
            
            sim_rng = np.random.default_rng(seed + i * 1000 + 1)
            choices_true, _ = true_model.simulate_session(
                stimuli, categories, rng=sim_rng
            )
            
            # Fit
            try:
                _, results = cls.fit(
                    stimuli, categories, choices_true,
                    fixed_params=fixed_params,
                    burn_in=burn_in,
                    burn_in_seed=burn_in_seed + i,
                    validation=validation,
                    validation_config=validation_config,
                    n_restarts=3,
                    seed=seed + i
                )
                
                for name in free_param_names:
                    true_params[name].append(true_free[name])
                    recovered_params[name].append(results['params'][name])
                
                diagnostics['train_nlls'].append(results['train_nll'])
                if 'test_nll' in results:
                    diagnostics['test_nlls'].append(results['test_nll'])
                
                if verbose:
                    print("OK")
            
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")
        
        # Convert to arrays
        for name in free_param_names:
            true_params[name] = np.array(true_params[name])
            recovered_params[name] = np.array(recovered_params[name])
        
        for key in ['train_nlls', 'test_nlls']:
            if diagnostics[key]:
                diagnostics[key] = np.array(diagnostics[key])
        
        # Compute correlations
        correlations = {}
        for name in free_param_names:
            if len(true_params[name]) >= 2:
                correlations[name] = np.corrcoef(
                    true_params[name], recovered_params[name]
                )[0, 1]
            else:
                correlations[name] = np.nan
        
        return true_params, recovered_params, correlations, diagnostics
    
    # =========================================================================
    # PLOTTING
    # =========================================================================
    
    def plot_session(self, 
                     stimuli: Optional[np.ndarray] = None,
                     choices: Optional[np.ndarray] = None,
                     categories: Optional[np.ndarray] = None,
                     **kwargs):
        """
        Plot trial-by-trial session visualisation.
        
        If called after simulate_session(..., store_history=True), can be called
        without arguments to use stored data.
        
        Args:
            stimuli: Stimulus values (optional if store_history was used)
            choices: Choice values (optional if store_history was used)
            categories: Category values (optional if store_history was used)
            **kwargs: Additional arguments passed to plot_session function
        
        Returns:
            Matplotlib Figure
        """
        from Plotting.session import plot_session as _plot_session
        
        if stimuli is None:
            if not self._history:
                raise ValueError(
                    "No data available. Run simulate_session(..., store_history=True) "
                    "first, or provide stimuli, choices, categories explicitly."
                )
            stimuli = self._history['stimuli']
            choices = self._history['choices']
            categories = self._history['categories']
            p_B = self._history['p_B']
            belief_mu = self._history.get('belief_mu')
            belief_std = self._history.get('belief_std')
        else:
            if choices is None or categories is None:
                raise ValueError("Must provide stimuli, choices, and categories together")
            p_B = kwargs.pop('p_B', None)
            belief_mu = kwargs.pop('belief_mu', None)
            belief_std = kwargs.pop('belief_std', None)
        
        return _plot_session(
            stimuli=stimuli,
            choices=choices,
            categories=categories,
            p_B=p_B,
            belief_mu=belief_mu,
            belief_std=belief_std,
            **kwargs
        )
    
    def plot_belief_evolution(self, stimuli: np.ndarray, categories: np.ndarray,
                               no_response: Optional[np.ndarray] = None,
                               n_snapshots: int = 5,
                               seed: int = 42,
                               figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
        """
        Plot how boundary belief evolves during a session.
        
        Args:
            stimuli: Stimulus values
            categories: True categories
            no_response: No-response mask
            n_snapshots: Number of belief snapshots to show
            seed: Random seed
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if no_response is None:
            no_response = np.zeros(len(stimuli), dtype=bool)
        
        # Store initial state
        initial_state = self._state.copy()
        
        rng = np.random.default_rng(seed)
        n_trials = len(stimuli)
        
        # Determine snapshot points
        snapshot_trials = np.linspace(0, n_trials - 1, n_snapshots + 1).astype(int)
        
        # Collect beliefs
        beliefs = [self._state.boundary_belief.copy()]
        trial_indices = [0]
        
        for t in range(n_trials):
            if no_response[t]:
                continue
            
            s_hat = BEModel.perceive_stimulus(
                stimuli[t], self._params, self._state.s_hat_prev, rng
            )
            self._state = BEModel.update_belief(s_hat, categories[t], self._params, self._state)
            
            if t + 1 in snapshot_trials:
                beliefs.append(self._state.boundary_belief.copy())
                trial_indices.append(t + 1)
        
        # Restore state
        self._state = initial_state
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(beliefs)))
        
        for i, (belief, trial_idx) in enumerate(zip(beliefs, trial_indices)):
            ax.plot(self._state.x, belief, color=colors[i], 
                   label=f'Trial {trial_idx}', alpha=0.8)
        
        ax.axvline(0, color='k', linestyle='--', alpha=0.3, label='True boundary')
        ax.set_xlabel('Stimulus space')
        ax.set_ylabel('Belief density')
        ax.set_title('Boundary belief evolution')
        ax.legend(loc='upper right')
        ax.set_xlim(self._x_min, self._x_max)
        
        plt.tight_layout()
        return fig
    
    @classmethod
    def plot_recovery(cls, true_params: Dict[str, np.ndarray],
                      recovered_params: Dict[str, np.ndarray],
                      correlations: Dict[str, float],
                      figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """Plot parameter recovery results."""
        param_names = list(true_params.keys())
        n_params = len(param_names)
        
        if figsize is None:
            cols = min(n_params, 2)
            rows = (n_params + cols - 1) // cols
            figsize = (5 * cols, 5 * rows)
        else:
            cols = min(n_params, 2)
            rows = (n_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        bounds = cls.get_bounds()
        
        for i, name in enumerate(param_names):
            ax = axes[i]
            
            true_vals = true_params[name]
            rec_vals = recovered_params[name]
            
            ax.scatter(true_vals, rec_vals, alpha=0.6)
            
            if name in bounds:
                lims = list(bounds[name])
            else:
                lims = [min(true_vals.min(), rec_vals.min()),
                        max(true_vals.max(), rec_vals.max())]
            ax.plot(lims, lims, 'k--', alpha=0.5)
            
            r = correlations.get(name, np.nan)
            ax.set_title(f'{name}\nr = {r:.3f}')
            ax.set_xlabel('True')
            ax.set_ylabel('Recovered')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')
        
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BoundaryEstimationModel',
]
