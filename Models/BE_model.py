import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
import warnings

from scipy.optimize import minimize
from scipy.stats import norm
from scipy.integrate import trapezoid

from Analysis.update_matrix import compute_update_matrix, matrix_error
from Helpers.psychometry import fit_psychometric
from Helpers.utils import generate_stimuli

# =============================================================================
# BOUNDARY ESTIMATION MODEL CLASS
# =============================================================================

class BoundaryEstimationModel:
    """
    Boundary Estimation (BE) model for sound categorisation.
    
    The agent maintains a belief distribution over the category boundary location
    and updates this belief based on trial-by-trial feedback (true category).
    
    Parameters:
        sigma_percep: Perceptual noise standard deviation
        A_repulsion: Serial dependence strength (repulsion from previous trial)
        mu_learning: Learning rate for boundary belief updates
        mu_relax: Relaxation rate toward uniform distribution
    
    Derived:
        sigma_boundary = 1 / sigma_percep (update precision in sigmoid)
    
    Usage:
        # Simulation
        model = BoundaryEstimationModel(sigma_percep=0.15, A_repulsion=0.1,
                                        mu_learning=0.3, mu_relax=0.1)
        choices, p_B = model.simulate_session(stimuli, categories)
        
        # Fitting
        model, results = BoundaryEstimationModel.fit(stimuli, categories, observed_choices)
        
        # Diagnostics
        comparison = model.compare_serial_dependence(stimuli, categories, 
                                                      observed_choices, rewards, ...)
    """
    
    # =========================================================================
    # CORE MODEL METHODS
    # =========================================================================
    
    def __init__(self, sigma_percep: float, A_repulsion: float,
                 mu_learning: float, mu_relax: float,
                 x_min: float = -1, x_max: float = 1, n_points: int = 500):
        """
        Initialise model with parameters.
        
        Args:
            sigma_percep: Perceptual noise standard deviation
            A_repulsion: Strength of serial dependence (repulsion)
            mu_learning: Learning rate for boundary belief updates
            mu_relax: Relaxation rate toward uniform distribution
            x_min, x_max: Stimulus space bounds
            n_points: Discretisation resolution for belief distribution
        """
        self.sigma_percep = sigma_percep
        self.A_repulsion = A_repulsion
        self.mu_learning = mu_learning
        self.mu_relax = mu_relax
        self.sigma_boundary = 1 / sigma_percep  # Derived parameter
        
        # Stimulus space discretisation
        self.x = np.linspace(x_min, x_max, n_points)
        self.x_min = x_min
        self.x_max = x_max
        self.n_points = n_points
        
        # Initialise state
        self.reset_belief()
        self.s_hat_prev = None
    
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
            if len(belief) != len(self.x):
                raise ValueError(f"Belief length ({len(belief)}) must match "
                                f"discretisation ({len(self.x)})")
            self.boundary_belief = belief.copy()
            # Normalise
            self.boundary_belief = self.boundary_belief / trapezoid(self.boundary_belief, self.x)
        elif burn_in > 0:
            self._run_burn_in(burn_in, burn_in_seed)
        else:
            # Uniform belief
            self.boundary_belief = np.ones_like(self.x) / (self.x_max - self.x_min)
            # Normalise properly
            self.boundary_belief = self.boundary_belief / trapezoid(self.boundary_belief, self.x)
        
        self.s_hat_prev = None
    
    def _run_burn_in(self, n_trials: int, seed: int):
        """
        Run burn-in simulation to establish expert-like belief.
        
        Simulates n_trials with uniform stimulus distribution, updating the
        boundary belief as if the animal had experienced these trials.
        
        Args:
            n_trials: Number of burn-in trials
            seed: Random seed
        """
        rng = np.random.default_rng(seed)
        
        # Start with uniform belief
        self.boundary_belief = np.ones_like(self.x) / (self.x_max - self.x_min)
        self.boundary_belief = self.boundary_belief / trapezoid(self.boundary_belief, self.x)
        self.s_hat_prev = None
        
        # Generate burn-in stimuli (uniform distribution)
        burn_in_stimuli, burn_in_categories, rng = generate_stimuli(x_min = self.x_min,
                                                                   x_max = self.x_max,
                                                                   n_trials = n_trials,
                                                                   seed=42)

        
        # Run through trials (don't need to store choices)
        for t in range(n_trials):
            s_hat = self._perceive_stimulus(burn_in_stimuli[t], rng)
            # Update belief based on true category
            self._update_belief(s_hat, burn_in_categories[t])
    
    def _perceive_stimulus(self, s_t: float, rng: np.random.Generator) -> float:
        """
        Apply perceptual noise and repulsion from previous trial.
        
        Args:
            s_t: True stimulus value
            rng: Random number generator
        
        Returns:
            s_hat: Perceived stimulus value
        """
        # Perceptual noise
        noise = rng.normal(0, self.sigma_percep)
        s_tilde = s_t + noise
        
        # Repulsion from previous trial
        if self.s_hat_prev is not None:
            diff = s_tilde - self.s_hat_prev
            repulsion = self.A_repulsion * diff * np.exp(-np.abs(diff))
            s_hat = s_tilde + repulsion
        else:
            s_hat = s_tilde
        
        return s_hat
    
    def _find_closest_idx(self, s_hat: float) -> int:
        """Find index of closest element in x to s_hat."""
        return int(np.abs(self.x - s_hat).argmin())
    
    def _get_choice_probability(self, s_hat: float) -> float:
        """
        Compute P(choose B) given perceived stimulus.
        
        P(choose B) = P(boundary < s_hat) = CDF of boundary belief at s_hat
        
        Args:
            s_hat: Perceived stimulus value
        
        Returns:
            P(choose B)
        """
        j = self._find_closest_idx(s_hat)
        p_B = trapezoid(self.boundary_belief[:j+1], self.x[:j+1])
        return np.clip(p_B, 1e-10, 1 - 1e-10)
    
    def _update_belief(self, s_hat: float, true_category: int):
        """
        Update boundary belief based on feedback.
        
        Uses TRUE CATEGORY (not choice) for update. This implements:
        1. Learning update: sigmoid shift based on feedback
        2. Relaxation: drift back toward uniform
        
        Args:
            s_hat: Perceived stimulus value
            true_category: True category (0 = A, 1 = B)
        """
        # C = +1 for category B, -1 for category A
        C = 1 if true_category == 1 else -1
        
        # Learning update (sigmoid)
        delta_learning = 1 / (1 + np.exp(-self.sigma_boundary * C * (self.x - s_hat)))
        y_prime = self.boundary_belief - self.mu_learning * delta_learning
        
        # Relaxation toward uniform (0.5)
        delta_relax = y_prime - 0.5
        y_double_prime = y_prime - self.mu_relax * delta_relax
        
        # Ensure non-negative
        min_val = np.min(y_double_prime)
        if min_val < 0:
            y_double_prime = y_double_prime + np.abs(min_val)
        
        # Normalise
        self.boundary_belief = y_double_prime / trapezoid(y_double_prime, self.x)
        
        # Store for next trial's repulsion
        self.s_hat_prev = s_hat
    
    def simulate_session(self, stimuli: np.ndarray, categories: np.ndarray,
                         no_response: Optional[np.ndarray] = None,
                         rng: Optional[np.random.Generator] = None
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
        
        Returns:
            choices: Simulated choice sequence (0 = A, 1 = B, NaN = no response)
            p_B_sequence: P(choose B) at each trial
        """
        if rng is None:
            rng = np.random.default_rng()
        if no_response is None:
            no_response = np.zeros(len(stimuli), dtype=bool)
        
        n = len(stimuli)
        choices = np.full(n, np.nan)
        p_B_sequence = np.full(n, np.nan)
        
        for t in range(n):
            if no_response[t]:
                continue
            
            # Perceive
            s_hat = self._perceive_stimulus(stimuli[t], rng)
            
            # Compute choice probability
            p_B = self._get_choice_probability(s_hat)
            p_B_sequence[t] = p_B
            
            # Model makes choice
            choice = rng.binomial(1, p_B)
            choices[t] = choice
            
            # Update based on TRUE CATEGORY (not choice)
            self._update_belief(s_hat, categories[t])
        
        return choices, p_B_sequence
    
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
            trial_log_liks: Array of per-trial log-likelihoods (only for evaluated trials)
            n_eval: Number of evaluated trials
        """
        rng = np.random.default_rng(seed)
        
        if no_response is None:
            no_response = np.isnan(observed_choices)
        
        log_liks = []
        n_eval = 0
        
        for t in range(len(stimuli)):
            if no_response[t]:
                continue
            
            # Perceive
            s_hat = self._perceive_stimulus(stimuli[t], rng)
            
            # Compute choice probability
            p_B = self._get_choice_probability(s_hat)
            
            # Accumulate LL if in eval set
            if eval_mask is None or eval_mask[t]:
                if observed_choices[t] == 1:
                    log_liks.append(np.log(p_B))
                else:
                    log_liks.append(np.log(1 - p_B))
                n_eval += 1
            
            # ALWAYS update belief (maintains correct state)
            self._update_belief(s_hat, categories[t])
        
        log_liks = np.array(log_liks)
        total_log_lik = np.sum(log_liks) if len(log_liks) > 0 else 0.0
        
        return total_log_lik, log_liks, n_eval
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def get_belief_copy(self) -> np.ndarray:
        """Return copy of current boundary belief distribution."""
        return self.boundary_belief.copy()
    
    def set_belief(self, belief: np.ndarray):
        """
        Set boundary belief distribution (for carrying across sessions).
        
        Args:
            belief: New belief distribution (will be normalised)
        """
        self.boundary_belief = belief.copy()
        # Normalise
        self.boundary_belief = self.boundary_belief / trapezoid(self.boundary_belief, self.x)
    
    def get_params(self) -> Dict[str, float]:
        """Return current parameters as dict."""
        return {
            'sigma_percep': self.sigma_percep,
            'A_repulsion': self.A_repulsion,
            'mu_learning': self.mu_learning,
            'mu_relax': self.mu_relax
        }
    
    def get_state(self) -> Dict:
        """Return complete model state (for checkpointing)."""
        return {
            'params': self.get_params(),
            'boundary_belief': self.boundary_belief.copy(),
            's_hat_prev': self.s_hat_prev,
            'x': self.x.copy(),
            'x_min': self.x_min,
            'x_max': self.x_max,
            'n_points': self.n_points
        }
    
    # =========================================================================
    # CLASS METHODS: PARAMETER INFO
    # =========================================================================
    
    @classmethod
    def get_bounds(cls) -> Dict[str, Tuple[float, float]]:
        """Parameter bounds for fitting."""
        return {
            'sigma_percep': (0.05, 0.5),
            'A_repulsion': (0.0, 0.5),
            'mu_learning': (0.05, 0.9),
            'mu_relax': (0.01, 0.4)
        }
    
    @classmethod
    def get_param_names(cls) -> List[str]:
        """Parameter names in canonical order."""
        return ['sigma_percep', 'A_repulsion', 'mu_learning', 'mu_relax']
    
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
            ) -> Tuple['BoundaryEstimationModel', Dict]: #type: ignore
        
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
            results: Dict with:
                - 'params': All parameter values
                - 'free_params': Dict of fitted parameters
                - 'fixed_params': Dict of fixed parameters
                - 'train_nll': Training negative log-likelihood
                - 'train_nll_per_trial': NLL per trial
                - 'test_nll': Test NLL (if validation used)
                - 'test_nll_per_trial': Test NLL per trial (if validation used)
                - 'aic', 'bic': Information criteria
                - 'n_train', 'n_test': Number of trials
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
            params = fixed_params.copy()
            for name, val in zip(free_param_names, free_param_values):
                params[name] = val
            
            # Create model
            model = cls(
                sigma_percep=params['sigma_percep'],
                A_repulsion=params['A_repulsion'],
                mu_learning=params['mu_learning'],
                mu_relax=params['mu_relax']
            )
            
            # Set initial belief
            if initial_belief is not None:
                model.set_belief(initial_belief)
            elif burn_in > 0:
                model.reset_belief(burn_in=burn_in, burn_in_seed=burn_in_seed)
            
            ll, _, _ = model.compute_log_likelihood(
                stimuli, categories, observed_choices,
                no_response=no_response
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
            mu_learning=best_params['mu_learning'],
            mu_relax=best_params['mu_relax']
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
            """NLL on training trials only, but process all trials for correct state."""
            params = fixed_params.copy()
            for name, val in zip(free_param_names, free_param_values):
                params[name] = val
            
            model = cls(
                sigma_percep=params['sigma_percep'],
                A_repulsion=params['A_repulsion'],
                mu_learning=params['mu_learning'],
                mu_relax=params['mu_relax']
            )
            
            if initial_belief is not None:
                model.set_belief(initial_belief)
            elif burn_in > 0:
                model.reset_belief(burn_in=burn_in, burn_in_seed=burn_in_seed)
            
            ll, _, _ = model.compute_log_likelihood(
                stimuli, categories, observed_choices,
                eval_mask=train_mask,
                no_response=no_response
            )
            
            return -ll
        
        # Multiple restarts
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
        
        # Reconstruct full params
        best_params = fixed_params.copy()
        for name, val in zip(free_param_names, best_free_params):
            best_params[name] = val
        
        # Create fitted model and compute test NLL
        best_model = cls(
            sigma_percep=best_params['sigma_percep'],
            A_repulsion=best_params['A_repulsion'],
            mu_learning=best_params['mu_learning'],
            mu_relax=best_params['mu_relax']
        )
        
        if initial_belief is not None:
            best_model.set_belief(initial_belief)
        elif burn_in > 0:
            best_model.reset_belief(burn_in=burn_in, burn_in_seed=burn_in_seed)
        
        # Compute test NLL (process all trials, eval only test)
        test_ll, _, n_test = best_model.compute_log_likelihood(
            stimuli, categories, observed_choices,
            eval_mask=test_mask,
            no_response=no_response
        )
        test_nll = -test_ll
        
        # Reset model for return
        if initial_belief is not None:
            best_model.set_belief(initial_belief)
        elif burn_in > 0:
            best_model.reset_belief(burn_in=burn_in, burn_in_seed=burn_in_seed)
        else:
            best_model.reset_belief()
        
        # Compute statistics
        n_train = int(np.sum(train_mask))
        n_free_params = len(free_param_names)
        aic = 2 * best_nll + 2 * n_free_params
        bic = 2 * best_nll + n_free_params * np.log(n_train)
        
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
            'aic': aic,
            'bic': bic,
            'split_idx': split_idx,
            'test_fraction': test_fraction,
            'burn_in': burn_in
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
        """MLE fitting with block-based cross-validation."""
        
        block_size = validation_config.get('block_size', 50)
        n_folds = validation_config.get('n_folds', 2)
        n_repetitions = validation_config.get('n_repetitions', 4)
        
        if no_response is None:
            no_response = np.isnan(observed_choices)
        
        n_trials = len(stimuli)
        valid_trials = ~no_response
        valid_indices = np.where(valid_trials)[0]
        
        # Create blocks
        n_blocks = len(valid_indices) // block_size
        if n_blocks < n_folds:
            warnings.warn(f"Not enough blocks ({n_blocks}) for {n_folds}-fold CV. "
                         f"Using holdout instead.")
            return cls._fit_mle_holdout(
                stimuli, categories, observed_choices, no_response,
                fixed_params, initial_belief, burn_in, burn_in_seed,
                {'test_fraction': 1/n_folds}, method, n_restarts, seed
            )
        
        all_bounds = cls.get_bounds()
        all_param_names = cls.get_param_names()
        
        if fixed_params is None:
            fixed_params = {}
        
        free_param_names = [p for p in all_param_names if p not in fixed_params]
        free_bounds = [all_bounds[p] for p in free_param_names]
        
        rng = np.random.default_rng(seed)
        
        # Storage for CV results
        all_test_nlls = []
        all_train_nlls = []
        
        for rep in range(n_repetitions):
            # Shuffle blocks
            block_indices = list(range(n_blocks))
            rng.shuffle(block_indices)
            
            # Split into folds
            fold_size = n_blocks // n_folds
            
            for fold in range(n_folds):
                # Determine test blocks
                test_block_start = fold * fold_size
                test_block_end = test_block_start + fold_size
                test_blocks = block_indices[test_block_start:test_block_end]
                train_blocks = [b for b in block_indices if b not in test_blocks]
                
                # Create masks
                train_mask = np.zeros(n_trials, dtype=bool)
                test_mask = np.zeros(n_trials, dtype=bool)
                
                for b in train_blocks:
                    start = b * block_size
                    end = min(start + block_size, len(valid_indices))
                    train_mask[valid_indices[start:end]] = True
                
                for b in test_blocks:
                    start = b * block_size
                    end = min(start + block_size, len(valid_indices))
                    test_mask[valid_indices[start:end]] = True
                
                # Fit on train, evaluate on test
                def neg_log_likelihood_train(free_param_values):
                    params = fixed_params.copy()
                    for name, val in zip(free_param_names, free_param_values):
                        params[name] = val
                    
                    model = cls(
                        sigma_percep=params['sigma_percep'],
                        A_repulsion=params['A_repulsion'],
                        mu_learning=params['mu_learning'],
                        mu_relax=params['mu_relax']
                    )
                    
                    if initial_belief is not None:
                        model.set_belief(initial_belief)
                    elif burn_in > 0:
                        model.reset_belief(burn_in=burn_in, burn_in_seed=burn_in_seed)
                    
                    ll, _, _ = model.compute_log_likelihood(
                        stimuli, categories, observed_choices,
                        eval_mask=train_mask,
                        no_response=no_response
                    )
                    
                    return -ll
                
                # Fit
                best_nll_fold = np.inf
                best_params_fold = None
                
                for i in range(n_restarts):
                    x0 = [rng.uniform(b[0], b[1]) for b in free_bounds]
                    
                    try:
                        result = minimize(
                            neg_log_likelihood_train,
                            x0=x0,
                            method=method,
                            bounds=free_bounds,
                            options={'maxiter': 500}
                        )
                        
                        if result.fun < best_nll_fold:
                            best_nll_fold = result.fun
                            best_params_fold = result.x
                    
                    except Exception:
                        continue
                
                if best_params_fold is not None:
                    # Compute test NLL
                    params = fixed_params.copy()
                    for name, val in zip(free_param_names, best_params_fold):
                        params[name] = val
                    
                    model = cls(
                        sigma_percep=params['sigma_percep'],
                        A_repulsion=params['A_repulsion'],
                        mu_learning=params['mu_learning'],
                        mu_relax=params['mu_relax']
                    )
                    
                    if initial_belief is not None:
                        model.set_belief(initial_belief)
                    elif burn_in > 0:
                        model.reset_belief(burn_in=burn_in, burn_in_seed=burn_in_seed)
                    
                    test_ll, _, n_test = model.compute_log_likelihood(
                        stimuli, categories, observed_choices,
                        eval_mask=test_mask,
                        no_response=no_response
                    )
                    
                    all_train_nlls.append(best_nll_fold / np.sum(train_mask))
                    all_test_nlls.append(-test_ll / n_test if n_test > 0 else np.nan)
        
        # Final fit on all data
        best_model, final_results = cls._fit_mle_no_validation(
            stimuli, categories, observed_choices, no_response,
            fixed_params, initial_belief, burn_in, burn_in_seed,
            method, n_restarts, seed
        )
        
        # Add CV results
        final_results['validation'] = 'cv'
        final_results['cv_train_nlls'] = np.array(all_train_nlls)
        final_results['cv_test_nlls'] = np.array(all_test_nlls)
        final_results['cv_train_nll_mean'] = np.nanmean(all_train_nlls)
        final_results['cv_test_nll_mean'] = np.nanmean(all_test_nlls)
        final_results['cv_train_nll_std'] = np.nanstd(all_train_nlls)
        final_results['cv_test_nll_std'] = np.nanstd(all_test_nlls)
        final_results['cv_config'] = validation_config
        
        return best_model, final_results
    
    # =========================================================================
    # DIAGNOSTICS: SERIAL DEPENDENCE COMPARISON
    # =========================================================================
    
    def compare_serial_dependence(self, stimuli: np.ndarray, categories: np.ndarray,
                                   observed_choices: np.ndarray, rewards: np.ndarray,
                                   no_response: np.ndarray, not_blockstart: np.ndarray,
                                   n_simulations: int = 10,
                                   n_bins: int = 8,
                                   trial_filter: str = 'post_correct',
                                   seed: int = 42) -> Dict:
        """
        Compare model's serial dependence pattern to observed data.
        
        Simulates choices from the model and computes update matrices for both
        data and model, allowing direct comparison of serial dependence patterns.
        
        Args:
            stimuli: Stimulus values
            categories: True categories
            observed_choices: Animal's actual choices
            rewards: Actual rewards (1 = correct)
            no_response: Boolean array (True = no response)
            not_blockstart: Boolean array (True = not start of block)
            n_simulations: Number of model simulations to average
            n_bins: Number of bins for update matrix
            trial_filter: 'post_correct' or 'all'
            seed: Random seed
        
        Returns:
            Dict with:
                - data_update_matrix, data_conditional_matrix
                - model_update_matrix (mean), model_conditional_matrix (mean)
                - model_update_matrices (all simulations)
                - matrix_error: MSE between data and model matrices
                - data_info, model_info: Fitting details
        """
        # Store initial state
        initial_belief = self.get_belief_copy()
        initial_s_hat_prev = self.s_hat_prev
        
        # Compute data matrices
        data_update, data_cond, data_info = compute_update_matrix(
            stimuli, observed_choices, rewards, no_response, not_blockstart,
            n_bins=n_bins, trial_filter=trial_filter
        )
        
        # Simulate and compute model matrices
        model_updates = []
        model_conds = []
        
        for sim in range(n_simulations):
            # Reset to initial state
            self.set_belief(initial_belief)
            self.s_hat_prev = initial_s_hat_prev
            
            # Simulate
            sim_rng = np.random.default_rng(seed + sim * 1000)
            sim_choices, _ = self.simulate_session(stimuli, categories, no_response, sim_rng)
            
            # Compute rewards for simulated choices
            sim_rewards = (sim_choices == categories).astype(float)
            sim_rewards[np.isnan(sim_choices)] = np.nan
            
            # Compute matrices
            sim_update, sim_cond, _ = compute_update_matrix(
                stimuli, sim_choices, sim_rewards, no_response, not_blockstart,
                n_bins=n_bins, trial_filter=trial_filter
            )
            
            model_updates.append(sim_update)
            model_conds.append(sim_cond)
        
        # Restore initial state
        self.set_belief(initial_belief)
        self.s_hat_prev = initial_s_hat_prev
        
        # Average model matrices
        model_updates = np.array(model_updates)
        model_conds = np.array(model_conds)
        
        mean_model_update = np.nanmean(model_updates, axis=0)
        mean_model_cond = np.nanmean(model_conds, axis=0)
        std_model_update = np.nanstd(model_updates, axis=0)
        
        # Compute error
        error = matrix_error(mean_model_update, data_update)
        
        return {
            'data_update_matrix': data_update,
            'data_conditional_matrix': data_cond,
            'model_update_matrix': mean_model_update,
            'model_conditional_matrix': mean_model_cond,
            'model_update_std': std_model_update,
            'model_update_matrices': model_updates,
            'model_conditional_matrices': model_conds,
            'matrix_error': error,
            'data_info': data_info,
            'n_simulations': n_simulations,
            'n_bins': n_bins,
            'trial_filter': trial_filter
        }
    
    def compare_to_data(self, stimuli: np.ndarray, categories: np.ndarray,
                        observed_choices: np.ndarray,
                        n_simulations: int = 10,
                        seed: int = 42) -> Dict:
        """
        Compare model psychometric predictions to actual data.
        
        Args:
            stimuli: Stimulus values
            categories: True categories
            observed_choices: Animal's actual choices
            n_simulations: Number of model simulations
            seed: Random seed
        
        Returns:
            Dict with psychometric comparison results
        """
        # Store initial state
        initial_belief = self.get_belief_copy()
        initial_s_hat_prev = self.s_hat_prev
        
        no_response = np.isnan(observed_choices)
        
        # Fit data psychometric
        data_psych = fit_psychometric(stimuli[~no_response], observed_choices[~no_response])
        
        # Simulate and fit model psychometrics
        model_psychs = []
        x_eval = np.linspace(-1, 1, 100)
        
        for sim in range(n_simulations):
            self.set_belief(initial_belief)
            self.s_hat_prev = initial_s_hat_prev
            
            sim_rng = np.random.default_rng(seed + sim * 1000)
            sim_choices, _ = self.simulate_session(stimuli, categories, no_response, sim_rng)
            
            valid = ~np.isnan(sim_choices)
            psych = fit_psychometric(stimuli[valid], sim_choices[valid], x_eval)
            model_psychs.append(psych)
        
        # Restore state
        self.set_belief(initial_belief)
        self.s_hat_prev = initial_s_hat_prev
        
        # Average model psychometric
        model_curves = np.array([p['y_fit'] for p in model_psychs if p['success']])
        
        if len(model_curves) > 0:
            mean_curve = np.mean(model_curves, axis=0)
            std_curve = np.std(model_curves, axis=0)
        else:
            mean_curve = np.full(len(x_eval), np.nan)
            std_curve = np.full(len(x_eval), np.nan)
        
        return {
            'data_psychometric': data_psych,
            'model_psychometrics': model_psychs,
            'model_curve_mean': mean_curve,
            'model_curve_std': std_curve,
            'x_eval': x_eval,
            'n_simulations': n_simulations
        }
    
    # =========================================================================
    # PARAMETER RECOVERY
    # =========================================================================
    
    @classmethod
    def parameter_recovery(cls, n_tests: int = 20, n_trials: int = 300,
                           fixed_params: Optional[Dict[str, float]] = None,
                           burn_in: int = 0,
                           burn_in_seed: int = 42,
                           validation: Optional[str] = None,
                           validation_config: Optional[Dict] = None,
                           seed: int = 42,
                           verbose: bool = True
                           ) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Test parameter recovery on simulated data.
        
        For each test:
        1. Sample true parameters from uniform prior
        2. Simulate a session
        3. Fit the model
        4. Compare recovered to true parameters
        
        Args:
            n_tests: Number of recovery tests
            n_trials: Trials per simulated session
            fixed_params: Parameters to fix (only test free parameters)
            burn_in: Burn-in trials for simulation
            burn_in_seed: Seed for burn-in
            validation: Validation method for fitting
            validation_config: Validation settings
            seed: Random seed
            verbose: Print progress
        
        Returns:
            true_params: Dict of arrays {param_name: true values}
            recovered_params: Dict of arrays {param_name: recovered values}
            correlations: Dict of correlations per parameter
            diagnostics: Dict with NLLs, additional metrics
        """
        all_bounds = cls.get_bounds()
        all_param_names = cls.get_param_names()
        
        if fixed_params is None:
            fixed_params = {}
        
        free_param_names = [p for p in all_param_names if p not in fixed_params]
        
        rng = np.random.default_rng(seed)
        
        # Storage
        true_params = {name: [] for name in free_param_names}
        recovered_params = {name: [] for name in free_param_names}
        diagnostics = {
            'train_nlls': [],
            'test_nlls': [],
        }
        
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
            stimuli, categories, rng = generate_stimuli(x_min = -1,
                                                        x_max = 1,
                                                        n_trials = n_trials)

            # Create model with true params
            model_true = cls(**all_true)
            if burn_in > 0:
                model_true.reset_belief(burn_in=burn_in, burn_in_seed=burn_in_seed + i)
            
            # Simulate
            sim_seed = seed + i * 1000
            choices_true, _ = model_true.simulate_session(
                stimuli, categories, rng=np.random.default_rng(sim_seed)
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
                
                # Store
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
            
            # Identity line
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
        
        # Hide unused axes
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
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
        initial_belief = self.get_belief_copy()
        initial_s_hat_prev = self.s_hat_prev
        
        rng = np.random.default_rng(seed)
        n_trials = len(stimuli)
        
        # Determine snapshot points
        snapshot_trials = np.linspace(0, n_trials - 1, n_snapshots + 1).astype(int)
        
        # Collect beliefs
        beliefs = [self.get_belief_copy()]
        trial_indices = [0]
        
        for t in range(n_trials):
            if no_response[t]:
                continue
            
            s_hat = self._perceive_stimulus(stimuli[t], rng)
            self._update_belief(s_hat, categories[t])
            
            if t + 1 in snapshot_trials:
                beliefs.append(self.get_belief_copy())
                trial_indices.append(t + 1)
        
        # Restore state
        self.set_belief(initial_belief)
        self.s_hat_prev = initial_s_hat_prev
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(beliefs)))
        
        for i, (belief, trial_idx) in enumerate(zip(beliefs, trial_indices)):
            ax.plot(self.x, belief, color=colors[i], 
                   label=f'Trial {trial_idx}', alpha=0.8)
        
        ax.axvline(0, color='k', linestyle='--', alpha=0.3, label='True boundary')
        ax.set_xlabel('Stimulus space')
        ax.set_ylabel('Belief density')
        ax.set_title('Boundary belief evolution')
        ax.legend(loc='upper right')
        ax.set_xlim(self.x_min, self.x_max)
        
        plt.tight_layout()
        return fig