from Models.BE_model import BoundaryEstimationModel

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

class MixedAgent:
    """
    Agent that combines BE model with heuristic strategies via probabilistic mixture.
    
    P(B) = α × P(B|BE) + (1-α) × P(B|heuristics)
    
    Heuristics include:
        - Side bias: constant preference for B (or A if negative)
        - Win-stay: repeat choice after reward
        - Lose-shift: switch choice after no reward
        - Random: 50/50
    
    Heuristics are combined via weighted mixture, renormalised based on which 
    are active (win-stay only after reward, lose-shift only after no reward).
    
    First trial uses only bias + random (no history available).
    
    Usage:
        agent = MixedAgent(alpha=0.7, bias=0.1, p_winstay=0.8, ...)
        choices, rewards = agent.simulate_session(stimuli, categories)
    """
    
    def __init__(
        self,
        # BE params
        sigma_percep: float = 0.15,
        A_repulsion: float = 0.1,
        mu_learning: float = 0.35,
        mu_relax: float = 0.12,
        # Mixture weight
        alpha: float = 1.0,  # BE weight; 0 = pure heuristic, 1 = pure BE
        # Heuristic params
        bias: float = 0.0,         # Side bias: P(B) = 0.5 + bias, range [-0.5, 0.5]
        p_winstay: float = 0.5,    # P(repeat choice | prev rewarded), range [0, 1]
        p_loseshift: float = 0.5,  # P(switch choice | prev unrewarded), range [0, 1]
        # Heuristic weights (unnormalised, will be renormalised internally)
        w_bias: float = 1.0,
        w_winstay: float = 1.0,
        w_loseshift: float = 1.0,
        w_random: float = 1.0,
        # Initialisation
        burn_in: int = 0,
        burn_in_seed: int = 42,
        # Grid resolution for BE model
        n_points: int = 500
    ):
        """
        Initialise MixedAgent.
        
        Args:
            sigma_percep: Perceptual noise (BE param)
            A_repulsion: Repulsion strength (BE param)
            mu_learning: Learning rate (BE param)
            mu_relax: Relaxation rate (BE param)
            alpha: BE weight in mixture (0 = pure heuristic, 1 = pure BE)
            bias: Side bias, added to P(B) = 0.5 + bias
            p_winstay: Probability of repeating choice after reward
            p_loseshift: Probability of switching choice after no reward
            w_bias: Weight for bias heuristic
            w_winstay: Weight for win-stay heuristic
            w_loseshift: Weight for lose-shift heuristic
            w_random: Weight for random heuristic
            burn_in: Number of burn-in trials for BE model
            burn_in_seed: Seed for burn-in simulation
            n_points: Grid resolution for BE model
        """
        # Validate parameters
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not -0.5 <= bias <= 0.5:
            raise ValueError(f"bias must be in [-0.5, 0.5], got {bias}")
        if not 0 <= p_winstay <= 1:
            raise ValueError(f"p_winstay must be in [0, 1], got {p_winstay}")
        if not 0 <= p_loseshift <= 1:
            raise ValueError(f"p_loseshift must be in [0, 1], got {p_loseshift}")
        if any(w < 0 for w in [w_bias, w_winstay, w_loseshift, w_random]):
            raise ValueError("Heuristic weights must be non-negative")
        
        # Store BE params
        self.sigma_percep = sigma_percep
        self.A_repulsion = A_repulsion
        self.mu_learning = mu_learning
        self.mu_relax = mu_relax
        
        # Mixture weight
        self.alpha = alpha
        
        # Heuristic params
        self.bias = bias
        self.p_winstay = p_winstay
        self.p_loseshift = p_loseshift
        
        # Heuristic weights
        self.w_bias = w_bias
        self.w_winstay = w_winstay
        self.w_loseshift = w_loseshift
        self.w_random = w_random
        
        # Initialisation params
        self.burn_in = burn_in
        self.burn_in_seed = burn_in_seed
        self.n_points = n_points
        
        # Create internal BE model
        self._be_model = BoundaryEstimationModel(
            sigma_percep=sigma_percep,
            A_repulsion=A_repulsion,
            mu_learning=mu_learning,
            mu_relax=mu_relax,
            n_points=n_points
        )
        
        # Initialise BE model
        self._be_model.reset_belief(burn_in=burn_in, burn_in_seed=burn_in_seed)
    
    def reset(self, burn_in: Optional[int] = None, burn_in_seed: Optional[int] = None):
        """
        Reset agent state (BE model belief).
        
        Args:
            burn_in: New burn-in value (optional, uses stored if None)
            burn_in_seed: New burn-in seed (optional, uses stored if None)
        """
        if burn_in is not None:
            self.burn_in = burn_in
        if burn_in_seed is not None:
            self.burn_in_seed = burn_in_seed
        self._be_model.reset_belief(burn_in=self.burn_in, burn_in_seed=self.burn_in_seed)
    
    def _get_heuristic_p_B(
        self,
        trial_idx: int,
        prev_choice: Optional[int] = None,
        prev_reward: Optional[bool] = None
    ) -> float:
        """
        Compute P(B) from heuristics using Option A (renormalise based on active).
        
        Args:
            trial_idx: Current trial index (0-indexed)
            prev_choice: Previous choice (0=A, 1=B) or None if first trial
            prev_reward: Whether previous trial was rewarded, or None if first trial
        
        Returns:
            P(B) from heuristic mixture
        """
        # P(B) for always-active heuristics
        p_bias = 0.5 + self.bias
        p_random = 0.5
        
        # First trial: only bias + random
        if trial_idx == 0 or prev_choice is None or prev_reward is None:
            total_weight = self.w_bias + self.w_random
            if total_weight == 0:
                return 0.5
            return (self.w_bias * p_bias + self.w_random * p_random) / total_weight
        
        # After reward: bias + win-stay + random
        if prev_reward:
            # Win-stay: P(repeat previous choice)
            # If prev_choice = B (1), P(B) = p_winstay (stay on B)
            # If prev_choice = A (0), P(B) = 1 - p_winstay (stay on A means not B)
            if prev_choice == 1:
                p_winstay = self.p_winstay
            else:
                p_winstay = 1 - self.p_winstay
            
            total_weight = self.w_bias + self.w_winstay + self.w_random
            if total_weight == 0:
                return 0.5
            return (self.w_bias * p_bias + 
                    self.w_winstay * p_winstay + 
                    self.w_random * p_random) / total_weight
        
        # After no reward: bias + lose-shift + random
        else:
            # Lose-shift: P(switch from previous choice)
            # If prev_choice = B (1), P(B) = 1 - p_loseshift (shift away from B)
            # If prev_choice = A (0), P(B) = p_loseshift (shift to B)
            if prev_choice == 1:
                p_loseshift = 1 - self.p_loseshift
            else:
                p_loseshift = self.p_loseshift
            
            total_weight = self.w_bias + self.w_loseshift + self.w_random
            if total_weight == 0:
                return 0.5
            return (self.w_bias * p_bias + 
                    self.w_loseshift * p_loseshift + 
                    self.w_random * p_random) / total_weight
    
    def get_choice_probability(
        self,
        s_hat: float,
        trial_idx: int,
        prev_choice: Optional[int] = None,
        prev_reward: Optional[bool] = None
    ) -> Tuple[float, float, float]:
        """
        Compute P(B) as mixture of BE and heuristics.
        
        Args:
            s_hat: Noisy stimulus percept
            trial_idx: Current trial index
            prev_choice: Previous choice (0=A, 1=B) or None
            prev_reward: Whether previous trial was rewarded, or None
        
        Returns:
            p_B: Mixed P(B)
            p_be: P(B) from BE model only
            p_heuristic: P(B) from heuristics only
        """
        # BE component
        p_be = self._be_model._get_choice_probability(s_hat)
        
        # Heuristic component
        p_heuristic = self._get_heuristic_p_B(trial_idx, prev_choice, prev_reward)
        
        # Mixture
        p_B = self.alpha * p_be + (1 - self.alpha) * p_heuristic
        p_B = np.clip(p_B, 0, 1)
        
        return p_B, p_be, p_heuristic
    
    def simulate_session(
        self,
        stimuli: np.ndarray,
        categories: np.ndarray,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a session of choices.
        
        Args:
            stimuli: Array of stimulus values
            categories: Array of true categories (0=A, 1=B)
            rng: Random number generator
        
        Returns:
            choices: Array of choices (0=A, 1=B)
            rewards: Array of rewards (0=incorrect, 1=correct)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        n_trials = len(stimuli)
        choices = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials, dtype=int)
        
        prev_choice = None
        prev_reward = None
        
        for t in range(n_trials):
            s = stimuli[t]
            
            # Noisy percept
            s_hat = s + rng.normal(0, self.sigma_percep)
            s_hat = np.clip(s_hat, -1, 1)
            
            # Get P(B) from mixture
            p_B, _, _ = self.get_choice_probability(s_hat, t, prev_choice, prev_reward)
            
            # Make choice
            choice = int(rng.random() < p_B)
            choices[t] = choice
            
            # Determine reward (correct = choice matches category)
            correct = (choice == categories[t])
            rewards[t] = int(correct)
            
            # Update BE model belief (always updates, even if alpha < 1)
            self._be_model._update_belief(s_hat, correct)
            
            # Update history for next trial
            prev_choice = choice
            prev_reward = correct
        
        return choices, rewards
    
    def simulate_session_detailed(
        self,
        stimuli: np.ndarray,
        categories: np.ndarray,
        rng: Optional[np.random.Generator] = None
    ) -> pd.DataFrame:
        """
        Simulate session with detailed trial-by-trial information.
        
        Returns DataFrame with columns:
            trial, stimulus, category, s_hat, p_be, p_heuristic, p_mixed,
            choice, correct, prev_choice, prev_reward
        """
        if rng is None:
            rng = np.random.default_rng()
        
        n_trials = len(stimuli)
        records = []
        
        prev_choice = None
        prev_reward = None
        
        for t in range(n_trials):
            s = stimuli[t]
            cat = categories[t]
            
            # Noisy percept
            s_hat = s + rng.normal(0, self.sigma_percep)
            s_hat = np.clip(s_hat, -1, 1)
            
            # Get probabilities
            p_B, p_be, p_heuristic = self.get_choice_probability(
                s_hat, t, prev_choice, prev_reward
            )
            
            # Make choice
            choice = int(rng.random() < p_B)
            correct = (choice == cat)
            
            # Record
            records.append({
                'trial': t,
                'stimulus': s,
                'category': cat,
                's_hat': s_hat,
                'p_be': p_be,
                'p_heuristic': p_heuristic,
                'p_mixed': p_B,
                'choice': choice,
                'correct': int(correct),
                'prev_choice': prev_choice,
                'prev_reward': prev_reward
            })
            
            # Update BE model
            self._be_model._update_belief(s_hat, correct)
            
            # Update history
            prev_choice = choice
            prev_reward = correct
        
        return pd.DataFrame(records)
    
    def get_params(self) -> Dict[str, float]:
        """Return all parameters as dictionary."""
        return {
            # BE params
            'sigma_percep': self.sigma_percep,
            'A_repulsion': self.A_repulsion,
            'mu_learning': self.mu_learning,
            'mu_relax': self.mu_relax,
            # Mixture
            'alpha': self.alpha,
            # Heuristics
            'bias': self.bias,
            'p_winstay': self.p_winstay,
            'p_loseshift': self.p_loseshift,
            'w_bias': self.w_bias,
            'w_winstay': self.w_winstay,
            'w_loseshift': self.w_loseshift,
            'w_random': self.w_random,
            # Initialisation
            'burn_in': self.burn_in
        }
    
    def get_be_params(self) -> Dict[str, float]:
        """Return only BE model parameters."""
        return {
            'sigma_percep': self.sigma_percep,
            'A_repulsion': self.A_repulsion,
            'mu_learning': self.mu_learning,
            'mu_relax': self.mu_relax
        }
    
    def get_heuristic_params(self) -> Dict[str, float]:
        """Return only heuristic parameters."""
        return {
            'bias': self.bias,
            'p_winstay': self.p_winstay,
            'p_loseshift': self.p_loseshift,
            'w_bias': self.w_bias,
            'w_winstay': self.w_winstay,
            'w_loseshift': self.w_loseshift,
            'w_random': self.w_random
        }
    
    @classmethod
    def from_params(cls, params: Dict[str, float]) -> 'MixedAgent':
        """Create MixedAgent from parameter dictionary."""
        return cls(**params) #type: ignore
    
    @property
    def boundary_belief(self) -> np.ndarray:
        """Access BE model's current boundary belief."""
        return self._be_model.boundary_belief
    
    @property
    def x(self) -> np.ndarray:
        """Access BE model's stimulus grid."""
        return self._be_model.x
    
    def __repr__(self) -> str:
        return (f"MixedAgent(alpha={self.alpha}, "
                f"bias={self.bias}, p_ws={self.p_winstay}, p_ls={self.p_loseshift}, "
                f"mu_learning={self.mu_learning}, burn_in={self.burn_in})")
