"""
SLDS / HMM state discovery and online prediction.

Utilities for fitting Hidden Markov Models and Switching Linear
Dynamical Systems to session-level behavioural statistics, and
predicting states for new sessions.

Extracted from NB 40. BIC-based model selection across K and D.

Public API:
    hmm_n_params    — Free parameter count for Gaussian-emission HMM
    slds_n_params   — Free parameter count for Gaussian-emission SLDS
    compute_bic     — BIC from log-likelihood, params, observations
    predict_state   — Online state prediction for a single session

Usage:
    from analysis.slds import predict_state, compute_bic

    state, info = predict_state(session, model, scaler, features)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


def hmm_n_params(K: int, D: int) -> int:
    """
    Free parameters in ssm Gaussian-emission HMM.

    Args:
        K: Number of discrete states
        D: Observation dimensionality
    """
    init = K - 1
    trans = K * (K - 1)
    means = K * D
    covariances = K * D * (D + 1) // 2
    return init + trans + means + covariances


def slds_n_params(K: int, D: int, N: int) -> int:
    """
    Free parameters in ssm Gaussian-emission SLDS.

    Args:
        K: Number of discrete states
        D: Latent dimensionality
        N: Observation dimensionality
    """
    init = K - 1
    trans = K * (K - 1)
    dynamics = K * (D * D + D)
    dynamics_cov = K * D * (D + 1) // 2
    emissions = K * (N * D + N)
    emission_cov = K * N * (N + 1) // 2
    return init + trans + dynamics + dynamics_cov + emissions + emission_cov


def compute_bic(ll: float, n_params: int, n_obs: int) -> float:
    """BIC = -2*LL + n_params*ln(n_obs). Lower is better."""
    return -2 * ll + n_params * np.log(n_obs)


def predict_state(
    session,
    model,
    scaler,
    features: List[str],
    model_type: str = 'hmm',
) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """
    Predict behavioural state for a single new session.

    Args:
        session: SessionData object
        model: Trained ssm HMM or SLDS model
        scaler: Fitted StandardScaler (from training)
        features: List of feature/stat names (must match training)
        model_type: 'hmm' or 'slds'

    Returns:
        (state_label, info_dict) where info_dict may contain
        'latent' for SLDS. Returns (None, None) if features
        contain NaN.
    """
    stats = session.stats(features)
    x = np.array([stats.get(f, np.nan) for f in features]).reshape(1, -1)

    if np.any(np.isnan(x)):
        return None, None

    x_std = scaler.transform(x)

    if model_type == 'hmm':
        state = model.most_likely_states(x_std)[0]
        return state, {}
    elif model_type == 'slds':
        z, x_lat = model.most_likely_states(x_std)
        return z[0], {'latent': x_lat[0]}
    else:
        raise ValueError(f'Unknown model_type: {model_type}')
