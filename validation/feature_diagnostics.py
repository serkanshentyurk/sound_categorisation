"""Synthetic feature diagnostics for the sound-categorisation models.

`compute_param_stat_correlations` samples model parameters from the prior, simulates one
session per draw on a fixed uniform stimulus sequence, computes the flattened summary-stat
vector, and returns the per-parameter / per-summary-stat correlation matrix plus the raw
draws. Built directly on the model simulators and behav_utils stats — no SBI/torch.

Noise is held fixed across draws (same simulator seed every time) so the only thing varying
is the parameter set; the correlations therefore reflect parameter sensitivity rather than
trial noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from behav_utils.data.synthetic import sample_stimuli
from behav_utils.analysis.summary_stats import (
    fit_summary_stats, flatten_stats, get_stat_names_expanded, list_available_stats,
)
from models.BE_core import BEModel, BEParams, BEState
from models.SC_core import SCModel, SCParams, SCState

# model_type -> (Model, Params, initial-state factory)
_MODELS = {
    'be': (BEModel, BEParams, BEState.initial_uniform),
    'sc': (SCModel, SCParams, SCState.initial_default),
}


def compute_param_stat_correlations(model_type, stat_names=None, n_samples=1000,
                                    n_trials=2000, seed=0):
    """Correlate each model parameter with each summary statistic, across the prior.

    Parameters
    ----------
    model_type : {'be', 'sc'}
    stat_names : list of str, optional
        Defaults to ``list_available_stats()`` (the full expanded vector, including the
        update matrix and conditional psychometric).
    n_samples : int
        Number of parameter draws from the prior.
    n_trials : int
        Trials in the (single, shared) simulated session per draw. Keep this high enough
        that the per-session update matrix is well populated, or draws with non-finite
        stat vectors are dropped (see ``n_valid``).
    seed : int

    Returns
    -------
    dict with keys:
        corr_matrix          (n_params, n_stats_expanded) Pearson r
        param_names          list of parameter names
        stat_names_expanded  expanded stat names (columns of corr_matrix and x)
        theta                (n_valid, n_params) sampled parameters
        x                    (n_valid, n_stats_expanded) simulated stat vectors
        n_valid              number of draws with a finite stat vector
    """
    model_type = model_type.lower()
    if model_type not in _MODELS:
        raise ValueError(f"model_type must be 'be' or 'sc', got {model_type!r}")
    Model, Params, initial_state = _MODELS[model_type]

    if stat_names is None:
        stat_names = list_available_stats()
    stat_names = list(stat_names)
    param_names = Params.get_param_names()
    stat_names_expanded = get_stat_names_expanded(stat_names)

    # one fixed stimulus sequence, shared across all draws
    stimuli, categories = sample_stimuli(n_trials=n_trials, rng=np.random.default_rng(seed))
    no_response = np.zeros(n_trials, dtype=bool)
    not_blockstart = np.ones(n_trials, dtype=bool)
    not_blockstart[0] = False

    param_rng = np.random.default_rng(seed + 1)
    theta, x = [], []
    for _ in range(n_samples):
        params = Params.sample_prior(rng=param_rng)
        choices, *_ = Model.simulate_session(
            stimuli=stimuli, categories=categories, params=params,
            initial_state=initial_state(),
            rng=np.random.default_rng(seed + 2),        # fixed noise across draws
            no_response=no_response, not_blockstart=not_blockstart,
        )
        stats = flatten_stats(fit_summary_stats(
            choices, stimuli, categories, stat_names=stat_names, return_dict=True))
        if np.all(np.isfinite(stats)):
            theta.append(params.to_array())
            x.append(stats)

    theta = np.asarray(theta)
    x = np.asarray(x)

    # pairwise-complete Pearson r; constant/degenerate columns come back as NaN (no warning)
    corr = np.full((len(param_names), len(stat_names_expanded)), np.nan)
    if len(theta) > 1:
        full = pd.DataFrame(np.hstack([theta, x])).corr().to_numpy()
        corr = full[:theta.shape[1], theta.shape[1]:]

    return {
        'corr_matrix': corr,
        'param_names': param_names,
        'stat_names_expanded': stat_names_expanded,
        'theta': theta,
        'x': x,
        'n_valid': len(theta),
    }
