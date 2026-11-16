"""Simulation-based calibration (SBC) of a trained amortised network.

For ``n_draws`` fresh parameter sets from the prior: simulate ``n_sessions`` x ``T`` trials
(the same ``simulate_sessions`` the training simulator uses), condition the network on them
exactly as on an animal, draw ``n_posterior_samples``, and record for each parameter how many
samples lie below the true value (the rank, 0..L). A calibrated posterior gives ranks that are
uniform over the draws (Talts et al. 2018). Two readouts come from the same ranks:

    ranks         -> ECDF differences with simultaneous bands (inference.validation, after
                     Säilynoja et al. 2022) and KS p-values
    coverage      -> at each level c, the fraction of draws whose true value lies inside the
                     central c interval of its posterior (the rank lies within L*(1-c)/2 ..
                     L*(1+c)/2), against the nominal c

Each draw also keeps its posterior mean, SD, median and quantiles and the network's acceptance
rate, so parameter recovery (error, bias, R^2, contraction, z-scores) comes from the same draws
(``inference.validation.recovery_table``). ``budget_hit`` marks draws where fewer than L samples
fell inside the prior within the network's proposal budget (the samples were then resampled).
Degenerate draws (the observation has a non-finite stat the net cannot impute) are skipped and
counted in ``n_skipped``.

The draws come from the network's own prior (``net.prior_name``) and, for a network trained with
the accuracy rule, are repeated until every simulated session passes it (``n_rejected`` counts the
repeats): the truths then follow the same distribution as the network's training set, which is what
calibration is defined against. Ranks are the same in any monotone scale. Everything else stored
per parameter (truth, posterior mean, SD, median, quantiles, ``prior_bounds``) is in flat space
(``models.priors``): the logarithm for a log-uniform parameter (listed in ``log_uniform``), the
parameter itself otherwise. That is where the prior is a uniform box, so "relative to the prior
width" and contraction mean what they say.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from sound_categorisation.inference.simulator import simulate_sessions, theta_to_params
from sound_categorisation.models import priors
from sound_categorisation.settings import SBC_COVERAGE_LEVELS, SBC_POSTERIOR_QUANTILES, VALIDATION_SEED


def coverage_from_ranks(ranks: np.ndarray, n_posterior_samples: int,
                        levels: Sequence[float] = SBC_COVERAGE_LEVELS,
                        param_names: Sequence[str] | None = None) -> pd.DataFrame:
    """Empirical coverage of the central ``c`` interval per parameter from SBC ranks.

    Tidy rows ``{param, nominal, empirical, n}``; the binomial standard error of ``empirical``
    is ``sqrt(c (1-c) / n)`` for the reader who wants a band.
    """
    ranks = np.asarray(ranks, dtype=float)
    n, n_params = ranks.shape
    names = list(param_names) if param_names is not None else [f'theta_{i}' for i in range(n_params)]
    u = ranks / n_posterior_samples                   # position of the truth in the posterior, 0..1
    rows = []
    for c in levels:
        lo, hi = (1 - c) / 2, (1 + c) / 2
        inside = (u >= lo) & (u <= hi)
        for j, name in enumerate(names):
            rows.append({'param': name, 'nominal': float(c),
                         'empirical': float(inside[:, j].mean()) if n else np.nan, 'n': int(n)})
    return pd.DataFrame(rows)


def compute_sbc_ranks(net, n_draws: int = 1000, n_sessions: int = 5, T=None,
                      n_posterior_samples: int = 1000, distribution: str | None = None,
                      seed: int | None = None, levels: Sequence[float] = SBC_COVERAGE_LEVELS,
                      quantiles: Sequence[float] = SBC_POSTERIOR_QUANTILES,
                      show_progress: bool = True, max_candidates: int = 500) -> Dict[str, Any]:
    """SBC ranks + coverage for one trained ``AmortisedSBI``.

    Args:
        net: trained network (its ``model``, ``T``, ``burn_in``, ``dist_schedule`` are used
            unless overridden).
        n_draws: prior draws (each one simulated animal).
        n_sessions: sessions per draw: 5 is an animal's full data, 2 and 3 the CV halves.
        T: trials per session, int or (lo, hi) range; default the net's training T.
        n_posterior_samples: L, samples per conditioning (ranks are 0..L).
        distribution: session distribution; default the net's schedule.
        seed: base seed, default VALIDATION_SEED (disjoint from every training simulation's
            seed). All ``n_draws`` prior draws come from ONE stream (``default_rng(seed)``);
            draw ``i``'s simulation uses ``default_rng(seed + 1 + i)``. (Taking each truth as
            the first output of a fresh sequential seed is NOT uniform enough at n~300 -- KS
            p~0.005 in a check -- which would show up as a false miscalibration.)
        quantiles: posterior quantiles stored per draw.
        max_candidates: most prior draws tried per kept draw, on average, under the accuracy rule.
    """
    from scipy.stats import kstest

    seed = VALIDATION_SEED if seed is None else int(seed)
    model = net.model
    T = net.T if T is None else T
    burn_in = int(getattr(net, 'burn_in', 1000))
    dist = distribution or (net.dist_schedule if isinstance(net.dist_schedule, str) else net.dist_schedule[0])
    prior_name = getattr(net, 'prior_name', None) or 'manuscript'
    min_accuracy = getattr(net, 'min_accuracy', None)
    key = model.value
    lo, hi = priors.flat_bounds(key, prior_name)
    names = list(net.param_names)

    q_levels = np.asarray(quantiles, dtype=float)
    theta_rng = np.random.default_rng(seed)             # one stream for every truth
    ranks, theta_true, theta_med, n_skipped, n_rejected = [], [], [], 0, 0
    theta_mean, theta_sd, theta_q, acceptance, budget_hit = [], [], [], [], []
    n_kept, candidate = 0, 0
    while n_kept < n_draws:
        if candidate >= max_candidates * n_draws:
            raise RuntimeError(f'{key.upper()}: {n_kept}/{n_draws} draws passed the accuracy rule '
                               f'(>= {min_accuracy}) after {candidate} candidates')
        theta = priors.sample_prior(key, prior_name, 1, theta_rng)[1][0]
        rng = np.random.default_rng(seed + 1 + candidate)
        candidate += 1
        sessions = simulate_sessions(model, theta_to_params(theta, model, prior_name),
                                     [dist] * n_sessions, T, burn_in, rng, min_accuracy=min_accuracy)
        if sessions is None:
            n_rejected += 1
            continue
        n_kept += 1
        try:
            cond = net.condition(sessions, n_samples=n_posterior_samples)
        except ValueError:
            n_skipped += 1
            continue
        samples = priors.to_flat(key, prior_name, np.asarray(cond['posterior_samples'], dtype=float))
        theta = priors.to_flat(key, prior_name, theta)
        if samples.shape[0] != n_posterior_samples:
            raise RuntimeError(f'expected {n_posterior_samples} posterior samples, got {samples.shape}')
        ranks.append((samples < theta[None, :]).sum(axis=0))
        theta_true.append(theta)
        theta_med.append(np.median(samples, axis=0))
        theta_mean.append(samples.mean(axis=0))
        theta_sd.append(samples.std(axis=0, ddof=1))
        theta_q.append(np.quantile(samples, q_levels, axis=0))       # (n_q, n_params)
        acceptance.append(float(cond.get('acceptance_rate', np.nan)))
        budget_hit.append(bool(cond.get('budget_hit', False)))
        if show_progress and n_kept % 25 == 0:
            print(f'  sbc {model.value.upper()} {n_kept}/{n_draws}')

    n_p = len(names)
    ranks = np.asarray(ranks, dtype=int).reshape(-1, n_p)
    theta_true = np.asarray(theta_true, dtype=float).reshape(-1, n_p)
    theta_med = np.asarray(theta_med, dtype=float).reshape(-1, n_p)
    theta_mean = np.asarray(theta_mean, dtype=float).reshape(-1, n_p)
    theta_sd = np.asarray(theta_sd, dtype=float).reshape(-1, n_p)
    theta_q = np.asarray(theta_q, dtype=float).reshape(-1, len(q_levels), n_p)
    if len(ranks):
        ks = [float(kstest(ranks[:, j] / (n_posterior_samples + 1), 'uniform').pvalue)
              for j in range(len(names))]
    else:
        ks = [np.nan] * len(names)

    return {
        'model': model.value.upper(),
        'mode': net.mode,
        'ranks': ranks,
        'n_posterior_samples': int(n_posterior_samples),
        'param_names': names,
        'ks_pvalues': ks,
        'theta_true': theta_true,
        'theta_median': theta_med,
        'theta_mean': theta_mean,
        'theta_sd': theta_sd,
        'theta_quantiles': theta_q,
        'quantile_levels': [float(q) for q in q_levels],
        'acceptance_rate': np.asarray(acceptance, dtype=float),
        'budget_hit': np.asarray(budget_hit, dtype=bool),
        'design_features': list(getattr(net, 'design_features', ()) or ()),
        'coverage': coverage_from_ranks(ranks, n_posterior_samples, levels, names),
        'prior_bounds': {n: (float(lo[j]), float(hi[j])) for j, n in enumerate(names)},
        'prior': prior_name,
        'log_uniform': list(priors.log_uniform_params(key, prior_name)),
        'min_accuracy': min_accuracy,
        'n_draws': int(n_draws),
        'n_skipped': int(n_skipped),
        'n_rejected': int(n_rejected),
        'n_sessions': int(n_sessions),
        'T': T if isinstance(T, int) else tuple(int(t) for t in T),
        'distribution': dist,
        'seed': int(seed),
    }
