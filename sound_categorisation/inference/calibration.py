"""Simulation-based calibration (SBC) of a trained amortised network.

For ``n_draws`` fresh parameter sets from the prior: simulate ``n_sessions`` x ``T`` trials
(the same ``simulate_sessions`` the training simulator uses), condition the network on them
exactly as on an animal, draw ``n_posterior_samples``, and record for each parameter how many
samples lie below the true value (the rank, 0..L). A calibrated posterior gives ranks that are
uniform over the draws (Talts et al. 2018). Two readouts come from the same ranks:

    ranks         -> plot_sbc_ranks / plot_sbc_ecdf (histogram, ECDF with KS band)
    coverage      -> at each level c, the fraction of draws whose true value lies inside the
                     central c interval of its posterior (the rank lies within L*(1-c)/2 ..
                     L*(1+c)/2), against the nominal c

``compute_sbc_ranks`` returns the dict the plotters in ``plotting/sbi_validation.py`` expect,
plus ``theta_true``, ``theta_median`` (a free recovery check), ``coverage`` and ``prior_bounds``.
Degenerate draws (the observation has a non-finite stat the net cannot impute) are skipped and
counted in ``n_skipped``.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from sound_categorisation.inference.simulator import get_bounds_arrays, simulate_sessions, theta_to_params
from sound_categorisation.settings import SBC_COVERAGE_LEVELS


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


def compute_sbc_ranks(net, n_draws: int = 300, n_sessions: int = 5, T=None,
                      n_posterior_samples: int = 1000, distribution: str | None = None,
                      seed: int = 0, levels: Sequence[float] = SBC_COVERAGE_LEVELS,
                      show_progress: bool = True) -> Dict[str, Any]:
    """SBC ranks + coverage for one trained ``AmortisedSBI``.

    Args:
        net: trained network (its ``model``, ``T``, ``burn_in``, ``dist_schedule`` are used
            unless overridden).
        n_draws: prior draws (each one simulated animal).
        n_sessions: sessions per draw (the networks train on 4-6; 5 is the synthetic cohort).
        T: trials per session, int or (lo, hi) range; default the net's training T.
        n_posterior_samples: L, samples per conditioning (ranks are 0..L).
        distribution: session distribution; default the net's schedule.
        seed: base seed. All ``n_draws`` prior draws come from ONE stream
            (``default_rng(seed)``); draw ``i``'s simulation uses ``default_rng(seed + 1 + i)``.
            (Taking each truth as the first output of a fresh sequential seed is NOT uniform
            enough at n~300 -- KS p~0.005 in a check -- which would show up as a false
            miscalibration.)
    """
    from scipy.stats import kstest

    model = net.model
    T = net.T if T is None else T
    burn_in = int(getattr(net, 'burn_in', 1000))
    dist = distribution or (net.dist_schedule if isinstance(net.dist_schedule, str) else net.dist_schedule[0])
    lo, hi = get_bounds_arrays(model)
    names = list(net.param_names)

    thetas = np.random.default_rng(seed).uniform(lo, hi, size=(n_draws, len(names)))
    ranks, theta_true, theta_med, n_skipped = [], [], [], 0
    for i in range(n_draws):
        theta = thetas[i]
        rng = np.random.default_rng(seed + 1 + i)
        sessions = simulate_sessions(model, theta_to_params(theta, model), [dist] * n_sessions,
                                     T, burn_in, rng)
        try:
            cond = net.condition(sessions, n_samples=n_posterior_samples)
        except ValueError:
            n_skipped += 1
            continue
        samples = np.asarray(cond['posterior_samples'], dtype=float)
        if samples.shape[0] != n_posterior_samples:
            raise RuntimeError(f'expected {n_posterior_samples} posterior samples, got {samples.shape}')
        ranks.append((samples < theta[None, :]).sum(axis=0))
        theta_true.append(theta)
        theta_med.append(np.median(samples, axis=0))
        if show_progress and (i + 1) % 25 == 0:
            print(f'  sbc {model.value.upper()} {i + 1}/{n_draws}')

    ranks = np.asarray(ranks, dtype=int).reshape(-1, len(names))
    theta_true = np.asarray(theta_true, dtype=float).reshape(-1, len(names))
    theta_med = np.asarray(theta_med, dtype=float).reshape(-1, len(names))
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
        'coverage': coverage_from_ranks(ranks, n_posterior_samples, levels, names),
        'prior_bounds': {n: (float(lo[j]), float(hi[j])) for j, n in enumerate(names)},
        'n_draws': int(n_draws),
        'n_skipped': int(n_skipped),
        'n_sessions': int(n_sessions),
        'T': T if isinstance(T, int) else tuple(int(t) for t in T),
        'distribution': dist,
        'seed': int(seed),
    }
