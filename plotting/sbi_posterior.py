"""
SBI Posterior Plotting

Marginal posterior distributions, corner/pairplots, and psychometric
curve overlays from posterior predictive samples.

Usage:
    from plotting.sbi_posteriors import (
        plot_marginal_posteriors,
        plot_pairplot,
        plot_posterior_psychometric,
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

from plotting._style import PARAM_COLOURS


# =============================================================================
# MARGINAL POSTERIORS
# =============================================================================

def plot_marginal_posteriors(
    posterior_samples: np.ndarray,
    param_names: List[str],
    true_theta: Optional[np.ndarray] = None,
    figsize: Optional[Tuple[float, float]] = None,
    n_bins: int = 40,
) -> plt.Figure:
    """Marginal posterior histogram per parameter.

    Static design: one posterior per conditioning, so one marginal histogram
    per parameter (no per-session panels).

    Args:
        posterior_samples: (n_samples, n_params), from
            AmortisedSBI.condition()['posterior_samples'].
        param_names: Parameter names, length == n_params
            (condition()['param_names']).
        true_theta: Optional (n_params,) ground-truth values (for synthetic
            recovery checks); drawn as a dashed line per panel.
        figsize: Figure size.
        n_bins: Histogram bins.

    Returns:
        Matplotlib figure (one panel per parameter).
    """
    samples = np.asarray(posterior_samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError(
            f"posterior_samples must be 2-D (n_samples, n_params); "
            f"got shape {samples.shape}")
    n_params = samples.shape[1]
    if len(param_names) != n_params:
        raise ValueError(
            f"{len(param_names)} param_names but posterior_samples has "
            f"{n_params} columns")

    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    if figsize is None:
        figsize = (4 * n_cols, 3.2 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    true_theta = None if true_theta is None else np.atleast_1d(true_theta)

    for i, name in enumerate(param_names):
        ax = axes_flat[i]
        colour = PARAM_COLOURS.get(name, f'C{i}')
        col = samples[:, i]
        col = col[np.isfinite(col)]

        if col.size:
            ax.hist(col, bins=n_bins, color=colour, alpha=0.75,
                    edgecolor='white', density=True)
            med = float(np.median(col))
            ax.axvline(med, color=colour, linewidth=1.5,
                       label=f'median {med:.3g}')

        if true_theta is not None and i < len(true_theta):
            ax.axvline(true_theta[i], color='k', linestyle='--', linewidth=2,
                       label=f'true {true_theta[i]:.3g}')

        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(name)
        ax.legend(fontsize=8)

    for j in range(n_params, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    return fig


# =============================================================================
# CORNER / PAIRPLOT
# =============================================================================

def plot_pairplot(
    samples: np.ndarray,
    param_names: List[str],
    ground_truth: Optional[np.ndarray] = None,
    figsize: Optional[Tuple[float, float]] = None,
    n_bins: int = 30,
    max_params: int = 8,
) -> plt.Figure:
    """
    Corner plot (pairwise scatter + marginals) for posterior samples.

    Args:
        samples: (n_samples, n_params) posterior samples.
        param_names: Parameter names matching columns.
        ground_truth: Optional true parameter values.
        figsize: Figure size.
        n_bins: Bins for marginal histograms.
        max_params: Maximum number of params to include.

    Returns:
        Matplotlib figure.
    """
    n_params = min(len(param_names), max_params, samples.shape[1])
    samples = samples[:, :n_params]
    names = param_names[:n_params]

    if figsize is None:
        figsize = (2.5 * n_params, 2.5 * n_params)

    fig, axes = plt.subplots(n_params, n_params, figsize=figsize)

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            if i == j:
                ax.hist(samples[:, i], bins=n_bins, color='steelblue',
                        alpha=0.7, edgecolor='white', density=True)
                if ground_truth is not None and i < len(ground_truth):
                    ax.axvline(ground_truth[i], color='red', linewidth=2)

            elif i > j:
                ax.scatter(samples[:, j], samples[:, i], alpha=0.05,
                           s=1, color='steelblue')
                if ground_truth is not None:
                    if j < len(ground_truth) and i < len(ground_truth):
                        ax.axvline(ground_truth[j], color='red',
                                   linewidth=1, alpha=0.7)
                        ax.axhline(ground_truth[i], color='red',
                                   linewidth=1, alpha=0.7)
            else:
                ax.set_visible(False)

            if i == n_params - 1:
                ax.set_xlabel(names[j], fontsize=8)
            else:
                ax.set_xticklabels([])

            if j == 0 and i != 0:
                ax.set_ylabel(names[i], fontsize=8)
            elif j != 0:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=6)

    fig.tight_layout()
    return fig


# =============================================================================
# PSYCHOMETRIC OVERLAY
# =============================================================================

def plot_posterior_psychometric(
    stimuli_per_session: List[np.ndarray],
    choices_per_session: List[np.ndarray],
    posterior_choices: Optional[List[List[np.ndarray]]] = None,
    sessions_to_show: Optional[List[int]] = None,
    n_bins: int = 8,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay observed psychometric curves with posterior predictive samples.

    Shows binned data points, fitted cumulative Gaussian, and (optionally)
    posterior predictive CI band.

    Args:
        stimuli_per_session: List of stimulus arrays.
        choices_per_session: List of observed choice arrays.
        posterior_choices: List (sessions) of lists (samples) of choice arrays.
        sessions_to_show: Which sessions to plot. Default: up to 9.
        n_bins: Bins for psychometric curves.
        figsize: Figure size.
        title: Overall title.

    Returns:
        Matplotlib figure.
    """
    from behav_utils.analysis.psychometry import fit_psychometric

    n_sessions = len(stimuli_per_session)

    if sessions_to_show is None:
        if n_sessions <= 9:
            sessions_to_show = list(range(n_sessions))
        else:
            step = max(1, n_sessions // 9)
            sessions_to_show = list(range(0, n_sessions, step))[:9]

    n_show = len(sessions_to_show)
    n_cols = min(3, n_show)
    n_rows = int(np.ceil(n_show / n_cols))

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    x_fine = np.linspace(-1, 1, 200)

    for idx, s in enumerate(sessions_to_show):
        ax = axes_flat[idx]
        stim = stimuli_per_session[s]
        choices = choices_per_session[s]

        valid = ~np.isnan(choices)
        stim_v = stim[valid]
        choices_v = choices[valid].astype(int)

        # Bin observed data
        bin_edges = np.linspace(stim_v.min() - 0.01,
                                stim_v.max() + 0.01, n_bins + 1)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_idx = np.digitize(stim_v, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        obs_prob = np.zeros(n_bins)
        obs_count = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                obs_prob[b] = choices_v[mask].mean()
                obs_count[b] = mask.sum()
            else:
                obs_prob[b] = np.nan

        # Posterior predictive curves
        if posterior_choices is not None and s < len(posterior_choices):
            pred_curves = []
            for pc in posterior_choices[s]:
                pc_v = pc[valid].astype(int)
                pred_prob = np.zeros(n_bins)
                for b in range(n_bins):
                    mask = bin_idx == b
                    if mask.sum() > 0:
                        pred_prob[b] = pc_v[mask].mean()
                    else:
                        pred_prob[b] = np.nan
                pred_curves.append(pred_prob)

            pred_curves = np.array(pred_curves)
            pred_mean = np.nanmean(pred_curves, axis=0)
            pred_lo = np.nanpercentile(pred_curves, 2.5, axis=0)
            pred_hi = np.nanpercentile(pred_curves, 97.5, axis=0)

            ax.fill_between(bin_centres, pred_lo, pred_hi,
                            alpha=0.2, color='steelblue',
                            label='95% pred. CI')
            ax.plot(bin_centres, pred_mean, '-', color='steelblue',
                    linewidth=1.5, label='Pred. mean')

        # Fit cumulative Gaussian to observed data
        fit_result = fit_psychometric(stim_v, choices_v.astype(float),
                                       x_eval=x_fine)
        if fit_result.get('success', False):
            ax.plot(fit_result['x_fit'], fit_result['y_fit'], '-',
                    color='k', linewidth=2, alpha=0.8, label='Fit')

        # Observed data points
        valid_bins = obs_count > 0
        sizes = np.clip(obs_count[valid_bins] * 3, 20, 150)
        ax.scatter(bin_centres[valid_bins], obs_prob[valid_bins],
                   s=sizes, color='k', zorder=5, label='Observed')

        ax.axhline(0.5, color='grey', linestyle=':', alpha=0.5)
        ax.axvline(0, color='grey', linestyle=':', alpha=0.5)

        ax.set_xlim(stim_v.min() - 0.1, stim_v.max() + 0.1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Stimulus')
        ax.set_ylabel('P(choose B)')

        acc = np.mean(choices_v == (stim_v > 0).astype(int))
        if fit_result.get('success', False):
            mu = fit_result['mu']
            sigma = fit_result['sigma']
            ax.set_title(
                f'Session {s} (acc={acc:.2f}, μ={mu:.2f}, σ={sigma:.2f})')
        else:
            ax.set_title(f'Session {s} (acc={acc:.2f}, n={len(choices_v)})')

        if idx == 0:
            ax.legend(fontsize=7, loc='lower right')

    for j in range(n_show, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    fig.tight_layout()
    return fig
