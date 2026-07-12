"""
plotting/sbi_validation.py — Visualisations for SBI validation.

Each plotter consumes a result dict from validation.sbi:

    plot_sbc_ranks(sbc_result)         ← compute_sbc_ranks
    plot_sbc_ecdf(sbc_result)          ← compute_sbc_ranks
    plot_recovery_scatter(rec_result)  ← compute_parameter_recovery
    plot_recovery_bias(rec_result)     ← compute_parameter_recovery
    plot_param_stat_correlations(res)  ← compute_param_stat_correlations
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# SBC
# =============================================================================

def plot_sbc_ranks(
    sbc_result: Dict[str, Any],
    param_indices: Optional[List[int]] = None,
    n_bins: int = 20,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Histograms of SBC ranks, one panel per parameter.

    Reference: uniform distribution (red dashed) is well-calibrated.
    U-shape → posterior too narrow; inverted U → too wide; skew → biased.

    Args:
        sbc_result: Output of compute_sbc_ranks.
        param_indices: Subset of parameter indices to plot. Default all.
        n_bins: Histogram bins.
        figsize: Figure size.
        title: Overall figure title.
    """
    from scipy.stats import binom

    ranks = sbc_result['ranks']
    names = sbc_result.get('param_names')
    ks_pvals = sbc_result.get('ks_pvalues')

    n_params = ranks.shape[1]
    indices = param_indices if param_indices is not None else list(range(n_params))
    n_plot = len(indices)
    n_cols = min(4, n_plot)
    n_rows = int(np.ceil(n_plot / n_cols))
    if figsize is None:
        figsize = (4 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    expected = len(ranks) / n_bins

    for idx, p in enumerate(indices):
        ax = axes_flat[idx]
        ax.hist(ranks[:, p], bins=n_bins, density=False,
                color='steelblue', edgecolor='white', alpha=0.8)
        ax.axhline(expected, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.7, label='Expected (uniform)')

        # 95% binomial band for uniform expectation
        ci_lo = binom.ppf(0.025, len(ranks), 1 / n_bins)
        ci_hi = binom.ppf(0.975, len(ranks), 1 / n_bins)
        ax.axhspan(ci_lo, ci_hi, alpha=0.1, color='red', zorder=0)

        label = names[p] if names is not None and p < len(names) else f'θ_{p}'
        ks_str = ''
        if ks_pvals is not None:
            pval = ks_pvals[p]
            ks_str = f'\nKS p={pval:.3f}' + (' ⚠' if pval < 0.05 else '')
        ax.set_title(f'{label}{ks_str}', fontsize=9)
        ax.set_xlabel('Rank')
        ax.set_ylabel('Count')

    for j in range(n_plot, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title or 'SBC Rank Histograms', fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def plot_sbc_ecdf(
    sbc_result: Dict[str, Any],
    param_indices: Optional[List[int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    ECDF of normalised SBC ranks, one panel per parameter.

    Should lie on the diagonal if calibrated. The red shaded band is a
    KS 95% envelope; the curve leaving the band indicates miscalibration.

    Args:
        sbc_result: Output of compute_sbc_ranks.
        param_indices: Subset of params to plot.
        figsize: Figure size.
        title: Figure title.
    """
    ranks = sbc_result['ranks']
    n_post = sbc_result['n_posterior_samples']
    names = sbc_result.get('param_names')

    n_params = ranks.shape[1]
    indices = param_indices if param_indices is not None else list(range(n_params))
    n_plot = len(indices)
    n_cols = min(4, n_plot)
    n_rows = int(np.ceil(n_plot / n_cols))
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    n_sbc = len(ranks)

    for idx, p in enumerate(indices):
        ax = axes_flat[idx]
        normalised = np.sort(ranks[:, p]) / (n_post + 1)
        ecdf_y = np.arange(1, n_sbc + 1) / n_sbc

        ax.plot(normalised, ecdf_y, color='steelblue', linewidth=1.5)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=1)

        ks_crit = 1.36 / np.sqrt(n_sbc)
        ax.fill_between(
            [0, 1],
            [-ks_crit, 1 - ks_crit],
            [ ks_crit, 1 + ks_crit],
            alpha=0.1, color='red',
        )

        label = names[p] if names is not None and p < len(names) else f'θ_{p}'
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Normalised rank')
        ax.set_ylabel('ECDF')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    for j in range(n_plot, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# =============================================================================
# PARAMETER RECOVERY
# =============================================================================

def plot_recovery_scatter(
    recovery_result: Dict[str, Any],
    param_indices: Optional[List[int]] = None,
    prior_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    param_links: Optional[Dict[str, Any]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    True vs recovered scatter, one panel per parameter. Identity line shown.

    Args:
        recovery_result: Output of compute_parameter_recovery.
        param_indices: Subset of params to plot.
        prior_bounds: {name: (lo, hi)} — sets axis range. If None and
            param_links provided, extracted from link.bounds.
        param_links: {name: spec} — alternative way to set bounds.
        figsize: Figure size.
        title: Figure title.
    """
    true_p = recovery_result['true_params']
    rec    = recovery_result['recovered_median']
    ci_lo  = recovery_result['recovered_ci_low']
    ci_hi  = recovery_result['recovered_ci_high']
    names  = recovery_result['param_names']
    corrs  = recovery_result['correlation']
    cov    = recovery_result['coverage_90']

    if prior_bounds is None and param_links is not None:
        prior_bounds = {
            name: link.bounds
            for name, link in param_links.items()
            if hasattr(link, 'bounds')
        }

    n_params = true_p.shape[1]
    indices = param_indices if param_indices is not None else list(range(n_params))
    n_plot = len(indices)
    n_cols = min(4, n_plot)
    n_rows = int(np.ceil(n_plot / n_cols))
    if figsize is None:
        figsize = (4.5 * n_cols, 4.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, p in enumerate(indices):
        ax = axes_flat[idx]
        name = names[p] if p < len(names) else f'θ_{p}'

        yerr_lo = rec[:, p] - ci_lo[:, p]
        yerr_hi = ci_hi[:, p] - rec[:, p]
        ax.errorbar(
            true_p[:, p], rec[:, p],
            yerr=[yerr_lo, yerr_hi],
            fmt='o', markersize=3, alpha=0.4,
            elinewidth=0.5, capsize=0, color='steelblue',
        )

        if prior_bounds is not None and name in prior_bounds:
            lo, hi = prior_bounds[name]
            padding = (hi - lo) * 0.08
            ax_lo, ax_hi = lo - padding, hi + padding
        else:
            all_vals = np.concatenate([true_p[:, p], rec[:, p]])
            lo_d, hi_d = float(np.min(all_vals)), float(np.max(all_vals))
            padding = (hi_d - lo_d) * 0.1
            ax_lo, ax_hi = lo_d - padding, hi_d + padding

        ax.set_xlim(ax_lo, ax_hi); ax.set_ylim(ax_lo, ax_hi)
        ax.plot([ax_lo, ax_hi], [ax_lo, ax_hi], 'k--', linewidth=1, alpha=0.5)

        r = corrs[p]
        r_str = f'r={r:.2f}' if np.isfinite(r) else 'r=N/A'
        ax.set_title(f'{name}\n{r_str}, 90% cov={cov[p]:.0%}', fontsize=9)
        ax.set_xlabel('True')
        ax.set_ylabel('Recovered (median)')
        ax.set_aspect('equal')

    for j in range(n_plot, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title or 'Parameter Recovery: True vs Recovered',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def plot_recovery_bias(
    recovery_result: Dict[str, Any],
    param_indices: Optional[List[int]] = None,
    prior_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    param_links: Optional[Dict[str, Any]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Recovery error (recovered − true) vs true value, with a running mean
    overlay. Horizontal scatter around zero = unbiased.

    Args:
        recovery_result: Output of compute_parameter_recovery.
        param_indices: Subset of params.
        prior_bounds: {name: (lo, hi)} sets x-axis range.
        param_links: {name: spec} — alternative bound source.
        figsize: Figure size.
        title: Figure title.
    """
    true_p = recovery_result['true_params']
    rec    = recovery_result['recovered_median']
    names  = recovery_result['param_names']
    rmse_v = recovery_result['rmse']
    bias_v = recovery_result['bias']

    if prior_bounds is None and param_links is not None:
        prior_bounds = {
            name: link.bounds
            for name, link in param_links.items()
            if hasattr(link, 'bounds')
        }

    n_params = true_p.shape[1]
    indices = param_indices if param_indices is not None else list(range(n_params))
    n_plot = len(indices)
    n_cols = min(4, n_plot)
    n_rows = int(np.ceil(n_plot / n_cols))
    if figsize is None:
        figsize = (4.5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, p in enumerate(indices):
        ax = axes_flat[idx]
        name = names[p] if p < len(names) else f'θ_{p}'
        error = rec[:, p] - true_p[:, p]

        ax.scatter(true_p[:, p], error, s=15, alpha=0.4,
                   color='steelblue', edgecolors='none')
        ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)

        # Running mean overlay
        try:
            sort_idx = np.argsort(true_p[:, p])
            x_sorted = true_p[sort_idx, p]
            e_sorted = error[sort_idx]
            window = max(5, len(e_sorted) // 10)
            if len(e_sorted) > window:
                running = np.convolve(e_sorted, np.ones(window) / window, mode='valid')
                x_run = x_sorted[window // 2: window // 2 + len(running)]
                ax.plot(x_run, running, 'r-', linewidth=2, alpha=0.7)
        except (ValueError, IndexError):
            pass

        if prior_bounds is not None and name in prior_bounds:
            lo, hi = prior_bounds[name]
            padding = (hi - lo) * 0.08
            ax.set_xlim(lo - padding, hi + padding)

        ax.set_title(
            f'{name}\nbias={bias_v[p]:.4f}, RMSE={rmse_v[p]:.4f}',
            fontsize=9,
        )
        ax.set_xlabel('True value')
        ax.set_ylabel('Error (recovered − true)')

    for j in range(n_plot, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title or 'Parameter Recovery: Bias Analysis',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# =============================================================================
# PARAMETER ↔ STAT CORRELATIONS
# =============================================================================

def plot_param_stat_correlations(
    result: Dict[str, Any],
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = 'RdBu_r',
    annot: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of |Pearson r| between each parameter and each summary stat.

    Args:
        result: Output of compute_param_stat_correlations.
        figsize: Figure size.
        cmap: Diverging colormap.
        annot: Annotate cells with the correlation value.
        title: Figure title.
    """
    corr   = result['corr_matrix']
    params = result['param_names']
    stats  = result['stat_names_expanded']

    n_params, n_stats = corr.shape
    if figsize is None:
        figsize = (max(6, n_stats * 0.45), max(2.5, n_params * 0.45))

    fig, ax = plt.subplots(figsize=figsize)
    vmax = float(np.nanmax(np.abs(corr)))
    im = ax.imshow(corr, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Pearson r')

    ax.set_xticks(np.arange(n_stats))
    ax.set_xticklabels(stats, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(n_params))
    ax.set_yticklabels(params, fontsize=9)

    if annot:
        for i in range(n_params):
            for j in range(n_stats):
                v = corr[i, j]
                if np.isnan(v):
                    continue
                colour = 'white' if abs(v) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=7, color=colour)

    ax.set_title(title or 'Parameter ↔ Summary-Stat Correlations',
                 fontsize=11, pad=10)
    fig.tight_layout()
    return fig



# =============================================================================
# Summary-stat selection / attribution  (draw-only, single-panel, plot_x(result, ax=None))
# Consume the dicts from validation.feature_diagnostics:
#   plot_um_scalar_correlation ← um_scalar_correlation   (what scalars miss)
#   plot_stat_correlation      ← stat_correlation        (redundancy)
#   plot_individual_identity   ← stat_individual_power    (Q2: identity AUC per stat)
#   plot_individual_recovery   ← stat_individual_power    (Q3: per-param R² per stat)
#   plot_selection_curve       ← select_stats            (Q4: 0→all, identity + min-R²)
#   plot_stat_contributions    ← stat_contributions      (parked: joint importance)
#   plot_stat_sensitivity      ← stat_parameter_sensitivity (parked: isolation)
# =============================================================================

def _heatmap(ax, M, row_labels, col_labels, cmap, vmin, vmax, cbar_label,
             annot=True, annot_thresh=None, rot=45):
    im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label(cbar_label)
    ax.set_xticks(np.arange(M.shape[1]))
    ax.set_xticklabels(col_labels, rotation=rot, ha='right' if rot else 'center', fontsize=7)
    ax.set_yticks(np.arange(M.shape[0])); ax.set_yticklabels(row_labels, fontsize=7)
    if annot:
        thr = annot_thresh if annot_thresh is not None else (vmin + vmax) / 2 + (vmax - vmin) * 0.25
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6,
                            color='white' if abs(v) > thr else 'black')
    return im


def plot_um_scalar_correlation(result, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """|r| between each UM cell (rows) and each scalar stat (cols). Dark rows = UM structure no
    scalar captures. result ← um_scalar_correlation."""
    C = result['corr']
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, C.shape[1] * 0.5), max(5, C.shape[0] * 0.18)))
    _heatmap(ax, C.values, list(C.index), list(C.columns), 'viridis', 0, 1, '|r|',
             annot=False, rot=90)
    ax.set_title(f"UM cell vs scalar · {result['model_type'].upper()} · {result['distribution']}",
                 fontsize=10, pad=8)
    return ax


def plot_stat_correlation(result, ax: Optional[plt.Axes] = None, annot: bool = False) -> plt.Axes:
    """|r| stat×stat (redundancy). High off-diagonal blocks = drop all but one. result ← stat_correlation."""
    C = result['corr']; names = list(C.columns)
    if ax is None:
        n = len(names); _, ax = plt.subplots(figsize=(max(5, n * 0.4), max(4, n * 0.4)))
    _heatmap(ax, C.values, names, names, 'viridis', 0, 1, '|r|', annot=annot,
             annot_thresh=0.5, rot=90)
    ax.set_title(f"Stat–stat correlation · {result['model_type'].upper()} · {result['distribution']}",
                 fontsize=10, pad=8)
    return ax


def plot_individual_identity(result, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Q2 — each stat alone: model-identity CV AUC. result ← stat_individual_power."""
    s = result['identity'].sort_values(ascending=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, max(2.5, 0.4 * len(s))))
    ax.barh(range(len(s)), s.values, color='slateblue')
    ax.axvline(0.5, ls='--', c='grey', lw=1)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=8)
    ax.set_xlim(0.4, 1.0); ax.set_xlabel('identity CV AUC (stat alone)')
    ax.set_title(f"Individual identity power · {result['distribution']}", fontsize=10, pad=8)
    return ax


def plot_individual_recovery(result, model: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Q3 — each stat alone: per-parameter recovery CV R² (clipped at 0 for display).
    result ← stat_individual_power; model in {'be','sc'}."""
    R = result['recovery'][model]
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, R.shape[1] * 0.9), max(3, R.shape[0] * 0.45)))
    _heatmap(ax, np.clip(R.values.astype(float), 0, 1), list(R.index), list(R.columns),
             'viridis', 0, 1, 'CV R² (≥0)', annot=True, annot_thresh=0.5, rot=45)
    ax.set_title(f"Individual recovery · {model.upper()} · {result['distribution']}",
                 fontsize=10, pad=8)
    return ax


def plot_selection_curve(result, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Q4 — best min-across-params CV R² vs #stats, with identity AUC overlaid. Marks the
    selected subset. result ← select_stats."""
    curve = result['best_by_k']
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(curve['k'], curve['r2_min'], '-o', color='steelblue', ms=4, label='min-param R²')
    sel_k = len(result['selected'])
    ax.axvline(sel_k, color='red', ls='--', lw=1, alpha=0.7, label=f'selected (k={sel_k})')
    ax.set_xlabel('number of stat groups'); ax.set_ylabel('min over params (CV R²)')
    ax2 = ax.twinx()
    ax2.plot(curve['k'], curve['identity_auc'], '-s', color='darkorange', ms=3,
             alpha=0.7, label='identity AUC')
    ax2.set_ylabel('identity AUC', color='darkorange'); ax2.set_ylim(0.4, 1.02)
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax.set_title(f"Selection · {result['distribution']} · {result['method']}", fontsize=10, pad=8)
    ax.legend(fontsize=8, loc='lower right')
    return ax


def plot_stat_contributions(result, ax: Optional[plt.Axes] = None, annot: bool = True) -> plt.Axes:
    """Parked — joint permutation importance (stat × target). result ← stat_contributions."""
    df = result['contribution']; M = df.values.astype(float)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, df.shape[1] * 0.7), max(3, df.shape[0] * 0.45)))
    vmax = float(np.nanmax(np.abs(M))) or 1.0
    _heatmap(ax, M, list(df.index), list(df.columns), 'RdBu_r', -vmax, vmax,
             'Δ CV score (permutation)', annot=annot, annot_thresh=vmax * 0.5)
    ax.set_title(f"Stat contributions · {result['distribution']} · scorer={result['scorer']}",
                 fontsize=10, pad=8)
    return ax


def plot_stat_sensitivity(result, ax: Optional[plt.Axes] = None, annot: bool = True) -> plt.Axes:
    """Parked — one-at-a-time |Spearman| of each stat to each parameter. result ← stat_parameter_sensitivity."""
    S = result['sensitivity']
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, S.shape[1] * 0.9), max(3, S.shape[0] * 0.45)))
    _heatmap(ax, S.values.astype(float), list(S.index), list(S.columns), 'viridis', 0, 1,
             '|Spearman ρ|', annot=annot, annot_thresh=0.5)
    ax.set_title(f"Stat→param sensitivity · {result['model_type'].upper()} · {result['distribution']}",
                 fontsize=10, pad=8)
    return ax
