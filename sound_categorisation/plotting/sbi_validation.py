"""
plotting/sbi_validation.py — Visualisations for SBI validation and summary-stat selection.

SBC, coverage and recovery (draw-only, single-panel, ``plot_x(table, ax=None)``). Each takes the
rows of one ``sc-validation-tables`` CSV (``inference.validation``) for ONE network, session
count and parameter -- select them first, e.g.
``ecdf[(ecdf.rep == 'pooled') & (ecdf.model == 'BE') & (ecdf.n_sessions == 5) & (ecdf.param == p)]``:

    plot_sbc_ecdf_diff(ecdf_rows)      ← sbc_ecdf.csv        ECDF − uniform with the simultaneous band
    plot_coverage(coverage_rows)       ← coverage.csv        all parameters of one network
    plot_recovery_scatter(draw_rows)   ← recovery_draws.csv  truth vs posterior median, 5–95 %
    plot_contraction(draw_rows)        ← recovery_draws.csv  posterior contraction vs z-score

Feature selection (below) and plot_param_stat_correlations keep their own producers.
"""

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# SBC, COVERAGE, RECOVERY  (rows of one network / session count / parameter)
# =============================================================================


def _label(rows) -> str:
    r = rows.iloc[0]
    return f"{r['rep']} {r['model']}, {int(r['n_sessions'])} sessions"


def plot_sbc_ecdf_diff(rows, ax: plt.Axes | None = None) -> plt.Axes:
    """ECDF of the SBC fractional ranks minus the uniform CDF, with the simultaneous band.

    A calibrated network stays inside the band; a curve bowing up means the truth sits low in
    the posterior too often (the posterior is shifted upwards), a ∩-shape (positive in the
    middle) or ∪-shape means it is too wide or too narrow respectively.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 2.8))
    rows = rows.sort_values('z')
    ax.fill_between(rows['z'], rows['band_lo'], rows['band_hi'], color='grey', alpha=0.25,
                    linewidth=0, label='95 % simultaneous band')
    ax.plot(rows['z'], rows['ecdf_diff'], color='black', lw=1.2)
    ax.axhline(0, color='grey', lw=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel('fractional rank')
    ax.set_ylabel('ECDF − uniform')
    ax.set_title(f"{rows['param'].iloc[0]} ({_label(rows)})", fontsize=9)
    return ax


def plot_coverage(rows, ax: plt.Axes | None = None) -> plt.Axes:
    """Empirical against nominal coverage of central posterior intervals, one line per
    parameter, with the range a calibrated network would give at each level (grey bars)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3.4, 3.2))
    levels = rows.drop_duplicates('nominal').sort_values('nominal')
    ax.vlines(levels['nominal'], levels['band_lo'], levels['band_hi'], color='grey', lw=6,
              alpha=0.3, label='calibrated range')
    for param, g in rows.groupby('param', sort=False):
        g = g.sort_values('nominal')
        ax.plot(g['nominal'], g['empirical'], 'o-', ms=3, lw=1, label=param)
    ax.plot([0.4, 1.0], [0.4, 1.0], '--', color='grey', lw=0.8)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel('nominal coverage')
    ax.set_ylabel('empirical coverage')
    ax.set_title(_label(rows), fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    return ax


def plot_recovery_scatter(rows, ax: plt.Axes | None = None, max_points: int = 300) -> plt.Axes:
    """Posterior median against the true value for one parameter, with 5–95 % intervals
    (a fixed subsample of ``max_points`` draws for legibility; statistics use them all)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.2))
    if len(rows) > max_points:
        rows = rows.sample(max_points, random_state=0)
    ax.errorbar(rows['truth'], rows['median'],
                yerr=[rows['median'] - rows['q05'], rows['q95'] - rows['median']],
                fmt='o', ms=2, lw=0.4, color='black', ecolor='grey', alpha=0.6)
    lo, hi = float(rows['prior_lo'].iloc[0]), float(rows['prior_hi'].iloc[0])
    ax.plot([lo, hi], [lo, hi], '--', color='grey', lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"true {rows['param'].iloc[0]}")
    ax.set_ylabel('posterior median')
    ax.set_title(_label(rows), fontsize=9)
    return ax


def plot_contraction(rows, ax: plt.Axes | None = None) -> plt.Axes:
    """Posterior z-score against posterior contraction for one parameter (Schad et al. 2021):
    good recovery sits at high contraction with |z| mostly below 2."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.scatter(rows['contraction'], rows['zscore'], s=4, color='black', alpha=0.4, linewidths=0)
    for y in (-2, 2):
        ax.axhline(y, color='grey', lw=0.6, ls='--')
    ax.set_xlim(min(0.0, float(rows['contraction'].min())), 1.0)
    ax.set_xlabel('posterior contraction')
    ax.set_ylabel('posterior z-score')
    ax.set_title(f"{rows['param'].iloc[0]} ({_label(rows)})", fontsize=9)
    return ax


# =============================================================================
# PARAMETER ↔ STAT CORRELATIONS
# =============================================================================

def plot_param_stat_correlations(
    result: Dict[str, Any],
    figsize: Tuple[float, float] | None = None,
    cmap: str = 'RdBu_r',
    annot: bool = True,
    title: str | None = None,
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
# Consume the dicts from sound_categorisation.features.feature_diagnostics:
#   plot_um_scalar_correlation ← um_scalar_correlation   (what scalars miss)
#   plot_stat_correlation      ← stat_correlation        (redundancy)
#   plot_individual_identity   ← stat_individual_power    (Q2: identity AUC per stat)
#   plot_individual_recovery   ← stat_individual_power    (Q3: per-param R² per stat)
#   plot_selection_curve       ← select_stats            (Q4: 0→all, identity + min-R²)
#   plot_forward_selection     ← forward_selection       (Q5: greedy over the whole pool)
#   plot_stat_contributions    ← stat_contributions      (parked: joint importance)
#   plot_stat_sensitivity      ← stat_parameter_sensitivity (parked: isolation)
# =============================================================================

def _heatmap(ax, M, row_labels, col_labels, cmap, vmin, vmax, cbar_label,
             annot=True, annot_thresh=None, rot=45):
    im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(cbar_label)
    ax.set_xticks(np.arange(M.shape[1]))
    ax.set_xticklabels(col_labels, rotation=rot, ha='right' if rot else 'center', fontsize=7)
    ax.set_yticks(np.arange(M.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=7)
    if annot:
        thr = annot_thresh if annot_thresh is not None else (vmin + vmax) / 2 + (vmax - vmin) * 0.25
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6,
                            color='white' if abs(v) > thr else 'black')
    return im


def plot_um_scalar_correlation(result, ax: plt.Axes | None = None) -> plt.Axes:
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


def plot_stat_correlation(result, ax: plt.Axes | None = None, annot: bool = False) -> plt.Axes:
    """|r| stat×stat (redundancy). High off-diagonal blocks = drop all but one. result ← stat_correlation."""
    C = result['corr']
    names = list(C.columns)
    if ax is None:
        n = len(names)
        _, ax = plt.subplots(figsize=(max(5, n * 0.4), max(4, n * 0.4)))
    _heatmap(ax, C.values, names, names, 'viridis', 0, 1, '|r|', annot=annot,
             annot_thresh=0.5, rot=90)
    ax.set_title(f"Stat–stat correlation · {result['model_type'].upper()} · {result['distribution']}",
                 fontsize=10, pad=8)
    return ax


def plot_individual_identity(result, ax: plt.Axes | None = None) -> plt.Axes:
    """Q2 — each stat alone: model-identity CV AUC. result ← stat_individual_power."""
    s = result['identity'].sort_values(ascending=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, max(2.5, 0.4 * len(s))))
    ax.barh(range(len(s)), s.values, color='slateblue')
    ax.axvline(0.5, ls='--', c='grey', lw=1)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=8)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel('identity CV AUC (stat alone)')
    ax.set_title(f"Individual identity power · {result['distribution']}", fontsize=10, pad=8)
    return ax


def plot_individual_recovery(result, model: str, ax: plt.Axes | None = None) -> plt.Axes:
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


def plot_selection_curve(result, ax: plt.Axes | None = None) -> plt.Axes:
    """Q4 — best min-across-params CV R² vs #stats, with identity AUC overlaid. Marks the
    selected subset. result ← select_stats."""
    curve = result['best_by_k']
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(curve['k'], curve['r2_min'], '-o', color='steelblue', ms=4, label='min-param R²')
    sel_k = len(result['selected'])
    ax.axvline(sel_k, color='red', ls='--', lw=1, alpha=0.7, label=f'selected (k={sel_k})')
    ax.set_xlabel('number of stat groups')
    ax.set_ylabel('min over params (CV R²)')
    ax2 = ax.twinx()
    ax2.plot(curve['k'], curve['identity_auc'], '-s', color='darkorange', ms=3,
             alpha=0.7, label='identity AUC')
    ax2.set_ylabel('identity AUC', color='darkorange')
    ax2.set_ylim(0.4, 1.02)
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax.set_title(f"Selection · {result['distribution']} · {result['method']}", fontsize=10, pad=8)
    ax.legend(fontsize=8, loc='lower right')
    return ax


def plot_forward_selection(result, ax: plt.Axes | None = None) -> plt.Axes:
    """Q5 — forward-selection curve: identity AUC and min-param CV R² vs the number of stat
    groups, phase boundary and suggested k marked, the added stat named at each step.
    result ← forward_selection (or ``{'curve': DataFrame, 'suggested_k': int, ...}`` reloaded
    from forward_selection.csv)."""
    curve = result['curve']
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.45 * len(curve)), 4))
    ax.plot(curve['k'], curve['identity_auc'], '-s', color='darkorange', ms=4, label='identity AUC')
    ax.plot(curve['k'], curve['r2_min'], '-o', color='steelblue', ms=4, label='min-param R²')
    ax.plot(curve['k'], curve['r2_mean'], ':', color='steelblue', lw=1, label='mean-param R²')
    n_id = int((curve['phase'] == 'identity').sum())
    if 0 < n_id < len(curve):
        ax.axvspan(0.5, n_id + 0.5, color='darkorange', alpha=0.06, lw=0)
    k_sug = result.get('suggested_k')
    if k_sug:
        ax.axvline(k_sug, color='red', ls='--', lw=1, alpha=0.7, label=f'suggested (k={k_sug})')
    ax.set_xticks(curve['k'])
    ax.set_xticklabels([f"{k}\n{a}" for k, a in zip(curve['k'], curve['added'])],
                       fontsize=7, rotation=90)
    ax.set_xlabel('stat groups added (in order)')
    ax.set_ylabel('CV score')
    ax.set_ylim(min(0.0, float(curve['r2_min'].min()) - 0.05), 1.02)
    ax.set_title(f"Forward selection · {result.get('distribution', '')} · "
                 f"scorer={result.get('scorer', '')}", fontsize=10, pad=8)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    return ax


def plot_stat_contributions(result, ax: plt.Axes | None = None, annot: bool = True) -> plt.Axes:
    """Parked — joint permutation importance (stat × target). result ← stat_contributions."""
    df = result['contribution']
    M = df.values.astype(float)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, df.shape[1] * 0.7), max(3, df.shape[0] * 0.45)))
    vmax = float(np.nanmax(np.abs(M))) or 1.0
    _heatmap(ax, M, list(df.index), list(df.columns), 'RdBu_r', -vmax, vmax,
             'Δ CV score (permutation)', annot=annot, annot_thresh=vmax * 0.5)
    ax.set_title(f"Stat contributions · {result['distribution']} · scorer={result['scorer']}",
                 fontsize=10, pad=8)
    return ax


def plot_stat_sensitivity(result, ax: plt.Axes | None = None, annot: bool = True) -> plt.Axes:
    """Parked — one-at-a-time |Spearman| of each stat to each parameter. result ← stat_parameter_sensitivity."""
    S = result['sensitivity']
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, S.shape[1] * 0.9), max(3, S.shape[0] * 0.45)))
    _heatmap(ax, S.values.astype(float), list(S.index), list(S.columns), 'viridis', 0, 1,
             '|Spearman ρ|', annot=annot, annot_thresh=0.5)
    ax.set_title(f"Stat→param sensitivity · {result['model_type'].upper()} · {result['distribution']}",
                 fontsize=10, pad=8)
    return ax
