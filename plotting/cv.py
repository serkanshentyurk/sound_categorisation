"""
Grid-Search CV Plotting

Pure plotting functions for BE/SC model comparison visualisation.
All computation (seed error extraction, dataframe building, statistical
tests) has been moved to utils/cv_utils.py.

Usage:
    from utils.cv_utils import compute_gs_seed_errors, compute_cv_dataframes
    from plotting.cv import plot_cv_comparison, plot_winner_summary

    errors, best = compute_gs_seed_errors(gs_pickle)
    long_df, comp_df = compute_cv_dataframes(animal_id, be_errors, sc_errors)
    fig = plot_cv_comparison(long_df, comp_df, animal_id)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
from pathlib import Path

from behav_utils.plotting.styles import COLOURS, UM_CMAP

# Convenience aliases
BE_COLOUR = COLOURS['BE']
SC_COLOUR = COLOURS['SC']
MODEL_COLOURS = {
    'BE': BE_COLOUR,
    'SC': SC_COLOUR,
    'Inconclusive': COLOURS.get('unknown', '#95A5A6'),
}


# =============================================================================
# HELPERS
# =============================================================================

def _save_fig(fig, output_dir, prefix):
    """Save as both PNG and PDF."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f'{prefix}.png', dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / f'{prefix}.pdf', bbox_inches='tight')


# =============================================================================
# PER-ANIMAL: VIOLIN + PAIRED SCATTER
# =============================================================================

def plot_cv_comparison(
    long_df,
    comparison_df,
    animal_id: str,
    fit_target: str = 'UM',
    suptitle: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """
    Per-animal split violin + paired scatter plot (manuscript style).

    Left: half-violins for BE and SC with paired seed lines + box-plot stats.
    Right: BE vs SC test error scatter with identity line.

    Args:
        long_df: DataFrame with columns animal_id, model, seed, avg_test_error.
                 From compute_cv_dataframes().
        comparison_df: DataFrame with animal_id, winner, p_value, be_mean, sc_mean.
                       From compute_cv_dataframes().
        animal_id: Which animal to plot.
        fit_target: Label for y-axis (e.g. 'UM', 'CP').
        suptitle: Override figure title. Default: '{animal_id} — CV — {fit_target}'.
        figsize: Figure size.
        output_dir: If provided, saves PNG + PDF.

    Returns:
        Figure.
    """
    from scipy.stats import wilcoxon

    sub = long_df[long_df['animal_id'] == animal_id]
    be_sub = sub[sub['model'] == 'BE'].sort_values('seed').reset_index(drop=True)
    sc_sub = sub[sub['model'] == 'SC'].sort_values('seed').reset_index(drop=True)

    be_vals = be_sub['avg_test_error'].dropna().values
    sc_vals = sc_sub['avg_test_error'].dropna().values

    if len(be_vals) == 0 or len(sc_vals) == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, f'{animal_id}: insufficient data',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    # Match seeds for paired lines
    paired_seeds = sorted(
        set(be_sub['seed'].values) & set(sc_sub['seed'].values)
    )
    be_paired = be_sub.set_index('seed')['avg_test_error']
    sc_paired = sc_sub.set_index('seed')['avg_test_error']
    be_p = np.array([be_paired[s] for s in paired_seeds])
    sc_p = np.array([sc_paired[s] for s in paired_seeds])

    # Get stats from comparison_df
    row = comparison_df[comparison_df['animal_id'] == animal_id]
    if len(row) > 0:
        p_val = row.iloc[0]['p_value']
        winner = row.iloc[0]['winner']
    else:
        n_paired = min(len(be_vals), len(sc_vals))
        try:
            _, p_val = wilcoxon(be_vals[:n_paired], sc_vals[:n_paired])
        except ValueError:
            p_val = np.nan
        winner = 'BE' if np.mean(be_vals) < np.mean(sc_vals) else 'SC'

    p_str = f'p={p_val:.2e}' if p_val < 0.001 else f'p={p_val:.3f}'

    # ── Figure ────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    _draw_split_violins(ax1, be_vals, sc_vals, be_p, sc_p)
    ax1.set_ylabel(f'Test {fit_target} MSE', fontsize=11)
    if p_val < 0.05:
        verdict = f'{p_str}, winner={winner}'
    elif p_val < 0.1:
        verdict = f'{p_str}, marginal {winner}'
    else:
        verdict = f'{p_str}, inconclusive'
    ax1.set_title(verdict, fontsize=11)

    _draw_paired_scatter(ax2, be_p, sc_p)

    fig.suptitle(
        suptitle or f'{animal_id} — CV — {fit_target}',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()

    if output_dir:
        _save_fig(fig, output_dir, f'cv_comparison_{animal_id}')

    return fig


def _draw_split_violins(ax, be_vals, sc_vals, be_p, sc_p):
    """Half-violin plots with paired lines and box-plot summaries."""
    vp_be = ax.violinplot(
        be_vals, positions=[0], showextrema=False,
        showmedians=False, widths=0.8,
    )
    vp_sc = ax.violinplot(
        sc_vals, positions=[1], showextrema=False,
        showmedians=False, widths=0.8,
    )

    for body in vp_be['bodies']:
        m = np.mean(body.get_paths()[0].vertices[:, 0])
        body.get_paths()[0].vertices[:, 0] = np.clip(
            body.get_paths()[0].vertices[:, 0], -np.inf, m)
        body.set_facecolor(BE_COLOUR)
        body.set_alpha(0.5)
        body.set_edgecolor(BE_COLOUR)

    for body in vp_sc['bodies']:
        m = np.mean(body.get_paths()[0].vertices[:, 0])
        body.get_paths()[0].vertices[:, 0] = np.clip(
            body.get_paths()[0].vertices[:, 0], m, np.inf)
        body.set_facecolor(SC_COLOUR)
        body.set_alpha(0.5)
        body.set_edgecolor(SC_COLOUR)

    # Paired lines
    for b, s in zip(be_p, sc_p):
        ax.plot([0, 1], [b, s], color='grey', alpha=0.25, linewidth=0.7)

    # Summary stats (median + IQR + whiskers)
    for pos, vals, colour in [
        (0, be_vals, BE_COLOUR),
        (1, sc_vals, SC_COLOUR),
    ]:
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        lo, hi = np.min(vals), np.max(vals)
        ax.hlines(med, pos - 0.15, pos + 0.15, color=colour, linewidth=2.5)
        ax.vlines(pos, q25, q75, color=colour, linewidth=2)
        ax.hlines([q25, q75], pos - 0.08, pos + 0.08,
                  color=colour, linewidth=1.5)
        ax.vlines(pos, lo, q25, color=colour, linewidth=1)
        ax.vlines(pos, q75, hi, color=colour, linewidth=1)
        ax.hlines([lo, hi], pos - 0.08, pos + 0.08,
                  color=colour, linewidth=1.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['BE', 'SC'], fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _draw_paired_scatter(ax, be_p, sc_p):
    """BE vs SC scatter with identity line."""
    ax.scatter(be_p, sc_p, c='grey', alpha=0.6, s=30, edgecolors='none')

    all_v = np.concatenate([be_p, sc_p])
    hi = np.max(all_v) * 1.1
    ax.plot([0, hi], [0, hi], '--', color='grey', alpha=0.5, linewidth=1)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel('BE test error', fontsize=11)
    ax.set_ylabel('SC test error', fontsize=11)
    ax.set_title(
        f'BE={np.mean(be_p):.5f}, SC={np.mean(sc_p):.5f}', fontsize=11,
    )
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# =============================================================================
# WINNER SUMMARY
# =============================================================================

def plot_winner_summary(
    comparison_df,
    figsize: Tuple[float, float] = (4, 3),
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of winning model counts across animals."""
    if len(comparison_df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        return fig

    winner_counts = comparison_df['winner'].value_counts()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        winner_counts.index, winner_counts.values,
        color=[MODEL_COLOURS.get(w, 'grey') for w in winner_counts.index],
        edgecolor='black', linewidth=0.5,
    )
    ax.set_ylabel('Number of animals')
    ax.set_title('Best-fit model per animal')
    for bar, val in zip(bars, winner_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(val), ha='center', va='bottom', fontweight='bold',
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if output_dir:
        _save_fig(fig, output_dir, 'winner_summary')

    return fig


# =============================================================================
# UPDATE MATRIX HEATMAP
# =============================================================================

def plot_update_matrix(
    um: np.ndarray,
    title: str = '',
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap=None,
    show_colourbar: bool = True,
) -> plt.Axes:
    """
    Plot a single 8x8 update matrix as a heatmap.

    Uses UM_CMAP from behav_utils styles by default.
    """
    if cmap is None:
        cmap = UM_CMAP

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    if vmin is None:
        vmax_auto = np.nanmax(np.abs(um))
        vmin, vmax = -vmax_auto, vmax_auto

    im = ax.imshow(
        um, cmap=cmap, vmin=vmin, vmax=vmax,
        aspect='equal', interpolation='nearest',
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Previous stim bin', fontsize=9)
    ax.set_ylabel('Current stim bin', fontsize=9)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))

    if show_colourbar:
        plt.colorbar(im, ax=ax, shrink=0.8)

    return ax


def plot_um_comparison(
    emp_um: np.ndarray,
    model_ums: Dict[str, np.ndarray],
    model_errors: Dict[str, float],
    animal_id: str,
    figsize: Optional[Tuple[float, float]] = None,
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side empirical vs model update matrices."""
    n_panels = 1 + len(model_ums)
    if figsize is None:
        figsize = (5 * n_panels, 4)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    # Shared colour scale
    all_ums = [emp_um] + list(model_ums.values())
    vmax = max(np.nanmax(np.abs(v)) for v in all_ums)
    vmin = -vmax

    plot_update_matrix(
        emp_um, title=f'{animal_id} — Empirical',
        ax=axes[0], vmin=vmin, vmax=vmax, show_colourbar=False,
    )

    for idx, (name, um) in enumerate(model_ums.items(), start=1):
        err = model_errors.get(name, np.nan)
        plot_update_matrix(
            um, title=f'{name} (MSE={err:.4f})',
            ax=axes[idx], vmin=vmin, vmax=vmax,
            show_colourbar=(idx == n_panels - 1),
        )

    plt.suptitle(f'{animal_id}: Empirical vs Best-Fit Models', fontsize=13)
    plt.tight_layout()

    if output_dir:
        _save_fig(fig, output_dir, f'um_comparison_{animal_id}')

    return fig


# =============================================================================
# PARAMETER DISTRIBUTIONS
# =============================================================================

def plot_param_distributions(
    param_df,
    animal_id: str,
    figsize: Tuple[float, float] = (16, 6),
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """
    Parameter histograms across seeds for one animal.

    Top row: BE params. Bottom row: SC params.
    """
    sub = param_df[param_df['animal_id'] == animal_id]

    fig, axes = plt.subplots(2, 4, figsize=figsize)

    be_cols = ['sigma_noise', 'A_repulsion', 'eta_learning', 'eta_relax']
    sc_cols = ['sigma_noise', 'A_repulsion', 'gamma', 'sigma_update']

    be_sub = sub[sub['model'] == 'BE']
    for ax, col in zip(axes[0], be_cols):
        vals = be_sub[col].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=15, color=BE_COLOUR, alpha=0.7,
                    edgecolor='black', linewidth=0.5)
        ax.set_title(f'BE: {col}', fontsize=9)
        ax.set_xlabel(col, fontsize=8)

    sc_sub = sub[sub['model'] == 'SC']
    for ax, col in zip(axes[1], sc_cols):
        vals = sc_sub[col].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=15, color=SC_COLOUR, alpha=0.7,
                    edgecolor='black', linewidth=0.5)
        ax.set_title(f'SC: {col}', fontsize=9)
        ax.set_xlabel(col, fontsize=8)

    plt.suptitle(
        f'{animal_id}: Parameter distributions across seeds', fontsize=12,
    )
    plt.tight_layout()

    if output_dir:
        _save_fig(fig, output_dir, f'param_distributions_{animal_id}')

    return fig
