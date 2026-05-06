"""
Assignment Strip Plot

Reusable coloured-grid visualisation for model assignment across
animals and methods. Handles both real-data (with significance) and
synthetic (with ground truth) variants.

Usage:
    from plotting.assignment import plot_assignment_strip

    # Real data: significance-based opacity
    plot_assignment_strip(assign_df, mode='real')

    # Synthetic: ground truth comparison
    plot_assignment_strip(strip_df, mode='synthetic', truth_col='true')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import Optional, List, Tuple

from behav_utils.plotting.styles import COLOURS

BE_COL = COLOURS['BE']
SC_COL = COLOURS['SC']
COLOUR_MAP = {'BE': BE_COL, 'SC': SC_COL}


def plot_assignment_strip(
    df: pd.DataFrame,
    mode: str = 'real',
    method_cols: Optional[List[str]] = None,
    id_col: str = 'id',
    truth_col: Optional[str] = None,
    alpha: float = 0.05,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot a coloured assignment strip for multiple animals × methods.

    Parameters
    ----------
    df : DataFrame
        One row per animal. Must contain `id_col` and method columns.
    mode : 'real' or 'synthetic'
        'real': uses p-value columns ({method}_p) for significance opacity,
            shows 'Consensus' as last column.
        'synthetic': uses `truth_col` as first column, marks wrong
            assignments with ✕, shows accuracy at bottom.
    method_cols : list of str, optional
        Columns to display. Auto-detected if not given:
        'real' → ['GS-UM', 'GS-CP', 'SBI-UM', 'SBI-CP']
        'synthetic' → all columns except id_col and truth_col
    id_col : str
        Column with animal identifiers.
    truth_col : str, optional
        Column with ground truth (required for mode='synthetic').
    alpha : float
        Significance threshold for 'real' mode.
    figsize : tuple, optional
        Figure size. Auto-calculated if not given.
    title : str, optional
        Suptitle.
    ax : Axes, optional
        Existing axes. Creates new figure if not given.

    Returns
    -------
    plt.Figure
    """
    if mode == 'synthetic':
        return _plot_synthetic_strip(
            df, method_cols=method_cols, id_col=id_col,
            truth_col=truth_col or 'true',
            figsize=figsize, title=title, ax=ax,
        )
    else:
        return _plot_real_strip(
            df, method_cols=method_cols, id_col=id_col,
            alpha=alpha, figsize=figsize, title=title, ax=ax,
        )


# ── Real data strip ─────────────────────────────────────────────────────

def _plot_real_strip(
    df, method_cols=None, id_col='id', alpha=0.05,
    figsize=None, title=None, ax=None,
):
    """Strip for real data: significance-based opacity + Consensus column."""
    if method_cols is None:
        method_cols = ['GS-UM', 'GS-CP', 'SBI-UM', 'SBI-CP']
    method_cols = [c for c in method_cols if c in df.columns]

    all_cols = method_cols + (['Consensus'] if 'Consensus' in df.columns else [])
    n_methods = len(all_cols)
    n_animals = len(df)

    if figsize is None:
        figsize = (1.5 * n_methods + 2, max(6, n_animals * 0.38))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for i, (_, row) in enumerate(df.iterrows()):
        for j, col in enumerate(all_cols):
            val = row.get(col)
            p_val = row.get(f'{col}_p', np.nan) if col != 'Consensus' else np.nan

            if val in ('BE', 'SC'):
                fc = COLOUR_MAP[val]
                if col == 'Consensus':
                    a = 0.9
                elif pd.notna(p_val) and p_val < alpha:
                    a = 0.9
                else:
                    a = 0.25
                rect = plt.Rectangle(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    facecolor=fc, alpha=a, edgecolor='black', lw=0.5,
                )
                ax.add_patch(rect)
                label = val
                if col != 'Consensus' and pd.notna(p_val):
                    label += f'\n{p_val:.3f}'
                ax.text(j, i, label, ha='center', va='center',
                        fontsize=7, fontweight='bold',
                        color='white' if a > 0.5 else 'black')
            elif val == 'Split':
                ax.add_patch(plt.Rectangle(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    facecolor='#FFE0B2', edgecolor='black', lw=0.5,
                ))
                ax.text(j, i, 'Split', ha='center', va='center',
                        fontsize=7, fontweight='bold', color='#E65100')
            elif val == 'Unclear':
                ax.add_patch(plt.Rectangle(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    facecolor='#F0F0F0', edgecolor='#CCC', lw=0.5,
                ))
                ax.text(j, i, '?', ha='center', va='center',
                        fontsize=9, color='#999')
            elif isinstance(val, str) and 'inconclusive' in val.lower():
                ax.add_patch(plt.Rectangle(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    facecolor='#D4D4D4', edgecolor='#CCC', lw=0.5,
                ))
                ax.text(j, i, '?', ha='center', va='center',
                        fontsize=9, fontweight='bold', color='#666')
            else:
                ax.add_patch(plt.Rectangle(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    facecolor='#F8F8F8', edgecolor='#DDD', lw=0.5,
                ))
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=9, color='#BBB')

    # Separator before Consensus
    if 'Consensus' in all_cols:
        ax.axvline(n_methods - 1.5, color='black', lw=2.5)

    ax.set_xlim(-0.6, n_methods - 0.4)
    ax.set_ylim(-0.6, n_animals - 0.4)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(all_cols, fontsize=10, fontweight='bold')
    ax.xaxis.set_ticks_position('top')
    ax.set_yticks(range(n_animals))
    ax.set_yticklabels(df[id_col].values, fontsize=8)
    ax.invert_yaxis()

    legend_elements = [
        Patch(facecolor=BE_COL, alpha=0.9, label=f'BE (p<{alpha})'),
        Patch(facecolor=BE_COL, alpha=0.25, label='BE (not sig.)'),
        Patch(facecolor=SC_COL, alpha=0.9, label=f'SC (p<{alpha})'),
        Patch(facecolor=SC_COL, alpha=0.25, label='SC (not sig.)'),
        Patch(facecolor='#FFE0B2', edgecolor='#E65100', label='Split'),
        Patch(facecolor='#D4D4D4', label='Inconclusive'),
        Patch(facecolor='#F0F0F0', label='No data'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(1.02, 1), fontsize=8)

    if title is None:
        cons = df.get('Consensus')
        if cons is not None:
            counts = cons.value_counts()
            cons_str = ', '.join(f'{k}: {v}' for k, v in counts.items())
            title = f'Model Assignment — All Animals\n{cons_str}'
        else:
            title = 'Model Assignment'

    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    if own_fig:
        plt.tight_layout()
    return fig


# ── Synthetic strip ──────────────────────────────────────────────────────

def _plot_synthetic_strip(
    df, method_cols=None, id_col='id', truth_col='true',
    figsize=None, title=None, ax=None,
):
    """Strip for synthetic data: ground truth + ✕ for wrong assignments."""
    from matplotlib.colors import ListedColormap

    if method_cols is None:
        exclude = {id_col, truth_col}
        method_cols = [c for c in df.columns if c not in exclude]

    all_cols = [truth_col] + method_cols
    n_m = len(all_cols)
    n_a = len(df)

    if figsize is None:
        figsize = (1.2 * n_m + 1, max(6, n_a * 0.32))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    cm = {'BE': 0, 'SC': 1}
    data_mat = np.full((n_a, n_m), np.nan)
    corr_mat = np.ones((n_a, n_m), dtype=bool)

    for i, (_, row) in enumerate(df.iterrows()):
        for j, col in enumerate(all_cols):
            val = row.get(col)
            if isinstance(val, str) and val in cm:
                data_mat[i, j] = cm[val]
                if col != truth_col:
                    corr_mat[i, j] = (val == row[truth_col])

    cmap = ListedColormap([BE_COL, SC_COL])
    ax.imshow(data_mat, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    for i in range(n_a):
        for j in range(n_m):
            if j == 0:
                continue
            val = df.iloc[i][all_cols[j]]
            if not corr_mat[i, j] and not np.isnan(data_mat[i, j]):
                ax.plot(j, i, 'x', color='white', ms=10, mew=2.5)
            elif np.isnan(data_mat[i, j]):
                fc = '#F0F0F0'
                label = '—' if pd.isna(val) else '?'
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=fc, edgecolor='#CCC', lw=0.5,
                ))
                ax.text(j, i, label, ha='center', va='center',
                        fontsize=9, color='#666', fontweight='bold')

    for j in range(n_m + 1):
        ax.axvline(j - 0.5, color='white', lw=2)
    ax.axvline(0.5, color='black', lw=2.5)

    ax.set_xticks(range(n_m))
    col_labels = ['True'] + method_cols
    ax.set_xticklabels(col_labels, fontsize=10, fontweight='bold')
    ax.xaxis.set_ticks_position('top')
    ax.set_yticks(range(n_a))
    ax.set_yticklabels(df[id_col].values, fontsize=7)

    # Accuracy per method at bottom
    for j, col in enumerate(all_cols):
        if col == truth_col:
            continue
        valid = ~np.isnan(data_mat[:, j])
        if valid.sum() > 0:
            acc = corr_mat[valid, j].mean()
            ax.text(j, n_a + 0.3, f'{acc:.0%}',
                    ha='center', fontsize=9, fontweight='bold')

    ax.legend(
        handles=[
            Patch(facecolor=BE_COL, label='BE'),
            Patch(facecolor=SC_COL, label='SC'),
            Line2D([0], [0], marker='x', color='white',
                   markeredgecolor='white', ms=10, mew=2.5,
                   ls='', label='Wrong'),
        ],
        loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9,
    )

    if title is None:
        title = 'Model Assignment vs Ground Truth'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)

    if own_fig:
        plt.tight_layout()
    return fig
