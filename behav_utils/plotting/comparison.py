import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from behav_utils.plotting.styles import PALETTE, UM_CMAP


def plot_comparison(
    result: dict,
    ax: Optional[plt.Axes] = None,
    metric: str = 'psychometric',
    color_a: Optional[str] = None,
    color_b: Optional[str] = None,
    show_stats: bool = True,
    title: str = '',
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a two-condition comparison from a compute_comparison() result.

    Three metric modes:
        'psychometric' — overlaid psychometric curves (requires fit params)
        'accuracy'     — bar chart of accuracy ± CI
        'um'           — side-by-side update matrices + difference

    Args:
        result: Dict from compute_comparison().
        ax: Matplotlib axes (creates one if None). For 'um' mode, creates 3 axes.
        metric: What to plot.
        color_a, color_b: Colours for conditions A and B.
        show_stats: Annotate p-values and effect sizes.
        title: Axes title.

    Returns:
        (fig, ax) or (fig, axes) for 'um' mode.
    """
    color_a = color_a or PALETTE[0]
    color_b = color_b or PALETTE[1]
    label_a = result.get('label_a', 'A')
    label_b = result.get('label_b', 'B')

    if metric == 'psychometric':
        return _plot_comparison_psychometric(
            result, ax, color_a, color_b, label_a, label_b, show_stats, title)

    elif metric == 'accuracy':
        return _plot_comparison_accuracy(
            result, ax, color_a, color_b, label_a, label_b, show_stats, title)

    elif metric == 'um':
        return _plot_comparison_um(
            result, color_a, color_b, label_a, label_b, title)

    else:
        raise ValueError(f"Unknown metric '{metric}'. Use 'psychometric', 'accuracy', or 'um'.")


def _plot_comparison_psychometric(result, ax, color_a, color_b,
                                   label_a, label_b, show_stats, title):
    """Overlaid psychometric curves from comparison params."""
    from behav_utils.analysis.utils import cumulative_gaussian

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    else:
        fig = ax.get_figure()

    x = np.linspace(-1, 1, 200)

    for params, color, label in [
        (result['params_a'], color_a, label_a),
        (result['params_b'], color_b, label_b),
    ]:
        pse = params.get('pse')
        slope = params.get('slope')
        if pse is not None and slope is not None and not np.isnan(pse):
            y = cumulative_gaussian(x, pse, slope,
                                     params.get('lapse_low', 0),
                                     params.get('lapse_high', 0))
            ax.plot(x, y, color=color, lw=2, label=label)

    ax.axhline(0.5, ls='--', color='grey', alpha=0.3)
    ax.axvline(0.0, ls='--', color='grey', alpha=0.3)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('P(choose B)')
    ax.legend(fontsize=9)

    if show_stats:
        diffs = result.get('diffs', {})
        perm_p = result.get('perm_p', {})
        pse_diff = diffs.get('pse', np.nan)
        pse_p = perm_p.get('pse', np.nan) if perm_p else np.nan
        text = f'\u0394PSE = {pse_diff:.3f}'
        if not np.isnan(pse_p):
            text += f'\np = {pse_p:.3f}'
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#ccc', alpha=0.9))

    if title:
        ax.set_title(title)
    return fig, ax


def _plot_comparison_accuracy(result, ax, color_a, color_b,
                               label_a, label_b, show_stats, title):
    """Bar chart of accuracy for two conditions."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    else:
        fig = ax.get_figure()

    acc_a = result['params_a'].get('accuracy', np.nan)
    acc_b = result['params_b'].get('accuracy', np.nan)

    bars = ax.bar([0, 1], [acc_a, acc_b], color=[color_a, color_b],
                  width=0.6, edgecolor='white')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls='--', color='grey', alpha=0.3)

    if show_stats:
        fisher_p = result.get('fisher_p', np.nan)
        if not np.isnan(fisher_p):
            ax.text(0.5, max(acc_a, acc_b) + 0.03, f'p = {fisher_p:.3f}',
                    ha='center', fontsize=9)

    if title:
        ax.set_title(title)
    return fig, ax


def _plot_comparison_um(result, color_a, color_b, label_a, label_b, title):
    """Side-by-side UMs + difference."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))

    um_a = result.get('um_a')
    um_b = result.get('um_b')
    um_diff = result.get('um_diff')

    if um_a is None or um_b is None:
        return fig, axes

    vmax = max(np.nanmax(np.abs(um_a)), np.nanmax(np.abs(um_b)), 0.01)

    for ax, um, label in [
        (axes[0], um_a, label_a),
        (axes[1], um_b, label_b),
    ]:
        im = ax.imshow(um, cmap=UM_CMAP, vmin=-vmax, vmax=vmax,
                       origin='lower', aspect='equal')
        ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.046)

    if um_diff is not None:
        vmax_d = max(np.nanmax(np.abs(um_diff)), 0.01)
        im = axes[2].imshow(um_diff, cmap=UM_CMAP, vmin=-vmax_d, vmax=vmax_d,
                            origin='lower', aspect='equal')
        rmse = result.get('um_rmse', np.nan)
        axes[2].set_title(f'{label_a} \u2212 {label_b}\n(RMSE={rmse:.3f})')
        plt.colorbar(im, ax=axes[2], fraction=0.046)

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes