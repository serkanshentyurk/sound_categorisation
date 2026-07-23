"""
behav_utils/plotting/comparison.py — Visualise compare_phases output.

Pairs with behav_utils.analysis.comparison.compare_phases. Overlays the two
psychometric curves of one contrast, with bootstrap bands, the binned data and
the Δ / p / CI annotation.

    from behav_utils.analysis import compare_phases
    from behav_utils.plotting import plot_comparison

    r = compare_phases({'non_opto': ctrl, 'opto': opto},
                       stats=['psychometric'], reference='non_opto')
    plot_comparison(r, 'opto_vs_non_opto')

Bands are drawn when the per-phase psychometric carries ``curve_band``
(``compute_psychometric`` produces it by default, and the downsampled
aggregate carries it too). Binned data points come from the same fit result,
so nothing has to be passed in alongside.
"""

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from behav_utils.analysis.utils import cumulative_gaussian


# Map dict-key (math name) → display label (literature name)
_LABEL_FOR_KEY = {
    'mu':         'PSE',
    'sigma':      'slope',
    'lapse_low':  'λ_low',
    'lapse_high': 'λ_high',
    'accuracy':   'Acc',
}


def plot_comparison(
    result: Dict,
    contrast: Optional[str] = None,
    ax: Optional[Axes] = None,
    color_a: str = '#d62728',
    color_b: str = '#444444',
    show_stats: bool = True,
    show_band: bool = True,
    show_data: bool = True,
    stats_keys: Sequence[str] = ('mu', 'sigma', 'accuracy'),
) -> Axes:
    """Overlay the two psychometric curves of one contrast, with statistics.

    Args:
        result:   dict from ``compare_phases``.
        contrast: which contrast to draw, e.g. ``'opto_vs_non_opto'``.
                  Defaults to the only one when there is exactly one, otherwise
                  raises with the available keys listed.
        ax:       axis to draw on (creates a 4.5x4 figure if None).
        color_a:  colour for the contrast phase; ``color_b`` for the reference.
        show_stats: annotate Δ, CI and p in the top-left.
        show_band:  shade the bootstrap fit bands when present.
        show_data:  overlay the binned choice proportions from the fit result.
        stats_keys: which diffs to annotate. Pass
                    ``('mu','sigma','lapse_low','lapse_high','accuracy')``
                    for everything; keys absent from the contrast are skipped.

    Returns:
        The axis used.

    Raises:
        KeyError: if ``contrast`` is absent, or omitted when several exist, or
                  the phases carry no psychometric.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))

    contrasts = result.get('contrasts', {})
    if contrast is None:
        if len(contrasts) != 1:
            raise KeyError(
                "plot_comparison: several contrasts present, pass one of "
                f"{sorted(contrasts)}"
            )
        contrast = next(iter(contrasts))
    if contrast not in contrasts:
        raise KeyError(
            f"plot_comparison: no contrast {contrast!r}; "
            f"available: {sorted(contrasts)}"
        )

    con = contrasts[contrast]
    phases = result.get('phases', {})
    label_a = con.get('label_a', 'A')
    label_b = con.get('label_b', 'B')
    psyc_a = phases.get(label_a, {}).get('psychometric', {})
    psyc_b = phases.get(label_b, {}).get('psychometric', {})
    if not psyc_a or not psyc_b:
        raise KeyError(
            "plot_comparison: no psychometric on one of the phases — call "
            "compare_phases with 'psychometric' in stats."
        )
    params_a = psyc_a.get('params', {})
    params_b = psyc_b.get('params', {})

    # ── Bands (under the lines) ────────────────────────────────────
    if show_band:
        _plot_band(ax, psyc_b.get('curve_band'), color_b)
        _plot_band(ax, psyc_a.get('curve_band'), color_a)

    # ── Fit curves (B first so A is on top) ────────────────────────
    x = np.linspace(-1, 1, 200)
    for params, colour, label in [
        (params_b, color_b, label_b),
        (params_a, color_a, label_a),
    ]:
        mu = params.get('mu')
        if mu is not None and np.isfinite(mu):
            y = cumulative_gaussian(
                x, mu, params['sigma'],
                params.get('lapse_low', 0.0), params.get('lapse_high', 0.0),
            )
            ax.plot(x, y, color=colour, label=label, linewidth=2)

    # ── Binned data, taken from the fit result ─────────────────────
    if show_data:
        _plot_binned(ax, psyc_b, color_b)
        _plot_binned(ax, psyc_a, color_a)

    # ── References + labels ───────────────────────────────────────
    ax.axhline(0.5, color='k', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.axvline(0.0, color='k', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Stimulus (distance from boundary)')
    ax.set_ylabel('P(choose B)')
    ax.legend(loc='lower right', fontsize=8)

    if show_stats:
        ax.text(
            0.03, 0.97, _stats_text(con, stats_keys),
            transform=ax.transAxes, va='top', ha='left',
            fontsize=8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.7, edgecolor='none'),
        )

    return ax


# ── private helpers ────────────────────────────────────────────────

def _plot_band(ax: Axes, band: Optional[Dict], colour: str):
    if not band or band.get('lo') is None:
        return
    ax.fill_between(band['x'], band['lo'], band['hi'],
                    color=colour, alpha=0.20, linewidth=0)


def _plot_binned(ax: Axes, psyc: Dict, colour: str):
    """Scatter the binned choice proportions already computed by the fit."""
    centres = psyc.get('bin_centres')
    means = psyc.get('bin_means')
    if centres is None or means is None:
        return
    centres = np.asarray(centres, dtype=float)
    means = np.asarray(means, dtype=float)
    ok = np.isfinite(means)
    if not np.any(ok):
        return
    counts = psyc.get('bin_counts')
    if counts is not None:
        sizes = (8 + 0.05 * np.asarray(counts, dtype=float))[ok]
    else:
        sizes = 20.0
    ax.scatter(centres[ok], means[ok], s=sizes, color=colour,
               alpha=0.7, edgecolors='none', zorder=3)


def _stats_text(con: Dict, stats_keys: Sequence[str]) -> str:
    diffs = con.get('diffs', {})
    boot_ci = con.get('boot_ci')
    perm_p = con.get('perm_p')

    lines = []
    for key in stats_keys:
        val = diffs.get(key, np.nan)
        if val is None or not np.isfinite(val):
            continue
        label = _LABEL_FOR_KEY.get(key, key)
        line = f"Δ{label:<5s} = {val:+.3f}"
        if boot_ci and key in boot_ci:
            lo, hi = boot_ci[key]
            if np.isfinite(lo) and np.isfinite(hi):
                line += f" [{lo:+.3f}, {hi:+.3f}]"
        if perm_p and key in perm_p:
            p = perm_p[key]
            if p is not None and np.isfinite(p):
                line += f"  p={p:.3f}"
        lines.append(line)

    lines.append(f"n_{con.get('label_a', 'A')} = {con.get('n_a', '?')}")
    lines.append(f"n_{con.get('label_b', 'B')} = {con.get('n_b', '?')}")
    return '\n'.join(lines)


def plot_stat_comparison(
    result: Dict,
    stats: Optional[Sequence[str]] = None,
    ncols: int = 3,
    phase_order: Optional[Sequence[str]] = None,
    palette: Optional[Sequence[str]] = None,
    panel_size: tuple = (3.0, 3.0),
    suptitle: Optional[str] = None,
    show_p: bool = True,
):
    """Grid of per-phase stat values with bootstrap CIs, one panel per stat.

    The companion to :func:`plot_comparison` for the parameter *values* rather
    than the curves — PSE, slope, lapses, win_stay, accuracy and so on, side by
    side across phases.

    Each phase contributes one pooled estimate, drawn as a point with its
    bootstrap CI. That interval is sampling uncertainty in the pooled estimate;
    it is **not** a spread across sessions, which is why this is a point-and-CI
    plot rather than a box plot — a box would imply a distribution that a
    pooled estimate does not have. For a session-level spread use
    ``extract_stats(..., mode='per_session')`` with ``plot_stat_swarm``.

    The p-value above each non-reference phase is the permutation p for that
    phase's contrast against the reference.

    Args:
        result:      dict from ``compare_phases`` (run with ``n_bootstrap > 0``
                     for the intervals, ``n_permutations > 0`` for the p-values).
        stats:       which stats to draw; default is every scalar computed.
        ncols:       panels per row.
        phase_order: x-axis order; default is the order in ``result['phases']``
                     with the reference first.
        palette:     colours per phase; defaults to the house palette.
        panel_size:  (width, height) inches per panel.
        suptitle:    figure title.
        show_p:      annotate the contrast p-value above each non-reference phase.

    Returns:
        ``(fig, axes)`` with ``axes`` a flat list of the axes used.
    """
    from behav_utils.plotting.styles import get_colour

    phases = result.get('phases', {})
    contrasts = result.get('contrasts', {})
    reference = result.get('meta', {}).get('reference')
    if not phases:
        raise KeyError("plot_stat_comparison: no phases in result")

    if stats is None:
        stats = list(result.get('meta', {}).get('scalar_names') or [])
    stats = [s for s in stats]
    if not stats:
        raise ValueError("plot_stat_comparison: no scalar stats to plot")

    order = list(phase_order) if phase_order else (
        ([reference] if reference in phases else [])
        + [p for p in phases if p != reference]
    )
    colours = list(palette) if palette else [get_colour(i) for i in range(len(order))]

    # p-value for each phase, from its contrast against the reference
    p_for = {}
    for con in contrasts.values():
        pp = con.get('perm_p') or {}
        p_for[con.get('label_a')] = pp

    ncols = max(1, min(ncols, len(stats)))
    nrows = int(np.ceil(len(stats) / ncols))
    fig, axarr = plt.subplots(nrows, ncols, squeeze=False,
                              figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    flat = axarr.ravel()

    for ax, stat in zip(flat, stats):
        vals, los, his = [], [], []
        for name in order:
            ph = phases.get(name, {})
            v = ph.get('stats', {}).get(stat, np.nan)
            lo, hi = (ph.get('stats_ci') or {}).get(stat, (np.nan, np.nan))
            vals.append(v)
            los.append(v - lo if np.isfinite(lo) and np.isfinite(v) else np.nan)
            his.append(hi - v if np.isfinite(hi) and np.isfinite(v) else np.nan)

        xs = np.arange(len(order))
        for i, name in enumerate(order):
            if not np.isfinite(vals[i]):
                continue
            err = None
            if np.isfinite(los[i]) and np.isfinite(his[i]):
                err = [[los[i]], [his[i]]]
            ax.errorbar(xs[i], vals[i], yerr=err, fmt='o', color=colours[i % len(colours)],
                        markersize=7, capsize=4, elinewidth=1.4,
                        markeredgecolor='white', markeredgewidth=0.6, zorder=3)

        # p-values above the non-reference phases
        if show_p:
            finite = [v for v in vals if np.isfinite(v)]
            if finite:
                top = max(v + (h if np.isfinite(h) else 0)
                          for v, h in zip(vals, his) if np.isfinite(v))
                span = (max(finite) - min(finite)) or (abs(max(finite)) or 1.0)
                for i, name in enumerate(order):
                    p = (p_for.get(name) or {}).get(stat)
                    if p is None or not np.isfinite(p):
                        continue
                    ax.annotate(f"p={p:.3g}", xy=(xs[i], top + 0.12 * span),
                                ha='center', va='bottom', fontsize=7.5)
                ax.margins(y=0.25)

        ax.set_xticks(xs)
        ax.set_xticklabels(order, fontsize=8)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_title(_LABEL_FOR_KEY.get(stat, stat), fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)

    for ax in flat[len(stats):]:
        fig.delaxes(ax)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig, list(flat[:len(stats)])
