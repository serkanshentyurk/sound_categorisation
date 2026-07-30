"""
behav_utils/plotting/comparison.py — Visualise compute_delta_stat output.

Pairs with behav_utils.analysis.comparison.compute_delta_stat. Overlays the two
psychometric curves of one contrast, with bootstrap bands, the binned data and
the Δ / p / CI annotation.

    from behav_utils.analysis import compute_delta_stat
    from behav_utils.plotting import plot_comparison

    r = compute_delta_stat({'non_opto': ctrl, 'opto': opto},
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
        result:   dict from ``compute_delta_stat``.
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
            "compute_delta_stat with 'psychometric' in stats."
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


def plot_stat_comparison_single(
    result: Dict,
    stat: str,
    ax: Optional[Axes] = None,
    phase_order: Optional[Sequence[str]] = None,
    palette: Optional[Sequence[str]] = None,
    show_p: bool = True,
    units: Optional[Sequence[str]] = None,
) -> Axes:
    """One stat's per-phase point-and-CI, drawn on a single Axes.

    The single-panel core of :func:`plot_stat_comparison`; use it to place one
    stat in a figure you lay out yourself. ``plot_stat_comparison`` calls this
    once per stat to build its grid, so there is one drawing implementation.

    Each phase contributes one pooled estimate drawn as a point with its
    bootstrap CI — sampling uncertainty in the pooled estimate, not a spread
    across sessions, which is why it is a point-and-CI rather than a box. The
    point can sit outside its interval (a boundary-pinned lapse or a skewed draw
    distribution); that is drawn as a segment with a hollow marker rather than
    via ``yerr``, which would raise on the negative offset.

    Args:
        result:      dict from ``compute_delta_stat`` (``n_bootstrap > 0`` for
                     the CI, ``n_permutations > 0`` for the p).
        stat:        the scalar to draw.
        ax:          Axes to draw on; a new one is made if None.
        phase_order: x-axis order; default is the reference first, then the rest.
        palette:     colours per phase; defaults to the house palette.
        show_p:      annotate the contrast p above each non-reference phase.
        units:       which resample unit(s) to draw the per-phase CI for. None
                     (default) draws the single legacy ``stats_ci`` in the phase
                     colour — unchanged behaviour. A tuple such as
                     ``('trials', 'sessions')`` draws one interval per unit from
                     ``stats_ci_by_unit``, offset at the same x: the trial CI thin
                     and grey (the diagnostic that ignores session scatter), the
                     session CI thick and in the phase colour (the honest interval
                     for a between-phase comparison). Units absent from
                     ``stats_ci_by_unit`` are skipped; if none are present it
                     falls back to the legacy single CI.

    Returns:
        the Axes drawn on.
    """
    from behav_utils.plotting.styles import get_colour

    phases = result.get('phases', {})
    contrasts = result.get('contrasts', {})
    reference = result.get('meta', {}).get('reference')
    if not phases:
        raise KeyError("plot_stat_comparison_single: no phases in result")

    order = list(phase_order) if phase_order else (
        ([reference] if reference in phases else [])
        + [p for p in phases if p != reference]
    )
    colours = list(palette) if palette else [get_colour(i) for i in range(len(order))]

    # p-value for each phase, from its contrast against the reference:
    # permutation p (within-phase) and the per-unit bootstrap p (between-phase).
    p_for, bootp_for = {}, {}
    for con in contrasts.values():
        p_for[con.get('label_a')] = con.get('perm_p') or {}
        bootp_for[con.get('label_a')] = con.get('boot_p_by_unit') or {}

    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 3.0))

    vals, his = [], []
    for name in order:
        ph = phases.get(name, {})
        v = ph.get('stats', {}).get(stat, np.nan)
        lo, hi = (ph.get('stats_ci') or {}).get(stat, (np.nan, np.nan))
        vals.append(v)
        his.append(hi - v if np.isfinite(hi) and np.isfinite(v) else np.nan)

    xs = np.arange(len(order))
    for i, name in enumerate(order):
        if not np.isfinite(vals[i]):
            continue
        colour = colours[i % len(colours)]
        ph = phases.get(name, {})

        # Which CIs to draw: one per requested unit if per-unit intervals exist,
        # else the single legacy stats_ci. Two units are offset at the same x —
        # trials thin/grey (ignores session scatter), sessions thick/coloured.
        by_unit = ph.get('stats_ci_by_unit') or {}
        draw = None
        if units:
            picked = [u for u in units if stat in (by_unit.get(u) or {})]
            if picked:
                draw = picked
        outside = False
        if draw:
            multi = len(draw) > 1
            for j, u in enumerate(draw):
                lo, hi = by_unit[u][stat]
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    continue
                dx = (j - (len(draw) - 1) / 2) * 0.16 if multi else 0.0
                is_sess = (u == 'sessions')
                c = colour if (is_sess or not multi) else '0.55'
                lw = 2.4 if (is_sess and multi) else 1.4
                ax.plot([xs[i] + dx, xs[i] + dx], [lo, hi], color=c, lw=lw,
                        solid_capstyle='butt', zorder=2)
                for cap in (lo, hi):
                    ax.plot([xs[i] + dx - 0.06, xs[i] + dx + 0.06], [cap, cap],
                            color=c, lw=lw, zorder=2)
                if is_sess or not multi:  # instability judged on the honest CI
                    outside = vals[i] < lo or vals[i] > hi
        else:
            # Segment rather than yerr: the point is the observed estimate and
            # the interval is a bootstrap percentile, so the point need not lie
            # inside it. yerr raises on the negative offset; a segment renders
            # it, and the hollow marker flags the estimate as unstable.
            lo, hi = (ph.get('stats_ci') or {}).get(stat, (np.nan, np.nan))
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([xs[i], xs[i]], [lo, hi], color=colour, lw=1.4,
                        solid_capstyle='butt', zorder=2)
                for cap in (lo, hi):
                    ax.plot([xs[i] - 0.07, xs[i] + 0.07], [cap, cap],
                            color=colour, lw=1.4, zorder=2)
                outside = vals[i] < lo or vals[i] > hi
        ax.plot(xs[i], vals[i], marker='o', markersize=7,
                markerfacecolor='white' if outside else colour,
                markeredgecolor=colour,
                markeredgewidth=1.6 if outside else 0.6, zorder=3)

    if show_p:
        finite = [v for v in vals if np.isfinite(v)]
        if finite:
            top = max(max(v, v + h) if np.isfinite(h) else v
                      for v, h in zip(vals, his) if np.isfinite(v))
            span = (max(finite) - min(finite)) or (abs(max(finite)) or 1.0)
            for i, name in enumerate(order):
                perm_p = (p_for.get(name) or {}).get(stat)
                if perm_p is not None and np.isfinite(perm_p):
                    # within-phase: the valid permutation p
                    ax.annotate(f"p={perm_p:.3g}", xy=(xs[i], top + 0.12 * span),
                                ha='center', va='bottom', fontsize=7.5)
                elif units:
                    # between-phase: pair each CI with its bootstrap p. The trial
                    # p is the over-confident one — small exactly when the trial
                    # CI is misleadingly narrow — so it is shown greyed next to
                    # the honest session p, never alone.
                    bp = bootp_for.get(name) or {}
                    parts = []
                    for u in units:
                        pu = (bp.get(u) or {}).get(stat)
                        if pu is not None and np.isfinite(pu):
                            tag = 's' if u == 'sessions' else 't'
                            parts.append((tag, pu))
                    for k, (tag, pu) in enumerate(parts):
                        ax.annotate(f"p{tag}={pu:.2g}",
                                    xy=(xs[i], top + (0.12 + 0.13 * k) * span),
                                    ha='center', va='bottom', fontsize=7,
                                    color=('0.55' if tag == 't' else 'black'))
            ax.margins(y=0.25 + 0.13 * (len(units) if units else 0))

    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=8)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_title(_LABEL_FOR_KEY.get(stat, stat), fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    return ax


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
    side across phases. Each panel is :func:`plot_stat_comparison_single`; call
    that directly to place a single stat in a figure you lay out yourself.

    Each phase contributes one pooled estimate with its bootstrap CI (sampling
    uncertainty, not a session spread — hence point-and-CI, not a box). The p
    above each non-reference phase is the permutation p for that phase's
    contrast against the reference.

    Args:
        result:      dict from ``compute_delta_stat`` (``n_bootstrap > 0`` for
                     the intervals, ``n_permutations > 0`` for the p-values).
        stats:       which stats to draw; default is every scalar computed.
        ncols:       panels per row.
        phase_order: x-axis order; default is the reference first, then the rest.
        palette:     colours per phase; defaults to the house palette.
        panel_size:  (width, height) inches per panel.
        suptitle:    figure title.
        show_p:      annotate the contrast p above each non-reference phase.

    Returns:
        ``(fig, axes)`` with ``axes`` a flat list of the axes used.
    """
    from behav_utils.plotting.styles import get_colour

    phases = result.get('phases', {})
    reference = result.get('meta', {}).get('reference')
    if not phases:
        raise KeyError("plot_stat_comparison: no phases in result")

    if stats is None:
        stats = list(result.get('meta', {}).get('scalar_names') or [])
    stats = [s for s in stats]
    if not stats:
        raise ValueError("plot_stat_comparison: no scalar stats to plot")

    # resolve order and colours once so every panel matches
    order = list(phase_order) if phase_order else (
        ([reference] if reference in phases else [])
        + [p for p in phases if p != reference]
    )
    colours = list(palette) if palette else [get_colour(i) for i in range(len(order))]

    ncols = max(1, min(ncols, len(stats)))
    nrows = int(np.ceil(len(stats) / ncols))
    fig, axarr = plt.subplots(nrows, ncols, squeeze=False,
                              figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    flat = axarr.ravel()

    for ax, stat in zip(flat, stats):
        plot_stat_comparison_single(result, stat, ax=ax, phase_order=order,
                                    palette=colours, show_p=show_p)

    for ax in flat[len(stats):]:
        fig.delaxes(ax)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig, list(flat[:len(stats)])


def plot_interaction_single(
    interaction: Dict,
    stat: str,
    ax: Optional[Axes] = None,
    show_p: bool = True,
    show_components: bool = True,
    colour_a: str = '#1f77b4',
    colour_b: str = '#ff7f0e',
    colour_interaction: str = '#7f2704',
    units: Optional[Sequence[str]] = None,
) -> Axes:
    """One stat's difference-of-differences, drawn on a single Axes.

    The single-panel core of :func:`plot_interaction`; use it to place one
    interaction in a figure you lay out yourself. ``plot_interaction`` calls
    this once per stat to build its grid.

    Three points on a common scale: the effect in the first result, the effect
    in the second, and their difference, each with a 95% bootstrap interval.
    Read the interaction, not the gap between the two components — two intervals
    can overlap while their difference is significant (independent equal-SE
    estimates stop overlapping only at 3.9 SE, significant from 2.8 SE), which
    is why the difference is drawn explicitly.

    Args:
        interaction: output of ``compute_interaction``.
        stat:        the scalar to draw.
        ax:          Axes to draw on; a new one is made if None.
        show_p:      annotate the interaction p-value.
        show_components: draw the two component effects beside the interaction;
                     False gives the interaction alone.
        colour_a, colour_b, colour_interaction: marker colours.
        units:       which resample unit(s) to draw the interaction CI for. None
                     (default) draws the single legacy interval — unchanged. A
                     tuple such as ``('trials', 'sessions')`` draws one interval
                     per unit from ``entry['by_unit']``, offset at the interaction
                     x: trials thin/grey, sessions thick/coloured, with a '∗' over
                     the session interval when it excludes 0. The component
                     effects keep their single (trial-level) interval — the dual
                     CI is only meaningful on the interaction, which is the
                     between-session quantity.

    Returns:
        the Axes drawn on.
    """
    meta = interaction.get('meta', {})
    label_a = meta.get('label_a', 'a')
    label_b = meta.get('label_b', 'b')
    if stat == 'meta' or stat not in interaction:
        raise KeyError(f"plot_interaction_single: stat {stat!r} not in interaction")

    if show_components:
        positions = [f'\u0394 {label_a}', f'\u0394 {label_b}', 'interaction']
        colours = [colour_a, colour_b, colour_interaction]
    else:
        positions = ['interaction']
        colours = [colour_interaction]

    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 3.0))

    entry = interaction[stat]
    if show_components:
        values = [entry.get('delta_a', np.nan),
                  entry.get('delta_b', np.nan),
                  entry.get('interaction', np.nan)]
        intervals = [entry.get('ci_a', (np.nan, np.nan)),
                     entry.get('ci_b', (np.nan, np.nan)),
                     (entry.get('ci_lo', np.nan), entry.get('ci_hi', np.nan))]
    else:
        values = [entry.get('interaction', np.nan)]
        intervals = [(entry.get('ci_lo', np.nan), entry.get('ci_hi', np.nan))]

    ax.axhline(0.0, color='0.6', lw=1, ls='--', zorder=0)

    for i, (value, (lo, hi), colour) in enumerate(zip(values, intervals, colours)):
        if not np.isfinite(value):
            continue
        is_interaction = (i == len(values) - 1)

        # Dual CI on the interaction bar only: one interval per requested unit
        # from entry['by_unit'], offset at the interaction x (trials thin/grey,
        # sessions thick/coloured, '∗' over the session interval if it excludes
        # 0). Components and the no-units case keep the single interval below.
        by_unit = entry.get('by_unit') or {}
        draw = None
        if is_interaction and units:
            picked = [u for u in units if u in by_unit]
            if picked:
                draw = picked

        # Segment rather than yerr, same reason as the phase plot: the point is
        # the observed estimate and the interval a bootstrap percentile, so the
        # point can fall outside it. A hollow marker then flags it as unstable.
        outside = False
        if draw:
            multi = len(draw) > 1
            for j, u in enumerate(draw):
                ulo = by_unit[u].get('ci_lo', np.nan)
                uhi = by_unit[u].get('ci_hi', np.nan)
                if not (np.isfinite(ulo) and np.isfinite(uhi)):
                    continue
                dx = (j - (len(draw) - 1) / 2) * 0.18 if multi else 0.0
                is_sess = (u == 'sessions')
                c = colour if (is_sess or not multi) else '0.55'
                lw = 2.6 if (is_sess and multi) else 1.8
                ax.plot([i + dx, i + dx], [ulo, uhi], color=c, lw=lw,
                        solid_capstyle='butt', zorder=2)
                for cap in (ulo, uhi):
                    ax.plot([i + dx - 0.06, i + dx + 0.06], [cap, cap],
                            color=c, lw=lw, zorder=2)
                if is_sess and (ulo > 0 or uhi < 0):  # session CI excludes 0
                    ax.annotate('\u2217', (i + dx, uhi), textcoords='offset points',
                                xytext=(0, 3), ha='center', va='bottom',
                                fontsize=11, color=c, zorder=5)
                if is_sess or not multi:
                    outside = value < ulo or value > uhi
        elif np.isfinite(lo) and np.isfinite(hi):
            ax.plot([i, i], [lo, hi], color=colour,
                    lw=1.8 if is_interaction else 1.3,
                    solid_capstyle='butt', zorder=2)
            for cap in (lo, hi):
                ax.plot([i - 0.07, i + 0.07], [cap, cap], color=colour,
                        lw=1.8 if is_interaction else 1.3, zorder=2)
            outside = value < lo or value > hi

        ax.plot(i, value, marker='o',
                markersize=8 if is_interaction else 6.5,
                markerfacecolor='white' if outside else colour,
                markeredgecolor=colour,
                markeredgewidth=1.6 if outside else 0.6,
                zorder=4 if is_interaction else 3)

    if show_p:
        finite = [(v, iv) for v, iv in zip(values, intervals) if np.isfinite(v)]
        by_unit = entry.get('by_unit') or {}
        # Paired p's when units are drawn (trial greyed, session black); else the
        # single legacy p. The trial p is never shown alone — it is small exactly
        # when the trial CI is misleadingly narrow.
        pairs = []
        if units and by_unit:
            for u in units:
                pu = (by_unit.get(u) or {}).get('p_two_sided')
                if pu is not None and np.isfinite(pu):
                    pairs.append(('s' if u == 'sessions' else 't', pu))
        else:
            p = entry.get('p_two_sided')
            if p is not None and np.isfinite(p):
                pairs = [('', p)]
        if finite and pairs:
            top = max(max(v, iv[1]) if np.isfinite(iv[1]) else v for v, iv in finite)
            bottom = min(min(v, iv[0]) if np.isfinite(iv[0]) else v for v, iv in finite)
            span = (top - bottom) or (abs(top) or 1.0)
            for k, (tag, pu) in enumerate(pairs):
                label = f"p = {pu:.3g}" if tag == '' else f"p{tag}={pu:.2g}"
                ax.annotate(label, xy=(len(values) - 1, top + (0.10 + 0.13 * k) * span),
                            ha='center', va='bottom', fontsize=8,
                            color=('0.55' if tag == 't' else colour_interaction))
            ax.margins(y=0.22 + 0.13 * max(0, len(pairs) - 1))

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(positions, fontsize=8, rotation=20 if show_components else 0,
                       ha='right' if show_components else 'center')
    ax.set_xlim(-0.6, len(positions) - 0.4)
    ax.set_title(_LABEL_FOR_KEY.get(stat, stat), fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    return ax


def plot_interaction(
    interaction: Dict,
    stats: Optional[Sequence[str]] = None,
    ncols: int = 4,
    panel_size: tuple = (3.0, 3.0),
    suptitle: Optional[str] = None,
    show_p: bool = True,
    show_components: bool = True,
    colour_a: str = '#1f77b4',
    colour_b: str = '#ff7f0e',
    colour_interaction: str = '#7f2704',
):
    """Grid of difference-of-differences, one panel per stat.

    The companion to :func:`plot_stat_comparison`: that shows phases within a
    session type, this shows whether the *effect* differs between two session
    types — masking vs opto, say. Each panel is :func:`plot_interaction_single`;
    call that directly to place a single interaction in your own layout.

    Every panel gets its own y-scale: PSE is in stimulus units, win_stay a
    proportion, and a shared axis would flatten most to slivers.

    Args:
        interaction: output of ``compute_interaction``.
        stats:       which stats to draw, in order; default is all of them.
        ncols:       panels per row; unused axes are removed.
        panel_size:  (width, height) inches per panel.
        suptitle:    figure title.
        show_p:      annotate the interaction p-value.
        show_components: draw the two component effects beside the interaction.
        colour_a, colour_b, colour_interaction: marker colours.

    Returns:
        ``(fig, axes)`` with ``axes`` a flat list of the axes used.

    Raises:
        ValueError: if there is nothing to plot.
    """
    if stats is None:
        stats = [k for k in interaction if k != 'meta']
    stats = [s for s in stats if s in interaction and s != 'meta']
    if not stats:
        raise ValueError("plot_interaction: no stats to plot")

    ncols = max(1, min(ncols, len(stats)))
    nrows = int(np.ceil(len(stats) / ncols))
    fig, axarr = plt.subplots(nrows, ncols, squeeze=False,
                              figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    flat = axarr.ravel()

    for ax, stat in zip(flat, stats):
        plot_interaction_single(interaction, stat, ax=ax, show_p=show_p,
                                show_components=show_components,
                                colour_a=colour_a, colour_b=colour_b,
                                colour_interaction=colour_interaction)

    for ax in flat[len(stats):]:
        fig.delaxes(ax)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig, list(flat[:len(stats)])
