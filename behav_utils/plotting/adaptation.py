"""
plot_adaptation — the convergence time-course from compute_adaptation.

Draws the windowed convergence curve over post-switch trials, with the
bootstrap band when present, a dashed line at the plateau, and reference lines
at 0 (pre-switch) and 1 (normative optimum, when normalised).
"""

from typing import Dict, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from behav_utils.plotting.styles import get_colour

__all__ = ['plot_adaptation']


def plot_adaptation(
    result: Dict,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
    show_band: bool = True,
    show_plateau: bool = True,
) -> plt.Axes:
    """Plot one convergence curve from ``compute_adaptation``.

    Call repeatedly on the same ``ax`` with different colours/labels to overlay
    animals or distributions.

    Args:
        result:       output of ``compute_adaptation``.
        ax:           axis to draw on (creates one if None).
        color:        curve colour (house palette[0] if None).
        label:        legend label.
        show_band:    shade the bootstrap CI band when present.
        show_plateau: dashed horizontal line at the plateau value.

    Returns:
        The axis used.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    colour = color or get_colour(0)

    curve = result['curve']
    trials = np.asarray(curve['trials'], dtype=float)
    values = np.asarray(curve['values'], dtype=float)
    normalise = result['meta'].get('normalise', True)

    # reference lines
    ax.axhline(0.0, color='0.6', lw=1, ls='--', zorder=0)   # pre-switch
    if normalise:
        ax.axhline(1.0, color='0.6', lw=1, ls=':', zorder=0)   # normative optimum

    if show_band and curve.get('ci_lo') is not None:
        lo = np.asarray(curve['ci_lo'], dtype=float)
        hi = np.asarray(curve['ci_hi'], dtype=float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        if ok.any():
            ax.fill_between(trials[ok], lo[ok], hi[ok], color=colour, alpha=0.18, linewidth=0)

    ok = np.isfinite(values)
    ax.plot(trials[ok], values[ok], color=colour, lw=2, label=label, zorder=3)

    if show_plateau:
        scalars = {s['stat']: s['value'] for s in result['scalars']}
        plateau = scalars.get('plateau')
        if plateau is not None and np.isfinite(plateau):
            ax.axhline(plateau, color=colour, lw=1, ls='--', alpha=0.5, zorder=1)

    ax.set_xlabel('Post-switch trial')
    ax.set_ylabel('Convergence' if normalise else 'PSE shift (stimulus units)')
    if label:
        ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    return ax
