"""
Update Matrix Plotting

plot_um(data, ax, **kwargs)

Accepts SessionData, List[SessionData], AnimalData, or raw np.ndarray.
NO FILTERING. Data must be pre-filtered via filter_trials / session.filter.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Dict

from behav_utils.plotting.styles import UM_CMAP

def plot_um(
    result: Union[Dict, np.ndarray],
    ax: Optional[plt.Axes] = None,
    session_idx: Optional[Union[int, str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap=None,
    colorbar: bool = True,
    title: str = '',
    xlabel: str = 'Previous stimulus',
    ylabel: str = 'Current stimulus',
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot an update matrix heatmap from a compute_um() result.

    Args:
        result: Dict from compute_um(), or raw ndarray for convenience.
        ax: Matplotlib axes (creates one if None).
        session_idx: Only for a per_session result. An int selects that
            session's matrix (matched on the entry's session_idx tag); 'all'
            plots the nan-aware mean over the per-session matrices. Must be None
            for a pooled result or a raw ndarray.
        vmin/vmax: Colour scale limits (auto-symmetric if None).
        cmap: Colourmap (default: UM_CMAP).
        colorbar: Show colour bar.
        title: Axes title.

    Returns:
        (fig, ax)
    """
    # Resolve the matrix to plot, handling pooled / per_session / ndarray.
    if isinstance(result, np.ndarray):
        if session_idx is not None:
            raise ValueError("session_idx is only valid for a per_session "
                             "compute_um() result, not a raw array.")
        um = result
        n_bins = um.shape[0]
    elif isinstance(result, dict) and result.get('mode') == 'per_session':
        entries = result.get('per_session', [])
        if not entries:
            raise ValueError("per_session result has no sessions to plot.")
        if session_idx is None:
            available = [e.get('session_idx') for e in entries]
            raise ValueError(
                "per_session result is a list of matrices; pass "
                f"session_idx=<int> or 'all'. Available session_idx: {available}.")
        if session_idx == 'all':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                um = np.nanmean(np.stack([e['um'] for e in entries]), axis=0)
        else:
            match = [e for e in entries if e.get('session_idx') == session_idx]
            if not match:
                available = [e.get('session_idx') for e in entries]
                raise ValueError(
                    f"session_idx={session_idx!r} not found. Available: {available}.")
            um = match[0]['um']
        n_bins = result.get('n_bins', um.shape[0])
    elif isinstance(result, dict):
        if session_idx is not None:
            raise ValueError("session_idx is only valid for a per_session "
                             "compute_um() result; this is a pooled result.")
        um = result['um']
        n_bins = result.get('n_bins', um.shape[0])
    else:
        raise TypeError(
            f"plot_um expects dict from compute_um() or ndarray, "
            f"got {type(result).__name__}."
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))
    else:
        fig = ax.get_figure()

    cmap = cmap or UM_CMAP

    if vmin is None or vmax is None:
        abs_max = np.nanmax(np.abs(um))
        abs_max = max(abs_max, 0.01)
        vmin = vmin if vmin is not None else -abs_max
        vmax = vmax if vmax is not None else abs_max

    im = ax.imshow(um, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin='lower', aspect='equal')

    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Tick labels
    edges = np.linspace(-1, 1, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    tick_labels = [f'{c:.1f}' for c in centres]
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels(tick_labels, fontsize=7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    return fig, ax
