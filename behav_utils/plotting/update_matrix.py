"""
Update Matrix Plotting

plot_um(data, ax, **kwargs)

Accepts SessionData, List[SessionData], AnimalData, or raw np.ndarray.
NO FILTERING. Data must be pre-filtered via filter_trials / session.filter.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, TYPE_CHECKING

from behav_utils.analysis.update_matrix import compute_update_matrix
from behav_utils.plotting.styles import UM_CMAP

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData, AnimalData


def plot_um(
    data, ax=None, n_bins=8,
    cmap=None, vmin=None, vmax=None, colorbar=True,
    title='', xlabel='Previous stimulus', ylabel='Current stimulus',
) -> Tuple[plt.Figure, plt.Axes, dict]:
    """Plot an update matrix heatmap. Data must be pre-filtered."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.get_figure()

    cmap = cmap or UM_CMAP
    um, info = _resolve_um(data, n_bins)

    if vmin is None or vmax is None:
        absmax = np.nanmax(np.abs(um)) if not np.all(np.isnan(um)) else 0.1
        absmax = max(absmax, 0.01)
        vmin = vmin if vmin is not None else -absmax
        vmax = vmax if vmax is not None else absmax

    im = ax.imshow(um, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin='lower', aspect='equal', interpolation='nearest')
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    info.update({'um': um, 'vmin': vmin, 'vmax': vmax})
    return fig, ax, info


def _resolve_um(data, n_bins):
    """Resolve input to (um_array, info_dict). No filtering."""
    from behav_utils.data.structures import SessionData, AnimalData

    if isinstance(data, np.ndarray):
        return data, {'source': 'array', 'n_bins': data.shape[0]}

    if isinstance(data, SessionData):
        sessions = [data]
    elif isinstance(data, AnimalData):
        sessions = list(data.sessions)
    elif isinstance(data, (list, tuple)) and len(data) > 0 and hasattr(data[0], 'trials'):
        sessions = list(data)
    else:
        try:
            sessions = list(data)
            if not sessions or not hasattr(sessions[0], 'trials'):
                raise TypeError()
        except TypeError:
            raise TypeError(f"Expected SessionData/List/AnimalData/ndarray, got {type(data).__name__}")

    # Pool arrays across sessions — no filtering
    all_stim, all_ch, all_cat = [], [], []
    for s in sessions:
        arr = s.get_arrays()
        v = ~arr['no_response']
        if v.sum() > 0:
            all_stim.append(arr['stimuli'][v])
            all_ch.append(arr['choices'][v])
            all_cat.append(arr['categories'][v])

    if not all_stim:
        return np.full((n_bins, n_bins), np.nan), {'source': 'empty', 'n_sessions': 0}

    stim = np.concatenate(all_stim)
    ch = np.concatenate(all_ch)
    cat = np.concatenate(all_cat)
    um, _, um_info = compute_update_matrix(stim, ch, cat, n_bins=n_bins)
    return um, {'source': 'sessions', 'n_sessions': len(sessions), **um_info}
