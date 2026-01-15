import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple

from Analysis.update_matrix import matrix_error

def plot_update_matrix(update_matrix: np.ndarray, 
                       title: str = 'Update Matrix',
                       vmin: float = -0.3, vmax: float = 0.3,
                       cmap: str = 'RdBu_r',
                       ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot update matrix as a heatmap.
    
    Args:
        update_matrix: (n_bins, n_bins) update matrix
        title: Plot title
        vmin, vmax: Colour scale limits
        cmap: Colourmap
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    n_bins = update_matrix.shape[0]
    
    im = ax.imshow(update_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin='lower', aspect='equal')
    
    ax.set_xlabel('Previous stimulus bin')
    ax.set_ylabel('Current stimulus bin')
    ax.set_title(title)
    
    # Tick labels
    tick_labels = [f'{i+1}' for i in range(n_bins)]
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels(tick_labels)
    
    plt.colorbar(im, ax=ax, label='Î”P(B)')
    
    return ax


def plot_update_matrix_comparison(data_matrix: np.ndarray, model_matrix: np.ndarray,
                                   figsize: Tuple[int, int] = (14, 4)) -> plt.Figure: #type: ignore
    """
    Plot data and model update matrices side by side with difference.
    
    Args:
        data_matrix: Data update matrix
        model_matrix: Model update matrix
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    vmax = max(np.nanmax(np.abs(data_matrix)), np.nanmax(np.abs(model_matrix)))
    vmin = -vmax
    
    plot_update_matrix(data_matrix, 'Data', vmin, vmax, ax=axes[0])
    plot_update_matrix(model_matrix, 'Model', vmin, vmax, ax=axes[1])
    
    diff = model_matrix - data_matrix
    diff_max = np.nanmax(np.abs(diff))
    plot_update_matrix(diff, 'Model - Data', -diff_max, diff_max, ax=axes[2])
    
    error = matrix_error(model_matrix, data_matrix)
    fig.suptitle(f'Update Matrix Comparison (MSE = {error:.4f})', y=1.02)
    
    plt.tight_layout()
    return fig