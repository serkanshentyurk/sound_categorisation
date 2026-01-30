"""
Session plotting functions for trial-by-trial visualisation.

Usage:
    from Plotting.session import plot_session
    
    fig = plot_session(stimuli, choices, categories, p_B)
    fig = plot_session(stimuli, choices, categories, p_B, belief_mu=belief_history)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Optional, Tuple, List
import warnings


def plot_session(
    stimuli: np.ndarray,
    choices: np.ndarray,
    categories: np.ndarray,
    p_B: np.ndarray,
    belief_mu: Optional[np.ndarray] = None,
    show_running_accuracy: bool = False,
    accuracy_window: int = 20,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    alpha_pB: float = 0.4,
    marker_size: float = 40,
    show_legend: bool = True,
    xlim: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot trial-by-trial session visualisation.
    
    Main panel shows:
        - Stimuli as triangles (▲ chose B, ▼ chose A)
        - Coloured green (correct) / red (error)
        - P(B) as semi-transparent line (right y-axis)
        - True category boundary at y=0
        - Model's boundary belief trajectory (if provided)
    
    Optional bottom panel shows:
        - Running accuracy with sliding window
    
    Args:
        stimuli: Array of stimulus values, shape (n_trials,)
        choices: Array of choices (0=A, 1=B), shape (n_trials,)
        categories: Array of true categories (0=A, 1=B), shape (n_trials,)
        p_B: Array of P(choose B), shape (n_trials,)
        belief_mu: Model's boundary belief per trial, shape (n_trials,). Optional.
        show_running_accuracy: Whether to show running accuracy panel. Default False.
        accuracy_window: Window size for running accuracy calculation.
        figsize: Figure size. Default (14, 4) or (14, 6) with accuracy panel.
        title: Plot title. Optional.
        alpha_pB: Transparency for P(B) line. Default 0.4.
        marker_size: Size of stimulus markers. Default 40.
        show_legend: Whether to show legend. Default True.
        xlim: Optional x-axis limits as (start, end) trial numbers.
    
    Returns:
        Matplotlib Figure
    
    Example:
        >>> stimuli, categories, rng = generate_stimuli(n_trials=300)
        >>> model = BoundaryEstimationModel(**params)
        >>> choices, p_B = model.simulate_session(stimuli, categories, rng=rng)
        >>> fig = plot_session(stimuli, choices, categories, p_B)
    """
    # Validate inputs
    n_trials = len(stimuli)
    assert len(choices) == n_trials, "choices must have same length as stimuli"
    assert len(categories) == n_trials, "categories must have same length as stimuli"
    assert len(p_B) == n_trials, "p_B must have same length as stimuli"
    
    if belief_mu is not None:
        assert len(belief_mu) == n_trials, "belief_mu must have same length as stimuli"
    
    # Convert to arrays
    stimuli = np.asarray(stimuli)
    choices = np.asarray(choices)
    categories = np.asarray(categories)
    p_B = np.asarray(p_B)
    
    # Compute correct/error
    correct = choices == categories
    
    # Set up figure
    if figsize is None:
        figsize = (14, 6) if show_running_accuracy else (14, 4)
    
    if show_running_accuracy:
        fig, (ax_main, ax_acc) = plt.subplots(
            2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True
        )
    else:
        fig, ax_main = plt.subplots(figsize=figsize)
    
    trials = np.arange(n_trials)
    
    # =========================================================================
    # Main panel: Stimuli + P(B)
    # =========================================================================
    
    # Create right axis for P(B)
    ax_pB = ax_main.twinx()
    
    # Plot P(B) first (background)
    ax_pB.plot(trials, p_B, color='steelblue', alpha=alpha_pB, linewidth=1.5,
               label='P(B)', zorder=1)
    ax_pB.axhline(0.5, color='steelblue', linestyle=':', alpha=0.3, linewidth=1)
    ax_pB.set_ylabel('P(choose B)', color='steelblue', fontsize=10)
    ax_pB.tick_params(axis='y', labelcolor='steelblue')
    ax_pB.set_ylim(-0.05, 1.05)
    
    # Plot boundary belief if provided
    if belief_mu is not None:
        ax_main.plot(trials, belief_mu, color='purple', linestyle='--', 
                     alpha=0.7, linewidth=1.5, label='Boundary belief (μ)', zorder=2)
    
    # True category boundary
    ax_main.axhline(0, color='black', linestyle='-', linewidth=1.5, 
                    alpha=0.7, label='True boundary', zorder=2)
    
    # Plot stimuli as markers
    # Separate by choice and correctness
    for chose_B in [False, True]:
        for is_correct in [False, True]:
            mask = (choices == int(chose_B)) & (correct == is_correct)
            
            if not mask.any():
                continue
            
            marker = '^' if chose_B else 'v'  # ▲ for B, ▼ for A
            color = 'green' if is_correct else 'red'
            
            ax_main.scatter(
                trials[mask], stimuli[mask],
                marker=marker, s=marker_size, c=color,
                alpha=0.7, edgecolors='none', zorder=3
            )
    
    # Main axis settings
    ax_main.set_ylabel('Stimulus', fontsize=10)
    ax_main.set_ylim(-1.15, 1.15)
    ax_main.set_xlim(xlim if xlim else (-5, n_trials + 5))
    
    # Light category shading
    ax_main.axhspan(0, 1.15, alpha=0.05, color='blue', zorder=0)   # Category B region
    ax_main.axhspan(-1.15, 0, alpha=0.05, color='orange', zorder=0)  # Category A region
    
    # Add category labels
    ax_main.text(n_trials + 2, 0.6, 'B', fontsize=12, fontweight='bold', 
                 color='blue', alpha=0.5, ha='left')
    ax_main.text(n_trials + 2, -0.6, 'A', fontsize=12, fontweight='bold', 
                 color='orange', alpha=0.5, ha='left')
    
    # Legend
    if show_legend:
        # Custom legend handles
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
                   markersize=8, label='Chose B, correct'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
                   markersize=8, label='Chose B, error'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='green',
                   markersize=8, label='Chose A, correct'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
                   markersize=8, label='Chose A, error'),
            Line2D([0], [0], color='black', linewidth=1.5, label='True boundary'),
            Line2D([0], [0], color='steelblue', alpha=alpha_pB, linewidth=1.5,
                   label='P(B)'),
        ]
        
        if belief_mu is not None:
            legend_elements.append(
                Line2D([0], [0], color='purple', linestyle='--', linewidth=1.5,
                       label='Boundary belief')
            )
        
        ax_main.legend(handles=legend_elements, loc='upper left', fontsize=8,
                       ncol=2, framealpha=0.9)
    
    # Title
    if title:
        ax_main.set_title(title, fontsize=12, fontweight='bold')
    else:
        accuracy = correct.mean()
        ax_main.set_title(f'Session: {n_trials} trials, {accuracy:.1%} accuracy',
                          fontsize=12)
    
    # =========================================================================
    # Optional: Running accuracy panel
    # =========================================================================
    
    if show_running_accuracy:
        running_acc = _compute_running_accuracy(correct, window=accuracy_window)
        
        ax_acc.plot(trials, running_acc, color='black', linewidth=1.5)
        ax_acc.axhline(0.5, color='red', linestyle='--', alpha=0.5, 
                       linewidth=1, label='Chance')
        ax_acc.fill_between(trials, 0.5, running_acc, 
                            where=(running_acc >= 0.5),
                            color='green', alpha=0.2)
        ax_acc.fill_between(trials, 0.5, running_acc,
                            where=(running_acc < 0.5),
                            color='red', alpha=0.2)
        
        ax_acc.set_ylabel(f'Accuracy\n(window={accuracy_window})', fontsize=10)
        ax_acc.set_xlabel('Trial', fontsize=10)
        ax_acc.set_ylim(0.3, 1.0)
        ax_acc.set_xlim(xlim if xlim else (-5, n_trials + 5))
    else:
        ax_main.set_xlabel('Trial', fontsize=10)
    
    plt.tight_layout()
    
    return fig


def _compute_running_accuracy(correct: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Compute running accuracy with a sliding window.
    
    Uses causal window (only past trials) with edge padding.
    
    Args:
        correct: Boolean array of correct/incorrect
        window: Window size
    
    Returns:
        Array of running accuracy, same length as input
    """
    n = len(correct)
    running_acc = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - window + 1)
        running_acc[i] = correct[start:i+1].mean()
    
    return running_acc


def plot_session_comparison(
    sessions: List[dict],
    labels: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    share_y: bool = True,
) -> plt.Figure:
    """
    Plot multiple sessions side by side for comparison.
    
    Args:
        sessions: List of dicts, each with keys:
                  'stimuli', 'choices', 'categories', 'p_B', 
                  and optionally 'belief_mu'
        labels: Optional list of labels for each session
        figsize: Figure size
        share_y: Whether to share y-axis across panels
    
    Returns:
        Matplotlib Figure
    
    Example:
        >>> sessions = [
        ...     {'stimuli': s1, 'choices': c1, 'categories': cat1, 'p_B': p1, 'label': 'burn_in=0'},
        ...     {'stimuli': s2, 'choices': c2, 'categories': cat2, 'p_B': p2, 'label': 'burn_in=1000'},
        ... ]
        >>> fig = plot_session_comparison(sessions)
    """
    n_sessions = len(sessions)
    
    if labels is None:
        labels = [f'Session {i+1}' for i in range(n_sessions)]
    
    if figsize is None:
        figsize = (6 * n_sessions, 4)
    
    fig, axes = plt.subplots(1, n_sessions, figsize=figsize, sharey=share_y)
    
    if n_sessions == 1:
        axes = [axes]
    
    for ax, session, label in zip(axes, sessions, labels):
        # Extract data
        stimuli = session['stimuli']
        choices = session['choices']
        categories = session['categories']
        p_B = session['p_B']
        belief_mu = session.get('belief_mu', None)
        
        n_trials = len(stimuli)
        trials = np.arange(n_trials)
        correct = choices == categories
        
        # Create right axis for P(B)
        ax_pB = ax.twinx()
        
        # Plot P(B)
        ax_pB.plot(trials, p_B, color='steelblue', alpha=0.4, linewidth=1.5)
        ax_pB.set_ylim(-0.05, 1.05)
        ax_pB.set_ylabel('P(B)', color='steelblue', fontsize=9)
        ax_pB.tick_params(axis='y', labelcolor='steelblue')
        
        # Plot belief if provided
        if belief_mu is not None:
            ax.plot(trials, belief_mu, color='purple', linestyle='--', 
                    alpha=0.7, linewidth=1.5)
        
        # Boundary
        ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        
        # Stimuli
        for chose_B in [False, True]:
            for is_correct in [False, True]:
                mask = (choices == int(chose_B)) & (correct == is_correct)
                if mask.any():
                    marker = '^' if chose_B else 'v'
                    color = 'green' if is_correct else 'red'
                    ax.scatter(trials[mask], stimuli[mask], marker=marker, 
                              s=30, c=color, alpha=0.7, edgecolors='none')
        
        # Settings
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel('Trial', fontsize=10)
        ax.set_title(f'{label}\n({correct.mean():.1%} acc)', fontsize=11)
        
        if ax == axes[0]:
            ax.set_ylabel('Stimulus', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_session_segment(
    stimuli: np.ndarray,
    choices: np.ndarray,
    categories: np.ndarray,
    p_B: np.ndarray,
    start_trial: int = 0,
    end_trial: int = 50,
    belief_mu: Optional[np.ndarray] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot a segment of a session (zoomed view).
    
    Convenience wrapper around plot_session with xlim set.
    
    Args:
        stimuli, choices, categories, p_B: Session data
        start_trial: First trial to show
        end_trial: Last trial to show
        belief_mu: Optional belief history
        **kwargs: Additional arguments passed to plot_session
    
    Returns:
        Matplotlib Figure
    """
    return plot_session(
        stimuli, choices, categories, p_B,
        belief_mu=belief_mu,
        xlim=(start_trial, end_trial),
        title=f'Trials {start_trial}-{end_trial}',
        **kwargs
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'plot_session',
    'plot_session_comparison',
    'plot_session_segment',
]
