"""
Psychometric curve plotting.

Core:
    plot_psychometric           Single curve on one axes
    plot_psychometric_overlay   Multiple curves overlaid on one axes

Session-aware:
    plot_session_psychometrics  Grid or pooled from a list of SessionData
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Tuple, List, Union

from Helpers.psychometry import fit_psychometric, compute_psychometric_gof


# =============================================================================
# Core: single psychometric curve
# =============================================================================

def plot_psychometric(
    stimuli: np.ndarray,
    choices: np.ndarray,
    ax: Optional[plt.Axes] = None,
    n_bins: int = 8,
    show_fit: bool = True,
    show_params: bool = True,
    show_gof: bool = False,
    show_lapse: bool = False,
    show_reference: bool = True,
    n_bootstrap: int = 0,
    show_ci: bool = True,
    color: Optional[str] = None,
    label: Optional[str] = None,
    marker: str = 'o',
    markersize: int = 8,
    linewidth: int = 2,
    capsize: int = 3,
    x_fine: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    text_position: str = 'upper left',
    seed: int = 42,
) -> Tuple[plt.Axes, Dict]:
    """
    Plot a single psychometric curve with binned data and optional fitted curve.

    Args:
        stimuli: Stimulus values
        choices: Binary choices (0=A, 1=B)
        ax: Axes (creates new if None)
        n_bins: Stimulus bins for data points
        show_fit: Show fitted cumulative Gaussian
        show_params: Show mu, sigma text
        show_gof: Show Acc, R2 text
        show_lapse: Show lapse rates text
        show_reference: Show 0.5 / 0 reference lines
        n_bootstrap: Bootstrap resamples for CIs (0 = none)
        show_ci: Show CI band (needs n_bootstrap > 0)
        color: Colour
        label: Legend label
        title: Axes title
        text_position: 'upper left', 'upper right', 'lower left', 'lower right'
        seed: Bootstrap seed

    Returns:
        (ax, info_dict)
    """
    stimuli = np.asarray(stimuli, dtype=float)
    choices = np.asarray(choices, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    if x_fine is None:
        x_fine = np.linspace(-1, 1, 100)
    if color is None:
        color = 'C0'

    # Drop NaN
    valid = ~np.isnan(stimuli) & ~np.isnan(choices)
    stim_v, ch_v = stimuli[valid], choices[valid]

    # --- Binned data ---
    bin_edges = np.linspace(-1, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(stim_v, bin_edges) - 1, 0, n_bins - 1)

    prop_B = np.zeros(n_bins)
    prop_B_se = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for b in range(n_bins):
        m = bin_idx == b
        counts[b] = m.sum()
        if counts[b] > 0:
            prop_B[b] = ch_v[m].mean()
            prop_B_se[b] = np.sqrt(prop_B[b] * (1 - prop_B[b]) / counts[b])

    ax.errorbar(bin_centers, prop_B, yerr=prop_B_se, fmt=marker,
                color=color, markersize=markersize, capsize=capsize, label=label)

    # --- Fit ---
    psych = fit_psychometric(stim_v, ch_v, x_fine,
                             n_bootstrap=n_bootstrap, seed=seed)
    gof = compute_psychometric_gof(stim_v, ch_v, psych, n_bins)

    # CI band
    if show_fit and show_ci and n_bootstrap > 0 and psych['success']:
        ci = psych.get('y_fit_ci', (None, None))
        if ci[0] is not None:
            ax.fill_between(x_fine, ci[0], ci[1], color=color, alpha=0.2)

    # Fitted curve
    if show_fit and psych['success']:
        ax.plot(x_fine, psych['y_fit'], '-', color=color, linewidth=linewidth)

    # Reference lines
    if show_reference:
        ax.axhline(0.5, color='gray', ls=':', alpha=0.5)
        ax.axvline(0, color='gray', ls=':', alpha=0.5)

    # --- Text box ---
    lines = []
    if show_gof:
        cats = (stim_v > 0).astype(int)
        acc = (ch_v == cats).mean()
        lines.append(f"Acc: {acc:.1%}")
        lines.append(f"R2: {gof['r_squared']:.3f}")
    if show_params and psych['success']:
        mu_s = f"mu: {psych['mu']:.3f}"
        sig_s = f"sigma: {psych['sigma']:.3f}"
        if n_bootstrap > 0 and 'mu_ci' in psych:
            mu_s += f" [{psych['mu_ci'][0]:.3f}, {psych['mu_ci'][1]:.3f}]"
            sig_s += f" [{psych['sigma_ci'][0]:.3f}, {psych['sigma_ci'][1]:.3f}]"
        lines.extend([mu_s, sig_s])
    if show_lapse and psych['success']:
        lines.append(f"lapse_lo: {psych['lapse_low']:.3f}")
        lines.append(f"lapse_hi: {psych['lapse_high']:.3f}")
    if lines:
        pos = {'upper left': (0.05, 0.95, 'left', 'top'),
               'upper right': (0.95, 0.95, 'right', 'top'),
               'lower left': (0.05, 0.05, 'left', 'bottom'),
               'lower right': (0.95, 0.05, 'right', 'bottom')}
        xp, yp, ha, va = pos.get(text_position, (0.05, 0.95, 'left', 'top'))
        ax.text(xp, yp, '\n'.join(lines), transform=ax.transAxes,
                fontsize=8, ha=ha, va=va, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Stimulus')
    ax.set_ylabel('P(choose B)')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    if title:
        ax.set_title(title)

    info = {'psych_params': psych, 'gof': gof,
            'bin_centers': bin_centers, 'prop_B': prop_B,
            'prop_B_se': prop_B_se, 'counts': counts}
    return ax, info


# =============================================================================
# Core: overlay multiple curves on one axes
# =============================================================================

def plot_psychometric_overlay(
    stimuli_list: List[np.ndarray],
    choices_list: List[np.ndarray],
    labels: List[str],
    ax: Optional[plt.Axes] = None,
    colors: Optional[List[str]] = None,
    n_bins: int = 8,
    n_bootstrap: int = 0,
    show_ci: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 5),
    seed: int = 42,
) -> Tuple[plt.Axes, List[Dict]]:
    """
    Multiple psychometric curves overlaid on one axes.

    Useful for comparing conditions (e.g. opto on vs off, uniform vs hard-A).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if colors is None:
        colors = [f'C{i}' for i in range(len(stimuli_list))]

    infos = []
    for i, (stim, ch, lab) in enumerate(zip(stimuli_list, choices_list, labels)):
        _, info = plot_psychometric(
            stim, ch, ax=ax, color=colors[i], label=lab,
            n_bins=n_bins, n_bootstrap=n_bootstrap, show_ci=show_ci,
            show_params=False, show_gof=False, show_reference=False,
            seed=seed + i,
        )
        infos.append(info)

    ax.axhline(0.5, color='gray', ls=':', alpha=0.5)
    ax.axvline(0, color='gray', ls=':', alpha=0.5)
    ax.legend(loc='lower right')
    if title:
        ax.set_title(title)
    return ax, infos


# =============================================================================
# Session-aware: grid or pooled from SessionData list
# =============================================================================

def _extract_valid_trials(sessions, exclude_abort=True, exclude_opto=True,
                          min_trials=5):
    """Extract (stimuli, choices) per session, skipping sessions with too few."""
    stim_list, ch_list, labels, used_sessions = [], [], [], []
    for sess in sessions:
        arrays = sess.trials.get_model_arrays(
            exclude_abort=exclude_abort, exclude_opto=exclude_opto,
        )
        valid = ~arrays['no_response']
        if valid.sum() < min_trials:
            continue
        stim_list.append(arrays['stimuli'][valid])
        ch_list.append(arrays['choices'][valid])
        labels.append(f"S{sess.session_idx} ({sess.date})")
        used_sessions.append(sess)
    return stim_list, ch_list, labels, used_sessions


def plot_session_psychometrics(
    sessions: list,
    mode: str = 'grid',
    n_max: int = 12,
    ncols: int = 4,
    n_bins: int = 8,
    n_bootstrap: int = 0,
    show_params: bool = False,
    show_gof: bool = False,
    show_lapse: bool = False,
    show_ci: bool = True,
    color: Optional[str] = None,
    suptitle: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    seed: int = 42,
    exclude_abort: bool = True,
    exclude_opto: bool = True,
) -> Tuple[plt.Figure, Union[List[Dict], Dict]]:
    """
    Plot psychometric curves from a list of SessionData objects.

    Args:
        sessions: List of SessionData objects
        mode:
            'grid'    - one subplot per session, evenly sampled if > n_max.
                        Shows psychometric evolution across sessions.
            'pooled'  - all trials pooled into a single curve.
                        Shows overall psychometric for these sessions.
                        Set n_bootstrap=500+ for CIs.
            'overlay' - fit each session separately, overlay all fitted curves
                        on one axes. Colour gradient early->late, mean in black.
                        Shows session-to-session variability.
        n_max: Max sessions to show in grid (evenly samples if exceeded)
        ncols: Grid columns
        n_bins: Stimulus bins
        n_bootstrap: Bootstrap resamples (0 = no CIs)
        show_params: Show mu, sigma
        show_gof: Show accuracy, R2
        show_lapse: Show lapse rates
        show_ci: Show CI bands (needs n_bootstrap > 0)
        color: Plot colour (auto if None)
        suptitle: Figure title
        figsize: Figure size (auto if None)
        seed: Bootstrap seed
        exclude_abort: Remove abort trials
        exclude_opto: Remove opto trials

    Returns:
        (fig, infos) - infos is list of dicts (grid) or single dict (pooled)
    """
    if len(sessions) == 0:
        raise ValueError("No sessions provided")

    stim_list, ch_list, labels, used = _extract_valid_trials(
        sessions, exclude_abort, exclude_opto,
    )
    if len(stim_list) == 0:
        raise ValueError("No sessions with enough valid trials")

    # ---- Overlay mode ----
    if mode == 'overlay':
        if figsize is None:
            figsize = (7, 5)
        fig, ax = plt.subplots(figsize=figsize)

        x_fine = np.linspace(-1, 1, 100)
        n_sess = len(stim_list)
        cmap = plt.cm.viridis
        norm = plt.Normalize(0, max(n_sess - 1, 1))

        all_curves = []
        infos = []
        for i in range(n_sess):
            psych = fit_psychometric(stim_list[i], ch_list[i], x_fine)
            infos.append({'psych_params': psych, 'label': labels[i]})
            if psych['success']:
                c = cmap(norm(i))
                ax.plot(x_fine, psych['y_fit'], '-', color=c,
                        alpha=0.4, linewidth=1)
                all_curves.append(psych['y_fit'])

        # Mean curve + SD band
        if len(all_curves) >= 2:
            curve_arr = np.array(all_curves)  # (n_sessions, n_points)
            mean_curve = curve_arr.mean(axis=0)
            sd_curve = curve_arr.std(axis=0)
            ax.fill_between(x_fine, mean_curve - sd_curve, mean_curve + sd_curve,
                            color='black', alpha=0.12, label='±1 SD')
            ax.plot(x_fine, mean_curve, '-', color='black',
                    linewidth=2.5, label='mean')
            ax.legend(loc='lower right')

        ax.axhline(0.5, color='gray', ls=':', alpha=0.5)
        ax.axvline(0, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('Stimulus')
        ax.set_ylabel('P(choose B)')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.05, 1.05)

        # Colourbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Session')

        subtitle = f"{n_sess} sessions, per-session fits"
        if suptitle:
            ax.set_title(f"{suptitle}\n{subtitle}", fontsize=11)
        else:
            ax.set_title(subtitle)

        plt.tight_layout()
        return fig, infos

    # ---- Pooled mode ----
    if mode == 'pooled':
        all_stim = np.concatenate(stim_list)
        all_ch = np.concatenate(ch_list)

        if figsize is None:
            figsize = (6, 5)
        fig, ax = plt.subplots(figsize=figsize)

        _, info = plot_psychometric(
            all_stim, all_ch, ax=ax,
            n_bins=n_bins, n_bootstrap=n_bootstrap,
            show_ci=show_ci, show_params=show_params,
            show_gof=show_gof, show_lapse=show_lapse,
            color=color or 'C0', seed=seed,
        )

        n_sess = len(stim_list)
        n_trials = len(all_stim)
        subtitle = f"{n_sess} sessions, {n_trials} trials"
        if suptitle:
            ax.set_title(f"{suptitle}\n{subtitle}", fontsize=11)
        else:
            ax.set_title(subtitle)

        return fig, info

    # ---- Grid mode ----
    # Evenly sample if too many sessions
    n_sess = len(stim_list)
    if n_sess > n_max:
        step = n_sess / n_max
        indices = [int(round(i * step)) for i in range(n_max)]
        indices = [min(i, n_sess - 1) for i in indices]
        # Deduplicate while preserving order
        seen = set()
        indices = [i for i in indices if not (i in seen or seen.add(i))]
    else:
        indices = list(range(n_sess))

    n_show = len(indices)
    nrows = int(np.ceil(n_show / ncols))
    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    infos = []
    for plot_i, sess_i in enumerate(indices):
        ax = axes_flat[plot_i]
        _, info = plot_psychometric(
            stim_list[sess_i], ch_list[sess_i], ax=ax,
            n_bins=n_bins, n_bootstrap=n_bootstrap,
            show_ci=show_ci, show_params=show_params,
            show_gof=show_gof, show_lapse=show_lapse,
            color=color or 'C0', title=labels[sess_i],
            seed=seed + sess_i,
        )
        infos.append(info)

        # Only y-label on left column
        if plot_i % ncols != 0:
            ax.set_ylabel('')

    # Hide unused axes
    for i in range(n_show, len(axes_flat)):
        axes_flat[i].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()

    return fig, infos
