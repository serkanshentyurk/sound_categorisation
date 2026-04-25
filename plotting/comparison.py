"""
SBI Comparison Plotting

Plotting functions extracted from sbi_comparison_utils.py:
- CV violin + scatter (SBI version)
- Session-by-session update matrices
- Session-by-session psychometric curves
- Pooled UM comparison

Uses behav_utils.plotting.styles for colour consistency.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any

from behav_utils.plotting.styles import COLOURS, UM_CMAP
from behav_utils.analysis.utils import cumulative_gaussian

BE_COLOUR = COLOURS['BE']
SC_COLOUR = COLOURS['SC']


# =============================================================================
# CV VIOLIN + SCATTER (SBI version)
# =============================================================================

def plot_sbi_cv_comparison(
    be_cv: Dict,
    sc_cv: Dict,
    comparison: Dict,
    animal_id: str,
    title_suffix: str = '',
    figsize: Tuple[float, float] = (12, 4.5),
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """Paired violin + scatter plot of SBI CV test errors."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    n = min(len(be_cv['test_errors']), len(sc_cv['test_errors']))

    ax = axes[0]
    for i in range(n):
        ax.plot(
            [0, 1],
            [be_cv['test_errors'][i], sc_cv['test_errors'][i]],
            'o-', color='grey', alpha=0.15, markersize=3,
        )
    parts = ax.violinplot(
        [be_cv['test_errors'], sc_cv['test_errors']],
        positions=[0, 1], showmedians=True,
    )
    for pc, col in zip(parts['bodies'], [BE_COLOUR, SC_COLOUR]):
        pc.set_facecolor(col)
        pc.set_alpha(0.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['BE', 'SC'])
    ax.set_ylabel('Test UM MSE')
    ax.set_title(f'p={comparison["p_value"]:.3g}, winner={comparison["winner"]}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    ax.scatter(
        be_cv['test_errors'][:n], sc_cv['test_errors'][:n],
        alpha=0.4, s=20, c='grey',
    )
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
    ax.set_xlabel('BE test error')
    ax.set_ylabel('SC test error')
    ax.set_title(f'BE={comparison["be_mean"]:.5f}, SC={comparison["sc_mean"]:.5f}')
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle(
        f'{animal_id} — SBI CV {title_suffix}',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    if output_dir:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_dir}/sbi_cv_{animal_id}.png',
                    dpi=200, bbox_inches='tight')

    return fig


# =============================================================================
# SESSION-BY-SESSION UPDATE MATRICES
# =============================================================================

def plot_session_by_session_um(
    session_data: List[Dict],
    animal_id: str,
    max_sessions: int = 20,
    figscale: float = 2.5,
    output_dir: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Grid of update matrices: rows = sessions, columns = Real | BE | SC.
    """
    from behav_utils.plotting.update_matrix import plot_update_matrix as _plot_um

    data = session_data[:max_sessions]
    n_sess = len(data)

    if n_sess == 0:
        return None

    # Shared colour scale
    all_ums = []
    for d in data:
        for um in [d['real_um'], d['be_um'], d['sc_um']]:
            if um is not None and not np.all(np.isnan(um)):
                all_ums.append(np.nanmax(np.abs(um)))
    vlim = max(all_ums) if all_ums else 0.3

    fig, axes = plt.subplots(
        n_sess, 3, figsize=(figscale * 3, figscale * n_sess),
        squeeze=False,
    )

    for row, d in enumerate(data):
        acc = d['accuracy']
        sid = d['session_idx']

        for col, (um, label) in enumerate([
            (d['real_um'], f"Real (acc={acc:.0%})"),
            (d['be_um'], f"BE (MSE={d['be_um_mse']:.4f})"),
            (d['sc_um'], f"SC (MSE={d['sc_um_mse']:.4f})"),
        ]):
            ax = axes[row, col]
            if um is not None and not np.all(np.isnan(um)):
                _plot_um(um, ax=ax, vmin=-vlim, vmax=vlim, show_colorbar=False)
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')

            if row == 0:
                ax.set_title(['Real', 'BE', 'SC'][col],
                             fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'S{sid}', fontsize=9, fontweight='bold')
            else:
                ax.set_ylabel('')

            if col > 0:
                mse_val = d['be_um_mse'] if col == 1 else d['sc_um_mse']
                if not np.isnan(mse_val):
                    colour = BE_COLOUR if col == 1 else SC_COLOUR
                    ax.text(0.02, 0.98, f'{mse_val:.4f}',
                            transform=ax.transAxes, fontsize=7,
                            va='top', ha='left', color=colour)

            if row < n_sess - 1:
                ax.set_xlabel('')
            ax.tick_params(labelsize=6)

    fig.suptitle(
        f'{animal_id} — Session-by-session Update Matrices',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()

    if output_dir:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_dir}/session_um_{animal_id}.png',
                    dpi=200, bbox_inches='tight')

    return fig


# =============================================================================
# SESSION-BY-SESSION PSYCHOMETRIC CURVES
# =============================================================================

def plot_session_by_session_psychometric(
    session_data: List[Dict],
    animal_id: str,
    max_sessions: int = 20,
    n_cols: int = 4,
    figscale: float = 3.0,
    output_dir: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Grid of psychometric curves: Real (black) + BE + SC overlaid.
    """
    data = session_data[:max_sessions]
    n_sess = len(data)

    if n_sess == 0:
        return None

    n_rows = int(np.ceil(n_sess / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figscale * n_cols, figscale * n_rows),
        squeeze=False,
    )

    x_fine = np.linspace(-1.1, 1.1, 200)

    for idx, d in enumerate(data):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        acc = d['accuracy']
        sid = d['session_idx']

        # Real
        p = d['real_psych']
        if p.get('success', False):
            y = cumulative_gaussian(
                x_fine, p['mu'], p['sigma'],
                p['lapse_low'], p['lapse_high'],
            )
            ax.plot(x_fine, y, 'k-', lw=2, label='Real')

        # BE
        p = d['be_psych']
        if p.get('success', False):
            y = cumulative_gaussian(
                x_fine, p['mu'], p['sigma'],
                p['lapse_low'], p['lapse_high'],
            )
            ax.plot(x_fine, y, '-', color=BE_COLOUR, lw=1.5, label='BE')

        # SC
        p = d['sc_psych']
        if p.get('success', False):
            y = cumulative_gaussian(
                x_fine, p['mu'], p['sigma'],
                p['lapse_low'], p['lapse_high'],
            )
            ax.plot(x_fine, y, '-', color=SC_COLOUR, lw=1.5, label='SC')

        ax.axhline(0.5, color='grey', ls='--', alpha=0.3, lw=0.5)
        ax.axvline(0, color='grey', ls='--', alpha=0.3, lw=0.5)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'S{sid} (acc={acc:.0%})', fontsize=9)
        ax.tick_params(labelsize=7)

        if idx == 0:
            ax.legend(fontsize=7, loc='lower right')
        if col > 0:
            ax.set_yticklabels([])
        if row < n_rows - 1:
            ax.set_xticklabels([])

    for idx in range(n_sess, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f'{animal_id} — Session-by-session Psychometric Curves',
        fontsize=14, fontweight='bold',
    )
    fig.text(0.5, 0.01, 'Stimulus', ha='center', fontsize=11)
    fig.text(0.01, 0.5, 'P(choose B)', va='center',
             rotation='vertical', fontsize=11)
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.97])

    if output_dir:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_dir}/session_psych_{animal_id}.png',
                    dpi=200, bbox_inches='tight')

    return fig


# =============================================================================
# POOLED UM COMPARISON
# =============================================================================

def plot_pooled_um_comparison(
    session_data: List[Dict],
    animal_id: str,
    n_bins: int = 8,
    output_dir: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Pooled update matrix comparison across all sessions:
    Real | BE | SC, each averaged across sessions.
    """
    from behav_utils.plotting.update_matrix import plot_update_matrix as _plot_um
    from behav_utils.analysis.update_matrix import matrix_error

    real_ums = [d['real_um'] for d in session_data
                if not np.all(np.isnan(d['real_um']))]
    be_ums = [d['be_um'] for d in session_data
              if not np.all(np.isnan(d['be_um']))]
    sc_ums = [d['sc_um'] for d in session_data
              if not np.all(np.isnan(d['sc_um']))]

    real_mean = np.nanmean(real_ums, axis=0) if real_ums else None
    be_mean = np.nanmean(be_ums, axis=0) if be_ums else None
    sc_mean = np.nanmean(sc_ums, axis=0) if sc_ums else None

    if real_mean is None:
        return None

    be_mse = matrix_error(be_mean, real_mean) if be_mean is not None else np.nan
    sc_mse = matrix_error(sc_mean, real_mean) if sc_mean is not None else np.nan

    vlim = max(
        np.nanmax(np.abs(real_mean)),
        np.nanmax(np.abs(be_mean)) if be_mean is not None else 0,
        np.nanmax(np.abs(sc_mean)) if sc_mean is not None else 0,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, um, title in [
        (axes[0], real_mean, f'Real (n={len(real_ums)} sessions)'),
        (axes[1], be_mean, f'BE (MSE={be_mse:.5f})'),
        (axes[2], sc_mean, f'SC (MSE={sc_mse:.5f})'),
    ]:
        if um is not None:
            _plot_um(um, ax=ax, vmin=-vlim, vmax=vlim)
        ax.set_title(title, fontsize=10)

    fig.suptitle(
        f'{animal_id} — Pooled UM Comparison',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    if output_dir:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_dir}/pooled_um_{animal_id}.png',
                    dpi=200, bbox_inches='tight')

    return fig


# =============================================================================
# SINGLE-SESSION EXAMPLE PLOT
# =============================================================================

def plot_example_session(
    data: Dict,
    animal_id: str,
    n_bins: int = 8,
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """
    4-panel session comparison + 3-panel UM comparison.

    Args:
        data: Output from inference.comparison.simulate_example_session
        animal_id: For title
        n_bins: Bins for update matrix
    """
    from behav_utils.plotting.update_matrix import plot_update_matrix as _plot_um
    from behav_utils.analysis.update_matrix import compute_update_matrix, matrix_error
    from behav_utils.analysis.psychometry import fit_psychometric

    stim, cat, ch = data['stimuli'], data['categories'], data['choices']
    info = data['session_info']
    n = len(stim)
    trials = np.arange(n)
    correct = (ch == cat).astype(bool)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Trial-by-trial p(B)
    ax = axes[0, 0]
    ax.scatter(trials, ch, c=ch, cmap='coolwarm', s=8, alpha=0.4)
    ax.plot(trials, data['be_pB'], '-', color=BE_COLOUR, lw=1.2, alpha=0.8, label='BE')
    ax.plot(trials, data['sc_pB'], '-', color=SC_COLOUR, lw=1.2, alpha=0.8, label='SC')
    ax.axhline(0.5, color='grey', ls='--', alpha=0.3)
    ax.set_xlabel('Trial')
    ax.set_ylabel('P(B)')
    ax.set_title('Trial-by-trial choice probability')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # Stimulus sequence
    ax = axes[0, 1]
    ax.scatter(trials[correct], stim[correct], c='green', s=6, alpha=0.5, label='Correct')
    ax.scatter(trials[~correct], stim[~correct], c='red', s=6, alpha=0.5, label='Error')
    ax.axhline(0, color='grey', ls='--', alpha=0.3)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Stimulus')
    ax.set_title('Stimulus sequence')
    ax.legend(fontsize=8)

    # Psychometric curves
    ax = axes[1, 0]
    x_fine = np.linspace(-1.1, 1.1, 200)
    real_psych = fit_psychometric(stim, ch)
    if real_psych.get('success', False):
        ax.plot(x_fine, cumulative_gaussian(
            x_fine, real_psych['mu'], real_psych['sigma'],
            real_psych['lapse_low'], real_psych['lapse_high'],
        ), 'k-', lw=2.5, label='Real', zorder=10)

    for label, calls, col in [
        ('BE', data['be_choices_all'], BE_COLOUR),
        ('SC', data['sc_choices_all'], SC_COLOUR),
    ]:
        mc = np.nanmean(calls, axis=0)
        p = fit_psychometric(stim, mc)
        if p.get('success', False):
            ax.plot(x_fine, cumulative_gaussian(
                x_fine, p['mu'], p['sigma'], p['lapse_low'], p['lapse_high'],
            ), '-', color=col, lw=2, label=label)

    ax.axhline(0.5, color='grey', ls='--', alpha=0.3)
    ax.axvline(0, color='grey', ls='--', alpha=0.3)
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('P(B)')
    ax.set_title('Psychometric curves')
    ax.legend(fontsize=8)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.05, 1.05)

    # Real UM
    real_um, _, _ = compute_update_matrix(stim, ch, cat, n_bins=n_bins)
    _plot_um(real_um, ax=axes[1, 1], title='Real UM (this session)')

    fig.suptitle(
        f'{animal_id} — {info["session_id"]} '
        f'({info["n_trials"]} trials, acc={info["accuracy"]:.0%})',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    # 3-panel UM comparison
    be_ums = [
        compute_update_matrix(stim[~np.isnan(c)], c[~np.isnan(c)],
                              cat[~np.isnan(c)], n_bins)[0]
        for c in data['be_choices_all']
        if (~np.isnan(c)).sum() > 50
    ]
    sc_ums = [
        compute_update_matrix(stim[~np.isnan(c)], c[~np.isnan(c)],
                              cat[~np.isnan(c)], n_bins)[0]
        for c in data['sc_choices_all']
        if (~np.isnan(c)).sum() > 50
    ]
    be_mu = np.nanmean(be_ums, axis=0) if be_ums else None
    sc_mu = np.nanmean(sc_ums, axis=0) if sc_ums else None

    fig2 = None
    if be_mu is not None and sc_mu is not None:
        fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4.5))
        vlim = max(
            np.nanmax(np.abs(real_um)),
            np.nanmax(np.abs(be_mu)),
            np.nanmax(np.abs(sc_mu)),
        )
        for a, u, t in [
            (ax2[0], real_um, 'Real'),
            (ax2[1], be_mu, f'BE (MSE={matrix_error(be_mu, real_um):.5f})'),
            (ax2[2], sc_mu, f'SC (MSE={matrix_error(sc_mu, real_um):.5f})'),
        ]:
            _plot_um(u, ax=a, vmin=-vlim, vmax=vlim)
            a.set_title(t)
        fig2.suptitle('Session UM Comparison', fontsize=12)
        plt.tight_layout()

    if output_dir:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_dir}/example_session_{animal_id}.png',
                    dpi=200, bbox_inches='tight')
        if fig2:
            fig2.savefig(f'{output_dir}/example_session_um_{animal_id}.png',
                         dpi=200, bbox_inches='tight')

    return fig
