"""
Per-Animal Report — Plotting

Pure rendering functions. Each takes a result dict from the corresponding
compute_ function in analysis/animal_report.py. No data loading, no
simulation, no filtering, no fitting.

Public API:
    plot_animal_summary       — Psychometric + accuracy trajectory + consensus strip
    plot_cv_results           — Wraps plotting.cv (unchanged)
    plot_model_fits           — UM comparison + psychometric overlay + per-session MSE
    plot_sbi_diagnostics      — Posterior marginals + PPC

Usage:
    from analysis.animal_report import compute_animal_summary, compute_model_fits
    from plotting.animal_report import plot_animal_summary, plot_model_fits

    summary = compute_animal_summary('SS05', sessions, RESULTS_DIR)
    plot_animal_summary(summary)

    fits = compute_model_fits('SS05', sessions, RESULTS_DIR, method='sbi')
    plot_model_fits(fits)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from behav_utils.plotting.psychometric import plot_psychometric
from behav_utils.plotting.update_matrix import plot_um
from behav_utils.plotting.trajectory import plot_trajectory
from behav_utils.plotting.styles import COLOURS, PALETTE
from behav_utils.analysis.utils import cumulative_gaussian

BE_COL = COLOURS.get('BE', '#2196F3')
SC_COL = COLOURS.get('SC', '#FF9800')


# ═════════════════════════════════════════════════════════════════════════════
# 1. ANIMAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def plot_animal_summary(result: dict) -> plt.Figure:
    """
    Draw animal summary: psychometric + accuracy trajectory + consensus strip.

    Args:
        result: Dict from compute_animal_summary().

    Returns:
        Figure.
    """
    animal_id = result['animal_id']
    distribution = result['distribution']
    n_sessions = result['n_sessions']
    n_trials = result['n_trials']
    true_params = result.get('true_params')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5),
                             gridspec_kw={'width_ratios': [1, 1, 0.8]})

    # Panel 1: Psychometric
    plot_psychometric(result['psychometric'], ax=axes[0],
                      color='black', show_params=True, title='Psychometric')

    # Panel 2: Accuracy trajectory
    plot_trajectory(result['trajectory'], stat_name='accuracy', ax=axes[1],
                    color='black', title='Accuracy')
    mean_acc = np.nanmean(result['trajectory']['values']['accuracy'])
    axes[1].axhline(mean_acc, color='grey', ls='--', alpha=0.5)
    axes[1].set_ylim(0.4, 1.0)
    axes[1].set_title(f'Accuracy ({mean_acc:.1%} mean)', fontsize=10)

    # Panel 3: Consensus strip
    _draw_consensus_strip(result['consensus'], animal_id, ax=axes[2])

    label = distribution.replace('_', ' ').title()
    title = f'{animal_id} — {label} ({n_sessions} sessions, {n_trials} trials)'
    if true_params:
        from analysis.animal_report import _params_to_str
        title += f'\nTrue: {_params_to_str(true_params)}'
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def _draw_consensus_strip(consensus_info, animal_id, ax=None):
    """Draw the method consensus strip panel."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2))

    methods = consensus_info.get('methods', {})
    method_names = consensus_info.get('method_names', [])
    consensus_label = consensus_info.get('consensus', 'Unknown')

    n_m = len(method_names)
    if n_m == 0:
        ax.text(0.5, 0.5, 'No consensus data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10)
        return

    for j, mn in enumerate(method_names):
        val = methods.get(mn)
        if val == 'BE':
            colour, label = BE_COL, 'BE'
        elif val == 'SC':
            colour, label = SC_COL, 'SC'
        else:
            colour, label = '#DDDDDD', '?'

        ax.add_patch(plt.Rectangle((j - 0.4, -0.4), 0.8, 0.8,
                                    facecolor=colour, edgecolor='white', lw=2))
        text_col = 'white' if val in ('BE', 'SC') else 'grey'
        ax.text(j, 0, label, ha='center', va='center',
                fontsize=12, fontweight='bold', color=text_col)

    ax.set_xlim(-0.6, n_m - 0.4)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xticks(range(n_m))
    ax.set_xticklabels(method_names, fontsize=9, rotation=15)
    ax.set_yticks([])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)

    if consensus_label in ('BE', 'SC'):
        sig_count = sum(1 for v in methods.values() if v == consensus_label)
        ax.set_title(f'{animal_id} — {sig_count}/{n_m} say {consensus_label}',
                     fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'{animal_id} — {consensus_label}',
                     fontsize=12, fontweight='bold')


# ═════════════════════════════════════════════════════════════════════════════
# 2. CV RESULTS (delegates to plotting.cv, no changes needed)
# ═════════════════════════════════════════════════════════════════════════════

def plot_cv_results(
    animal_id: str,
    results_dir: Union[str, Path],
    distribution: str = 'uniform',
    fit_target: str = 'update_matrix',
) -> Optional[plt.Figure]:
    """
    GS CV seed-level comparison. Wraps plotting.cv.plot_cv_comparison.

    This function loads data internally (pickle loading is tightly coupled
    to the CV output format). Not worth splitting for now.
    """
    from plotting.cv import plot_cv_comparison, build_cv_dataframes

    results_dir = Path(results_dir)
    long_df, comparison_df = build_cv_dataframes(
        animal_id, results_dir, distribution, fit_target)

    if long_df is None:
        print(f'  No GS data for {animal_id} ({distribution}/{fit_target})')
        return None

    fig = plot_cv_comparison(long_df, comparison_df, animal_id,
                             fit_target=fit_target)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 3. MODEL FITS
# ═════════════════════════════════════════════════════════════════════════════

def plot_model_fits(result: dict) -> Dict[str, plt.Figure]:
    """
    Draw model fit comparison panels from a compute_model_fits() result.

    Produces three figures:
        1. UM comparison: empirical vs BE vs SC (3 panels)
        2. Psychometric overlay: data + BE curve + SC curve
        3. Per-session MSE trajectory: BE vs SC

    Args:
        result: Dict from compute_model_fits().

    Returns:
        Dict with 'um', 'psychometric', 'mse' figure keys.
    """
    animal_id = result['animal_id']
    distribution = result['distribution']
    source = result['source']
    src_label = f"{source['method'].upper()}-{source['fit_target']}"
    if source.get('fallback'):
        src_label += ' (fallback)'
    label = distribution.replace('_', ' ').title()

    figs = {}

    # ── Figure 1: UM comparison ──────────────────────────────────────────
    fig_um, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, um, title_str in [
        (axes[0], result['emp_um'], f'Real (n={len(result["session_data"])}s)'),
        (axes[1], result['be_um'], f'BE (MSE={result["be_mse"]:.5f})'),
        (axes[2], result['sc_um'], f'SC (MSE={result["sc_mse"]:.5f})'),
    ]:
        plot_um(um, ax=ax, title=title_str)
    fig_um.suptitle(f'{animal_id} — UM Comparison ({label}, {src_label})',
                    fontsize=13, fontweight='bold')
    fig_um.tight_layout()
    figs['um'] = fig_um

    # ── Figure 2: Psychometric overlay ───────────────────────────────────
    fig_psych, ax = plt.subplots(figsize=(7, 5))

    # Data psychometric (from compute_psychometric result)
    plot_psychometric(result['emp_psych'], ax=ax,
                      color='black', label='Data', show_ci=True)

    # Model curves overlaid
    x_fine = np.linspace(-1.1, 1.1, 200)
    for mk, col, params in [
        ('BE', BE_COL, result['be_psych_params']),
        ('SC', SC_COL, result['sc_psych_params']),
    ]:
        if params and 'mu' in params and not np.isnan(params['mu']):
            y = cumulative_gaussian(x_fine, params['mu'], params['sigma'],
                                    params.get('lapse_low', 0),
                                    params.get('lapse_high', 0))
            ax.plot(x_fine, y, '-', color=col, lw=2,
                    label=f'{mk} (PSE={params["mu"]:.3f})')

    ax.legend(fontsize=9)
    ax.set_title(f'{animal_id} — Psychometric ({label}, {src_label})',
                 fontsize=12, fontweight='bold')
    fig_psych.tight_layout()
    figs['psychometric'] = fig_psych

    # ── Figure 3: Per-session MSE ────────────────────────────────────────
    fig_mse, ax = plt.subplots(figsize=(10, 4))
    mse_data = result['per_session_mse']
    si = [d['session_idx'] for d in mse_data]
    be_mses = [d['be_mse'] for d in mse_data]
    sc_mses = [d['sc_mse'] for d in mse_data]

    ax.plot(si, be_mses, 'o-', color=BE_COL, label='BE', ms=5)
    ax.plot(si, sc_mses, 'o-', color=SC_COL, label='SC', ms=5)

    bmu, smu = np.nanmean(be_mses), np.nanmean(sc_mses)
    ax.axhline(bmu, color=BE_COL, ls='--', alpha=0.4)
    ax.axhline(smu, color=SC_COL, ls='--', alpha=0.4)

    winner = 'BE' if bmu < smu else 'SC'
    be_wins = sum(b < s for b, s in zip(be_mses, sc_mses))
    ax.set(xlabel='Session index', ylabel='UM MSE')
    ax.set_title(
        f'{animal_id} — Per-Session MSE ({label}, {src_label})\n'
        f'BE={bmu:.5f}, SC={smu:.5f} → {winner} (BE wins {be_wins}/{len(be_mses)})',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    fig_mse.tight_layout()
    figs['mse'] = fig_mse

    # ── Print parameter comparison ───────────────────────────────────────
    from analysis.animal_report import _params_to_str
    print(f'  BE params: {_params_to_str(result["be_params"])}')
    print(f'  SC params: {_params_to_str(result["sc_params"])}')

    return figs


# ═════════════════════════════════════════════════════════════════════════════
# 4. SBI DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════════════

def plot_sbi_diagnostics(result: dict) -> Dict[str, plt.Figure]:
    """
    Draw SBI posterior marginals and PPC from a compute_sbi_diagnostics() result.

    Args:
        result: Dict from compute_sbi_diagnostics().

    Returns:
        Dict with figure keys per model ('be_marginals', 'sc_marginals', etc.)
    """
    animal_id = result['animal_id']
    true_params = result.get('true_params')
    gs_params = result.get('gs_params')
    figs = {}

    for model_key in ('be', 'sc'):
        if model_key not in result['samples']:
            continue

        samps = result['samples'][model_key]
        pnames = result['param_names'][model_key]
        col = BE_COL if model_key == 'be' else SC_COL

        n_params = len(pnames)
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
        axes = np.atleast_1d(axes)

        for j, pn in enumerate(pnames):
            ax = axes[j]
            vals = samps[:, j]
            ax.hist(vals, bins=50, color=col, alpha=0.4,
                    density=True, edgecolor='none')

            med = np.median(vals)
            lo, hi = np.percentile(vals, [2.5, 97.5])
            ax.axvline(med, color=col, lw=2, label=f'Median={med:.3f}')
            ax.axvspan(lo, hi, alpha=0.12, color=col)

            # True value
            if true_params:
                true_val = true_params.get(pn)
                if true_val is not None:
                    ax.axvline(true_val, color='red', lw=2, ls='--',
                               label=f'True={true_val:.3f}')

            # GS point estimates
            if gs_params:
                for ft_key, style in [
                    (f'{model_key}_update_matrix', ('green', '-', 'GS-UM')),
                    (f'{model_key}_conditional_psych', ('green', '--', 'GS-CP')),
                ]:
                    gs_p = gs_params.get(ft_key, {})
                    if pn in gs_p:
                        ax.axvline(gs_p[pn], color=style[0], lw=2, ls=style[1],
                                   label=f'{style[2]}={gs_p[pn]:.3f}')

            ax.set_xlabel(pn, fontsize=11)
            ax.set_title(f'{med:.3f} [{lo:.3f}, {hi:.3f}]', fontsize=8)
            ax.legend(fontsize=7)

        fig.suptitle(f'{animal_id} — {model_key.upper()} Posteriors',
                     fontsize=13, fontweight='bold')
        fig.tight_layout()
        figs[f'{model_key}_marginals'] = fig

    return figs
