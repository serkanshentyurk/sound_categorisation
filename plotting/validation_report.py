"""
Synthetic Validation Report — Plotting

Pure rendering. Each function takes a result dict from the corresponding
compute_ function in analysis/validation_report.py.

Public API:
    plot_synth_summary          — Psychometric + UM panels
    plot_synth_model_fits       — UM comparison + psychometric + MSE (same as animal_report)
    plot_recovery_overlay       — True vs recovered scatter per param
    plot_recovery_summary       — Correlation bars across methods
    plot_confusion_matrix       — Heatmap from build_confusion_matrix result

Usage:
    from analysis.validation_report import compute_synth_summary
    from plotting.validation_report import plot_synth_summary

    result = compute_synth_summary(sa)
    plot_synth_summary(result)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from behav_utils.plotting.psychometric import plot_psychometric
from behav_utils.plotting.update_matrix import plot_um
from behav_utils.plotting.styles import COLOURS, PALETTE
from behav_utils.analysis.utils import cumulative_gaussian

BE_COL = COLOURS.get('BE', '#2196F3')
SC_COL = COLOURS.get('SC', '#FF9800')


# ═════════════════════════════════════════════════════════════════════════════
# 1. SYNTH SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def plot_synth_summary(result: dict) -> plt.Figure:
    """
    Draw psychometric + UM for a synthetic animal.

    Args:
        result: Dict from compute_synth_summary().
    """
    aid = result['animal_id']
    mt = result['true_model']
    col = BE_COL if mt == 'BE' else SC_COL

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    plot_psychometric(result['psychometric'], ax=axes[0],
                      color=col, show_ci=True, show_params=True)
    plot_um(result['um'], ax=axes[1], title='Update Matrix')

    from analysis.validation_report import _params_to_str
    fig.suptitle(
        f'{aid} — True {mt} ({result["n_sessions"]} sessions, {result["n_trials"]} trials)\n'
        f'{_params_to_str(result["true_params"])}',
        fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 2. SYNTH MODEL FITS (reuses animal_report.plot_model_fits pattern)
# ═════════════════════════════════════════════════════════════════════════════

def plot_synth_model_fits(result: dict) -> Dict[str, plt.Figure]:
    """
    Draw model fit panels for a synthetic animal.

    Same layout as plotting.animal_report.plot_model_fits but adds
    true-model marker.

    Args:
        result: Dict from compute_synth_model_fits().
    """
    from analysis.validation_report import _params_to_str, FT_LABEL

    aid = result['animal_id']
    mt = result['true_model']
    ft = FT_LABEL.get(result['fit_target'], result['fit_target'])
    figs = {}

    # UM comparison
    fig_um, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, um, lbl in [
        (axes[0], result['emp_um'], f'Real'),
        (axes[1], result['be_um'], f'BE MSE={result["be_mse"]:.5f}' + (' ★' if mt == 'BE' else '')),
        (axes[2], result['sc_um'], f'SC MSE={result["sc_mse"]:.5f}' + (' ★' if mt == 'SC' else '')),
    ]:
        plot_um(um, ax=ax, title=lbl)
    fig_um.suptitle(f'{aid} — UM (GS-{ft}, true: {mt})', fontsize=13, fontweight='bold')
    fig_um.tight_layout()
    figs['um'] = fig_um

    # Psychometric overlay
    fig_psych, ax = plt.subplots(figsize=(7, 5))
    plot_psychometric(result['emp_psych'], ax=ax, color='black', label='Data', show_ci=True)

    x_fine = np.linspace(-1.1, 1.1, 200)
    for mk, col, params in [
        ('BE', BE_COL, result['be_psych_params']),
        ('SC', SC_COL, result['sc_psych_params']),
    ]:
        if params and 'mu' in params and not np.isnan(params['mu']):
            y = cumulative_gaussian(x_fine, params['mu'], params['sigma'],
                                    params.get('lapse_low', 0), params.get('lapse_high', 0))
            star = ' ★' if mk == mt else ''
            ax.plot(x_fine, y, '-', color=col, lw=2,
                    label=f'{mk} (PSE={params["mu"]:.3f}){star}')
    ax.legend(fontsize=9)
    ax.set_title(f'{aid} — Psychometric (GS-{ft}, true: {mt})', fontsize=12, fontweight='bold')
    fig_psych.tight_layout()
    figs['psychometric'] = fig_psych

    # Per-session MSE
    fig_mse, ax = plt.subplots(figsize=(10, 4))
    mse_data = result['per_session_mse']
    si = [d['session_idx'] for d in mse_data]
    be_mses = [d['be_mse'] for d in mse_data]
    sc_mses = [d['sc_mse'] for d in mse_data]
    ax.plot(si, be_mses, 'o-', color=BE_COL, label='BE', ms=5)
    ax.plot(si, sc_mses, 'o-', color=SC_COL, label='SC', ms=5)
    bmu, smu = np.nanmean(be_mses), np.nanmean(sc_mses)
    w = 'BE' if bmu < smu else 'SC'
    bw = sum(b < s for b, s in zip(be_mses, sc_mses))
    ax.axhline(bmu, color=BE_COL, ls='--', alpha=0.4)
    ax.axhline(smu, color=SC_COL, ls='--', alpha=0.4)
    ax.set(xlabel='Session', ylabel='UM MSE')
    ax.set_title(f'{aid} — Per-Session (GS-{ft}, true: {mt})\n'
                 f'BE={bmu:.5f}, SC={smu:.5f} → {w} (BE wins {bw}/{len(be_mses)})',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    fig_mse.tight_layout()
    figs['mse'] = fig_mse

    print(f'  BE: {_params_to_str(result["be_params"])}')
    print(f'  SC: {_params_to_str(result["sc_params"])}')

    return figs


# ═════════════════════════════════════════════════════════════════════════════
# 3. RECOVERY PLOTS (already take pre-computed dicts — just fix imports)
# ═════════════════════════════════════════════════════════════════════════════

def plot_recovery_overlay(
    recovery: dict,
    model_type: str = 'BE',
    methods: Optional[List[Tuple[str, dict, str]]] = None,
) -> plt.Figure:
    """
    True vs recovered scatter for each parameter.

    Args:
        recovery: {model_type: {param_name: {'true': [...], 'recovered': [...]}}}
        model_type: 'BE' or 'SC'.
        methods: List of (label, recovery_dict, colour) to overlay.
                 If None, plots recovery[model_type] as a single method.
    """
    mt = model_type.upper()
    if methods is None:
        methods = [(mt, recovery.get(mt, {}), BE_COL if mt == 'BE' else SC_COL)]

    all_params = sorted(set(
        pn for _, src, _ in methods for pn in src
    ))
    if not all_params:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No recovery data', transform=ax.transAxes, ha='center')
        return fig

    n_p = len(all_params)
    fig, axes = plt.subplots(1, n_p, figsize=(5 * n_p, 4.5))
    axes = np.atleast_1d(axes)

    for j, pn in enumerate(all_params):
        ax = axes[j]
        for label, src, col in methods:
            if pn not in src:
                continue
            t = np.array(src[pn]['true'])
            r = np.array(src[pn]['recovered'])
            ax.scatter(t, r, c=col, s=30, alpha=0.6, label=label, edgecolor='white', lw=0.5)
            corr = np.corrcoef(t, r)[0, 1] if len(t) > 2 else np.nan
            ax.set_title(f'{pn}\nr={corr:.3f}', fontsize=10)

        rng = ax.get_xlim()
        ax.plot(rng, rng, 'k--', alpha=0.3, zorder=0)
        ax.set_xlabel('True')
        ax.set_ylabel('Recovered')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{mt} — Parameter Recovery', fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = '',
    labels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw a confusion matrix heatmap.

    Args:
        cm: 2D ndarray from build_confusion_matrix().
        title: Axes title.
        labels: Class labels (default: ['BE', 'SC']).
    """
    labels = labels or ['BE', 'SC']
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
    else:
        fig = ax.get_figure()

    cm_n = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1, aspect='equal')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            c = 'white' if cm_n[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i, j]}\n({cm_n[i, j] * 100:.0f}%)',
                    ha='center', va='center', fontsize=12, fontweight='bold', color=c)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    for s in ax.spines.values():
        s.set_visible(False)

    if not title:
        total = cm.sum()
        title = f'{cm.trace()}/{total} ({cm.trace() / total:.0%})'
    ax.set_title(title, fontsize=10, fontweight='bold')

    return fig, ax
