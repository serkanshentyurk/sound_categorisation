"""
Optogenetic effect plotting.

Visualisations for within-session opto vs control comparison,
cross-phase stability, update matrix comparison,
genotype interaction, model-assignment grouping,
equivalence testing, and phase × opto interaction.

Colours imported from behav_utils.plotting.styles for consistency.
UM_CMAP used for update matrix heatmaps (matching all other UM plots).

Public API:
    plot_opto_psychometric       — Overlaid opto/control psychometric curves
    plot_phase_trajectory        — Stat across experimental phases
    plot_opto_um_comparison      — Side-by-side opto vs control UMs
    plot_expert_stability        — Baseline → opto → washout dot plot
    plot_within_session_summary  — Per-session effect sizes across sessions
    plot_genotype_interaction    — Het vs WT interaction bar plot
    plot_model_assignment_effects — Opto effects grouped by BE/SC/unclear
    plot_equivalence_test        — TOST result with CI and bounds
    plot_phase_interaction       — Paired expert vs post-shift opto effects
    plot_animal_opto_report      — Full-page report for one animal

Usage:
    from analysis.opto import assign_opto_phases, phase_pooled_comparison
    from plotting.opto import plot_opto_um_comparison

    phases = assign_opto_phases(animal)
    comp = phase_pooled_comparison(animal.sessions, phases, OptoPhase.EXPERT_OPTO)
    fig = plot_opto_um_comparison(comp['opto_um'], comp['control_um'])
"""

from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.analysis.utils import cumulative_gaussian
from behav_utils.plotting.styles import COLOURS, UM_CMAP, apply_style

from analysis.opto import (
    OptoPhase, split_trials_by_opto, get_post_opto_mask,
    extract_trial_arrays,
)

apply_style()

# ─── Colours ─────────────────────────────────────────────────────────────────

OPTO_COLOUR = '#E24A33'
POST_OPTO_COLOUR = '#8B6DAF'
CTRL_COLOUR = '#348ABD'
BE_COLOUR = COLOURS.get('BE', 'steelblue')
SC_COLOUR = COLOURS.get('SC', 'darkorange')
UNCLEAR_COLOUR = '#999999'

MODEL_COLOURS = {
    'BE': BE_COLOUR,
    'SC': SC_COLOUR,
    'unclear': UNCLEAR_COLOUR,
}

PHASE_COLOURS = {
    OptoPhase.EXPERT_BASELINE: COLOURS.get('BE', '#5DA5DA'),
    OptoPhase.EXPERT_OPTO: '#E24A33',
    OptoPhase.EXPERT_WASHOUT: '#60BD68',
    OptoPhase.MASKING: '#AAAAAA',
    OptoPhase.SHIFT_1_OPTO: '#E24A33',
    OptoPhase.SHIFT_1_RECOVERY: '#F5A623',
    OptoPhase.SHIFT_2_OPTO: '#E24A33',
    OptoPhase.SHIFT_2_RECOVERY: '#B276B2',
    OptoPhase.PRE_EXPERIMENT: '#333333',
}


# ─── Psychometric overlay ────────────────────────────────────────────────────

def plot_opto_psychometric(
    session_or_sessions,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    n_bins: int = 8,
    n_bootstrap: int = 200,
    show_post_opto: bool = True,
) -> plt.Figure:
    """
    Plot overlaid psychometric curves for opto, post-opto, and control trials.

    Shows binned data points, fitted curves, and bootstrap 95% CI bands.
    Uses three-way split when show_post_opto=True.

    Args:
        session_or_sessions: Single SessionData or list of SessionData.
            If list, trials are pooled.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.
        n_bins: Number of stimulus bins for data points.
        n_bootstrap: Bootstrap iterations for CI bands (0 to skip).
        show_post_opto: Include post-opto curve (default True).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    else:
        fig = ax.get_figure()

    sessions = (session_or_sessions
                if isinstance(session_or_sessions, list)
                else [session_or_sessions])

    # Define conditions: (label, mask_getter, colour)
    # Control = all non-opto (70%), post-opto = subset of control
    conditions = [
        ('Control', lambda s: split_trials_by_opto(s)[1], CTRL_COLOUR),
        ('Opto', lambda s: split_trials_by_opto(s)[0], OPTO_COLOUR),
    ]
    if show_post_opto:
        conditions.append(
            ('Post-opto', lambda s: get_post_opto_mask(s), POST_OPTO_COLOUR))

    for label, mask_fn, colour in conditions:
        all_stim, all_choice = [], []
        for sess in sessions:
            mask = mask_fn(sess)
            arrays = extract_trial_arrays(sess, mask)
            if arrays is None:
                continue
            valid = ~arrays['no_response']
            all_stim.append(arrays['stimuli'][valid])
            all_choice.append(arrays['choices'][valid])

        if not all_stim:
            continue

        stim = np.concatenate(all_stim)
        choice = np.concatenate(all_choice)

        # Binned data points
        bins = np.linspace(-1, 1, n_bins + 1)
        centres = (bins[:-1] + bins[1:]) / 2
        means = np.full(n_bins, np.nan)
        for b in range(n_bins):
            in_bin = (stim >= bins[b]) & (stim < bins[b + 1])
            if b == n_bins - 1:
                in_bin |= (stim == bins[b + 1])
            if in_bin.sum() > 0:
                means[b] = np.nanmean(choice[in_bin])

        ax.plot(centres, means, 'o', color=colour, markersize=6,
                label=f'{label} (n={len(stim)})')

        # Fitted curve with bootstrap CI
        try:
            pfit = fit_psychometric(
                stim, choice, n_bootstrap=n_bootstrap)
            x_fine = np.linspace(-1, 1, 200)

            # CI band
            if n_bootstrap > 0 and pfit.get('y_fit_ci') is not None:
                ci_lo, ci_hi = pfit['y_fit_ci']
                if ci_lo is not None and ci_hi is not None:
                    x_fit = pfit.get('x_fit', np.linspace(-1, 1, 100))
                    ci_lo_fine = np.interp(x_fine, x_fit, ci_lo)
                    ci_hi_fine = np.interp(x_fine, x_fit, ci_hi)
                    ax.fill_between(
                        x_fine, ci_lo_fine, ci_hi_fine,
                        color=colour, alpha=0.15)

            # Fitted line
            y_fine = cumulative_gaussian(
                x_fine, pfit['mu'], pfit['sigma'],
                pfit['lapse_low'], pfit['lapse_high'])
            ax.plot(x_fine, y_fine, '-', color=colour, alpha=0.7)
        except Exception:
            pass

    ax.axhline(0.5, ls='--', color='grey', alpha=0.3)
    ax.axvline(0.0, ls='--', color='grey', alpha=0.3)
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('P(choose B)')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


# ─── Phase trajectory ────────────────────────────────────────────────────────

def plot_phase_trajectory(
    sessions: list,
    phases: List[OptoPhase],
    stat_name: str = 'accuracy',
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a statistic across all sessions, coloured by phase.

    Shows phase transitions as vertical lines.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    else:
        fig = ax.get_figure()

    values = []
    colours = []
    for sess, phase in zip(sessions, phases):
        try:
            st = sess.stats(stat_names=[stat_name], exclude_opto=True)
            values.append(st[stat_name])
        except Exception:
            values.append(np.nan)
        colours.append(PHASE_COLOURS.get(phase, '#333333'))

    x = np.arange(len(values))
    ax.plot(x, values, 'k-', alpha=0.3, linewidth=0.8)
    for i in range(len(values)):
        ax.scatter(x[i], values[i], c=colours[i], s=40, zorder=3,
                   edgecolor='white', linewidth=0.5)

    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            ax.axvline(i - 0.5, ls=':', color='grey', alpha=0.5)

    ax.set_xlabel('Session')
    ax.set_ylabel(stat_name)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


# ─── Update matrix comparison ────────────────────────────────────────────────

def plot_opto_um_comparison(
    opto_um: np.ndarray,
    control_um: np.ndarray,
    diff: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side update matrices: control, opto, and optionally difference.

    Uses UM_CMAP from behav_utils for all panels (consistent with
    all other UM plots in the codebase).
    """
    n_panels = 3 if diff else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4))

    vmax = max(np.nanmax(np.abs(control_um)), np.nanmax(np.abs(opto_um)))
    vmax = max(vmax, 0.01)

    im0 = axes[0].imshow(
        control_um, cmap=UM_CMAP, vmin=-vmax, vmax=vmax,
        origin='lower', aspect='equal')
    axes[0].set_title('Control')
    axes[0].set_xlabel('Previous stimulus')
    axes[0].set_ylabel('Current stimulus')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(
        opto_um, cmap=UM_CMAP, vmin=-vmax, vmax=vmax,
        origin='lower', aspect='equal')
    axes[1].set_title('Opto')
    axes[1].set_xlabel('Previous stimulus')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    if diff:
        diff_um = opto_um - control_um
        vmax_d = max(np.nanmax(np.abs(diff_um)), 0.01)
        im2 = axes[2].imshow(
            diff_um, cmap=UM_CMAP, vmin=-vmax_d, vmax=vmax_d,
            origin='lower', aspect='equal')
        axes[2].set_title('Opto − Control')
        axes[2].set_xlabel('Previous stimulus')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ─── Expert stability ────────────────────────────────────────────────────────

def plot_expert_stability(
    stability_result: Dict[str, Any],
    stat_name: str = 'accuracy',
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot baseline → opto → washout as grouped dots with means.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    else:
        fig = ax.get_figure()

    data = [
        ('Baseline', stability_result['baseline_values'],
         PHASE_COLOURS[OptoPhase.EXPERT_BASELINE]),
        ('Opto', stability_result['opto_values'],
         PHASE_COLOURS[OptoPhase.EXPERT_OPTO]),
        ('Washout', stability_result['washout_values'],
         PHASE_COLOURS[OptoPhase.EXPERT_WASHOUT]),
    ]

    for i, (label, values, colour) in enumerate(data):
        if len(values) == 0:
            continue
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(values))
        ax.scatter(i + jitter, values, c=colour, s=30, alpha=0.6,
                   edgecolor='white', linewidth=0.5)
        ax.plot([i - 0.2, i + 0.2], [np.mean(values)] * 2,
                color=colour, linewidth=2.5)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Baseline', 'Opto', 'Washout'])
    ax.set_ylabel(stat_name)

    p = stability_result.get('p_value', np.nan)
    if not np.isnan(p):
        ax.set_xlabel(f'Baseline vs Opto: p={p:.3f}')

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_phase_stability(
    stability_result: Dict[str, Any],
    stat_name: str = 'accuracy',
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot stat values across all phases with pairwise comparisons.

    Shows baseline, masking, opto, washout as grouped dots with means.
    Annotates significant pairwise comparisons.

    Args:
        stability_result: From phase_stability().
        stat_name: Which stat to plot (must be in stability_result['stat_names']).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    else:
        fig = ax.get_figure()

    phase_order = [
        (OptoPhase.EXPERT_BASELINE, 'Baseline'),
        (OptoPhase.MASKING, 'Masking'),
        (OptoPhase.EXPERT_OPTO, 'Opto'),
        (OptoPhase.EXPERT_WASHOUT, 'Washout'),
    ]

    # Only show phases with data
    present = []
    for phase, label in phase_order:
        vals = stability_result['per_phase'].get(phase, {}).get(stat_name, np.array([]))
        if len(vals) > 0:
            present.append((phase, label, vals))

    for i, (phase, label, values) in enumerate(present):
        colour = PHASE_COLOURS.get(phase, '#333333')
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(values))
        ax.scatter(i + jitter, values, c=colour, s=30, alpha=0.6,
                   edgecolor='white', linewidth=0.5)
        ax.plot([i - 0.2, i + 0.2], [np.mean(values)] * 2,
                color=colour, linewidth=2.5)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([label for _, label, _ in present])
    ax.set_ylabel(stat_name)

    # Annotate significant comparisons
    comparisons = stability_result.get('comparisons', {})
    sig_pairs = []
    for (pa, pb), stats in comparisons.items():
        p_val = stats.get(stat_name, {}).get('p', np.nan)
        if not np.isnan(p_val) and p_val < 0.05:
            # Find x positions
            idx_a = next((i for i, (ph, _, _) in enumerate(present) if ph == pa), None)
            idx_b = next((i for i, (ph, _, _) in enumerate(present) if ph == pb), None)
            if idx_a is not None and idx_b is not None:
                sig_pairs.append((idx_a, idx_b, p_val))

    # Draw significance brackets
    if sig_pairs:
        y_max = max(np.max(v) for _, _, v in present)
        y_step = (y_max - min(np.min(v) for _, _, v in present)) * 0.08
        for k, (ia, ib, p_val) in enumerate(sig_pairs):
            y = y_max + y_step * (k + 1)
            ax.plot([ia, ia, ib, ib], [y - y_step * 0.3, y, y, y - y_step * 0.3],
                    'k-', linewidth=0.8)
            ax.text((ia + ib) / 2, y, f'p={p_val:.3f}',
                    ha='center', va='bottom', fontsize=7)

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# ─── Within-session effect across sessions ───────────────────────────────────

def plot_within_session_summary(
    within_results: List[Dict],
    metric: str = 'accuracy',
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot per-session opto effect sizes across sessions.

    Args:
        within_results: List of dicts from animal_opto_report['within_session']
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
    else:
        fig = ax.get_figure()

    x_vals, y_vals, colours = [], [], []
    for entry in within_results:
        if entry['effect'] is None:
            continue
        x_vals.append(entry['session_idx'])
        y_vals.append(entry['effect']['diff'][metric])
        colours.append(PHASE_COLOURS.get(entry['phase'], '#333333'))

    ax.bar(x_vals, y_vals, color=colours, alpha=0.7, edgecolor='white')
    ax.axhline(0, ls='-', color='grey', alpha=0.3)
    ax.set_xlabel('Session')
    ax.set_ylabel(f'Opto − Control ({metric})')

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# ─── Genotype interaction ────────────────────────────────────────────────────

def plot_genotype_interaction(
    interaction_result: Dict[str, Any],
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Bar plot comparing opto effect sizes: Het vs WT.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    else:
        fig = ax.get_figure()

    het = interaction_result['het_diffs']
    wt = interaction_result['wt_diffs']
    metric = interaction_result['metric']

    positions = [0, 1]
    means = [interaction_result['het_mean'], interaction_result['wt_mean']]
    colours_bar = [OPTO_COLOUR, '#888888']

    ax.bar(positions, means, color=colours_bar, alpha=0.7,
           edgecolor='white', width=0.5)

    for vals, pos in [(het, 0), (wt, 1)]:
        if len(vals) > 0:
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax.scatter(pos + jitter, vals, c='black', s=20, alpha=0.4, zorder=3)

    ax.axhline(0, ls='-', color='grey', alpha=0.3)
    ax.set_xticks(positions)
    ax.set_xticklabels(['Het', 'WT'])
    ax.set_ylabel(f'Opto effect ({metric})')

    p = interaction_result.get('p_value', np.nan)
    if not np.isnan(p):
        ax.set_xlabel(f'p={p:.3f}')

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# ─── Model-assignment grouped effects ────────────────────────────────────────

def plot_model_assignment_effects(
    groups: Dict[str, Dict],
    metric: str = 'accuracy',
    comparison: Optional[Dict] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Opto effect sizes grouped by model assignment (BE / SC / unclear).

    Bars show group mean; individual dots show per-session effects.
    Annotates with n_animals and n_sessions per group.

    Args:
        groups: From opto_by_model_assignment()['groups'].
        metric: Metric name for axis label.
        comparison: From opto_by_model_assignment()['comparison'].
            If provided, annotates BE vs SC p-value.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    else:
        fig = ax.get_figure()

    order = ['BE', 'SC', 'unclear']
    present = [k for k in order if groups.get(k, {}).get('n_sessions', 0) > 0]

    for i, key in enumerate(present):
        grp = groups[key]
        colour = MODEL_COLOURS.get(key, '#999999')
        diffs = grp['diffs']
        mean_val = grp['mean_diff']

        # Bar
        ax.bar(i, mean_val, color=colour, alpha=0.7,
               edgecolor='white', width=0.5)

        # Individual session dots
        if len(diffs) > 0:
            jitter = np.random.default_rng(42).uniform(
                -0.12, 0.12, len(diffs))
            ax.scatter(i + jitter, diffs, c='black', s=18,
                       alpha=0.35, zorder=3)

        # Annotation: n animals, n sessions
        label = f'n={grp["n_animals"]}a, {grp["n_sessions"]}s'
        y_pos = min(0, mean_val) - 0.015
        ax.text(i, y_pos, label, ha='center', va='top', fontsize=8,
                color='#555555')

    ax.axhline(0, ls='-', color='grey', alpha=0.3)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present)
    ax.set_ylabel(f'Opto − Control ({metric})')

    # Annotate BE vs SC comparison
    if comparison is not None and len(present) >= 2:
        p_val = comparison.get('be_vs_sc_p', np.nan)
        if not np.isnan(p_val) and 'BE' in present and 'SC' in present:
            ax.set_xlabel(f'BE vs SC: p={p_val:.3f}')

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# ─── Equivalence test visualisation ──────────────────────────────────────────

def plot_equivalence_test(
    test_result: Dict[str, Any],
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Visualise TOST equivalence test result for expert null prediction.

    Shows:
    - Per-animal effect dots
    - Grand mean with 90% CI
    - Equivalence bounds (shaded region)
    - TOST and t-test conclusion annotations

    Args:
        test_result: From expert_null_test().
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    else:
        fig = ax.get_figure()

    effects = test_result['effects']
    bound = test_result['equivalence_bound']
    metric = test_result['metric']
    n = test_result['n_animals']

    if n == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='grey')
        fig.tight_layout()
        return fig

    # Equivalence bounds
    ax.axvspan(-bound, bound, alpha=0.10, color='green',
               label=f'Equivalence region (±{bound})')
    ax.axvline(-bound, ls='--', color='green', alpha=0.5, lw=0.8)
    ax.axvline(bound, ls='--', color='green', alpha=0.5, lw=0.8)
    ax.axvline(0, ls='-', color='grey', alpha=0.3)

    # Per-animal dots
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, n)
    ax.scatter(effects, jitter, c=OPTO_COLOUR, s=40, alpha=0.6,
               edgecolor='white', linewidth=0.5, zorder=3)

    # Grand mean + 90% CI
    mean = test_result['grand_mean']
    ci = test_result.get('ci_90', (np.nan, np.nan))

    if not np.isnan(ci[0]):
        ax.errorbar(mean, 0.0, xerr=[[mean - ci[0]], [ci[1] - mean]],
                     fmt='D', color='black', markersize=8,
                     capsize=5, capthick=1.5, elinewidth=1.5,
                     zorder=5, label=f'Mean ± 90% CI')
    else:
        ax.plot(mean, 0.0, 'D', color='black', markersize=8, zorder=5)

    # Annotation
    tost_p = test_result.get('tost_p', np.nan)
    ttest_p = test_result.get('ttest_p', np.nan)
    mw_p = test_result.get('mann_whitney_p', np.nan)
    reject = test_result.get('tost_reject', False)

    lines = [f'n = {n} animals']
    lines.append(f'Mean effect = {mean:.4f}')
    if not np.isnan(ttest_p):
        lines.append(f't-test p = {ttest_p:.3f}')
    if not np.isnan(mw_p):
        lines.append(f'Mann-Whitney p = {mw_p:.3f}')
    if not np.isnan(tost_p):
        verdict = 'equivalent' if reject else 'inconclusive'
        lines.append(f'TOST p = {tost_p:.3f} ({verdict})')

    text = '\n'.join(lines)
    ax.text(0.98, 0.98, text, transform=ax.transAxes,
            ha='right', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.9))

    ax.set_xlabel(f'Opto effect ({metric})')
    ax.set_yticks([])
    ax.legend(loc='lower left', fontsize=8)

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# ─── Phase × opto interaction ────────────────────────────────────────────────

def plot_phase_interaction(
    interaction_result: Dict[str, Any],
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Paired comparison of opto effects: expert vs post-shift.

    Shows per-animal paired dots connected by lines, with group
    means highlighted. This is the key figure for the thesis:
    if the lines slope downward (larger negative effect post-shift),
    the hypothesis is supported.

    Args:
        interaction_result: From phase_opto_interaction().
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    else:
        fig = ax.get_figure()

    expert = interaction_result['expert_effects']
    shift = interaction_result['shift_effects']
    paired_ids = interaction_result['paired_animals']
    n = interaction_result['n_paired']
    metric = interaction_result['metric']

    if n == 0:
        ax.text(0.5, 0.5, 'No paired data yet', transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='grey')
        warning = interaction_result.get('warning', '')
        if warning:
            ax.text(0.5, 0.35, warning, transform=ax.transAxes,
                    ha='center', va='center', fontsize=8, color='#888888',
                    wrap=True)
        fig.tight_layout()
        return fig

    # Paired lines + dots for each animal
    expert_phase = interaction_result['expert_phase']
    shift_phase = interaction_result['shift_phase']

    expert_colour = PHASE_COLOURS.get(expert_phase, '#348ABD')
    shift_colour = PHASE_COLOURS.get(shift_phase, '#E24A33')

    for i in range(n):
        ax.plot([0, 1], [expert[i], shift[i]], '-', color='#AAAAAA',
                linewidth=0.8, alpha=0.6, zorder=1)

    jitter_expert = np.random.default_rng(42).uniform(-0.06, 0.06, n)
    jitter_shift = np.random.default_rng(43).uniform(-0.06, 0.06, n)

    ax.scatter(0 + jitter_expert, expert, c=expert_colour, s=45,
               alpha=0.7, edgecolor='white', linewidth=0.5, zorder=3)
    ax.scatter(1 + jitter_shift, shift, c=shift_colour, s=45,
               alpha=0.7, edgecolor='white', linewidth=0.5, zorder=3)

    # Group means
    exp_mean = interaction_result['expert_mean']
    shf_mean = interaction_result['shift_mean']
    ax.plot([0, 1], [exp_mean, shf_mean], 'k-', linewidth=2.5, zorder=4)
    ax.plot(0, exp_mean, 'ko', markersize=10, zorder=5)
    ax.plot(1, shf_mean, 'ko', markersize=10, zorder=5)

    ax.axhline(0, ls='-', color='grey', alpha=0.3)

    # Labels
    exp_label = expert_phase.value.replace('_', ' ').title()
    shf_label = shift_phase.value.replace('_', ' ').title()
    ax.set_xticks([0, 1])
    ax.set_xticklabels([exp_label, shf_label])
    ax.set_ylabel(f'Opto effect ({metric})')

    # Stats annotation
    lines = [f'n = {n} paired animals']
    lines.append(
        f'Interaction: {interaction_result["interaction_mean"]:.4f}'
    )
    paired_p = interaction_result.get('paired_ttest_p', np.nan)
    wilcox_p = interaction_result.get('wilcoxon_p', np.nan)
    if not np.isnan(paired_p):
        lines.append(f'Paired t-test p = {paired_p:.3f}')
    if not np.isnan(wilcox_p):
        lines.append(f'Wilcoxon p = {wilcox_p:.3f}')

    text = '\n'.join(lines)
    ax.text(0.98, 0.02, text, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.9))

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# ─── Full animal report ──────────────────────────────────────────────────────

def plot_animal_opto_report(
    animal,
    report: Dict[str, Any],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Full-page opto report for one animal.

    Top row: phase trajectory (accuracy coloured by phase)
    Middle row: expert stability + within-session effect bars
    Bottom row: UM difference maps for each opto phase
    """
    phases = report['phases']
    sessions = animal.sessions

    n_um_phases = len(report['phase_comparisons'])
    n_rows = 3 if n_um_phases > 0 else 2

    fig = plt.figure(figsize=(14, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Row 1: phase trajectory
    ax_traj = fig.add_subplot(gs[0, :])
    plot_phase_trajectory(
        sessions, phases, stat_name='accuracy',
        ax=ax_traj, title=f'{animal.animal_id} — Accuracy across phases')

    # Row 2: expert stability + within-session effects
    ax_stab = fig.add_subplot(gs[1, 0])
    plot_expert_stability(
        report['expert_stability'], stat_name='accuracy',
        ax=ax_stab, title='Expert stability')

    ax_within = fig.add_subplot(gs[1, 1:])
    plot_within_session_summary(
        report['within_session'], metric='accuracy',
        ax=ax_within, title='Per-session opto effect')

    # Row 3: UM difference maps
    if n_um_phases > 0:
        for idx, (phase, comp) in enumerate(report['phase_comparisons'].items()):
            if idx >= 3:
                break
            ax_um = fig.add_subplot(gs[2, idx])
            diff_um = comp['opto_um'] - comp['control_um']
            vmax = max(np.nanmax(np.abs(diff_um)), 0.01)
            im = ax_um.imshow(
                diff_um, cmap=UM_CMAP, vmin=-vmax, vmax=vmax,
                origin='lower', aspect='equal')
            ax_um.set_title(f'{phase.value}\nOpto−Control UM')
            plt.colorbar(im, ax=ax_um, fraction=0.046)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
