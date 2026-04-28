"""
Validation Report — Per-animal synthetic validation plots.

For synthetic animals where ground truth is known. Takes pre-loaded
data structures, not paths.

Usage:
    from plotting.validation_report import (
        plot_synth_summary, plot_synth_cv_results,
        plot_synth_model_fits, plot_synth_sbi_diagnostics,
        plot_recovery_overlay, plot_recovery_summary,
    )
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from behav_utils.analysis.summary_stats import (
    compute_summary_stats, get_stat_names_expanded,
)
from behav_utils.analysis.update_matrix import compute_update_matrix, matrix_error
from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.analysis.utils import cumulative_gaussian
from behav_utils.plotting.psychometric import plot_psychometric
from behav_utils.plotting.update_matrix import (
    plot_update_matrix, plot_phase_update_matrices,
)
from behav_utils.plotting.styles import COLOURS

BE_COL = COLOURS['BE']
SC_COL = COLOURS['SC']

FT_LABEL = {'update_matrix': 'UM', 'conditional_psych': 'CP'}


# =============================================================================
# HELPERS
# =============================================================================

def _params_to_str(params) -> str:
    if params is None: return ''
    if hasattr(params, '__dict__'):
        d = {k: v for k, v in vars(params).items()
             if not str(k).startswith('_') and isinstance(v, (int, float))}
    elif isinstance(params, dict):
        d = {k: v for k, v in params.items() if isinstance(v, (int, float))}
    else: return str(params)
    return ', '.join(f'{k}={v:.3f}' for k, v in d.items())


def _params_to_dict(params) -> dict:
    if params is None: return {}
    if hasattr(params, '__dict__'):
        return {k: v for k, v in vars(params).items()
                if not str(k).startswith('_') and isinstance(v, (int, float))}
    elif isinstance(params, dict):
        return {k: v for k, v in params.items() if isinstance(v, (int, float))}
    return {}


# Re-export for callers that already import from here
from plotting.cv import gs_seed_errors as _gs_seed_errors


def _pool_sessions(sessions):
    stim = np.concatenate([s.trials.stimulus for s in sessions])
    ch = np.concatenate([s.trials.choice for s in sessions])
    cat = np.concatenate([s.trials.category for s in sessions])
    return stim, ch, cat


def _get_observed_stats(sessions, stat_names):
    all_stats = []
    for sess in sessions:
        s = compute_summary_stats(
            sess.trials.choice, sess.trials.stimulus, sess.trials.category,
            stat_names=stat_names, return_dict=False)
        all_stats.append(s)
    return np.nanmean(all_stats, axis=0)


def build_gs_index(synth_gs_raw: dict) -> dict:
    """
    Build nested index: {fit_target: {animal_id: {'BE': pickle, 'SC': pickle}}}.

    Args:
        synth_gs_raw: {(cohort, fit_target): [list of raw pickles]}

    Returns:
        {fit_target: {animal_id: {'BE': data, 'SC': data}}}
    """
    index = {}
    for (cohort, ft), pickles in synth_gs_raw.items():
        if ft not in index:
            index[ft] = {}
        for r in pickles:
            aid = r['animal_id']
            if aid not in index[ft]:
                index[ft][aid] = {}
            index[ft][aid][r['model']] = r
    return index


# =============================================================================
# PER-ANIMAL PLOTS
# =============================================================================

def plot_synth_summary(sa: dict, n_bootstrap: int = 1000):
    """Raw behaviour: psychometric + UM. Shows true model + params."""
    aid = sa['animal_id']
    mt = sa['true_model']
    sessions = sa['sessions']
    stim, ch, cat = _pool_sessions(sessions)
    col = BE_COL if mt == 'BE' else SC_COL

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_psychometric(stim, ch, ax=axes[0], n_bootstrap=n_bootstrap,
                      show_ci=True, show_params=True, color=col)
    um, _, _ = compute_update_matrix(stim, ch, cat, n_bins=8)
    plot_update_matrix(um, ax=axes[1], title='Update Matrix')
    fig.suptitle(
        f'{aid} — True {mt} ({len(sessions)} sessions, {len(stim)} trials)\n'
        f'{_params_to_str(sa["true_params"])}',
        fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()


def plot_synth_cv_results(
    sa: dict, gs_index: dict, fit_target: str = 'update_matrix',
):
    """
    GS CV violin + scatter for a synthetic animal.

    Uses the shared plot_cv_comparison from plotting.cv for consistent
    visual style across synthetic and real data.

    Args:
        sa: Synthetic animal dict
        gs_index: {fit_target: {animal_id: {'BE': pickle, 'SC': pickle}}}
        fit_target: Which fit target to show
    """
    from plotting.cv import build_cv_dataframes, plot_cv_comparison

    aid = sa['animal_id']
    mt = sa['true_model']
    ft_short = FT_LABEL[fit_target]

    if fit_target not in gs_index or aid not in gs_index[fit_target]:
        print(f'  No GS-{ft_short} data for {aid}')
        return None

    fits = gs_index[fit_target][aid]
    if 'BE' not in fits or 'SC' not in fits:
        print(f'  Incomplete GS-{ft_short} data for {aid}')
        return None

    be_e, _ = _gs_seed_errors(fits['BE'])
    sc_e, _ = _gs_seed_errors(fits['SC'])
    if not be_e or not sc_e:
        print(f'  No valid seeds for {aid}')
        return None

    long_df, comparison_df = build_cv_dataframes(aid, be_e, sc_e)
    if long_df is None:
        return None

    # Build synthetic-specific suptitle showing ground truth
    row = comparison_df.iloc[0]
    winner = row['winner']
    p_val = row['p_value']
    n_paired = min(len(be_e), len(sc_e))

    if p_val < 0.05:
        correct = winner == mt
        tick = '✓' if correct else '✗'
        sig_str = f'Assigned: {winner} {tick}'
    elif p_val < 0.1:
        sig_str = f'Marginal {winner} (true: {mt})'
    else:
        sig_str = f'Inconclusive, {winner} lower (true: {mt})'

    title = f'{aid} — GS-{ft_short} CV ({n_paired} seeds) — {sig_str}'

    fig = plot_cv_comparison(
        long_df, comparison_df, aid,
        fit_target=ft_short, suptitle=title,
    )
    plt.show()
    return fig


def plot_synth_model_fits(
    sa: dict, gs_index: dict,
    fit_target: str = 'update_matrix',
    n_reps: int = 20, burn_in: int = 1000, seed: int = 42,
):
    """
    UM comparison + psychometric overlay + per-session MSE.
    Uses params from specified fit_target's GS results.
    """
    aid = sa['animal_id']
    mt = sa['true_model']
    ft_short = FT_LABEL[fit_target]
    sessions = sa['sessions']

    if fit_target not in gs_index or aid not in gs_index[fit_target]:
        print(f'  No GS-{ft_short} data for {aid}'); return None

    fits = gs_index[fit_target][aid]
    if 'BE' not in fits or 'SC' not in fits:
        print(f'  Incomplete GS-{ft_short} data for {aid}'); return None

    _, be_params = _gs_seed_errors(fits['BE'])
    _, sc_params = _gs_seed_errors(fits['SC'])
    if not be_params or not sc_params:
        print(f'  No valid params for {aid}'); return None

    print(f'  Params from GS-{ft_short}:')
    print(f'    BE: {_params_to_str(be_params)}')
    print(f'    SC: {_params_to_str(sc_params)}')
    print(f'    True ({mt}): {_params_to_str(sa["true_params"])}')

    from behav_utils.data.selection import fitting_data_from_sessions
    from inference.comparison import simulate_all_sessions

    fd = fitting_data_from_sessions(sessions, aid)
    print(f'  Simulating ({fd.n_sessions} sessions, {n_reps} reps)...')
    session_data = simulate_all_sessions(
        fd, be_params, sc_params,
        burn_in=burn_in, n_reps=n_reps, n_bins=8, seed=seed)
    print(f'  {len(session_data)} sessions done')

    # UM comparison
    emp_um = np.nanmean([d['real_um'] for d in session_data], axis=0)
    be_um = np.nanmean([d['be_um'] for d in session_data], axis=0)
    sc_um = np.nanmean([d['sc_um'] for d in session_data], axis=0)
    be_mse = matrix_error(be_um, emp_um)
    sc_mse = matrix_error(sc_um, emp_um)

    fig, axes = plot_phase_update_matrices(
        {'Real': emp_um, 'BE': be_um, 'SC': sc_um},
        annotations={
            'Real': f'n={len(session_data)} sessions',
            'BE': f'MSE={be_mse:.5f}' + (' ★' if mt == 'BE' else ''),
            'SC': f'MSE={sc_mse:.5f}' + (' ★' if mt == 'SC' else ''),
        },
        suptitle=f'{aid} — UM Comparison (params: GS-{ft_short}, true: {mt})',
    )
    plt.show()

    # Psychometric overlay
    all_stim = np.concatenate([d['stimuli'] for d in session_data])
    all_ch = np.concatenate([d['choices'] for d in session_data])
    x_fine = np.linspace(-1.1, 1.1, 200)

    fig, ax = plt.subplots(figsize=(7, 5))
    _, ax, _ = plot_psychometric(
        all_stim, all_ch, ax=ax, n_bootstrap=500, show_ci=True,
        show_params=False, color='black', label='Data')
    for mk, col in [('be', BE_COL), ('sc', SC_COL)]:
        ps = [d[f'{mk}_psych'] for d in session_data
              if d[f'{mk}_psych'].get('success')]
        if ps:
            mu = np.mean([p['mu'] for p in ps])
            sig = np.mean([p['sigma'] for p in ps])
            ll = np.mean([p['lapse_low'] for p in ps])
            lh = np.mean([p['lapse_high'] for p in ps])
            star = ' ★' if mk.upper() == mt else ''
            ax.plot(x_fine, cumulative_gaussian(x_fine, mu, sig, ll, lh),
                    '-', color=col, lw=2,
                    label=f'{mk.upper()} (PSE={mu:.3f}){star}')
    ax.legend(fontsize=9)
    ax.set_title(f'{aid} — Psychometric (params: GS-{ft_short}, true: {mt})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.show()

    # Per-session MSE
    si = [d['session_idx'] for d in session_data]
    be_mses = [d['be_um_mse'] for d in session_data]
    sc_mses = [d['sc_um_mse'] for d in session_data]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(si, be_mses, 'o-', color=BE_COL, label='BE', ms=5)
    ax.plot(si, sc_mses, 'o-', color=SC_COL, label='SC', ms=5)
    bmu, smu = np.nanmean(be_mses), np.nanmean(sc_mses)
    ax.axhline(bmu, color=BE_COL, ls='--', alpha=0.4)
    ax.axhline(smu, color=SC_COL, ls='--', alpha=0.4)
    bw = sum(b < s for b, s in zip(be_mses, sc_mses))
    w = 'BE' if bmu < smu else 'SC'
    ax.set(xlabel='Session index', ylabel='UM MSE')
    ax.set_title(
        f'{aid} — Per-Session (GS-{ft_short}, true: {mt})\n'
        f'BE={bmu:.5f}, SC={smu:.5f} → {w} (BE wins {bw}/{len(be_mses)})',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()

    return {'session_data': session_data, 'be_params': be_params, 'sc_params': sc_params}


def plot_synth_sbi_diagnostics(
    sa: dict, snpe_dict: dict,
    gs_index: Optional[dict] = None,
    n_samples: int = 10000,
    n_ppc_samples: int = 200,
    show_ppc: bool = True,
):
    """
    SBI posteriors with true values + GS point estimates from both fit targets.
    """
    import torch

    aid = sa['animal_id']
    mt = sa['true_model']
    sessions = sa['sessions']
    true_dict = _params_to_dict(sa['true_params'])

    if not snpe_dict or 'be' not in snpe_dict or 'sc' not in snpe_dict:
        print(f'  SNPE networks not available'); return None

    stat_names_raw = snpe_dict['be']['stat_names']
    stat_names_expanded = get_stat_names_expanded(stat_names_raw)

    try:
        x_obs = _get_observed_stats(sessions, stat_names_raw)
        if np.any(np.isnan(x_obs)):
            print(f'  NaN in observed stats'); return None
    except Exception as e:
        print(f'  Stats failed: {e}'); return None

    x_t = torch.tensor(x_obs, dtype=torch.float32)
    all_samples = {}
    for mk in ['be', 'sc']:
        try:
            all_samples[mk] = snpe_dict[mk]['posterior'].sample(
                (n_samples,), x=x_t).numpy()
        except Exception as e:
            warnings.warn(f'Sampling failed for {mk}: {e}')
            all_samples[mk] = None

    if all_samples.get('be') is None or all_samples.get('sc') is None:
        print(f'  Posterior sampling failed'); return None

    # ── Posterior marginals with true values + GS points ─────────────────
    # Collect GS params from all fit targets
    gs_params_by_ft = {}
    if gs_index:
        for ft in ['update_matrix', 'conditional_psych']:
            if ft in gs_index and aid in gs_index[ft]:
                fits = gs_index[ft][aid]
                mk_upper = mt  # only show true model's params
                if mk_upper in fits:
                    _, params = _gs_seed_errors(fits[mk_upper])
                    if params:
                        gs_params_by_ft[ft] = params

    # Plot: true model's posteriors with all reference lines
    mk = mt.lower()
    pnames = snpe_dict[mk]['param_names']
    samples = all_samples[mk]
    col = BE_COL if mt == 'BE' else SC_COL

    fig, axes = plt.subplots(1, len(pnames), figsize=(5 * len(pnames), 5))
    axes = np.atleast_1d(axes)

    for j, pn in enumerate(pnames):
        ax = axes[j]
        ax.hist(samples[:, j], bins=50, color=col, alpha=0.4,
                density=True, edgecolor='none', label='SBI posterior')
        med = np.median(samples[:, j])
        ax.axvline(med, color=col, lw=1.5, ls=':',
                   label=f'SBI median={med:.3f}')

        # True value
        if pn in true_dict:
            ax.axvline(true_dict[pn], color='red', lw=2.5,
                       label=f'True={true_dict[pn]:.3f}')

        # GS points from each fit target
        gs_styles = {
            'update_matrix': ('grey', '--', 'GS-UM'),
            'conditional_psych': ('dimgrey', '-.', 'GS-CP'),
        }
        for ft, params in gs_params_by_ft.items():
            if pn in params:
                style_col, style_ls, style_label = gs_styles[ft]
                ax.axvline(params[pn], color=style_col, lw=2, ls=style_ls,
                           label=f'{style_label}={params[pn]:.3f}')

        ax.set_xlabel(pn, fontsize=11)
        ax.legend(fontsize=7)

    fig.suptitle(
        f'{aid} — {mt} Posteriors: SBI vs GS-UM vs GS-CP vs Truth',
        fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()
    
    # ── Pair plot ─────────────────────────────────────────────────────────
    from plotting.sbi import plot_pairplot
    
    mk = mt.lower()
    pnames = snpe_dict[mk]['param_names']
    samples = all_samples[mk]
    
    ground_truth = np.array([true_dict.get(pn, np.nan) for pn in pnames])
    
    fig = plot_pairplot(samples, pnames, ground_truth=ground_truth)
    fig.suptitle(f'{aid} — {mt} Posterior Correlations (red = true)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.show()

    # ── Also show the other model's posteriors (without true params) ──────
    other_mk = 'sc' if mk == 'be' else 'be'
    other_mt = 'SC' if mt == 'BE' else 'BE'
    other_pnames = snpe_dict[other_mk]['param_names']
    other_samples = all_samples[other_mk]
    other_col = SC_COL if mt == 'BE' else BE_COL

    fig, axes = plt.subplots(1, len(other_pnames), figsize=(5 * len(other_pnames), 4))
    axes = np.atleast_1d(axes)
    for j, pn in enumerate(other_pnames):
        ax = axes[j]
        ax.hist(other_samples[:, j], bins=50, color=other_col, alpha=0.4,
                density=True, edgecolor='none')
        med = np.median(other_samples[:, j])
        lo, hi = np.percentile(other_samples[:, j], [5, 95])
        ax.axvline(med, color=other_col, lw=2)
        ax.axvspan(lo, hi, alpha=0.12, color=other_col)
        ax.set_xlabel(pn, fontsize=9)
        ax.set_title(f'{med:.3f} [{lo:.3f}, {hi:.3f}]', fontsize=8)
    fig.suptitle(f'{aid} — {other_mt} Posteriors (wrong model, for reference)',
                 fontsize=11)
    plt.tight_layout(); plt.show()

    # ── PPC ───────────────────────────────────────────────────────────────
    if show_ppc:
        from inference.simulator import (
            create_be_simulator, create_sc_simulator, wrap_for_sbi,
        )
        from plotting.sbi import plot_summary_stats_comparison

        ref_stim = np.concatenate([s.trials.stimulus for s in sessions[:3]])
        ref_cat = np.concatenate([s.trials.category for s in sessions[:3]])

        for mk_i, col_i, lab_i, create_fn in [
            ('be', BE_COL, 'BE', create_be_simulator),
            ('sc', SC_COL, 'SC', create_sc_simulator),
        ]:
            samp = all_samples[mk_i][:n_ppc_samples]
            try:
                sim = create_fn(stimuli=ref_stim, categories=ref_cat,
                                stat_names=stat_names_raw,
                                burn_in=snpe_dict[mk_i]['burn_in'], seed=42)
                sbi_sim = wrap_for_sbi(sim)
            except Exception as e:
                print(f'  {lab_i} PPC: simulator failed: {e}'); continue

            pred = []
            for si in range(len(samp)):
                import torch as _torch
                theta = _torch.tensor(samp[si], dtype=_torch.float32)
                try:
                    x_sim = sbi_sim(theta)
                    if x_sim is not None and not _torch.any(_torch.isnan(x_sim)):
                        pred.append(x_sim.numpy())
                except Exception: pass

            if pred:
                star = ' ★' if mk_i.upper() == mt else ''
                fig = plot_summary_stats_comparison(
                    x_obs, np.array(pred), stat_names_expanded,
                    title=f'{aid} — {lab_i}{star} PPC')
                plt.show()
            else:
                print(f'  {lab_i} PPC: all simulations failed')

    return {'be_samples': all_samples.get('be'),
            'sc_samples': all_samples.get('sc'), 'x_obs': x_obs}


# =============================================================================
# COHORT-LEVEL: PARAMETER RECOVERY
# =============================================================================

def extract_gs_recovery(synth_gs_raw_pickles: list) -> dict:
    """
    Extract true vs recovered params from raw GS pickles.
    Only for correctly identified animals.

    Returns: {model_type: {param_name: {'true': [], 'recovered': []}}}
    """
    by_animal = {}
    for r in synth_gs_raw_pickles:
        aid = r['animal_id']
        if aid not in by_animal: by_animal[aid] = {}
        by_animal[aid][r['model']] = r

    recovery = {'BE': {}, 'SC': {}}
    for aid, fits in by_animal.items():
        if 'BE' not in fits or 'SC' not in fits: continue
        be_err, sc_err = fits['BE']['mean_error'], fits['SC']['mean_error']
        if np.isnan(be_err) or np.isnan(sc_err): continue
        winner = 'BE' if be_err < sc_err else 'SC'
        if winner != fits['BE']['true_model']: continue
        tp = fits[winner]['true_params']
        td = _params_to_dict(tp)
        _, rec = _gs_seed_errors(fits[winner])
        if not rec: continue
        for pn, tv in td.items():
            if pn in rec:
                if pn not in recovery[winner]:
                    recovery[winner][pn] = {'true': [], 'recovered': []}
                recovery[winner][pn]['true'].append(tv)
                recovery[winner][pn]['recovered'].append(rec[pn])
    return recovery


def extract_sbi_recovery(
    cohort_animals: list, snpe_dict: dict, ci_level: int = 90,
) -> dict:
    """
    Re-condition SNPE on each synthetic animal, extract recovery + CIs.

    Returns: {model_type: {param_name: {'true': [], 'recovered': [],
                                         'ci_lo': [], 'ci_hi': []}}}
    """
    import torch
    recovery = {'BE': {}, 'SC': {}}
    lo_pct, hi_pct = (100 - ci_level) / 2, 100 - (100 - ci_level) / 2

    for sa in cohort_animals:
        mt = sa['true_model']
        mk = mt.lower()
        if mk not in snpe_dict: continue
        td = _params_to_dict(sa['true_params'])
        sessions = sa.get('sessions', [])
        if not sessions: continue
        stat_names = snpe_dict[mk]['stat_names']
        pnames = snpe_dict[mk]['param_names']
        try:
            x_obs = _get_observed_stats(sessions, stat_names)
            if np.any(np.isnan(x_obs)): continue
            x_t = torch.tensor(x_obs, dtype=torch.float32)
            samples = snpe_dict[mk]['posterior'].sample((5000,), x=x_t).numpy()
        except Exception: continue
        for j, pn in enumerate(pnames):
            if pn not in td: continue
            true_val = td[pn]
            median = np.median(samples[:, j])
            lo, hi = np.percentile(samples[:, j], [lo_pct, hi_pct])
            if pn not in recovery[mt]:
                recovery[mt][pn] = {'true': [], 'recovered': [],
                                    'ci_lo': [], 'ci_hi': []}
            recovery[mt][pn]['true'].append(true_val)
            recovery[mt][pn]['recovered'].append(median)
            recovery[mt][pn]['ci_lo'].append(lo)
            recovery[mt][pn]['ci_hi'].append(hi)
    return recovery


def plot_recovery_overlay(
    gs_um_recovery: dict, gs_cp_recovery: dict, sbi_recovery: dict,
):
    """
    Triple overlay: GS-UM (×), GS-CP (△), SBI (○ + CI) on same axes.
    One figure per model type (BE, SC).
    """
    for mt in ['BE', 'SC']:
        gs_um = gs_um_recovery.get(mt, {})
        gs_cp = gs_cp_recovery.get(mt, {})
        sbi = sbi_recovery.get(mt, {})

        # Find all params that appear in at least one method
        all_params = sorted(set(
            list(gs_um.keys()) + list(gs_cp.keys()) + list(sbi.keys())
        ))
        if not all_params: continue

        n = len(all_params)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        axes = np.atleast_1d(axes)
        col = BE_COL if mt == 'BE' else SC_COL

        for i, pn in enumerate(all_params):
            ax = axes[i]
            r_values = {}

            # GS-UM: × markers
            if pn in gs_um:
                t = np.array(gs_um[pn]['true'])
                r = np.array(gs_um[pn]['recovered'])
                ax.scatter(t, r, marker='x', color='grey', s=50,
                           linewidths=1.5, alpha=0.7, label='GS-UM', zorder=4)
                if len(t) > 2:
                    r_values['GS-UM'] = np.corrcoef(t, r)[0, 1]

            # GS-CP: △ markers
            if pn in gs_cp:
                t = np.array(gs_cp[pn]['true'])
                r = np.array(gs_cp[pn]['recovered'])
                ax.scatter(t, r, marker='^', color='dimgrey', s=40,
                           alpha=0.7, label='GS-CP', zorder=4)
                if len(t) > 2:
                    r_values['GS-CP'] = np.corrcoef(t, r)[0, 1]

            # SBI: ○ with CI error bars
            if pn in sbi:
                t = np.array(sbi[pn]['true'])
                r = np.array(sbi[pn]['recovered'])
                lo = np.array(sbi[pn]['ci_lo'])
                hi = np.array(sbi[pn]['ci_hi'])
                yerr = np.array([r - lo, hi - r])
                ax.errorbar(t, r, yerr=yerr, fmt='o', color=col, ms=5,
                            capsize=2, elinewidth=0.8, alpha=0.7,
                            label='SBI (±90% CI)', zorder=5)
                if len(t) > 2:
                    r_values['SBI'] = np.corrcoef(t, r)[0, 1]

            # Identity line
            all_vals = []
            for src in [gs_um, gs_cp, sbi]:
                if pn in src:
                    all_vals.extend(src[pn]['true'])
                    all_vals.extend(src[pn]['recovered'])
            if all_vals:
                lo_v, hi_v = min(all_vals), max(all_vals)
                margin = (hi_v - lo_v) * 0.1
                lims = [lo_v - margin, hi_v + margin]
                ax.plot(lims, lims, 'k--', alpha=0.3)
                ax.set_xlim(lims); ax.set_ylim(lims)
                ax.set_aspect('equal')

            # r values box
            if r_values:
                text = '\n'.join(f'{k}: r={v:.3f}' for k, v in r_values.items())
                ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
                        va='top', bbox=dict(boxstyle='round', facecolor='white',
                                            alpha=0.8))

            ax.set(xlabel=f'True {pn}', ylabel='Recovered')
            ax.set_title(pn, fontsize=11)
            ax.legend(fontsize=7, loc='lower right')

        # Count animals per method
        ns = {}
        for label, src in [('GS-UM', gs_um), ('GS-CP', gs_cp), ('SBI', sbi)]:
            if all_params[0] in src:
                ns[label] = len(src[all_params[0]]['true'])
        n_str = ', '.join(f'{k} n={v}' for k, v in ns.items())

        fig.suptitle(f'{mt} — Parameter Recovery: GS-UM vs GS-CP vs SBI ({n_str})',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(); plt.show()


def plot_recovery_summary(
    gs_um_recovery: dict, gs_cp_recovery: dict, sbi_recovery: dict,
):
    """
    Grouped bar chart: correlation per parameter per method.
    Quick visual answer to "which method recovers best?"
    """
    for mt in ['BE', 'SC']:
        gs_um = gs_um_recovery.get(mt, {})
        gs_cp = gs_cp_recovery.get(mt, {})
        sbi = sbi_recovery.get(mt, {})

        all_params = sorted(set(
            list(gs_um.keys()) + list(gs_cp.keys()) + list(sbi.keys())
        ))
        if not all_params: continue

        methods = [
            ('GS-UM', gs_um, 'grey'),
            ('GS-CP', gs_cp, 'dimgrey'),
            ('SBI', sbi, BE_COL if mt == 'BE' else SC_COL),
        ]
        n_params = len(all_params)
        n_methods = len(methods)
        w = 0.25
        x = np.arange(n_params)

        fig, ax = plt.subplots(figsize=(max(8, n_params * 2.5), 5))

        for j, (label, src, colour) in enumerate(methods):
            vals = []
            for pn in all_params:
                if pn in src:
                    t = np.array(src[pn]['true'])
                    r = np.array(src[pn]['recovered'])
                    vals.append(np.corrcoef(t, r)[0, 1] if len(t) > 2 else np.nan)
                else:
                    vals.append(np.nan)
            offset = (j - n_methods / 2 + 0.5) * w
            # Plot bars (use actual values including negative)
            plot_vals = [v if not np.isnan(v) else 0 for v in vals]
            bars = ax.bar(x + offset, plot_vals, w * 0.9, label=label,
                          color=colour, alpha=0.8)
            for bar, v in zip(bars, vals):
                if np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                            'n/a', ha='center', va='bottom', fontsize=6,
                            color='grey')
                else:
                    # Label above positive bars, below negative bars
                    y_pos = bar.get_height() + 0.02 if v >= 0 else bar.get_height() - 0.05
                    va = 'bottom' if v >= 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                            f'{v:.2f}', ha='center', va=va, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(all_params, fontsize=10)
        ax.set_ylabel('Correlation (r)', fontsize=11)
        ax.axhline(0, color='black', lw=0.8)
        ax.axhline(1.0, color='grey', ls='--', alpha=0.3)
        # Extend y-axis to show negatives
        all_r = [v for vals_list in [
            [np.corrcoef(np.array(src[pn]['true']), np.array(src[pn]['recovered']))[0, 1]
             for pn in all_params if pn in src and len(src[pn]['true']) > 2]
            for _, src, _ in methods] for v in vals_list]
        y_min = min(min(all_r, default=0) - 0.1, -0.1)
        ax.set_ylim(y_min, 1.1)
        ax.legend(fontsize=9)
        ax.set_title(f'{mt} — Recovery Correlation by Method',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(); plt.show()

        # Print table
        print(f'\n  {mt} Recovery Summary:')
        print(f'  {"Param":20s} {"GS-UM":>8s} {"GS-CP":>8s} {"SBI":>8s}')
        for pn in all_params:
            row = f'  {pn:20s}'
            for label, src, _ in methods:
                if pn in src:
                    t = np.array(src[pn]['true'])
                    r = np.array(src[pn]['recovered'])
                    corr = np.corrcoef(t, r)[0, 1] if len(t) > 2 else np.nan
                    rmse = np.sqrt(np.mean((t - r) ** 2))
                    row += f' {corr:7.3f}'
                else:
                    row += f'     —'
            print(row)
