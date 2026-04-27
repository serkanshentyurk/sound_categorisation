"""
Animal Report — Modular per-animal analysis plots.

Four public functions:

    plot_animal_summary   — Raw behaviour + method consensus
    plot_cv_results       — GS CV seed-level comparison
    plot_model_fits       — UM comparison, psychometric overlay, per-session MSE
    plot_sbi_diagnostics  — Posterior marginals, PPC, GS vs SBI overlay

Wraps existing plotting functions where possible:
    behav_utils.plotting.psychometric.plot_psychometric  (bootstrapped psychometric)
    behav_utils.plotting.update_matrix.plot_phase_update_matrices  (UM comparison)
    plotting.cv.plot_cv_comparison  (CV violin + scatter)
    plotting.sbi.plot_summary_stats_comparison  (PPC)

Usage:
    from plotting.animal_report import (
        plot_animal_summary, plot_cv_results,
        plot_model_fits, plot_sbi_diagnostics,
    )

    sessions = select_sessions(animal, 'expert_uniform')
    plot_animal_summary('SS05', sessions, results_dir=RESULTS_DIR)
    plot_cv_results('SS05', results_dir=RESULTS_DIR)
    plot_model_fits('SS05', sessions, results_dir=RESULTS_DIR, method='sbi')
    plot_sbi_diagnostics('SS05', sessions, snpe_dict=snpe, results_dir=RESULTS_DIR)
"""

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from behav_utils.analysis.summary_stats import compute_summary_stats, get_stat_names_expanded
from behav_utils.analysis.update_matrix import (
    compute_update_matrix, matrix_error,
)
from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.analysis.utils import cumulative_gaussian
from behav_utils.plotting.psychometric import plot_psychometric
from behav_utils.plotting.update_matrix import (
    plot_phase_update_matrices, plot_update_matrix,
)
from behav_utils.plotting.styles import COLOURS

BE_COL = COLOURS['BE']
SC_COL = COLOURS['SC']

FIT_TARGETS = ('update_matrix', 'conditional_psych')
FT_LABEL = {'update_matrix': 'UM', 'conditional_psych': 'CP'}


# =============================================================================
# PRIVATE HELPERS — DATA LOADING
# =============================================================================

def _load_gs_raw_pickles(
    animal_id: str, results_dir: Path,
    distribution: str = 'uniform', fit_target: str = 'update_matrix',
) -> Tuple[Optional[dict], Optional[dict]]:
    """Load raw GS per-model pickles. Returns (be_data, sc_data)."""
    cv_dir = results_dir / 'cv' / f'{distribution}_{fit_target}'
    data = {}
    for model in ['BE', 'SC']:
        path = cv_dir / f'cv_{animal_id}_{model}.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                data[model] = pickle.load(f)
    return data.get('BE'), data.get('SC')


def _gs_seed_errors(gs_data: dict) -> Tuple[list, Optional[dict]]:
    """Extract per-seed errors and best params from a raw GS pickle."""
    results = gs_data.get('results', [])
    errors = [r['avg_test_error'] for r in results
              if not np.isnan(r.get('avg_test_error', np.nan))]
    valid = [r for r in results
             if not np.isnan(r.get('avg_test_error', np.nan))
             and r.get('best_params_single')]
    best_params = (min(valid, key=lambda r: r['avg_test_error'])['best_params_single']
                   if valid else None)
    return errors, best_params


def _build_cv_dataframes(
    animal_id: str, results_dir: Path,
    distribution: str = 'uniform', fit_target: str = 'update_matrix',
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Build long_df and comparison_df for plotting.cv.plot_cv_comparison."""
    be_data, sc_data = _load_gs_raw_pickles(
        animal_id, results_dir, distribution, fit_target)
    if be_data is None or sc_data is None:
        return None, None

    rows = []
    for model, data in [('BE', be_data), ('SC', sc_data)]:
        for i, r in enumerate(data.get('results', [])):
            if np.isnan(r.get('avg_test_error', np.nan)):
                continue
            rows.append({
                'animal_id': animal_id, 'model': model,
                'seed': i, 'avg_test_error': r['avg_test_error'],
            })
    if not rows:
        return None, None

    long_df = pd.DataFrame(rows)
    be_errors = long_df[long_df['model'] == 'BE']['avg_test_error']
    sc_errors = long_df[long_df['model'] == 'SC']['avg_test_error']
    be_mean, sc_mean = be_errors.mean(), sc_errors.mean()
    winner = 'BE' if be_mean < sc_mean else 'SC'

    try:
        from scipy.stats import f_oneway
        _, p_val = f_oneway(be_errors.values, sc_errors.values)
    except Exception:
        p_val = np.nan

    comparison_df = pd.DataFrame([{
        'animal_id': animal_id, 'winner': winner, 'p_value': p_val,
        'be_mean': be_mean, 'sc_mean': sc_mean,
    }])
    return long_df, comparison_df


def _load_sbi_comparison(
    animal_id: str, results_dir: Path,
    distribution: str = 'uniform', fit_target: str = 'update_matrix',
) -> Optional[dict]:
    """Load SBI comparison pickle."""
    path = (results_dir / 'sbi_static' / 'comparisons'
            / f'{distribution}_{fit_target}' / f'animal_{animal_id}.pkl')
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_sbi_posterior_data(
    animal_id: str, results_dir: Path,
    distribution: str = 'uniform',
) -> Optional[dict]:
    """Load SBI posterior pickle (may contain samples in future)."""
    path = (results_dir / 'sbi_static' / distribution
            / f'animal_{animal_id}.pkl')
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def _get_params(
    animal_id: str, results_dir: Path,
    method: str = 'sbi', fit_target: str = 'update_matrix',
    distribution: str = 'uniform',
) -> Tuple[Optional[dict], Optional[dict], dict]:
    """
    Load best-fit BE and SC params from specified method.
    Fallback chain: SBI → SBI other fit target → GS.
    Returns (be_params, sc_params, source_info).
    """
    source = {'method': method, 'fit_target': fit_target,
              'distribution': distribution}

    if method == 'sbi':
        for ft in [fit_target] + [f for f in FIT_TARGETS if f != fit_target]:
            comp = _load_sbi_comparison(animal_id, results_dir, distribution, ft)
            if comp:
                bp, sp = comp.get('be_params'), comp.get('sc_params')
                if bp and sp:
                    source['fit_target'] = ft
                    source['winner'] = comp.get('winner')
                    source['p'] = comp.get('p')
                    if ft != fit_target:
                        source['fallback'] = True
                    return bp, sp, source

    # GS fallback (also used when method='gs')
    for ft in [fit_target] + [f for f in FIT_TARGETS if f != fit_target]:
        be_data, sc_data = _load_gs_raw_pickles(
            animal_id, results_dir, distribution, ft)
        if be_data and sc_data:
            _, be_params = _gs_seed_errors(be_data)
            _, sc_params = _gs_seed_errors(sc_data)
            if be_params and sc_params:
                source['method'] = 'gs'
                source['fit_target'] = ft
                if method != 'gs':
                    source['fallback'] = True
                return be_params, sc_params, source

    return None, None, source


def _get_sbi_samples(
    animal_id: str, sessions: list, results_dir: Path,
    snpe_dict: dict, distribution: str = 'uniform',
    n_samples: int = 10000,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get posterior samples. Fast path: saved in pickle. Slow path: re-condition.
    Returns (be_samples, sc_samples, x_obs).
    """
    import torch

    # Fast path
    post_data = _load_sbi_posterior_data(animal_id, results_dir, distribution)
    if (post_data and 'be_samples' in post_data
            and post_data['be_samples'] is not None):
        return (post_data['be_samples'], post_data['sc_samples'],
                post_data.get('observed_stats'))

    # Slow path: re-condition
    if not snpe_dict or 'be' not in snpe_dict or 'sc' not in snpe_dict:
        return None, None, None

    stat_names = snpe_dict['be']['stat_names']
    all_stats = []
    for sess in sessions:
        s = compute_summary_stats(
            sess.trials.choice, sess.trials.stimulus, sess.trials.category,
            stat_names=stat_names, return_dict=False)
        all_stats.append(s)
    x_obs = np.nanmean(all_stats, axis=0)

    if np.any(np.isnan(x_obs)):
        return None, None, x_obs

    x_t = torch.tensor(x_obs, dtype=torch.float32)
    samples = {}
    for mk in ['be', 'sc']:
        try:
            samples[mk] = snpe_dict[mk]['posterior'].sample(
                (n_samples,), x=x_t).numpy()
        except Exception as e:
            warnings.warn(f'Posterior sampling failed for {mk}: {e}')
            samples[mk] = None

    return samples.get('be'), samples.get('sc'), x_obs


def _params_to_str(params) -> str:
    """Format params (dict or dataclass) as compact string."""
    if params is None:
        return ''
    if hasattr(params, '__dict__'):
        d = {k: v for k, v in vars(params).items()
             if not str(k).startswith('_') and isinstance(v, (int, float))}
    elif isinstance(params, dict):
        d = {k: v for k, v in params.items() if isinstance(v, (int, float))}
    else:
        return str(params)
    return ', '.join(f'{k}={v:.3f}' for k, v in d.items())


def _plot_consensus_strip(
    animal_id: str, results_dir: Path,
    distribution: str = 'uniform',
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, dict]:
    """Plot 4-column consensus strip for one animal."""
    methods = {}
    for ft in FIT_TARGETS:
        ft_short = FT_LABEL[ft]
        # GS
        be_data, sc_data = _load_gs_raw_pickles(
            animal_id, results_dir, distribution, ft)
        if be_data and sc_data:
            be_e, _ = _gs_seed_errors(be_data)
            sc_e, _ = _gs_seed_errors(sc_data)
            if be_e and sc_e:
                from scipy.stats import wilcoxon
                n_paired = min(len(be_e), len(sc_e))
                try:
                    _, p_val = wilcoxon(be_e[:n_paired], sc_e[:n_paired])
                except Exception:
                    p_val = np.nan
                if p_val < 0.05:
                    methods[f'GS-{ft_short}'] = 'BE' if np.mean(be_e) < np.mean(sc_e) else 'SC'
                else:
                    methods[f'GS-{ft_short}'] = 'Inconclusive'
            else:
                methods[f'GS-{ft_short}'] = None
        else:
            methods[f'GS-{ft_short}'] = None
        # SBI
        comp = _load_sbi_comparison(animal_id, results_dir, distribution, ft)
        if comp:
            w = comp.get('winner')
            methods[f'SBI-{ft_short}'] = w if w in ('BE', 'SC') else w
        else:
            methods[f'SBI-{ft_short}'] = None

    method_names = list(methods.keys())
    n_m = len(method_names)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(n_m * 1.5 + 1, 1.5))
    else:
        fig = ax.figure

    colour_map = {'BE': BE_COL, 'SC': SC_COL}
    for j, mname in enumerate(method_names):
        val = methods[mname]
        fc = colour_map.get(val, '#EEEEEE')
        rect = plt.Rectangle((j - 0.4, -0.4), 0.8, 0.8,
                              facecolor=fc, edgecolor='black', lw=1)
        ax.add_patch(rect)
        label = val if val and val in ('BE', 'SC') else '?'
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

    valid = [v for v in methods.values() if v in ('BE', 'SC')]
    if valid:
        from collections import Counter
        majority = Counter(valid).most_common(1)[0]
        ax.set_title(f'{animal_id} — {majority[1]}/{len(valid)} say {majority[0]}',
                     fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'{animal_id} — No clear assignment',
                     fontsize=12, fontweight='bold')

    if own_fig:
        plt.tight_layout()
    return fig, methods


# =============================================================================
# PUBLIC API
# =============================================================================

def plot_animal_summary(
    animal_id: str,
    sessions: list,
    results_dir: Union[str, Path],
    distribution: str = 'uniform',
    phase_label: Optional[str] = None,
    true_params: Optional[Any] = None,
) -> dict:
    """
    Raw behaviour overview + method consensus strip.

    Produces:
        - Bootstrapped psychometric curve (via behav_utils.plot_psychometric)
        - Accuracy trajectory
        - Consensus strip (GS-UM | GS-CP | SBI-UM | SBI-CP)
    """
    results_dir = Path(results_dir)
    label = phase_label or distribution.replace('_', ' ').title()
    all_stim = np.concatenate([s.trials.stimulus for s in sessions])
    all_ch = np.concatenate([s.trials.choice for s in sessions])
    n_trials = len(all_stim)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5),
                             gridspec_kw={'width_ratios': [1, 1, 0.8]})

    # Panel 1: Psychometric (bootstrapped, via behav_utils)
    _, ax, psych_info = plot_psychometric(
        all_stim, all_ch, ax=axes[0],
        n_bootstrap=1000, show_ci=True, show_params=True,
        color='black', title='Psychometric',
    )

    # Panel 2: Accuracy trajectory
    ax = axes[1]
    accs = [np.nanmean(s.trials.correct) for s in sessions]
    ax.plot(range(len(accs)), accs, 'ko-', ms=4)
    ax.axhline(np.mean(accs), color='grey', ls='--', alpha=0.5)
    ax.set(xlabel=f'Session ({label})', ylabel='Accuracy', ylim=(0.4, 1.0))
    ax.set_title(f'Accuracy ({np.mean(accs):.1%} mean)', fontsize=10)

    # Panel 3: Consensus strip
    _plot_consensus_strip(animal_id, results_dir, distribution, ax=axes[2])

    title = f'{animal_id} — {label} ({len(sessions)} sessions, {n_trials} trials)'
    if true_params:
        title += f'\nTrue: {_params_to_str(true_params)}'
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return {'psychometric': psych_info}


def plot_cv_results(
    animal_id: str,
    results_dir: Union[str, Path],
    distribution: str = 'uniform',
    fit_target: str = 'update_matrix',
) -> Optional[plt.Figure]:
    """
    GS CV seed-level comparison: split violin + paired scatter.
    Wraps plotting.cv.plot_cv_comparison.
    """
    results_dir = Path(results_dir)
    long_df, comparison_df = _build_cv_dataframes(
        animal_id, results_dir, distribution, fit_target)

    if long_df is None:
        print(f'  No GS data for {animal_id} ({distribution}/{fit_target})')
        return None

    from plotting.cv import plot_cv_comparison
    fig = plot_cv_comparison(
        long_df, comparison_df, animal_id,
        fit_target=FT_LABEL[fit_target],
    )
    plt.show()
    return fig


def plot_model_fits(
    animal_id: str,
    sessions: list,
    results_dir: Union[str, Path],
    method: str = 'sbi',
    fit_target: str = 'update_matrix',
    distribution: str = 'uniform',
    phase_label: Optional[str] = None,
    true_params: Optional[Any] = None,
    n_boot: int = 500,
    n_reps: int = 20,
    burn_in: int = 1000,
    seed: int = 42,
) -> Optional[dict]:
    """
    Model fit assessment.

    Loads params from specified method, simulates both models.

    Produces:
        - UM comparison (via behav_utils.plot_phase_update_matrices)
        - Psychometric overlay (data via behav_utils.plot_psychometric + model curves)
        - Per-session MSE trajectory
        - Psychometric parameter comparison table
    """
    results_dir = Path(results_dir)
    label = phase_label or distribution.replace('_', ' ').title()

    # ── Load params ──────────────────────────────────────────────────────
    be_params, sc_params, source = _get_params(
        animal_id, results_dir, method, fit_target, distribution)

    if not be_params or not sc_params:
        print(f'  No params for {animal_id} '
              f'(method={method}, ft={fit_target}, dist={distribution})')
        return None

    src_label = f'{source["method"].upper()}-{FT_LABEL[source["fit_target"]]}'
    if source.get('fallback'):
        src_label += ' (fallback)'
    print(f'  {animal_id}: params from {src_label}')
    print(f'    BE: {_params_to_str(be_params)}')
    print(f'    SC: {_params_to_str(sc_params)}')

    # ── Simulate ─────────────────────────────────────────────────────────
    from behav_utils.data.selection import fitting_data_from_sessions
    from inference.comparison import simulate_all_sessions

    fd = fitting_data_from_sessions(sessions, animal_id)
    print(f'  Simulating ({fd.n_sessions} sessions, {n_reps} reps)...')
    session_data = simulate_all_sessions(
        fd, be_params, sc_params,
        burn_in=burn_in, n_reps=n_reps, n_bins=8, seed=seed,
    )
    print(f'  {len(session_data)} sessions done')

    # ── Plot 1: UM comparison (via behav_utils) ──────────────────────────
    emp_um = np.nanmean([d['real_um'] for d in session_data], axis=0)
    be_um = np.nanmean([d['be_um'] for d in session_data], axis=0)
    sc_um = np.nanmean([d['sc_um'] for d in session_data], axis=0)
    be_mse = matrix_error(be_um, emp_um)
    sc_mse = matrix_error(sc_um, emp_um)

    fig, axes = plot_phase_update_matrices(
        {'Real': emp_um, 'BE': be_um, 'SC': sc_um},
        annotations={
            'Real': f'n={len(session_data)} sessions',
            'BE': f'MSE={be_mse:.5f}',
            'SC': f'MSE={sc_mse:.5f}',
        },
        suptitle=f'{animal_id} — UM Comparison ({label}, {src_label})',
    )
    plt.show()

    # ── Plot 2: Psychometric overlay ─────────────────────────────────────
    all_stim = np.concatenate([d['stimuli'] for d in session_data])
    all_ch = np.concatenate([d['choices'] for d in session_data])
    x_fine = np.linspace(-1.1, 1.1, 200)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Real data with bootstrap CI (via behav_utils)
    _, ax, data_fit = plot_psychometric(
        all_stim, all_ch, ax=ax,
        n_bootstrap=n_boot, show_ci=True, show_params=False,
        color='black', label='Data',
    )

    # Model curves overlaid
    for mk, col in [('be', BE_COL), ('sc', SC_COL)]:
        ps = [d[f'{mk}_psych'] for d in session_data
              if d[f'{mk}_psych'].get('success')]
        if ps:
            mu = np.mean([p['mu'] for p in ps])
            sig = np.mean([p['sigma'] for p in ps])
            ll = np.mean([p['lapse_low'] for p in ps])
            lh = np.mean([p['lapse_high'] for p in ps])
            ax.plot(x_fine, cumulative_gaussian(x_fine, mu, sig, ll, lh),
                    '-', color=col, lw=2, label=f'{mk.upper()} (PSE={mu:.3f})')

    ax.legend(fontsize=9)
    ax.set_title(f'{animal_id} — Psychometric Overlay ({label}, {src_label})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ── Psychometric parameter comparison ────────────────────────────────
    print(f'\n  Psychometric parameter comparison:')
    header = f'  {"":12s} {"PSE":>12s}  {"Slope (σ)":>12s}  {"Lapse low":>12s}  {"Lapse high":>12s}'
    print(header)

    if data_fit.get('success'):
        print(f'  {"Data":12s} {data_fit["mu"]:12.3f}  {data_fit["sigma"]:12.3f}  '
              f'{data_fit["lapse_low"]:12.3f}  {data_fit["lapse_high"]:12.3f}')

    for mk in ['be', 'sc']:
        ps = [d[f'{mk}_psych'] for d in session_data
              if d[f'{mk}_psych'].get('success')]
        if ps:
            mu = np.mean([p['mu'] for p in ps])
            sig = np.mean([p['sigma'] for p in ps])
            ll = np.mean([p['lapse_low'] for p in ps])
            lh = np.mean([p['lapse_high'] for p in ps])
            print(f'  {mk.upper():12s} {mu:12.3f}  {sig:12.3f}  {ll:12.3f}  {lh:12.3f}')

    # ── Plot 3: Per-session MSE trajectory ───────────────────────────────
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
        f'{animal_id} — Per-Session Fit ({label}, {src_label})\n'
        f'BE={bmu:.5f}, SC={smu:.5f} → {w} '
        f'(BE wins {bw}/{len(be_mses)} sessions)',
        fontsize=11, fontweight='bold',
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    return {
        'session_data': session_data,
        'be_params': be_params,
        'sc_params': sc_params,
        'source': source,
    }


def plot_sbi_diagnostics(
    animal_id: str,
    sessions: list,
    snpe_dict: dict,
    results_dir: Union[str, Path],
    distribution: str = 'uniform',
    true_params: Optional[Any] = None,
    n_samples: int = 10000,
    n_ppc_samples: int = 200,
    show_ppc: bool = True,
) -> Optional[dict]:
    """
    SBI-specific diagnostics.

    Produces:
        - Posterior marginals (BE row + SC row) — custom (different input
          format from plotting.sbi.plot_marginal_posteriors)
        - GS vs SBI overlay (if GS params available)
        - PPC (via plotting.sbi.plot_summary_stats_comparison)
    """
    import torch
    results_dir = Path(results_dir)

    if not snpe_dict or 'be' not in snpe_dict or 'sc' not in snpe_dict:
        print(f'  SNPE networks not available — skipping SBI diagnostics')
        return None

    # ── Get posterior samples ─────────────────────────────────────────────
    be_samples, sc_samples, x_obs = _get_sbi_samples(
        animal_id, sessions, results_dir, snpe_dict, distribution, n_samples)

    if be_samples is None or sc_samples is None:
        print(f'  Could not get posterior samples for {animal_id}')
        return None

    true_dict = None
    if true_params is not None:
        if hasattr(true_params, '__dict__'):
            true_dict = {k: v for k, v in vars(true_params).items()
                         if not str(k).startswith('_')
                         and isinstance(v, (int, float))}
        elif isinstance(true_params, dict):
            true_dict = true_params

    all_samples = {'be': be_samples, 'sc': sc_samples}

    # ── Plot 1: Posterior marginals ──────────────────────────────────────
    # Custom implementation: takes raw (n_samples, n_params) arrays.
    # plotting.sbi.plot_marginal_posteriors expects trajectories format
    # (SBIFitter output with link_type), which doesn't apply here.
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for row, (mk, col, lab) in enumerate([
        ('be', BE_COL, 'BE'), ('sc', SC_COL, 'SC'),
    ]):
        pnames = snpe_dict[mk]['param_names']
        samples = all_samples[mk]

        for j, pn in enumerate(pnames):
            ax = axes[row, j]
            ax.hist(samples[:, j], bins=50, color=col, alpha=0.5,
                    density=True, edgecolor='none')
            med = np.median(samples[:, j])
            lo, hi = np.percentile(samples[:, j], [5, 95])
            ax.axvline(med, color=col, lw=2)
            ax.axvspan(lo, hi, alpha=0.12, color=col)

            if true_dict and pn in true_dict:
                ax.axvline(true_dict[pn], color='red', lw=2.5, ls='-',
                           label=f'True={true_dict[pn]:.3f}')
                ax.legend(fontsize=7)

            ax.set_xlabel(pn, fontsize=9)
            ax.set_title(f'{med:.3f} [{lo:.3f}, {hi:.3f}]', fontsize=8)
            if j == 0:
                ax.set_ylabel(lab, fontsize=11, fontweight='bold')

    fig.suptitle(f'{animal_id} — SBI Posterior Marginals',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ── Plot 2: GS vs SBI overlay ────────────────────────────────────────
    gs_params_by_model = {}
    for ft in FIT_TARGETS:
        be_data, sc_data = _load_gs_raw_pickles(
            animal_id, results_dir, distribution, ft)
        if be_data and sc_data:
            _, be_p = _gs_seed_errors(be_data)
            _, sc_p = _gs_seed_errors(sc_data)
            if be_p:
                gs_params_by_model['be'] = be_p
            if sc_p:
                gs_params_by_model['sc'] = sc_p
            break

    if gs_params_by_model:
        for mk, col, lab in [('be', BE_COL, 'BE'), ('sc', SC_COL, 'SC')]:
            gs_p = gs_params_by_model.get(mk)
            if not gs_p:
                continue
            pnames = snpe_dict[mk]['param_names']
            samples = all_samples[mk]

            fig, axes_row = plt.subplots(1, len(pnames),
                                         figsize=(5 * len(pnames), 4.5))
            axes_row = np.atleast_1d(axes_row)

            for j, pn in enumerate(pnames):
                ax = axes_row[j]
                ax.hist(samples[:, j], bins=50, color=col, alpha=0.4,
                        density=True, edgecolor='none', label='SBI posterior')
                med = np.median(samples[:, j])
                ax.axvline(med, color=col, lw=1.5, ls=':',
                           label=f'SBI median={med:.3f}')
                if pn in gs_p:
                    ax.axvline(gs_p[pn], color='grey', lw=2.5, ls='--',
                               label=f'GS={gs_p[pn]:.3f}')
                if true_dict and pn in true_dict:
                    ax.axvline(true_dict[pn], color='red', lw=2.5,
                               label=f'True={true_dict[pn]:.3f}')
                ax.set_xlabel(pn, fontsize=11)
                ax.legend(fontsize=8)

            fig.suptitle(f'{animal_id} — {lab}: GS vs SBI',
                         fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.show()

    # ── Plot 3: Posterior Predictive Check (via plotting.sbi) ────────────
    if show_ppc and x_obs is not None:
        from inference.simulator import (
            create_be_simulator, create_sc_simulator, wrap_for_sbi,
        )
        from plotting.sbi import plot_summary_stats_comparison

        stat_names_raw = snpe_dict['be']['stat_names']
        stat_names = get_stat_names_expanded(stat_names_raw)        
        ref_stim = np.concatenate([s.trials.stimulus for s in sessions[:3]])
        ref_cat = np.concatenate([s.trials.category for s in sessions[:3]])
                
        for mk, col, lab, create_fn in [
            ('be', BE_COL, 'BE', create_be_simulator),
            ('sc', SC_COL, 'SC', create_sc_simulator),
        ]:
            samples = all_samples[mk][:n_ppc_samples]
            sim = create_fn(
                stimuli=ref_stim, categories=ref_cat,
                stat_names=stat_names,
                burn_in=snpe_dict[mk]['burn_in'], seed=42,
            )
            sbi_sim = wrap_for_sbi(sim)

            pred = []
            for si in range(len(samples)):
                theta = torch.tensor(samples[si], dtype=torch.float32)
                try:
                    x_sim = sbi_sim(theta)
                    if x_sim is not None and not torch.any(torch.isnan(x_sim)):
                        pred.append(x_sim.numpy())
                except Exception:
                    pass

            if pred:
                pred_arr = np.array(pred)
                fig = plot_summary_stats_comparison(
                    x_obs, pred_arr, stat_names,
                    title=f'{animal_id} — {lab} Posterior Predictive Check',
                )
                plt.show()
            else:
                print(f'  {lab} PPC: all simulations failed')

    return {
        'be_samples': be_samples,
        'sc_samples': sc_samples,
        'x_obs': x_obs,
    }
