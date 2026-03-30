"""
SBI Model Comparison Utilities

Functions for BE vs SC model comparison via:
  - Amortised SNPE (Parts 1 & 3)
  - Per-animal SNPE with UM in stats (Part 2 — matches manuscript protocol)
  - Cross-validated model assignment (2-fold × 64 repeats, ANOVA)

Used by notebook 3_SBI_Model_Comparison.ipynb.
"""

import numpy as np
import warnings
import time
from typing import Dict, List, Tuple, Optional, Any

from behav_utils.data.structures import AnimalData, FittingData
from behav_utils.analysis.summary_stats import (
    compute_summary_stats, get_stat_names_expanded,
)
from behav_utils.analysis.update_matrix import compute_update_matrix, matrix_error
from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.analysis.utils import cumulative_gaussian
from behav_utils.data.synthetic import sample_stimuli

from Inference.simulator import (
    ModelType, create_be_simulator, create_sc_simulator,
    get_sbi_prior, wrap_for_sbi,
)


# =============================================================================
# EXPERT SESSION SELECTION
# =============================================================================

def select_expert_sessions(
    animal: AnimalData,
    stage: str = 'Full_Task_Cont',
    distribution: str = 'Uniform',
    min_accuracy: float = 0.70,
    last_fraction: float = 0.50,
    min_valid_trials: int = 30,
) -> FittingData:
    """
    Select expert sessions: intersection of accuracy >= threshold
    AND last fraction of qualifying sessions.
    """
    fd = animal.get_fitting_data(
        stage=stage, distribution=distribution,
        min_valid_trials=min_valid_trials,
    )
    if fd.n_sessions == 0:
        raise ValueError(f"No sessions for {animal.animal_id}")

    n_last = max(1, int(np.ceil(fd.n_sessions * last_fraction)))
    start_idx = fd.n_sessions - n_last

    keep = []
    for i in range(start_idx, fd.n_sessions):
        v = ~fd.no_response[i]
        if v.sum() < min_valid_trials:
            continue
        acc = np.mean(fd.choices[i][v] == fd.categories[i][v])
        if acc >= min_accuracy:
            keep.append(i)

    if not keep:
        raise ValueError(
            f"No expert sessions for {animal.animal_id} "
            f"(acc>={min_accuracy}, last {last_fraction:.0%})"
        )

    return _slice_fitting_data(fd, keep)


def select_all_sessions(
    animal: AnimalData,
    stage: str = 'Full_Task_Cont',
    distribution: str = 'Uniform',
    min_valid_trials: int = 30,
) -> FittingData:
    """Select all qualifying sessions including learning."""
    return animal.get_fitting_data(
        stage=stage, distribution=distribution,
        min_valid_trials=min_valid_trials,
    )


def _slice_fitting_data(fd: FittingData, indices: List[int]) -> FittingData:
    """Slice FittingData to keep only given session indices."""
    idx = np.array(indices)
    return FittingData(
        animal_id=fd.animal_id,
        session_ids=[fd.session_ids[i] for i in indices],
        session_dates=[fd.session_dates[i] for i in indices],
        session_indices=fd.session_indices[idx],
        stimuli=[fd.stimuli[i] for i in indices],
        categories=[fd.categories[i] for i in indices],
        choices=[fd.choices[i] for i in indices],
        no_response=[fd.no_response[i] for i in indices],
        not_blockstart=[fd.not_blockstart[i] for i in indices],
        n_sessions=len(indices),
        trials_per_session=fd.trials_per_session[idx],
    )


# =============================================================================
# TIMING ESTIMATION
# =============================================================================

def estimate_timing(
    stat_names: List[str],
    n_trials: int = 2500,
    burn_in: int = 1000,
    n_sbi_sims: int = 50_000,
    n_test: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Estimate per-simulation cost for BE and SC.

    Runs n_test forward simulations with each model and reports
    timing + NaN rate + projected total training time.

    Returns dict with per-model timing info.
    """
    stim, cat = sample_stimuli(n_trials, 'uniform', np.random.default_rng(seed))
    results = {}

    for model_type in ['be', 'sc']:
        creator = create_be_simulator if model_type == 'be' else create_sc_simulator
        sim = creator(stim, cat, stat_names=stat_names, burn_in=burn_in)

        times = []
        nan_count = 0
        for i in range(n_test):
            theta = sim.sample_prior(seed=seed + i)
            t0 = time.time()
            stats = sim(theta, seed=seed + i)
            times.append(time.time() - t0)
            if np.any(np.isnan(stats)):
                nan_count += 1

        ms_per_sim = np.mean(times) * 1000
        total_min = np.mean(times) * n_sbi_sims / 60
        n_stat_dims = len(stats)

        results[model_type] = {
            'ms_per_sim': ms_per_sim,
            'total_minutes': total_min,
            'total_hours': total_min / 60,
            'nan_rate': nan_count / n_test,
            'stat_dims': n_stat_dims,
            'theta_dims': sim.n_free_params,
        }

    return results


def print_timing_report(
    timing: Dict[str, Any],
    n_sbi_sims: int,
    n_animals: int = 1,
    label: str = '',
):
    """Print a formatted timing report."""
    print(f"\n{'='*60}")
    if label:
        print(f"  Timing estimate: {label}")
    print(f"  {n_sbi_sims:,} simulations")
    print(f"{'='*60}")
    print(f"  {'Model':<6s} {'ms/sim':>8s} {'Total':>10s} {'NaN%':>6s} "
          f"{'θ dims':>7s} {'Stat dims':>10s}")
    print(f"  {'-'*50}")

    for mt in ['be', 'sc']:
        t = timing[mt]
        total_str = f"{t['total_hours']:.1f}h" if t['total_hours'] >= 1 else f"{t['total_minutes']:.0f}min"
        print(f"  {mt.upper():<6s} {t['ms_per_sim']:8.0f} {total_str:>10s} "
              f"{t['nan_rate']:5.0%} {t['theta_dims']:>7d} {t['stat_dims']:>10d}")

    if n_animals > 1:
        be_h = timing['be']['total_hours']
        sc_h = timing['sc']['total_hours']
        total = (be_h + sc_h) * n_animals
        print(f"\n  {n_animals} animals × 2 models = ~{total:.0f} hours total")


# =============================================================================
# AMORTISED SNPE (Parts 1 & 3)
# =============================================================================

def train_amortised_snpe(
    model_type: str,
    stat_names: List[str],
    n_simulations: int = 50_000,
    n_trials: int = 2500,
    burn_in: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train amortised SNPE with generic Uniform stimuli.
    Condition on any animal's stats without retraining.

    stat_names should NOT include update_matrix (sequence-dependent).
    """
    import torch
    from sbi.inference import SNPE

    name = model_type.upper()
    print(f"\nTraining amortised SNPE [{name}] "
          f"({n_simulations:,} sims, {n_trials} trials, burn_in={burn_in})...")

    stim, cat = sample_stimuli(n_trials, 'uniform', np.random.default_rng(seed))
    creator = create_be_simulator if model_type == 'be' else create_sc_simulator
    sim = creator(stim, cat, stat_names=stat_names, burn_in=burn_in)
    prior = get_sbi_prior(sim)
    sbi_sim = wrap_for_sbi(sim)

    t0 = time.time()
    theta = prior.sample((n_simulations,))
    print(f"  Simulating...")
    x = torch.stack([sbi_sim(t) for t in theta])

    valid = ~torch.any(torch.isnan(x), dim=1)
    n_valid = valid.sum().item()
    print(f"  {n_valid}/{n_simulations} valid ({100*n_valid/n_simulations:.0f}%)")

    inference = SNPE(prior=prior)
    inference.append_simulations(theta[valid], x[valid])
    posterior = inference.build_posterior(inference.train())

    dt = time.time() - t0
    print(f"  Done in {dt/60:.1f} min")

    return {
        'posterior': posterior, 'prior': prior,
        'simulator': sim, 'sbi_sim': sbi_sim,
        'param_names': sim.get_param_names(),
        'model_type': model_type, 'stat_names': stat_names,
        'burn_in': burn_in, 'training_time': dt, 'n_valid': n_valid,
    }


# =============================================================================
# PER-ANIMAL SNPE WITH UM (Part 2)
# =============================================================================

def train_per_animal_snpe(
    model_type: str,
    fitting_data: FittingData,
    stat_names: List[str],
    n_simulations: int = 10_000,
    burn_in: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train SNPE for one animal using its real stimulus sequence.

    stat_names CAN include update_matrix here because the simulator
    uses the animal's actual stimuli — no sequence mismatch.

    This is the SNPE equivalent of the manuscript's grid search:
    parameters are optimised to reproduce the UM pattern from this
    specific animal's stimulus sequence.
    """
    import torch
    from sbi.inference import SNPE

    name = model_type.upper()
    aid = fitting_data.animal_id
    pooled = fitting_data.pool()
    stim, cat = pooled['stimuli'], pooled['categories']

    print(f"  Training per-animal SNPE [{name}] for {aid} "
          f"({n_simulations:,} sims, {len(stim)} trials)...")

    creator = create_be_simulator if model_type == 'be' else create_sc_simulator
    sim = creator(stim, cat, stat_names=stat_names, burn_in=burn_in)
    prior = get_sbi_prior(sim)
    sbi_sim = wrap_for_sbi(sim)

    t0 = time.time()
    theta = prior.sample((n_simulations,))
    x = torch.stack([sbi_sim(t) for t in theta])

    valid = ~torch.any(torch.isnan(x), dim=1)
    n_valid = valid.sum().item()
    print(f"    {n_valid}/{n_simulations} valid ({100*n_valid/n_simulations:.0f}%)")

    inference = SNPE(prior=prior)
    inference.append_simulations(theta[valid], x[valid])
    posterior = inference.build_posterior(inference.train())

    dt = time.time() - t0
    print(f"    Done in {dt/60:.1f} min")

    return {
        'posterior': posterior, 'prior': prior,
        'simulator': sim, 'sbi_sim': sbi_sim,
        'param_names': sim.get_param_names(),
        'model_type': model_type, 'stat_names': stat_names,
        'burn_in': burn_in, 'training_time': dt, 'n_valid': n_valid,
    }


# =============================================================================
# POSTERIOR CONDITIONING
# =============================================================================

def condition_on_animal(
    snpe_result: Dict[str, Any],
    fitting_data: FittingData,
    n_samples: int = 2000,
) -> Dict[str, Any]:
    """Condition posterior on one animal's observed stats."""
    import torch

    pooled = fitting_data.pool()
    obs = compute_summary_stats(
        pooled['choices'], pooled['stimuli'], pooled['categories'],
        stat_names=snpe_result['stat_names'], return_dict=False,
    )
    obs = np.nan_to_num(obs, nan=0.0)

    x_obs = torch.tensor(obs, dtype=torch.float32)
    samples = snpe_result['posterior'].sample((n_samples,), x=x_obs).numpy()

    param_names = snpe_result['param_names']
    median_params = {
        name: float(np.median(samples[:, i]))
        for i, name in enumerate(param_names)
    }

    return {
        'samples': samples, 'median_params': median_params,
        'param_names': param_names, 'observed_stats': obs,
        'animal_id': fitting_data.animal_id,
    }


# =============================================================================
# CROSS-VALIDATION (shared by Parts 1, 2, 3)
# =============================================================================

def cv_um_comparison(
    snpe_result: Dict[str, Any],
    fitting_data: FittingData,
    n_folds: int = 2,
    n_repeats: int = 64,
    n_posterior_samples: int = 50,
    n_stochastic_reps: int = 10,
    n_bins: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Cross-validated UM comparison.

    For each fold:
    1. Compute training-fold observed stats
    2. Condition posterior on training-fold stats → get params
    3. Simulate on test-fold stimuli with those params
    4. Compute UM MSE on test fold

    Works for BOTH amortised (Parts 1&3) and per-animal (Part 2) SNPE.
    The posterior was trained on either generic or real stimuli — the CV
    reconditioning uses the same stat format either way.
    """
    import torch

    pooled = fitting_data.pool()
    stim, cat, choices = pooled['stimuli'], pooled['categories'], pooled['choices']
    n_trials = len(stim)

    model_type = snpe_result['model_type']
    stat_names = snpe_result['stat_names']
    burn_in = snpe_result['burn_in']
    param_names = snpe_result['param_names']
    creator = create_be_simulator if model_type == 'be' else create_sc_simulator

    rng = np.random.default_rng(seed)
    test_errors = []

    for rep in range(n_repeats):
        perm = rng.permutation(n_trials)
        fold_size = n_trials // n_folds
        fold_errors = []

        for fold in range(n_folds):
            test_idx = perm[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.setdiff1d(np.arange(n_trials), test_idx)

            if len(test_idx) < 100 or len(train_idx) < 100:
                continue

            # Condition posterior on TRAINING fold
            train_obs = compute_summary_stats(
                choices[train_idx], stim[train_idx], cat[train_idx],
                stat_names=stat_names, return_dict=False,
            )
            train_obs = np.nan_to_num(train_obs, nan=0.0)

            x_train = torch.tensor(train_obs, dtype=torch.float32)
            try:
                post_samples = snpe_result['posterior'].sample(
                    (n_posterior_samples,), x=x_train
                ).numpy()
            except Exception:
                continue

            fold_params = {
                name: float(np.median(post_samples[:, i]))
                for i, name in enumerate(param_names)
            }

            # Empirical UM on TEST fold
            emp_um, _, _ = compute_update_matrix(
                stim[test_idx], choices[test_idx], cat[test_idx], n_bins=n_bins,
            )

            # Simulated UM on test fold
            sim = creator(
                stim[test_idx], cat[test_idx],
                fixed_params=fold_params,
                stat_names=['accuracy'], burn_in=burn_in,
            )
            sim_ums = []
            for j in range(n_stochastic_reps):
                try:
                    _, sim_ch = sim.simulate(
                        sim.sample_prior(seed=rep * 1000 + fold * 100 + j),
                        seed=rep * 1000 + fold * 100 + j,
                        return_choices=True,
                    )
                    um_j, _, _ = compute_update_matrix(
                        stim[test_idx], sim_ch.flatten(), cat[test_idx], n_bins=n_bins,
                    )
                    if not np.all(np.isnan(um_j)):
                        sim_ums.append(um_j)
                except Exception:
                    continue

            if sim_ums:
                fold_errors.append(matrix_error(np.nanmean(sim_ums, axis=0), emp_um))

        if fold_errors:
            test_errors.append(np.mean(fold_errors))

    return {
        'test_errors': np.array(test_errors),
        'mean_error': np.nanmean(test_errors),
        'std_error': np.nanstd(test_errors),
    }


# =============================================================================
# MODEL COMPARISON (ANOVA)
# =============================================================================

def compare_models(be_cv: Dict, sc_cv: Dict, alpha: float = 0.05) -> Dict:
    """ANOVA on CV test errors. Returns winner + p-value."""
    from scipy.stats import f_oneway

    be, sc = be_cv['test_errors'], sc_cv['test_errors']
    if len(be) < 2 or len(sc) < 2:
        return {'f_stat': np.nan, 'p_value': np.nan, 'winner': 'insufficient_data',
                'be_mean': np.nanmean(be), 'sc_mean': np.nanmean(sc),
                'be_std': np.nanstd(be), 'sc_std': np.nanstd(sc)}

    f_stat, p = f_oneway(be, sc)
    winner = ('BE' if np.mean(be) < np.mean(sc) else 'SC') if p < alpha else 'tied'

    return {
        'f_stat': f_stat, 'p_value': p, 'winner': winner,
        'be_mean': np.mean(be), 'be_std': np.std(be),
        'sc_mean': np.mean(sc), 'sc_std': np.std(sc),
    }


# =============================================================================
# EXAMPLE SESSION VISUALIZATION
# =============================================================================

def simulate_example_session(
    animal: AnimalData, session_idx: int,
    be_params: Dict, sc_params: Dict,
    stage: str = 'Full_Task_Cont', distribution: str = 'Uniform',
    burn_in: int = 1000, n_reps: int = 20, seed: int = 42,
) -> Dict[str, Any]:
    """Simulate BE and SC on one real session for visualisation."""
    from Models.BE_core import BEParams, BEModel
    from Models.SC_core import SCParams, SCModel

    sessions = animal.get_sessions(stage=stage, distribution=distribution)
    sess = sessions[session_idx]
    arrays = sess.trials.get_arrays(exclude_abort=True, exclude_opto=True)
    valid = ~arrays['no_response']
    stim, cat, ch = arrays['stimuli'][valid], arrays['categories'][valid], arrays['choices'][valid]

    # p(B) trajectories
    be_p = BEParams(**be_params)
    be_state = BEModel.create_initial_state(params=be_p, burn_in=burn_in, seed=seed)
    _, be_pB, _, _ = BEModel.simulate_session(
        be_p, be_state, stim, cat, np.random.default_rng(seed), return_history=False)

    sc_p = SCParams(**sc_params)
    sc_state = SCModel.create_initial_state(params=sc_p, burn_in=burn_in, seed=seed)
    _, sc_pB, _, _ = SCModel.simulate_session(
        sc_p, sc_state, stim, cat, np.random.default_rng(seed), return_history=False)

    # Stochastic realisations
    be_all, sc_all = [], []
    for r in range(n_reps):
        rng_r = np.random.default_rng(seed + r + 1)
        s1 = BEModel.create_initial_state(params=be_p, burn_in=burn_in, seed=seed)
        c1, _, _, _ = BEModel.simulate_session(be_p, s1, stim, cat, rng_r, return_history=False)
        be_all.append(c1)
        s2 = SCModel.create_initial_state(params=sc_p, burn_in=burn_in, seed=seed)
        c2, _, _, _ = SCModel.simulate_session(sc_p, s2, stim, cat, rng_r, return_history=False)
        sc_all.append(c2)

    return {
        'stimuli': stim, 'categories': cat, 'choices': ch,
        'be_pB': be_pB, 'sc_pB': sc_pB,
        'be_choices_all': be_all, 'sc_choices_all': sc_all,
        'session_info': {
            'session_id': sess.session_id, 'n_trials': len(stim),
            'accuracy': float(np.mean(ch == cat)),
        },
    }


def plot_example_session(
    data: Dict, animal_id: str,
    be_colour: str = 'steelblue', sc_colour: str = 'darkorange',
    n_bins: int = 8,
):
    """4-panel session comparison + 3-panel UM comparison."""
    import matplotlib.pyplot as plt
    from behav_utils.plotting.update_matrix import plot_update_matrix as _plot_um

    stim, cat, ch = data['stimuli'], data['categories'], data['choices']
    info = data['session_info']
    n = len(stim)
    trials = np.arange(n)
    correct = (ch == cat).astype(bool)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Trial-by-trial p(B)
    ax = axes[0, 0]
    ax.scatter(trials, ch, c=ch, cmap='coolwarm', s=8, alpha=0.4)
    ax.plot(trials, data['be_pB'], '-', color=be_colour, lw=1.2, alpha=0.8, label='BE')
    ax.plot(trials, data['sc_pB'], '-', color=sc_colour, lw=1.2, alpha=0.8, label='SC')
    ax.axhline(0.5, color='grey', ls='--', alpha=0.3)
    ax.set_xlabel('Trial'); ax.set_ylabel('P(B)')
    ax.set_title('Trial-by-trial choice probability')
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05)

    # Stimulus sequence
    ax = axes[0, 1]
    ax.scatter(trials[correct], stim[correct], c='green', s=6, alpha=0.5, label='Correct')
    ax.scatter(trials[~correct], stim[~correct], c='red', s=6, alpha=0.5, label='Error')
    ax.axhline(0, color='grey', ls='--', alpha=0.3)
    ax.set_xlabel('Trial'); ax.set_ylabel('Stimulus')
    ax.set_title('Stimulus sequence'); ax.legend(fontsize=8)

    # Psychometric curves
    ax = axes[1, 0]
    x_fine = np.linspace(-1.1, 1.1, 200)
    real_psych = fit_psychometric(stim, ch)
    if real_psych['success']:
        ax.plot(x_fine, cumulative_gaussian(x_fine, real_psych['mu'], real_psych['sigma'],
                real_psych['lapse_low'], real_psych['lapse_high']),
                'k-', lw=2.5, label='Real', zorder=10)
    for label, calls, col in [('BE', data['be_choices_all'], be_colour),
                               ('SC', data['sc_choices_all'], sc_colour)]:
        mc = np.nanmean(calls, axis=0)
        p = fit_psychometric(stim, mc)
        if p['success']:
            ax.plot(x_fine, cumulative_gaussian(x_fine, p['mu'], p['sigma'],
                    p['lapse_low'], p['lapse_high']), '-', color=col, lw=2, label=label)
    ax.axhline(0.5, color='grey', ls='--', alpha=0.3)
    ax.axvline(0, color='grey', ls='--', alpha=0.3)
    ax.set_xlabel('Stimulus'); ax.set_ylabel('P(B)')
    ax.set_title('Psychometric curves'); ax.legend(fontsize=8)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.05, 1.05)

    # Real UM
    real_um, _, _ = compute_update_matrix(stim, ch, cat, n_bins=n_bins)
    _plot_um(real_um, ax=axes[1, 1], title='Real UM (this session)')

    fig.suptitle(f'{animal_id} — {info["session_id"]} '
                 f'({info["n_trials"]} trials, acc={info["accuracy"]:.0%})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()

    # 3-panel UM comparison
    be_ums = [compute_update_matrix(stim[~np.isnan(c)], c[~np.isnan(c)],
              cat[~np.isnan(c)], n_bins)[0] for c in data['be_choices_all']
              if (~np.isnan(c)).sum() > 50]
    sc_ums = [compute_update_matrix(stim[~np.isnan(c)], c[~np.isnan(c)],
              cat[~np.isnan(c)], n_bins)[0] for c in data['sc_choices_all']
              if (~np.isnan(c)).sum() > 50]
    be_mu = np.nanmean(be_ums, axis=0) if be_ums else None
    sc_mu = np.nanmean(sc_ums, axis=0) if sc_ums else None

    if be_mu is not None and sc_mu is not None:
        fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4.5))
        vlim = max(np.nanmax(np.abs(real_um)), np.nanmax(np.abs(be_mu)), np.nanmax(np.abs(sc_mu)))
        for a, u, t in [(ax2[0], real_um, 'Real'),
                         (ax2[1], be_mu, f'BE (MSE={matrix_error(be_mu, real_um):.5f})'),
                         (ax2[2], sc_mu, f'SC (MSE={matrix_error(sc_mu, real_um):.5f})')]:
            _plot_um(u, ax=a, vmin=-vlim, vmax=vlim); a.set_title(t)
        fig2.suptitle('Session UM Comparison', fontsize=12)
        plt.tight_layout(); plt.show()


# =============================================================================
# CV RESULT PLOTTING
# =============================================================================

def plot_cv_comparison(
    be_cv: Dict, sc_cv: Dict, comparison: Dict,
    animal_id: str, title_suffix: str = '',
    be_colour: str = 'steelblue', sc_colour: str = 'darkorange',
):
    """Paired violin + scatter plot of CV test errors."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    n = min(len(be_cv['test_errors']), len(sc_cv['test_errors']))

    ax = axes[0]
    for i in range(n):
        ax.plot([0, 1], [be_cv['test_errors'][i], sc_cv['test_errors'][i]],
                'o-', color='grey', alpha=0.15, markersize=3)
    parts = ax.violinplot([be_cv['test_errors'], sc_cv['test_errors']],
                          positions=[0, 1], showmedians=True)
    for pc, col in zip(parts['bodies'], [be_colour, sc_colour]):
        pc.set_facecolor(col); pc.set_alpha(0.3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['BE', 'SC'])
    ax.set_ylabel('Test UM MSE')
    ax.set_title(f'p={comparison["p_value"]:.3g}, winner={comparison["winner"]}')

    ax = axes[1]
    ax.scatter(be_cv['test_errors'][:n], sc_cv['test_errors'][:n], alpha=0.4, s=20, c='grey')
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
    ax.set_xlabel('BE test error'); ax.set_ylabel('SC test error')
    ax.set_title(f'BE={comparison["be_mean"]:.5f}, SC={comparison["sc_mean"]:.5f}')
    ax.set_aspect('equal')

    fig.suptitle(f'{animal_id} — CV {title_suffix}', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()


# =============================================================================
# FULL PER-ANIMAL PIPELINE
# =============================================================================

def run_animal_pipeline(
    animal: AnimalData,
    fitting_data: FittingData,
    be_snpe: Dict, sc_snpe: Dict,
    n_cv_repeats: int = 64,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline for one animal using pre-trained amortised SNPE.
    Used by Parts 1 and 3.
    """
    aid = animal.animal_id
    if verbose:
        print(f"\n  {aid}: {fitting_data.n_sessions} sessions, "
              f"{fitting_data.trials_per_session.sum()} trials")

    be_cond = condition_on_animal(be_snpe, fitting_data)
    sc_cond = condition_on_animal(sc_snpe, fitting_data)

    if verbose:
        print(f"    BE: {_fmt_params(be_cond['median_params'])}")
        print(f"    SC: {_fmt_params(sc_cond['median_params'])}")

    be_cv = cv_um_comparison(be_snpe, fitting_data, n_repeats=n_cv_repeats, seed=seed)
    sc_cv = cv_um_comparison(sc_snpe, fitting_data, n_repeats=n_cv_repeats, seed=seed)
    comp = compare_models(be_cv, sc_cv)

    if verbose:
        print(f"    CV: BE={comp['be_mean']:.5f} SC={comp['sc_mean']:.5f} "
              f"p={comp['p_value']:.3g} → {comp['winner']}")

    return {
        'animal_id': aid, 'n_sessions': fitting_data.n_sessions,
        'n_trials': int(fitting_data.trials_per_session.sum()),
        'be_params': be_cond['median_params'],
        'sc_params': sc_cond['median_params'],
        'winner': comp['winner'], 'p': comp['p_value'],
        'be_mean': comp['be_mean'], 'sc_mean': comp['sc_mean'],
        'be_cv': be_cv, 'sc_cv': sc_cv,
    }


def run_animal_pipeline_part2(
    animal: AnimalData,
    fitting_data: FittingData,
    stat_names_with_um: List[str],
    n_sbi_sims: int = 10_000,
    n_cv_repeats: int = 64,
    burn_in: int = 1000,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline for one animal using per-animal SNPE with UM in stats.
    Used by Part 2 (manuscript approach).
    """
    aid = animal.animal_id
    if verbose:
        print(f"\n{'='*50}")
        print(f"  {aid}: {fitting_data.n_sessions} sessions, "
              f"{fitting_data.trials_per_session.sum()} trials")

    be_snpe = train_per_animal_snpe(
        'be', fitting_data, stat_names_with_um, n_sbi_sims, burn_in, seed)
    sc_snpe = train_per_animal_snpe(
        'sc', fitting_data, stat_names_with_um, n_sbi_sims, burn_in, seed + 1)

    be_cond = condition_on_animal(be_snpe, fitting_data)
    sc_cond = condition_on_animal(sc_snpe, fitting_data)

    be_cv = cv_um_comparison(be_snpe, fitting_data, n_repeats=n_cv_repeats, seed=seed)
    sc_cv = cv_um_comparison(sc_snpe, fitting_data, n_repeats=n_cv_repeats, seed=seed)
    comp = compare_models(be_cv, sc_cv)

    if verbose:
        print(f"    CV: BE={comp['be_mean']:.5f} SC={comp['sc_mean']:.5f} "
              f"p={comp['p_value']:.3g} → {comp['winner']}")

    return {
        'animal_id': aid, 'n_sessions': fitting_data.n_sessions,
        'n_trials': int(fitting_data.trials_per_session.sum()),
        'be_params': be_cond['median_params'],
        'sc_params': sc_cond['median_params'],
        'winner': comp['winner'], 'p': comp['p_value'],
        'be_mean': comp['be_mean'], 'sc_mean': comp['sc_mean'],
        'be_cv': be_cv, 'sc_cv': sc_cv,
    }


def _fmt_params(d: Dict) -> str:
    return ', '.join(f'{k}={v:.3f}' for k, v in d.items())


# =============================================================================
# SESSION-BY-SESSION VISUALIZATION
# =============================================================================

def simulate_all_sessions(
    animal: AnimalData,
    be_params: Dict[str, float],
    sc_params: Dict[str, float],
    stage: str = 'Full_Task_Cont',
    distribution: str = 'Uniform',
    min_accuracy: float = 0.0,
    last_fraction: float = 1.0,
    burn_in: int = 1000,
    n_reps: int = 20,
    n_bins: int = 8,
    min_valid_trials: int = 30,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Simulate BE and SC on every qualifying session.

    Returns list of dicts, one per session, each containing:
        'stimuli', 'categories', 'choices' (real),
        'session_id', 'session_idx', 'accuracy',
        'real_um', 'be_um', 'sc_um',
        'real_psych', 'be_psych', 'sc_psych',
        'be_pB', 'sc_pB'
    """
    from Models.BE_core import BEParams, BEModel
    from Models.SC_core import SCParams, SCModel

    # Get sessions
    try:
        fd = select_expert_sessions(
            animal, stage, distribution, min_accuracy, last_fraction,
            min_valid_trials,
        )
    except ValueError:
        fd = animal.get_fitting_data(
            stage=stage, distribution=distribution,
            min_valid_trials=min_valid_trials,
        )

    be_p = BEParams(**be_params)
    sc_p = SCParams(**sc_params)

    results = []

    for i in range(fd.n_sessions):
        v = ~fd.no_response[i]
        stim = fd.stimuli[i][v]
        cat = fd.categories[i][v]
        ch = fd.choices[i][v]

        if len(stim) < min_valid_trials:
            continue

        acc = float(np.mean(ch == cat))

        # Real UM and psychometric
        real_um, _, _ = compute_update_matrix(stim, ch, cat, n_bins=n_bins)
        real_psych = fit_psychometric(stim, ch)

        # BE: p(B) + simulated UMs and psychometrics
        be_state = BEModel.create_initial_state(params=be_p, burn_in=burn_in, seed=seed)
        _, be_pB, _, _ = BEModel.simulate_session(
            be_p, be_state, stim, cat,
            np.random.default_rng(seed), return_history=False,
        )

        be_ums, be_psychs = [], []
        for r in range(n_reps):
            rng_r = np.random.default_rng(seed + r + 1)
            s_be = BEModel.create_initial_state(params=be_p, burn_in=burn_in, seed=seed)
            c_be, _, _, _ = BEModel.simulate_session(
                be_p, s_be, stim, cat, rng_r, return_history=False)
            vv = ~np.isnan(c_be)
            if vv.sum() > 50:
                um, _, _ = compute_update_matrix(stim[vv], c_be[vv], cat[vv], n_bins)
                be_ums.append(um)
            be_psychs.append(c_be)

        be_mean_um = np.nanmean(be_ums, axis=0) if be_ums else np.full((n_bins, n_bins), np.nan)
        be_mean_choices = np.nanmean(be_psychs, axis=0)
        be_psych = fit_psychometric(stim, be_mean_choices)

        # SC: same
        sc_state = SCModel.create_initial_state(params=sc_p, burn_in=burn_in, seed=seed)
        _, sc_pB, _, _ = SCModel.simulate_session(
            sc_p, sc_state, stim, cat,
            np.random.default_rng(seed), return_history=False,
        )

        sc_ums, sc_psychs = [], []
        for r in range(n_reps):
            rng_r = np.random.default_rng(seed + r + 1)
            s_sc = SCModel.create_initial_state(params=sc_p, burn_in=burn_in, seed=seed)
            c_sc, _, _, _ = SCModel.simulate_session(
                sc_p, s_sc, stim, cat, rng_r, return_history=False)
            vv = ~np.isnan(c_sc)
            if vv.sum() > 50:
                um, _, _ = compute_update_matrix(stim[vv], c_sc[vv], cat[vv], n_bins)
                sc_ums.append(um)
            sc_psychs.append(c_sc)

        sc_mean_um = np.nanmean(sc_ums, axis=0) if sc_ums else np.full((n_bins, n_bins), np.nan)
        sc_mean_choices = np.nanmean(sc_psychs, axis=0)
        sc_psych = fit_psychometric(stim, sc_mean_choices)

        results.append({
            'stimuli': stim, 'categories': cat, 'choices': ch,
            'session_id': fd.session_ids[i],
            'session_idx': int(fd.session_indices[i]),
            'accuracy': acc,
            'n_trials': len(stim),
            'real_um': real_um, 'be_um': be_mean_um, 'sc_um': sc_mean_um,
            'real_psych': real_psych, 'be_psych': be_psych, 'sc_psych': sc_psych,
            'be_um_mse': matrix_error(be_mean_um, real_um),
            'sc_um_mse': matrix_error(sc_mean_um, real_um),
        })

    return results


def plot_session_by_session_um(
    session_data: List[Dict],
    animal_id: str,
    be_colour: str = 'steelblue',
    sc_colour: str = 'darkorange',
    max_sessions: int = 20,
    figscale: float = 2.5,
):
    """
    Grid of update matrices: rows = sessions, columns = Real | BE | SC.

    Also shows per-session UM MSE in the title of each model panel.
    """
    import matplotlib.pyplot as plt
    from behav_utils.plotting.update_matrix import plot_update_matrix as _plot_um

    data = session_data[:max_sessions]
    n_sess = len(data)

    if n_sess == 0:
        print("No sessions to plot.")
        return

    # Shared colour scale across all panels
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
                ax.set_title(['Real', 'BE', 'SC'][col], fontsize=11, fontweight='bold')

            # Row label on the left
            if col == 0:
                ax.set_ylabel(f'S{sid}', fontsize=9, fontweight='bold')
            else:
                ax.set_ylabel('')

            # Small MSE text
            if col > 0:
                mse_val = d['be_um_mse'] if col == 1 else d['sc_um_mse']
                if not np.isnan(mse_val):
                    ax.text(0.02, 0.98, f'{mse_val:.4f}', transform=ax.transAxes,
                            fontsize=7, va='top', ha='left',
                            color=be_colour if col == 1 else sc_colour)

            # Remove tick labels for interior panels
            if row < n_sess - 1:
                ax.set_xlabel('')
            ax.tick_params(labelsize=6)

    fig.suptitle(f'{animal_id} — Session-by-session Update Matrices',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_session_by_session_psychometric(
    session_data: List[Dict],
    animal_id: str,
    be_colour: str = 'steelblue',
    sc_colour: str = 'darkorange',
    max_sessions: int = 20,
    n_cols: int = 4,
    figscale: float = 3.0,
):
    """
    Grid of psychometric curves: Real (black) + BE + SC overlaid.

    Each panel is one session.
    """
    import matplotlib.pyplot as plt

    data = session_data[:max_sessions]
    n_sess = len(data)

    if n_sess == 0:
        print("No sessions to plot.")
        return

    n_rows = int(np.ceil(n_sess / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(figscale * n_cols, figscale * n_rows),
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
        if p['success']:
            y = cumulative_gaussian(x_fine, p['mu'], p['sigma'],
                                    p['lapse_low'], p['lapse_high'])
            ax.plot(x_fine, y, 'k-', lw=2, label='Real')

        # BE
        p = d['be_psych']
        if p['success']:
            y = cumulative_gaussian(x_fine, p['mu'], p['sigma'],
                                    p['lapse_low'], p['lapse_high'])
            ax.plot(x_fine, y, '-', color=be_colour, lw=1.5, label='BE')

        # SC
        p = d['sc_psych']
        if p['success']:
            y = cumulative_gaussian(x_fine, p['mu'], p['sigma'],
                                    p['lapse_low'], p['lapse_high'])
            ax.plot(x_fine, y, '-', color=sc_colour, lw=1.5, label='SC')

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

    # Hide unused axes
    for idx in range(n_sess, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(f'{animal_id} — Session-by-session Psychometric Curves',
                 fontsize=14, fontweight='bold')
    fig.text(0.5, 0.01, 'Stimulus', ha='center', fontsize=11)
    fig.text(0.01, 0.5, 'P(choose B)', va='center', rotation='vertical', fontsize=11)
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.97])
    plt.show()


def plot_pooled_um_comparison(
    session_data: List[Dict],
    animal_id: str,
    be_colour: str = 'steelblue',
    sc_colour: str = 'darkorange',
    n_bins: int = 8,
):
    """
    Pooled update matrix comparison across all sessions:
    Real | BE | SC, each averaged across sessions.
    """
    import matplotlib.pyplot as plt
    from behav_utils.plotting.update_matrix import plot_update_matrix as _plot_um

    real_ums = [d['real_um'] for d in session_data if not np.all(np.isnan(d['real_um']))]
    be_ums = [d['be_um'] for d in session_data if not np.all(np.isnan(d['be_um']))]
    sc_ums = [d['sc_um'] for d in session_data if not np.all(np.isnan(d['sc_um']))]

    real_mean = np.nanmean(real_ums, axis=0) if real_ums else None
    be_mean = np.nanmean(be_ums, axis=0) if be_ums else None
    sc_mean = np.nanmean(sc_ums, axis=0) if sc_ums else None

    if real_mean is None:
        print("No valid UMs to plot.")
        return

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

    fig.suptitle(f'{animal_id} — Pooled UM Comparison',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return {'real_mean_um': real_mean, 'be_mean_um': be_mean, 'sc_mean_um': sc_mean,
            'be_mse': be_mse, 'sc_mse': sc_mse}
