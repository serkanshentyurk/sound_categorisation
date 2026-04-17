"""
Validation Utilities

Shared helpers for the 2-series validation notebooks.
Kept separate from behav_utils because these depend on
project-specific models (BE/SC) and stimulus distributions.
"""

import numpy as np
import pandas as pd
import warnings
from typing import List, Dict, Optional, Tuple
from datetime import date, timedelta


# =============================================================================
# SESSION GENERATION WITH PROJECT-SPECIFIC DISTRIBUTIONS
# =============================================================================

def generate_session_with_distribution(
    session_idx, n_trials, distribution, animal_id, simulator,
    stage='Full_Task_Cont', rng=None, abort_rate=0.05,
    simulator_kwargs=None,
):
    """
    Generate a synthetic session supporting 'hard_a'/'hard_b' distributions.

    Unlike behav_utils.generate_synthetic_session, this uses
    analysis.stimulus_distribution.sample_distribution for project-specific
    distributions.
    """
    from analysis.stimulus_distribution import sample_distribution
    from behav_utils.data.synthetic import sample_stimuli
    from behav_utils.data.structures import SessionData, SessionMetadata, TrialData

    if rng is None:
        rng = np.random.default_rng()
    if simulator_kwargs is None:
        simulator_kwargs = {}

    if distribution in ('hard_a', 'hard_b'):
        stimuli, categories = sample_distribution(n_trials, distribution, rng=rng)
    else:
        stimuli, categories = sample_stimuli(n_trials, distribution=distribution, rng=rng)

    abort = rng.random(n_trials) < abort_rate
    choices = simulator(stimuli, categories, rng, **simulator_kwargs)
    choices[abort] = np.nan
    correct = (choices == categories)
    correct[np.isnan(choices)] = False
    outcome = np.where(abort, 'Abort', np.where(correct, 'Correct', 'Incorrect'))
    rt = np.abs(rng.normal(300, 100, n_trials))
    rt[abort] = np.nan

    trials = TrialData(
        trial_number=np.arange(1, n_trials + 1),
        stimulus=stimuli, category=categories, choice=choices,
        choice_raw=choices.copy(),
        correct=correct, outcome=outcome,
        reaction_time=rt, abort=abort,
        opto_on=np.zeros(n_trials, dtype=bool),
    )
    session_date = date(2025, 1, 1) + timedelta(days=session_idx)
    metadata = SessionMetadata(fields={
        'animal_id': animal_id,
        'stage': stage,
        'protocol': 'Synthetic',
        'sound_contingency': 'Low_Left_High_Right',
        'stim_range_min': -1.0,
        'stim_range_max': 1.0,
    })
    session_id = f'{animal_id}_S{session_idx:03d}'
    return SessionData(
        session_id=session_id,
        session_idx=session_idx,
        date=session_date,
        metadata=metadata,
        trials=trials,
    )


# =============================================================================
# COHORT GENERATORS
# =============================================================================

def _make_simulator(model_type, params, burn_in, seed):
    """Create a simulator for the given model and params."""
    from models.BE_core import BEModel
    from models.SC_core import SCModel
    if model_type == 'BE':
        return BEModel.make_simulator(params, burn_in=burn_in, seed=seed)
    else:
        return SCModel.make_simulator(params, burn_in=burn_in, seed=seed)


def _sample_params(model_type, rng):
    """Sample params from prior for the given model."""
    from models.BE_core import BEParams
    from models.SC_core import SCParams
    if model_type == 'BE':
        return BEParams.sample_prior(rng)
    else:
        return SCParams.sample_prior(rng)


def make_synthetic_cohort(
    n_per_model=5, n_sessions=15, trials_per_session=350,
    burn_in=1000, seed=42, stage='Full_Task_Cont',
):
    """
    Static expert parameters, uniform distribution.
    Returns list of dicts: animal_id, true_model, true_params, animal, sessions.
    """
    from behav_utils.data.synthetic import generate_synthetic_animal

    animals = []
    for model_type, base_seed in [('BE', seed), ('SC', seed + 5000)]:
        rng = np.random.default_rng(base_seed)
        for i in range(n_per_model):
            seed_i = base_seed + i * 100
            params = _sample_params(model_type, rng)
            sim = _make_simulator(model_type, params, burn_in, seed_i)

            aid = f'{model_type}_static_{i:02d}'
            animal, _ = generate_synthetic_animal(
                animal_id=aid, n_sessions=n_sessions,
                trials_per_session=trials_per_session,
                seed=seed_i, simulator=sim, stage=stage,
            )
            animals.append({
                'animal_id': aid, 'true_model': model_type,
                'true_params': params, 'animal': animal,
                'sessions': animal.get_sessions(stage=stage),
            })
    return animals


def make_learning_cohort(
    n_per_model=5, n_sessions=20, trials_per_session=350,
    burn_in=1000, seed=42, stage='Full_Task_Cont',
):
    """
    Dynamic learning trajectory on uniform stimuli.

    BE: eta_learning follows ~0.02 -> peak -> expert_value (hump-shaped)
    SC: gamma follows 0.5 -> expert_value (monotonic increase)
    """
    from models.BE_core import BEParams
    from models.SC_core import SCParams
    from behav_utils.data.structures import AnimalData

    animals = []
    for model_type, base_seed in [('BE', seed), ('SC', seed + 5000)]:
        rng = np.random.default_rng(base_seed)
        for i in range(n_per_model):
            seed_i = base_seed + i * 100
            sess_rng = np.random.default_rng(seed_i)
            expert_params = _sample_params(model_type, rng)

            sessions = []
            for s in range(n_sessions):
                frac = s / max(n_sessions - 1, 1)

                if model_type == 'BE':
                    peak_eta = min(expert_params.eta_learning * 2.5, 0.9)
                    if frac < 0.4:
                        eta = 0.02 + (peak_eta - 0.02) * (frac / 0.4)
                    else:
                        eta = peak_eta + (expert_params.eta_learning - peak_eta) * ((frac - 0.4) / 0.6)
                    sess_params = BEParams(
                        sigma_percep=expert_params.sigma_percep,
                        A_repulsion=expert_params.A_repulsion,
                        eta_learning=eta,
                        eta_relax=expert_params.eta_relax,
                    )
                else:
                    gamma = 0.5 + (expert_params.gamma - 0.5) * frac
                    gamma = min(gamma, 1.0)
                    sess_params = SCParams(
                        sigma_percep=expert_params.sigma_percep,
                        A_repulsion=expert_params.A_repulsion,
                        gamma=gamma,
                        sigma_update=expert_params.sigma_update,
                    )

                sim = _make_simulator(model_type, sess_params, burn_in, seed_i + s)
                sess = generate_session_with_distribution(
                    session_idx=s, n_trials=trials_per_session,
                    distribution='uniform',
                    animal_id=f'{model_type}_learn_{i:02d}',
                    simulator=sim, stage=stage, rng=sess_rng,
                )
                sessions.append(sess)

            aid = f'{model_type}_learn_{i:02d}'
            animal = AnimalData(animal_id=aid, sessions=sessions)
            animals.append({
                'animal_id': aid, 'true_model': model_type,
                'true_params': expert_params, 'animal': animal,
                'sessions': sessions,
            })
    return animals


def make_shift_cohort(
    n_per_model=5, n_uniform=10, n_hard_a=10,
    trials_per_session=350, burn_in=1000, dynamic_hard_a=False,
    seed=42, stage='Full_Task_Cont',
):
    """
    Uniform expert sessions -> Hard-A sessions.

    If dynamic_hard_a: eta/gamma spikes then decays during Hard-A
    (mimicking re-learning). Otherwise static expert params throughout.

    Returns list of dicts with extra keys: uniform_sessions, hard_a_sessions.
    """
    from models.BE_core import BEParams
    from models.SC_core import SCParams
    from behav_utils.data.structures import AnimalData

    animals = []
    for model_type, base_seed in [('BE', seed), ('SC', seed + 5000)]:
        rng = np.random.default_rng(base_seed)
        for i in range(n_per_model):
            seed_i = base_seed + i * 100
            sess_rng = np.random.default_rng(seed_i)
            expert_params = _sample_params(model_type, rng)

            sessions = []
            n_total = n_uniform + n_hard_a
            for s in range(n_total):
                is_hard_a = s >= n_uniform
                dist = 'hard_a' if is_hard_a else 'uniform'

                if is_hard_a and dynamic_hard_a:
                    shift_frac = (s - n_uniform) / max(n_hard_a - 1, 1)
                    if model_type == 'BE':
                        spike_eta = min(expert_params.eta_learning * 2.5, 0.9)
                        eta = spike_eta + (expert_params.eta_learning - spike_eta) * shift_frac
                        sess_params = BEParams(
                            sigma_percep=expert_params.sigma_percep,
                            A_repulsion=expert_params.A_repulsion,
                            eta_learning=eta,
                            eta_relax=expert_params.eta_relax,
                        )
                    else:
                        drop_gamma = max(expert_params.gamma * 0.6, 0.3)
                        gamma = drop_gamma + (expert_params.gamma - drop_gamma) * shift_frac
                        sess_params = SCParams(
                            sigma_percep=expert_params.sigma_percep,
                            A_repulsion=expert_params.A_repulsion,
                            gamma=gamma,
                            sigma_update=expert_params.sigma_update,
                        )
                    sim = _make_simulator(model_type, sess_params, burn_in, seed_i + s)
                else:
                    sim = _make_simulator(model_type, expert_params, burn_in, seed_i + s)

                tag = 'dshift' if dynamic_hard_a else 'shift'
                aid = f'{model_type}_{tag}_{i:02d}'
                sess = generate_session_with_distribution(
                    session_idx=s, n_trials=trials_per_session,
                    distribution=dist, animal_id=aid,
                    simulator=sim, stage=stage, rng=sess_rng,
                )
                sessions.append(sess)

            aid = f'{model_type}_{tag}_{i:02d}'
            animal = AnimalData(animal_id=aid, sessions=sessions)
            animals.append({
                'animal_id': aid, 'true_model': model_type,
                'true_params': expert_params, 'animal': animal,
                'sessions': sessions,
                'uniform_sessions': sessions[:n_uniform],
                'hard_a_sessions': sessions[n_uniform:],
            })
    return animals


# =============================================================================
# MODEL IDENTIFICATION RUNNERS
# =============================================================================

def run_gs_model_id(
    animals, sessions_key='sessions', grid=None, n_seeds=2, burn_in=1000,
    fit_target='update_matrix',
):
    """
    Run grid-search model identification on synthetic animals.

    Args:
        fit_target: 'update_matrix' or 'conditional_psych'.

    Returns DataFrame with gs_winner, gs_correct, gs_be_mean, gs_sc_mean,
    gs_recovered_params, fit_target.
    """
    from analysis.grid_search import grid_search_cv, COARSE_GRID
    if grid is None:
        grid = COARSE_GRID

    rows = []
    for sa in animals:
        aid = sa['animal_id']
        sessions = sa[sessions_key]
        print(f'  GS [{fit_target}] {aid} [{sa["true_model"]}]...', end=' ')

        errors = {'BE': [], 'SC': []}
        errors_detail = {'BE': [], 'SC': []}
        for seed in range(1, n_seeds + 1):
            for mt in ['BE', 'SC']:
                try:
                    r = grid_search_cv(
                        sessions, mt, grid=grid[mt],
                        n_folds=2, seed=seed, burn_in=burn_in,
                        fit_target=fit_target,
                    )
                    errors[mt].append(r['avg_test_error'])
                    errors_detail[mt].append(r)
                except Exception:
                    pass

        be_mean = np.mean(errors['BE']) if errors['BE'] else np.nan
        sc_mean = np.mean(errors['SC']) if errors['SC'] else np.nan
        winner = 'BE' if be_mean < sc_mean else 'SC'
        correct = winner == sa['true_model']
        # Store recovered params from winning model's best seed
        recovered = {}
        winner_results = errors_detail.get(winner, [])
        if winner_results:
            best_seed_idx = int(np.argmin([r['avg_test_error'] for r in winner_results]))
            recovered = winner_results[best_seed_idx].get('best_params_single', {})

        rows.append({
            'animal_id': aid, 'true_model': sa['true_model'],
            'gs_winner': winner, 'gs_correct': correct,
            'gs_be_mean': be_mean, 'gs_sc_mean': sc_mean,
            'gs_recovered_params': recovered,
            'fit_target': fit_target,
        })
        print(f'{winner} {"✓" if correct else "✗"}')

    return pd.DataFrame(rows)


def run_sbi_model_id(
    animals, sessions_key='sessions', stat_names=None,
    n_sbi_sims=1000, n_generic_trials=300, n_cv_repeats=4,
    burn_in=1000, seed=42,
    return_networks=False,
    method='update_matrix',
):
    """
    Run SBI CV model identification on synthetic animals.

    Args:
        method: 'update_matrix' or 'conditional_psych' — which matrix
                to score BE vs SC against during CV.

    Returns:
        DataFrame (or (DataFrame, be_snpe, sc_snpe) if return_networks=True)
    """
    from inference.comparison import train_amortised_snpe, run_animal_pipeline
    from behav_utils.data.selection import fitting_data_from_sessions

    if stat_names is None:
        stat_names = [
            'accuracy', 'psychometric', 'recency', 'stimulus_recency',
            'win_stay', 'lose_shift', 'side_bias', 'stimulus_sensitivity',
            'choice_entropy', 'perseveration',
        ]

    print('  Training SNPE...')
    be_snpe = train_amortised_snpe(
        'be', stat_names, n_sbi_sims, n_generic_trials, burn_in, seed)
    sc_snpe = train_amortised_snpe(
        'sc', stat_names, n_sbi_sims, n_generic_trials, burn_in, seed + 1)

    rows = []
    for sa in animals:
        aid = sa['animal_id']
        sessions = sa[sessions_key]
        print(f'  SBI [{method}] {aid} [{sa["true_model"]}]...', end=' ')
        try:
            fd = fitting_data_from_sessions(sessions, aid)
            r = run_animal_pipeline(
                fd, be_snpe, sc_snpe,
                n_cv_repeats=n_cv_repeats, seed=seed, verbose=False,
                method=method,
            )
            correct = r['winner'] == sa['true_model']
            rows.append({
                'animal_id': aid, 'true_model': sa['true_model'],
                'sbi_winner': r['winner'], 'sbi_correct': correct,
                'sbi_be_mean': r['be_mean'], 'sbi_sc_mean': r['sc_mean'],
                'sbi_p': r['p'],
                'method': method,
            })
            print(f'{r["winner"]} {"✓" if correct else "✗"}')
        except Exception as e:
            print(f'FAILED ({e})')
            rows.append({
                'animal_id': aid, 'true_model': sa['true_model'],
                'sbi_winner': '?', 'sbi_correct': False,
                'sbi_be_mean': np.nan, 'sbi_sc_mean': np.nan,
                'sbi_p': np.nan,
                'method': method,
            })
    df = pd.DataFrame(rows)
    if return_networks:
        return df, be_snpe, sc_snpe
    return df


def summarise_agreement(gs_df, sbi_df, label=''):
    """Merge GS and SBI results, report accuracy and agreement."""
    merged = gs_df.merge(sbi_df, on=['animal_id', 'true_model'], how='outer')

    print(f'\n{"=" * 60}')
    if label:
        print(f'  {label}')
        print(f'{"=" * 60}')

    for method, col in [('GS', 'gs_correct'), ('SBI', 'sbi_correct')]:
        if col in merged.columns:
            valid = merged[col].dropna()
            print(f'  {method} accuracy: {valid.sum():.0f}/{len(valid)} ({valid.mean():.0%})')

    if 'gs_winner' in merged.columns and 'sbi_winner' in merged.columns:
        agree = (merged['gs_winner'] == merged['sbi_winner']).mean()
        print(f'  GS-SBI agreement: {agree:.0%}')
        if 'gs_correct' in merged.columns and 'sbi_correct' in merged.columns:
            both = (merged['gs_correct'] & merged['sbi_correct']).mean()
            print(f'  Both correct:     {both:.0%}')

    return merged
