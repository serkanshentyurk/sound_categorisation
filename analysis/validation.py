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
    from analysis.grid_search import compute_grid_search_cv, COARSE_GRID
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
                    r = compute_grid_search_cv(
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
