"""
Shared Notebook Setup

Single entry point for all notebooks. Handles:
- Path configuration
- Data loading (real CSV, pickle, or synthetic fallback)
- Common imports re-exported for convenience

Usage:
    from shared_setup import load_data, STAGE, PATH_CONFIG

    experiment, info = load_data()
    # info = {'mode': 'csv'|'pickle'|'synthetic', 'config_path': ..., ...}

    # Or force a specific mode:
    experiment, info = load_data(mode='synthetic', n_animals=5)
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Path setup ──────────────────────────────────────────────────────────────
# Notebooks live in sound_categorisation/notebooks/
# Project root is one level up.
_NOTEBOOK_DIR = Path(os.path.abspath(''))
_PROJECT_ROOT = _NOTEBOOK_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Project constants ───────────────────────────────────────────────────────
# Edit these to match your local setup.
PATH_CONFIG = _PROJECT_ROOT / 'config.yaml'
PATH_PICKLE = _PROJECT_ROOT / 'data' / 'experiment.pkl'
STAGE = 'Full_Task_Cont'
MIN_SESSIONS = 5

# ── Common imports ──────────────────────────────────────────────────────────
# Re-exported so notebooks can do: from shared_setup import ExperimentData, ...
from behav_utils.data.structures import (
    ExperimentData, AnimalData, SessionData, FittingData,
)
from behav_utils.data.loading import load_experiment
from behav_utils.data.selection import select_sessions, SessionFilter
from behav_utils.data.synthetic import (
    generate_synthetic_animal,
    sample_stimuli,
    noisy_psychometric_simulator,
)
from behav_utils.analysis.summary_stats import (
    compute_summary_stats,
    list_available_stats,
    FEATURE_MATRIX_STATS,
    DEFAULT_STATS,
)
from behav_utils.analysis.session_features import (
    build_feature_matrix,
    build_feature_matrix_multi,
)
from behav_utils.analysis.update_matrix import (
    compute_update_matrix,
    compute_update_matrix_from_sessions,
    matrix_error,
)
from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.analysis.utils import cumulative_gaussian

from behav_utils.plotting.psychometric import (
    plot_psychometric,
    plot_session_psychometrics,
    plot_psychometric_overlay,
)
from behav_utils.plotting.trajectory import (
    plot_stat_trajectory,
    plot_multi_animal_trajectory,
    plot_stat_grid,
)
from behav_utils.plotting.update_matrix import (
    plot_update_matrix,
    plot_phase_update_matrices,
    plot_conditional_psychometrics,
)


# ── Data loading ────────────────────────────────────────────────────────────

def _generate_synthetic_cohort(
    n_animals: int = 5,
    n_sessions: int = 25,
    shift_session: int = 15,
    seed: int = 42,
) -> ExperimentData:
    """Generate a synthetic cohort with a distribution shift for testing."""
    from scipy.stats import norm as sp_norm

    rng = np.random.default_rng(seed)
    experiment = ExperimentData(metadata={'cohort': 'synthetic_demo'})

    def _learning_simulator(stimuli, categories, rng, sigma=0.3, lapse=0.05, **kw):
        p_b = lapse + (1 - 2 * lapse) * sp_norm.cdf(stimuli, 0, sigma)
        return (rng.random(len(stimuli)) < p_b).astype(float)

    for a in range(n_animals):
        animal_id = f'SYN{a + 1:02d}'
        animal_seed = int(rng.integers(0, 2**31))
        rate = 0.1 + rng.uniform(-0.03, 0.03)
        sigmas = [max(0.12, 0.6 * np.exp(-rate * s)) for s in range(n_sessions)]
        lapses = [max(0.02, 0.15 * np.exp(-0.12 * s)) for s in range(n_sessions)]
        per_session_kwargs = [
            {'sigma': sigmas[s], 'lapse': lapses[s]} for s in range(n_sessions)
        ]
        dist_schedule = (
            ['uniform'] * shift_session
            + ['exponential_left'] * (n_sessions - shift_session)
        )
        animal, _ = generate_synthetic_animal(
            animal_id=animal_id,
            n_sessions=n_sessions,
            trials_per_session=300,
            seed=animal_seed,
            simulator=_learning_simulator,
            per_session_simulator_kwargs=per_session_kwargs,
            distribution_schedule=dist_schedule,
            stage=STAGE,
        )
        experiment.add_animal(animal)
    return experiment


def load_data(
    mode: str = 'auto',
    config_path: Path = None,
    pickle_path: Path = None,
    **synthetic_kwargs,
):
    """
    Load experimental data.

    Args:
        mode: 'auto' (try CSV then pickle then synthetic),
              'csv', 'pickle', or 'synthetic'
        config_path: Override PATH_CONFIG
        pickle_path: Override PATH_PICKLE
        **synthetic_kwargs: Passed to _generate_synthetic_cohort

    Returns:
        (experiment, info) where info is a dict with loading metadata
    """
    config_path = config_path or PATH_CONFIG
    pickle_path = pickle_path or PATH_PICKLE

    if mode == 'csv' or (mode == 'auto' and config_path.exists()):
        try:
            experiment = load_experiment(config_path)
            n_total = sum(a.n_sessions for a in experiment.animals.values())
            print(
                f"Loaded {experiment.n_animals} animals, "
                f"{n_total} sessions from CSV"
            )
            return experiment, {
                'mode': 'csv',
                'config_path': str(config_path),
            }
        except Exception as e:
            if mode == 'csv':
                raise
            warnings.warn(f"CSV loading failed ({e}), trying pickle...")

    if mode == 'pickle' or (mode == 'auto' and pickle_path.exists()):
        try:
            experiment = ExperimentData.load(pickle_path)
            print(
                f"Loaded {experiment.n_animals} animals from pickle"
            )
            return experiment, {
                'mode': 'pickle',
                'pickle_path': str(pickle_path),
            }
        except Exception as e:
            if mode == 'pickle':
                raise
            warnings.warn(f"Pickle loading failed ({e}), generating synthetic...")

    # Synthetic fallback
    kw = {'n_animals': 5, 'n_sessions': 25, 'shift_session': 15, 'seed': 42}
    kw.update(synthetic_kwargs)
    experiment = _generate_synthetic_cohort(**kw)
    print(
        f"Generated synthetic cohort: {experiment.n_animals} animals, "
        f"{kw['n_sessions']} sessions each "
        f"(shift at session {kw['shift_session']})"
    )
    return experiment, {
        'mode': 'synthetic',
        'shift_session': kw['shift_session'],
        **kw,
    }
