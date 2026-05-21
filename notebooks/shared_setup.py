"""
Shared Notebook Setup

Usage:
    from shared_setup import *
    experiment, info = load_data()
"""

import os
import sys
import platform
from typing import Optional
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Path setup ──────────────────────────────────────────────────────────────
_NOTEBOOK_DIR = Path(os.path.abspath(''))
_PROJECT_ROOT = _NOTEBOOK_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Snapshot and config paths ───────────────────────────────────────────────
#
# Folder structure (local, any OS):
#   some_folder/data/snapshots/sound_cat_snapshot.pkl
#   some_folder/repos/sound_categorisation/   ← _PROJECT_ROOT
#
# Cluster (fixed):
#   /ceph/akrami/Serkan/.../Processed/snapshots/sound_cat_snapshot.pkl
#
from scripts.snapshot import snapshot_dir
PATH_SNAPSHOT = snapshot_dir(_PROJECT_ROOT) / 'sound_cat_snapshot.pkl'
PATH_CONFIG = _PROJECT_ROOT / 'config.yaml'

STAGE = 'Full_Task_Cont'
MIN_SESSIONS = 5

# ── Results paths (single source of truth for all notebooks) ───────────────
RESULTS_DIR = _PROJECT_ROOT / 'results'
SNPE_DIR = RESULTS_DIR / 'snpe'
CV_DIR = RESULTS_DIR / 'cv'
SBI_STATIC_DIR = RESULTS_DIR / 'sbi_static'
SBI_DYNAMIC_DIR = RESULTS_DIR / 'sbi_dynamic'
VALIDATION_DIR = RESULTS_DIR / 'validation'

FIT_TARGETS = ['update_matrix', 'conditional_psych']
FT_LABEL = {'update_matrix': 'UM', 'conditional_psych': 'CP'}

# ── Common imports ──────────────────────────────────────────────────────────
from behav_utils.data.structures import (
    ExperimentData, AnimalData, SessionData, FittingData,
)
from behav_utils.data.loading import load_experiment
from behav_utils.data.selection import select_sessions, SessionFilter
from behav_utils.data.filtering import (
    filter_trials, pool_arrays,
    build_mask, opto_mask,
    filter_session, get_arrays,
)
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
# Low-level analysis (arrays)
from behav_utils.analysis.psychometry import fit_psychometric
from behav_utils.analysis.update_matrix import compute_update_matrix, matrix_error
from behav_utils.analysis.comparison import compare_conditions
from behav_utils.analysis.utils import cumulative_gaussian

# Session-level analysis (sessions → result dicts)
from behav_utils.analysis.psychometry import compute_psychometric
from behav_utils.analysis.update_matrix import compute_um
from behav_utils.analysis.trajectory import compute_trajectory
from behav_utils.analysis.comparison import compute_comparison
from behav_utils.analysis.session_raster import compute_session_raster

# Plotting (result dicts → axes)
from behav_utils.plotting import (
    plot_psychometric, plot_um, plot_trajectory,
    plot_comparison, plot_session_raster,
    PALETTE, COLOURS, UM_CMAP,
    apply_style, get_colour,
)

# ── Results loading helpers ─────────────────────────────────────────────────

def load_snpe_networks(snpe_dir: Optional[Path] = None, distribution: str = 'uniform') -> dict:
    """
    Load trained SNPE pickles for BE and SC.

    Returns dict keyed by model type ('be', 'sc'), each containing
    the trained posterior + metadata.
    """
    import pickle

    snpe_dir = snpe_dir or SNPE_DIR
    snpe = {}
    for model in ['be', 'sc']:
        p = snpe_dir / f'{distribution}_{model}.pkl'
        if p.exists():
            with open(p, 'rb') as f:
                snpe[model] = pickle.load(f)
            print(f'SNPE {model}: {snpe[model]["param_names"]}')
        else:
            print(f'SNPE {model}: not found at {p}')
    return snpe


# ── Data loading ────────────────────────────────────────────────────────────

def _generate_synthetic_cohort(
    n_animals: int = 5,
    n_sessions: int = 25,
    shift_session: int = 15,
    seed: int = 42,
) -> ExperimentData:
    """Generate a synthetic cohort with a distribution shift for testing."""
    from scipy.stats import norm as sp_norm # type: ignore

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
    config_path: Optional[Path] = None,
    snapshot_path: Optional[Path] = None,
    warn_age_hours: float = 72,
    **synthetic_kwargs,
):
    """
    Load experimental data.

    Priority order for 'auto':
        1. Snapshot — versioned pickle with staleness checks
        2. CSV — full reload from raw data (needs BEHAV_DATA_DIR env var)
        3. Synthetic — always works, for testing notebooks

    Returns:
        (experiment, info_dict)
    """
    config_path = config_path or PATH_CONFIG
    snapshot_path = snapshot_path or PATH_SNAPSHOT

    # 1. Snapshot
    if mode in ('snapshot', 'auto') and snapshot_path.exists():
        try:
            from scripts.snapshot import load_snapshot
            experiment, meta = load_snapshot(
                snapshot_path,
                config_path=config_path if config_path.exists() else None,
                warn_age_hours=warn_age_hours,
            )
            return experiment, {
                'mode': 'snapshot',
                'snapshot_path': str(snapshot_path),
                'metadata': meta,
            }
        except Exception as e:
            if mode == 'snapshot':
                raise
            warnings.warn(f'Snapshot loading failed ({e}), trying CSV...')

    # 2. CSV
    if mode in ('csv', 'auto') and config_path.exists():
        try:
            experiment = load_experiment(config_path)
            n_total = sum(a.n_sessions for a in experiment.animals.values())
            print(
                f'Loaded {experiment.n_animals} animals, '
                f'{n_total} sessions from CSV'
            )
            return experiment, {
                'mode': 'csv',
                'config_path': str(config_path),
            }
        except Exception as e:
            if mode == 'csv':
                raise
            warnings.warn(f'CSV loading failed ({e}), generating synthetic...')

    # 3. Synthetic fallback
    if mode not in ('synthetic', 'auto'):
        raise FileNotFoundError(
            f'Could not load data in mode={mode!r}. '
            f'Check paths: snapshot={snapshot_path}, config={config_path}'
        )

    kw = {'n_animals': 5, 'n_sessions': 25, 'shift_session': 15, 'seed': 42}
    kw.update(synthetic_kwargs)
    experiment = _generate_synthetic_cohort(**kw)
    print(
        f'Generated synthetic cohort: {experiment.n_animals} animals, '
        f'{kw["n_sessions"]} sessions each '
        f'(shift at session {kw["shift_session"]})'
    )
    return experiment, {
        'mode': 'synthetic',
        'shift_session': kw['shift_session'],
        **kw,
    }
