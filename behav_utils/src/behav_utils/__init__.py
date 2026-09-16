"""
behav_utils — Behavioural Neuroscience Data Utilities

Config-driven library for loading, filtering, analysing, and
plotting trial-based behavioural data.

Architecture — three levels per domain:

    Low-level:     fit_psychometric(stim, ch)       — raw arrays, any source
    Session-level: compute_psychometric(sessions)   — pre-filtered sessions → result dict
    Plotting:      plot_psychometric(result)         — result dict → axes

Pipeline:
    experiment = load_experiment('config.yaml')
    sessions   = select_sessions(animal, preset='expert_uniform')
    clean      = filter_trials(sessions)

    psych = compute_psychometric(clean, mode='pooled')
    fig, ax = plt.subplots()
    plot_psychometric(psych, ax=ax)

Modules:
    behav_utils.data        — structures, loading, selection, filtering, synthetic
    behav_utils.analysis    — psychometry, update matrix, trajectory, comparison, stats
    behav_utils.plotting    — psychometric, update matrix, trajectory, comparison, session
"""

# ── Config ───────────────────────────────────────────────────────────────────
from behav_utils.analysis.comparison import (
    DeltaStats,
    Interaction,
    compute_delta_stat,
    compute_interaction,
)
from behav_utils.analysis.psychometry import fit_psychometric, fit_psychometric_gof
from behav_utils.analysis.rolling import RollingStats, compute_rolling_stats
from behav_utils.analysis.session_features import compute_session_features
from behav_utils.analysis.session_raster import compute_session_raster
from behav_utils.analysis.statistics import PhaseStats, compute_stat
from behav_utils.analysis.update_matrix import fit_update_matrix, matrix_error
from behav_utils.analysis.utils import cumulative_gaussian, generate_stimuli
from behav_utils.config.schema import ProjectConfig, load_config

# ── Analysis: low-level (arrays) ─────────────────────────────────────────────
from behav_utils.data.arrays import TrialArrays

# ── Loading ──────────────────────────────────────────────────────────────────
from behav_utils.data.loading import load_animal, load_experiment, load_session_csv

# ── Trial filtering ──────────────────────────────────────────────────────────
from behav_utils.data.ops.filtering import (
    build_mask,
    filter_session,
    filter_trial_data,
    filter_trials,
    get_arrays,
    opto_mask,
    pool_arrays,
)

# ── Session selection ────────────────────────────────────────────────────────
from behav_utils.data.ops.selection import (
    SessionFilter,
    list_presets,
    register_preset,
    register_presets_from_config,
    select_sessions,
)

# ── Data structures ──────────────────────────────────────────────────────────
from behav_utils.data.structures import (
    AnimalData,
    ExperimentData,
    SessionData,
    SessionMetadata,
    TrialData,
)

# ── Synthetic data ───────────────────────────────────────────────────────────
from behav_utils.data.synthetic import (
    generate_synthetic_animal,
    generate_synthetic_session,
    noisy_psychometric_simulator,
    sample_stimuli,
)
from behav_utils.plotting import (
    COLOURS,
    PALETTE,
    UM_CMAP,
    apply_style,
    get_colour,
    plot_comparison,
    plot_interaction,
    plot_psychometric_curve,
    plot_session_raster,
    plot_stat_comparison,
    plot_trajectory,
    plot_update_matrix,
)
from behav_utils.readouts import (
    BinnedCurve,
    ConditionalPsychometric,
    PsychometricCurve,
    SerialDependenceProfile,
    UpdateMatrix,
    compute_binned_curve,
    compute_conditional_psychometric,
    compute_psychometric_curve,
    compute_sd_profile,
    compute_update_matrix,
)
from behav_utils.stats import PSYCHOMETRIC, compute_stats, is_exchangeable, list_stats

__version__ = '0.4.0'

__all__ = [
    # Config
    'load_config', 'ProjectConfig',

    # Structures
    'ExperimentData', 'AnimalData', 'SessionData',
    'SessionMetadata', 'TrialData',

    # Loading
    'load_experiment', 'load_session_csv', 'load_animal',

    # Session selection
    'select_sessions', 'SessionFilter',
    'register_preset', 'list_presets', 'register_presets_from_config',

    # Trial filtering
    'filter_trials', 'pool_arrays',
    'build_mask', 'opto_mask',
    'filter_session', 'filter_trial_data', 'get_arrays',

    # Synthetic
    'generate_synthetic_animal', 'generate_synthetic_session',
    'sample_stimuli', 'noisy_psychometric_simulator',


    # Arrays, stats, readouts
    'TrialArrays', 'compute_stats', 'list_stats', 'is_exchangeable', 'PSYCHOMETRIC',
    'compute_psychometric_curve', 'compute_update_matrix', 'compute_conditional_psychometric',
    'compute_binned_curve', 'compute_sd_profile',
    'PsychometricCurve', 'UpdateMatrix', 'ConditionalPsychometric', 'BinnedCurve',
    'SerialDependenceProfile',
    # Analysis
    'fit_psychometric', 'fit_psychometric_gof', 'fit_update_matrix', 'matrix_error',
    'PhaseStats', 'compute_stat',
    'DeltaStats', 'Interaction', 'compute_delta_stat', 'compute_interaction',
    'RollingStats', 'compute_rolling_stats',
    'compute_session_features', 'cumulative_gaussian', 'generate_stimuli',
    'compute_session_raster',
    # Plotting
    'plot_psychometric_curve', 'plot_update_matrix', 'plot_trajectory',
    'plot_comparison', 'plot_stat_comparison', 'plot_interaction', 'plot_session_raster',
    'PALETTE', 'COLOURS', 'UM_CMAP', 'apply_style', 'get_colour',
]
