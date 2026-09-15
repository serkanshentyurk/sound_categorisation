"""
behav_utils.analysis — phase-level statistics, contrasts, resampling, group tests.

    phase = filter_trials(select_sessions(animal, preset='expert_uniform'))
    r  = compute_stat(phase, ['accuracy', *PSYCHOMETRIC], per_session=True)
    d  = compute_delta_stat({'non_opto': a, 'opto': b}, ['mu', 'sigma'], reference='non_opto')
    ix = compute_interaction(d_opto, d_mask, 'opto_vs_non_opto')

Scalar statistics themselves live in ``behav_utils.stats``; array-valued
readouts in ``behav_utils.readouts``. The two fit engines on raw arrays
(``fit_psychometric``, ``fit_update_matrix``) are re-exported here.
"""

from behav_utils.analysis.utils import cumulative_gaussian, generate_stimuli
from behav_utils.analysis.psychometry import fit_psychometric, fit_psychometric_gof
from behav_utils.analysis.update_matrix import fit_update_matrix, matrix_error

from behav_utils.analysis.statistics import PhaseStats, compute_stat, infer_animal_id
from behav_utils.analysis.comparison import (
    DeltaStats, PhaseSummary, Contrast, Interaction,
    compute_delta_stat, compute_interaction, contrast_key,
)
from behav_utils.analysis.resampling import (
    bootstrap_phase_stats, permute_phase_difference, summarise_draws, summarise_draw_frame,
    DrawSummary,
)
from behav_utils.analysis.downsample import (
    downsample, calculate_min_n, resample_stat_vectors,
    resample_psychometric_curve, resample_update_matrix,
)
from behav_utils.analysis.rolling import RollingStats, compute_rolling_stats
from behav_utils.analysis.adaptation import (
    compute_normative_pse, resolve_sigma, resolve_sigma_sbi,
    compute_adaptation, compute_adaptation_per_session, detect_shifts,
)
from behav_utils.analysis.across_animals import collect_rows, compare_groups, compare_genotypes
from behav_utils.analysis.group import (
    combine, paired_diff, bootstrap_units, rank_test, average_arrays, min_achievable_p,
)
from behav_utils.analysis.session_raster import compute_session_raster
from behav_utils.analysis.session_features import compute_session_features

__all__ = [
    'cumulative_gaussian', 'generate_stimuli',
    'fit_psychometric', 'fit_psychometric_gof', 'fit_update_matrix', 'matrix_error',
    'PhaseStats', 'compute_stat', 'infer_animal_id',
    'DeltaStats', 'PhaseSummary', 'Contrast', 'Interaction',
    'compute_delta_stat', 'compute_interaction', 'contrast_key',
    'bootstrap_phase_stats', 'permute_phase_difference', 'summarise_draws',
    'summarise_draw_frame', 'DrawSummary',
    'downsample', 'calculate_min_n', 'resample_stat_vectors',
    'resample_psychometric_curve', 'resample_update_matrix',
    'RollingStats', 'compute_rolling_stats',
    'compute_normative_pse', 'resolve_sigma', 'resolve_sigma_sbi',
    'compute_adaptation', 'compute_adaptation_per_session', 'detect_shifts',
    'collect_rows', 'compare_groups', 'compare_genotypes',
    'combine', 'paired_diff', 'bootstrap_units', 'rank_test', 'average_arrays', 'min_achievable_p',
    'compute_session_raster', 'compute_session_features',
]
