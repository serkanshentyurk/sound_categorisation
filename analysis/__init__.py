"""analysis — Project-specific behavioural and computational analyses."""
from analysis.consensus import compute_consensus_summary, load_all_assignments
from analysis.cv_utils import (
    compute_empirical_um, simulate_model_um, sessions_to_old_df,
    compute_gs_seed_errors, compute_cv_dataframes, params_to_str,
)
from analysis.grid_search import (
    compute_grid_search_cv, compute_sessions_blocked, compute_static_vs_dynamic,
    simulate_model_matrices, ParameterGrid, DEFAULT_GRID, COARSE_GRID,
)
from analysis.validation import (
    generate_session_with_distribution, make_synthetic_cohort,
    make_learning_cohort, run_gs_model_id,
)
from analysis.fold_utils import split_folds_by_block, merge_smallest_adjacent
from analysis.stimulus_distribution import (
    sample_distribution, compute_distribution_density, compute_normative_pse,
)
from analysis.opto import assign_opto_phases
from analysis.adaptation import detect_shifts
from analysis.sbi_validation import (
    compute_sbc_ranks, compute_parameter_recovery, compute_param_stat_correlations,
)