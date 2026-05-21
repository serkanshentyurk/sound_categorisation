"""
analysis — Project-specific analysis modules

Naming convention: all analysis functions use compute_* prefix,
matching behav_utils. Utilities (detection, classification, I/O)
keep descriptive names.

Modules:
    consensus           — BE/SC model assignment consensus
    opto                — Optogenetic effect analysis
    adaptation          — Post-shift adaptation characterisation
    grid_search         — Grid-search cross-validation
    cv_utils            — CV helper utilities
    fold_utils          — Fold splitting for CV
    stimulus_distribution — Hard-A/B distributions
    validation          — Synthetic data generation
"""

# ── Consensus ────────────────────────────────────────────────────────────────
from analysis.consensus import (
    load_all_assignments,
    compute_consensus_summary,
)

# ── Opto ─────────────────────────────────────────────────────────────────────
from analysis.opto import (
    OptoPhase,
    assign_opto_phases,
    compute_within_session_effect,
    compute_phase_comparison,
    compute_opto_um,
    compute_phase_stability,
    compute_genotype_interaction,
    compute_opto_by_assignment,
    compute_expert_null_test,
    compute_expert_um_test,
    compute_phase_interaction,
    compute_opto_psychometric,
    simulate_with_opto,
    split_opto_session,
    opto_relative_mask,
    split_trials_by_opto,
    get_post_opto_mask,
)

# ── Adaptation ───────────────────────────────────────────────────────────────
from analysis.adaptation import (
    detect_all_manipulations,
    detect_first_manipulation,
    compute_adaptation_trajectory,
    compute_recovery_curve,
    compute_phase_comparison as compute_adaptation_phase_comparison,
    compute_shift_magnitude,
    compute_convergence_metrics,
    classify_sessions,
    compute_group_trajectories,
    classify_shift_type,
    group_shifts_by_type,
    build_phase_blocks,
)

# ── Grid Search ──────────────────────────────────────────────────────────────
from analysis.grid_search import (
    ParameterGrid,
    compute_grid_search_cv,
    compute_grid_search_fit,
    compute_parameter_sweep,
    simulate_model_matrices,
    sessions_to_arrays,
    compute_sessions_blocked,
    compute_sessions_individual,
    compute_static_vs_dynamic,
)

# ── CV Utilities ─────────────────────────────────────────────────────────────
from analysis.cv_utils import (
    compute_empirical_um,
    simulate_model_um,
    sessions_to_old_df,
    compute_gs_seed_errors,
    compute_cv_dataframes,
    params_to_str,
)

# ── Validation ───────────────────────────────────────────────────────────────
from analysis.validation import (
    make_synthetic_cohort,
)

# Note: compute_phase_comparison exists in both opto and adaptation.
# Import as compute_adaptation_phase_comparison to avoid collision.
# In notebooks, import directly from the module:
#   from analysis.adaptation import compute_phase_comparison
#   from analysis.opto import compute_phase_comparison as compute_opto_phase_comparison
