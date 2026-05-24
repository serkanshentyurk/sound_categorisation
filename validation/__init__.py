"""validation — Synthetic-data testing of the analysis pipeline.

Generators produce synthetic cohorts with known ground truth, then
analysis pipelines are run on them to verify recovery.
"""
from validation.cohorts import (
    generate_session_with_distribution,
    make_synthetic_cohort,
    make_learning_cohort,
)
from validation.sbi import (
    compute_sbc_ranks,
    compute_parameter_recovery,
    compute_param_stat_correlations,
    recovery_summary_table,
)

from validation.model_id import run_gs_model_id