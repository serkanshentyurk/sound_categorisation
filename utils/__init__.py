"""utils — Math primitives and CV helpers shared across analyses.

Lower-level than `analysis/`. Nothing here is project-specific behaviour.
"""
from utils.stimulus_distributions import (
    sample_distribution,
    compute_distribution_density,
    compute_normative_pse,
)
from utils.cv_utils import (
    compute_seed_errors, compute_cv_dataframes, compare_models, params_to_str,
    save_cv_result, load_cv_results, CVResults,
)
from utils.fold_utils import split_folds_by_block, merge_smallest_adjacent