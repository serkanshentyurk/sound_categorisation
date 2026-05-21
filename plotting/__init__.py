"""
plotting — Project-specific plotting modules

All plotting functions take pre-computed result dicts from the
corresponding compute_* functions. No computation inside plotting.

Modules:
    cv                  — Grid-search CV visualisation
    assignment          — Cohort-level BE/SC assignment strip
    sbi_trajectories    — Parameter trajectory plots
    sbi_posteriors      — Posterior distribution plots
    sbi_validation      — PPC and correlation plots
    adaptation          — Post-shift adaptation plots
    opto                — Opto inactivation plots
"""

# ── CV ───────────────────────────────────────────────────────────────────────
from plotting.cv import (
    plot_cv_comparison,
    plot_winner_summary,
    plot_update_matrix,
    plot_um_comparison,
    plot_param_distributions,
)

# ── Assignment ───────────────────────────────────────────────────────────────
from plotting.assignment import plot_assignment_strip

# ── SBI: trajectories ────────────────────────────────────────────────────────
from plotting.sbi_trajectories import (
    plot_parameter_trajectories,
    plot_performance_trajectory,
    plot_learning_trajectory,
    PARAM_COLOURS,
    PHASE_COLOURS,
)

# ── SBI: posteriors ──────────────────────────────────────────────────────────
from plotting.sbi_posteriors import (
    plot_marginal_posteriors,
    plot_pairplot,
    plot_posterior_psychometric,
)

# ── SBI: validation ──────────────────────────────────────────────────────────
from plotting.sbi_validation import (
    plot_summary_stats_comparison,
    plot_param_stat_correlations,
)

# ── Adaptation ───────────────────────────────────────────────────────────────
from plotting.adaptation import (
    plot_animal_trajectory,
    plot_shift_um_evolution,
    plot_shift_psychometric,
    plot_group_trajectories,
)

# ── Opto ─────────────────────────────────────────────────────────────────────
from plotting.opto import (
    plot_opto_psychometric,
    plot_phase_trajectory,
    plot_opto_um_comparison,
    plot_equivalence_test,
    plot_phase_interaction,
    plot_model_assignment_effects,
    plot_within_session_summary,
    plot_genotype_interaction,
    plot_expert_stability,
)

# Note: plotting/sbi.py is DELETED. Import from the three submodules above.
# Note: plotting/animal_report.py and plotting/validation_report.py are DELETED.
