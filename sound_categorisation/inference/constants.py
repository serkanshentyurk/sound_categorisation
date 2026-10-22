# =============================================================================
# SUMMARY STATS FOR SBI (scalar summaries only; no update matrix, no conditional psych)
#
# Chosen by sc-feature-selection (results/feature_selection/uniform, Oct 2026; 2,000 BE +
# 2,000 SC simulated animals x 2,500 Uniform trials, HGB surrogate, 5-fold CV):
#   - identity (BE vs SC) saturates at AUC ~0.97 with a handful of history stats;
#     psychometric + win_stay/lose_shift/recency carry it (logistic_history scored
#     higher alone but is a fitted regression and adds nothing the simple stats lack);
#   - parameters: sigma_percep from the psychometric block (BE R^2 ~0.77), A_repulsion from
#     perseveration / recency (~0.5), eta_learning from win_stay / lose_shift (~0.45);
#   - NOT recoverable from any scalar in the pool on Uniform sessions: eta_relax (BE) and
#     gamma (SC); SC's sigma_percep / sigma_update only moderately (0.3-0.5). Their
#     posteriors stay close to the prior -- report that, do not claim recovery.
# Dropped from the previous vector: stimulus_recency (identical to recency on these data),
# side_bias and choice_entropy (uninformative on Uniform). The moments representation uses
# the same stats (mean + variance of each across sessions).
# Changing this list invalidates every trained network (AmortisedSBI.condition warns).
# =============================================================================

SBI_STATS = [
    'accuracy', 'mu', 'sigma', 'lapse_low', 'lapse_high', 'stimulus_sensitivity',
    'recency', 'win_stay', 'lose_shift', 'perseveration',
]
