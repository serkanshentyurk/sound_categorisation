"""
Project settings: distributions, model types, fit grid sizes, SBI simulation sizes,
expert-selection thresholds, seeds. Pure constants — no I/O, no imports of project code.
"""

# =============================================================================
# FIT TARGETS (shared vocabulary)
# =============================================================================

FIT_TARGETS = ('update_matrix', 'conditional_psych')


# =============================================================================
# MODEL TYPES
# =============================================================================

MODEL_TYPES = ('BE', 'SC')
MODEL_TYPES_LOWER = ('be', 'sc')


# =============================================================================
# DISTRIBUTIONS
# =============================================================================

DISTRIBUTIONS = ('uniform', 'hard_a', 'hard_b')


# =============================================================================
# SIMULATION & FITTING PARAMETERS
# =============================================================================

# Grid search
GS_N_FOLDS = 2
GS_N_SEEDS = 32
SYNTH_GS_N_SEEDS = 8        # Synthetic validation needs fewer seeds than real data
GS_BURN_IN = 1000
GS_N_BINS = 8

# SBI training
SBI_N_SIMULATIONS = 50_000
SBI_N_GENERIC_TRIALS = 2500
SBI_BURN_IN = 1000

# SBI conditioning / CV
SBI_N_CV_REPEATS = 64
SBI_N_POSTERIOR_SAMPLES = 50
SBI_N_STOCHASTIC_REPS = 10

# Synthetic validation cohorts
SYNTH_N_PER_MODEL = 20
SYNTH_N_SESSIONS = 15
SYNTH_TRIALS_PER_SESSION = 600   # cohort/validation session length (~ real expert sessions)

# SBI representations: 3 networks x 2 models = 6. N/mode flow into AmortisedSBI;
# n_simulations is per-rep (moments is 2*D-dim, so it needs more than the D-dim reps).
SBI_SESSIONS_RANGE = (4, 6)   # multi-session reps draw N ~ uniform{4,5,6} per simulation,
                              # matching the 4-6 sessions a real animal typically has.
SBI_TRAIN_T = 500   # trials per session for ALL reps -- matches one real session, so the
                    # single net conditions in-distribution and its midpoint CV folds land
                    # at ~250 trials (accepted: noisier held-out UM, honest to real data).
SBI_REPRESENTATIONS = {
    'pooled':  {'N': SBI_SESSIONS_RANGE, 'T': SBI_TRAIN_T, 'mode': 'pooled',  'n_simulations':  50_000},
    'moments': {'N': SBI_SESSIONS_RANGE, 'T': SBI_TRAIN_T, 'mode': 'moments', 'n_simulations': 100_000},
    'single':  {'N': 1,                  'T': SBI_TRAIN_T, 'mode': 'pooled',  'n_simulations':  50_000},
}
# Per-distribution SBI training: conditioning is always on a single, KNOWN phase
# (uniform / hard_a / hard_b), so a specialist network matched to each phase beats
# one network marginalising over phase (which you'd never need, since the phase is
# always supplied). 3 reps x 2 models x 3 dists = 18 networks; conditioning routes
# to the matching network automatically via snpe_net_path's filename.
SBI_TRAIN_DISTRIBUTIONS = DISTRIBUTIONS

# Smoke test: used when --smoke-test is passed on the command line
SMOKE_GS_N_SEEDS = 2
SMOKE_SBI_N_SIMULATIONS = 500
SMOKE_SBI_N_GENERIC_TRIALS = 200
SMOKE_N_ANIMALS_LIMIT = 2
SMOKE_SYNTH_N_PER_MODEL = 2


# =============================================================================
# SESSION SELECTION
# =============================================================================

EXPERT_MIN_ACCURACY = 0.70
EXPERT_LAST_FRACTION = 0.50
MIN_VALID_TRIALS = 30
STAGE = 'Full_Task_Cont'


# =============================================================================
# RANDOM SEED (base)
# =============================================================================

BASE_SEED = 42
