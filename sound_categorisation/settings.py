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
GS_N_SEEDS = 32             # real animals
SYNTH_GS_N_SEEDS = 8        # synthetic cohorts (picked by cohort kind in sc-grid-search)
GS_BURN_IN = 1000
GS_N_BINS = 8

# SBI training
SBI_N_SIMULATIONS = 50_000
SBI_N_GENERIC_TRIALS = 2500
SBI_BURN_IN = 1000

# SBI conditioning / CV
SBI_N_CV_REPEATS = 64
SBI_N_POSTERIOR_SAMPLES = 50        # samples per conditioning INSIDE the CV repeats (median = point estimate)
SBI_N_POSTERIOR_SAMPLES_SAVED = 1000  # samples of the one all-session conditioning saved per animal
                                      # (intervals, narrowing, the example posterior)

# SBC calibration (sc-sbi-sbc): fresh prior draws per network, each processed like an animal
SBC_N_DRAWS = 300
SBC_N_SESSIONS = 5
SBC_COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)

# Model identification: an animal needs this many selected sessions (the networks saw 4-6)
MODEL_ID_MIN_SESSIONS = 4

# SBI representations: 3 networks x 2 models = 6. N/mode flow into AmortisedSBI;
# n_simulations is per-rep (moments is 2*D-dim, so it needs more than the D-dim reps).
SBI_SESSIONS_RANGE = (4, 6)   # multi-session reps draw N ~ uniform{4,5,6} per simulation,
                              # matching the 4-6 sessions a real animal typically has.
SBI_TRAIN_T = (250, 500)   # trials per session for ALL reps: T ~ uniform{250..500} per session,
                           # the range of valid trials a real Uniform session has (median ~300,
                           # 250 is the model_id preset's min_trials; a few animals reach ~500).
                           # Drawn like N, so no animal's session length is out of distribution.
                           # An int fixes T instead.
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
SBI_TRAIN_DISTRIBUTIONS = ('uniform',)
# Which representations / distributions the training and conditioning arrays cover NOW. The
# others stay available by name (sc-sbi-train --rep single --distribution hard_a) but are not in
# the job grids. Widen these tuples to bring them back.
SBI_TRAIN_REPS = ('pooled', 'moments')

# Fast runs: used when --fast is passed on the command line (same code path, fewer draws)
FAST_GS_N_SEEDS = 2
FAST_SBI_N_SIMULATIONS = 500
FAST_SBI_N_GENERIC_TRIALS = 200
FAST_N_ANIMALS_LIMIT = 2
FAST_SBC_N_DRAWS = 10


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
