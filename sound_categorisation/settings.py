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
GS_N_SEEDS = 16             # real and synthetic cohorts alike, so a validation run repeats the
                            # real run's procedure; sc-grid-search --n-seeds overrides
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

# SBC calibration (sc-sbi-sbc): fresh prior draws per network, each processed like an animal.
# One array task per (rep, model, session count): 5 sessions is an animal's full data; 3 and 2
# are the CV halves the same networks are conditioned on inside the model comparison.
SBC_DRAWS_BY_SESSIONS = {5: 1000, 3: 500, 2: 500}
SBC_COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)
SBC_POSTERIOR_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)   # stored per draw (recovery, widths)

# Model identification: an animal needs this many selected sessions, so that each of the two
# CV halves has at least 2
MODEL_ID_MIN_SESSIONS = 4

# SBI representations: 3 networks x 2 models = 6. N/mode flow into AmortisedSBI;
# n_simulations is per-rep (moments is 2*D-dim, so it needs more than the D-dim reps).
SBI_SESSIONS_RANGE = (2, 6)   # multi-session reps draw N ~ uniform{2..6} per simulation: an
                              # animal's full data (4-5 sessions) and the CV halves the networks
                              # are also conditioned on (2-3 sessions). Before Nov 2026: (4, 6).
SBI_TRAIN_T = (250, 500)   # trials per session for ALL reps: T ~ uniform{250..500} per session,
                           # the range of valid trials a real Uniform session has (median ~300,
                           # 250 is the model_id preset's min_trials; a few animals reach ~500).
                           # Drawn like N, so no animal's session length is out of distribution.
                           # An int fixes T instead.
SBI_DESIGN_FEATURES = ('n_sessions', 'n_valid_trials')
                           # how much data x was built from, appended after the statistics, so
                           # the posterior's width follows the data actually given (a CV half
                           # or an animal's full data). Networks saved before Nov 2026 have none.
SBI_REPRESENTATIONS = {
    'pooled':  {'N': SBI_SESSIONS_RANGE, 'T': SBI_TRAIN_T, 'mode': 'pooled',  'n_simulations': 100_000},
    'moments': {'N': SBI_SESSIONS_RANGE, 'T': SBI_TRAIN_T, 'mode': 'moments', 'n_simulations': 150_000},
    'single':  {'N': 1,                  'T': SBI_TRAIN_T, 'mode': 'pooled',  'n_simulations':  50_000},
}
# Per-distribution SBI training: conditioning is always on a single, KNOWN phase
# (uniform / hard_a / hard_b), so a specialist network matched to each phase beats
# one network marginalising over phase (which you'd never need, since the phase is
# always supplied). 3 reps x 2 models x 3 dists = 18 networks are available; the job
# grids train the SBI_TRAIN_REPS x SBI_TRAIN_DISTRIBUTIONS subset below (4 now).
# Conditioning routes to the matching network via snpe_net_path's filename.
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
# Validation data (SBC draws, synthetic cohorts) are seeded from here. Training simulation i
# uses numpy seed BASE_SEED + i (i < 150,000), so nothing seeded from this base can replay a
# training simulation's stimulus stream.
VALIDATION_SEED = 10_000_000
