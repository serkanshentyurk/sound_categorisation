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
SBI_PRIOR = 'learning'      # the prior variant (models/priors.py) the model fits use: the networks, the
                            # grid, synthetic cohorts and SBC draws. 'manuscript' is the ranges of the
                            # networks trained before mid-November 2026; a saved network records its own.
SBI_MAX_DRAWS = 5_000_000   # most prior draws one network may simulate to reach its training set when
                            # only draws passing the accuracy rule are kept (a stop, not a target)

# SBI conditioning / CV
SBI_N_CV_REPEATS = 64
SBI_N_POSTERIOR_SAMPLES = 50        # samples per conditioning INSIDE the CV repeats (median = point estimate)
SBI_N_POSTERIOR_SAMPLES_SAVED = 1000  # samples of the one all-session conditioning saved per animal
                                      # (intervals, narrowing, the example posterior)

# SBC calibration (sc-sbi-sbc): fresh prior draws per network, each processed like an animal.
# One array task per (rep, model, session count): for pooled, 5 sessions is an animal's full data
# and 3 and 2 are the CV halves the same network is conditioned on inside the model comparison; the
# single-session network is calibrated on one session.
SBC_DRAWS = {'pooled': {5: 1000, 3: 500, 2: 500}, 'moments': {5: 1000, 3: 500, 2: 500}, 'single': {1: 1000}}
SBC_COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)
SBC_POSTERIOR_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)   # stored per draw (recovery, widths)

# Model identification: an animal needs this many selected sessions, so that each of the two
# CV halves has at least 2
MODEL_ID_MIN_SESSIONS = 4

# SBI representations. N / T / mode flow into AmortisedSBI.
#   n_simulations   the size of the training set (simulations KEPT)
#   accuracy_rule   keep a simulation only if every simulated session reaches the session filter's
#                   minimum accuracy (config.yaml: model_id_<distribution>), as every selected real
#                   session does. For several sessions the rule is information the observation
#                   does not carry, so the filtered network is the one that matches selected
#                   animals. For one session the rule can be read off the accuracy statistic, so it
#                   changes nothing about the posterior, and the half-sessions the single-session
#                   test conditions on are not selected by it: no filter there.
SBI_SESSIONS_RANGE = (2, 5)   # multi-session reps draw N ~ uniform{2..5} per simulation: an animal's
                              # full data (4-5 sessions) and the CV halves the networks are also
                              # conditioned on (2-3 sessions). Until mid-Nov 2026: (2, 6); before
                              # Nov 2026: (4, 6).
SBI_TRAIN_T = (250, 600)   # trials per session of the multi-session reps: T ~ uniform{250..600} per
                           # session. 250 is the model_id preset's min_trials; some animals' sessions
                           # pass 500 (two animals average 501 and 524 valid trials a session).
                           # Until mid-Nov 2026: (250, 500). An int fixes T instead.
SBI_SINGLE_T = (100, 600)  # the single-session network is also conditioned on half a session
                           # (inference.selection._condition_single), so it trains from 100 trials
SBI_DESIGN_FEATURES = ('n_sessions', 'n_valid_trials')
                           # how much data x was built from, appended after the statistics, so
                           # the posterior's width follows the data actually given (a CV half
                           # or an animal's full data). Networks saved before Nov 2026 have none.
SBI_REPRESENTATIONS = {
    'pooled':  {'N': SBI_SESSIONS_RANGE, 'T': SBI_TRAIN_T,  'mode': 'pooled',  'n_simulations': 100_000,
                'accuracy_rule': True},
    'moments': {'N': SBI_SESSIONS_RANGE, 'T': SBI_TRAIN_T,  'mode': 'moments', 'n_simulations': 150_000,
                'accuracy_rule': True},
    'single':  {'N': 1,                  'T': SBI_SINGLE_T, 'mode': 'pooled',  'n_simulations': 200_000,
                'accuracy_rule': False},
}
# Per-distribution SBI training: conditioning is always on a single, KNOWN phase
# (uniform / hard_a / hard_b), so a specialist network matched to each phase beats
# one network marginalising over phase (which you'd never need, since the phase is
# always supplied). The job grids train the SBI_TRAIN_REPS x SBI_TRAIN_DISTRIBUTIONS subset
# below (4 networks now). Conditioning routes to the matching network via snpe_net_path's filename.
SBI_TRAIN_DISTRIBUTIONS = ('uniform',)
# Which representations / distributions the training and conditioning arrays cover NOW. The
# others stay available by name (sc-sbi-train --rep moments --distribution hard_a) but are not in
# the job grids. Widen these tuples to bring them back.
# 'pooled' votes in the consensus; 'single' is reported beside it (per-session fits, exploratory).
# 'moments' was trained until mid-Nov 2026: its between-session variances rest on 2-5 sessions, and
# the animals vary across days more than a model with fixed parameters does.
SBI_TRAIN_REPS = ('pooled', 'single')

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
# EQUIVALENCE MARGINS (prediction 1: PPC dispensable in experts)
# =============================================================================
# Fixed in November 2026, before the laser effect was read against them. mu and sigma: about the
# criterion shift an ideal observer makes when the distribution changes, so an effect this size
# would matter against the task's own demands. The rest: five percentage points. A laser effect
# whose 95 % interval lies inside +/- margin counts as absent (behaviour/contrasts.py:
# equivalence_verdict; the opto-contrasts report writes equivalence.csv). A statistic without a
# margin gets no verdict.
EQUIVALENCE_MARGINS = {
    'mu': 0.10, 'sigma': 0.10,
    'accuracy': 0.05, 'hard_accuracy': 0.05, 'easy_accuracy': 0.05, 'side_bias': 0.05,
    'lapse_low': 0.05, 'lapse_high': 0.05, 'recency': 0.05, 'win_stay': 0.05, 'lose_shift': 0.05,
}


# =============================================================================
# RANDOM SEED (base)
# =============================================================================

BASE_SEED = 42
# Validation data (SBC draws, synthetic cohorts) are seeded from here. Training draw i uses numpy
# seed BASE_SEED + i (i < SBI_MAX_DRAWS), so nothing seeded from this base can replay a training
# simulation's stimulus stream.
VALIDATION_SEED = 10_000_000
