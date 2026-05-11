"""
Shared Configuration for Scripts

Central location for constants used by cluster scripts, local scripts,
and notebooks. Change here, not in the scripts.

Everything that affects a run (settings, versions, paths) should be
saved alongside results via the `build_metadata()` helper, so runs
are reproducible and traceable.
"""

from pathlib import Path
from datetime import datetime
import platform
import socket


# =============================================================================
# PATHS
# =============================================================================

# Relative to repo root. Scripts should Path(__file__).parent.parent to reach it.
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / 'results'

# Sub-directories — scripts create these lazily.
SNPE_DIR = RESULTS_DIR / 'snpe'
CV_DIR = RESULTS_DIR / 'cv'
SBI_STATIC_DIR = RESULTS_DIR / 'sbi_static'
SBI_DYNAMIC_DIR = RESULTS_DIR / 'sbi_dynamic'
VALIDATION_DIR = RESULTS_DIR / 'validation'
SYNTH_COHORTS_DIR = VALIDATION_DIR / 'synthetic_cohorts'
LOGS_DIR = RESULTS_DIR / 'logs'


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
# SUMMARY STATS FOR SBI (heuristics-only, no UM or conditional psych)
# =============================================================================

SBI_STATS = [
    'accuracy', 'psychometric', 'recency', 'stimulus_recency',
    'win_stay', 'lose_shift', 'side_bias', 'stimulus_sensitivity',
    'choice_entropy', 'perseveration',
]


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

# Dynamic SBI (per-animal RandomWalk)
DYNAMIC_SBI_N_SIMULATIONS = 30_000

# Per-parameter sigma_drift — scaled to ~3-5% of each parameter's range.
# A single global value was too aggressive for narrow-range parameters
# (gamma range=0.20 with drift=0.05 → 25% per step) and about right
# for wide-range ones (eta_learning range=0.94 with drift=0.05 → 5%).
DYNAMIC_SBI_SIGMA_DRIFT = {
    'BE': {'eta_learning': 0.04, 'eta_relax': 0.02},
    'SC': {'gamma': 0.015, 'sigma_update': 0.04},
}

# Parameters to fit with RandomWalk linking (the rest get ConstantSpec).
# For real data, include all potentially varying params.
# For synthetic validation, only mark the actually-varying ones (see
# SYNTH_DYNAMIC_VARYING_PARAMS below).
DYNAMIC_SBI_VARYING_PARAMS = {
    'BE': ('eta_learning', 'eta_relax'),
    'SC': ('gamma', 'sigma_update'),
}

# Wider bounds for dynamic SBI: must accommodate the full naive→expert
# learning arc, not just expert-phase values. The static bounds in
# BEParams.get_bounds() / SCParams.get_bounds() are for expert-phase
# fitting only. These override them for dynamic fitting.
DYNAMIC_SBI_BOUNDS = {
    'BE': {
        'sigma_percep': (0.05, 0.5),    # same as static
        'A_repulsion': (0.0, 0.5),      # same as static
        'eta_learning': (0.01, 0.95),   # was (0.05, 0.90) — learning starts ~0.02
        'eta_relax': (0.01, 0.4),       # same as static
    },
    'SC': {
        'sigma_percep': (0.05, 0.5),    # same as static
        'A_repulsion': (0.0, 0.5),      # same as static
        'gamma': (0.3, 0.999),          # was (0.80, 0.999) — learning starts ~0.5
        'sigma_update': (0.05, 1.0),    # same as static
    },
}

# Synthetic validation: only the params that actually vary in the
# synthetic trajectory. eta_relax and sigma_update are held constant
# by _compute_session_params, so fitting them as RandomWalk wastes
# capacity and produces spurious drift.
SYNTH_DYNAMIC_VARYING_PARAMS = {
    'BE': ('eta_learning',),
    'SC': ('gamma',),
}

# Synthetic validation cohorts
SYNTH_N_PER_MODEL = 20
SYNTH_N_SESSIONS = 15
SYNTH_TRIALS_PER_SESSION = 350

# Smoke test: used when --smoke-test is passed on the command line
SMOKE_GS_N_SEEDS = 2
SMOKE_SBI_N_SIMULATIONS = 500
SMOKE_SBI_N_GENERIC_TRIALS = 200
SMOKE_DYNAMIC_SBI_N_SIMULATIONS = 200
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


# =============================================================================
# METADATA HELPERS
# =============================================================================

def build_metadata(script_name: str, args: dict) -> dict:
    """
    Build a metadata dict to save alongside every result file.

    Captures: script, args, timestamp, host, platform, config constants,
    and library versions where we can get them.
    """
    import sys

    meta = {
        'script': script_name,
        'args': dict(args),  # shallow copy; caller should pass serialisable dict
        'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python_version': sys.version.split()[0],
        'config': {
            'GS_N_FOLDS': GS_N_FOLDS,
            'GS_N_SEEDS': GS_N_SEEDS,
            'GS_BURN_IN': GS_BURN_IN,
            'GS_N_BINS': GS_N_BINS,
            'SBI_N_SIMULATIONS': SBI_N_SIMULATIONS,
            'SBI_N_GENERIC_TRIALS': SBI_N_GENERIC_TRIALS,
            'SBI_BURN_IN': SBI_BURN_IN,
            'SBI_N_CV_REPEATS': SBI_N_CV_REPEATS,
            'SBI_STATS': list(SBI_STATS),
            'DYNAMIC_SBI_N_SIMULATIONS': DYNAMIC_SBI_N_SIMULATIONS,
            'DYNAMIC_SBI_SIGMA_DRIFT': DYNAMIC_SBI_SIGMA_DRIFT,
            'DYNAMIC_SBI_BOUNDS': DYNAMIC_SBI_BOUNDS,
            'DYNAMIC_SBI_VARYING_PARAMS': {
                k: list(v) for k, v in DYNAMIC_SBI_VARYING_PARAMS.items()
            },
            'SYNTH_DYNAMIC_VARYING_PARAMS': {
                k: list(v) for k, v in SYNTH_DYNAMIC_VARYING_PARAMS.items()
            },
            'EXPERT_MIN_ACCURACY': EXPERT_MIN_ACCURACY,
            'EXPERT_LAST_FRACTION': EXPERT_LAST_FRACTION,
            'MIN_VALID_TRIALS': MIN_VALID_TRIALS,
            'STAGE': STAGE,
            'BASE_SEED': BASE_SEED,
        },
        'versions': _get_versions(),
        'git_sha': _get_git_sha(),
    }
    return meta


def _get_versions() -> dict:
    """Collect version info for key libraries. Silent on failure."""
    versions = {}
    for pkg in ('numpy', 'scipy', 'pandas', 'torch', 'sbi', 'sklearn', 'joblib'):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, '__version__', 'unknown')
        except Exception:
            versions[pkg] = 'not installed'
    return versions


def _get_git_sha() -> str:
    """Return the current git SHA or 'unknown' if not a git repo."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=5,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            # Also check if working tree is dirty
            dirty = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True, text=True, cwd=REPO_ROOT, timeout=5,
            )
            if dirty.returncode == 0 and dirty.stdout.strip():
                sha += '-dirty'
            return sha
    except Exception:
        pass
    return 'unknown'


def ensure_dirs():
    """Create all standard results subdirectories if they don't exist."""
    for d in [RESULTS_DIR, SNPE_DIR, CV_DIR, SBI_STATIC_DIR,
              SBI_DYNAMIC_DIR, VALIDATION_DIR, SYNTH_COHORTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def apply_smoke_test_overrides(config_dict: dict) -> dict:
    """
    Return a copy of config_dict with smoke-test overrides applied.

    Use in scripts: `if args.smoke_test: cfg = apply_smoke_test_overrides(cfg)`.
    """
    overrides = {
        'GS_N_SEEDS': SMOKE_GS_N_SEEDS,
        'SBI_N_SIMULATIONS': SMOKE_SBI_N_SIMULATIONS,
        'SBI_N_GENERIC_TRIALS': SMOKE_SBI_N_GENERIC_TRIALS,
        'DYNAMIC_SBI_N_SIMULATIONS': SMOKE_DYNAMIC_SBI_N_SIMULATIONS,
        'SYNTH_N_PER_MODEL': SMOKE_SYNTH_N_PER_MODEL,
        'N_ANIMALS_LIMIT': SMOKE_N_ANIMALS_LIMIT,
    }
    out = dict(config_dict)
    out.update(overrides)
    return out


# =============================================================================
# DATA LOADING HELPERS
# =============================================================================

# Default config files — scripts use --config to override
DEFAULT_CONFIG = REPO_ROOT / 'config.yaml'
CLUSTER_CONFIG = REPO_ROOT / 'config_slurm.yaml'


def load_project_config(config_path=None):
    """Load ProjectConfig from YAML. Auto-detects cluster vs local."""
    from behav_utils.config.schema import load_config

    if config_path is not None:
        return load_config(str(config_path))

    # Auto-detect: use config_slurm.yaml if it exists and we're on the cluster
    if CLUSTER_CONFIG.exists():
        import socket
        hostname = socket.gethostname()
        if any(x in hostname for x in ('hpc', 'gpu', 'enc', 'sgw')):
            return load_config(str(CLUSTER_CONFIG))

    return load_config(str(DEFAULT_CONFIG))


def load_animal_data(animal_id, config=None, config_path=None):
    """
    Load one animal's data, handling path construction.

    Args:
        animal_id: e.g. 'SS01'
        config: ProjectConfig (if already loaded)
        config_path: Path to config YAML (loads if config is None)

    Returns:
        AnimalData
    """
    from behav_utils.data.loading import load_animal

    if config is None:
        config = load_project_config(config_path)

    data_dir = Path(config.file_structure.data_dir)
    return load_animal(data_dir / animal_id, config)


def list_animal_ids(config=None, config_path=None):
    """List available animal IDs from the data directory."""
    if config is None:
        config = load_project_config(config_path)

    data_dir = Path(config.file_structure.data_dir)
    if not data_dir.exists():
        return []

    return sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
