"""
Shared Configuration for Scripts

Central location for constants used by cluster scripts, local scripts,
and notebooks. Change here, not in the CLIs.

Everything that affects a run (settings, versions, paths) should be
saved alongside results via the `build_metadata()` helper, so runs
are reproducible and traceable.
"""

import platform
import socket
from datetime import datetime
from pathlib import Path

from sound_categorisation.inference.constants import SBI_STATS
from sound_categorisation.settings import (
    BASE_SEED,
    EXPERT_LAST_FRACTION,
    EXPERT_MIN_ACCURACY,
    GS_BURN_IN,
    GS_N_BINS,
    GS_N_FOLDS,
    GS_N_SEEDS,
    MIN_VALID_TRIALS,
    SBI_BURN_IN,
    SBI_N_CV_REPEATS,
    SBI_N_GENERIC_TRIALS,
    SBI_N_SIMULATIONS,
    STAGE,
)

# =============================================================================
# PATHS
# =============================================================================

# Relative to repo root. Scripts should Path(__file__).parent.parent to reach it.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent   # <repo>/sound_categorisation/data/paths.py


# --- External data root (results live outside the repo) -----------------------
# Mirrors data/snapshot.py: local = <repo>/../../data, cluster = ceph Processed.
# All results / validation / cohort paths derive from data_root() — see
# results_dir(), cohort_path(), snpe_networks_dir() below.
_CLUSTER_DATA_ROOT = Path('/ceph/akrami/Serkan/Head_Fixed_Behavior/Data/Processed')


def _on_cluster() -> bool:
    return any(x in socket.gethostname() for x in ('hpc', 'gpu', 'enc', 'sgw'))


def data_root(repo_root: Path = None) -> Path:
    """Machine-aware results root. Local: <repo>/../../data. Cluster: ceph Processed."""
    if _on_cluster():
        return _CLUSTER_DATA_ROOT
    return (repo_root or REPO_ROOT).parent.parent / 'data'


def cohort_path(cohort: str) -> Path:
    """Shared synthetic-cohort pickle (used by both GS and SBI validation)."""
    return data_root() / 'synthetic_cohorts' / f'{cohort}.pkl'


def results_dir(method: str, run: str, cohort: str, fit_target: str) -> Path:
    """Per-(animal, model) results directory. method='grid_search'|'sbi', run='quick'|'full'."""
    return data_root() / method / run / f'{cohort}_{fit_target}'


def snpe_networks_dir() -> Path:
    return data_root() / 'snpe_networks'


def snpe_net_path(rep: str, model: str, distribution: str) -> Path:
    """Explicit per-(rep, model, distribution) network file.

    Filename: 'snpe_{rep}_{model}_{distribution}.pkl' (e.g. 'snpe_moments_SC_hard_a.pkl'). The
    'snpe_' prefix marks these as SNPE/SBI networks. Single source of the naming — train_sbi
    writes it, run_sbi reads it.
    """
    return snpe_networks_dir() / f'snpe_{rep}_{model}_{distribution}.pkl'


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

