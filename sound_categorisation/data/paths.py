"""
Locations and run bookkeeping.

Inputs (snapshot, synthetic cohorts, trained SBI networks) live under ``data_root()``; results live
under ``results_root()`` as ``<report>/<cohort>/<run_id>/`` (see the RESULTS section). Every producer
calls ``start_run`` and stamps ``build_metadata(...)`` next to what it writes; every consumer finds a
run with ``resolve_run``. Constants are in ``sound_categorisation.settings``.
"""

import os
import platform
import re
import socket
from datetime import datetime, timezone
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent   # <repo>/sound_categorisation/data/paths.py

# Inputs that are not results: the snapshot, synthetic cohorts and trained SBI networks live under the
# data root. Local: <repo>/../../data. Cluster: ceph Processed. See data/snapshot.py for the snapshot.
_CLUSTER_DATA_ROOT = Path('/ceph/akrami/Serkan/Head_Fixed_Behavior/Data/Processed')
_CLUSTER_HOST_MARKERS = ('hpc', 'gpu', 'enc', 'sgw')


def _on_cluster() -> bool:
    return any(x in socket.gethostname() for x in _CLUSTER_HOST_MARKERS)


def data_root(repo_root: Path = None) -> Path:
    """Machine-aware data root (inputs). Local: <repo>/../../data. Cluster: ceph Processed.
    Override with the SC_DATA_ROOT environment variable."""
    env = os.environ.get('SC_DATA_ROOT')
    if env:
        return Path(env)
    if _on_cluster():
        return _CLUSTER_DATA_ROOT
    return (repo_root or REPO_ROOT).parent.parent / 'data'


def cohort_path(cohort: str) -> Path:
    """Synthetic-cohort pickle (an input to GS and SBI validation runs)."""
    return data_root() / 'synthetic_cohorts' / f'{cohort}.pkl'


def snpe_networks_dir() -> Path:
    return data_root() / 'snpe_networks'


def snpe_net_path(rep: str, model: str, distribution: str) -> Path:
    """Explicit per-(rep, model, distribution) network file.

    Filename: 'snpe_{rep}_{model}_{distribution}.pkl' (e.g. 'snpe_moments_SC_hard_a.pkl'). Single
    source of the naming — train_sbi writes it, run_sbi reads it.
    """
    return snpe_networks_dir() / f'snpe_{rep}_{model}_{distribution}.pkl'


# =============================================================================
# RESULTS: one root, one layout, one run id
#
#   <results_root>/<report>/<cohort>/<run_id>/...
#   <results_root>/<report>/<cohort>/latest      -> symlink to the newest run  (+ latest.txt)
#
#   report  ∈ REPORTS (the analysis, not the mechanism)
#   run_id  = YYYY-MM-DD_HHMM_<git sha7>[_fast][_<label>]
#
# Every producer (reports CLI, model-identification runners) writes under a run directory and
# stamps run_id, argv and git state into its meta.json. Consumers resolve runs through
# resolve_run(), never by a hard-coded path.
# =============================================================================

REPORTS = ('opto_contrasts', 'switch_adaptation', 'light_artefact', 'model_identification')


def results_root() -> Path:
    """Where results go. Local: <repo>/results. Cluster: <data root>/results.
    Override with the SC_RESULTS_ROOT environment variable."""
    env = os.environ.get('SC_RESULTS_ROOT')
    if env:
        return Path(env)
    if _on_cluster():
        return _CLUSTER_DATA_ROOT / 'results'
    return REPO_ROOT / 'results'


def git_state() -> dict:
    """{'sha': full sha or None, 'short': 7 chars or None, 'dirty': bool}. Never raises."""
    import subprocess
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                             cwd=REPO_ROOT, timeout=5)
        if sha.returncode != 0:
            return {'sha': None, 'short': None, 'dirty': False}
        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True,
                                cwd=REPO_ROOT, timeout=5)
        full = sha.stdout.strip()
        return {'sha': full, 'short': full[:7], 'dirty': bool(status.stdout.strip())}
    except Exception:
        return {'sha': None, 'short': None, 'dirty': False}


def new_run_id(fast: bool = False, label: str | None = None, now: datetime | None = None) -> str:
    """'YYYY-MM-DD_HHMM_<sha7>' plus '_fast' for reduced runs and an optional free label. The time keeps two
    runs of the same kind on the same day at the same commit apart."""
    g = git_state()
    now = now or datetime.now()
    parts = [now.strftime('%Y-%m-%d_%H%M'), g['short'] or 'nogit']
    if fast:
        parts.append('fast')
    if label:
        parts.append(re.sub(r'[^A-Za-z0-9.-]+', '-', label).strip('-'))
    return '_'.join(parts)


def _check_report(report: str) -> None:
    if report not in REPORTS:
        raise ValueError(f'unknown report {report!r}; one of {REPORTS}')


def run_dir(report: str, cohort: str, run_id: str, root: Path | None = None) -> Path:
    """The directory of one run (not created)."""
    _check_report(report)
    return Path(root or results_root()) / report / cohort / run_id


def mark_latest(run_path: Path) -> None:
    """Point <report>/<cohort>/latest at this run: a relative symlink, plus latest.txt holding the
    run id for filesystems or tools that cannot follow symlinks."""
    run_path = Path(run_path)
    parent = run_path.parent
    (parent / 'latest.txt').write_text(run_path.name + '\n')
    link = parent / 'latest'
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(run_path.name, target_is_directory=True)
    except OSError:
        pass                                   # latest.txt is the fallback


def latest_run(report: str, cohort: str, root: Path | None = None) -> Path:
    """The newest run directory for (report, cohort): the `latest` link, else latest.txt, else the
    newest run-id-shaped directory. Raises FileNotFoundError if there is none."""
    _check_report(report)
    parent = Path(root or results_root()) / report / cohort
    link = parent / 'latest'
    if link.is_symlink() and link.resolve().is_dir():
        return link.resolve()
    txt = parent / 'latest.txt'
    if txt.exists() and (parent / txt.read_text().strip()).is_dir():
        return parent / txt.read_text().strip()
    runs = sorted((p for p in parent.glob('*') if p.is_dir() and _RUN_ID.match(p.name)),
                  key=lambda p: (p.stat().st_mtime, p.name))          # newest by time, not by name
    if runs:
        return runs[-1]
    raise FileNotFoundError(f'no runs under {parent}. {_available(report, root)}')


def list_runs(report: str, cohort: str, root: Path | None = None) -> list:
    """Run ids under <report>/<cohort>, oldest first."""
    parent = Path(root or results_root()) / report / cohort
    return sorted(p.name for p in parent.glob('*') if p.is_dir() and _RUN_ID.match(p.name))


def _available(report: str, root: Path | None = None) -> str:
    base = Path(root or results_root()) / report
    cohorts = sorted(p.name for p in base.glob('*') if p.is_dir()) if base.exists() else []
    return f'cohorts with {report} runs: {cohorts}' if cohorts else f'no {report} runs under {base}'


_RUN_ID = re.compile(r'^\d{4}-\d{2}-\d{2}_(\d{4}_)?([0-9a-f]{7}|nogit)')   # time optional: older runs


def resolve_run(report: str, cohort: str, run: str = 'latest', root: Path | None = None) -> Path:
    """'latest', a run id, or an absolute/relative path to a run directory."""
    if run in (None, '', 'latest'):
        return latest_run(report, cohort, root)
    p = Path(run)
    if p.is_dir():
        return p
    d = run_dir(report, cohort, run, root)
    if not d.is_dir():
        raise FileNotFoundError(f'no run {run!r} under {d.parent}; runs there: {list_runs(report, cohort, root)}')
    return d


def start_run(report: str, cohort: str, run_id: str | None = None, *, fast: bool = False,
              label: str | None = None, root: Path | None = None) -> Path:
    """Create (or reopen) a run directory with a logs/ folder and point `latest` at it."""
    rid = run_id or new_run_id(fast=fast, label=label)
    d = run_dir(report, cohort, rid, root)
    (d / 'logs').mkdir(parents=True, exist_ok=True)
    mark_latest(d)
    return d


def model_id_dir(run: Path, method: str, fit_target: str, distribution: str, rep: str | None = None) -> Path:
    """``<run>/<method>/<fit_target>/<distribution>[/<rep>]`` — the one layout shared by run_gs
    (method='grid_search'), run_sbi (method='sbi', with rep) and consensus (which reads both)."""
    d = Path(run) / method / fit_target / distribution
    return d / rep if rep else d


# =============================================================================
# METADATA HELPERS
# =============================================================================

def build_metadata(script_name: str, args: dict, run_id: str | None = None) -> dict:
    """Metadata dict to save alongside every result file: command, args, run id, time, host,
    settings, package versions, git state."""
    import sys

    g = git_state()
    return {
        'script': script_name,
        'argv': list(sys.argv),
        'args': dict(args),  # shallow copy; caller passes a serialisable dict
        'run_id': run_id,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python_version': sys.version.split()[0],
        'settings': {
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
        'git_sha': g['sha'],
        'git_dirty': g['dirty'],
    }


def _get_versions() -> dict:
    """Version info for key libraries. Silent on failure."""
    versions = {}
    for pkg in ('behav_utils', 'sound_categorisation', 'numpy', 'scipy', 'pandas', 'torch', 'sbi', 'joblib'):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, '__version__', 'unknown')
        except Exception:
            versions[pkg] = 'not installed'
    return versions


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
    if CLUSTER_CONFIG.exists() and _on_cluster():
        return load_config(str(CLUSTER_CONFIG))

    return load_config(str(DEFAULT_CONFIG))

