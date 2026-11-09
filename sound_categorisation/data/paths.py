"""
Locations and run bookkeeping. Every path the code uses comes from here.

On the cluster everything hangs off one setting, the environment variable ``BEHAV_DATA_DIR`` (your folder
on ceph, e.g. ``/ceph/akrami/<Name>/Head_Fixed_Behavior/Data``):

    $BEHAV_DATA_DIR/Raw                    raw session folders (config.yaml, file_structure.data_dir)
    $BEHAV_DATA_DIR/Processed              data_root(): snapshot, SBI networks, synthetic cohorts
    $BEHAV_DATA_DIR/Processed/results      results_root()

On a laptop: data_root() is <repo>/../../data and results_root() is <repo>/results (override with
SC_DATA_ROOT / SC_RESULTS_ROOT). ``sc-fetch`` copies cluster folders to the same places on the laptop.

Results are ``<report>/<cohort>/<run_id>/`` (see the RESULTS section). Every producer calls ``start_run``
and stamps ``build_metadata(...)`` next to what it writes; every consumer finds a run with ``resolve_run``.
Constants are in ``sound_categorisation.settings``.
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

_CLUSTER_HOST_MARKERS = ('hpc', 'gpu', 'enc', 'sgw')

# How sc-fetch and sc-daily-tracker reach ceph from a laptop: through the SWC gateway to a login node, as
# $SWC_USER. The node is hpc-gw2 unless SWC_NODE names another (hpc-gw1 when hpc-gw2 does not answer).
SWC_GATEWAY = 'ssh.swc.ucl.ac.uk'
DEFAULT_SWC_NODE = 'hpc-gw2'


def swc_node() -> str:
    """The login node to go through: $SWC_NODE if set, else hpc-gw2."""
    return os.environ.get('SWC_NODE') or DEFAULT_SWC_NODE


def _on_cluster() -> bool:
    return any(x in socket.gethostname() for x in _CLUSTER_HOST_MARKERS)


def ceph_root() -> Path:
    """Your folder on ceph: the BEHAV_DATA_DIR environment variable. Stops with a clear message if it is
    not set, rather than falling back to someone else's folder."""
    value = os.environ.get('BEHAV_DATA_DIR')
    if not value:
        raise RuntimeError(
            'BEHAV_DATA_DIR is not set. Add your folder on ceph to ~/.bashrc (cluster) or ~/.zshrc (Mac):\n'
            '    export BEHAV_DATA_DIR=/ceph/akrami/<Name>/Head_Fixed_Behavior/Data\n'
            'See START_HERE.md, "Your own data folder".')
    return Path(value)


def cluster_data_root() -> Path:
    """The data root on the cluster: $BEHAV_DATA_DIR/Processed."""
    return ceph_root() / 'Processed'


def data_root(repo_root: Path = None) -> Path:
    """Machine-aware data root (inputs: snapshot, SBI networks, synthetic cohorts).
    Cluster: $BEHAV_DATA_DIR/Processed. Laptop: <repo>/../../data. SC_DATA_ROOT overrides both."""
    env = os.environ.get('SC_DATA_ROOT')
    if env:
        return Path(env)
    if _on_cluster():
        return cluster_data_root()
    return (repo_root or REPO_ROOT).parent.parent / 'data'


def cohort_path(cohort: str) -> Path:
    """Synthetic-cohort pickle (an input to GS and SBI validation runs)."""
    return data_root() / 'synthetic_cohorts' / f'{cohort}.pkl'


# ── the daily tracker's raw data ─────────────────────────────────────────────

def raw_source() -> tuple:
    """Where the daily tracker reads the raw session folders: ('local', Path) or ('ssh', remote).

    'local' when $BEHAV_DATA_DIR/Raw exists on this machine: the cluster, or a rig PC with BEHAV_DATA_DIR
    set to its mapped drive (e.g. Z:\\Serkan\\Head_Fixed_Behavior\\Data). Otherwise 'ssh', with the
    remote ('user@<login node>', '/ceph/.../Raw') reached through the SWC gateway as $SWC_USER, as sc-fetch
    does; the login node is swc_node().
    """
    raw = ceph_root() / 'Raw'
    if raw.is_dir():
        return 'local', raw
    user = os.environ.get('SWC_USER')
    if not user:
        raise RuntimeError(
            f'{raw} is not on this machine and SWC_USER is not set, so the session files cannot be reached.\n'
            'On a laptop add your SWC username to ~/.zshrc:  export SWC_USER=<your SWC username>\n'
            'On a rig PC set BEHAV_DATA_DIR to the data folder on the mapped drive.\n'
            'See START_HERE.md, "Daily behaviour tracker".')
    return 'ssh', (f'{user}@{swc_node()}', raw.as_posix())


def raw_mirror_dir() -> Path:
    """The daily tracker's local copy of the raw session files: <data root>/raw_mirror/Raw."""
    return data_root() / 'raw_mirror' / 'Raw'


def daily_tracker_dir() -> Path:
    """Where the daily tracker writes its reports: <results root>/daily_tracker."""
    return results_root() / 'daily_tracker'


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

REPORTS = ('opto_contrasts', 'switch_adaptation', 'light_artefact', 'model_identification',
           'feature_selection')


def results_root() -> Path:
    """Where results go. Cluster: $BEHAV_DATA_DIR/Processed/results. Laptop: <repo>/results.
    SC_RESULTS_ROOT overrides both."""
    env = os.environ.get('SC_RESULTS_ROOT')
    if env:
        return Path(env)
    if _on_cluster():
        return cluster_data_root() / 'results'
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


def write_sessions_table(run: Path, distribution: str, table) -> Path:
    """``<run>/sessions/<distribution>.csv`` — which sessions each animal's fits saw, and which animals
    were left out (from ``cohort.select_animals``). Written by every model-ID runner; identical content
    from each, so a rewrite is harmless."""
    out = Path(run) / 'sessions' / f'{distribution}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return out


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

# One config file on every machine; its raw-data path comes from BEHAV_DATA_DIR. --config overrides.
DEFAULT_CONFIG = REPO_ROOT / 'config.yaml'


def load_project_config(config_path=None):
    """Load ProjectConfig from config.yaml (or ``config_path``)."""
    from behav_utils.config.schema import load_config
    return load_config(str(config_path if config_path is not None else DEFAULT_CONFIG))

