"""
Experiment Snapshot Export / Import

Usage (export — on cluster):
    python scripts/export_snapshot.py

Usage (load — in notebooks):
    from scripts.snapshot import load_snapshot
    experiment, meta = load_snapshot(PATH_SNAPSHOT)
"""

import hashlib
import os
import pickle
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

SNAPSHOT_FORMAT_VERSION = 1
SNAPSHOT_FILENAME = 'sound_cat_snapshot.pkl'

# Cluster path — fixed, will not change
_CLUSTER_SNAPSHOT_DIR = Path(
    '/ceph/akrami/Serkan/Head_Fixed_Behavior/Data/Processed/behaviour/snapshots'
)

def snapshot_dir(repo_root: Optional[Path] = None) -> Path:
    """
    Return the snapshot directory for the current machine.

    Cluster (Linux): /ceph/akrami/.../Processed/snapshots/
    Local (any OS):  some_folder/data/snapshots/
                     (derived from repo at some_folder/repos/sound_categorisation/)
    """
    if platform.system() == 'Linux':
        return _CLUSTER_SNAPSHOT_DIR
    else:
        if repo_root is None:
            from scripts.config import REPO_ROOT
            repo_root = REPO_ROOT
        return repo_root.parent.parent / 'data' / 'behaviour' / 'snapshots'


def default_output_path(repo_root: Optional[Path] = None) -> Path:
    return snapshot_dir(repo_root) / SNAPSHOT_FILENAME


def _config_hash(config_path: Union[str, Path]) -> str:
    content = Path(config_path).read_bytes()
    return hashlib.sha256(content).hexdigest()[:8]


def _get_behav_utils_version() -> str:
    try:
        import behav_utils
        return getattr(behav_utils, '__version__', 'unknown')
    except Exception:
        return 'unknown'


def _session_summary(experiment) -> Dict[str, int]:
    return {
        aid: animal.n_sessions
        for aid, animal in experiment.animals.items()
    }


def export_snapshot(
    config_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Path:
    """Load data from CSV via config, save as versioned snapshot."""
    from behav_utils.config.schema import load_config
    from behav_utils.data.loading import load_experiment

    config_path = Path(config_path)

    if output_path is None:
        output_path = default_output_path()
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f'Loading from config: {config_path}')

    config = load_config(str(config_path))
    experiment = load_experiment(config)

    # Clean ALL config references
    experiment.config = None
    for animal in experiment.animals.values():
        if hasattr(animal, '_config'):
            animal._config = None

    # Build metadata
    session_counts = _session_summary(experiment)
    total_sessions = sum(session_counts.values())
    total_trials = 0
    for animal in experiment.animals.values():
        for session in animal.sessions:
            total_trials += session.n_trials

    meta = {
        'format_version': SNAPSHOT_FORMAT_VERSION,
        'exported_at': datetime.now(timezone.utc).isoformat(),
        'config_path': str(config_path.resolve()),
        'config_hash': _config_hash(config_path),
        'data_dir': str(config.file_structure.data_dir),
        'behav_utils_version': _get_behav_utils_version(),
        'n_animals': experiment.n_animals,
        'n_sessions_total': total_sessions,
        'n_trials_total': total_trials,
        'session_counts': session_counts,
        'animal_ids': sorted(experiment.animals.keys()),
    }

    snapshot = {'experiment': experiment, 'metadata': meta}
    with open(output_path, 'wb') as f:
        pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_path.stat().st_size / 1e6
    if verbose:
        print(f'Exported snapshot:')
        print(f'  Animals:  {meta["n_animals"]}')
        print(f'  Sessions: {total_sessions}')
        print(f'  Trials:   {total_trials}')
        print(f'  Config:   {meta["config_hash"]}')
        print(f'  Size:     {size_mb:.1f} MB')
        print(f'  Saved to: {output_path}')

    return output_path


def load_snapshot(
    path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    warn_age_hours: float = 72,
) -> Tuple:
    """Load a snapshot, with staleness and version checks."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Snapshot not found: {path}')

    with open(path, 'rb') as f:
        try:
            snapshot = pickle.load(f)
        except Exception as e:
            raise ValueError(
                f'Failed to unpickle snapshot. This usually means '
                f'behav_utils data classes changed since export. '
                f'Re-export from CSV.\nOriginal error: {e}'
            ) from e

    if not isinstance(snapshot, dict) or 'experiment' not in snapshot:
        raise ValueError(
            f'Not a valid snapshot file. Re-export with '
            f'scripts/export_snapshot.py.'
        )

    meta = snapshot.get('metadata', {})
    if meta.get('format_version', 0) != SNAPSHOT_FORMAT_VERSION:
        raise ValueError('Snapshot format version mismatch. Re-export.')

    experiment = snapshot['experiment']

    # Staleness warning
    exported_at = meta.get('exported_at', '')
    if exported_at:
        try:
            export_time = datetime.fromisoformat(exported_at)
            age_hours = (datetime.now(timezone.utc) - export_time).total_seconds() / 3600
            if age_hours > warn_age_hours:
                warnings.warn(
                    f'Snapshot is {age_hours:.0f}h old '
                    f'(exported {exported_at}). '
                    f'Re-export if new sessions have been collected.',
                    stacklevel=2,
                )
        except (ValueError, TypeError):
            pass

    # Config hash check
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            current_hash = _config_hash(config_path)
            export_hash = meta.get('config_hash', '')
            if export_hash and current_hash != export_hash:
                warnings.warn(
                    f'Config has changed since snapshot was exported. '
                    f'Re-export if column mappings changed.',
                    stacklevel=2,
                )

    print(
        f'Loaded snapshot: {meta.get("n_animals", "?")} animals, '
        f'{meta.get("n_sessions_total", "?")} sessions '
        f'(exported {exported_at[:10] if exported_at else "unknown"})'
    )
    return experiment, meta


def check_staleness(
    snapshot_path: Union[str, Path],
    config_path: Union[str, Path],
) -> Dict:
    """Compare snapshot contents against current CSV data on disk."""
    from behav_utils.config.schema import load_config
    from behav_utils.data.loading import load_experiment

    with open(snapshot_path, 'rb') as f:
        snapshot = pickle.load(f)
    snap_counts = snapshot['metadata']['session_counts']

    config = load_config(str(config_path))
    experiment = load_experiment(config)
    current_counts = _session_summary(experiment)

    comparison = {}
    all_ids = sorted(set(snap_counts.keys()) | set(current_counts.keys()))
    for aid in all_ids:
        snap_n = snap_counts.get(aid, 0)
        curr_n = current_counts.get(aid, 0)
        comparison[aid] = {
            'snapshot': snap_n, 'current': curr_n,
            'new_sessions': curr_n - snap_n,
        }

    n_new = sum(c['new_sessions'] for c in comparison.values() if c['new_sessions'] > 0)
    new_animals = [aid for aid in current_counts if aid not in snap_counts]

    print(f'Staleness check:')
    print(f'  Snapshot: {sum(snap_counts.values())} sessions '
          f'across {len(snap_counts)} animals')
    print(f'  Current:  {sum(current_counts.values())} sessions '
          f'across {len(current_counts)} animals')
    if n_new > 0:
        print(f'  → {n_new} new sessions detected. Re-export recommended.')
        for aid, c in comparison.items():
            if c['new_sessions'] > 0:
                print(f'    {aid}: {c["snapshot"]} → {c["current"]} '
                      f'(+{c["new_sessions"]})')
    if new_animals:
        print(f'  → {len(new_animals)} new animals: {new_animals}')
    if n_new == 0 and not new_animals:
        print(f'  → Snapshot is up to date.')

    return comparison
