"""
Experiment Snapshot Export / Import

Usage (export — on cluster):
    sc-export-snapshot

Usage (load — in notebooks):
    from sound_categorisation.data.snapshot import load_snapshot
    experiment, meta = load_snapshot(PATH_SNAPSHOT)

Export removes trials whose sound was never sent (data/stimulus_check.py); see remove_silent_trials.
"""

import hashlib
import pickle
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Union

# 2: trials whose sound was never sent are removed at export (remove_silent_trials). Snapshots of
# version 1 still contain them, so load_snapshot refuses them and asks for a re-export.
SNAPSHOT_FORMAT_VERSION = 2
SNAPSHOT_FILENAME = 'sound_cat_snapshot.pkl'

# Cluster path — fixed for this project's SWC/ceph layout. Off-cluster,
# snapshot_dir() falls back to a path derived from the repo location, so
# this constant is only ever used on Linux/cluster.
_CLUSTER_SNAPSHOT_DIR = Path(
    '/ceph/akrami/Serkan/Head_Fixed_Behavior/Data/Processed/behaviour/snapshots'
)

def snapshot_dir(repo_root: Path | None = None) -> Path:
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
            from sound_categorisation.data.paths import REPO_ROOT
            repo_root = REPO_ROOT
        return repo_root.parent.parent / 'data' / 'behaviour' / 'snapshots'


def default_output_path(repo_root: Path | None = None) -> Path:
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


def _rows_as_loaded(session, config):
    """
    The stimulus check and the stimulus column for the trial tables behind one loaded session, trimmed and
    merged the way behav_utils' loader (v0.7) does it: files in sorted order, each file's last row dropped
    (``drop_last_row``), and, only when a session has several files, files left with fewer than
    ``min_trials_per_file`` rows skipped.

    Returns ``(checked, stimulus)``: float arrays, ``checked`` being 1 sent / 0 not sent / NaN cannot tell;
    ``(None, None)`` if the tables cannot be read.
    """
    import glob

    import numpy as np
    import pandas as pd

    from sound_categorisation.data.stimulus_check import check_stimulus_delivery

    fs = config.file_structure
    files = sorted(glob.glob(str(Path(session.csv_path).parent / fs.behaviour_file)))
    stim_col = config.columns['stimulus'].csv_name
    checked, stimulus = [], []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except (pd.errors.ParserError, UnicodeDecodeError, OSError):
            continue
        if stim_col not in df.columns:
            return None, None
        check = check_stimulus_delivery(f, df).astype('Float64').to_numpy(dtype=float, na_value=np.nan)
        if fs.drop_last_row and len(df) > 1:
            df, check = df.iloc[:-1], check[:-1]
        if len(files) > 1 and len(df) < fs.min_trials_per_file:
            continue
        checked.append(check)
        stimulus.append(pd.to_numeric(df[stim_col], errors='coerce').to_numpy(dtype=float))
    if not checked:
        return None, None
    return np.concatenate(checked), np.concatenate(stimulus)


def remove_silent_trials(experiment, config, verbose: bool = True) -> Dict:
    """
    Remove, in place, every trial whose sound was never sent (data/stimulus_check.py).

    The trial after a removed one loses its lag-1 history (``prev_has_prev`` False, prev values NaN): its
    predecessor's Stim_Relative was left over from an earlier trial and nothing was heard. A removed trial
    that was also an abort was never anyone's predecessor (aborts are skipped), so it changes no history.
    Trials that cannot be checked are kept; every remaining trial carries the result in
    ``trials.extra['stimulus_check']`` (1 sent, NaN cannot tell). A session whose files do not line up with
    its loaded trials (same number of rows, same Stim_Relative row for row) is left as it is, with a warning.

    Returns the counts export_snapshot stores in the snapshot metadata: ``removed`` and ``unchecked``
    totals, ``per_session`` counts and the ``not_aligned`` session ids.
    """
    import numpy as np
    from behav_utils.data.ops.filtering import filter_trial_data

    prev_values = ('prev_stimulus', 'prev_choice', 'prev_correct', 'prev_category', 'prev_reaction_time',
                   'prev_opto_on')
    per_session, not_aligned = {}, []
    for animal in experiment.animals.values():
        for session in animal.sessions:
            if session.csv_path is None:
                continue
            t = session.trials
            checked, stimulus = _rows_as_loaded(session, config)
            if (checked is None or len(checked) != t.n_trials
                    or not np.allclose(stimulus, t.stimulus.astype(float), rtol=0, atol=1e-12, equal_nan=True)):
                warnings.warn(f'{session.session_id}: trial tables do not line up with the loaded trials; '
                              f'no trials removed', stacklevel=2)
                not_aligned.append(session.session_id)
                continue
            silent = checked == 0
            # A trial's predecessor is the latest earlier non-abort trial (behav_utils' lag-1 rule).
            lose_history = np.zeros(t.n_trials, dtype=bool)
            last = -1
            for i in range(t.n_trials):
                if t.abort[i]:
                    continue
                lose_history[i] = last >= 0 and silent[last]
                last = i
            if lose_history.any():
                t.prev_has_prev = np.where(lose_history, False, t.prev_has_prev)
                for name in prev_values:
                    setattr(t, name, np.where(lose_history, np.nan, getattr(t, name)))
            t.extra['stimulus_check'] = checked
            if silent.any():
                session.trials = filter_trial_data(t, ~silent, clear_flags=False)
            per_session[session.session_id] = {'removed': int(silent.sum()),
                                               'unchecked': int(np.isnan(checked).sum())}
    summary = {
        'removed': sum(c['removed'] for c in per_session.values()),
        'unchecked': sum(c['unchecked'] for c in per_session.values()),
        'per_session': per_session,
        'not_aligned': not_aligned,
    }
    if verbose:
        print(f"Sound check: removed {summary['removed']} trials whose sound was never sent; "
              f"{summary['unchecked']} could not be checked (kept)"
              + (f"; {len(not_aligned)} sessions could not be lined up (left as loaded)" if not_aligned else ''))
    return summary


def export_snapshot(
    config_path: Union[str, Path],
    output_path: Union[str, Path] | None = None,
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
    stimulus_check = remove_silent_trials(experiment, config, verbose=verbose)

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
        'stimulus_check': stimulus_check,
    }

    snapshot = {'experiment': experiment, 'metadata': meta}
    with open(output_path, 'wb') as f:
        pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_path.stat().st_size / 1e6
    if verbose:
        print('Exported snapshot:')
        print(f'  Animals:  {meta["n_animals"]}')
        print(f'  Sessions: {total_sessions}')
        print(f'  Trials:   {total_trials}')
        print(f'  Config:   {meta["config_hash"]}')
        print(f'  Size:     {size_mb:.1f} MB')
        print(f'  Saved to: {output_path}')

    return output_path


def load_snapshot(
    path: Union[str, Path],
    config_path: Union[str, Path] | None = None,
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
            'Not a valid snapshot file. Re-export with '
            'sc-export-snapshot.'
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
                    'Config has changed since snapshot was exported. '
                    'Re-export if column mappings changed.',
                    stacklevel=2,
                )
            # Presets and session types live in the config, not the pickle: re-apply both
            # so a snapshot behaves exactly like a fresh load.
            from behav_utils.config.schema import load_config
            from behav_utils.data.loading import apply_session_type
            from behav_utils.data.ops.selection import register_presets_from_config
            cfg = load_config(config_path)
            for type_name, mapping in cfg.session_types.items():
                apply_session_type(experiment, mapping, type_name)
            if cfg.session_presets:
                register_presets_from_config({'session_presets': cfg.session_presets})

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

    print('Staleness check:')
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
        print('  → Snapshot is up to date.')

    return comparison
