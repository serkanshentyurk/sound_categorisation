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
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Tuple, Union

# 2: trials whose sound was never sent are removed at export (remove_silent_trials). Snapshots of
# version 1 still contain them, so load_snapshot refuses them and asks for a re-export.
# 3: exported with behav_utils 1.0. A version-2 snapshot holds behav_utils 0.10 objects, which unpickle
# under 1.0 without their new fields (outcome codes, the recorded columns, source), so it is refused too.
SNAPSHOT_FORMAT_VERSION = 3
SNAPSHOT_FILENAME = 'sound_cat_snapshot.pkl'


def snapshot_dir(repo_root: Path | None = None) -> Path:
    """
    The snapshot directory, ``<data root>/behaviour/snapshots``.

    The data root is ``paths.data_root()``: ``SC_DATA_ROOT`` if set, the ceph Processed folder on the
    cluster, else ``<repo>/../../data``. (Up to 0.7.0 this function picked ceph whenever the machine
    ran Linux, so on any other Linux machine it ignored ``SC_DATA_ROOT``.)
    """
    from sound_categorisation.data.paths import data_root
    return data_root(repo_root) / 'behaviour' / 'snapshots'


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
    merged the way behav_utils' csv_folders reader does it (1.0): empty files ignored as if absent, files in
    name order, each file's last row dropped (``drop_last_row``), and files left with fewer than
    ``min_trials_per_file`` rows skipped (in 1.0 also when the session has only one file). The session's
    ``source`` is its folder (or its one file, with ``merge_files: false``).

    Returns ``(checked, stimulus)``: float arrays, ``checked`` being 1 sent / 0 not sent / NaN cannot tell;
    ``(None, None)`` if the tables cannot be read.
    """
    import glob

    import numpy as np
    import pandas as pd
    from behav_utils.data import is_empty_csv

    from sound_categorisation.data.stimulus_check import check_stimulus_delivery

    reader = config.reader
    source = Path(session.source)
    found = [str(source)] if source.is_file() else sorted(glob.glob(str(source / reader.behaviour_file)))
    files = [f for f in found if not is_empty_csv(f)]
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
        if reader.drop_last_row and len(df) > 1:
            df, check = df.iloc[:-1], check[:-1]
        if len(df) < reader.min_trials_per_file:
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
    its loaded trials (same number of rows, same Stim_Relative row for row, compared in CSV units:
    ``trials.stimulus_raw``) is left as it is, with a warning.

    Returns the counts export_snapshot stores in the snapshot metadata: ``removed`` and ``unchecked``
    totals, ``per_session`` counts and the ``not_aligned`` session ids.
    """
    import numpy as np
    from behav_utils.data import filter_trial_data

    prev_values = ('prev_stimulus', 'prev_choice', 'prev_correct', 'prev_category', 'prev_reaction_time',
                   'prev_opto_on')
    per_session, not_aligned = {}, []
    for animal in experiment.animals.values():
        for session in animal.sessions:
            if not session.source:
                continue
            t = session.trials
            checked, stimulus = _rows_as_loaded(session, config)
            # CSV units: since behav_utils 0.8 ``stimulus`` is normalised at load, ``stimulus_raw`` is the column
            raw = getattr(t, 'stimulus_raw', None)
            raw = t.stimulus if raw is None or len(raw) != t.n_trials else raw
            if (checked is None or len(checked) != t.n_trials
                    or not np.allclose(stimulus, np.asarray(raw, dtype=float), rtol=0, atol=1e-12, equal_nan=True)):
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
    from behav_utils.config import load_config
    from behav_utils.data import load_experiment

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
        'exported_at': datetime.now(UTC).isoformat(),
        'config_path': str(config_path.resolve()),
        'config_hash': _config_hash(config_path),
        'data_dir': str(config.reader.data_dir),
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
        raise ValueError(f"Snapshot format {meta.get('format_version', 0)}, this code reads "
                         f"{SNAPSHOT_FORMAT_VERSION} (behav_utils {meta.get('behav_utils_version', '?')} wrote it). "
                         f'Re-export with sc-export-snapshot.')

    experiment = snapshot['experiment']

    # Staleness warning
    exported_at = meta.get('exported_at', '')
    if exported_at:
        try:
            export_time = datetime.fromisoformat(exported_at)
            age_hours = (datetime.now(UTC) - export_time).total_seconds() / 3600
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
                    'config.yaml has changed since the snapshot was exported; re-export it if column '
                    'mappings or session_types changed (session types are set when the data are loaded).',
                    stacklevel=2,
                )
            # Presets live in the config, not the pickle: register them for select_sessions. Session types
            # were applied when the experiment was loaded for export (1.0 keeps that step private).
            from behav_utils.config import load_config
            from behav_utils.data import register_presets_from_config
            cfg = load_config(config_path)
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
    from behav_utils.config import load_config
    from behav_utils.data import load_experiment

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
