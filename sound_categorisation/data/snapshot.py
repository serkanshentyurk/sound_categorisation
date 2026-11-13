"""
Experiment snapshot: export and load

Usage (export, on the cluster, where the raw data are):
    sc-export-snapshot

Usage (load, in notebooks):
    from sound_categorisation.data.snapshot import load_snapshot
    experiment, meta = load_snapshot(PATH_SNAPSHOT)

The snapshot is a behav_utils store (1.0): a folder of Parquet tables and JSON that reloads exactly. It keeps
every loaded trial, each with its stimulus check in ``trials.extra['stimulus_check']``, and loading removes
the trials whose sound was never sent (data/stimulus_check.py; apply_stimulus_check). Removal waits until
loading because a store rebuilds the lag-1 view from the trials it holds: removed at export, the trial after a
silent one would get the trial before the silent one as its predecessor instead of no history.

The export also reports where behav_utils 1.0 can give another number than 0.10 did on these data
(changes_since_0_10); all zero means no statistic moved for those reasons.
"""

import hashlib
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Tuple, Union

# 2: trials whose sound was never sent are removed at export (a pickle).
# 3: the same pickle, written with behav_utils 1.0.
# 4: a behav_utils store: every trial kept with its stimulus check; silent trials are removed when loading.
#    Pickles (formats 2 and 3) are refused: export again.
SNAPSHOT_FORMAT_VERSION = 4
SNAPSHOT_NAME = 'sound_cat_store'


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
    """The snapshot store: ``<data root>/behaviour/snapshots/sound_cat_store`` (a folder)."""
    return snapshot_dir(repo_root) / SNAPSHOT_NAME


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
    """Number of sessions per animal."""
    return {aid: len(animal.sessions) for aid, animal in sorted(experiment.animals.items())}


def _refuse_pickle(path: Path) -> None:
    if path.suffix == '.pkl' or path.is_file():
        raise ValueError(f'{path} is a pickle snapshot, which this code no longer reads (behav_utils 1.0 '
                         f'cannot open it reliably). Export the store with sc-export-snapshot (it goes to '
                         f'<data root>/behaviour/snapshots/{SNAPSHOT_NAME}), or fetch it with '
                         f'sc-fetch snapshot.')


def read_snapshot_metadata(path: Union[str, Path]) -> Dict:
    """What export_snapshot recorded, read from the store's store.json without loading any trials."""
    path = Path(path)
    _refuse_pickle(path)
    info = path / 'store.json'
    if not info.is_file():
        raise FileNotFoundError(f'no snapshot store at {path}: run sc-export-snapshot, or sc-fetch snapshot')
    return dict(json.loads(info.read_text()).get('experiment_metadata') or {})


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


def attach_stimulus_check(experiment, config, verbose: bool = True) -> Dict:
    """
    Check every trial's sound (data/stimulus_check.py) and keep the result in
    ``trials.extra['stimulus_check']``: 1 sent, 0 never sent, NaN cannot tell. Removes nothing; see
    apply_stimulus_check. A session whose files do not line up with its loaded trials (same number of rows,
    same Stim_Relative row for row, compared in CSV units: ``trials.stimulus_raw``) gets no check, with a
    warning, and so keeps every trial.

    Returns ``silent`` and ``unchecked`` totals, ``per_session`` counts and the ``not_aligned`` session ids.
    """
    import numpy as np

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
            t.extra['stimulus_check'] = checked
            per_session[session.session_id] = {'silent': int(np.sum(checked == 0)),
                                               'unchecked': int(np.isnan(checked).sum())}
    summary = {
        'silent': sum(c['silent'] for c in per_session.values()),
        'unchecked': sum(c['unchecked'] for c in per_session.values()),
        'per_session': per_session,
        'not_aligned': not_aligned,
    }
    if verbose:
        print(f"Sound check: {summary['silent']} trials whose sound was never sent (removed when loading); "
              f"{summary['unchecked']} could not be checked (kept)"
              + (f'; {len(not_aligned)} sessions could not be lined up (left as loaded)'
                 if not_aligned else ''))
    return summary


def apply_stimulus_check(experiment, verbose: bool = False) -> Dict:
    """
    Remove, in place, every trial whose stored stimulus check says its sound was never sent.

    The trial after a removed one loses its lag-1 history (``prev_has_prev`` False, prev values NaN): its
    predecessor's Stim_Relative was left over from an earlier trial and nothing was heard. A removed trial
    that was also an abort was never anyone's predecessor (aborts are skipped), so it changes no history.
    Sessions without a stored check are left as they are. Returns the ``removed`` total and ``per_session``
    counts.
    """
    import numpy as np
    from behav_utils.data import filter_trial_data

    prev_values = ('prev_stimulus', 'prev_choice', 'prev_correct', 'prev_category', 'prev_reaction_time',
                   'prev_opto_on')
    per_session = {}
    for animal in experiment.animals.values():
        for session in animal.sessions:
            t = session.trials
            checked = (t.extra or {}).get('stimulus_check')
            if checked is None or len(checked) != t.n_trials:
                continue
            silent = np.asarray(checked, dtype=float) == 0
            if not silent.any():
                continue
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
            session.trials = filter_trial_data(t, ~silent, clear_flags=False)
            per_session[session.session_id] = int(silent.sum())
    summary = {'removed': sum(per_session.values()), 'per_session': per_session}
    if verbose:
        print(f"Sound check: removed {summary['removed']} trials whose sound was never sent")
    return summary


def remove_silent_trials(experiment, config, verbose: bool = True) -> Dict:
    """
    Check every trial's sound and remove, in place, the trials whose sound was never sent: attach then apply,
    what a fresh CSV load needs to match the snapshot. Every remaining trial carries its check in
    ``trials.extra['stimulus_check']`` (1 sent, NaN cannot tell).

    Returns ``removed`` and ``unchecked`` totals, ``per_session`` counts and the ``not_aligned`` session ids.
    """
    attached = attach_stimulus_check(experiment, config, verbose=False)
    applied = apply_stimulus_check(experiment)
    summary = {
        'removed': applied['removed'],
        'unchecked': attached['unchecked'],
        'per_session': {sid: {'removed': c['silent'], 'unchecked': c['unchecked']}
                        for sid, c in attached['per_session'].items()},
        'not_aligned': attached['not_aligned'],
    }
    if verbose:
        not_aligned = summary['not_aligned']
        print(f"Sound check: removed {summary['removed']} trials whose sound was never sent; "
              f"{summary['unchecked']} could not be checked (kept)"
              + (f'; {len(not_aligned)} sessions could not be lined up (left as loaded)'
                 if not_aligned else ''))
    return summary


# ── where behav_utils 1.0 can differ from 0.10 ───────────────────────────────

_CHANGES = {
    'not_loaded': 'sessions found but not loaded (min_trials_per_file now drops a short only file)',
    'boundary': 'trials with the stimulus on the boundary (scored by the task)',
    'nan_stimulus': 'trials with a NaN stimulus (no category, no score)',
    'after_unscored': 'trials after a trial without a score (prev_correct NaN; was 0 after a no-response)',
    'recorded_disagrees': "trials where the rig's Correct disagrees with the task's score",
}


def _recorded_disagreements(t) -> int:
    """Scored trials whose recorded correct column says otherwise (0 when none is mapped)."""
    import numpy as np
    import pandas as pd
    from behav_utils.tasks import Outcome

    recorded = getattr(t, 'correct_recorded', None)
    if recorded is None or len(recorded) != t.n_trials:
        return 0
    recorded = pd.to_numeric(pd.Series(np.asarray(recorded, dtype=object)), errors='coerce')
    recorded = recorded.to_numpy(dtype=float)
    codes = np.asarray(t.outcome)
    scored = (codes == Outcome.CORRECT) | (codes == Outcome.ERROR)
    known = scored & ~np.isnan(recorded)
    return int(np.sum(known & ((recorded == 1) != np.asarray(t.correct, dtype=bool))))


def changes_since_0_10(experiment, config) -> Dict:
    """
    Where behav_utils 1.0 can give another number than 0.10 did on these data (the changes its changelog marks
    with a triangle): sessions the reader finds but loading drops, trials with the stimulus on the boundary or
    NaN, trials after a trial without a score, and trials where the rig's recorded score disagrees with the
    task's. All zero means no statistic can have moved for these reasons.

    Returns the totals (``not_loaded`` lists session ids) and ``per_session`` counts where any is non-zero.
    """
    import numpy as np
    from behav_utils.data import discover_sessions

    loaded = {s.session_id for a in experiment.animals.values() for s in a.sessions}
    try:
        found = discover_sessions(config, peek=False)
        found = found[~found['ignored'].astype(bool)]
        not_loaded = sorted(set(found['session_id']) - loaded)
    except Exception as e:                          # listing is a courtesy; the export does not depend on it
        not_loaded = [f'(could not list the sessions: {type(e).__name__}: {e})']
    totals = {'boundary': 0, 'nan_stimulus': 0, 'after_unscored': 0, 'recorded_disagrees': 0}
    per_session = {}
    for animal in experiment.animals.values():
        for s in animal.sessions:
            t = s.trials
            stimulus = np.asarray(t.stimulus, dtype=float)
            kept = ~np.asarray(t.abort, dtype=bool)
            boundary = float(getattr(s.task, 'boundary', 0.0) or 0.0)
            counts = {
                'boundary': int(np.sum(kept & (stimulus == boundary))),
                'nan_stimulus': int(np.sum(kept & np.isnan(stimulus))),
                'after_unscored': int(np.sum(np.asarray(t.prev_has_prev, dtype=bool)
                                             & np.isnan(np.asarray(t.prev_correct, dtype=float)))),
                'recorded_disagrees': _recorded_disagreements(t),
            }
            for k, v in counts.items():
                totals[k] += v
            if any(counts.values()):
                per_session[s.session_id] = {k: v for k, v in counts.items() if v}
    return {'not_loaded': not_loaded, **totals, 'per_session': per_session}


def _print_changes(changes: Dict) -> None:
    print('Where behav_utils 1.0 can differ from 0.10 on these data (0 = it cannot have moved a number):')
    for key, label in _CHANGES.items():
        value = changes[key]
        n = len(value) if isinstance(value, list) else value
        extra = ''
        if isinstance(value, list) and value:
            extra = '  ' + ', '.join(value[:8]) + (' …' if len(value) > 8 else '')
        print(f'  {n:>7}  {label}{extra}')


# ── export and load ──────────────────────────────────────────────────────────

def export_snapshot(
    config_path: Union[str, Path],
    output_path: Union[str, Path] | None = None,
    verbose: bool = True,
) -> Path:
    """Load every session from the raw CSVs via the config, attach the stimulus check and save the store
    (replacing an existing one). Returns the store's path."""
    from behav_utils.config import load_config
    from behav_utils.data import load_experiment

    config_path = Path(config_path)
    output_path = Path(output_path) if output_path is not None else default_output_path()
    _refuse_pickle(output_path)
    if verbose:
        print(f'Loading from config: {config_path}')
    config = load_config(str(config_path))
    experiment = load_experiment(config)
    stimulus_check = attach_stimulus_check(experiment, config, verbose=verbose)
    changes = changes_since_0_10(experiment, config)
    if verbose:
        _print_changes(changes)

    session_counts = _session_summary(experiment)
    total_trials = sum(s.n_trials for a in experiment.animals.values() for s in a.sessions)
    meta = {
        'format_version': SNAPSHOT_FORMAT_VERSION,
        'exported_at': datetime.now(UTC).isoformat(),
        'config_path': str(config_path.resolve()),
        'config_hash': _config_hash(config_path),
        'data_dir': str(config.reader.data_dir),
        'behav_utils_version': _get_behav_utils_version(),
        'n_animals': experiment.n_animals,
        'n_sessions_total': sum(session_counts.values()),
        'n_trials_total': total_trials,
        'session_counts': session_counts,
        'animal_ids': sorted(experiment.animals.keys()),
        'stimulus_check': stimulus_check,
        'changes_since_0_10': changes,
    }
    experiment.metadata.update(meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    experiment.save(output_path, overwrite=True)

    if verbose:
        size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1e6
        print('Exported snapshot:')
        print(f'  Animals:  {meta["n_animals"]}')
        print(f'  Sessions: {meta["n_sessions_total"]}')
        print(f'  Trials:   {total_trials} (silent ones included; removed when loading)')
        print(f'  Config:   {meta["config_hash"]}')
        print(f'  Size:     {size_mb:.1f} MB')
        print(f'  Saved to: {output_path}')
    return output_path


def load_snapshot(
    path: Union[str, Path],
    config_path: Union[str, Path] | None = None,
    warn_age_hours: float = 72,
) -> Tuple:
    """Load the snapshot store, with version, age and config checks, and remove the trials whose sound was
    never sent. Returns ``(experiment, meta)``; ``meta['removed_at_load']`` counts the removed trials."""
    from behav_utils.data import ExperimentData

    path = Path(path)
    meta = read_snapshot_metadata(path)
    if meta.get('format_version', 0) != SNAPSHOT_FORMAT_VERSION:
        raise ValueError(f"Snapshot format {meta.get('format_version', 0)}, this code reads "
                         f"{SNAPSHOT_FORMAT_VERSION} (behav_utils {meta.get('behav_utils_version', '?')} wrote it). "
                         f'Re-export with sc-export-snapshot.')
    experiment = ExperimentData.load(path)

    exported_at = meta.get('exported_at', '')
    if exported_at:
        try:
            age_hours = (datetime.now(UTC) - datetime.fromisoformat(exported_at)).total_seconds() / 3600
            if age_hours > warn_age_hours:
                warnings.warn(f'Snapshot is {age_hours:.0f}h old (exported {exported_at}). '
                              f'Re-export if new sessions have been collected.', stacklevel=2)
        except (ValueError, TypeError):
            pass

    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            export_hash = meta.get('config_hash', '')
            if export_hash and _config_hash(config_path) != export_hash:
                warnings.warn(
                    'config.yaml has changed since the snapshot was exported; re-export it if column '
                    'mappings or session_types changed (session types are set when the data are loaded).',
                    stacklevel=2,
                )
            # Presets live in the config, not the store: register them for select_sessions. Session types
            # were applied when the experiment was loaded for export (1.0 keeps that step private).
            from behav_utils.config import load_config
            from behav_utils.data import register_presets_from_config
            cfg = load_config(config_path)
            if cfg.session_presets:
                register_presets_from_config({'session_presets': cfg.session_presets})

    meta['removed_at_load'] = apply_stimulus_check(experiment)
    print(
        f'Loaded snapshot: {meta.get("n_animals", "?")} animals, '
        f'{meta.get("n_sessions_total", "?")} sessions '
        f'(exported {exported_at[:10] if exported_at else "unknown"}); '
        f'{meta["removed_at_load"]["removed"]} trials whose sound was never sent removed'
    )
    return experiment, meta


def check_staleness(
    snapshot_path: Union[str, Path],
    config_path: Union[str, Path],
) -> Dict:
    """Compare the snapshot's sessions per animal with what the raw data hold now."""
    from behav_utils.config import load_config
    from behav_utils.data import load_experiment

    snap_counts = read_snapshot_metadata(snapshot_path)['session_counts']
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
