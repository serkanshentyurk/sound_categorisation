"""
Was each trial's sound actually sent? Read from a rig session's own Bonsai logs.

Specific to this rig's Bonsai output. Each run writes, next to its trial table (``Trial_Summary*.csv``), an
epoch log (``Trial_Epochs*.csv``: ``Epoch``, ``Time``) and an event log (``Long_Form_Timestamps*.csv``:
``Label``, ``Data``, ``Time``). Every trial starts with the ``Sound`` epoch, and the workflow logs a ``Sound``
event each time it triggers the sound card. A trial whose span holds no such event had no sound sent: in
Asym_Left / Asym_Right sessions run before 5 Oct 2026 this happened on about 6 % of trials (all on the hard
side), and the trial's row then carries the previous trial's ``Stim_Relative``.

Self-contained (numpy and pandas only), so the file can be used on its own:

    import pandas as pd
    from stimulus_check import check_stimulus_delivery

    df = pd.read_csv(path_to_trial_summary)
    df['sound_sent'] = check_stimulus_delivery(path_to_trial_summary)
"""

import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

__all__ = ['check_stimulus_delivery']

EPOCHS_FILE = 'Trial_Epochs*.csv'
EVENTS_FILE = 'Long_Form_Timestamps*.csv'
TRIAL_START = 'Sound'               # epoch that starts every trial
SOUND_EVENT = 'Sound'               # event logged when the sound card is triggered
TRIAL_END_COLUMN = 'Trial_End_Time'
MAX_STAMP_GAP_S = 5.0               # one run's files can be stamped a second or two apart
TOLERANCE_S = 0.002                 # an event may be logged this long before its trial's start epoch

_STAMP = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})')


def _stamp(path: Path) -> datetime | None:
    m = _STAMP.search(path.name)
    return datetime.strptime(m.group(1), '%Y-%m-%dT%H_%M_%S') if m else None


def _sibling(trial_summary: Path, pattern: str) -> Path | None:
    """The file matching ``pattern`` in the same folder from the same run: the closest file-name stamp
    within MAX_STAMP_GAP_S (a folder can hold several runs, e.g. after a restart)."""
    candidates = sorted(trial_summary.parent.glob(pattern))
    t0 = _stamp(trial_summary)
    if not candidates:
        return None
    if t0 is None:
        return candidates[0] if len(candidates) == 1 else None
    timed = [(abs((s - t0).total_seconds()), c) for c in candidates if (s := _stamp(c)) is not None]
    if not timed:
        return None
    gap, best = min(timed, key=lambda x: x[0])
    return best if gap <= MAX_STAMP_GAP_S else None


def _read_log(path: Path | None, needed: tuple) -> pd.DataFrame | None:
    """A log with numeric ``Time``, or None if absent, empty or unreadable. Logs of runs that ended with
    Bonsai being killed stop mid-line; malformed lines are skipped."""
    if path is None:
        return None
    try:
        df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError):
        return None
    if not set(needed) <= set(df.columns):
        return None
    df = df.assign(Time=pd.to_numeric(df['Time'], errors='coerce'))
    return df[df['Time'].notna()]


def check_stimulus_delivery(trial_summary_csv: Union[str, Path],
                            trial_summary: pd.DataFrame | None = None) -> pd.Series:
    """
    Whether each trial of one run had its sound sent.

    Row ``k`` of the trial table is the ``k``-th ``Sound`` epoch; the trial spans from that epoch to the next
    one (the last row: to its ``Trial_End_Time``).

    Args:
        trial_summary_csv: The run's ``Trial_Summary*.csv``; its folder is searched for the run's logs.
        trial_summary:     That table already read, so the result lines up with exactly those rows
                           (read with ``pd.read_csv`` if omitted).

    Returns:
        A nullable-boolean Series on the table's index, named ``stimulus_delivered``:
            True   a ``Sound`` event was logged during the trial;
            False  the logs cover the whole trial and hold none: no sound was sent;
            <NA>   cannot tell: a log is missing, a log ends before the trial does (killed runs lose the end
                   of every file), the stage does not start trials with ``Sound``, or the rows cannot be
                   lined up with the epochs.
    """
    path = Path(trial_summary_csv)
    if trial_summary is None:
        trial_summary = pd.read_csv(path, low_memory=False)
    n = len(trial_summary)
    out = pd.Series(pd.array([pd.NA] * n, dtype='boolean'), index=trial_summary.index, name='stimulus_delivered')
    if n == 0:
        return out

    epochs = _read_log(_sibling(path, EPOCHS_FILE), ('Epoch', 'Time'))
    events = _read_log(_sibling(path, EVENTS_FILE), ('Label', 'Time'))
    if epochs is None or events is None:
        return out

    # Logs only ever lose their end, each file at a different point, so extra starts after the last row are
    # trials whose rows were lost and rows after the last start are trials whose epochs were lost; neither
    # shifts the alignment. A shift anywhere else shows up in the end-time check below.
    starts = epochs.loc[epochs['Epoch'] == TRIAL_START, 'Time'].to_numpy(dtype=float)
    if len(starts) == 0:
        return out

    end_time = np.full(n, np.nan)
    if TRIAL_END_COLUMN in trial_summary.columns:
        end_time = pd.to_numeric(trial_summary[TRIAL_END_COLUMN], errors='coerce').to_numpy(dtype=float)
        k = min(n, len(starts))
        nxt = np.append(starts[1:], np.inf)[:k]          # the next trial's start, or inf for the last one
        ok = np.isnan(end_time[:k]) | ((starts[:k] <= end_time[:k] + TOLERANCE_S)
                                       & (end_time[:k] <= nxt + TOLERANCE_S))
        if not ok.all():
            warnings.warn(f'{path.name}: trial end times do not fall between consecutive trial starts; '
                          f'cannot line the rows up with the epochs', stacklevel=2)
            return out

    sounds = np.sort(events.loc[events['Label'] == SOUND_EVENT, 'Time'].to_numpy(dtype=float))
    covered_until = min(epochs['Time'].max(), events['Time'].max())

    values = [pd.NA] * n
    for k in range(min(n, len(starts))):
        start = starts[k]
        end = starts[k + 1] if k + 1 < len(starts) else end_time[k]
        if np.isnan(end):
            continue
        lo, hi = np.searchsorted(sounds, [start - TOLERANCE_S, end])
        if hi > lo:
            values[k] = True
        elif end <= covered_until:
            values[k] = False
    out[:] = pd.array(values, dtype='boolean')
    return out
