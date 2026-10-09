"""data/stimulus_check + snapshot.remove_silent_trials: finding trials whose sound was never sent, and removing
them at snapshot export.

Driven by small synthetic Bonsai runs in tmp_path (a trial table, an epoch log and an event log per run): silent
trials, logs cut off at the end (killed runs), file stamps a second apart, two runs in one folder (a restart), a
misaligned table.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import yaml
from behav_utils.config.schema import ChoiceMapping, ColumnMapping, FileStructure, ProjectConfig, TaskConfig
from behav_utils.data.loading import load_experiment
from sound_categorisation.data.snapshot import (
    SNAPSHOT_FORMAT_VERSION,
    export_snapshot,
    load_snapshot,
    remove_silent_trials,
)
from sound_categorisation.data.stimulus_check import check_stimulus_delivery

STAMP = '2026-05-23T10_12_44'
SESSION = ('A1', 'SOUND_CAT_A1_2026_5_23')


def _write_run(folder, stamp=STAMP, n=8, silent=(), abort=(), t0=100.0, stim_offset=0.0, summary_stamp=None,
               events_until=None, extra_starts=0, end_times=None):
    """One run: trial k starts at t0 + 5k, ends 4 s later, then its inter-trial interval (an epoch at the end
    and a lick 0.5 s into it). `silent` trials get no Sound event; `events_until` cuts the event log off."""
    folder.mkdir(parents=True, exist_ok=True)
    starts = t0 + 5.0 * np.arange(n + extra_starts)
    ends = starts[:n] + 4.0 if end_times is None else np.asarray(end_times, dtype=float)
    pd.DataFrame({'Trial_Number': range(1, n + 1), 'Stim_Relative': np.linspace(-0.9, 0.9, n) + stim_offset,
                  'Choice': np.tile([0, 1], n)[:n], 'Abort_Trial': [k in abort for k in range(n)],
                  'Trial_End_Time': ends}).to_csv(folder / f'Trial_Summary{summary_stamp or stamp}.csv', index=False)
    epochs = ([('Sound', t) for t in starts] + [('Go_Cue', t + 0.3) for t in starts]
              + [('Response_Window', t + 0.6) for t in starts] + [('Inter_Trial_Interval', t + 4.0) for t in starts])
    pd.DataFrame(epochs, columns=['Epoch', 'Time']).sort_values('Time').to_csv(
        folder / f'Trial_Epochs{stamp}.csv', index=False)
    events = [('Sound', 0.0, t0 - 2.0)]                                       # value logged at start-up
    events += [('Sound', 0.5, t + 0.005) for k, t in enumerate(starts) if k not in silent]
    events += [('Lick', 1.0, t + 1.0) for t in starts] + [('Lick', 1.0, t + 4.5) for t in starts]
    ev = pd.DataFrame(events, columns=['Label', 'Data', 'Time']).sort_values('Time')
    if events_until is not None:
        ev = ev[ev.Time <= events_until]
    ev.to_csv(folder / f'Long_Form_Timestamps{stamp}.csv', index=False)
    return folder / f'Trial_Summary{summary_stamp or stamp}.csv'


def _as_list(s):
    return [None if pd.isna(v) else bool(v) for v in s]


# ── check_stimulus_delivery ──────────────────────────────────────────────────
def test_all_sent(tmp_path):
    assert _as_list(check_stimulus_delivery(_write_run(tmp_path))) == [True] * 8


def test_silent_trials_are_false(tmp_path):
    s = check_stimulus_delivery(_write_run(tmp_path, silent=(0, 5)))      # 0: the start-up event does not count
    assert _as_list(s) == [False, True, True, True, True, False, True, True]
    assert s.dtype == 'boolean' and s.name == 'stimulus_delivered'


def test_cut_off_log_gives_na_after_the_cut(tmp_path):
    # trial k spans [100 + 5k, 105 + 5k); the event log ends at 117, inside trial 3
    s = check_stimulus_delivery(_write_run(tmp_path, silent=(1, 3, 5), events_until=117.0))
    assert _as_list(s) == [True, False, True, None, None, None, None, None]


def test_extra_starts_after_the_last_row_keep_the_alignment(tmp_path):
    assert _as_list(check_stimulus_delivery(_write_run(tmp_path, silent=(6,), extra_starts=4))) == [True] * 6 + [False, True]


def test_last_row_is_closed_by_its_end_time(tmp_path):
    assert _as_list(check_stimulus_delivery(_write_run(tmp_path, silent=(7,))))[-1] is False


def test_files_stamped_a_second_apart_are_one_run(tmp_path):
    path = _write_run(tmp_path, silent=(4,), summary_stamp='2026-05-23T10_12_45')
    assert _as_list(check_stimulus_delivery(path))[4] is False


def test_two_runs_in_one_folder_use_their_own_logs(tmp_path):
    a = _write_run(tmp_path, stamp='2026-05-23T10_12_44', silent=(1,))
    b = _write_run(tmp_path, stamp='2026-05-23T11_30_02', silent=(6,), t0=900.0)
    assert _as_list(check_stimulus_delivery(a)) == [True, False] + [True] * 6
    assert _as_list(check_stimulus_delivery(b)) == [True] * 6 + [False, True]


def test_missing_logs_give_na(tmp_path):
    path = _write_run(tmp_path)
    next(tmp_path.glob('Long_Form*')).write_text('')
    assert _as_list(check_stimulus_delivery(path)) == [None] * 8


def test_misaligned_table_gives_na_and_warns(tmp_path):
    ends = 104.0 + 5.0 * np.arange(8)
    ends[3] += 30.0                                     # row 3 ends after the next trial has started
    with pytest.warns(UserWarning, match='cannot line'):
        s = check_stimulus_delivery(_write_run(tmp_path, end_times=ends))
    assert _as_list(s) == [None] * 8


# ── remove_silent_trials ─────────────────────────────────────────────────────
def _config(data_dir):
    cols = {'trial_number': ColumnMapping(csv_name='Trial_Number', dtype='int'),
            'stimulus': ColumnMapping(csv_name='Stim_Relative', dtype='float'),
            'choice': ColumnMapping(csv_name='Choice', dtype='float'),
            'abort': ColumnMapping(csv_name='Abort_Trial', dtype='bool', optional=True, default=False)}
    task = TaskConfig(inputs=['stimulus'], outputs=['choice'], boundary=0.0,
                      choice_mapping=ChoiceMapping(type='identity', no_response_value=-1))
    fs = FileStructure(data_dir=str(data_dir), drop_last_row=True, behaviour_file='Trial_Summary*.csv',
                       date_regex=r'(\d{4})_(\d{1,2})_(\d{1,2})', min_trials_per_file=2)
    return ProjectConfig(name='test', columns=cols, task=task, file_structure=fs)


def _load(tmp_path):
    cfg = _config(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        exp = load_experiment(cfg)
    return exp, cfg, exp.animals['A1'].sessions[0]


def test_removes_silent_trials_and_breaks_the_next_trials_history(tmp_path):
    _write_run(tmp_path.joinpath(*SESSION), silent=(2, 5))
    exp, cfg, sess = _load(tmp_path)
    stim = sess.trials.stimulus.copy()                  # 7 rows: the last row is dropped by the loader
    summary = remove_silent_trials(exp, cfg, verbose=False)
    t = exp.animals['A1'].sessions[0].trials
    assert summary['removed'] == 2 and summary['not_aligned'] == []
    assert list(t.stimulus) == list(np.delete(stim, [2, 5]))
    # survivors 0 1 3 4 6: trials 3 and 6 followed a silent trial
    assert list(t.prev_has_prev) == [False, True, False, True, False]
    assert np.isnan(t.prev_stimulus[2]) and t.prev_stimulus[1] == pytest.approx(stim[0])
    assert np.all(t.extra['stimulus_check'] == 1.0)


def test_a_silent_abort_is_skipped_like_any_abort(tmp_path):
    _write_run(tmp_path.joinpath(*SESSION), silent=(2,), abort=(2,))
    exp, cfg, sess = _load(tmp_path)
    stim = sess.trials.stimulus.copy()
    remove_silent_trials(exp, cfg, verbose=False)
    t = exp.animals['A1'].sessions[0].trials
    assert t.prev_has_prev[2] and t.prev_stimulus[2] == pytest.approx(stim[1])   # trial 3 keeps trial 1


def test_unchecked_trials_are_kept(tmp_path):
    _write_run(tmp_path.joinpath(*SESSION), silent=(1, 5), events_until=117.0)
    exp, cfg, _ = _load(tmp_path)
    summary = remove_silent_trials(exp, cfg, verbose=False)
    t = exp.animals['A1'].sessions[0].trials
    assert summary['removed'] == 1 and summary['unchecked'] == 3          # rows 4-6 (row 7 is dropped)
    assert t.n_trials == 6 and int(np.isnan(t.extra['stimulus_check']).sum()) == 3


def test_restarted_session_removes_the_right_trials(tmp_path):
    folder = tmp_path.joinpath(*SESSION)
    _write_run(folder, stamp='2026-05-23T10_12_44', silent=(1,))
    _write_run(folder, stamp='2026-05-23T11_30_02', silent=(4,), t0=900.0, stim_offset=0.05)
    exp, cfg, sess = _load(tmp_path)
    stim = sess.trials.stimulus.copy()                  # 7 + 7 rows, merged in file order
    remove_silent_trials(exp, cfg, verbose=False)
    assert list(exp.animals['A1'].sessions[0].trials.stimulus) == list(np.delete(stim, [1, 7 + 4]))


def test_session_that_does_not_line_up_is_left_alone(tmp_path):
    path = _write_run(tmp_path.joinpath(*SESSION), silent=(2,))
    exp, cfg, sess = _load(tmp_path)
    df = pd.read_csv(path)
    df.loc[0, 'Stim_Relative'] += 0.5                   # the file changed after loading
    df.to_csv(path, index=False)
    with pytest.warns(UserWarning, match='do not line up'):
        summary = remove_silent_trials(exp, cfg, verbose=False)
    assert summary['not_aligned'] == [sess.session_id] and exp.animals['A1'].sessions[0].trials.n_trials == 7


def test_export_removes_them_and_records_the_counts(tmp_path):
    _write_run(tmp_path / 'data' / SESSION[0] / SESSION[1], silent=(2,))
    cfg = {'file_structure': {'data_dir': str(tmp_path / 'data'), 'session_pattern': 'SOUND_CAT_{animal_id}_{date}',
                              'behaviour_file': 'Trial_Summary*.csv', 'drop_last_row': True,
                              'date_regex': r'(\d{4})_(\d{1,2})_(\d{1,2})'},
           'columns': {'trial_number': {'csv_name': 'Trial_Number', 'dtype': 'int'},
                       'stimulus': {'csv_name': 'Stim_Relative', 'dtype': 'float'},
                       'choice': {'csv_name': 'Choice', 'dtype': 'float'}},
           'task': {'inputs': ['stimulus'], 'outputs': ['choice'], 'boundary': 0.0,
                    'choice_mapping': {'type': 'identity', 'no_response_value': -1}}}
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(cfg))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = export_snapshot(tmp_path / 'config.yaml', output_path=tmp_path / 'snap.pkl', verbose=False)
        exp, meta = load_snapshot(out)[:2]
    assert meta['format_version'] == SNAPSHOT_FORMAT_VERSION == 2
    assert meta['stimulus_check']['removed'] == 1 and meta['n_trials_total'] == 6
    assert exp.animals['A1'].sessions[0].trials.n_trials == 6
