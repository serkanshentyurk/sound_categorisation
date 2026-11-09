"""behaviour/daily_tracker + reports/daily_tracker: what the daily tracker computes and writes.

One small fake cohort, copied with the local route and loaded with the unchanged loader:
    SS30  habituation on the last two days (early step: trials only, move on after 300 across the step)
    SS31  regular-antibias-200 for three days, good (training: ready to move on); started after the others
    SS32  the other sound contingency; regular, then opto-regular with a light shift, a test run on 3 Nov,
          no session on 5 Nov, few trials on 6 Nov and a strong lean on 7 Nov
"""
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from behav_utils.config.schema import load_config
from sound_categorisation.behaviour.adaptation import BiasRule, session_bias_checks
from sound_categorisation.behaviour.daily_tracker import (
    TrackerSettings,
    compute_daily_tracker,
    load_mirror,
    load_tracker_settings,
)
from sound_categorisation.data.raw_mirror import mirror_sessions
from sound_categorisation.reports.daily_tracker import write_daily_report

CONFIG = Path(__file__).resolve().parents[2] / 'config.yaml'
TODAY = date(2026, 11, 7)
D = lambda m, d: date(2026, m, d)  # noqa: E731


def _cohort(raw: Path, write_rig_session) -> None:
    k = iter(range(1000))
    write_rig_session(raw, 'SS30', D(11, 6), 'habituation', n=200, seed=next(k))
    write_rig_session(raw, 'SS30', D(11, 7), 'habituation', n=150, seed=next(k))
    for d in (D(11, 5), D(11, 6), D(11, 7)):
        write_rig_session(raw, 'SS31', d, 'regular-antibias-200', n=400, seed=next(k))
    other = 'Low_Right_High_Left'
    for i in range(5):
        write_rig_session(raw, 'SS32', D(10, 25) + timedelta(days=i), 'regular', contingency=other, seed=next(k))
    for d in (D(10, 30), D(10, 31), D(11, 1), D(11, 2), D(11, 3), D(11, 4), D(11, 6), D(11, 7)):
        write_rig_session(raw, 'SS32', d, 'opto-regular', contingency=other, light_shift=0.2,
                          n=150 if d == D(11, 6) else 300, pse=0.6 if d == D(11, 7) else 0.0,
                          emulator=d == D(11, 3), seed=next(k))


@pytest.fixture(scope='module')
def loaded(rig_session):
    root = Path(tempfile.mkdtemp(prefix='tracker_'))
    _cohort(root / 'ceph' / 'Raw', rig_session)
    config = load_config(CONFIG)
    mirror_sessions(config, days=14, summary_days=None, today=TODAY, source=('local', root / 'ceph' / 'Raw'),
                    dest=root / 'mirror')
    return load_mirror(CONFIG, root / 'mirror')


@pytest.fixture(scope='module')
def tracker(loaded):
    experiment, notes = loaded
    return compute_daily_tracker(experiment, load_tracker_settings(CONFIG), today=TODAY, notes=notes)


def _row(t, animal, day):
    return t.sessions[(t.sessions['animal'] == animal) & (t.sessions['date'] == day)].iloc[0]


def _flags(t, animal):
    return {(r.date, r.flag) for r in t.flags[t.flags['animal'] == animal].itertuples()}


class TestSettings:
    def test_the_repository_block_loads(self):
        s = load_tracker_settings(CONFIG)
        assert s.days == 14 and s.summary_days is None and 'habituation' in s.early_steps

    def test_a_missing_key_is_named(self):
        with pytest.raises(ValueError, match='daily_tracker.flags.accuracy_drop'):
            TrackerSettings.from_block({'days': 14, 'flags': {}})

    def test_every_number_is_listed_with_where_it_is_set(self):
        table = load_tracker_settings(CONFIG).table(BiasRule())
        assert {'config.yaml', 'sound_categorisation/behaviour/adaptation.py'} == set(table['set in'])


class TestDays:
    def test_one_row_per_animal_and_day(self, tracker):
        assert tracker.animals == ['SS30', 'SS31', 'SS32'] and len(tracker.sessions) == 3 * 14

    def test_early_steps_carry_trials_and_no_statistics(self, tracker):
        r = _row(tracker, 'SS30', D(11, 6))
        assert r['group'] == 'early step' and r['trials'] == 199 and np.isnan(r['accuracy'])
        assert np.isnan(r['correct']) and np.isnan(r['responded'])     # not asked for, so not zero

    def test_counts_and_rig(self, tracker):
        r = _row(tracker, 'SS31', D(11, 7))
        assert r['rig'] == '31107' and r['responded'] + r['no_response'] == r['trials']
        assert 0 < r['correct'] <= r['responded'] and r['light_trials'] == 0

    def test_right_and_left_follow_the_contingency(self, tracker):
        r = _row(tracker, 'SS32', D(11, 7))
        assert r['side_bias'] < -0.1 and r['right_choices_minus_50'] == pytest.approx(-r['side_bias'])

    def test_test_runs_are_marked(self, tracker):
        assert bool(_row(tracker, 'SS32', D(11, 3))['test_run'])

    def test_tidy_rows(self, tracker):
        rows = tracker.to_rows()
        assert list(rows.columns) == ['animal', 'date', 'preset', 'stat', 'value']
        assert 'accuracy_right_minus_left' in set(rows['stat'])


class TestSessionNumbers:
    def test_counted_per_animal_without_test_runs(self, tracker):
        r = _row(tracker, 'SS32', D(11, 7))
        assert r['session_no'] == 12 and r['preset_no'] == 7
        assert np.isnan(_row(tracker, 'SS32', D(11, 3))['session_no'])
        assert _row(tracker, 'SS31', D(11, 5))['session_no'] == 1


def test_expected_animals_without_a_session(loaded):
    experiment, notes = loaded
    t = compute_daily_tracker(experiment, load_tracker_settings(CONFIG), today=TODAY, notes=notes,
                              expected={'SS30', 'SS98', 'SS99'}, last_seen={'SS99': D(10, 1)})
    m = t.missing.set_index('animal')
    assert list(m.index) == ['SS98', 'SS99'] and m.loc['SS99', 'days_since'] == 37
    assert m.loc['SS98', 'last_session'] is None


class TestFlags:
    def test_the_experiment_animal(self, tracker):
        f = _flags(tracker, 'SS32')
        assert {(D(11, 5), 'missing day'), (D(11, 6), 'low trials'), (D(11, 7), 'side bias'),
                (D(11, 7), 'accuracy drop')} <= f
        assert not any(d == D(11, 3) for d, _ in f)

    def test_days_before_an_animal_started_are_not_missing(self, tracker):
        assert _flags(tracker, 'SS31') == set()

    def test_the_bias_flag_names_what_failed(self, tracker):
        detail = tracker.flags[(tracker.flags['animal'] == 'SS32') & (tracker.flags['flag'] == 'side bias')]
        assert detail['detail'].iloc[0].startswith('PSE +0.')


class TestMoveOn:
    def test_marks(self, tracker):
        m = tracker.move_on.set_index('animal')
        total = int(tracker.sessions.loc[tracker.sessions['animal'] == 'SS30', 'trials'].sum())
        assert bool(m.loc['SS30', 'ready']) and f'{total} trials over 2 session(s)' in m.loc['SS30', 'detail']
        assert bool(m.loc['SS31', 'ready']) and m.loc['SS31', 'group'] == 'training'
        assert m.loc['SS32', 'ready'] is None and m.loc['SS32', 'group'] == 'experiment'


class TestConditions:
    def test_one_summary_per_preset_and_light_rows_only_with_light(self, tracker):
        c = tracker.conditions[tracker.conditions['animal'] == 'SS32']
        assert set(map(tuple, c[['preset', 'trials']].drop_duplicates().to_numpy())) == {
            ('regular', 'no light'), ('opto-regular', 'no light'), ('opto-regular', 'light')}

    def test_last_five_real_sessions_with_the_individuals(self, tracker):
        cs = tracker.condition_sessions
        mine = cs[(cs['animal'] == 'SS32') & (cs['preset'] == 'opto-regular') & (cs['trials'] == 'no light')]
        assert sorted(mine['date'].unique()) == [D(11, 1), D(11, 2), D(11, 4), D(11, 6), D(11, 7)]

    def test_pooled_and_mean_both_there_and_light_shifts_the_pse(self, tracker):
        c = tracker.conditions.set_index(['animal', 'preset', 'trials', 'stat'])
        no_light, light = c.loc[('SS32', 'opto-regular', 'no light', 'pse')], c.loc[('SS32', 'opto-regular', 'light', 'pse')]
        assert np.isfinite(no_light['pooled']) and np.isfinite(no_light['mean']) and np.isfinite(no_light['sd'])
        assert light['pooled'] > no_light['pooled'] + 0.1

    def test_curves(self, tracker):
        cur = tracker.curves[('SS32', 'opto-regular', 'light')]
        assert cur.pooled is not None and len(cur.sessions) == 5 and cur.mean_y.shape == cur.x.shape
        assert tracker.latest['SS32'][0] == D(11, 7) and 'SS30' not in tracker.latest


def test_session_bias_checks():
    df = pd.DataFrame({'pse': [0.5, 0.0, 0.0, np.nan], 'lapse_low': [0.0, 0.4, 0.0, np.nan],
                       'lapse_high': [0.0, 0.0, 0.0, np.nan], 'accuracy': [0.8, 0.8, 0.5, np.nan]})
    c = session_bias_checks(df, BiasRule(), pse_column='pse')
    assert c['pse_out'].tolist() == [True, False, False, False]
    assert c['lapse_out'].tolist() == [False, True, False, False]
    assert c['accuracy_out'].tolist() == [False, False, True, False]
    assert c['flagged'].tolist() == [True, True, True, False]


def test_early_steps_only(rig_session, tmp_path):
    """A cohort that has only reached habituation: no conditions yet, and the report still writes."""
    raw = tmp_path / 'ceph' / 'Raw'
    rig_session(raw, 'SS25', D(11, 6), 'habituation', n=215, seed=1)
    rig_session(raw, 'SS30', D(11, 6), 'habituation', n=180, seed=2)
    mirror_sessions(load_config(CONFIG), days=14, summary_days=None, today=TODAY, source=('local', raw),
                    dest=tmp_path / 'mirror')
    experiment, notes = load_mirror(CONFIG, tmp_path / 'mirror')
    t = compute_daily_tracker(experiment, load_tracker_settings(CONFIG), today=TODAY, notes=notes)
    assert t.conditions.empty and 'animal' in t.conditions and t.latest == {}
    assert set(t.move_on['ready']) == {False}
    page = write_daily_report(t, tmp_path / 'r.html').read_text()
    assert 'id="SS25"' in page and 'No full-task conditions yet' in page


def test_no_copy_builds_the_page_without_ceph(rig_session, tmp_path, monkeypatch):
    """--no-copy: ceph unreachable (no such folder, no SWC_USER), the page is built from the local copy."""
    from sound_categorisation.cli import daily_tracker as cli
    monkeypatch.setenv('SC_DATA_ROOT', str(tmp_path / 'data'))
    monkeypatch.setenv('SC_RESULTS_ROOT', str(tmp_path / 'results'))
    monkeypatch.setenv('BEHAV_DATA_DIR', str(tmp_path / 'nowhere'))
    monkeypatch.delenv('SWC_USER', raising=False)
    rig_session(tmp_path / 'data' / 'raw_mirror' / 'Raw', 'SS25', date.today() - timedelta(days=1), 'habituation',
                n=200, seed=3)
    assert cli.main(['--no-copy', '--no-open']) == 0
    page = (tmp_path / 'results' / 'daily_tracker' / 'daily_tracker_latest.html').read_text()
    assert 'not refreshed from ceph' in page and 'id="SS25"' in page


def test_report(tracker, tmp_path):
    path = write_daily_report(tracker, tmp_path / 'r' / 'daily_tracker_2026-11-07.html', source='fake')
    page = path.read_text()
    assert all(f'id="{a}"' in page for a in tracker.animals)
    assert page.count('<img') >= 5 and 'Numbers used' in page and 'daily_tracker.flags.min_correct_trials' in page
    assert 'missing day' in page and 'class="ready">ready<' in page
    assert page.count('class="view animal"') == 3 and 'id="home"' in page and 'class="totop"' in page
    assert '<svg class="spark"' in page and '07 Nov · #12' in page and '← overview' in page
    assert page.index('id="SS32"') < page.index('id="SS30"')      # flagged animals first
