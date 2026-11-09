"""data/raw_mirror + paths.raw_source: the daily tracker's copy of the raw session files.

Driven by fake ceph trees in tmp_path (the rig_session fixture writes the rig's current files plus a lick
file and a video): which animals and sessions are copied, only the two small files, timestamps kept, nothing
copied twice, old folders removed, both routes giving the same copy, the Session_Parameters reader, and the
choice of route.
"""
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest
from behav_utils.config.schema import load_config
from sound_categorisation.data import paths
from sound_categorisation.data import raw_mirror as rm
from sound_categorisation.data.raw_mirror import mirror_sessions, select_folders, session_parameters

CONFIG = Path(__file__).resolve().parents[2] / 'config.yaml'
TODAY = date(2026, 11, 7)


@pytest.fixture
def config():
    return load_config(CONFIG)


@pytest.fixture
def ceph(tmp_path, rig_session):
    """SS24 trains this week, SS25 started yesterday, SS13 last ran in September (not shown)."""
    raw = tmp_path / 'ceph' / 'Raw'
    for k, d in enumerate([date(2026, 10, 1), date(2026, 11, 5), date(2026, 11, 6), date(2026, 11, 7)]):
        rig_session(raw, 'SS24', d, 'regular', seed=k)
    rig_session(raw, 'SS25', date(2026, 11, 6), 'habituation', seed=10)
    rig_session(raw, 'SS13', date(2026, 9, 1), 'regular', seed=11)
    return raw


def _files(root: Path) -> list:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file())


def _mirror(config, raw, dest, **kw):
    kw = {'days': 14, 'summary_days': None, 'today': TODAY, **kw}
    return mirror_sessions(config, source=('local', raw), dest=dest, **kw)


class TestSelection:
    def test_shown_animals_and_whole_history(self):
        folders = [('A', 'f1', date(2026, 10, 1)), ('A', 'f2', date(2026, 11, 6)), ('B', 'f3', date(2026, 9, 1))]
        shown, keep = select_folders(folders, today=TODAY, days=14, summary_days=None)
        assert shown == ('A',)
        assert keep == {('A', 'f1'), ('A', 'f2')}

    def test_summary_days_caps_the_history(self):
        folders = [('A', 'f1', date(2026, 10, 1)), ('A', 'f2', date(2026, 11, 6))]
        assert select_folders(folders, today=TODAY, days=14, summary_days=10)[1] == {('A', 'f2')}

    def test_animals_limits_the_shown_set_and_future_folders_are_ignored(self):
        folders = [('A', 'f1', date(2026, 11, 6)), ('B', 'f2', date(2026, 11, 6)), ('A', 'f3', date(2026, 11, 9))]
        shown, keep = select_folders(folders, today=TODAY, days=14, summary_days=None, animals={'A'})
        assert shown == ('A',) and keep == {('A', 'f1')}


class TestLocalRoute:
    def test_copies_only_the_two_files_of_shown_animals(self, config, ceph, tmp_path):
        dest = tmp_path / 'mirror'
        r = _mirror(config, ceph, dest)
        assert r.shown_animals == ('SS24', 'SS25') and r.n_folders == 5
        names = _files(dest)
        assert len(names) == 10
        assert all(Path(n).name.startswith(('Trial_Summary', 'Session_Parameters')) for n in names)
        assert not any(n.startswith('SS13') for n in names)

    def test_keeps_timestamps_and_copies_nothing_twice(self, config, ceph, tmp_path):
        dest = tmp_path / 'mirror'
        _mirror(config, ceph, dest)
        for n in _files(dest):
            assert int((dest / n).stat().st_mtime) == int((ceph / n).stat().st_mtime)
        again = _mirror(config, ceph, dest)
        assert again.copied == () and again.unchanged == 10

    def test_a_changed_file_is_copied_again(self, config, ceph, tmp_path):
        dest = tmp_path / 'mirror'
        _mirror(config, ceph, dest)
        target = next(ceph.glob('SS24/*_2026_11_7/Trial_Summary*.csv'))
        target.write_text(target.read_text() + 'SS24,SOUND_CAT,Full_Task_Cont,301\n')
        r = _mirror(config, ceph, dest)
        assert r.copied == (target.relative_to(ceph).as_posix(),)

    def test_reports_progress_and_last_session_dates(self, config, ceph, tmp_path):
        messages = []
        r = _mirror(config, ceph, tmp_path / 'mirror', progress=messages.append)
        assert messages[0].startswith('listing ') and messages[1] == 'copying 10 files …'
        assert r.last_session == {'SS13': date(2026, 9, 1), 'SS24': date(2026, 11, 7), 'SS25': date(2026, 11, 6)}

    def test_dry_run_changes_nothing(self, config, ceph, tmp_path):
        dest = tmp_path / 'mirror'
        r = _mirror(config, ceph, dest, dry_run=True)
        assert len(r.copied) == 10 and not dest.exists()

    def test_folders_no_longer_selected_are_removed(self, config, ceph, tmp_path):
        dest = tmp_path / 'mirror'
        _mirror(config, ceph, dest)
        r = _mirror(config, ceph, dest, summary_days=14)
        assert r.removed_folders == ('SS24/SOUND_CAT_SS24_2026_10_1',)
        assert not (dest / 'SS24' / 'SOUND_CAT_SS24_2026_10_1').exists()


def test_local_copy_status_reads_only_the_copy(config, ceph, tmp_path):
    dest = tmp_path / 'mirror'
    _mirror(config, ceph, dest)
    shown, last = rm.local_copy_status(config, days=14, today=TODAY, dest=dest)
    assert shown == ('SS24', 'SS25') and last == {'SS24': date(2026, 11, 7), 'SS25': date(2026, 11, 6)}
    assert rm.local_copy_status(config, days=14, today=TODAY, dest=tmp_path / 'none') == ((), {})


def test_a_cohort_run_leaves_the_other_animals_copies_alone(config, ceph, tmp_path):
    dest = tmp_path / 'mirror'
    _mirror(config, ceph, dest)
    r = _mirror(config, ceph, dest, animals={'SS25'})
    assert r.removed_folders == () and (dest / 'SS24').is_dir()


def _gnu_find() -> bool:
    try:
        return subprocess.run(['find', '.', '-maxdepth', '0', '-printf', ''], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _gnu_find() or shutil.which('bash') is None, reason='needs GNU find and bash')
def test_ssh_route_gives_the_same_copy(config, ceph, tmp_path, monkeypatch):
    """The remote commands (find, then tar) run here through bash in place of ssh."""
    commands = []

    def local_shell(host, remote_command):
        commands.append(remote_command)
        return ['bash', '-c', remote_command]
    monkeypatch.setattr(rm, '_ssh_command', local_shell)
    via_ssh = mirror_sessions(config, days=14, summary_days=None, today=TODAY,
                              source=('ssh', ('me@hpc-gw2', ceph.as_posix())), dest=tmp_path / 'ssh')
    _mirror(config, ceph, tmp_path / 'local')
    assert _files(tmp_path / 'ssh') == _files(tmp_path / 'local')
    assert len(via_ssh.copied) == 10 and commands[0].startswith('find ') and commands[1].startswith('tar ')
    n = _files(tmp_path / 'ssh')[0]
    assert int((tmp_path / 'ssh' / n).stat().st_mtime) == int((ceph / n).stat().st_mtime)


def test_ssh_command_goes_through_the_gateway():
    cmd = rm._ssh_command('me@hpc-gw2', 'echo hi')
    assert cmd[:3] == ['ssh', '-J', f'me@{paths.SWC_GATEWAY}'] and cmd[-2:] == ['me@hpc-gw2', 'echo hi']


def test_session_parameters(tmp_path):
    f = tmp_path / 'S'
    f.mkdir()
    (f / 'Session_Parameters2026-11-06T07_00_00.csv').write_text('h1,h2,h3,h4\nRig_ID,31107,Agent_Sim,False\n')
    assert session_parameters(f) == {'Rig_ID': '31107', 'Agent_Sim': 'False'}
    (f / 'Session_Parameters2026-11-06T08_00_00.csv').write_text('h1,h2\nRig_ID,31108\n')
    assert session_parameters(f)['Rig_ID'] == '31107/31108'
    assert session_parameters(tmp_path / 'missing') == {}


class TestRoute:
    def test_local_when_the_raw_folder_is_here(self, tmp_path, monkeypatch):
        (tmp_path / 'Raw').mkdir()
        monkeypatch.setenv('BEHAV_DATA_DIR', str(tmp_path))
        assert paths.raw_source() == ('local', tmp_path / 'Raw')

    def test_ssh_otherwise(self, monkeypatch):
        monkeypatch.setenv('BEHAV_DATA_DIR', '/ceph/akrami/Someone/Head_Fixed_Behavior/Data')
        monkeypatch.setenv('SWC_USER', 'someone')
        monkeypatch.delenv('SWC_NODE', raising=False)
        assert paths.raw_source() == ('ssh', ('someone@hpc-gw2', '/ceph/akrami/Someone/Head_Fixed_Behavior/Data/Raw'))

    def test_swc_node_picks_another_login_node(self, monkeypatch):
        monkeypatch.setenv('BEHAV_DATA_DIR', '/ceph/akrami/Someone/Head_Fixed_Behavior/Data')
        monkeypatch.setenv('SWC_USER', 'someone')
        monkeypatch.setenv('SWC_NODE', 'hpc-gw1')
        assert paths.raw_source()[1][0] == 'someone@hpc-gw1'

    def test_stops_with_a_message_without_either(self, monkeypatch):
        monkeypatch.setenv('BEHAV_DATA_DIR', '/ceph/akrami/Someone/Head_Fixed_Behavior/Data')
        monkeypatch.delenv('SWC_USER', raising=False)
        with pytest.raises(RuntimeError, match='SWC_USER'):
            paths.raw_source()


def test_mirror_dirs_follow_the_roots(tmp_path, monkeypatch):
    monkeypatch.setenv('SC_DATA_ROOT', str(tmp_path / 'data'))
    monkeypatch.setenv('SC_RESULTS_ROOT', str(tmp_path / 'results'))
    assert paths.raw_mirror_dir() == tmp_path / 'data' / 'raw_mirror' / 'Raw'
    assert paths.daily_tracker_dir() == tmp_path / 'results' / 'daily_tracker'
