"""sc-update and sc-fetch: the commands they would run (nothing is executed here)."""
import pytest
from sound_categorisation.cli import fetch, update


def test_update_pulls_then_installs():
    cmds = update.commands()
    assert cmds[0] == ['git', 'pull', '--ff-only']
    assert cmds[1][-3:] == ['install', '-e', '.']
    keep = update.commands(keep_behav_utils=True)
    assert keep[1][-1] == '--no-deps' and '_install_other_requirements' in keep[2][-1]


def test_keep_behav_utils_still_installs_the_other_dependencies(monkeypatch):
    """--keep-behav-utils must not skip new dependencies (pip's --no-deps alone would)."""
    reqs = ['numpy', 'joblib', 'scikit-learn',
            'behav_utils@ git+https://github.com/serkanshentyurk/behav_utils.git@v0.9.0',
            'torch; extra == "fit"', 'pytest; extra == "dev"']
    monkeypatch.setattr(update.importlib.metadata, 'requires', lambda dist: reqs)
    assert update.other_requirements() == ['numpy', 'joblib', 'scikit-learn']


@pytest.fixture
def laptop(tmp_path, monkeypatch):
    monkeypatch.setenv('BEHAV_DATA_DIR', '/ceph/akrami/Someone/Head_Fixed_Behavior/Data')
    monkeypatch.setenv('SWC_USER', 'someone')
    monkeypatch.setenv('SC_DATA_ROOT', str(tmp_path / 'data'))
    monkeypatch.setenv('SC_RESULTS_ROOT', str(tmp_path / 'results'))
    return tmp_path


def test_fetch_snapshot_mirrors_the_cluster_layout(laptop):
    cmd = fetch.command('snapshot')
    assert cmd[:2] == ['rsync', '-av'] and cmd[2:4] == ['-e', 'ssh -J someone@ssh.swc.ucl.ac.uk']
    assert cmd[-2] == ('someone@hpc-gw2:/ceph/akrami/Someone/Head_Fixed_Behavior/Data/Processed/'
                       'behaviour/snapshots/sound_cat_snapshot.pkl')
    assert cmd[-1] == f'{laptop}/data/behaviour/snapshots/'


def test_fetch_results_and_dry_run(laptop):
    cmd = fetch.command('results', 'model_identification/all-mice', dry_run=True)
    assert '-n' in cmd
    assert cmd[-2].endswith('/Processed/results/model_identification/all-mice/')
    assert cmd[-1] == f'{laptop}/results/model_identification/all-mice/'


def test_fetch_needs_both_settings(monkeypatch):
    monkeypatch.delenv('SWC_USER', raising=False)
    monkeypatch.setenv('BEHAV_DATA_DIR', '/ceph/x')
    with pytest.raises(SystemExit, match='SWC_USER is not set'):
        fetch.command('networks')
    monkeypatch.setenv('SWC_USER', 'someone')
    monkeypatch.delenv('BEHAV_DATA_DIR')
    with pytest.raises(RuntimeError, match='BEHAV_DATA_DIR is not set'):
        fetch.command('networks')
