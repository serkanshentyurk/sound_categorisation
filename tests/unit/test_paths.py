"""data/paths: results root, run ids, latest resolution, the model-identification layout, metadata."""

import json
import os
from datetime import date

import pytest
from sound_categorisation.data import paths
from sound_categorisation.data.paths import (
    REPORTS,
    build_metadata,
    latest_run,
    mark_latest,
    model_id_dir,
    new_run_id,
    resolve_run,
    results_root,
    run_dir,
    start_run,
)


def test_results_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv('SC_RESULTS_ROOT', str(tmp_path))
    assert results_root() == tmp_path
    monkeypatch.delenv('SC_RESULTS_ROOT')
    assert results_root().name == 'results'           # <repo>/results locally, <data root>/results on the cluster


def test_run_id_shape():
    rid = new_run_id(today=date(2026, 10, 1))
    assert rid.startswith('2026-10-01_') and len(rid.split('_')[1]) == 7
    assert new_run_id(fast=True, today=date(2026, 10, 1)).endswith('_fast')
    assert new_run_id(label='hard a / b', today=date(2026, 10, 1)).endswith('_hard-a-b')


def test_unknown_report_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_dir('nonsense', 'c', 'r', root=tmp_path)
    assert set(REPORTS) >= {'opto_contrasts', 'switch_adaptation', 'model_identification'}


def test_start_run_creates_logs_and_latest(tmp_path):
    run = start_run('opto_contrasts', 'coh', root=tmp_path)
    assert (run / 'logs').is_dir()
    parent = run.parent
    assert (parent / 'latest.txt').read_text().strip() == run.name
    assert latest_run('opto_contrasts', 'coh', root=tmp_path) == run
    assert resolve_run('opto_contrasts', 'coh', 'latest', root=tmp_path) == run
    assert resolve_run('opto_contrasts', 'coh', run.name, root=tmp_path) == run
    assert resolve_run('opto_contrasts', 'coh', str(run), root=tmp_path) == run


def test_latest_moves_with_each_start(tmp_path):
    a = start_run('opto_contrasts', 'coh', '2026-01-01_0000000', root=tmp_path)
    b = start_run('opto_contrasts', 'coh', '2026-01-02_0000000', root=tmp_path)
    assert latest_run('opto_contrasts', 'coh', root=tmp_path) == b
    mark_latest(a)
    assert latest_run('opto_contrasts', 'coh', root=tmp_path) == a


def test_latest_falls_back_to_txt_then_newest_dir(tmp_path):
    a = start_run('switch_adaptation', 'coh', '2026-01-01_0000000', root=tmp_path)
    b = start_run('switch_adaptation', 'coh', '2026-01-03_0000000', root=tmp_path)
    link = a.parent / 'latest'
    if link.is_symlink():
        link.unlink()
    assert latest_run('switch_adaptation', 'coh', root=tmp_path) == b        # latest.txt
    (a.parent / 'latest.txt').unlink()
    assert latest_run('switch_adaptation', 'coh', root=tmp_path) == b        # newest run-id-shaped dir
    (a.parent / 'not_a_run').mkdir()
    assert latest_run('switch_adaptation', 'coh', root=tmp_path) == b


def test_latest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_run('opto_contrasts', 'nobody', root=tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_run('opto_contrasts', 'nobody', '2026-01-01_0000000', root=tmp_path)


def test_model_id_layout(tmp_path):
    run = start_run('model_identification', 'synthetic_uniform', root=tmp_path)
    gs = model_id_dir(run, 'grid_search', 'update_matrix', 'uniform')
    sbi = model_id_dir(run, 'sbi', 'update_matrix', 'uniform', 'pooled')
    assert gs == run / 'grid_search' / 'update_matrix' / 'uniform'
    assert sbi == run / 'sbi' / 'update_matrix' / 'uniform' / 'pooled'
    assert sbi.parent.parent == run / 'sbi' / 'update_matrix'      # same run, same fit-target level as gs


def test_build_metadata_is_json_and_stamped():
    m = build_metadata('x', {'a': 1}, run_id='2026-01-01_0000000')
    s = json.dumps(m, default=str)
    assert '2026-01-01_0000000' in s
    assert {'argv', 'run_id', 'git_sha', 'git_dirty', 'settings', 'versions', 'timestamp_utc'} <= set(m)
    assert isinstance(m['git_dirty'], bool)
    assert m['versions']['sound_categorisation'] != 'not installed'


def test_data_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv('SC_DATA_ROOT', str(tmp_path))
    assert paths.data_root() == tmp_path
    assert paths.cohort_path('c') == tmp_path / 'synthetic_cohorts' / 'c.pkl'
    assert os.path.commonpath([paths.snpe_networks_dir(), tmp_path]) == str(tmp_path)
