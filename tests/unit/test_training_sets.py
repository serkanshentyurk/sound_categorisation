"""Saved training sets (inference/training_sets.py), checked without torch: the file, the
configuration a reuse depends on, and the comparison a network makes before reusing one."""

import json

import numpy as np
import pytest
from sound_categorisation.data.paths import snpe_net_path, snpe_training_set_path
from sound_categorisation.inference.amortised import AmortisedSBI
from sound_categorisation.inference.representation import feature_names
from sound_categorisation.inference.training_sets import config_changes, read_training_set, save_training_set

STATS = ['accuracy', 'mu', 'win_stay']
SMALL = dict(dist_schedule='uniform', N=(2, 3), T=(60, 90), burn_in=40, mode='pooled', stat_names=STATS,
             design_features=('n_sessions', 'n_valid_trials'), min_accuracy=0.62)


def _net(**changes):
    return AmortisedSBI('be', **{**SMALL, **changes})


def _save(net, path, n=6, seed=5):
    theta, x, n_drawn, n_passed = net.simulate_training_set(n, seed=seed, n_jobs=1, show_progress=False)
    save_training_set(path, theta, x, config=net.training_config(n, seed), param_names=net.param_names,
                      feature_names=feature_names(net.mode, net.stat_names, net.design_features),
                      n_drawn=n_drawn, n_passed=n_passed, simulation_time=1.5)
    return theta, x, n_drawn, n_passed


def test_a_saved_set_reads_back_as_written_without_pickle(tmp_path):
    net, path = _net(), tmp_path / 'set.npz'
    theta, x, n_drawn, n_passed = _save(net, path)
    ts = read_training_set(path)
    np.testing.assert_array_equal(ts.theta, theta)
    np.testing.assert_array_equal(ts.x, x)
    assert ts.param_names == net.param_names and ts.feature_names == [*STATS, 'n_sessions', 'n_valid_trials']
    assert (ts.n_drawn, ts.n_passed, ts.meta['n_kept']) == (n_drawn, n_passed, 6) and n_drawn > 6
    assert ts.meta['simulation_time'] == 1.5 and ts.meta['created_utc'] and 'sha' in ts.meta['git']
    assert config_changes(ts.config, net.training_config(6, 5)) == []
    with np.load(path, allow_pickle=False) as raw:                 # plain arrays and a JSON string
        assert set(raw.files) == {'theta', 'x', 'meta'}
        assert json.loads(str(raw['meta']))['format'] == 1
    assert list(tmp_path.iterdir()) == [path]                       # no temporary file left behind


def test_non_finite_cells_are_kept_as_simulated(tmp_path):
    """Median imputation happens when a network trains, so the file keeps what the simulator gave."""
    x = np.array([[1.0, np.nan], [np.inf, 2.0], [3.0, 4.0]])
    save_training_set(tmp_path / 's.npz', np.ones((3, 1)), x, config={'seed': 1}, param_names=['a'],
                      feature_names=['f', 'g'], n_drawn=3, n_passed=3, simulation_time=0.0)
    np.testing.assert_array_equal(read_training_set(tmp_path / 's.npz').x, x)


def test_an_interrupted_write_leaves_no_file_to_reuse(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(np, 'savez', fail)
    with pytest.raises(OSError, match='disk full'):
        save_training_set(tmp_path / 's.npz', np.ones((2, 1)), np.ones((2, 1)), config={}, param_names=['a'],
                          feature_names=['f'], n_drawn=2, n_passed=2, simulation_time=0.0)
    assert list(tmp_path.iterdir()) == []


def test_rows_and_columns_must_match_the_names(tmp_path):
    common = dict(config={}, n_drawn=3, n_passed=3, simulation_time=0.0)
    with pytest.raises(ValueError, match='one row per simulation'):
        save_training_set(tmp_path / 's.npz', np.ones((3, 1)), np.ones((2, 1)), param_names=['a'],
                          feature_names=['f'], **common)
    with pytest.raises(ValueError, match='do not match'):
        save_training_set(tmp_path / 's.npz', np.ones((3, 2)), np.ones((3, 1)), param_names=['a'],
                          feature_names=['f'], **common)
    assert list(tmp_path.iterdir()) == []


def test_config_changes_names_what_differs_as_the_file_stores_it():
    a = {'N': (2, 5), 'T': (250, 600), 'seed': 42, 'min_accuracy': 0.7, 'prior': 'learning'}
    assert config_changes(a, {**a, 'N': [2, 5]}) == []                  # a tuple and its list are one value
    assert config_changes(a, {**a, 'seed': 43, 'T': (250, 500)}) == ['T', 'seed']
    assert config_changes(a, {k: v for k, v in a.items() if k != 'prior'}) == ['prior']
    assert config_changes(a, {**a, 'min_accuracy': None}) == ['min_accuracy']


def test_the_configuration_holds_everything_that_shapes_the_simulations():
    net = _net()
    cfg = net.training_config(100, 7)
    json.dumps(cfg)                                                      # storable as it is
    assert cfg['prior'] == 'learning' and cfg['log_uniform'] == ['eta_learning', 'eta_relax']
    assert cfg['min_accuracy'] == 0.62 and cfg['param_names'] == net.param_names
    assert cfg['prior_low'] == net.bounds()[0].tolist() and cfg['prior_high'] == net.bounds()[1].tolist()
    assert config_changes(cfg, net.training_config(101, 7)) == ['n_simulations']
    assert config_changes(cfg, net.training_config(100, 8)) == ['seed']
    assert net.training_config(100, None)['seed'] == 0                  # train() simulates with seed or 0
    others = {'N': _net(N=(2, 4)), 'T': _net(T=(60, 120)), 'burn_in': _net(burn_in=50),
              'stat_names': _net(stat_names=STATS[:2]), 'design_features': _net(design_features=()),
              'min_accuracy': _net(min_accuracy=None), 'dist_schedule': _net(dist_schedule='hard_a'),
              'prior': _net(prior='manuscript')}
    for key, other in others.items():
        assert key in config_changes(cfg, other.training_config(100, 7)), key


def test_a_network_compares_a_saved_set_with_the_one_it_would_make(tmp_path):
    net, path = _net(), tmp_path / 'set.npz'
    assert net.training_set_changes(path, 6, 5) == [] and net.training_set_changes(None, 6, 5) == []
    _save(net, path, n=6, seed=5)
    assert net.training_set_changes(path, 6, 5) == []
    assert net.training_set_changes(path, 6, 6) == ['seed']
    assert _net(prior='manuscript').training_set_changes(path, 6, 5) == [     # the same upper bounds
        'log_uniform', 'prior', 'prior_low']


def test_the_training_set_lives_beside_its_network(tmp_path, monkeypatch):
    monkeypatch.setenv('SC_DATA_ROOT', str(tmp_path))
    network, saved = snpe_net_path('pooled', 'BE', 'uniform'), snpe_training_set_path('pooled', 'BE', 'uniform')
    assert saved.parent == network.parent and saved.name == 'snpe_pooled_BE_uniform_training.npz'
    assert saved.match('snpe_*') and not saved.match('snpe_*.pkl')     # archived with it, not listed as one
