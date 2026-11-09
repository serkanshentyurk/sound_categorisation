"""inference.network_check and the network-file guards, on network files written without torch.

A network file is the dict AmortisedSBI.save writes; the posterior is replaced by a stand-in that
only carries the prior (as sbi's posterior does), so everything here runs without torch.
"""

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sound_categorisation.inference.network_check import (
    config_rows,
    fingerprint_rows,
    parameter_rows,
    parse_network_name,
    sessions_diff,
    simulate_features,
)
from sound_categorisation.inference.representation import compute_feature_medians
from sound_categorisation.inference.simulator import get_bounds_arrays, get_param_names
from sound_categorisation.settings import SBI_DESIGN_FEATURES

STATS = ['accuracy', 'mu', 'win_stay']


def _network(design=(), N=(4, 6), names=None, medians=None, prior_in_posterior=True):
    lo, hi = get_bounds_arrays('be')
    prior = SimpleNamespace(base_dist=SimpleNamespace(low=lo, high=hi))
    return {'model': 'be', 'dist_schedule': 'uniform', 'N': N, 'T': (40, 60), 'burn_in': 20,
            'mode': 'pooled', 'stat_names': STATS, 'design_features': list(design),
            'param_names': names or get_param_names('be'),
            'trained_posterior': SimpleNamespace(prior=prior) if prior_in_posterior else None,
            'training_metadata': {'n_simulations': 500, 'seed': 42}, 'feature_medians': medians}


def test_parse_network_name():
    assert parse_network_name('snpe_moments_SC_hard_a.pkl') == {
        'rep': 'moments', 'model': 'SC', 'distribution': 'hard_a'}
    with pytest.raises(ValueError):
        parse_network_name('net.pkl')


def test_config_flags_what_changed():
    rows = pd.DataFrame(config_rows(_network(), 'snpe_pooled_BE_uniform.pkl')).set_index('check')
    assert rows.loc['model', 'ok'] and rows.loc['distribution', 'ok'] and rows.loc['mode', 'ok']
    assert not rows.loc['N (sessions)', 'ok']                   # (4, 6) against the current (2, 6)
    assert not rows.loc['design_features', 'ok']                # none against the current two
    assert not rows.loc['stat_names', 'ok']                     # a 3-stat test network


def test_parameters_and_prior_bounds():
    rows = pd.DataFrame(parameter_rows(_network(), 'snpe_pooled_BE_uniform.pkl'))
    assert rows['ok'].all() and len(rows) == 1 + len(get_param_names('be'))
    swapped = list(reversed(get_param_names('be')))
    rows = pd.DataFrame(parameter_rows(_network(names=swapped), 'snpe_pooled_BE_uniform.pkl'))
    assert not rows.set_index('check').loc['param_names', 'ok']
    rows = pd.DataFrame(parameter_rows(_network(prior_in_posterior=False), 'snpe_pooled_BE_uniform.pkl'))
    assert rows.set_index('check').loc['prior bounds', 'ok'] is None     # not recorded: unknown


def test_fingerprint_matches_itself_and_flags_a_shift():
    data = _network(design=SBI_DESIGN_FEATURES)
    X = simulate_features(data, n_draws=120, seed=7)
    data['feature_medians'] = compute_feature_medians(X)
    X2 = simulate_features(data, n_draws=120, seed=8)            # fresh draws, same simulator
    rows = pd.DataFrame(fingerprint_rows(data, 'snpe_pooled_BE_uniform.pkl', X=X2, n_boot=100))
    assert len(rows) == len(STATS) + len(SBI_DESIGN_FEATURES)
    assert rows['ok'].all(), rows[['check', 'stored', 'current', 'z']]
    data['feature_medians'] = data['feature_medians'] + np.r_[0.2, 0, 0, 0, 0]   # accuracy moved
    rows = pd.DataFrame(fingerprint_rows(data, 'snpe_pooled_BE_uniform.pkl', X=X2, n_boot=100))
    assert list(rows.loc[~rows['ok'], 'check']) == ['median accuracy']


def test_sessions_diff():
    old = pd.DataFrame({'animal_id': ['A', 'A', 'B', 'B'], 'kept': [True] * 4,
                        'date': ['2026-05-01', '2026-05-02', '2026-05-01', '2026-05-02'],
                        'n_valid': [300, 310, 280, 290]})
    new = old.copy()
    new.loc[3, 'date'], new.loc[3, 'n_valid'] = '2026-05-03', 295
    d = sessions_diff(old, new).set_index('animal_id')
    assert d.loc['A', 'same'] and not d.loc['B', 'same']
    assert 'dropped' in d.loc['B', 'detail'] and 'added' in d.loc['B', 'detail']


def test_load_raises_on_reordered_parameters(tmp_path):
    from sound_categorisation.inference.amortised import AmortisedSBI
    good, bad = tmp_path / 'good.pkl', tmp_path / 'bad.pkl'
    for path, names in ((good, None), (bad, list(reversed(get_param_names('be'))))):
        with open(path, 'wb') as f:
            pickle.dump(_network(names=names), f)
    net = AmortisedSBI.load(good)
    assert net.design_features == ()                          # an October-style file: none
    with pytest.raises(ValueError, match='mislabelled'):
        AmortisedSBI.load(bad)


def test_train_refuses_a_different_representation(tmp_path):
    from sound_categorisation.cli.sbi_train import representation_changes
    path = tmp_path / 'snpe_pooled_BE_uniform.pkl'
    with open(path, 'wb') as f:
        pickle.dump(_network(), f)
    changed = representation_changes(path, 'pooled', 'uniform')
    assert {'N', 'design_features', 'stat_names', 'burn_in'} <= set(changed)
