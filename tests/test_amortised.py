"""Tests for inference.amortised.AmortisedSBI.

torch + sbi are required for training/conditioning, so the whole module is
skipped where they are unavailable. The non-torch scaffolding (constructor
wiring, accessors, and the pre-train raises) is exercised by the first class,
which only needs the package import.
"""

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('sbi')

from inference import AmortisedSBI
from inference.types import ModelType, get_default_param_configs
from models.simulate import simulate_choices
from behav_utils.data.synthetic import session_from_arrays
from utils.stimulus_distributions import sample_distribution

# Small, fast configuration for the test run.
STATS = ['accuracy', 'psychometric', 'side_bias', 'win_stay']
N, T, BURN = 3, 150, 50
BE_NAMES = ['sigma_percep', 'A_repulsion', 'eta_learning', 'eta_relax']


def _mid(model):
    return {n: 0.5 * (c.bounds[0] + c.bounds[1])
            for n, c in get_default_param_configs(model).items()}


def _be_sessions(n=N, t=T, seed0=0):
    be = _mid(ModelType.BE)
    out = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        s, c = sample_distribution(t, 'uniform', rng=rng)
        ch = simulate_choices(ModelType.BE, be, s, c, burn_in=BURN,
                              seed=seed0 + i)
        out.append(session_from_arrays(s, ch, c, session_idx=i,
                                       distribution='uniform'))
    return out


def _new_net():
    return AmortisedSBI(ModelType.BE, dist_schedule='uniform', N=N, T=T,
                        burn_in=BURN, mode='pooled', stat_names=STATS)


@pytest.fixture(scope='module')
def trained_net():
    net = _new_net()
    net.train(n_simulations=500, seed=0, show_progress=False)
    return net


class TestInitAndGuards:
    """No training needed -- just the constructor wiring and guard rails."""

    def test_wiring(self):
        net = _new_net()
        assert net.model == ModelType.BE
        assert net.param_names == BE_NAMES
        assert net.mode == 'pooled'
        assert net.N == N and net.T == T and net.burn_in == BURN
        assert net.stat_names == STATS

    def test_accessors(self):
        net = _new_net()
        assert callable(net.simulator)      # sim_fn theta -> x
        assert net.prior is not None        # BoxUniform (torch present)
        assert net.posterior is None        # not trained yet

    def test_condition_before_train_raises(self):
        with pytest.raises(RuntimeError):
            _new_net().condition(_be_sessions())

    def test_save_before_train_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            _new_net().save(tmp_path / 'untrained.pkl')


class TestTrainCondition:

    def test_train_sets_posterior(self, trained_net):
        assert trained_net.posterior is not None
        assert trained_net._training_metadata['n_valid'] > 0

    def test_condition_shapes(self, trained_net):
        out = trained_net.condition(_be_sessions(seed0=50), n_samples=200)
        n_params = len(trained_net.param_names)
        assert out['posterior_samples'].shape == (200, n_params)
        assert set(out['point_estimate'].keys()) == set(trained_net.param_names)
        assert out['theta_median'].shape == (n_params,)
        assert np.all(np.isfinite(out['theta_median']))


class TestSaveLoad:

    def test_roundtrip(self, trained_net, tmp_path):
        path = tmp_path / 'net.pkl'
        trained_net.save(path)
        loaded = AmortisedSBI.load(path)

        assert loaded.model == trained_net.model
        assert loaded.param_names == trained_net.param_names
        assert loaded.mode == trained_net.mode
        assert loaded.stat_names == trained_net.stat_names
        assert loaded.N == trained_net.N and loaded.T == trained_net.T

        # the loaded posterior is usable
        out = loaded.condition(_be_sessions(seed0=99), n_samples=100)
        assert out['posterior_samples'].shape == (100, len(loaded.param_names))
