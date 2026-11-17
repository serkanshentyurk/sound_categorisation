"""Tests for inference.amortised.AmortisedSBI.

torch + sbi are required for training/conditioning, so the whole module is
skipped where they are unavailable. The non-torch scaffolding (constructor
wiring, accessors, and the pre-train raises) is exercised by the first class,
which only needs the package import.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.fit

torch = pytest.importorskip('torch')
pytest.importorskip('sbi')

from behav_utils.data import session_from_arrays  # noqa: E402
from sound_categorisation.data.stimuli import sample_distribution  # noqa: E402
from sound_categorisation.inference import AmortisedSBI  # noqa: E402
from sound_categorisation.inference.training_sets import read_training_set  # noqa: E402
from sound_categorisation.inference.types import ModelType, get_default_param_configs  # noqa: E402
from sound_categorisation.models.simulate import simulate_choices  # noqa: E402

# Small, fast configuration for the test run.
STATS = ['accuracy', 'mu', 'sigma', 'lapse_low', 'lapse_high', 'side_bias', 'win_stay']
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

    def test_feature_medians_stored(self, trained_net):
        med = trained_net._feature_medians
        assert med is not None
        assert np.ndim(med) == 1 and np.all(np.isfinite(med))

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
        assert np.allclose(loaded._feature_medians, trained_net._feature_medians)

        # the loaded posterior is usable
        out = loaded.condition(_be_sessions(seed0=99), n_samples=100)
        assert out['posterior_samples'].shape == (100, len(loaded.param_names))


class TestDesignFeaturesAndFiles:

    def test_saved_and_restored(self, trained_net, tmp_path):
        from sound_categorisation.inference.amortised import read_network_file, trained_prior_bounds
        from sound_categorisation.inference.simulator import get_bounds_arrays
        from sound_categorisation.settings import SBI_DESIGN_FEATURES
        assert trained_net.design_features == tuple(SBI_DESIGN_FEATURES)
        assert len(trained_net._feature_medians) == len(STATS) + len(SBI_DESIGN_FEATURES)
        path = tmp_path / 'net.pkl'
        trained_net.save(path)
        loaded = AmortisedSBI.load(path)
        assert loaded.design_features == trained_net.design_features
        lo, hi = trained_prior_bounds(read_network_file(path))
        assert np.allclose(lo, get_bounds_arrays('be')[0]) and np.allclose(hi, get_bounds_arrays('be')[1])

    def test_condition_reports_budget(self, trained_net):
        out = trained_net.condition(_be_sessions(seed0=5), n_samples=50)
        assert isinstance(out['budget_hit'], bool)

    def test_file_without_design_features(self, tmp_path):
        """An October-style file (no design features, no stored bounds) loads, conditions on the statistics
        alone with a warning, and still gives its prior bounds from the pickled sbi posterior."""
        import pickle

        from sound_categorisation.inference.amortised import read_network_file, trained_prior_bounds
        net = AmortisedSBI(ModelType.BE, dist_schedule='uniform', N=N, T=T, burn_in=BURN, mode='pooled',
                           stat_names=STATS, design_features=())
        net.train(n_simulations=300, seed=1, show_progress=False)
        path = tmp_path / 'old.pkl'
        net.save(path)
        data = read_network_file(path)
        data.pop('design_features')
        data.pop('prior_bounds')
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        loaded = AmortisedSBI.load(path)
        assert loaded.design_features == ()
        with pytest.warns(UserWarning, match='design features'):
            out = loaded.condition(_be_sessions(seed0=6), n_samples=50)
        assert out['posterior_samples'].shape == (50, len(BE_NAMES))
        assert trained_prior_bounds(read_network_file(path)) is not None



RULE = 0.62                    # an accuracy rule about half of these short BE simulations pass


@pytest.fixture(scope='module')
def ruled_net():
    net = AmortisedSBI(ModelType.BE, dist_schedule='uniform', N=(2, 3), T=(60, 90), burn_in=BURN,
                       mode='pooled', stat_names=STATS, min_accuracy=RULE)
    net.train(n_simulations=300, seed=0, show_progress=False)
    return net


class TestPriorVariantAndAccuracyRule:
    """A network belongs to a prior variant and, optionally, to the accuracy rule. Inside it works
    in flat space (BE's two rates as logarithms under 'learning'); what comes out is in the
    parameters' own units."""

    RULE = RULE

    def test_a_new_network_is_on_the_prior_in_use(self, trained_net):
        from sound_categorisation.settings import SBI_PRIOR
        assert trained_net.prior_name == SBI_PRIOR == 'learning' and trained_net.min_accuracy is None
        assert trained_net._training_metadata['prior'] == 'learning'

    def test_the_density_estimator_sees_flat_space(self, trained_net):
        from sound_categorisation.inference.amortised import _as_np
        from sound_categorisation.models.priors import flat_bounds
        low, high = flat_bounds('be', 'learning')
        base = trained_net.prior.base_dist
        np.testing.assert_allclose(_as_np(base.low), low, rtol=1e-6)
        np.testing.assert_allclose(_as_np(base.high), high, rtol=1e-6)
        assert low[2] == pytest.approx(np.log(0.005)) and low[0] == pytest.approx(0.05)

    def test_posterior_samples_are_in_own_units_inside_the_prior(self, trained_net):
        out = trained_net.condition(_be_sessions(seed0=20), n_samples=300)
        low, high = trained_net.bounds()
        s = out['posterior_samples']
        assert np.all(s >= low - 1e-9) and np.all(s <= high + 1e-9)
        assert s[:, 2].min() > 0 and s[:, 3].min() > 0              # rates, not their logarithms
        assert 0.0 <= out['acceptance_rate'] <= 1.0
        assert all(low[i] <= v <= high[i] for i, v in enumerate(out['theta_median']))

    def test_rule_keeps_passing_simulations_and_records_the_draws(self, ruled_net):
        meta = ruled_net._training_metadata
        assert meta['n_simulations'] == 300 and meta['n_drawn'] > 300
        assert 300 / meta['n_drawn'] <= meta['pass_rate'] < 1 and meta['min_accuracy'] == self.RULE
        assert meta['simulation_time'] > 0 and meta['fit_time'] >= 0
        assert meta['training_time'] >= meta['simulation_time']
        # every kept simulation passed the rule, so the stored median accuracy is above it
        assert ruled_net._feature_medians[STATS.index('accuracy')] >= self.RULE

    def test_prior_and_rule_survive_save_and_load(self, ruled_net, tmp_path):
        from sound_categorisation.inference.amortised import network_prior, read_network_file
        path = tmp_path / 'ruled.pkl'
        ruled_net.save(path)
        data = read_network_file(path)
        assert network_prior(data) == 'learning' and data['min_accuracy'] == self.RULE
        assert data['log_uniform'] == ['eta_learning', 'eta_relax'] and data['_version'] == 7
        loaded = AmortisedSBI.load(path)
        assert loaded.prior_name == 'learning' and loaded.min_accuracy == self.RULE
        out = loaded.condition(_be_sessions(seed0=30), n_samples=100)
        assert out['posterior_samples'].shape == (100, len(BE_NAMES))

    def test_a_file_from_before_the_variants_is_a_manuscript_network(self, tmp_path):
        """Such a file records no prior and no rule: it loads on the manuscript ranges, without a
        rule, samples inside those ranges, and says so once when conditioned."""
        import pickle

        from sound_categorisation.inference.amortised import read_network_file
        from sound_categorisation.models.priors import bounds_arrays
        net = AmortisedSBI(ModelType.BE, dist_schedule='uniform', N=N, T=T, burn_in=BURN, mode='pooled',
                           stat_names=STATS, prior='manuscript')
        net.train(n_simulations=300, seed=2, show_progress=False)
        path = tmp_path / 'november.pkl'
        net.save(path)
        data = read_network_file(path)
        for key in ('prior', 'log_uniform', 'min_accuracy'):
            data.pop(key)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        loaded = AmortisedSBI.load(path)
        assert loaded.prior_name == 'manuscript' and loaded.min_accuracy is None
        with pytest.warns(UserWarning, match="'manuscript' prior"):
            out = loaded.condition(_be_sessions(seed0=7), n_samples=100)
        low, high = bounds_arrays('be', 'manuscript')
        assert np.all(out['posterior_samples'] >= low - 1e-9) and np.all(out['posterior_samples'] <= high + 1e-9)

    def test_condition_posterior_reports_the_networks_prior(self, trained_net):
        from sound_categorisation.inference.selection import condition_posterior
        post = condition_posterior(_be_sessions(seed0=40), trained_net, n_samples=100)
        assert post['prior'] == 'learning' and post['log_uniform'] == ['eta_learning', 'eta_relax']
        assert post['prior_bounds']['eta_learning'] == (0.005, 0.9)

    def test_sbc_runs_on_a_ruled_network(self, ruled_net):
        from sound_categorisation.inference.calibration import compute_sbc_ranks
        res = compute_sbc_ranks(ruled_net, n_draws=6, n_sessions=2, n_posterior_samples=50, seed=3,
                                show_progress=False)
        assert res['ranks'].shape == (6, len(BE_NAMES)) and res['prior'] == 'learning'
        assert res['min_accuracy'] == self.RULE and res['n_rejected'] >= 0
        assert res['theta_true'][:, 2].max() < 0                    # log eta_learning


def _no_simulating(*args, **kwargs):
    raise AssertionError('simulated again')


class TestTrainingSet:
    """The simulations are saved before the density estimator trains; a run started again with the
    same settings reads them back instead of simulating."""

    def test_saved_before_the_density_estimator_trains(self, tmp_path, monkeypatch):
        import sbi.inference

        def stop(self, *args, **kwargs):
            raise RuntimeError('density estimator stopped')
        monkeypatch.setattr(sbi.inference.SNPE, 'train', stop)
        net, path = _new_net(), tmp_path / 'set.npz'
        with pytest.raises(RuntimeError, match='density estimator stopped'):
            net.train(n_simulations=200, seed=3, show_progress=False, training_set=path)
        saved = read_training_set(path)
        assert saved.theta.shape == (200, len(BE_NAMES)) and saved.param_names == BE_NAMES
        assert net.training_set_changes(path, 200, 3) == [] and saved.n_drawn == 200

    def test_a_restart_reads_the_simulations_back(self, tmp_path, monkeypatch):
        path = tmp_path / 'set.npz'
        first = _new_net()
        first.train(n_simulations=200, seed=3, show_progress=False, training_set=path)
        monkeypatch.setattr(AmortisedSBI, 'simulate_training_set', _no_simulating)
        again = _new_net()
        again.train(n_simulations=200, seed=3, show_progress=False, training_set=path)
        m1, m2 = first._training_metadata, again._training_metadata
        assert not m1['training_set_reused'] and m2['training_set_reused'] and m2['training_set'] == 'set.npz'
        assert (m2['n_drawn'], m2['simulation_time']) == (m1['n_drawn'], m1['simulation_time'])
        assert m2['training_time'] >= m2['simulation_time']
        np.testing.assert_array_equal(again._feature_medians, first._feature_medians)
        out = again.condition(_be_sessions(seed0=60), n_samples=100)
        assert out['posterior_samples'].shape == (100, len(BE_NAMES))

    def test_a_set_made_with_other_settings_is_refused_before_simulating(self, tmp_path, monkeypatch):
        path = tmp_path / 'set.npz'
        _new_net().train(n_simulations=200, seed=3, show_progress=False, training_set=path)
        monkeypatch.setattr(AmortisedSBI, 'simulate_training_set', _no_simulating)
        with pytest.raises(FileExistsError, match='seed'):
            _new_net().train(n_simulations=200, seed=4, show_progress=False, training_set=path)
        with pytest.raises(FileExistsError, match='n_simulations'):
            _new_net().train(n_simulations=300, seed=3, show_progress=False, training_set=path)

    def test_a_set_that_cannot_be_saved_does_not_stop_training(self, tmp_path, monkeypatch):
        from sound_categorisation.inference import amortised

        def fail(*args, **kwargs):
            raise OSError('disk full')
        monkeypatch.setattr(amortised, 'save_training_set', fail)
        net = _new_net()
        with pytest.warns(UserWarning, match='could not save the training set'):
            net.train(n_simulations=200, seed=3, show_progress=False, training_set=tmp_path / 'set.npz')
        assert net.posterior is not None and net._training_metadata['training_set'] is None


class TestTrainCommand:
    """sc-sbi-train on a scratch data root: the four networks of the array, small."""

    def test_fast_training_writes_networks_with_the_current_settings(self, tmp_path, monkeypatch):
        monkeypatch.setenv('SC_DATA_ROOT', str(tmp_path))
        from sound_categorisation.cli import sbi_train
        from sound_categorisation.data.paths import snpe_net_path
        from sound_categorisation.inference.amortised import read_network_file
        from sound_categorisation.settings import SBI_REPRESENTATIONS, SBI_TRAIN_REPS
        monkeypatch.setattr(sbi_train, 'FAST_SBI_N_SIMULATIONS', 200)
        sbi_train.main(['--fast', '--model', 'SC'])
        for rep in SBI_TRAIN_REPS:
            path = snpe_net_path(rep, 'SC', 'uniform')
            assert path.exists() and path.is_relative_to(tmp_path)
            data = read_network_file(path)
            cfg = SBI_REPRESENTATIONS[rep]
            assert (data['N'], data['T'], data['prior']) == (cfg['N'], cfg['T'], 'learning')
            assert data['min_accuracy'] == (0.70 if cfg['accuracy_rule'] else None)
            assert data['training_metadata']['n_simulations'] == 200
            assert sbi_train.representation_changes(path, rep, 'uniform') == []
        # a second run finds the same representation and may replace it; another one is refused
        sbi_train.main(['--fast', '--rep', 'single', '--model', 'SC'])
        monkeypatch.setitem(SBI_REPRESENTATIONS['single'], 'T', (100, 300))
        with pytest.raises(FileExistsError, match='different representation'):
            sbi_train.main(['--fast', '--rep', 'single', '--model', 'SC'])

    def test_each_network_has_its_training_set_and_a_restart_reads_it_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv('SC_DATA_ROOT', str(tmp_path))
        from sound_categorisation.cli import sbi_train
        from sound_categorisation.data.paths import snpe_net_path, snpe_training_set_path
        from sound_categorisation.inference.amortised import read_network_file
        monkeypatch.setattr(sbi_train, 'FAST_SBI_N_SIMULATIONS', 200)
        args = ['--fast', '--rep', 'single', '--model', 'SC']
        sbi_train.main(args)
        network, saved = snpe_net_path('single', 'SC', 'uniform'), snpe_training_set_path('single', 'SC', 'uniform')
        assert read_training_set(saved).theta.shape == (200, 4)
        assert read_network_file(network)['training_metadata']['training_set'] == saved.name

        network.unlink()                                   # as if the job had stopped after simulating
        simulate = AmortisedSBI.simulate_training_set
        monkeypatch.setattr(AmortisedSBI, 'simulate_training_set', _no_simulating)
        sbi_train.main(args)
        assert read_network_file(network)['training_metadata']['training_set_reused']

        with pytest.raises(FileExistsError, match='seed'):  # another seed: refused before simulating
            sbi_train.main([*args, '--seed', '7'])
        monkeypatch.setattr(AmortisedSBI, 'simulate_training_set', simulate)
        sbi_train.main([*args, '--seed', '7', '--overwrite'])
        assert read_training_set(saved).config['seed'] == 7
        assert not read_network_file(network)['training_metadata']['training_set_reused']
