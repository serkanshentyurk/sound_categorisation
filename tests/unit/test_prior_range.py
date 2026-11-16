"""sc-sbi-range: where an animal's statistics fall in what the models simulate (inference/prior_range.py)."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sound_categorisation.cli.sbi_range import find_run, write_comparison
from sound_categorisation.data.cohort import AnimalRecord
from sound_categorisation.inference.constants import SBI_STATS
from sound_categorisation.inference.prior_range import (
    ANIMAL_COLUMNS,
    COMPARISON_COLUMNS,
    MIN_REFERENCE,
    RECOGNISED_COLUMNS,
    SIMULATED_COLUMNS,
    SUMMARY_COLUMNS,
    comparison_tables,
    group_designs,
    is_outside,
    nearest_percentile,
    percentile_of,
    range_tables,
    read_acceptance,
    recognised,
    recognised_rows,
    simulate_draw,
    simulate_reference,
)
from sound_categorisation.inference.simulator import (
    build_simulator,
    get_bounds_arrays,
    get_param_names,
    simulate_sessions,
)
from sound_categorisation.models.priors import sample_prior
from sound_categorisation.settings import SBI_DESIGN_FEATURES, SBI_REPRESENTATIONS

SMALL = dict(distribution='uniform', T=(40, 60), burn_in=50, stat_names=SBI_STATS,
             design_features=SBI_DESIGN_FEATURES)
REPS = {'pooled': 'pooled', 'moments': 'moments'}


@pytest.mark.parametrize('model', ['be', 'sc'])
@pytest.mark.parametrize('n_sessions', [(2, 4), 3])
@pytest.mark.parametrize('prior', ['learning', 'manuscript'])
def test_a_draw_is_what_the_training_simulator_gives(model, n_sessions, prior):
    """The reference simulations must be the networks' training distribution, draw for draw: the
    same parameters, and the same simulation of each."""
    names, thetas = sample_prior(model, prior, 3, np.random.default_rng(1))
    assert names == get_param_names(model)
    theta = thetas[0]
    draw = simulate_draw(model, dict(zip(names, theta)), 77, N=n_sessions, reps=REPS, **SMALL)
    for mode in REPS:
        sim, _, _ = build_simulator(model, 'uniform', N=n_sessions, T=SMALL['T'], burn_in=SMALL['burn_in'],
                                    mode=mode, design_features=SBI_DESIGN_FEATURES, prior=prior)
        np.testing.assert_array_equal(draw['x'][mode], sim(theta, seed=77))
    assert 0.0 <= draw['min_accuracy'] <= 1.0


def test_manuscript_draws_are_the_first_networks_prior():
    lo, hi = get_bounds_arrays('be', 'manuscript')
    _, thetas = sample_prior('be', 'manuscript', 3, np.random.default_rng(1))
    np.testing.assert_array_equal(thetas, np.random.default_rng(1).uniform(lo, hi, size=(3, len(lo))))


@pytest.mark.parametrize('model', ['be', 'sc'])
def test_reference_draws_follow_the_prior_variant(model):
    """Another prior is other parameter draws through the same seeds: different statistics."""
    kwargs = dict(N=2, reps={'pooled': 'pooled'}, n_draws=4, seed=5, **SMALL)
    plain = simulate_reference(model, prior='manuscript', **kwargs)
    wider = simulate_reference(model, prior='learning', **kwargs)
    assert plain['names'] == get_param_names(model) == wider['names']
    assert plain['theta'].shape == (4, 4) == wider['theta'].shape
    assert not np.array_equal(wider['theta'], plain['theta'])
    assert not np.array_equal(wider['x']['pooled'], plain['x']['pooled'], equal_nan=True)


def test_representations_with_the_same_sessions_share_their_simulations():
    groups = group_designs(SBI_REPRESENTATIONS, ('pooled', 'moments', 'single'))
    assert [g['reps'] for g in groups] == [{'pooled': 'pooled', 'moments': 'moments'}, {'single': 'pooled'}]
    assert groups[0]['N'] == SBI_REPRESENTATIONS['pooled']['N'] and groups[1]['N'] == 1


def test_percentile_and_outside():
    ref = np.array([0.0, 1.0, 2.0, 3.0, np.nan])
    assert percentile_of(1.5, ref) == 50.0 and percentile_of(1.0, ref) == 37.5       # a tie counts half
    assert percentile_of(-1.0, ref) == 0.0 and percentile_of(9.0, ref) == 100.0
    assert np.isnan(percentile_of(np.nan, ref)) and np.isnan(percentile_of(1.0, np.array([np.nan])))
    assert is_outside(100.0, 200) is True and is_outside(0.2, 200) is True
    assert is_outside(50.0, 200) is False and is_outside(np.nan, 200) is False       # missing is not outside
    assert is_outside(100.0, MIN_REFERENCE - 1) is None                              # too few draws to say


def test_nearest_percentile_separates_a_simulated_point_from_a_far_one():
    X = np.random.default_rng(0).normal(size=(300, 4))
    assert nearest_percentile(X[0], X[1:]) < 99
    assert nearest_percentile(X[0] + 20.0, X) == 100.0
    assert np.isnan(nearest_percentile(X[0], X[:5]))                                  # fewer than k + 1 rows
    with_gaps = X.copy()
    with_gaps[::7, 2] = np.nan                                                        # NaN cells are imputed
    assert np.isfinite(nearest_percentile(X[0], with_gaps))


def _record(name, model, theta, seed):
    sessions = simulate_sessions(model, dict(zip(get_param_names(model), theta)), ['uniform'] * 3, 50, 50,
                                 np.random.default_rng(seed))
    return AnimalRecord(name, sessions, model.upper(), None)


@pytest.fixture(scope='module')
def tables():
    lo, hi = get_bounds_arrays('be', 'manuscript')
    records = [_record('inside', 'be', (lo + hi) / 2, 3)]
    representations = {'pooled': {'N': (2, 4), 'T': (40, 60), 'mode': 'pooled'},
                       'moments': {'N': (2, 4), 'T': (40, 60), 'mode': 'moments'}}
    return range_tables(records, distribution='uniform', representations=representations,
                        reps=('pooled', 'moments'), models=('BE',), stat_names=SBI_STATS,
                        design_features=SBI_DESIGN_FEATURES, burn_in=50, min_accuracy=0.99,
                        n_draws=MIN_REFERENCE + 10, seed=11, prior='manuscript', log=lambda line: None)


def test_range_tables_shape(tables):
    assert list(tables['simulated'].columns) == SIMULATED_COLUMNS
    assert list(tables['animals'].columns) == ANIMAL_COLUMNS
    assert list(tables['summary'].columns) == SUMMARY_COLUMNS
    n_pooled, n_moments = len(SBI_STATS) + 2, 2 * len(SBI_STATS) + 2
    a = tables['animals']
    assert len(a[a.rep == 'pooled']) == n_pooled and len(a[a.rep == 'moments']) == n_moments
    s = tables['simulated']
    assert set(s['set']) == {'prior', 'passing'} and len(s) == 2 * (n_pooled + n_moments)
    assert (s.loc[s['set'] == 'prior', 'n'] == MIN_REFERENCE + 10).all()
    summary = tables['summary'].set_index('rep')
    assert summary.loc['pooled', 'n_features'] == n_pooled and summary.loc['pooled', 'true_model'] == 'BE'
    assert np.isnan(summary['acceptance_rate']).all()                     # no saved posterior was given
    assert list(tables['recognised'].columns) == RECOGNISED_COLUMNS
    assert tables['recognised'].empty                                     # one model: nothing to tell apart


def test_an_animal_from_the_prior_is_inside_and_a_tiny_passing_set_gives_no_verdict(tables):
    """The animal was simulated at the middle of the BE prior, so it is not far from the BE draws; no
    BE draw reaches 99 % accuracy in every session, so the passing set is empty and says nothing."""
    summary = tables['summary']
    assert (summary['nearest_percentile'] < 99.5).all()
    assert (summary['n_passing'] == 0).all() and np.isnan(summary['n_outside_passing']).all()
    a = tables['animals']
    assert a['outside_passing'].isna().all() and a['percentile'].between(0, 100).all()


def test_read_acceptance(tmp_path):
    with open(tmp_path / 'SS01_BE.pkl', 'wb') as f:
        pickle.dump({'posterior': {'acceptance_rate': 0.25}}, f)
    with open(tmp_path / 'SS01_SC.pkl', 'wb') as f:
        pickle.dump({'posterior': None}, f)
    assert read_acceptance(tmp_path, 'SS01', 'be') == 0.25
    assert np.isnan(read_acceptance(tmp_path, 'SS01', 'SC')) and np.isnan(read_acceptance(tmp_path, 'SS02', 'BE'))


def test_read_acceptance_is_of_a_network_of_the_asked_prior(tmp_path):
    """A posterior that records no prior came from a manuscript network."""
    with open(tmp_path / 'SS01_BE.pkl', 'wb') as f:
        pickle.dump({'posterior': {'acceptance_rate': 0.25}}, f)
    with open(tmp_path / 'SS02_BE.pkl', 'wb') as f:
        pickle.dump({'posterior': {'acceptance_rate': 0.5, 'prior': 'learning'}}, f)
    assert read_acceptance(tmp_path, 'SS01', 'BE', prior='manuscript') == 0.25
    assert np.isnan(read_acceptance(tmp_path, 'SS01', 'BE', prior='learning'))
    assert read_acceptance(tmp_path, 'SS02', 'BE', prior='learning') == 0.5
    assert np.isnan(read_acceptance(tmp_path, 'SS02', 'BE', prior='manuscript'))


def test_read_acceptance_without_the_module_a_result_was_saved_with(tmp_path, monkeypatch):
    """Results are saved on the cluster with torch objects in their metadata and read on a laptop that has
    no torch: the reader skips what it cannot import and still finds the rate."""
    import sys
    import types

    cluster_only = types.ModuleType('cluster_only_module')

    class Tensor:
        def __init__(self, values=()):
            self.values = list(values)

    Tensor.__module__, Tensor.__qualname__ = 'cluster_only_module', 'Tensor'
    cluster_only.Tensor = Tensor
    monkeypatch.setitem(sys.modules, 'cluster_only_module', cluster_only)
    with open(tmp_path / 'SS01_BE.pkl', 'wb') as f:
        pickle.dump({'posterior': {'acceptance_rate': 0.4, 'samples': np.zeros((3, 2))},
                     'metadata': {'prior_low': Tensor([0.0, 1.0]), 'nested': [Tensor(), {'t': Tensor([2.0])}]}}, f)
    monkeypatch.delitem(sys.modules, 'cluster_only_module')
    with pytest.raises(ModuleNotFoundError), open(tmp_path / 'SS01_BE.pkl', 'rb') as f:
        pickle.load(f)                                                    # the plain reader cannot
    assert read_acceptance(tmp_path, 'SS01', 'BE') == 0.4
    (tmp_path / 'SS02_BE.pkl').write_bytes(b'not a pickle')
    assert np.isnan(read_acceptance(tmp_path, 'SS02', 'BE'))             # unreadable: missing, not an error


def test_find_run_uses_the_latest_run_or_starts_one(tmp_path, monkeypatch):
    monkeypatch.setenv('SC_RESULTS_ROOT', str(tmp_path))
    first = find_run('some-cohort', 'latest')                             # none yet: a new run
    assert first.is_dir() and first.parent.name == 'some-cohort'
    assert find_run('some-cohort', 'latest') == first
    assert find_run('some-cohort', first.name) == first
    with pytest.raises(FileNotFoundError, match='runs there'):
        find_run('some-cohort', '1999-01-01_0000_abcdefg')


def test_recognised_is_high_for_models_that_differ_and_chance_for_models_that_do_not():
    rng = np.random.default_rng(0)
    apart = recognised({'BE': rng.normal(0.0, 1.0, (200, 3)), 'SC': rng.normal(4.0, 1.0, (200, 3))})
    assert apart['BE'].shape == (200,) and apart['BE'].mean() > 0.95 and apart['SC'].mean() > 0.95
    same = recognised({'BE': rng.normal(size=(300, 3)), 'SC': rng.normal(size=(300, 3))})
    assert 0.35 < np.mean([same['BE'].mean(), same['SC'].mean()]) < 0.65
    with_gaps = rng.normal(0.0, 1.0, (200, 3))
    with_gaps[::5, 1] = np.nan                                             # a failed fit is a missing value
    assert recognised({'BE': with_gaps, 'SC': rng.normal(4.0, 1.0, (200, 3))})['BE'].mean() > 0.95


def test_recognised_rows_over_every_draw_and_over_the_passing_draws():
    """The two models differ only among the passing draws here: recognised there, at chance overall."""
    rng = np.random.default_rng(1)
    be, sc = rng.normal(size=(400, 2)), rng.normal(size=(400, 2))
    passing = {'BE': np.arange(400) < 100, 'SC': np.arange(400) < 100}
    be[:100] += 6.0
    sc[:100] -= 6.0
    rows = {r['model']: r for r in recognised_rows('pooled', {'BE': be, 'SC': sc}, passing)}
    assert rows['BE']['n_prior'] == 400 and rows['BE']['n_passing'] == 100
    assert rows['BE']['recognised_passing'] > 0.95 and rows['SC']['recognised_passing'] > 0.95
    assert rows['BE']['recognised'] < 0.8
    assert recognised_rows('pooled', {'BE': be}, passing) == []                       # one model
    assert recognised_rows('pooled', {'BE': be[:10], 'SC': sc}, passing) == []        # too few draws


def _summary(rows):
    """A summary table from (animal, rep, model, n_outside, n_outside_passing, nearest_percentile) rows."""
    names = ['animal_id', 'rep', 'model', 'n_outside', 'n_outside_passing', 'nearest_percentile']
    return pd.DataFrame([dict(zip(names, r), n_prior=100, n_passing=60) for r in rows]).reindex(
        columns=SUMMARY_COLUMNS)


MANUSCRIPT = [('a', 'pooled', 'BE', 0, 0, 50.0), ('a', 'pooled', 'SC', 1, np.nan, 50.0),
              ('b', 'pooled', 'BE', 0, 2, 99.5), ('b', 'pooled', 'SC', 3, np.nan, 10.0)]
LEARNING = [('a', 'pooled', 'BE', 0, 0, 50.0), ('a', 'pooled', 'SC', 0, 0, 50.0),
        ('b', 'pooled', 'BE', 3, 2, 20.0), ('b', 'pooled', 'SC', 0, 1, 10.0)]


def test_comparison_counts_the_animals_each_prior_covers():
    """Covered = no feature outside the prior set and a simulation nearby (nearest_percentile <= 99)."""
    tables = comparison_tables({'learning': _summary(LEARNING), 'manuscript': _summary(MANUSCRIPT)})
    c = tables['comparison']
    assert list(c.columns) == COMPARISON_COLUMNS and list(c['prior']) == ['manuscript'] * 2 + ['learning'] * 2
    c = c.set_index(['prior', 'model'])
    assert c.loc[('manuscript', 'BE'), ['n_inside', 'n_near', 'n_covered']].tolist() == [2, 1, 1]
    assert c.loc[('manuscript', 'SC'), ['n_inside', 'n_near', 'n_covered']].tolist() == [0, 2, 0]
    assert c.loc[('manuscript', 'SC'), 'n_covered_passing'] == 0      # the passing set was too small to say
    assert c.loc[('learning', 'SC'), ['n_covered', 'n_covered_passing']].tolist() == [2, 1]
    animals = tables['comparison_animals'].set_index('animal_id')
    assert animals.loc['a', ['manuscript', 'learning']].tolist() == ['BE', 'BE+SC']
    assert animals.loc['b', ['manuscript', 'learning']].tolist() == ['none', 'SC']
    assert c['recognised'].isna().all()                              # no recognised table was given


def test_comparison_carries_what_each_prior_costs():
    told = pd.DataFrame([{'rep': 'pooled', 'model': 'BE', 'n_prior': 100, 'n_passing': 60, 'recognised': 0.9,
                          'recognised_passing': 0.6},
                         {'rep': 'pooled', 'model': 'SC', 'n_prior': 100, 'n_passing': 60, 'recognised': 0.95,
                          'recognised_passing': 0.97}])
    c = comparison_tables({'manuscript': _summary(MANUSCRIPT), 'learning': _summary(LEARNING)},
                          {'learning': told})['comparison'].set_index(['prior', 'model'])
    assert c.loc[('learning', 'BE'), ['recognised', 'recognised_passing']].tolist() == [0.9, 0.6]
    assert c.loc[('learning', 'SC'), 'recognised_passing'] == 0.97
    assert np.isnan(c.loc[('manuscript', 'BE'), 'recognised'])


def test_write_comparison_reads_the_priors_simulated_so_far(tmp_path):
    assert write_comparison(tmp_path) == {}                                # nothing simulated yet
    for prior, rows in (('manuscript', MANUSCRIPT), ('learning', LEARNING)):
        (tmp_path / prior).mkdir()
        _summary(rows).to_csv(tmp_path / prior / 'summary.csv', index=False)
    tables = write_comparison(tmp_path)
    assert set(tables['comparison']['prior']) == {'manuscript', 'learning'}
    saved = pd.read_csv(tmp_path / 'comparison_animals.csv')
    assert list(saved.columns) == ['animal_id', 'rep', 'manuscript', 'learning']
    assert len(pd.read_csv(tmp_path / 'comparison.csv')) == 4
