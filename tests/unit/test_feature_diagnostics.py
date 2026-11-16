"""SBI feature selection (features/feature_diagnostics.py) on a small simulated cohort: the tables
sc-feature-selection writes have the right shape and sensible values. Not a check of which statistics win;
that needs the full run."""

import numpy as np
import pytest
from sound_categorisation.features.feature_diagnostics import (
    forward_selection,
    list_producers,
    simulate_selection_cohort,
    stat_correlation,
    stat_individual_power,
)

POOL = ['accuracy', 'win_stay', 'lose_shift', 'recency']
N_SIMS = 40


@pytest.fixture(scope='module')
def cohort():
    return simulate_selection_cohort('uniform', n_sims=N_SIMS, n_trials=300, stat_pool=POOL, seed=0)


def test_cohort_holds_both_models_and_the_pool(cohort):
    assert cohort.predictors == POOL and set(cohort.pred_groups) == set(POOL)
    for model in ('be', 'sc'):
        n = cohort.n_valid[model]
        assert 0 < n <= N_SIMS
        assert cohort.theta[model].shape == (n, len(cohort.param_names[model]))
        assert cohort.Xp(model).shape == (n, len(cohort.pred_cols()))
        assert np.isfinite(cohort.Xp(model)).all()          # predictor columns are finite by construction


def test_the_cohort_is_reproducible():
    a = simulate_selection_cohort('uniform', n_sims=4, n_trials=120, stat_pool=POOL[:2], seed=3)
    b = simulate_selection_cohort('uniform', n_sims=4, n_trials=120, stat_pool=POOL[:2], seed=3)
    for model in ('be', 'sc'):
        np.testing.assert_array_equal(a.theta[model], b.theta[model])
        np.testing.assert_array_equal(a.Xp(model), b.Xp(model))


def test_each_stat_alone(cohort):
    power = stat_individual_power(cohort, scorer='ridge', n_splits=3)
    identity = power['identity']
    assert list(identity.index) == POOL and identity.between(0, 1).all()
    assert identity.max() > 0.6                              # the history statistics tell BE from SC
    for model in ('be', 'sc'):
        recovery = power['recovery'][model]
        assert list(recovery.index) == POOL and list(recovery.columns) == cohort.param_names[model]
        assert np.isfinite(recovery.to_numpy()).all()


def test_stat_correlation_is_a_correlation_matrix(cohort):
    corr = stat_correlation(cohort, 'be')['corr']
    assert corr.shape[0] == corr.shape[1] > 0
    values = corr.to_numpy(dtype=float)
    np.testing.assert_allclose(np.diag(values), 1.0)
    np.testing.assert_allclose(values, values.T, atol=1e-12)
    assert (np.abs(values) <= 1 + 1e-12).all()


def test_forward_selection_adds_every_stat_once(cohort):
    result = forward_selection(cohort, pool=POOL, scorer='ridge', n_splits=3, n_jobs=1)
    curve = result['curve']
    assert list(curve['k']) == [1, 2, 3, 4] and sorted(curve['added']) == sorted(POOL)
    assert curve['identity_auc'].between(0, 1).all() and set(curve['phase']) <= {'identity', 'recovery'}
    assert set(result['suggested']) <= set(POOL) and result['suggested_k'] == len(result['suggested'])


def test_every_producer_names_its_statistics():
    producers = list_producers()
    assert set(POOL) <= set(producers) and all(len(names) >= 1 for names in producers.values())
