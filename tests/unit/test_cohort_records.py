"""data.cohort.select_animals / load_animals and the saved SBI posterior.

Synthetic path only (the real path needs the snapshot): a cohort pickle is written under a
temporary SC_DATA_ROOT, then the min-session rule and the per-session table are checked.
"""

import numpy as np
import pytest
from sound_categorisation.cli.make_synthetic_cohort import make_cohort
from sound_categorisation.data.cohort import is_synthetic_cohort, load_animals, select_animals
from sound_categorisation.inference.cv_utils import load_cv_results, save_cv_result


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    monkeypatch.setenv('SC_DATA_ROOT', str(tmp_path))
    make_cohort('synthetic_test', distribution='uniform', n_per_model=2, n_sessions=5,
                trials=60, burn_in=20, seed=1)
    return 'synthetic_test'


def test_synthetic_cohort_records_and_table(cohort):
    assert is_synthetic_cohort(cohort)
    records, table = select_animals(cohort)
    assert [r.animal_id for r in records] == ['BE00', 'BE01', 'SC00', 'SC01']
    assert all(len(r.sessions) == 5 for r in records)
    assert all(r.true_model in ('BE', 'SC') and r.true_params for r in records)
    assert set(table.columns) >= {'animal_id', 'kept', 'reason', 'n_sessions', 'session_idx',
                                  'date', 'n_valid', 'accuracy', 'true_model'}
    assert len(table) == 20 and table.kept.all() and (table.reason == '').all()
    assert table.groupby('animal_id').session_idx.apply(list).map(lambda x: x == [0, 1, 2, 3, 4]).all()


def test_min_sessions_drops_and_lists(cohort):
    records, table = select_animals(cohort, min_sessions=6)
    assert records == []
    assert (~table.kept).all() and (table.reason == '5 sessions < 6').all()
    assert len(load_animals(cohort, min_sessions=5)) == 4
    assert [r.animal_id for r in load_animals(cohort)] == [r.animal_id for r in select_animals(cohort)[0]]


def test_truths_come_from_one_stream_per_model(cohort):
    records, _ = select_animals(cohort)
    be = [r.true_params for r in records if r.true_model == 'BE']
    assert be[0] != be[1]                                 # two draws, not a repeated first draw


def test_save_cv_result_keeps_posterior(tmp_path):
    post = {'posterior_samples': np.zeros((10, 4)), 'param_names': ['a', 'b', 'c', 'd'],
            'point_estimate': {'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0},
            'prior_bounds': {'a': (0, 1), 'b': (0, 1), 'c': (0, 1), 'd': (0, 1)}}
    res = [{'rep': 0, 'test_error': 0.1, 'best_params': post['point_estimate']}]
    for model in ('BE', 'SC'):
        save_cv_result(tmp_path / f'X_{model}.pkl', 'X', model, res, 'update_matrix',
                       true_model='BE', true_params={'a': 0.5}, posterior=post if model == 'BE' else None)
    import pickle
    d = pickle.load(open(tmp_path / 'X_BE.pkl', 'rb'))
    assert d['posterior']['posterior_samples'].shape == (10, 4)
    assert pickle.load(open(tmp_path / 'X_SC.pkl', 'rb'))['posterior'] is None
    cv = load_cv_results(tmp_path)                        # the loader ignores the extra key
    assert len(cv.comparison) == 1
