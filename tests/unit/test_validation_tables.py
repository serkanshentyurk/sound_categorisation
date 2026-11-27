"""cli.validation_tables.write_tables on fabricated runs: the October tables (real_votes.csv with --real and
--october, call_summary_compare.csv with --october-validation) next to the existing ones."""

import pandas as pd
from sound_categorisation.cli.validation_tables import write_tables
from sound_categorisation.inference.consensus import _method_dir
from sound_categorisation.inference.cv_utils import save_cv_result

DIST = 'uniform'
GS = ('grid_search', None, 'update_matrix')
POOLED = ('sbi', 'pooled', 'update_matrix')
SINGLE = ('sbi', 'single', 'update_matrix')
MOMENTS = ('sbi', 'moments', 'update_matrix')


def _write(run, method, animal, be_lo, sc_lo, true_model=None):
    """A paired BE/SC result over 8 reps in the neutral schema; BE wins iff be_lo < sc_lo."""
    d = _method_dir(*method, run, DIST)
    tp = {'sigma_percep': 0.2, 'eta_learning': 0.1} if true_model else None
    for model, lo in (('BE', be_lo), ('SC', sc_lo)):
        save_cv_result(d / f'{animal}_{model}.pkl', animal, model,
                       [{'rep': i, 'test_error': lo + 0.001 * i, 'best_params': {'sigma_percep': 0.2}}
                        for i in range(8)],
                       'update_matrix', true_model=true_model, true_params=tp)


def _runs(root):
    runs = {name: root / 'model_identification' / name / 'r'
            for name in ('val', 'val_oct', 'real', 'real_oct')}
    for method in (GS, POOLED):
        _write(runs['val'], method, 'A', 0.10, 0.20, 'BE')
        _write(runs['val'], method, 'B', 0.20, 0.10, 'SC')
    for method in (GS, POOLED, MOMENTS):
        _write(runs['val_oct'], method, 'A', 0.10, 0.20, 'BE')
        _write(runs['val_oct'], method, 'B', 0.10, 0.20, 'SC')         # the October networks call B wrong
    for method in (GS, POOLED, SINGLE):
        _write(runs['real'], method, 'R1', 0.10, 0.20)
    for method in (GS, POOLED, MOMENTS):
        _write(runs['real_oct'], method, 'R1', 0.20, 0.10)
    return runs


def test_october_tables(tmp_path):
    runs = _runs(tmp_path)
    out = write_tables(runs['val'], DIST, real=runs['real'], october=runs['real_oct'],
                       october_validation=runs['val_oct'])
    comp = pd.read_csv(out / 'call_summary_compare.csv')
    assert comp.columns[0] == 'run' and set(comp['run']) == {'current', 'october'}
    assert 'SBI-moments-UM' in set(comp.loc[comp['run'] == 'october', 'voter'])
    assert 'SBI-moments-UM' not in set(comp.loc[comp['run'] == 'current', 'voter'])
    oct_sc = comp[(comp['run'] == 'october') & (comp['rule'] == 1) & (comp['voter'] == 'Consensus')
                  & (comp['true_model'] == 'SC')].iloc[0]
    assert (oct_sc['wrong'], oct_sc['n']) == (1, 1)                    # scored with today's two voters
    current = comp[comp['run'] == 'current'].drop(columns='run').reset_index(drop=True)
    pd.testing.assert_frame_equal(current, pd.read_csv(out / 'call_summary.csv'))

    votes = pd.read_csv(out / 'real_votes.csv')
    methods = votes.groupby('run')['method'].apply(set).to_dict()
    assert methods == {'current': {'GS-UM', 'SBI-pooled-UM', 'SBI-single-UM'},
                       'october': {'GS-UM', 'SBI-pooled-UM', 'SBI-moments-UM'}}
    listed = votes.loc[~votes['voting'], ['run', 'method']].apply(tuple, axis=1).tolist()
    assert sorted(listed) == [('current', 'SBI-single-UM'), ('october', 'SBI-moments-UM')]
    assert (votes.loc[votes['run'] == 'october', 'call'] == 'SC').all()
    calls = pd.read_csv(out / 'real_calls.csv').set_index('id')
    assert calls.loc['R1', 'october_rule1'] == 'SC' and calls.loc['R1', 'current_rule1'] == 'BE'


def test_without_the_october_options_nothing_new_is_written(tmp_path):
    runs = _runs(tmp_path)
    out = write_tables(runs['val'], DIST, real=runs['real'])
    assert not (out / 'call_summary_compare.csv').exists()
    votes = pd.read_csv(out / 'real_votes.csv')
    assert set(votes['run']) == {'current'}
