"""inference.validation — intervals, the simultaneous ECDF band, and the model-recovery tables."""

import numpy as np
import pandas as pd
import pytest
from sound_categorisation.inference import validation as V


def test_clopper_pearson():
    lo, hi = V.clopper_pearson(0, 50)
    assert lo == 0 and hi == pytest.approx(0.0711, abs=1e-3)       # the 7 % quoted for K = 50
    lo, hi = V.clopper_pearson(50, 50)
    assert hi == 1 and lo == pytest.approx(1 - 0.0711, abs=1e-3)
    lo, hi = V.clopper_pearson(10, 20)
    assert lo < 0.5 < hi
    assert np.isnan(V.clopper_pearson(0, 0)[0])


def test_binomial_band_contains_nominal():
    lo, hi = V.binomial_band(1000, 0.9)
    assert lo < 0.9 < hi and hi - lo < 0.05


def test_ecdf_band_is_simultaneous():
    """Uniform samples stay entirely inside the band about 95 % of the time."""
    z = np.arange(1, 31) / 31
    lo, hi = V.ecdf_band(200, z, n_sim=500, seed=1)
    rng = np.random.default_rng(2)
    inside = []
    for _ in range(300):
        ecdf = np.searchsorted(np.sort(rng.uniform(size=200)), z, side='right') / 200
        inside.append(np.all((ecdf >= lo) & (ecdf <= hi)))
    assert 0.9 <= np.mean(inside) <= 0.99


def _assign(rule):
    """Four BE and four SC animals; one wrong grid-search vote, one contested call."""
    rows = []
    for i, tm in enumerate(['BE'] * 4 + ['SC'] * 4):
        other = 'SC' if tm == 'BE' else 'BE'
        gs = other if i == 0 else tm
        rows.append({'id': f'{tm}{i}', 'true_model': tm,
                     'GS-UM': gs, 'GS-UM_p': 0.01,
                     'SBI-pooled-UM': tm, 'SBI-pooled-UM_p': 0.01 if i != 7 else 0.4,
                     'SBI-moments-UM': tm, 'SBI-moments-UM_p': 0.01})
    df = pd.DataFrame(rows)
    df['n_sig_BE'] = [(r[['GS-UM', 'SBI-pooled-UM', 'SBI-moments-UM']].values ==
                       'BE')[[r['GS-UM_p'] < .05, r['SBI-pooled-UM_p'] < .05, r['SBI-moments-UM_p'] < .05]].sum()
                      for _, r in df.iterrows()]
    df['n_sig_SC'] = [(r[['GS-UM', 'SBI-pooled-UM', 'SBI-moments-UM']].values ==
                       'SC')[[r['GS-UM_p'] < .05, r['SBI-pooled-UM_p'] < .05, r['SBI-moments-UM_p'] < .05]].sum()
                      for _, r in df.iterrows()]
    df['Consensus'] = df['true_model']
    df['contested'] = (df['id'] == 'BE0')
    return df


def test_confusion_and_summaries():
    assign = {1: _assign(1), 2: _assign(2)}
    conf = V.confusion_table(assign)
    gs_be = conf[(conf.rule == 1) & (conf.voter == 'GS-UM') & (conf.true_model == 'BE')]
    assert dict(zip(gs_be['called'], gs_be['n'])) == {'BE': 3, 'SC': 1}
    pooled_sc = conf[(conf.rule == 1) & (conf.voter == 'SBI-pooled-UM') & (conf.true_model == 'SC')]
    assert dict(zip(pooled_sc['called'], pooled_sc['n'])) == {'SC': 3, 'no call': 1}
    summ = V.call_summary(conf).set_index(['rule', 'voter', 'true_model'])
    assert summ.loc[(1, 'GS-UM', 'BE'), 'wrong'] == 1 and summ.loc[(1, 'Consensus', 'SC'), 'correct'] == 4
    cont = V.contested_table(assign).set_index(['rule', 'true_model'])
    assert cont.loc[(1, 'BE'), 'n_contested'] == 1 and cont.loc[(1, 'SC'), 'n_contested'] == 0


def test_call_summary_by_run_stacks_runs_and_lists_the_retired_voter():
    """The October networks' validation run beside the current one; moments, present only in the
    October table, appears there and nowhere else."""
    current = {r: _assign(r).drop(columns=['SBI-moments-UM', 'SBI-moments-UM_p']) for r in V.RULES}
    october = {r: _assign(r) for r in V.RULES}
    both = V.call_summary_by_run({'current': current, 'october': october})
    assert both.columns[0] == 'run' and set(both['run']) == {'current', 'october'}
    voters = {'GS-UM', 'SBI-pooled-UM', 'SBI-moments-UM', 'Consensus'}
    assert set(both.loc[both['run'] == 'october', 'voter']) == voters
    assert 'SBI-moments-UM' not in set(both.loc[both['run'] == 'current', 'voter'])
    mine = both[both['run'] == 'current'].drop(columns='run').reset_index(drop=True)
    alone = V.call_summary(V.confusion_table(current))
    pd.testing.assert_frame_equal(mine, alone)                      # stacking changes no number


def test_real_votes_table_lists_every_method_with_its_role():
    a = pd.DataFrame([
        {'id': 'R1', 'GS-UM': 'BE', 'GS-UM_p': 0.20, 'GS-UM_be': 0.10, 'GS-UM_sc': 0.11,
         'SBI-pooled-UM': 'BE', 'SBI-pooled-UM_p': 0.001, 'SBI-pooled-UM_be': 0.09, 'SBI-pooled-UM_sc': 0.12,
         'SBI-moments-UM_lower': 'SC', 'SBI-moments-UM_lower_pvalue': 0.01,
         'SBI-moments-UM_be': 0.13, 'SBI-moments-UM_sc': 0.10,
         'Consensus': 'BE', 'n_sig_BE': 1, 'n_sig_SC': 0},
        {'id': 'R2', 'GS-UM': None, 'GS-UM_p': np.nan, 'GS-UM_be': np.nan, 'GS-UM_sc': np.nan,
         'SBI-pooled-UM': 'SC', 'SBI-pooled-UM_p': 0.03, 'SBI-pooled-UM_be': 0.12, 'SBI-pooled-UM_sc': 0.10,
         'SBI-moments-UM_lower': None, 'SBI-moments-UM_lower_pvalue': np.nan,
         'SBI-moments-UM_be': np.nan, 'SBI-moments-UM_sc': np.nan,
         'Consensus': 'SC', 'n_sig_BE': 0, 'n_sig_SC': 1},
    ])
    v = V.real_votes_table({'october': a}).set_index(['id', 'method'])
    assert list(V.real_votes_table({'october': a}).columns) == V.VOTE_COLUMNS
    assert len(v) == 6 and set(v['run']) == {'october'}
    assert (v.loc[('R1', 'GS-UM'), 'lower'], v.loc[('R1', 'GS-UM'), 'call']) == ('BE', 'no call')   # p 0.20
    assert v.loc[('R1', 'SBI-pooled-UM'), 'call'] == 'BE' and bool(v.loc[('R1', 'SBI-pooled-UM'), 'voting'])
    row = v.loc[('R1', 'SBI-moments-UM')]
    assert not bool(row['voting']) and row['call'] == 'SC' and row['sc_error'] == 0.10           # listed
    assert pd.isna(v.loc[('R2', 'GS-UM'), 'lower']) and v.loc[('R2', 'GS-UM'), 'call'] == 'no call'


def test_gs_sigma_split_and_real_calls():
    assign = _assign(1)
    truths = pd.DataFrame({'id': assign['id'], 'sigma_percep': [0.1, 0.4, 0.2, 0.2, 0.45, 0.1, 0.1, 0.1]})
    split = V.gs_sigma_table(assign, truths, 0.30).set_index(['true_model', 'sigma_band'])
    assert split.loc[('BE', '<= 0.3'), 'wrong'] == 1 and split.loc[('BE', '> 0.3'), 'n'] == 1
    both = V.real_calls_table({'oct': assign, 'new': assign.assign(Consensus='Unclear')})
    assert list(both.columns[:3]) == ['id', 'oct', 'oct_contested'] and (both['new'] == 'Unclear').all()
