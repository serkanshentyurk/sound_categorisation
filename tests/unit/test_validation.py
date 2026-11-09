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


def test_gs_sigma_split_and_real_calls():
    assign = _assign(1)
    truths = pd.DataFrame({'id': assign['id'], 'sigma_percep': [0.1, 0.4, 0.2, 0.2, 0.45, 0.1, 0.1, 0.1]})
    split = V.gs_sigma_table(assign, truths, 0.30).set_index(['true_model', 'sigma_band'])
    assert split.loc[('BE', '<= 0.3'), 'wrong'] == 1 and split.loc[('BE', '> 0.3'), 'n'] == 1
    both = V.real_calls_table({'oct': assign, 'new': assign.assign(Consensus='Unclear')})
    assert list(both.columns[:3]) == ['id', 'oct', 'oct_contested'] and (both['new'] == 'Unclear').all()
