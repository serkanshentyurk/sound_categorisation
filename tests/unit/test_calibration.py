"""inference.calibration — SBC ranks and coverage, with a stub net (no torch).

A stub whose "posterior" is the prior itself is calibrated by construction, so its ranks must
be uniform and its coverage must match the nominal levels; a stub whose posterior is a narrow
band around a wrong value must fail both. The coverage arithmetic is checked exactly on
hand-made ranks.
"""

import numpy as np
import pandas as pd
from sound_categorisation.inference.calibration import compute_sbc_ranks, coverage_from_ranks
from sound_categorisation.inference.types import ModelType
from sound_categorisation.models.priors import flat_bounds, from_flat, sample_prior

PARAMS = ['sigma_percep', 'A_repulsion', 'eta_learning', 'eta_relax']


class _StubNet:
    """Mimics the AmortisedSBI attributes compute_sbc_ranks reads, with a chosen posterior."""

    def __init__(self, kind='prior', prior_name='learning', min_accuracy=None):
        self.model = ModelType.BE
        self.T, self.burn_in, self.dist_schedule, self.mode = 40, 10, 'uniform', 'pooled'
        self.param_names = list(PARAMS)
        self.kind, self.prior_name, self.min_accuracy = kind, prior_name, min_accuracy
        self._k = 0

    def condition(self, sessions, n_samples=1000):
        lo, hi = flat_bounds('be', self.prior_name)
        self._k += 1
        rng = np.random.default_rng(10_000 + self._k)
        if self.kind == 'prior':                 # the prior itself: calibrated by construction
            s = sample_prior('be', self.prior_name, n_samples, rng)[1]
        else:                                    # overconfident and biased: a sliver at the lower end
            s = from_flat('be', self.prior_name, lo + 0.05 * (hi - lo) * rng.uniform(size=(n_samples, 4)))
        return {'posterior_samples': s, 'param_names': self.param_names,
                'point_estimate': dict(zip(self.param_names, np.median(s, axis=0)))}


def test_coverage_from_ranks_exact():
    L = 100
    # ranks spread evenly over 0..100 -> coverage of the central c interval is ~c
    ranks = np.tile(np.arange(0, L + 1), (2, 1)).T               # (101, 2)
    cov = coverage_from_ranks(ranks, L, levels=(0.5, 0.9), param_names=['a', 'b'])
    assert isinstance(cov, pd.DataFrame) and set(cov.columns) == {'param', 'nominal', 'empirical', 'n'}
    c50 = cov[(cov.param == 'a') & (cov.nominal == 0.5)].empirical.iloc[0]
    c90 = cov[(cov.param == 'b') & (cov.nominal == 0.9)].empirical.iloc[0]
    assert abs(c50 - 51 / 101) < 1e-12 and abs(c90 - 91 / 101) < 1e-12
    assert (cov.n == 101).all()


def test_calibrated_stub_is_uniform_and_covers():
    res = compute_sbc_ranks(_StubNet('prior'), n_draws=200, n_sessions=2, T=40,
                            n_posterior_samples=200, seed=3, show_progress=False)
    assert res['ranks'].shape == (200, 4) and res['theta_true'].shape == (200, 4)
    assert res['n_skipped'] == 0 and res['param_names'] == PARAMS
    assert min(res['ks_pvalues']) > 0.01
    cov = res['coverage'].pivot(index='param', columns='nominal', values='empirical')
    for c in (0.5, 0.8, 0.9, 0.95):
        assert (abs(cov[c] - c) < 3 * np.sqrt(c * (1 - c) / 200) + 0.01).all()


def test_overconfident_stub_fails():
    res = compute_sbc_ranks(_StubNet('narrow'), n_draws=100, n_sessions=2, T=40,
                            n_posterior_samples=100, seed=5, show_progress=False)
    assert max(res['ks_pvalues']) < 1e-6                 # ranks pile at the top
    cov = res['coverage'].pivot(index='param', columns='nominal', values='empirical')
    assert (cov[0.95] < 0.3).all()


def test_per_draw_posterior_summaries():
    res = compute_sbc_ranks(_StubNet('prior'), n_draws=20, n_sessions=2, T=40,
                            n_posterior_samples=100, seed=2, show_progress=False)
    n, p = res['theta_true'].shape
    assert res['theta_mean'].shape == (n, p) and res['theta_sd'].shape == (n, p)
    assert res['theta_quantiles'].shape == (n, len(res['quantile_levels']), p)
    q = res['quantile_levels']
    assert np.all(res['theta_quantiles'][:, q.index(0.05)] <= res['theta_quantiles'][:, q.index(0.95)])
    assert res['acceptance_rate'].shape == (n,) and res['budget_hit'].dtype == bool
    assert res['design_features'] == []                    # the stub has none


def test_recovery_is_stored_where_the_prior_is_flat():
    """Under the learning prior BE's two rates are log-uniform: their truths, posterior summaries
    and bounds are stored as logarithms, the other parameters as they are."""
    res = compute_sbc_ranks(_StubNet('prior'), n_draws=30, n_sessions=2, T=40,
                            n_posterior_samples=100, seed=6, show_progress=False)
    assert res['prior'] == 'learning' and res['log_uniform'] == ['eta_learning', 'eta_relax']
    low, high = flat_bounds('be', 'learning')
    for j, name in enumerate(PARAMS):
        assert res['prior_bounds'][name] == (low[j], high[j])
        assert np.all(res['theta_true'][:, j] >= low[j]) and np.all(res['theta_true'][:, j] <= high[j])
        assert np.all(res['theta_median'][:, j] >= low[j]) and np.all(res['theta_median'][:, j] <= high[j])
    assert res['theta_true'][:, 2].max() < 0 and res['theta_true'][:, 0].min() > 0     # log rate, plain sigma
    old = compute_sbc_ranks(_StubNet('prior', prior_name='manuscript'), n_draws=5, n_sessions=2, T=40,
                            n_posterior_samples=50, seed=6, show_progress=False)
    assert old['prior'] == 'manuscript' and old['log_uniform'] == [] and old['theta_true'][:, 2].min() >= 0.05


def test_accuracy_rule_redraws_until_every_session_passes():
    """A network trained with the rule is calibrated on truths that pass it, like its training set."""
    kwargs = dict(n_draws=25, n_sessions=2, T=60, n_posterior_samples=50, seed=8, show_progress=False)
    ruled = compute_sbc_ranks(_StubNet('prior', min_accuracy=0.62), **kwargs)
    plain = compute_sbc_ranks(_StubNet('prior'), **kwargs)
    assert ruled['min_accuracy'] == 0.62 and ruled['n_rejected'] > 0 and len(ruled['ranks']) == 25
    assert plain['min_accuracy'] is None and plain['n_rejected'] == 0
    assert not np.array_equal(ruled['theta_true'], plain['theta_true'])     # other truths survive


def test_default_seed_is_the_validation_seed():
    from sound_categorisation.settings import VALIDATION_SEED
    res = compute_sbc_ranks(_StubNet('prior'), n_draws=3, n_sessions=2, T=30,
                            n_posterior_samples=20, show_progress=False)
    assert res['seed'] == VALIDATION_SEED


def test_tables_and_single_panel_plotters(tmp_path):
    """SBC results laid out as sc-sbi-sbc writes them -> validation tables -> plotters."""
    import pickle

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sound_categorisation.inference import validation as V
    from sound_categorisation.plotting.sbi_validation import (
        plot_contraction,
        plot_coverage,
        plot_recovery_scatter,
        plot_sbc_ecdf_diff,
    )
    for n_sess in (2, 5):
        res = compute_sbc_ranks(_StubNet('prior'), n_draws=40, n_sessions=n_sess, T=40,
                                n_posterior_samples=60, seed=4, show_progress=False)
        d = tmp_path / 'sbc' / 'uniform' / f'n{n_sess}' / 'pooled'
        d.mkdir(parents=True)
        with open(d / 'BE.pkl', 'wb') as f:
            pickle.dump(res, f)
    results = V.load_sbc(tmp_path, 'uniform')
    assert sorted(r['n_sessions'] for r in results) == [2, 5]
    summ = V.sbc_summary_table(results)
    assert set(summ['param']) == set(PARAMS) and (summ['n_used'] == 40).all()
    cov = V.coverage_table(results)
    assert ((cov['band_lo'] <= cov['nominal']) & (cov['nominal'] <= cov['band_hi'])).all()
    ecdf = V.ecdf_table(results, n_points=20, n_sim=200)
    assert len(ecdf) == 2 * len(PARAMS) * 20
    assert (ecdf['band_lo'] < 0).all() and (ecdf['band_hi'] > 0).all()
    draws = V.recovery_draws(results)
    rec = V.recovery_table(draws)
    assert len(draws) == 2 * 40 * len(PARAMS) and len(rec) == 2 * len(PARAMS)
    # a posterior equal to the prior recovers nothing: R^2 <= ~0, no contraction
    assert (rec['r2'] < 0.3).all() and (rec['contraction_median'].abs() < 0.3).all()

    sel = dict(rep='pooled', model='BE', n_sessions=5)
    one = ecdf[(ecdf.rep == 'pooled') & (ecdf.n_sessions == 5) & (ecdf.param == 'eta_learning')]
    assert plot_sbc_ecdf_diff(one) is not None
    assert plot_coverage(cov[(cov.rep == sel['rep']) & (cov.n_sessions == 5)]) is not None
    d5 = draws[(draws.n_sessions == 5) & (draws.param == 'sigma_percep')]
    fig, ax = plt.subplots()
    assert plot_recovery_scatter(d5, ax=ax) is ax and plot_contraction(d5) is not None
    plt.close('all')
