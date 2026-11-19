"""What complete blocking would do (inference/blocked_update.py): no learning gives no effect, strong learning gives
a clear one, the draws are the saved posterior's, the observation is the opto report's post-laser delta-of-deltas,
the verdicts, and sc-blocked-update writing its tables."""

import numpy as np
import pandas as pd
import pytest
from behav_utils.data import session_from_arrays
from sound_categorisation.data.synthetic import synthetic_experiment
from sound_categorisation.inference.blocked_update import (
    PREDICTED_COLUMNS,
    STATS,
    SUMMARY_COLUMNS,
    animal_predictions,
    blocking_table,
    blocking_verdict,
    observed_dod,
    posterior_draws,
    posterior_path,
    summarise,
)
from sound_categorisation.inference.cv_utils import save_cv_result
from sound_categorisation.models.priors import sample_prior

BE_LEARNS = dict(sigma_percep=0.15, A_repulsion=0.0, eta_learning=0.5, eta_relax=0.05)
SC_NEVER_LEARNS = dict(sigma_percep=0.15, A_repulsion=0.0, gamma=1.0, sigma_update=0.2)


def _sessions(n_sessions=3, n=400, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_sessions):
        s = rng.uniform(-1, 1, n)
        out.append(session_from_arrays(s, (s > 0).astype(float), session_idx=i, laser=rng.random(n) < 0.3))
    return out


def _save_posterior(run, animal, model, samples):
    names = list(BE_LEARNS if model == 'BE' else SC_NEVER_LEARNS)
    save_cv_result(posterior_path(run, animal, model), animal, model, [], 'update_matrix',
                   posterior=None if samples is None else {'posterior_samples': np.asarray(samples, dtype=np.float32),
                                                           'param_names': names})


def test_a_model_that_does_not_learn_gives_no_effect():
    """SC with retention 1 never updates, so blocking changes nothing: with the same random numbers the two runs are
    the same choice for choice, and the effect is exactly zero."""
    d = animal_predictions(_sessions(), 'SC', [SC_NEVER_LEARNS] * 2, animal='T', burn_in=200)
    assert list(d.columns) == PREDICTED_COLUMNS and len(d) == 2 * len(STATS)
    assert (d['effect'] == 0).all() and d['within_normal'].notna().all()


def test_blocking_a_strong_learner_takes_the_laser_trial_out_of_the_next_choice():
    """BE learning fast from every outcome: with learning blocked on laser trials, the trial after one no longer
    repeats a rewarded choice more than an unrewarded one, so win-stay after a laser trial falls well below the
    other laser-off trials; without blocking the two do not differ."""
    d = animal_predictions(_sessions(), 'BE', [BE_LEARNS] * 3, animal='T', burn_in=300)
    win_stay = d[d['stat'] == 'win_stay']
    assert (win_stay['effect'] < -0.2).all()
    assert abs(win_stay['within_normal'].mean()) < 0.1


def test_draws_are_the_first_samples_of_the_saved_posterior(tmp_path):
    samples = np.array([[0.2, 0.1, 0.4, 0.05], [0.3, 0.2, 0.5, 0.06], [0.4, 0.3, 0.6, 0.07]])
    _save_posterior(tmp_path, 'A1', 'BE', samples)
    draws = posterior_draws(tmp_path, 'A1', 'BE', n=2)
    assert draws == [pytest.approx(dict(zip(BE_LEARNS, row))) for row in samples[:2]]
    with pytest.raises(FileNotFoundError, match='no SC fit of A1'):
        posterior_draws(tmp_path, 'A1', 'SC')
    _save_posterior(tmp_path, 'A2', 'BE', None)
    with pytest.raises(ValueError, match='no posterior samples'):
        posterior_draws(tmp_path, 'A2', 'BE')


def _contrasts_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_the_observation_is_the_post_laser_delta_of_deltas(tmp_path):
    f = tmp_path / 'Uniform' / 'ppc_post_opto' / 'A1' / 'contrasts.csv'
    rows = []
    for st, dod in zip(STATS, (-0.02, 0.01, 0.03)):
        rows += [{'kind': 'dod', 'unit': 'trials', 'stat': st, 'diff': dod, 'ci_lo': dod - 0.05, 'ci_hi': dod + 0.05,
                  'n_a': np.nan},
                 {'kind': 'dod', 'unit': 'sessions', 'stat': st, 'diff': 9.0, 'ci_lo': 8.0, 'ci_hi': 10.0, 'n_a': np.nan},
                 {'kind': 'within', 'unit': 'trials', 'stat': st, 'diff': 0.5, 'ci_lo': 0.4, 'ci_hi': 0.6, 'n_a': 310}]
    rows.append({'kind': 'dod', 'unit': 'trials', 'stat': 'mu', 'diff': 1.0, 'ci_lo': 0.9, 'ci_hi': 1.1, 'n_a': np.nan})
    _contrasts_csv(f, rows)
    o = observed_dod(tmp_path, 'A1').set_index('stat')
    assert list(o.index) == list(STATS)
    assert o.loc['win_stay', 'observed'] == pytest.approx(0.01) and o.loc['win_stay', 'ci_hi'] == pytest.approx(0.06)
    assert (o['n_post_opto'] == 310).all()
    assert observed_dod(tmp_path, 'NOBODY').empty


@pytest.mark.parametrize('effect, lo, hi, verdict', [
    (-0.04, -0.10, 0.10, 'too weak to test'),
    (-0.20, -0.08, 0.06, 'not fully blocked'),
    (-0.20, -0.15, -0.02, 'not fully blocked'),          # partly: zero and the predicted effect both excluded
    (-0.20, -0.30, -0.10, 'consistent with blocking'),
    (-0.20, -0.45, -0.25, 'beyond blocking'),
    (-0.20, -0.30, 0.05, 'unclear'),
    (0.20, -0.05, 0.10, 'not fully blocked'),
    (0.20, 0.25, 0.40, 'beyond blocking'),
    (0.20, np.nan, 0.10, 'missing'),
    (np.nan, -0.1, 0.1, 'missing'),
])
def test_verdicts(effect, lo, hi, verdict):
    assert blocking_verdict(effect, lo, hi, 0.05) == verdict


def test_the_summary_takes_the_median_draw():
    predicted = pd.DataFrame([{'animal': 'A1', 'genotype': 'het', 'model': 'BE', 'draw': d, 'stat': 'win_stay',
                               'within_blocked': np.nan, 'within_normal': np.nan, 'effect': e}
                              for d, e in enumerate((-0.4, -0.3, -0.2, -0.1, -0.25))])
    observed = pd.DataFrame([{'animal': 'A1', 'stat': 'win_stay', 'observed': -0.01, 'ci_lo': -0.06, 'ci_hi': 0.05,
                              'n_post_opto': 300}])
    s = summarise(predicted, observed).iloc[0]
    assert list(summarise(predicted, observed).columns) == SUMMARY_COLUMNS
    assert s['effect'] == pytest.approx(-0.25) and s['n_draws'] == 5 and s['margin'] == pytest.approx(0.05)
    assert s['effect_lo'] == pytest.approx(np.quantile([-0.4, -0.3, -0.2, -0.1, -0.25], 0.1))
    assert s['verdict'] == 'not fully blocked'
    assert summarise(predicted, pd.DataFrame()).iloc[0]['verdict'] == 'missing'


def test_jobs_give_the_same_numbers():
    sessions = _sessions(n_sessions=2, n=200)
    jobs = [('T1', 'het', 'BE', sessions, [BE_LEARNS]), ('T2', 'wt', 'SC', sessions, [SC_NEVER_LEARNS])]
    one, two = blocking_table(jobs, n_jobs=1, burn_in=100), blocking_table(jobs, n_jobs=2, burn_in=100)
    pd.testing.assert_frame_equal(one, two)
    assert set(one['animal']) == {'T1', 'T2'}


def test_the_command_writes_its_tables(tmp_path, monkeypatch):
    """sc-blocked-update on the synthetic experiment: fits from a model-identification run, observations from an
    opto run, an animal without fits left out."""
    monkeypatch.setenv('SC_RESULTS_ROOT', str(tmp_path / 'results'))
    from sound_categorisation.cli import blocked_update as cli
    from sound_categorisation.data.paths import start_run

    exp = synthetic_experiment(n_animals=3, n_trials=150)
    monkeypatch.setattr(cli, 'load_experiment_any', lambda config, snapshot: exp)
    monkeypatch.setattr(cli, 'load_cohorts', lambda path: {'synthetic': ['ST00', 'ST01', 'ST02']})
    fits = start_run('model_identification', 'all-mice', 'fits')
    rng = np.random.default_rng(0)
    for aid in ('ST00', 'ST01'):
        for model in ('BE', 'SC'):
            names, thetas = sample_prior(model, None, 3, rng)
            save_cv_result(posterior_path(fits, aid, model), aid, model, [], 'update_matrix',
                           posterior={'posterior_samples': thetas, 'param_names': names})
    opto = start_run('opto_contrasts', 'synthetic', 'opto')
    _contrasts_csv(opto / 'Uniform' / 'ppc_post_opto' / 'ST00' / 'contrasts.csv',
                   [{'kind': 'dod', 'unit': 'trials', 'stat': st, 'diff': 0.0, 'ci_lo': -0.1, 'ci_hi': 0.1, 'n_a': 1}
                    for st in STATS])
    run = cli.main(['--fits', 'all-mice/fits', '--cohort', 'synthetic', '--opto-run', 'opto', '--draws', '1'])
    assert run.parent.name == 'synthetic' and run.parent.parent.name == 'blocked_update'
    predicted, summary = pd.read_csv(run / 'predicted.csv'), pd.read_csv(run / 'summary.csv')
    assert set(predicted['animal']) == {'ST00', 'ST01'} and set(predicted['draw']) == {0}
    assert len(summary) == 2 * 2 * len(STATS) and (summary['n_draws'] == 1).all()
    assert (summary.loc[summary['animal'] == 'ST01', 'verdict'] == 'missing').all()      # no observation
    assert summary.loc[summary['animal'] == 'ST00', 'observed'].notna().all()
    assert set(pd.read_csv(run / 'sessions.csv')['animal']) == {'ST00', 'ST01'}
    assert (run / 'meta.json').exists()
