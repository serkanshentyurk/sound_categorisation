"""What a network trains on, checked without torch: the accuracy rule in the simulator and the
training set ``AmortisedSBI.simulate_training_set`` builds (prior draws kept until enough pass)."""

import numpy as np
import pytest
from sound_categorisation.inference.amortised import AmortisedSBI
from sound_categorisation.inference.simulator import (
    accuracy_rule,
    build_simulator,
    get_bounds_arrays,
    session_accuracy,
    simulate_sessions,
    theta_to_params,
)
from sound_categorisation.models.priors import bounds_arrays, prior_bounds, sample_prior

STATS = ['accuracy', 'mu', 'win_stay']
SMALL = dict(dist_schedule='uniform', N=(2, 3), T=(60, 90), burn_in=40, mode='pooled', stat_names=STATS)
RULE = 0.62                              # a rule about half of these short BE simulations pass


def test_the_fits_use_the_prior_in_use_unless_one_is_named():
    for model in ('be', 'sc'):
        for got, want in zip(get_bounds_arrays(model), bounds_arrays(model, 'learning')):
            np.testing.assert_array_equal(got, want)
        for got, want in zip(get_bounds_arrays(model, 'manuscript'), bounds_arrays(model, 'manuscript')):
            np.testing.assert_array_equal(got, want)


def test_theta_is_clipped_to_the_prior_it_belongs_to():
    """A slow BE learning rate is inside the learning prior and outside the manuscript one."""
    theta = np.array([0.1, 0.2, 0.01, 0.002])
    assert theta_to_params(theta, 'be') == dict(zip(prior_bounds('be'), theta))
    clipped = theta_to_params(theta, 'be', 'manuscript')
    assert clipped['eta_learning'] == 0.05 and clipped['eta_relax'] == 0.01


def test_accuracy_rule_is_the_session_filters_minimum_where_a_representation_has_one():
    assert accuracy_rule('pooled', 'uniform') == pytest.approx(0.70)
    assert accuracy_rule('moments', 'uniform') == pytest.approx(0.70)
    assert accuracy_rule('single', 'uniform') is None


@pytest.mark.parametrize('model', ['be', 'sc'])
def test_the_rule_gives_up_at_the_first_failing_session_and_leaves_passing_animals_alone(model):
    _, thetas = sample_prior(model, 'learning', 12, np.random.default_rng(0))
    n_passed = n_failed = 0
    for i, theta in enumerate(thetas):
        params = theta_to_params(theta, model)
        plain = simulate_sessions(model, params, ['uniform'] * 3, (60, 90), 40, np.random.default_rng(i))
        ruled = simulate_sessions(model, params, ['uniform'] * 3, (60, 90), 40, np.random.default_rng(i),
                                  min_accuracy=RULE)
        accuracies = [session_accuracy(s) for s in plain]
        if min(accuracies) >= RULE:
            n_passed += 1
            assert ruled is not None and len(ruled) == 3
            for a, b in zip(plain, ruled):                       # the same animal, session for session
                np.testing.assert_array_equal(a.get_arrays()['choices'], b.get_arrays()['choices'])
        else:
            n_failed += 1
            assert ruled is None
    assert n_passed and n_failed                                 # the test saw both cases


def test_sim_fn_returns_none_for_a_rejected_draw_and_the_same_x_otherwise():
    plain, _, _ = build_simulator('be', **SMALL)
    ruled, _, _ = build_simulator('be', min_accuracy=RULE, **SMALL)
    _, thetas = sample_prior('be', None, 10, np.random.default_rng(2))
    outputs = [(plain(t, seed=i), ruled(t, seed=i)) for i, t in enumerate(thetas)]
    assert any(r is None for _, r in outputs) and any(r is not None for _, r in outputs)
    for x, r in outputs:
        if r is not None:
            np.testing.assert_array_equal(x, r)
        else:
            assert x is not None                                 # without the rule nothing is dropped


def _net(model='be', min_accuracy=RULE, prior=None):
    return AmortisedSBI(model, design_features=('n_sessions', 'n_valid_trials'), prior=prior,
                        min_accuracy=min_accuracy, **SMALL)


def test_training_set_is_the_first_passing_draws_whatever_the_parallelism():
    net = _net()
    theta, x, n_drawn, n_passed = net.simulate_training_set(12, seed=5, n_jobs=1, show_progress=False)
    theta2, x2, n_drawn2, n_passed2 = net.simulate_training_set(12, seed=5, n_jobs=2, show_progress=False)
    np.testing.assert_array_equal(theta, theta2)
    np.testing.assert_array_equal(x, x2)
    assert theta.shape == (12, 4) and x.shape == (12, len(STATS) + 2) and n_drawn == n_drawn2 > 12
    assert n_passed == n_passed2 and 12 <= n_passed < n_drawn       # the last round may pass a few more

    # by hand: draw i has the i-th parameters of one stream and seed 5 + i; keep the first 12 passing
    _, stream = sample_prior('be', None, n_drawn, np.random.default_rng(5))
    kept = []
    for i, t in enumerate(stream):
        out = net.simulator(t, seed=5 + i)
        if out is not None:
            kept.append((t, out))
        if len(kept) == 12:
            break
    np.testing.assert_array_equal(theta, np.vstack([t for t, _ in kept]))
    np.testing.assert_array_equal(x, np.vstack([o for _, o in kept]))


def test_training_set_is_in_the_parameters_own_units_inside_the_networks_prior():
    for prior in ('learning', 'manuscript'):
        net = _net(prior=prior)
        theta = net.simulate_training_set(8, seed=1, n_jobs=1, show_progress=False)[0]
        low, high = net.bounds()
        np.testing.assert_array_equal(low, bounds_arrays('be', prior)[0])
        assert np.all(theta >= low) and np.all(theta <= high)
        assert net.prior_name == prior


def test_without_the_rule_every_draw_is_kept_and_none_is_wasted():
    net = _net(min_accuracy=None)
    for n in (70, 7):                                            # however few, none beyond them
        theta, x, n_drawn, n_passed = net.simulate_training_set(n, seed=3, n_jobs=1, show_progress=False)
        assert n_drawn == n_passed == n and len(theta) == n and len(x) == n
    theta = net.simulate_training_set(70, seed=3, n_jobs=1, show_progress=False)[0]
    _, stream = sample_prior('be', None, 70, np.random.default_rng(3))
    np.testing.assert_array_equal(theta, stream)


def test_a_rule_almost_nothing_passes_stops_at_max_draws():
    net = _net(min_accuracy=0.999)
    with pytest.raises(RuntimeError, match='prior draws'):
        net.simulate_training_set(5, seed=0, n_jobs=1, max_draws=100, show_progress=False)
