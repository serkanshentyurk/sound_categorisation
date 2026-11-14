"""BE and SC as behav_utils models (models/simulate.py) and the rig's distributions as a behav_utils
StimulusDistribution (data/stimuli.RigDistribution).

The models must draw exactly what simulate_choices draws, so that simulations, the synthetic cohort and the
trained SBI networks do not move: a run is simulate_choices with one seed drawn from the run's rng.
"""
import numpy as np
import pytest
from behav_utils.models import Design, generate_synthetic_animal, make_design, simulate, simulate_responses
from behav_utils.tasks import TwoAFC
from sound_categorisation.data.stimuli import RigDistribution, sample_distribution
from sound_categorisation.inference.simulator import get_bounds_arrays, simulate_sessions, theta_to_params
from sound_categorisation.inference.types import ModelType
from sound_categorisation.models import (
    BoundaryEstimation,
    StatisticalCategorisation,
    make_model,
    simulate_choices,
)
from sound_categorisation.models.simulate import _run

BURN_IN = 200


def _params(model):
    lo, hi = get_bounds_arrays(model)
    return theta_to_params((lo + hi) / 2, model)


@pytest.mark.parametrize('model', ['be', 'sc'])
def test_a_run_is_simulate_choices_with_the_seed_drawn_from_rng(model):
    stim, cat = sample_distribution(300, 'hard_a', rng=np.random.default_rng(1))
    seed = int(np.random.default_rng(3).integers(0, 2**31 - 1))
    expected = simulate_choices(model, _params(model), stim, cat, burn_in=BURN_IN, seed=seed)
    got = make_model(model, _params(model), burn_in=BURN_IN).simulate(Design(TwoAFC(), stim),
                                                                      np.random.default_rng(3)).choice
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize('model', ['be', 'sc'])
def test_simulate_sessions_draws_what_it_drew_before(model):
    """simulate_sessions now runs through behav_utils' simulate; the draws are those of its old loop."""
    params, schedule = _params(model), ['uniform', 'hard_b', 'hard_a']
    rng = np.random.default_rng(11)
    old = []
    for dist in schedule:
        stim, cat = sample_distribution(250, dist, rng=rng)
        seed = int(rng.integers(0, 2**31 - 1))
        old.append((stim, simulate_choices(model, params, stim, cat, burn_in=BURN_IN, seed=seed)))
    new = simulate_sessions(model, params, schedule, 250, BURN_IN, np.random.default_rng(11))
    for (stim, choices), session, dist in zip(old, new, schedule):
        np.testing.assert_array_equal(session.trials.stimulus, stim)
        np.testing.assert_array_equal(session.trials.choice, choices)
        assert session.distribution == dist


def test_behav_utils_runs_them_on_its_own_designs():
    rng = np.random.default_rng(5)
    be = BoundaryEstimation(_params('be'), burn_in=BURN_IN)
    sc = StatisticalCategorisation(_params('sc'), burn_in=BURN_IN)
    design = make_design(TwoAFC(), RigDistribution('Hard-B'), 200, label='Hard-B', laser_fraction=0.3,
                         abort_rate=0.1, no_response_rate=0.05, rng=rng)
    session = simulate(be, design, rng)
    skipped = design.abort | design.no_response
    assert session.trials.n_trials == 200 and session.distribution == 'Hard-B'
    assert np.isnan(session.trials.choice[skipped]).all()
    assert set(np.unique(session.trials.choice[~skipped])) <= {0.0, 1.0}
    assert set(np.unique(simulate_responses(sc, design.stimulus, rng=rng))) <= {0.0, 1.0}
    animal = generate_synthetic_animal(sc, n_sessions=3, trials_per_session=120,
                                       distribution=RigDistribution('Hard-A'), label='Hard-A', rng=7)
    assert [s.distribution for s in animal.sessions] == ['Hard-A'] * 3


@pytest.mark.parametrize('model', ['be', 'sc'])
def test_the_laser_blocks_learning_only_when_asked(model):
    stim, cat = sample_distribution(300, 'uniform', rng=np.random.default_rng(2))
    laser = np.ones(300, dtype=bool)
    seed = int(np.random.default_rng(4).integers(0, 2**31 - 1))
    blocked = make_model(model, _params(model), burn_in=BURN_IN, laser_blocks_update=True)
    ignored = make_model(model, _params(model), burn_in=BURN_IN)
    got_blocked = blocked.simulate(Design(TwoAFC(), stim, laser=laser), np.random.default_rng(4)).choice
    got_ignored = ignored.simulate(Design(TwoAFC(), stim, laser=laser), np.random.default_rng(4)).choice
    np.testing.assert_array_equal(got_blocked, _run(model, _params(model), stim, cat, BURN_IN, seed,
                                                    update_mask=np.zeros(300, dtype=bool)))
    np.testing.assert_array_equal(got_ignored, simulate_choices(model, _params(model), stim, cat,
                                                                burn_in=BURN_IN, seed=seed))


def test_make_model_takes_a_model_type_or_a_name():
    assert isinstance(make_model(ModelType.BE, _params('be')), BoundaryEstimation)
    assert isinstance(make_model('SC', _params('sc')), StatisticalCategorisation)
    with pytest.raises(ValueError, match='Unknown model'):
        make_model('xx', {})


def test_the_rig_distributions():
    for name in ('Uniform', 'Hard-A', 'Hard-B'):
        dist = RigDistribution(name)
        np.testing.assert_array_equal(dist.sample(50, np.random.default_rng(9)),
                                      sample_distribution(50, name, rng=np.random.default_rng(9))[0])
        points, weights = dist.grid()
        assert weights.sum() == pytest.approx(1.0)
        assert np.sum(dist.pdf(points)) * (points[1] - points[0]) == pytest.approx(1.0, rel=1e-3)
    hard_a = RigDistribution('Hard-A')
    assert hard_a.pdf(-0.05) > hard_a.pdf(0.05)                  # the hard side is A, crowded near 0
    with pytest.raises(ValueError, match='unknown distribution'):
        RigDistribution('Hard-C')
