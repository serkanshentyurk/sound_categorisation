"""The prior variants (models/priors.py): the ranges BE and SC are simulated over."""

import numpy as np
import pytest
from sound_categorisation.models.be_core import BEParams
from sound_categorisation.models.priors import (
    PRIOR_VARIANTS,
    WIDE_BOUNDS,
    bounds_arrays,
    current,
    flat_bounds,
    from_flat,
    log_uniform_params,
    prior_bounds,
    resolve,
    sample_prior,
    to_flat,
)
from sound_categorisation.models.sc_core import SCParams
from sound_categorisation.models.simulate import simulate_choices
from sound_categorisation.settings import SBI_PRIOR

PARAMS = {'be': BEParams, 'sc': SCParams}


@pytest.mark.parametrize('model', ['be', 'sc'])
def test_manuscript_is_the_first_networks_prior(model):
    """The ranges of the model classes (``get_bounds``), in the theta order."""
    bounds = prior_bounds(model, 'manuscript')
    assert list(bounds) == PARAMS[model].get_param_names()
    assert bounds == {k: tuple(v) for k, v in PARAMS[model].get_bounds().items()}
    assert log_uniform_params(model, 'manuscript') == ()


@pytest.mark.parametrize('model', ['be', 'sc'])
def test_no_variant_means_the_one_in_use(model):
    """settings.SBI_PRIOR is what every model fit uses unless a variant is named."""
    assert current() == SBI_PRIOR == 'learning' and resolve(None) == SBI_PRIOR
    assert prior_bounds(model) == prior_bounds(model, SBI_PRIOR)
    assert log_uniform_params(model) == log_uniform_params(model, SBI_PRIOR)
    low, high = bounds_arrays(model)
    assert low.tolist() == [b[0] for b in prior_bounds(model).values()]
    assert high.tolist() == [b[1] for b in prior_bounds(model).values()]


@pytest.mark.parametrize('model', ['be', 'sc'])
@pytest.mark.parametrize('variant', PRIOR_VARIANTS)
def test_flat_space_is_where_the_prior_is_a_uniform_box(model, variant):
    """Log-uniform parameters are logged, the rest untouched; there and back is the identity; and a
    draw is uniform between the flat bounds."""
    names, thetas = sample_prior(model, variant, 4000, np.random.default_rng(1))
    flat = to_flat(model, variant, thetas)
    np.testing.assert_allclose(from_flat(model, variant, flat), thetas, rtol=1e-12)
    low, high = flat_bounds(model, variant)
    logs = log_uniform_params(model, variant)
    for j, name in enumerate(names):
        expected = np.log(thetas[:, j]) if name in logs else thetas[:, j]
        np.testing.assert_allclose(flat[:, j], expected)
        assert low[j] <= flat[:, j].min() and flat[:, j].max() <= high[j]
        u = (flat[:, j] - low[j]) / (high[j] - low[j])               # uniform on 0..1
        assert abs(u.mean() - 0.5) < 0.03 and abs(np.mean(u < 0.25) - 0.25) < 0.03
    assert to_flat(model, variant, thetas[0]).shape == (len(names),)    # a single vector works too


@pytest.mark.parametrize('model', ['be', 'sc'])
def test_learning_widens_the_learning_ranges_only(model):
    assert PRIOR_VARIANTS == ('manuscript', 'learning')
    manuscript, learning = (prior_bounds(model, v) for v in PRIOR_VARIANTS)
    assert list(learning) == list(manuscript)                             # the same four parameters
    changed = {k for k in learning if learning[k] != manuscript[k]}
    assert changed == set(WIDE_BOUNDS[model])
    for name in changed:                                                   # the wider range contains the trained one
        assert learning[name][0] <= manuscript[name][0] and learning[name][1] >= manuscript[name][1]
    assert set(log_uniform_params(model, 'learning')) <= changed


@pytest.mark.parametrize('model', ['be', 'sc'])
@pytest.mark.parametrize('variant', PRIOR_VARIANTS)
def test_draws_lie_in_the_bounds_and_run_the_model(model, variant):
    bounds = prior_bounds(model, variant)
    names, thetas = sample_prior(model, variant, 500, np.random.default_rng(0))
    assert names == list(bounds) and thetas.shape == (500, len(bounds))
    for j, name in enumerate(names):
        low, high = bounds[name]
        assert thetas[:, j].min() >= low and thetas[:, j].max() <= high
        middle = np.sqrt(low * high) if name in log_uniform_params(model, variant) else (low + high) / 2
        assert abs(np.median(thetas[:, j]) - middle) < 0.15 * (high - low)
    stimuli = np.random.default_rng(1).uniform(-1, 1, 60)
    choices = simulate_choices(model, dict(zip(names, thetas[0])), stimuli, (stimuli > 0).astype(int),
                               burn_in=50, seed=2)
    assert set(np.unique(choices)) <= {0.0, 1.0}


def test_a_log_uniform_range_is_covered_at_its_low_end():
    """Half of BE's wider learning rates lie below the geometric middle of the range, far below the
    arithmetic one: a uniform draw would put almost none there."""
    names, thetas = sample_prior('be', 'learning', 2000, np.random.default_rng(3))
    low, high = prior_bounds('be', 'learning')['eta_learning']
    eta = thetas[:, names.index('eta_learning')]
    assert 0.4 < np.mean(eta < np.sqrt(low * high)) < 0.6 and np.mean(eta < 0.05) > 0.3


def test_unknown_names_are_errors():
    with pytest.raises(ValueError, match='prior variant'):
        prior_bounds('be', 'broad')
    with pytest.raises(ValueError, match='model'):
        prior_bounds('xx', 'learning')
