"""Tests for models.simulate.simulate_choices (the single params->choices core)."""

import numpy as np
import pytest

from inference.types import ModelType, get_default_param_configs
from models.simulate import simulate_choices


def _mid(model):
    return {n: 0.5 * (c.bounds[0] + c.bounds[1])
            for n, c in get_default_param_configs(model).items()}


def _stim(n=300, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1, 1, n)
    return s, (s > 0).astype(int)


class TestSimulateChoices:

    @pytest.mark.parametrize('model', [ModelType.BE, ModelType.SC])
    def test_shape_and_binary(self, model):
        s, c = _stim()
        ch = simulate_choices(model, _mid(model), s, c, burn_in=200, seed=1)
        assert ch.shape == (300,)
        finite = ch[np.isfinite(ch)]
        assert set(np.unique(finite)) <= {0.0, 1.0}

    @pytest.mark.parametrize('model_id', ['be', 'sc'])
    def test_string_id_accepted(self, model_id):
        model = ModelType.BE if model_id == 'be' else ModelType.SC
        s, c = _stim()
        ch = simulate_choices(model_id, _mid(model), s, c, burn_in=200, seed=1)
        assert ch.shape == (300,)

    def test_reproducible(self):
        s, c = _stim(seed=3)
        a = simulate_choices(ModelType.BE, _mid(ModelType.BE), s, c,
                             burn_in=200, seed=7)
        b = simulate_choices(ModelType.BE, _mid(ModelType.BE), s, c,
                             burn_in=200, seed=7)
        assert np.array_equal(a, b)

    def test_enum_equals_string(self):
        s, c = _stim(seed=5)
        a = simulate_choices(ModelType.BE, _mid(ModelType.BE), s, c,
                             burn_in=200, seed=9)
        b = simulate_choices('be', _mid(ModelType.BE), s, c,
                             burn_in=200, seed=9)
        assert np.array_equal(a, b)

    def test_unknown_model_raises(self):
        s, c = _stim()
        with pytest.raises(ValueError):
            simulate_choices('xyz', {}, s, c)
