"""
Tests for behav_utils/analysis/trajectory.py.

This module had a silent-failure bug (round 3 fix): compute_trajectory
was returning all-NaN values because sess.stats() was deleted. These
tests verify the fix and prevent regression.
"""

import numpy as np
import pytest


class TestComputeTrajectory:
    """compute_trajectory must return real numbers, not NaN."""

    def test_returns_real_numbers_not_nan(self, rng):
        """Critical regression test: must return real values, not silent NaN."""
        from behav_utils.analysis.trajectory import compute_trajectory
        from behav_utils.data.synthetic import generate_synthetic_animal

        animal, _ = generate_synthetic_animal(
            animal_id='X', n_sessions=3, trials_per_session=200, seed=1,
        )

        result = compute_trajectory(animal.sessions, ['accuracy'])

        # The bug: this used to return all NaN. Now should be real.
        assert 'values' in result
        values = result['values']
        assert 'accuracy' in values
        # At least 2 of 3 sessions should give non-NaN accuracy
        non_nan = np.sum(~np.isnan(values['accuracy']))
        assert non_nan >= 2, f"too many NaN in accuracy trajectory: {values['accuracy']}"

    def test_returns_expected_structure(self, rng):
        """compute_trajectory returns the documented dict shape."""
        from behav_utils.analysis.trajectory import compute_trajectory
        from behav_utils.data.synthetic import generate_synthetic_animal

        animal, _ = generate_synthetic_animal(
            animal_id='X', n_sessions=2, trials_per_session=200, seed=1,
        )

        result = compute_trajectory(animal.sessions, ['accuracy'])
        assert 'stat_names' in result
        assert 'session_indices' in result
        assert 'values' in result
        assert 'per_session' in result

    def test_accuracy_in_unit_interval(self, rng):
        """Accuracy values are valid probabilities."""
        from behav_utils.analysis.trajectory import compute_trajectory
        from behav_utils.data.synthetic import generate_synthetic_animal

        animal, _ = generate_synthetic_animal(
            animal_id='X', n_sessions=3, trials_per_session=200, seed=1,
        )

        result = compute_trajectory(animal.sessions, ['accuracy'])
        for v in result['values']['accuracy']:
            if not np.isnan(v):
                assert 0 <= v <= 1, f"accuracy {v} out of [0,1]"
