"""Tests for behav_utils.data.selection.SessionFilter."""

import numpy as np
import pytest
from datetime import date, timedelta

from behav_utils.data.selection import SessionFilter, list_presets


class TestSessionFilterBasic:
    """Basic filtering tests."""

    def test_no_constraints(self, synthetic_animal):
        """Empty filter returns all sessions."""
        f = SessionFilter(exclude_masking=False)
        result = f.apply(synthetic_animal)
        assert len(result) == len(synthetic_animal.sessions)

    def test_stage_filter(self, synthetic_animal):
        """Filter by stage name."""
        f = SessionFilter(stage='Full_Task_Cont', exclude_masking=False)
        result = f.apply(synthetic_animal)
        assert len(result) == len(synthetic_animal.sessions)

        f2 = SessionFilter(stage='Nonexistent')
        result2 = f2.apply(synthetic_animal)          
        assert len(result2) == 0

    def test_distribution_filter(self, synthetic_animal):
        """Filter by distribution."""
        f = SessionFilter(distribution='Uniform', exclude_masking=False)
        result = f.apply(synthetic_animal)
        assert len(result) == len(synthetic_animal.sessions)

    def test_min_trials(self, synthetic_animal):
        """Filter by minimum trial count."""
        f = SessionFilter(min_trials=500)
        result = f.apply(synthetic_animal)
        # Fixture has 300 trials per session
        assert len(result) == 0

        f2 = SessionFilter(min_trials=100, exclude_masking=False)
        result2 = f2.apply(synthetic_animal)  
        assert len(result2) == len(synthetic_animal.sessions)


class TestMaskingFilter:
    """Tests for exclude_masking."""

    def test_exclude_masking_default(self, synthetic_animal):
        """exclude_masking=True by default."""
        f = SessionFilter()
        assert f.exclude_masking is True

    def test_masking_excluded(self, synthetic_animal):
        """Masking sessions filtered out by default."""
        f = SessionFilter()
        result = f.apply(synthetic_animal)
        for sess in result:
            assert not getattr(sess, 'masking', False)

    def test_masking_included(self, synthetic_animal):
        """exclude_masking=False keeps masking sessions."""
        f = SessionFilter(exclude_masking=False)
        result = f.apply(synthetic_animal)
        assert len(result) == len(synthetic_animal.sessions)

    def test_masking_count(self, synthetic_animal):
        """Correct number filtered out."""
        all_count = len(synthetic_animal.sessions)
        masking_count = sum(
            1 for s in synthetic_animal.sessions
            if getattr(s, 'masking', False)
        )
        f = SessionFilter()
        result = f.apply(synthetic_animal)
        assert len(result) == all_count - masking_count


class TestPresets:
    """Tests for preset registry."""

    def test_presets_exist(self):
        """At least some presets should be registered."""
        presets = list_presets()
        assert len(presets) > 0
        assert 'expert_uniform' in presets

    def test_preset_returns_filter(self):
        """Each preset should be a SessionFilter."""
        presets = list_presets()
        for name in presets:
            from behav_utils.data.selection import get_preset
            f = get_preset(name)
            assert isinstance(f, SessionFilter)


class TestWithOverrides:
    """Tests for SessionFilter.with_overrides()."""

    def test_override_creates_new(self):
        """with_overrides returns a new object."""
        base = SessionFilter(min_trials=100)
        modified = base.with_overrides(min_trials=200)
        assert base.min_trials == 100
        assert modified.min_trials == 200
        assert base is not modified
