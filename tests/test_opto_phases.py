"""Tests for analysis.opto phase assignment and within-session effects."""

import numpy as np
import pytest
from datetime import date

from analysis.opto import (
    OptoPhase, assign_opto_phases, split_trials_by_opto,
    within_session_effect, phase_pooled_comparison,
    expert_stability,
)


class TestAssignOptoPhases:
    """Tests for sequential phase assignment logic."""

    def test_baseline_only(self, synthetic_animal):
        """Animal with no opto → all sessions expert_baseline."""
        for sess in synthetic_animal.sessions:
            sess.trials.opto_on = np.zeros(len(sess.trials.stimulus), dtype=bool)
            sess.masking = False
        phases = assign_opto_phases(synthetic_animal)
        assert all(p == OptoPhase.EXPERT_BASELINE for p in phases)

    def test_masking_detected(self, synthetic_animal):
        """Sessions with masking=True → OptoPhase.MASKING."""
        phases = assign_opto_phases(synthetic_animal)
        assert phases[10] == OptoPhase.MASKING
        assert phases[11] == OptoPhase.MASKING

    def test_expert_opto_after_masking(self, synthetic_animal):
        """Opto sessions after masking → EXPERT_OPTO."""
        phases = assign_opto_phases(synthetic_animal)
        for i in range(12, 15):
            assert phases[i] == OptoPhase.EXPERT_OPTO

    def test_full_timeline(self, synthetic_opto_animal):
        """Full opto timeline with shifts."""
        phases = assign_opto_phases(synthetic_opto_animal)

        for i in range(5):
            assert phases[i] == OptoPhase.EXPERT_BASELINE, f'Session {i}'

        assert phases[5] == OptoPhase.MASKING
        assert phases[6] == OptoPhase.MASKING

        for i in range(7, 12):
            assert phases[i] == OptoPhase.EXPERT_OPTO, f'Session {i}'

        for i in range(12, 14):
            assert phases[i] == OptoPhase.EXPERT_WASHOUT, f'Session {i}'

        for i in range(14, 19):
            assert phases[i] == OptoPhase.SHIFT_1_OPTO, f'Session {i}'

        for i in range(19, 24):
            assert phases[i] == OptoPhase.SHIFT_1_RECOVERY, f'Session {i}'

    def test_length_matches_sessions(self, synthetic_opto_animal):
        """Output length matches number of sessions."""
        phases = assign_opto_phases(synthetic_opto_animal)
        assert len(phases) == len(synthetic_opto_animal.sessions)


class TestSplitTrials:
    """Tests for trial splitting."""

    def test_mask_sizes(self, synthetic_opto_trial_data):
        """Opto + control masks should cover all non-abort trials."""
        from behav_utils.data.structures import SessionData

        sess = SessionData(
            session_id='test_000',
            session_idx=0,
            date=date(2026, 1, 1),
            metadata={'distribution': 'Uniform'},
            trials=synthetic_opto_trial_data,
        )
        opto_mask, ctrl_mask = split_trials_by_opto(sess)
        assert opto_mask.sum() + ctrl_mask.sum() == 300
        assert not np.any(opto_mask & ctrl_mask)

    def test_no_opto_session(self, synthetic_session):
        """Session with no opto → all trials are control."""
        opto_mask, ctrl_mask = split_trials_by_opto(synthetic_session)
        assert opto_mask.sum() == 0
        assert ctrl_mask.sum() == len(synthetic_session.trials.stimulus)


class TestWithinSessionEffect:
    """Tests for within-session opto vs control comparison."""

    def test_returns_none_for_no_opto(self, synthetic_session):
        """Session without opto trials → None."""
        result = within_session_effect(synthetic_session)
        assert result is None

    def test_returns_dict_with_opto(self, synthetic_opto_animal):
        """Opto session → dict with expected keys."""
        sess = synthetic_opto_animal.sessions[7]  # First real opto
        result = within_session_effect(sess)
        assert result is not None
        assert 'opto_stats' in result
        assert 'control_stats' in result
        assert 'diff' in result
        assert 'accuracy' in result['diff']

    def test_diff_is_opto_minus_control(self, synthetic_opto_animal):
        """Diff should be opto - control."""
        sess = synthetic_opto_animal.sessions[7]
        result = within_session_effect(sess)
        expected = (result['opto_stats']['accuracy']
                    - result['control_stats']['accuracy'])
        assert abs(result['diff']['accuracy'] - expected) < 1e-10


class TestExpertStability:
    """Tests for baseline → opto → washout trajectory."""

    def test_returns_phase_values(self, synthetic_opto_animal):
        """Should return values for each phase."""
        phases = assign_opto_phases(synthetic_opto_animal)
        result = expert_stability(
            synthetic_opto_animal.sessions, phases, stat_name='accuracy')
        assert len(result['baseline_values']) == 5
        assert len(result['opto_values']) > 0
        assert len(result['washout_values']) == 2

    def test_p_value_exists(self, synthetic_opto_animal):
        """Should compute p-value when enough data."""
        phases = assign_opto_phases(synthetic_opto_animal)
        result = expert_stability(
            synthetic_opto_animal.sessions, phases, stat_name='accuracy')
        assert not np.isnan(result['p_value'])
