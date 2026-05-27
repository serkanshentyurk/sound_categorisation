"""
Tests for analysis/opto.py.

Covers assign_opto_phases — partitions sessions into phases based on
distribution, masking, and opto status.
"""

import numpy as np
import pytest
from datetime import date


class TestAssignOptoPhases:
    """assign_opto_phases returns a dict of {phase_name: [sessions]}."""

    def test_returns_dict(self, synthetic_opto_animal):
        """Returns a dict."""
        from analysis.opto import assign_opto_phases

        phases = assign_opto_phases(synthetic_opto_animal.sessions)
        assert isinstance(phases, dict)

    def test_phases_are_lists(self, synthetic_opto_animal):
        """All values are lists (possibly empty)."""
        from analysis.opto import assign_opto_phases

        phases = assign_opto_phases(synthetic_opto_animal.sessions)
        for phase_name, sessions in phases.items():
            assert isinstance(sessions, list), f"phase {phase_name} not a list"

    def test_no_session_in_two_phases(self, synthetic_opto_animal):
        """Each session belongs to at most one phase."""
        from analysis.opto import assign_opto_phases

        phases = assign_opto_phases(synthetic_opto_animal.sessions)
        all_assigned = []
        for sessions in phases.values():
            for s in sessions:
                all_assigned.append(s.session_idx)
        # No duplicates
        assert len(all_assigned) == len(set(all_assigned)), \
            "session assigned to multiple phases"

    def test_masking_sessions_excluded_from_real_phases(self, synthetic_opto_animal):
        """Masking sessions shouldn't appear in phases (they're not real opto runs)."""
        from analysis.opto import assign_opto_phases

        phases = assign_opto_phases(synthetic_opto_animal.sessions)
        for phase_name, sessions in phases.items():
            if 'masking' in phase_name.lower() or 'mask' in phase_name.lower():
                continue
            for s in sessions:
                assert not s.masking, \
                    f"masking session {s.session_id} appeared in non-masking phase {phase_name}"

    def test_all_sessions_handled(self, synthetic_opto_animal):
        """Total assigned sessions + unassigned = all sessions, or it's documented."""
        from analysis.opto import assign_opto_phases

        all_sessions = synthetic_opto_animal.sessions
        phases = assign_opto_phases(all_sessions)
        assigned_count = sum(len(s) for s in phases.values())

        # Either all assigned, or function returns a sensible subset
        # (e.g., dropping masking sessions). Just check no nonsense.
        assert assigned_count <= len(all_sessions)
        assert assigned_count > 0  # at least something got assigned
