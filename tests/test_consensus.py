"""Tests for analysis.consensus majority vote logic."""

import numpy as np
import pandas as pd
import pytest

from analysis.consensus import _compute_consensus


class TestComputeConsensus:
    """Tests for _compute_consensus majority vote."""

    def _make_row(self, gs_um='BE', gs_cp='BE', sbi_um='BE', sbi_cp='BE',
                  gs_um_p=0.001, gs_cp_p=0.001, sbi_um_p=0.001, sbi_cp_p=0.001):
        """Helper: create a row dict matching expected format."""
        return {
            'GS-UM': gs_um, 'GS-UM_p': gs_um_p,
            'GS-CP': gs_cp, 'GS-CP_p': gs_cp_p,
            'SBI-UM': sbi_um, 'SBI-UM_p': sbi_um_p,
            'SBI-CP': sbi_cp, 'SBI-CP_p': sbi_cp_p,
        }

    def test_unanimous_be(self):
        """All methods say BE with p<0.05 → BE."""
        row = self._make_row('BE', 'BE', 'BE', 'BE')
        assert _compute_consensus(row) == 'BE'

    def test_unanimous_sc(self):
        """All methods say SC with p<0.05 → SC."""
        row = self._make_row('SC', 'SC', 'SC', 'SC')
        assert _compute_consensus(row) == 'SC'

    def test_majority_be(self):
        """3 BE, 1 SC → BE."""
        row = self._make_row('BE', 'BE', 'BE', 'SC')
        assert _compute_consensus(row) == 'BE'

    def test_majority_sc(self):
        """1 BE, 3 SC → SC."""
        row = self._make_row('BE', 'SC', 'SC', 'SC')
        assert _compute_consensus(row) == 'SC'

    def test_split_2v2(self):
        """2 BE, 2 SC → Split."""
        row = self._make_row('BE', 'BE', 'SC', 'SC')
        assert _compute_consensus(row) == 'Split'

    def test_non_significant_excluded(self):
        """Methods with p>0.05 are excluded from vote."""
        row = self._make_row(
            'BE', 'SC', 'SC', 'SC',
            gs_um_p=0.001, gs_cp_p=0.5, sbi_um_p=0.001, sbi_cp_p=0.001,
        )
        # GS-CP excluded (p=0.5). Votes: BE(GS-UM), SC(SBI-UM), SC(SBI-CP) → SC
        assert _compute_consensus(row) == 'SC'

    def test_all_non_significant(self):
        """All methods p>0.05 → Unclear."""
        row = self._make_row(
            gs_um_p=0.5, gs_cp_p=0.5, sbi_um_p=0.5, sbi_cp_p=0.5,
        )
        assert _compute_consensus(row) == 'Unclear'

    def test_missing_values(self):
        """NaN assignments → excluded from vote."""
        row = self._make_row()
        row['GS-UM'] = np.nan
        row['GS-CP'] = np.nan
        row['GS-UM_p'] = np.nan
        row['GS-CP_p'] = np.nan
        # Only SBI votes: BE, BE → BE
        assert _compute_consensus(row) == 'BE'

    def test_custom_alpha(self):
        """Custom alpha threshold."""
        row = self._make_row(
            gs_um_p=0.02, gs_cp_p=0.02, sbi_um_p=0.02, sbi_cp_p=0.02,
        )
        # With alpha=0.01, all are excluded
        assert _compute_consensus(row, alpha=0.01) == 'Unclear'
        # With alpha=0.05, all pass
        assert _compute_consensus(row, alpha=0.05) == 'BE'
