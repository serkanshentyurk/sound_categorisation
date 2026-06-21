"""Tests for analysis.consensus: the majority-vote rule and the per-animal
× per-method assignment table."""

import numpy as np
import pandas as pd
import pytest

from analysis.consensus import _compute_consensus, load_all_assignments, _method_dir
from utils.cv_utils import save_cv_result


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


class TestLoadAllAssignments:
    """load_all_assignments builds the per-animal × per-method winner table.

    results_dir is monkeypatched into a tmp tree so no real data root is touched;
    fabricated BE/SC files in the neutral schema stand in for run_gs / run_sbi
    output.
    """

    RUN, COH = 'r', 'c'

    def _patch(self, monkeypatch, root):
        monkeypatch.setattr(
            'analysis.consensus.results_dir',
            lambda source, run, cohort, ft: root / source / run / f'{cohort}_{ft}')

    def _write_method(self, source, rep, ft, animal, true_model, be_lo, sc_lo):
        """Write a paired BE/SC result (8 reps) where BE wins iff be_lo < sc_lo."""
        d = _method_dir(source, rep, ft, self.RUN, self.COH)
        tp = {'sigma_percep': 0.2, 'eta_learning': 0.1}
        be = [be_lo + 0.001 * i for i in range(8)]
        sc = [sc_lo + 0.001 * i for i in range(8)]
        save_cv_result(d / f'{animal}_BE.pkl', animal, 'BE',
                       [{'rep': i, 'test_error': be[i], 'best_params': tp}
                        for i in range(8)],
                       ft, true_model=true_model, true_params=tp)
        save_cv_result(d / f'{animal}_SC.pkl', animal, 'SC',
                       [{'rep': i, 'test_error': sc[i], 'best_params': {'gamma': 0.5}}
                        for i in range(8)],
                       ft, true_model=true_model, true_params=tp)

    def test_winner_per_method_and_consensus(self, tmp_path, monkeypatch):
        """Two agreeing methods → per-method winners + consensus + correctness."""
        self._patch(monkeypatch, tmp_path)
        methods = [('grid_search', None, 'update_matrix'),
                   ('sbi', 'pooled', 'update_matrix')]
        for s, rp, ft in methods:
            self._write_method(s, rp, ft, 'A', 'BE', be_lo=0.10, sc_lo=0.20)
            self._write_method(s, rp, ft, 'B', 'SC', be_lo=0.20, sc_lo=0.10)
        df = load_all_assignments(self.RUN, self.COH, methods=methods).set_index('id')
        assert df.loc['A', 'GS-UM'] == 'BE' and df.loc['A', 'SBI-pooled-UM'] == 'BE'
        assert df.loc['A', 'Consensus'] == 'BE'
        assert df.loc['B', 'Consensus'] == 'SC'
        assert bool(df.loc['A', 'consensus_correct']) is True
        assert bool(df.loc['B', 'consensus_correct']) is True

    def test_column_schema(self, tmp_path, monkeypatch):
        """Each method contributes winner / p / be / sc columns."""
        self._patch(monkeypatch, tmp_path)
        methods = [('grid_search', None, 'update_matrix')]
        self._write_method('grid_search', None, 'update_matrix', 'A', 'BE', 0.10, 0.20)
        df = load_all_assignments(self.RUN, self.COH, methods=methods)
        for col in ['id', 'GS-UM', 'GS-UM_p', 'GS-UM_be', 'GS-UM_sc',
                    'true_model', 'Consensus', 'consensus_correct']:
            assert col in df.columns

    def test_disagreement_is_split(self, tmp_path, monkeypatch):
        """One BE method + one SC method, both significant → Split."""
        self._patch(monkeypatch, tmp_path)
        methods = [('grid_search', None, 'update_matrix'),
                   ('sbi', 'pooled', 'update_matrix')]
        self._write_method('grid_search', None, 'update_matrix', 'A', 'BE', 0.10, 0.20)
        self._write_method('sbi', 'pooled', 'update_matrix', 'A', 'BE', 0.20, 0.10)
        df = load_all_assignments(self.RUN, self.COH, methods=methods).set_index('id')
        assert df.loc['A', 'GS-UM'] == 'BE' and df.loc['A', 'SBI-pooled-UM'] == 'SC'
        assert df.loc['A', 'Consensus'] == 'Split'

    def test_missing_method_dir_is_nan(self, tmp_path, monkeypatch):
        """A requested method with no results directory → NaN column, others vote."""
        self._patch(monkeypatch, tmp_path)
        methods = [('grid_search', None, 'update_matrix'),
                   ('sbi', 'pooled', 'update_matrix')]
        self._write_method('grid_search', None, 'update_matrix', 'A', 'BE', 0.10, 0.20)
        df = load_all_assignments(self.RUN, self.COH, methods=methods).set_index('id')
        assert df.loc['A', 'GS-UM'] == 'BE'
        assert pd.isna(df.loc['A', 'SBI-pooled-UM'])
        assert df.loc['A', 'Consensus'] == 'BE'      # single significant vote

    def test_custom_single_method(self, tmp_path, monkeypatch):
        """The method list is configurable (here a single GS-CP method)."""
        self._patch(monkeypatch, tmp_path)
        methods = [('grid_search', None, 'conditional_psych')]
        self._write_method('grid_search', None, 'conditional_psych', 'A', 'BE', 0.10, 0.20)
        df = load_all_assignments(self.RUN, self.COH, methods=methods).set_index('id')
        assert 'GS-CP' in df.columns
        assert df.loc['A', 'Consensus'] == 'BE'

    def test_no_truth_omits_correct_columns(self, tmp_path, monkeypatch):
        """Files without true_model → no true_model / consensus_correct columns."""
        self._patch(monkeypatch, tmp_path)
        d = _method_dir('grid_search', None, 'update_matrix', self.RUN, self.COH)
        save_cv_result(d / 'A_BE.pkl', 'A', 'BE',
                       [{'rep': i, 'test_error': 0.1 + 0.001 * i, 'best_params': {'a': 1}}
                        for i in range(8)], 'update_matrix')
        save_cv_result(d / 'A_SC.pkl', 'A', 'SC',
                       [{'rep': i, 'test_error': 0.2 + 0.001 * i, 'best_params': {'a': 1}}
                        for i in range(8)], 'update_matrix')
        df = load_all_assignments(self.RUN, self.COH,
                                  methods=[('grid_search', None, 'update_matrix')])
        assert 'consensus_correct' not in df.columns
        assert 'true_model' not in df.columns
        assert df.set_index('id').loc['A', 'Consensus'] == 'BE'
