"""
Tests for utils/cv_utils.py — the cross-method CV comparator and schema I/O.

compare_models / save_cv_result / load_cv_results / compute_seed_errors are the
load-bearing, method-agnostic core shared by grid-search and SBI, so they are
exercised directly here (torch-free).
"""

import numpy as np
import pytest


# ── params_to_str ────────────────────────────────────────────────────────────

class TestParamsToStr:
    def test_returns_string(self):
        from utils.cv_utils import params_to_str
        result = params_to_str({'sigma_percep': 0.1, 'eta_learning': 0.3})
        assert isinstance(result, str) and len(result) > 0

    def test_none_returns_empty(self):
        from utils.cv_utils import params_to_str
        assert params_to_str(None) == ''


# ── compute_seed_errors ──────────────────────────────────────────────────────

class TestComputeSeedErrors:
    def test_extracts_errors_and_best(self):
        from utils.cv_utils import compute_seed_errors
        data = {'results': [
            {'rep': 0, 'test_error': 0.3, 'best_params': {'a': 1}},
            {'rep': 1, 'test_error': 0.1, 'best_params': {'a': 2}},
        ]}
        errors, best = compute_seed_errors(data)
        assert errors == [0.3, 0.1]
        assert best == {'a': 2}                 # from the lowest-error rep

    def test_filters_nan_errors(self):
        from utils.cv_utils import compute_seed_errors
        data = {'results': [
            {'rep': 0, 'test_error': np.nan, 'best_params': {'a': 1}},
            {'rep': 1, 'test_error': 0.2, 'best_params': {'a': 2}},
        ]}
        errors, best = compute_seed_errors(data)
        assert errors == [0.2]
        assert best == {'a': 2}

    def test_empty_results(self):
        from utils.cv_utils import compute_seed_errors
        errors, best = compute_seed_errors({'results': []})
        assert errors == [] and best is None


# ── compare_models (within-method BE vs SC) ──────────────────────────────────

class TestCompareModels:
    def test_be_wins_on_lower_mean(self):
        from utils.cv_utils import compare_models
        _, comp = compare_models('A', [0.1, 0.1, 0.1], [0.3, 0.3, 0.3])
        assert comp['winner'].iloc[0] == 'BE'
        assert comp['be_mean'].iloc[0] < comp['sc_mean'].iloc[0]

    def test_sc_wins_on_lower_mean(self):
        from utils.cv_utils import compare_models
        _, comp = compare_models('A', [0.5, 0.5], [0.2, 0.2])
        assert comp['winner'].iloc[0] == 'SC'

    def test_p_value_significant_when_separated(self):
        from utils.cv_utils import compare_models
        be = [0.10 + 0.001 * i for i in range(8)]
        sc = [0.20 + 0.001 * i for i in range(8)]
        _, comp = compare_models('A', be, sc)
        assert comp['p_value'].iloc[0] < 0.05   # all BE<SC -> significant

    def test_empty_returns_none(self):
        from utils.cv_utils import compare_models
        long_df, comp = compare_models('A', [], [0.1])
        assert long_df is None and comp is None

    def test_long_df_has_seed_column(self):
        from utils.cv_utils import compare_models
        long_df, _ = compare_models('A', [0.1, 0.2], [0.3, 0.4])
        assert 'seed' in long_df.columns        # plot_cv_comparison compatibility
        assert set(long_df['model']) == {'BE', 'SC'}


# ── save_cv_result / load_cv_results round-trip ──────────────────────────────

class TestSaveLoadRoundtrip:
    def _write_pair(self, d, aid, be_err, sc_err, true_model, tp):
        from utils.cv_utils import save_cv_result
        save_cv_result(d / f'{aid}_BE.pkl', aid, 'BE',
                       [{'rep': i, 'test_error': e, 'best_params': tp}
                        for i, e in enumerate(be_err)],
                       'update_matrix', true_model=true_model, true_params=tp)
        save_cv_result(d / f'{aid}_SC.pkl', aid, 'SC',
                       [{'rep': i, 'test_error': e, 'best_params': {'gamma': 0.5}}
                        for i, e in enumerate(sc_err)],
                       'update_matrix', true_model=true_model, true_params=tp)

    def test_roundtrip_comparison_and_recovery(self, tmp_path):
        from utils.cv_utils import load_cv_results
        tp = {'sigma_percep': 0.2, 'eta_learning': 0.1}
        self._write_pair(tmp_path, 'A', [0.1, 0.1, 0.1], [0.3, 0.3, 0.3], 'BE', tp)
        cv = load_cv_results(tmp_path)
        assert cv.comparison['winner'].iloc[0] == 'BE'
        assert bool(cv.comparison['correct'].iloc[0]) is True   # winner == true_model
        assert set(cv.recovery['param']) == {'sigma_percep', 'eta_learning'}

    def test_unpaired_animal_dropped(self, tmp_path):
        from utils.cv_utils import save_cv_result, load_cv_results
        save_cv_result(tmp_path / 'X_BE.pkl', 'X', 'BE',
                       [{'rep': 0, 'test_error': 0.1, 'best_params': {'a': 1}}],
                       'update_matrix')                          # only BE, no SC
        cv = load_cv_results(tmp_path)
        assert len(cv.comparison) == 0                          # needs both fits

    def test_partials_subdir_skipped(self, tmp_path):
        from utils.cv_utils import save_cv_result, load_cv_results
        tp = {'sigma_percep': 0.2}
        self._write_pair(tmp_path, 'A', [0.1, 0.1], [0.3, 0.3], 'BE', tp)
        (tmp_path / 'partials').mkdir()
        save_cv_result(tmp_path / 'partials' / 'A_BE_seed0.pkl', 'A', 'BE',
                       [{'rep': 0, 'test_error': 0.9, 'best_params': {'a': 1}}],
                       'update_matrix')
        cv = load_cv_results(tmp_path)
        assert len(cv.comparison) == 1                          # partials ignored
