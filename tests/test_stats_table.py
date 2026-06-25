"""Tests for behav_utils.analysis.stats_table (Tier A: extract_stats /
extract_matched) and the selection-provenance contract with filter_phase.

Uses the conftest session fixtures (synthetic_animal / synthetic_opto_animal) and
filter_trials, mirroring tests/test_summary_stats.py. The _resolve_meta guards are
tested by stamping filter_info['selection'] directly; one integration test runs
the real filter_phase -> extract_stats path so the writer and reader are checked
together.
"""
import numpy as np
import pandas as pd
import pytest

from behav_utils.analysis.stats_table import extract_stats, extract_matched, StatTable
from behav_utils.analysis.downsample import calculate_min_n
from behav_utils.analysis.summary_stats import is_exchangeable
from behav_utils.data.ops.filtering import filter_trials

SYMMETRIC = ['recency', 'win_stay', 'lose_shift']


def _clean(animal, n=5):
    return filter_trials(animal.sessions[:n])


def _stamp(sessions, **selection):
    """Set filter_info['selection'] directly (what filter_phase does)."""
    for s in sessions:
        s.filter_info = {**(s.filter_info or {}), 'selection': dict(selection)}
    return sessions


class TestExtractStatsPooled:

    def test_one_row_per_stat(self, synthetic_animal):
        table = extract_stats(_clean(synthetic_animal), animal_id='SS01',
                              stats=SYMMETRIC, mode='pooled')
        assert isinstance(table, StatTable)
        assert set(table.estimates['stat']) == set(SYMMETRIC)
        assert table.estimates['session'].isna().all()      # animal-level
        assert table.replicates is None

    def test_animal_id_and_meta_carried(self, synthetic_animal):
        table = extract_stats(_clean(synthetic_animal), animal_id='SS01',
                              stats=['recency'], mode='pooled',
                              meta={'genotype': 'het'})
        assert (table.estimates['animal'] == 'SS01').all()
        assert (table.estimates['genotype'] == 'het').all()

    def test_psychometric_expands(self, synthetic_animal):
        table = extract_stats(_clean(synthetic_animal), animal_id='SS01',
                              stats=['psychometric'], mode='pooled')
        assert {'mu', 'sigma', 'lapse_low', 'lapse_high'} <= set(table.estimates['stat'])


class TestExtractStatsPerSession:

    def test_row_per_session(self, synthetic_animal):
        table = extract_stats(_clean(synthetic_animal, n=5), animal_id='SS01',
                              stats=['recency', 'win_stay'], mode='per_session')
        assert table.estimates['session'].nunique() == 5
        assert len(table.estimates) == 5 * 2

    def test_invalid_mode_raises(self, synthetic_animal):
        with pytest.raises(ValueError):
            extract_stats(_clean(synthetic_animal), animal_id='SS01',
                          stats=['recency'], mode='average')


class TestBootstrap:

    def test_n_boot_fills_ci_and_reps(self, synthetic_animal):
        table = extract_stats(_clean(synthetic_animal), animal_id='SS01',
                              stats=SYMMETRIC, mode='pooled', n_boot=100, seed=0)
        assert table.estimates['ci_lo_within'].notna().all()
        assert (table.estimates['ci_lo_within'] <= table.estimates['ci_hi_within']).all()
        assert table.replicates is not None
        assert len(table.replicates) == len(SYMMETRIC) * 100
        assert table.replicates['stat'].nunique() == len(SYMMETRIC)

    def test_n_boot_requires_pooled(self, synthetic_animal):
        with pytest.raises(ValueError):
            extract_stats(_clean(synthetic_animal), animal_id='SS01',
                          stats=['recency'], mode='per_session', n_boot=50)


class TestExchangeability:

    def test_lag1_stats_are_exchangeable(self):
        for name in ('recency', 'win_stay', 'lose_shift', 'accuracy'):
            assert is_exchangeable(name) is True

    def test_order_dependent_stats_flagged(self):
        for name in ('update_matrix', 'logistic_history',
                     'perseveration', 'history_interaction_r2'):
            assert is_exchangeable(name) is False

    def test_bootstrapping_order_dependent_raises(self, synthetic_animal):
        with pytest.raises(ValueError):
            extract_stats(_clean(synthetic_animal), animal_id='SS01',
                          stats=['logistic_history'], mode='pooled', n_boot=10)


class TestProvenance:

    def test_filter_phase_autofills_columns(self, synthetic_opto_animal):
        # integration: real writer (filter_phase) -> real reader (extract_stats)
        from analysis.phase import filter_phase
        clean = filter_phase(synthetic_opto_animal, dist='uniform',
                             session_type='regular')
        assert clean, 'fixture should yield uniform/regular sessions'
        table = extract_stats(clean, animal_id='SS01', stats=['recency'], mode='pooled')
        assert (table.estimates['distribution'] == 'uniform').all()
        assert (table.estimates['session_type'] == 'regular').all()
        assert (table.estimates['trial_type'] == 'all').all()

    def test_explicit_meta_overrides_provenance(self, synthetic_animal):
        clean = _stamp(_clean(synthetic_animal), distribution='hard_a',
                       session_type='opto', trial_type='opto')
        table = extract_stats(clean, animal_id='SS01', stats=['recency'],
                              mode='pooled', meta={'distribution': 'OVERRIDE'})
        assert (table.estimates['distribution'] == 'OVERRIDE').all()
        assert (table.estimates['trial_type'] == 'opto').all()        # untouched

    def test_disagreeing_selections_raise(self, synthetic_animal):
        a = _stamp(_clean(synthetic_animal, n=3), distribution='hard_a',
                   session_type='opto', trial_type='opto')
        b = _stamp(_clean(synthetic_animal, n=2), distribution='hard_b',
                   session_type='opto', trial_type='opto')
        with pytest.raises(ValueError):
            extract_stats(a + b, animal_id='SS01', stats=['recency'], mode='pooled')

    def test_mixed_tagged_untagged_raise(self, synthetic_animal):
        tagged = _stamp(_clean(synthetic_animal, n=3), distribution='hard_a',
                        session_type='opto', trial_type='opto')
        untagged = filter_trials(synthetic_animal.sessions[5:8])
        with pytest.raises(ValueError):
            extract_stats(tagged + untagged, animal_id='SS01',
                          stats=['recency'], mode='pooled')

    def test_all_untagged_has_no_selection_columns(self, synthetic_animal):
        table = extract_stats(_clean(synthetic_animal), animal_id='SS01',
                              stats=['recency'], mode='pooled')
        for col in ('distribution', 'session_type', 'trial_type'):
            assert col not in table.estimates.columns


class TestExtractMatched:

    def test_matched_count(self, synthetic_animal):
        clean = _clean(synthetic_animal)
        target = min(calculate_min_n([clean], unit='trials'), 80)
        table = extract_matched(clean, target, animal_id='SS01',
                                stats=['recency', 'win_stay'], n_repeats=20)
        assert (table.estimates['n_trials'] == target).all()
        assert table.estimates['session'].isna().all()
        assert table.replicates is None

    def test_matched_autofills_provenance(self, synthetic_animal):
        clean = _stamp(_clean(synthetic_animal), distribution='hard_a',
                       session_type='opto', trial_type='opto')
        target = min(calculate_min_n([clean], unit='trials'), 80)
        table = extract_matched(clean, target, animal_id='SS01',
                                stats=['recency'], n_repeats=10)
        assert (table.estimates['distribution'] == 'hard_a').all()
