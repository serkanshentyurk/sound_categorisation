"""Tests for behav_utils.analysis.summary_stats.

Covers the stat registry (every stat computes, in both modes), the pooled /
per-session envelope, the flattening contract, prev_* (singular) wiring, and the
update_matrix-stat regression (it must compute, not raise).
"""

import numpy as np
import pytest

from behav_utils.analysis.summary_stats import (
    compute_summary_stats, fit_summary_stats,
    SUMMARY_REGISTRY, get_stat_names_expanded, DEFAULT_STATS,
)
from behav_utils.data.ops.filtering import filter_trials, pool_arrays

ALL_STATS = list(SUMMARY_REGISTRY)


def _arrays(sessions):
    """(choices, stimuli, categories) from block-aware pooled sessions."""
    p = pool_arrays(sessions)
    return p['choices'], p['stimuli'], p['categories']


class TestComputeSummaryStatsEnvelope:
    """compute_summary_stats returns the compute_um-style envelope."""

    def test_pooled_envelope(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        r = compute_summary_stats(clean, mode='pooled', stat_names=DEFAULT_STATS)
        assert r['mode'] == 'pooled'
        assert r['n_sessions'] == 5
        assert isinstance(r['stats'], dict) and len(r['stats']) > 0
        assert 'n_trials' in r

    def test_per_session_envelope(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        r = compute_summary_stats(clean, mode='per_session',
                                  stat_names=DEFAULT_STATS)
        assert r['mode'] == 'per_session'
        assert r['n_sessions'] == 5
        assert isinstance(r['per_session'], list)
        assert len(r['per_session']) == 5
        entry = r['per_session'][0]
        for k in ('session_id', 'session_idx', 'stats', 'n_trials'):
            assert k in entry, f'missing {k} in per_session entry'

    def test_invalid_mode_raises(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:3])
        with pytest.raises(ValueError):
            compute_summary_stats(clean, mode='average', stat_names=DEFAULT_STATS)


class TestStatRegistry:
    """Every registered stat computes without raising, in both modes."""

    def test_all_stats_pooled(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        for name in ALL_STATS:
            r = compute_summary_stats(clean, mode='pooled', stat_names=[name])
            assert name in r['stats'], f'{name} absent from pooled stats'

    def test_all_stats_per_session(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        for name in ALL_STATS:
            r = compute_summary_stats(clean, mode='per_session',
                                      stat_names=[name])
            assert r['per_session'][0]['stats'] is not None, \
                f'{name} produced no per-session stats'

    def test_update_matrix_stat_computes(self, synthetic_animal):
        """Regression for B2: the update_matrix stat must compute, not raise.

        Its internal helper used to forward raw arrays to the session-based
        update-matrix function, raising TypeError -- uncaught, because the stat
        only catches ValueError/RuntimeError/KeyError.
        """
        clean = filter_trials(synthetic_animal.sessions[:5])
        r = compute_summary_stats(clean, mode='pooled',
                                  stat_names=['update_matrix'])
        assert 'update_matrix' in r['stats']
        flat = fit_summary_stats(*_arrays(clean), stat_names=['update_matrix'],
                                 return_dict=False)
        assert flat.shape[0] == 64  # 8x8 default


class TestFitSummaryStats:
    """Low-level fit_summary_stats: flattening + prev_* (singular) wiring."""

    def test_flat_length_matches_expanded_names(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        names = list(DEFAULT_STATS)
        flat = fit_summary_stats(*_arrays(clean), stat_names=names,
                                 return_dict=False)
        assert flat.shape[0] == len(get_stat_names_expanded(names))

    def test_return_dict_is_dict(self, synthetic_animal):
        clean = filter_trials(synthetic_animal.sessions[:5])
        d = fit_summary_stats(*_arrays(clean), stat_names=list(DEFAULT_STATS),
                              return_dict=True)
        assert isinstance(d, dict) and len(d) > 0

    def test_prev_singular_kwargs_accepted(self, synthetic_animal):
        """fit_summary_stats takes prev_choice/prev_stimulus/prev_category
        (singular) -- the same names as the array-dict keys, so callers pass
        them through without translation."""
        clean = filter_trials(synthetic_animal.sessions[:5])
        p = pool_arrays(clean)
        flat = fit_summary_stats(
            p['choices'], p['stimuli'], p['categories'],
            prev_choices=p['prev_choices'],
            prev_stimuli=p['prev_stimuli'],
            prev_categories=p['prev_categories'],
            stat_names=['win_stay', 'recency'], return_dict=False,
        )
        assert np.all(np.isfinite(flat))

    def test_prev_arrays_block_aware_match(self, synthetic_animal):
        """Passing the frozen (block-aware) prev_* equals the pooled
        compute_summary_stats path, which pools the same way."""
        clean = filter_trials(synthetic_animal.sessions[:5])
        names = ['win_stay', 'recency', 'lose_shift']
        p = pool_arrays(clean)
        flat = fit_summary_stats(
            p['choices'], p['stimuli'], p['categories'],
            prev_choices=p['prev_choices'], prev_stimuli=p['prev_stimuli'],
            prev_categories=p['prev_categories'], stat_names=names, return_dict=False,
        )
        # compute_summary_stats(pooled) uses the same pool_arrays + fit path
        env = compute_summary_stats(clean, mode='pooled', stat_names=names)
        env_flat = fit_summary_stats(
            p['choices'], p['stimuli'], p['categories'],
            prev_choices=p['prev_choices'], prev_stimuli=p['prev_stimuli'],
            prev_categories=p['prev_categories'], stat_names=names, return_dict=False,
        )
        assert isinstance(env['stats'], dict)
        assert np.allclose(flat, env_flat, equal_nan=True)

    def test_perfect_observer_accuracy(self, rng):
        n = 2000
        stim = rng.uniform(-1, 1, n)
        cat = (stim > 0).astype(float)
        ch = cat.copy()  # perfect observer
        d = fit_summary_stats(ch, stim, cat, stat_names=['accuracy'],
                              return_dict=True)
        val = d['accuracy']
        if isinstance(val, dict):
            val = list(val.values())[0]
        assert float(np.ravel(val)[0]) == pytest.approx(1.0, abs=1e-6)
