"""Tests for ``behav_utils.analysis.comparison`` — the phase-comparison surface.

A *phase* is a list of ``SessionData`` (the output of ``filter_trials``), pooled
into one trial set. The public entry points are:

    compute_delta_stat   — N phases vs a reference, with permutation p + bootstrap CI
    compute_interaction  — difference of differences between two results
    compare_phases       — deprecated alias of compute_delta_stat

built on the pooling / resampling helpers ``pool_phase_arrays``,
``compute_stats_from_arrays``, ``bootstrap_phase_stats``,
``permute_phase_difference`` and ``summarise_draw_distribution``.

Performance note: the per-permutation / per-bootstrap psychometric fit costs
~0.4 s each and ``compute_delta_stat``'s psychometric *display* object adds a
large fixed CI-band cost, so exercising that path hundreds of times would make
the suite unusably slow (flagged as a real perf smell, not a test artefact).
The delta / interaction machinery is stat-agnostic, so it is driven here with
the fit-free ``accuracy`` statistic; the ``mu`` / ``sigma`` behaviour that needs
a psychometric fit is checked directly via ``compute_stats_from_arrays``, which
fits once.
"""
import numpy as np
import pytest
from datetime import date, timedelta

from behav_utils.data.structures import (
    SessionData, SessionMetadata, TrialData, AnimalData)
from behav_utils.analysis.comparison import (
    compute_delta_stat, compute_interaction, compare_phases,
    pool_phase_arrays, compute_stats_from_arrays, bootstrap_phase_stats,
    permute_phase_difference, summarise_draw_distribution)


# ── synthetic phase builder (criterion shift = mu displacement) ──────────────
def _phase(n_sessions=2, n=250, shift=0.0, noise=0.15, seed=0, with_aborts=False):
    sessions = []
    for i in range(n_sessions):
        rng = np.random.default_rng(seed * 100 + i)
        stim = rng.uniform(-1, 1, n)
        cat = (stim > 0).astype(float)
        ch = ((stim + rng.normal(0, noise, n) + shift) > 0).astype(float)
        abort = np.zeros(n, bool)
        if with_aborts:                      # ~10% no-response (NaN choice) trials
            miss = rng.choice(n, n // 10, replace=False)
            ch[miss] = np.nan
            abort[miss] = True
        tr = TrialData(trial_number=np.arange(n), stimulus=stim, category=cat,
                       choice=ch, outcome=(ch == cat).astype(float),
                       correct=(ch == cat), abort=abort, opto_on=np.zeros(n, bool))
        sessions.append(SessionData(
            session_id=f's{seed}_{i}', session_idx=i,
            date=date(2026, 1, 1) + timedelta(days=i),
            metadata=SessionMetadata(fields={'stage': 'Full_Task_Cont',
                                             'distribution': 'Uniform'}),
            trials=tr, session_type='regular'))
    return sessions


# fit-free accuracy runs are cheap, so counts can be realistic
FF = dict(n_permutations=60, n_bootstrap=40, seed=1)
SIG = dict(n_permutations=200, n_bootstrap=40, seed=1)


# ── compute_delta_stat (driven with the fit-free accuracy stat) ──────────────
class TestComputeDeltaStat:

    def test_returns_phases_contrasts_meta(self):
        r = compute_delta_stat({'ref': _phase(seed=1), 'x': _phase(shift=0.3, seed=2)},
                               stats=['accuracy'], reference='ref', **FF)
        assert {'phases', 'contrasts', 'meta'} <= set(r)
        assert set(r['phases']) == {'ref', 'x'}
        assert 'x_vs_ref' in r['contrasts']

    def test_phase_entry_shape(self):
        r = compute_delta_stat({'ref': _phase(seed=1), 'x': _phase(shift=0.3, seed=2)},
                               stats=['accuracy'], reference='ref', **FF)
        e = r['phases']['x']
        assert e['n_sessions'] == 2 and e['n_trials'] > 0
        assert 'accuracy' in e['stats']
        assert e['bootstrap_draws']['accuracy'].shape == (FF['n_bootstrap'],)
        lo, hi = e['stats_ci']['accuracy']
        assert lo <= hi

    def test_contrast_diff_equals_stat_difference(self):
        # correctness: diffs[stat] must equal stat(x) − stat(ref) exactly.
        r = compute_delta_stat({'ref': _phase(seed=1), 'x': _phase(shift=0.4, seed=2)},
                               stats=['accuracy'], reference='ref', **FF)
        con = r['contrasts']['x_vs_ref']
        expected = r['phases']['x']['stats']['accuracy'] - r['phases']['ref']['stats']['accuracy']
        assert con['diffs']['accuracy'] == pytest.approx(expected, abs=1e-9)
        assert con['label_a'] == 'x' and con['label_b'] == 'ref'
        assert con['n_a'] > 0 and con['n_b'] > 0

    def test_perm_p_in_unit_interval(self):
        r = compute_delta_stat({'ref': _phase(seed=1), 'x': _phase(shift=0.3, seed=2)},
                               stats=['accuracy'], reference='ref', **FF)
        assert 0.0 <= r['contrasts']['x_vs_ref']['perm_p']['accuracy'] <= 1.0

    def test_real_shift_is_flagged(self):
        # a clear criterion shift lowers accuracy → large diff, significant p.
        r = compute_delta_stat({'ref': _phase(seed=1, noise=0.1),
                                'x': _phase(shift=0.6, seed=2, noise=0.1)},
                               stats=['accuracy'], reference='ref', **SIG)
        con = r['contrasts']['x_vs_ref']
        assert abs(con['diffs']['accuracy']) > 0.03
        assert con['perm_p']['accuracy'] < 0.05

    def test_identical_phases_null_result(self):
        base = _phase(seed=7, noise=0.1)
        r = compute_delta_stat({'ref': base, 'x': base}, stats=['accuracy'],
                               reference='ref', **SIG)
        con = r['contrasts']['x_vs_ref']
        assert abs(con['diffs']['accuracy']) < 1e-9      # exact, no difference
        assert con['perm_p']['accuracy'] > 0.05          # no false positive

    def test_list_input_with_labels_matches_dict(self):
        r = compute_delta_stat([_phase(seed=1), _phase(shift=0.3, seed=2)],
                               stats=['accuracy'], labels=['ref', 'x'],
                               reference='ref', **FF)
        assert 'x_vs_ref' in r['contrasts']

    def test_three_phases_give_two_contrasts(self):
        r = compute_delta_stat({'ref': _phase(seed=1),
                                'a': _phase(shift=0.3, seed=2),
                                'b': _phase(shift=-0.3, seed=3)},
                               stats=['accuracy'], reference='ref', **FF)
        assert set(r['contrasts']) == {'a_vs_ref', 'b_vs_ref'}

    def test_fewer_than_two_phases_raises(self):
        with pytest.raises(ValueError):
            compute_delta_stat({'only': _phase()}, stats=['accuracy'],
                               reference='only', **FF)

    def test_reference_not_in_phases_raises(self):
        with pytest.raises(ValueError):
            compute_delta_stat({'ref': _phase(seed=1), 'x': _phase(seed=2)},
                               stats=['accuracy'], reference='absent', **FF)

    def test_compare_phases_is_deprecated_alias(self):
        with pytest.warns(DeprecationWarning):
            r = compare_phases({'ref': _phase(seed=1), 'x': _phase(shift=0.3, seed=2)},
                               stats=['accuracy'], reference='ref', **FF)
        assert 'x_vs_ref' in r['contrasts']


# ── compute_interaction (difference of differences) ──────────────────────────
class TestComputeInteraction:

    def _paired_results(self):
        C = _phase(seed=3, noise=0.1)                     # shared reference
        A = _phase(shift=0.3, seed=1, noise=0.1)
        B = _phase(shift=-0.3, seed=2, noise=0.1)
        rA = compute_delta_stat({'C': C, 'A': A}, stats=['accuracy'], reference='C', **FF)
        rB = compute_delta_stat({'C': C, 'B': B}, stats=['accuracy'], reference='C', **FF)
        return rA, rB

    def test_returns_per_stat_dicts(self):
        rA, rB = self._paired_results()
        inter = compute_interaction(rA, rB, contrast='A_vs_C', contrast_b='B_vs_C')
        assert 'meta' in inter and 'accuracy' in inter
        for k in ('delta_a', 'delta_b', 'interaction', 'ci_lo', 'ci_hi'):
            assert k in inter['accuracy']

    def test_shared_reference_cancellation(self):
        # (A − C) − (B − C) == A − B, exactly (the cancellation identity).
        rA, rB = self._paired_results()
        inter = compute_interaction(rA, rB, contrast='A_vs_C', contrast_b='B_vs_C')
        d_a = rA['contrasts']['A_vs_C']['diffs']['accuracy']
        d_b = rB['contrasts']['B_vs_C']['diffs']['accuracy']
        assert inter['accuracy']['interaction'] == pytest.approx(d_a - d_b, abs=1e-9)
        assert inter['accuracy']['delta_a'] == pytest.approx(d_a, abs=1e-9)
        assert inter['accuracy']['delta_b'] == pytest.approx(d_b, abs=1e-9)


# ── psychometric mu / sigma behaviour (single-fit, via the helper) ───────────
class TestPsychometricShift:

    def test_criterion_shift_moves_mu_preserves_sigma(self):
        # A criterion shift displaces mu but not sigma (sensitivity) — checked on
        # a single fit each side rather than through the slow display path.
        ref = compute_stats_from_arrays(pool_phase_arrays(_phase(seed=1, noise=0.1)),
                                        ['psychometric'])
        shifted = compute_stats_from_arrays(pool_phase_arrays(_phase(shift=0.5, seed=2, noise=0.1)),
                                            ['psychometric'])
        assert abs(shifted['mu'] - ref['mu']) > 0.2       # criterion moves …
        assert abs(shifted['sigma'] - ref['sigma']) < 0.06  # … sensitivity intact


# ── pooling / resampling helpers ─────────────────────────────────────────────
class TestPoolPhaseArrays:

    def test_keys_and_equal_lengths(self):
        arr = pool_phase_arrays(_phase(n_sessions=2, n=200, seed=1))
        assert {'stimuli', 'choices', 'categories'} <= set(arr)
        assert len({len(arr[k]) for k in ('stimuli', 'choices', 'categories')}) == 1

    def test_no_response_trials_dropped(self):
        # pool_phase_arrays drops non-responses (NaN choice), so a phase with
        # missed trials pools to fewer rows than the same phase without.
        clean = pool_phase_arrays(_phase(n_sessions=1, n=300, seed=1, with_aborts=False))
        missed = pool_phase_arrays(_phase(n_sessions=1, n=300, seed=1, with_aborts=True))
        assert len(missed['stimuli']) < len(clean['stimuli'])
        assert np.all(np.isfinite(missed['choices']))       # NaNs really gone


class TestComputeStatsFromArrays:

    def test_flattens_families_to_scalar_names(self):
        arr = pool_phase_arrays(_phase(seed=1))
        stats = compute_stats_from_arrays(arr, ['psychometric', 'accuracy'])
        assert {'mu', 'sigma', 'lapse_low', 'lapse_high', 'accuracy'} <= set(stats)
        assert all(np.isscalar(v) for v in stats.values())

    def test_unknown_family_strict_raises(self):
        arr = pool_phase_arrays(_phase(seed=1))
        with pytest.raises(Exception):
            compute_stats_from_arrays(arr, ['not_a_real_family'], strict=True)


class TestBootstrapPhaseStats:

    def test_shape_and_reproducible(self):
        ph = _phase(seed=1)
        a = bootstrap_phase_stats(ph, ['accuracy'], n_draws=64, seed=0)
        b = bootstrap_phase_stats(ph, ['accuracy'], n_draws=64, seed=0)
        assert a['accuracy'].shape == (64,)
        np.testing.assert_array_equal(a['accuracy'], b['accuracy'])

    def test_draws_centre_near_observed(self):
        ph = _phase(seed=1, noise=0.1)
        obs = compute_stats_from_arrays(pool_phase_arrays(ph), ['accuracy'])['accuracy']
        draws = bootstrap_phase_stats(ph, ['accuracy'], n_draws=200, seed=0)['accuracy']
        assert abs(np.mean(draws) - obs) < 0.05


class TestPermutePhaseDifference:

    def test_shape(self):
        d = permute_phase_difference(_phase(seed=1), _phase(shift=0.3, seed=2),
                                     ['accuracy'], n_draws=64, seed=0)
        assert d['accuracy'].shape == (64,)

    def test_null_centres_near_zero_for_like_phases(self):
        d = permute_phase_difference(_phase(seed=1, noise=0.1), _phase(seed=1, noise=0.1),
                                     ['accuracy'], n_draws=200, seed=0)['accuracy']
        assert abs(np.mean(d)) < 0.03


class TestSummariseDrawDistribution:

    def test_keys_and_ordering(self):
        draws = np.random.default_rng(0).normal(0.5, 0.1, 500)
        s = summarise_draw_distribution(draws, observed=0.5)
        assert {'ci_lo', 'ci_hi', 'p_two_sided', 'n_draws', 'median'} <= set(s)
        assert s['ci_lo'] <= s['median'] <= s['ci_hi']
        assert 0.0 <= s['p_two_sided'] <= 1.0
        assert s['n_draws'] == 500

    def test_far_null_gives_small_p(self):
        draws = np.random.default_rng(0).normal(1.0, 0.05, 500)   # far from 0
        assert summarise_draw_distribution(draws, null_value=0.0)['p_two_sided'] < 0.05

    def test_null_inside_distribution_gives_large_p(self):
        draws = np.random.default_rng(0).normal(0.0, 0.2, 500)    # straddles 0
        assert summarise_draw_distribution(draws, null_value=0.0)['p_two_sided'] > 0.2
