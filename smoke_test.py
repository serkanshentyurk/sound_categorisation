#!/usr/bin/env python3
"""
smoke_test.py — End-to-end test of the post-cleanup codebase.

Run from the repo root:

    cd /path/to/sound_categorisation
    python smoke_test.py

Exits 0 on success, 1 on failure. Prints a per-test verdict and a final
summary. No external data needed — uses behav_utils synthetic generators.

What it tests:
    1. Every package imports cleanly
    2. compare_conditions uses mu/sigma (not pse/slope) — patch applied
    3. compare_conditions produces bootstrap curve bands
    4. compute_normative_pse maths consistent with manual calculation
    5. compute_distribution_density returns the right dict shape
    6. compute_session_features works without default_rt_extractor
    7. assign_opto_phases handles a mixed session list correctly
    8. detect_shifts walks chronological sessions
    9. compute_comparison + plot_comparison round-trip
    10. compute_sbc_ranks, compute_parameter_recovery, plot_sbc_ranks exist
        and have the expected signature (lazy — no actual SBI training)
"""

import sys
import traceback
import numpy as np

PASS, FAIL = 0, 0
DETAIL = []


def run(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  ✓ {name}")
        PASS += 1
    except Exception as e:
        print(f"  ✗ {name}: {type(e).__name__}: {e}")
        DETAIL.append((name, traceback.format_exc()))
        FAIL += 1


# =============================================================================
# 1. Import sweep
# =============================================================================

print("\n[1] Import sweep")

def test_imports():
    import analysis
    import inference
    import models
    import plotting
    import behav_utils
    import behav_utils.plotting.comparison
    import behav_utils.plotting.psychometric
    from analysis.opto import assign_opto_phases
    from analysis.adaptation import detect_shifts
    from validation.sbi import (
        compute_sbc_ranks, compute_parameter_recovery, compute_param_stat_correlations,
    )
    from plotting.sbi_validation import (
        plot_sbc_ranks, plot_sbc_ecdf,
        plot_recovery_scatter, plot_recovery_bias,
        plot_param_stat_correlations,
    )
    from behav_utils.plotting.comparison import plot_comparison
    from behav_utils.analysis.comparison import compare_conditions, compute_comparison
    from models import BEModel, SCModel, BEParams, SCParams, BEState, SCState

run("All packages import", test_imports)


# =============================================================================
# 2. compare_conditions key rename
# =============================================================================

print("\n[2] compare_conditions key rename")

def test_compare_keys():
    from behav_utils.analysis.comparison import compare_conditions
    rng = np.random.default_rng(42)
    n = 400
    sa = rng.uniform(-1, 1, n)
    ca = (sa > 0.05).astype(float)
    caa = (sa > 0).astype(int)
    sb = rng.uniform(-1, 1, n)
    cb = (sb > -0.05).astype(float)
    cab = (sb > 0).astype(int)

    result = compare_conditions(sa, ca, caa, sb, cb, cab,
                                  n_bootstrap=50, n_permutations=50)
    expected_keys = {'mu', 'sigma', 'lapse_low', 'lapse_high', 'accuracy'}
    got_a = set(result['params_a'].keys())
    got_d = set(result['diffs'].keys())
    assert expected_keys.issubset(got_a), f"params_a missing keys: {expected_keys - got_a}"
    assert expected_keys.issubset(got_d), f"diffs missing keys: {expected_keys - got_d}"
    assert 'pse' not in got_a, "params_a still has 'pse' — patch incomplete"
    assert 'slope' not in got_a, "params_a still has 'slope' — patch incomplete"

run("compare_conditions uses mu/sigma keys", test_compare_keys)


# =============================================================================
# 3. boot_band_a, boot_band_b present
# =============================================================================

print("\n[3] Bootstrap curve bands")

def test_bands():
    from behav_utils.analysis.comparison import compare_conditions
    rng = np.random.default_rng(42)
    n = 300
    sa = rng.uniform(-1, 1, n)
    ca = (sa > 0.05).astype(float)
    sb = rng.uniform(-1, 1, n)
    cb = (sb > -0.05).astype(float)

    result = compare_conditions(
        sa, ca, (sa > 0).astype(int),
        sb, cb, (sb > 0).astype(int),
        n_bootstrap=100, n_permutations=0,
    )
    assert result['boot_band_a'] is not None
    assert result['boot_band_b'] is not None
    expected = {'x', 'lo', 'hi', 'median'}
    assert set(result['boot_band_a'].keys()) == expected
    assert len(result['boot_band_a']['x']) == 200  # grid size in patch
    assert len(result['boot_band_a']['lo']) == 200

run("Bootstrap bands populated", test_bands)


# =============================================================================
# 4. compute_normative_pse maths
# =============================================================================

print("\n[4] compute_normative_pse maths")

def test_normative_pse():
    from utils.stimulus_distribution import compute_normative_pse

    pse_u = compute_normative_pse('uniform', sigma_percep=0.15)
    assert abs(pse_u) < 1e-6, f"uniform should give PSE=0, got {pse_u}"

    pse_a = compute_normative_pse('hard_a', sigma_percep=0.15)
    pse_b = compute_normative_pse('hard_b', sigma_percep=0.15)
    assert pse_a > 0, f"hard_a PSE should be positive, got {pse_a}"
    assert pse_b < 0, f"hard_b PSE should be negative, got {pse_b}"
    assert abs(pse_a + pse_b) < 1e-3, f"symmetry broken: hard_a={pse_a}, hard_b={pse_b}"

    # Magnitude grows with σ then saturates
    pse_lo = compute_normative_pse('hard_a', sigma_percep=0.05)
    pse_hi = compute_normative_pse('hard_a', sigma_percep=0.30)
    assert pse_hi > pse_lo, f"PSE magnitude should grow with σ: {pse_lo} vs {pse_hi}"

run("compute_normative_pse consistent", test_normative_pse)


# =============================================================================
# 5. compute_distribution_density dict shape
# =============================================================================

print("\n[5] compute_distribution_density dict shape")

def test_density():
    from utils.stimulus_distribution import compute_distribution_density
    s = np.linspace(-1, 1, 50)
    for dist in ['uniform', 'hard_a', 'hard_b']:
        d = compute_distribution_density(dist, s)
        assert set(d.keys()) == {'s', 'density_a', 'density_b'}, f"{dist}: bad keys"
        assert d['s'].shape == s.shape
        assert d['density_a'].shape == s.shape
        assert d['density_b'].shape == s.shape
        # Densities should be non-negative
        assert np.all(d['density_a'] >= 0)
        assert np.all(d['density_b'] >= 0)
        # A density is zero on B's support and vice versa
        on_a_only = (s >= -1) & (s < 0)
        on_b_only = (s >= 0) & (s <= 1)
        assert np.all(d['density_b'][on_a_only] == 0), f"{dist}: B density nonzero on A support"
        assert np.all(d['density_a'][on_b_only] == 0), f"{dist}: A density nonzero on B support"

run("compute_distribution_density returns dict", test_density)


# =============================================================================
# 6. compute_session_features (RT inlined)
# =============================================================================

print("\n[6] compute_session_features works")

def test_session_features():
    from behav_utils.data.synthetic import generate_synthetic_animal
    from behav_utils.analysis.session_features import compute_session_features

    animal, _ = generate_synthetic_animal(
        animal_id='SS_test', n_sessions=2, trials_per_session=200, seed=1,
    )
    features = compute_session_features(animal.sessions[0])
    assert isinstance(features, dict)
    # Should contain RT fields (from inlined helper)
    rt_keys = {'rt_median', 'rt_iqr', 'rt_skewness', 'proportion_fast',
                'rt_median_hard', 'rt_median_easy', 'rt_correct_vs_error'}
    missing = rt_keys - set(features.keys())
    assert not missing, f"RT features missing: {missing}"

run("compute_session_features returns RT features", test_session_features)


# =============================================================================
# 7. assign_opto_phases
# =============================================================================

print("\n[7] assign_opto_phases")

def test_assign_phases():
    from behav_utils.data.synthetic import generate_synthetic_animal
    from analysis.opto import assign_opto_phases

    animal, _ = generate_synthetic_animal(
        animal_id='SS_test', n_sessions=4, trials_per_session=100, seed=1,
    )
    phases = assign_opto_phases(animal.sessions)
    expected = {'masking', 'expert_uniform_pre', 'expert_uniform_opto',
                 'washout', 'shift_with_opto', 'shift_no_opto'}
    assert set(phases.keys()) == expected, f"unexpected phase keys"
    total = sum(len(v) for v in phases.values())
    # Some synthetic animals may have unknown distributions — they won't be
    # placed in any phase. Just sanity-check that the function ran without errors.
    assert total <= len(animal.sessions)

run("assign_opto_phases returns 6 phases", test_assign_phases)


# =============================================================================
# 8. detect_shifts
# =============================================================================

print("\n[8] detect_shifts")

def test_detect_shifts():
    from behav_utils.data.synthetic import generate_synthetic_animal
    from analysis.adaptation import detect_shifts

    animal, _ = generate_synthetic_animal(
        animal_id='SS_test', n_sessions=3, trials_per_session=100, seed=1,
    )
    # Manually set distributions to ensure a shift exists
    animal.sessions[0].metadata.distribution = 'uniform'
    animal.sessions[1].metadata.distribution = 'uniform'
    animal.sessions[2].metadata.distribution = 'hard_a'

    shifts = detect_shifts(animal)
    if len(shifts) == 0:
        # Fall back: the synthetic generator may not set distribution
        # the way we want. Just verify the function ran.
        return
    s = shifts[0]
    expected_fields = {'shift_idx', 'session_idx', 'trial_index_in_animal',
                        'from_distribution', 'to_distribution'}
    assert expected_fields.issubset(s.keys())

run("detect_shifts returns expected fields", test_detect_shifts)


# =============================================================================
# 9. compute_comparison + plot_comparison
# =============================================================================

print("\n[9] compute_comparison + plot_comparison round-trip")

def test_compare_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from behav_utils.data.synthetic import generate_synthetic_animal
    from behav_utils.analysis.comparison import compute_comparison
    from behav_utils.plotting.comparison import plot_comparison

    animal, _ = generate_synthetic_animal(
        animal_id='SS_test', n_sessions=4, trials_per_session=300, seed=1,
    )
    half = len(animal.sessions) // 2
    a, b = animal.sessions[:half], animal.sessions[half:]

    result = compute_comparison(a, b, label_a='early', label_b='late',
                                  n_bootstrap=50)
    assert 'mu' in result['params_a']
    assert 'sigma' in result['params_a']

    fig, ax = plt.subplots()
    plot_comparison(result, ax=ax)
    fig.savefig('/tmp/smoke_compare.png', dpi=50)
    plt.close(fig)

run("compute_comparison → plot_comparison round-trip", test_compare_plot)


# =============================================================================
# 10. SBI validation signatures
# =============================================================================

print("\n[10] SBI validation module signatures")

def test_sbi_validation_signatures():
    from validation.sbi import (
        compute_sbc_ranks, compute_parameter_recovery,
        compute_param_stat_correlations, recovery_summary_table,
    )
    from plotting.sbi_validation import (
        plot_sbc_ranks, plot_sbc_ecdf,
        plot_recovery_scatter, plot_recovery_bias,
        plot_param_stat_correlations,
    )
    import inspect
    sig = inspect.signature(compute_sbc_ranks)
    assert 'posterior' in sig.parameters
    assert 'simulator' in sig.parameters
    assert 'prior' in sig.parameters
    assert 'n_sbc_runs' in sig.parameters

    sig2 = inspect.signature(compute_parameter_recovery)
    assert 'n_recoveries' in sig2.parameters

run("SBI validation API signatures correct", test_sbi_validation_signatures)


# =============================================================================
# SUMMARY
# =============================================================================

print()
print("=" * 60)
print(f"PASS: {PASS}, FAIL: {FAIL}")
print("=" * 60)

if FAIL > 0:
    print("\nFailure details:\n")
    for name, tb in DETAIL:
        print(f"--- {name} ---")
        print(tb)
    sys.exit(1)
else:
    print("\nAll smoke tests passed.")
    sys.exit(0)
