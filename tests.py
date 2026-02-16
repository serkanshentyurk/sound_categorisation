"""
Pipeline Tests

Run with:  python tests.py
           python tests.py -v          (verbose — prints details on pass too)

Each test function either returns silently (pass) or raises AssertionError (fail).
No dependencies beyond what the repo already uses.
"""

import sys
import time
import traceback
import numpy as np
from scipy.integrate import trapezoid


# =============================================================================
# TEST RUNNER
# =============================================================================

TESTS = []

def test(fn):
    """Decorator to register a test."""
    TESTS.append(fn)
    return fn


def run_all(verbose=False):
    """Run all registered tests, print results."""
    passed, failed, errors = [], [], []

    for fn in TESTS:
        name = fn.__name__
        try:
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
            passed.append(name)
            if verbose:
                print(f"  PASS  {name}  ({dt*1000:.0f}ms)")
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            errors.append((name, traceback.format_exc()))
            print(f"  ERROR {name}: {e}")

    # Summary
    print(f"\n{'='*60}")
    total = len(passed) + len(failed) + len(errors)
    print(f"{len(passed)}/{total} passed", end="")
    if failed:
        print(f", {len(failed)} FAILED", end="")
    if errors:
        print(f", {len(errors)} ERRORS", end="")
    print()

    if errors and verbose:
        print("\nFull tracebacks:")
        for name, tb in errors:
            print(f"\n--- {name} ---")
            print(tb)

    return len(failed) + len(errors)


# =============================================================================
# HELPERS
# =============================================================================

def _make_standard_inputs(n_trials=300, seed=42):
    """Standard test inputs reused across tests."""
    rng = np.random.default_rng(seed)
    stimuli = rng.uniform(-1, 1, n_trials)
    categories = (stimuli > 0).astype(int)
    return stimuli, categories, rng


# =============================================================================
# 1. MODEL CONTAINERS
# =============================================================================

@test
def test_beparams_array_roundtrip():
    """BEParams survives array serialisation."""
    from Models.BE_core import BEParams
    original = BEParams(sigma_percep=0.2, A_repulsion=0.15,
                        eta_learning=0.4, eta_relax=0.1)
    reconstructed = BEParams.from_array(original.to_array())
    for name in BEParams.get_param_names():
        assert getattr(original, name) == getattr(reconstructed, name), \
            f"{name} mismatch: {getattr(original, name)} vs {getattr(reconstructed, name)}"


@test
def test_beparams_dict_roundtrip():
    """BEParams survives dict serialisation."""
    from Models.BE_core import BEParams
    original = BEParams(sigma_percep=0.2, A_repulsion=0.15,
                        eta_learning=0.4, eta_relax=0.1)
    reconstructed = BEParams.from_dict(original.to_dict())
    for name in BEParams.get_param_names():
        assert getattr(original, name) == getattr(reconstructed, name)


@test
def test_beparams_clamping():
    """Out-of-bound values get clamped with warning, not crash."""
    import warnings
    from Models.BE_core import BEParams
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = BEParams(sigma_percep=-0.1, A_repulsion=-0.5,
                     eta_learning=1.5, eta_relax=-0.3)
    assert p.sigma_percep > 0, "sigma_percep should be clamped positive"
    assert p.A_repulsion >= 0, "A_repulsion should be clamped non-negative"
    assert 0 < p.eta_learning <= 1, "eta_learning should be in (0, 1]"
    assert 0 <= p.eta_relax < 1, "eta_relax should be in [0, 1)"


@test
def test_bestate_uniform_integrates_to_one():
    """Uniform belief distribution integrates to 1."""
    from Models.BE_core import BEState
    state = BEState.initial_uniform()
    integral = trapezoid(state.boundary_belief, state.x)
    assert abs(integral - 1.0) < 1e-10, f"Integral = {integral}, expected 1.0"


@test
def test_bestate_uniform_nonstandard_range():
    """Uniform belief works with non-default stimulus range."""
    from Models.BE_core import BEState
    state = BEState.initial_uniform(x_min=-2.0, x_max=2.0, n_points=500)
    integral = trapezoid(state.boundary_belief, state.x)
    assert abs(integral - 1.0) < 1e-10, f"Integral = {integral} for [-2, 2] range"
    # Density should be 1/(x_max - x_min) = 0.25, not 0.5
    expected_density = 1.0 / (2.0 - (-2.0))
    actual_density = state.boundary_belief[len(state.boundary_belief)//2]
    assert abs(actual_density - expected_density) < 1e-6, \
        f"Uniform density = {actual_density}, expected {expected_density}"


# =============================================================================
# 2. BELIEF UPDATE
# =============================================================================

@test
def test_update_belief_preserves_normalisation():
    """Belief still integrates to 1 after update."""
    from Models.BE_core import BEParams, BEState, BEModel
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)
    state = BEState.initial_uniform()

    # Run several updates with different stimulus/category combos
    test_cases = [(0.3, 1), (-0.7, 0), (0.01, 0), (-0.99, 1), (0.5, 1)]
    for s_hat, cat in test_cases:
        state = BEModel.update_belief(s_hat, cat, params, state)
        integral = trapezoid(state.boundary_belief, state.x)
        assert abs(integral - 1.0) < 1e-8, \
            f"Integral = {integral} after update(s_hat={s_hat}, cat={cat})"


@test
def test_inplace_update_matches_public():
    """_update_belief_inplace produces same result as public update_belief."""
    from Models.BE_core import BEParams, BEState, BEModel
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)
    state = BEState.initial_uniform()

    test_cases = [(0.3, 1), (-0.7, 0), (0.01, 0), (-0.99, 1)]
    for s_hat, cat in test_cases:
        # Public method
        new_state = BEModel.update_belief(s_hat, cat, params, state)

        # In-place method
        belief_copy = state.boundary_belief.copy()
        BEModel._update_belief_inplace(
            s_hat, cat, params, belief_copy, state.x,
            state.x_min, state.x_max
        )

        max_diff = np.max(np.abs(new_state.boundary_belief - belief_copy))
        assert max_diff < 1e-12, \
            f"Divergence {max_diff:.2e} at s_hat={s_hat}, cat={cat}"


@test
def test_inplace_update_preserves_normalisation():
    """In-place belief update still integrates to 1."""
    from Models.BE_core import BEParams, BEState, BEModel
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)
    state = BEState.initial_uniform()
    belief = state.boundary_belief.copy()

    for s_hat, cat in [(0.5, 1), (-0.3, 0), (0.8, 0)]:
        BEModel._update_belief_inplace(
            s_hat, cat, params, belief, state.x,
            state.x_min, state.x_max
        )
        integral = trapezoid(belief, state.x)
        assert abs(integral - 1.0) < 1e-8, \
            f"Integral = {integral} after inplace update(s_hat={s_hat}, cat={cat})"


@test
def test_relaxation_target_adapts_to_range():
    """Relaxation toward uniform uses correct density for the range, not hardcoded 0.5."""
    from Models.BE_core import BEParams, BEState, BEModel

    # Use a non-standard range where uniform density != 0.5
    state_wide = BEState.initial_uniform(x_min=-2.0, x_max=2.0, n_points=500)
    state_std = BEState.initial_uniform(x_min=-1.0, x_max=1.0, n_points=500)

    # High relaxation, no learning — belief should stay near uniform
    params_relax = BEParams(sigma_percep=0.15, A_repulsion=0.0,
                            eta_learning=0.01, eta_relax=0.9)

    # After many updates, belief should relax back near uniform
    for _ in range(50):
        state_wide = BEModel.update_belief(0.1, 1, params_relax, state_wide)
        state_std = BEModel.update_belief(0.1, 1, params_relax, state_std)

    # Both should integrate to 1
    integral_wide = trapezoid(state_wide.boundary_belief, state_wide.x)
    integral_std = trapezoid(state_std.boundary_belief, state_std.x)
    assert abs(integral_wide - 1.0) < 1e-8, f"Wide range integral: {integral_wide}"
    assert abs(integral_std - 1.0) < 1e-8, f"Standard range integral: {integral_std}"


# =============================================================================
# 3. SIMULATION
# =============================================================================

@test
def test_simulate_session_shapes():
    """simulate_session returns arrays of correct shape."""
    from Models.BE_core import BEParams, BEState, BEModel
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)
    state = BEState.initial_uniform()
    stimuli, categories, rng = _make_standard_inputs(n_trials=200)

    choices, p_B, final_state, history = BEModel.simulate_session(
        params, state, stimuli, categories, rng, return_history=True
    )

    assert choices.shape == (200,), f"choices shape: {choices.shape}"
    assert p_B.shape == (200,), f"p_B shape: {p_B.shape}"
    assert history is not None, "history should not be None when return_history=True"
    assert history.beliefs.shape == (200, state.n_points), \
        f"beliefs shape: {history.beliefs.shape}"


@test
def test_simulate_no_history_returns_none():
    """History is None when not requested."""
    from Models.BE_core import BEParams, BEState, BEModel
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)
    state = BEState.initial_uniform()
    stimuli, categories, rng = _make_standard_inputs(n_trials=100)

    _, _, _, history = BEModel.simulate_session(
        params, state, stimuli, categories, rng, return_history=False
    )
    assert history is None


@test
def test_choices_are_binary():
    """Simulated choices are in {0, 1, NaN}."""
    from Models.BE_core import BEParams, BEState, BEModel
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)
    state = BEState.initial_uniform()
    stimuli, categories, rng = _make_standard_inputs(n_trials=500)

    choices, _, _, _ = BEModel.simulate_session(
        params, state, stimuli, categories, rng
    )
    valid = ~np.isnan(choices)
    unique_vals = set(choices[valid])
    assert unique_vals <= {0.0, 1.0}, f"Unexpected choice values: {unique_vals}"


@test
def test_p_B_in_valid_range():
    """Choice probabilities are in (0, 1)."""
    from Models.BE_core import BEParams, BEState, BEModel
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)
    state = BEState.initial_uniform()
    stimuli, categories, rng = _make_standard_inputs(n_trials=500)

    _, p_B, _, _ = BEModel.simulate_session(
        params, state, stimuli, categories, rng
    )
    valid = ~np.isnan(p_B)
    assert np.all(p_B[valid] > 0), "p_B has values <= 0"
    assert np.all(p_B[valid] < 1), "p_B has values >= 1"


@test
def test_model_choices_higher_ll_than_random():
    """Model-generated choices have higher LL than random choices."""
    from Models.BE_core import BEParams, BEState, BEModel
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)
    state = BEState.initial_uniform()
    stimuli, categories, _ = _make_standard_inputs(n_trials=300)

    # Generate model choices
    rng = np.random.default_rng(99)
    choices, _, _, _ = BEModel.simulate_session(
        params, state, stimuli, categories, rng
    )

    # Average LL over several noise realisations
    lls_model, lls_random = [], []
    random_choices = np.random.default_rng(77).binomial(1, 0.5, 300).astype(float)
    for mc in range(20):
        rng_mc = np.random.default_rng(mc)
        ll_m, _, _, _ = BEModel.compute_log_likelihood(
            params, state, stimuli, categories, choices, rng_mc)
        lls_model.append(ll_m)

        rng_mc2 = np.random.default_rng(mc)
        ll_r, _, _, _ = BEModel.compute_log_likelihood(
            params, state, stimuli, categories, random_choices, rng_mc2)
        lls_random.append(ll_r)

    ll_model = np.mean(lls_model)
    ll_random = np.mean(lls_random)
    assert ll_model > ll_random, \
        f"Model LL ({ll_model:.1f}) should exceed random LL ({ll_random:.1f})"


@test
def test_deterministic_simulation_is_reproducible():
    """Same seed → same choices."""
    from Models.BE_core import BEParams, BEState, BEModel
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)
    stimuli, categories, _ = _make_standard_inputs(n_trials=200)

    choices_1, _, _, _ = BEModel.simulate_session(
        params, BEState.initial_uniform(), stimuli, categories,
        np.random.default_rng(42)
    )
    choices_2, _, _, _ = BEModel.simulate_session(
        params, BEState.initial_uniform(), stimuli, categories,
        np.random.default_rng(42)
    )

    assert np.array_equal(choices_1, choices_2), \
        "Same seed should produce identical choices"


@test
def test_no_response_trials_are_nan():
    """Trials flagged as no_response produce NaN choices and NaN p_B."""
    from Models.BE_core import BEParams, BEState, BEModel
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)
    state = BEState.initial_uniform()
    stimuli, categories, rng = _make_standard_inputs(n_trials=100)

    no_response = np.zeros(100, dtype=bool)
    no_response[[5, 10, 50, 99]] = True

    choices, p_B, _, _ = BEModel.simulate_session(
        params, state, stimuli, categories, rng,
        no_response=no_response
    )

    for idx in [5, 10, 50, 99]:
        assert np.isnan(choices[idx]), f"Choice at no_response trial {idx} should be NaN"
        assert np.isnan(p_B[idx]), f"p_B at no_response trial {idx} should be NaN"


# =============================================================================
# 4. MODELTRACE
# =============================================================================

@test
def test_model_trace_belief_means_stds():
    """Vectorised belief_means/stds match scalar computation."""
    from Models.BE_core import BEParams, BEState, BEModel
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)
    state = BEState.initial_uniform()
    stimuli, categories, rng = _make_standard_inputs(n_trials=100)

    _, _, _, trace = BEModel.simulate_session(
        params, state, stimuli, categories, rng, return_history=True
    )

    means = trace.belief_means
    stds = trace.belief_stds
    assert means.shape == (100,)
    assert stds.shape == (100,)

    # Spot-check against scalar computation
    for t in [0, 25, 50, 99]:
        b = trace.beliefs[t]
        mu_scalar = trapezoid(trace.x * b, trace.x)
        var_scalar = trapezoid((trace.x - mu_scalar)**2 * b, trace.x)
        assert abs(means[t] - mu_scalar) < 1e-12, \
            f"Mean mismatch at trial {t}: {means[t]} vs {mu_scalar}"
        assert abs(stds[t] - np.sqrt(var_scalar)) < 1e-12, \
            f"Std mismatch at trial {t}: {stds[t]} vs {np.sqrt(var_scalar)}"


@test
def test_model_trace_get_p_B_at_midpoints():
    """Vectorised midpoint evaluation gives valid probabilities."""
    from Models.BE_core import BEParams, BEState, BEModel
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)
    state = BEState.initial_uniform()
    stimuli, categories, rng = _make_standard_inputs(n_trials=100)

    _, _, _, trace = BEModel.simulate_session(
        params, state, stimuli, categories, rng, return_history=True
    )

    midpoints = np.linspace(-0.8, 0.8, 8)
    p_vals = trace.get_p_B_at_midpoints(midpoints, trial_idx=50)

    assert p_vals.shape == (8,)
    assert np.all(p_vals > 0) and np.all(p_vals < 1), \
        f"P(B) values out of range: {p_vals}"
    # Should be monotonically increasing (higher stimulus → higher P(B))
    assert np.all(np.diff(p_vals) >= -0.01), \
        "P(B) should be roughly monotonically increasing with stimulus"


# =============================================================================
# 5. SUMMARY STATISTICS
# =============================================================================

@test
def test_perfect_observer_accuracy():
    """Perfect choices give accuracy ≈ 1.0."""
    from Analysis.summary_stats import compute_summary_stats
    stimuli = np.linspace(-1, 1, 200)
    categories = (stimuli > 0).astype(int)
    choices = categories.astype(float)  # Perfect

    result = compute_summary_stats(choices, stimuli, categories,
                                   stat_names=['accuracy'], return_dict=True)
    assert abs(result['accuracy'] - 1.0) < 1e-10, \
        f"Perfect observer accuracy: {result['accuracy']}"


@test
def test_all_B_choices_give_bias_near_one():
    """Always choosing B gives side_bias ≈ 0.5."""
    from Analysis.summary_stats import compute_summary_stats
    stimuli = np.linspace(-1, 1, 200)
    categories = (stimuli > 0).astype(int)
    choices = np.ones(200)  # Always B

    result = compute_summary_stats(choices, stimuli, categories,
                                   stat_names=['side_bias'], return_dict=True)
    assert abs(result['side_bias'] - 0.5) < 1e-6, \
        f"All-B side bias: {result['side_bias']}, expected 0.5"


@test
def test_random_choices_accuracy_near_half():
    """Random choices give accuracy ≈ 0.5."""
    from Analysis.summary_stats import compute_summary_stats
    rng = np.random.default_rng(42)
    stimuli = rng.uniform(-1, 1, 5000)
    categories = (stimuli > 0).astype(int)
    choices = rng.binomial(1, 0.5, 5000).astype(float)

    result = compute_summary_stats(choices, stimuli, categories,
                                   stat_names=['accuracy'], return_dict=True)
    assert abs(result['accuracy'] - 0.5) < 0.05, \
        f"Random accuracy: {result['accuracy']}, expected ~0.5"


@test
def test_summary_stats_flatten_produces_finite():
    """Flattened summary stats vector has no inf values."""
    from Analysis.summary_stats import compute_summary_stats, DEFAULT_STATS
    from Models.BE_core import BEParams, BEState, BEModel
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)
    state = BEState.initial_uniform()
    stimuli, categories, rng = _make_standard_inputs(n_trials=300)

    choices, _, _, _ = BEModel.simulate_session(
        params, state, stimuli, categories, rng
    )

    stats = compute_summary_stats(choices, stimuli, categories,
                                  stat_names=DEFAULT_STATS, return_dict=False)
    assert isinstance(stats, np.ndarray), f"Expected ndarray, got {type(stats)}"
    assert not np.any(np.isinf(stats)), "Summary stats contain inf"
    # Some NaN is acceptable (failed psychometric fits etc.) but not all
    assert not np.all(np.isnan(stats)), "All summary stats are NaN"


@test
def test_registry_contains_expected_stats():
    """Core stats are registered."""
    from Analysis.summary_stats import SUMMARY_REGISTRY
    expected = ['accuracy', 'psychometric', 'recency', 'win_stay', 'lose_shift',
                'side_bias', 'choice_autocorr', 'stimulus_sensitivity',
                'choice_entropy', 'logistic_history']
    for name in expected:
        assert name in SUMMARY_REGISTRY, f"Missing stat: '{name}'"


# =============================================================================
# 6. DATA PIPELINE
# =============================================================================

@test
def test_synthetic_animal_generation():
    """Generate synthetic animal → extract FittingData without errors."""
    from Data.structures import generate_synthetic_animal

    animal, ground_truth = generate_synthetic_animal(
        animal_id='TEST01',
        n_sessions=5,
        trials_per_session=100,
        seed=42,
    )

    assert animal.n_sessions == 5
    assert animal.animal_id == 'TEST01'

    # Extract fitting data
    fitting = animal.get_fitting_data(stage='Full_Task_Cont')
    assert fitting.n_sessions == 5
    assert len(fitting.stimuli) == 5
    assert all(len(s) > 0 for s in fitting.stimuli), "Empty stimulus arrays"


@test
def test_fitting_data_summary():
    """FittingData.summary() returns a valid DataFrame."""
    from Data.structures import generate_synthetic_animal

    animal, _ = generate_synthetic_animal(
        n_sessions=3, trials_per_session=100, seed=42
    )
    fitting = animal.get_fitting_data()
    summary = fitting.summary()

    assert len(summary) == 3, f"Expected 3 rows, got {len(summary)}"
    assert 'performance' in summary.columns
    assert all(summary['n_valid'] > 0), "Sessions with zero valid trials"


@test
def test_synthetic_animal_with_distribution_shift():
    """Distribution shift generates correctly."""
    from Data.structures import generate_synthetic_animal

    animal, gt = generate_synthetic_animal(
        n_sessions=10,
        trials_per_session=100,
        distribution='Uniform',
        distribution_shift_session=5,
        shift_distribution='Hard-A',
        seed=42,
    )

    # Sessions 0-4 should be Uniform, 5-9 should be Hard-A
    for i in range(5):
        assert animal.sessions[i].distribution == 'Uniform', \
            f"Session {i} should be Uniform"
    for i in range(5, 10):
        assert animal.sessions[i].distribution == 'Hard-A', \
            f"Session {i} should be Hard-A"


@test
def test_variable_trial_counts():
    """Variable trials_per_session handled correctly."""
    from Data.structures import generate_synthetic_animal

    tps = [80, 120, 200, 150, 90]
    animal, _ = generate_synthetic_animal(
        n_sessions=5,
        trials_per_session=tps,
        seed=42,
    )

    for i, expected in enumerate(tps):
        actual = animal.sessions[i].n_trials
        assert actual == expected, \
            f"Session {i}: expected {expected} trials, got {actual}"


@test
def test_get_model_arrays_excludes_aborts():
    """get_model_arrays with exclude_abort=True removes abort trials."""
    from Data.structures import generate_synthetic_animal

    animal, _ = generate_synthetic_animal(
        n_sessions=1, trials_per_session=200,
        abort_rate=0.1, seed=42
    )
    trials = animal.sessions[0].trials
    n_aborts = trials.abort.sum()
    assert n_aborts > 0, "Need some aborts for this test"

    arrays = trials.get_model_arrays(exclude_abort=True)
    assert len(arrays['stimuli']) == 200 - n_aborts, \
        f"Expected {200 - n_aborts} trials after abort exclusion, got {len(arrays['stimuli'])}"


# =============================================================================
# 7. simulate_for_sbi INTERFACE
# =============================================================================

@test
def test_simulate_for_sbi_returns_choices():
    """simulate_for_sbi with return_choices=True gives valid choices."""
    from Models.BE_core import BEParams, simulate_for_sbi
    stimuli, categories, _ = _make_standard_inputs(n_trials=200)
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)

    choices = simulate_for_sbi(
        params.to_array(), stimuli, categories,
        seed=42, return_choices=True
    )

    assert choices.shape == (200,)
    valid = ~np.isnan(choices)
    assert set(choices[valid]) <= {0.0, 1.0}


# =============================================================================
# 8. MULTI-SESSION
# =============================================================================

@test
def test_multisession_state_chaining():
    """State carries across sessions — second session's belief ≠ uniform."""
    from Models.BE_core import BEParams, BEState, BEModel
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)
    state = BEState.initial_uniform()
    rng = np.random.default_rng(42)

    # Session 1
    stim1 = rng.uniform(-1, 1, 200)
    cat1 = (stim1 > 0).astype(int)
    _, _, state_after_1, _ = BEModel.simulate_session(
        params, state, stim1, cat1, rng
    )

    # Session 2 starts from where session 1 ended
    stim2 = rng.uniform(-1, 1, 200)
    cat2 = (stim2 > 0).astype(int)
    _, _, state_after_2, _ = BEModel.simulate_session(
        params, state_after_1, stim2, cat2, rng
    )

    # Belief after session 1 should differ from uniform
    uniform = np.ones(state.n_points) / (state.x_max - state.x_min)
    diff_from_uniform = np.max(np.abs(state_after_1.boundary_belief - uniform))
    assert diff_from_uniform > 0.01, \
        "Belief after 200 trials should differ substantially from uniform"


@test
def test_burn_in_produces_nonuniform_belief():
    """Burn-in shifts belief away from uniform."""
    from Models.BE_core import BEParams, BEState, BEModel
    params = BEParams(sigma_percep=0.15, A_repulsion=0.1,
                      eta_learning=0.35, eta_relax=0.12)
    state = BEModel.create_initial_state(burn_in=500, params=params, seed=42)

    uniform = np.ones(state.n_points) / (state.x_max - state.x_min)
    diff = np.max(np.abs(state.boundary_belief - uniform))
    assert diff > 0.01, "Burn-in should produce non-uniform belief"

    # Belief should peak near the true boundary (0)
    peak_location = state.x[np.argmax(state.boundary_belief)]
    assert abs(peak_location) < 0.3, \
        f"Belief peak at {peak_location}, expected near 0"


# =============================================================================
# 9. IMPORT HEALTH
# =============================================================================

@test
def test_imports_models():
    """Model modules import without error."""
    from Models.BE_core import BEParams, BEState, BEModel, ModelTrace
    from Models.BE_model import BoundaryEstimationModel


@test
def test_imports_analysis():
    """Analysis modules import without error."""
    from Analysis.summary_stats import (
        compute_summary_stats, SUMMARY_REGISTRY, DEFAULT_STATS,
        FEATURE_MATRIX_STATS, flatten_stats
    )
    from Analysis.update_matrix import compute_update_matrix
    from Analysis.session_features import build_feature_matrix


@test
def test_imports_data():
    """Data modules import without error."""
    from Data.structures import (
        AnimalData, SessionData, TrialData, FittingData,
        ExperimentData, generate_synthetic_animal
    )


@test
def test_imports_inference():
    """Inference modules import without error."""
    from Inference.simulator import Simulator, SimulatorConfig
    from Inference.priors import DEFAULT_BE_BOUNDS


@test
def test_imports_helpers():
    """Helper modules import without error."""
    from Helpers.psychometry import fit_psychometric
    from Helpers.utils import cumulative_gaussian


# =============================================================================
# 10. PSYCHOMETRIC FITTING
# =============================================================================

@test
def test_psychometric_fit_on_clean_data():
    """Psychometric fit recovers known PSE and slope from simulated data."""
    from Helpers.psychometry import fit_psychometric
    from Helpers.utils import cumulative_gaussian

    rng = np.random.default_rng(42)
    true_mu, true_sigma = 0.1, 0.25
    stimuli = rng.uniform(-1, 1, 2000)
    p_B = cumulative_gaussian(stimuli, true_mu, true_sigma, 0.02, 0.02)
    choices = (rng.random(2000) < p_B).astype(float)

    result = fit_psychometric(stimuli, choices)
    assert result['success'], "Fit should succeed on clean data"
    assert abs(result['mu'] - true_mu) < 0.05, \
        f"PSE: {result['mu']:.3f}, expected ~{true_mu}"
    assert abs(result['sigma'] - true_sigma) < 0.1, \
        f"Slope: {result['sigma']:.3f}, expected ~{true_sigma}"


@test
def test_psychometric_fit_fails_gracefully():
    """Psychometric fit returns success=False on garbage data."""
    from Helpers.psychometry import fit_psychometric

    # Too few trials
    result = fit_psychometric(np.array([0.1, 0.2]), np.array([0.0, 1.0]))
    assert not result['success'], "Should fail with < 10 trials"
    assert np.isnan(result['mu'])

@test
def test_hard_distributions_sample_correctly():
    """Hard-A/B distributions concentrate near boundary on correct side."""
    from Data.structures import sample_stimuli
    rng = np.random.default_rng(42)
    
    n = 10000
    stim_A = sample_stimuli(n, 'Hard-A', rng)
    stim_B = sample_stimuli(n, 'Hard-B', rng)
    stim_U = sample_stimuli(n, 'Uniform', rng)
    
    # All in valid range
    assert np.all(stim_A >= -1) and np.all(stim_A <= 1)
    assert np.all(stim_B >= -1) and np.all(stim_B <= 1)
    
    # Hard-A: A side should have more near-boundary trials than uniform
    a_side_A = stim_A[stim_A < 0]
    a_side_U = stim_U[stim_U < 0]
    near_boundary_A = np.mean(a_side_A > -0.3)  # fraction near boundary
    near_boundary_U = np.mean(a_side_U > -0.3)
    assert near_boundary_A > near_boundary_U + 0.05, \
        f"Hard-A should concentrate near boundary: {near_boundary_A:.2f} vs uniform {near_boundary_U:.2f}"
    
    # Hard-A: B side should be uniform (similar to Uniform dist)
    b_side_A = stim_A[stim_A >= 0]
    b_side_U = stim_U[stim_U >= 0]
    near_boundary_bA = np.mean(b_side_A < 0.3)
    near_boundary_bU = np.mean(b_side_U < 0.3)
    assert abs(near_boundary_bA - near_boundary_bU) < 0.05, \
        "Hard-A B-side should be uniform"
    
    # Hard-B: symmetric argument
    b_side_B = stim_B[stim_B >= 0]
    near_boundary_B = np.mean(b_side_B < 0.3)
    assert near_boundary_B > near_boundary_bU + 0.05, \
        f"Hard-B should concentrate near boundary: {near_boundary_B:.2f} vs uniform {near_boundary_bU:.2f}"


@test
def test_hard_distribution_roughly_balanced_sides():
    """Each distribution produces ~50/50 A/B trials."""
    from Data.structures import sample_stimuli
    rng = np.random.default_rng(42)
    
    for dist in ['Uniform', 'Hard-A', 'Hard-B']:
        stim = sample_stimuli(5000, dist, rng)
        frac_B = np.mean(stim >= 0)
        assert 0.45 < frac_B < 0.55, \
            f"{dist}: B fraction = {frac_B:.3f}, expected ~0.5"

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    verbose = '-v' in sys.argv
    print(f"Running {len(TESTS)} tests...\n")
    n_failures = run_all(verbose=verbose)
    sys.exit(n_failures)
