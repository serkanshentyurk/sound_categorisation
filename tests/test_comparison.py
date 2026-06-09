# """
# Tests for behav_utils/analysis/comparison.py.

# Covers the pair-comparison primitives (compare_conditions, compute_comparison)
# and the group-level statistics primitives (compute_per_animal_stats,
# compute_group_comparison).

# The mutable-default bug in compute_per_animal_stats / compute_group_comparison
# (F1 from audit v3) would have been caught here.
# """

# import numpy as np
# import pytest
# from datetime import date, timedelta


# # ============================================================================
# # compare_conditions — the low-level array primitive
# # ============================================================================

# class TestCompareConditions:
#     """compare_conditions takes raw arrays for two conditions."""

#     def test_returns_expected_keys(self, rng):
#         """compare_conditions returns the documented dict keys with mu/sigma naming."""
#         from behav_utils.analysis.comparison import compare_conditions

#         n = 500
#         stim_a = rng.uniform(-1, 1, n)
#         ch_a = (stim_a > 0.05).astype(float)  # boundary slightly off
#         cat_a = (stim_a > 0).astype(float)

#         stim_b = rng.uniform(-1, 1, n)
#         ch_b = (stim_b > -0.05).astype(float)  # boundary other side
#         cat_b = (stim_b > 0).astype(float)

#         result = compare_conditions(stim_a, ch_a, cat_a, stim_b, ch_b, cat_b)

#         # mu/sigma keys (post-rename), not pse/slope
#         for top in ['params_a', 'params_b', 'diffs', 'perm_p', 'boot_ci']:
#             assert top in result, f"missing top-level key: {top}"
#             assert 'mu' in result[top], f"missing 'mu' in {top}"
#             assert 'sigma' in result[top], f"missing 'sigma' in {top}"
#         assert 'um_rmse' in result
#         assert 'boot_band_a' in result
#         assert 'boot_band_b' in result

#     def test_boot_band_shape(self, rng):
#         """Bootstrap band has expected shape."""
#         from behav_utils.analysis.comparison import compare_conditions

#         n = 400
#         stim_a = rng.uniform(-1, 1, n)
#         ch_a = (stim_a > 0).astype(float)
#         cat_a = (stim_a > 0).astype(float)

#         result = compare_conditions(
#             stim_a, ch_a, cat_a, stim_a, ch_a, cat_a,
#             n_bootstrap=50,
#         )

#         band_a = result['boot_band_a']
#         assert 'x' in band_a
#         assert 'lo' in band_a
#         assert 'hi' in band_a
#         assert len(band_a['x']) == len(band_a['lo']) == len(band_a['hi'])
#         # Upper >= lower at every x
#         assert np.all(band_a['hi'] >= band_a['lo'])

#     def test_perm_p_values_in_unit_interval(self, rng):
#         """All permutation p-values should be valid probabilities."""
#         from behav_utils.analysis.comparison import compare_conditions

#         n = 500
#         stim_a = rng.uniform(-1, 1, n)
#         ch_a = (stim_a > 0).astype(float)
#         cat_a = (stim_a > 0).astype(float)

#         stim_b = rng.uniform(-1, 1, n)
#         ch_b = (stim_b > 0.3).astype(float)
#         cat_b = (stim_b > 0).astype(float)

#         result = compare_conditions(stim_a, ch_a, cat_a, stim_b, ch_b, cat_b,
#                                      n_permutations=100)

#         for key, p in result['perm_p'].items():
#             if not np.isnan(p):
#                 assert 0 <= p <= 1, f"p_{key} = {p} out of [0,1]"


# # ============================================================================
# # compute_comparison — the session-level wrapper
# # ============================================================================

# class TestComputeComparison:
#     """compute_comparison takes sessions, pools, calls compare_conditions."""

#     def test_basic_call(self, synthetic_animal):
#         """Returns dict with same shape as compare_conditions."""
#         from behav_utils.analysis.comparison import compute_comparison
#         from behav_utils.data.ops.filtering import filter_session, opto_mask

#         sessions_with_opto = [s for s in synthetic_animal.sessions
#                               if not s.masking and s.trials.opto_on.any()]
#         assert len(sessions_with_opto) >= 2, "fixture must provide opto sessions"

#         on = [filter_session(s, opto_mask(s.trials, 0))
#               for s in sessions_with_opto]
#         off = [filter_session(s, opto_mask(s.trials, 'control'))
#                for s in sessions_with_opto]

#         result = compute_comparison(
#             on, off,
#             label_a='opto_on', label_b='opto_off',
#             n_bootstrap=50, n_permutations=50,
#         )

#         assert 'diffs' in result
#         assert 'mu' in result['diffs']
#         assert 'um_rmse' in result
#         assert result['label_a'] == 'opto_on'
#         assert result['label_b'] == 'opto_off'

#     def test_identical_inputs_give_small_diff(self, synthetic_animal):
#         """Comparing a session to itself should give near-zero diffs."""
#         from behav_utils.analysis.comparison import compute_comparison

#         sess = [s for s in synthetic_animal.sessions if not s.masking][:3]
#         result = compute_comparison(
#             sess, sess,
#             n_bootstrap=20, n_permutations=20,
#         )

#         # mu/sigma diffs should be exactly zero (same data, both psychometric fits)
#         assert abs(result['diffs']['mu']) < 1e-9
#         assert abs(result['diffs']['sigma']) < 1e-9


# # ============================================================================
# # compute_per_animal_stats — the F1-bug-prone function
# # ============================================================================

# class TestComputePerAnimalStats:
#     """compute_per_animal_stats: one row per animal, columns = stats."""

#     def test_default_args_dont_crash(self, rng):
#         """The F1 bug: stat_keys=... (Ellipsis) made this crash. Verify it doesn't."""
#         from behav_utils.data.synthetic import generate_synthetic_animal
#         from behav_utils.analysis.comparison import compute_per_animal_stats

#         animals = [
#             generate_synthetic_animal(animal_id=f'A{i}', n_sessions=2,
#                                        trials_per_session=200, seed=i)[0]
#             for i in range(3)
#         ]

#         df = compute_per_animal_stats(animals)
#         assert len(df) == 3
#         assert 'animal_id' in df.columns
#         assert 'mu' in df.columns
#         assert 'sigma' in df.columns
#         assert 'accuracy' in df.columns

#     def test_accuracy_is_in_unit_interval(self, rng):
#         """Accuracy must be a valid probability."""
#         from behav_utils.data.synthetic import generate_synthetic_animal
#         from behav_utils.analysis.comparison import compute_per_animal_stats

#         animals = [
#             generate_synthetic_animal(animal_id=f'A{i}', n_sessions=2,
#                                        trials_per_session=200, seed=i)[0]
#             for i in range(5)
#         ]

#         df = compute_per_animal_stats(animals)
#         for acc in df['accuracy']:
#             assert 0 <= acc <= 1

#     def test_custom_stat_keys(self, rng):
#         """Custom stat_keys filters the columns correctly."""
#         from behav_utils.data.synthetic import generate_synthetic_animal
#         from behav_utils.analysis.comparison import compute_per_animal_stats

#         animals = [
#             generate_synthetic_animal(animal_id=f'A{i}', n_sessions=2,
#                                        trials_per_session=200, seed=i)[0]
#             for i in range(3)
#         ]

#         df = compute_per_animal_stats(animals, stat_keys=('accuracy',))
#         assert 'accuracy' in df.columns
#         # sigma should NOT be present (filtered out)
#         assert 'sigma' not in df.columns

#     def test_sessions_per_animal_override(self, synthetic_animal):
#         """sessions_per_animal lets caller restrict to a subset."""
#         from behav_utils.analysis.comparison import compute_per_animal_stats

#         # All sessions
#         df_all = compute_per_animal_stats([synthetic_animal])

#         # Only first 3 sessions
#         sessions_subset = {
#             synthetic_animal.animal_id: synthetic_animal.sessions[:3]
#         }
#         df_subset = compute_per_animal_stats(
#             [synthetic_animal], sessions_per_animal=sessions_subset,
#         )

#         assert df_subset['n_sessions'].iloc[0] == 3
#         assert df_all['n_sessions'].iloc[0] > 3
#         assert df_subset['n_trials_total'].iloc[0] < df_all['n_trials_total'].iloc[0]


# # ============================================================================
# # compute_group_comparison — paired and unpaired
# # ============================================================================

# class TestComputeGroupComparison:
#     """compute_group_comparison: cross-animal test on per-animal stats."""

#     @pytest.fixture
#     def two_groups(self, rng):
#         """Two groups of 5 animals each."""
#         from behav_utils.data.synthetic import generate_synthetic_animal
#         from behav_utils.analysis.comparison import compute_per_animal_stats

#         animals_a = [
#             generate_synthetic_animal(animal_id=f'A{i}', n_sessions=3,
#                                        trials_per_session=300, seed=i)[0]
#             for i in range(5)
#         ]
#         animals_b = [
#             generate_synthetic_animal(animal_id=f'B{i}', n_sessions=3,
#                                        trials_per_session=300, seed=i + 100)[0]
#             for i in range(5)
#         ]
#         return (
#             compute_per_animal_stats(animals_a),
#             compute_per_animal_stats(animals_b),
#         )

#     def test_unpaired_runs(self, two_groups):
#         """Unpaired Mann-Whitney comparison returns valid result."""
#         from behav_utils.analysis.comparison import compute_group_comparison

#         df_a, df_b = two_groups
#         result = compute_group_comparison(df_a, df_b, paired=False)

#         assert result['paired'] is False
#         assert result['n_a'] == 5
#         assert result['n_b'] == 5
#         for key in ('mu', 'sigma', 'accuracy'):
#             assert key in result['p_values']
#             p = result['p_values'][key]
#             assert np.isnan(p) or (0 <= p <= 1)

#     def test_paired_runs(self, two_groups):
#         """Paired Wilcoxon comparison returns valid result."""
#         from behav_utils.analysis.comparison import compute_group_comparison

#         df_a, df_b = two_groups
#         result = compute_group_comparison(df_a, df_b, paired=True)

#         assert result['paired'] is True
#         for key in ('mu', 'sigma', 'accuracy'):
#             assert key in result['p_values']

#     def test_paired_requires_matching_lengths(self, two_groups):
#         """Paired comparison should produce NaN p-values when lengths mismatch."""
#         from behav_utils.analysis.comparison import compute_group_comparison

#         df_a, df_b = two_groups
#         # Drop one row from df_b to mismatch
#         df_b_short = df_b.iloc[:-1]
#         result = compute_group_comparison(df_a, df_b_short, paired=True)

#         for key, p in result['p_values'].items():
#             assert np.isnan(p), f"paired comparison with mismatched lengths should give NaN for {key}, got {p}"

#     def test_labels_propagated(self, two_groups):
#         """Labels show up in the result."""
#         from behav_utils.analysis.comparison import compute_group_comparison

#         df_a, df_b = two_groups
#         result = compute_group_comparison(
#             df_a, df_b, label_a='HET', label_b='WT', paired=False,
#         )
#         assert result['label_a'] == 'HET'
#         assert result['label_b'] == 'WT'

#     def test_diffs_are_medians_diff(self, two_groups):
#         """diffs should equal medians_a - medians_b."""
#         from behav_utils.analysis.comparison import compute_group_comparison

#         df_a, df_b = two_groups
#         result = compute_group_comparison(df_a, df_b, paired=False)

#         for key in result['diffs']:
#             expected = result['medians_a'][key] - result['medians_b'][key]
#             assert abs(result['diffs'][key] - expected) < 1e-9
