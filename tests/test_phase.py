"""Tests for analysis.phase: phase/condition selection and the report assembler.

Fixtures are built from the conftest helpers. Note two deliberate facts these
tests pin down:
  * filter_phase maps 'uniform'/'hard_a'/'hard_b' to the *mapped* distribution
    strings ('Uniform'/'Hard-A'/'Hard-B'); a raw config name like 'Asym_Right'
    is not matched (it must be remapped upstream first).
  * the opto trial-type masks are checked by surviving-trial counts via the
    laser+control == all-valid invariant, because filter_session rebuilds the
    session and does not preserve the opto_on flag on the selected subset.
"""

from datetime import date

import numpy as np
import pytest

from conftest import _make_trial_data, _make_session
from behav_utils.data.structures import AnimalData
from analysis.phase import (
    filter_phase, is_opto_cohort, compute_phase,
    PHASE_ORDER, PANELS, _DIST, MIN_TRIALS,
)


# ── fixtures (plain helpers so they run under the manual harness too) ─────────

def _animal(rng):
    """Uniform regular/masking/opto/washout, plus Hard-A opto and Hard-B regular."""
    specs = [  # (idx, dist, opto_frac, masking, washout, n)
        (0, 'Uniform', 0.0, False, False, 120),
        (1, 'Uniform', 0.0, False, False, 120),
        (2, 'Uniform', 0.0, False, False, 120),
        (3, 'Uniform', 0.3, True, False, 120),    # masking
        (4, 'Uniform', 0.3, False, False, 120),   # opto
        (5, 'Uniform', 0.3, False, False, 120),   # opto
        (6, 'Uniform', 0.0, False, True, 120),    # washout
        (7, 'Hard-A', 0.3, False, False, 120),    # hard_a opto
        (8, 'Hard-A', 0.3, False, False, 120),    # hard_a opto
        (9, 'Hard-B', 0.0, False, False, 120),    # hard_b regular
    ]
    sess = [_make_session(i, date(2026, 1, 1),
                          _make_trial_data(n, rng, opto_frac=of),
                          distribution=d, masking=m, washout=w)
            for i, d, of, m, w, n in specs]
    return AnimalData(animal_id='PHASE', sessions=sess)


def _regular_animal(rng):
    return AnimalData(animal_id='REG', sessions=[
        _make_session(i, date(2026, 1, 1), _make_trial_data(120, rng),
                      distribution='Uniform') for i in range(4)])


def _n_trials(sessions):
    return sum(s.trials.n_trials for s in sessions)


# ── constants / panels ───────────────────────────────────────────────────────

class TestConstantsAndPanels:
    def test_phase_order_and_dist_map(self):
        assert PHASE_ORDER == ['uniform', 'hard_a', 'hard_b']
        assert _DIST == {'uniform': 'Uniform', 'hard_a': 'Hard-A', 'hard_b': 'Hard-B'}

    def test_opto_panels_structure(self):
        for dist in PHASE_ORDER:
            labels = set(PANELS['opto'][dist])
            assert {'masking', 'all_opto', 'opto_off', 'opto_on', 'post_opto'} <= labels
        assert 'baseline' in PANELS['opto']['uniform']    # uniform also has a baseline

    def test_non_opto_panels_structure(self):
        assert 'baseline' in PANELS['non-opto']['uniform']
        assert 'regular' in PANELS['non-opto']['hard_a']
        assert 'regular' in PANELS['non-opto']['hard_b']


# ── is_opto_cohort ───────────────────────────────────────────────────────────

class TestIsOptoCohort:
    def test_true_when_opto_or_masking_present(self, rng):
        assert is_opto_cohort(_animal(rng))

    def test_false_for_regular_only(self, rng):
        assert not is_opto_cohort(_regular_animal(rng))


# ── filter_phase ─────────────────────────────────────────────────────────────

class TestFilterPhase:
    def test_selects_by_distribution_and_session_type(self, rng):
        an = _animal(rng)
        assert len(filter_phase(an, 'uniform', 'regular')) == 3
        assert len(filter_phase(an, 'uniform', 'masking')) == 1
        assert len(filter_phase(an, 'uniform', 'opto')) == 2
        assert len(filter_phase(an, 'uniform', 'washout')) == 1

    def test_distribution_mapping(self, rng):
        an = _animal(rng)
        assert len(filter_phase(an, 'hard_a', 'opto')) == 2
        assert len(filter_phase(an, 'hard_b', 'regular')) == 1
        # a raw, unmapped config name is NOT matched by 'hard_a'
        asym = AnimalData(animal_id='A', sessions=[
            _make_session(0, date(2026, 1, 1),
                          _make_trial_data(120, rng, opto_frac=0.3),
                          distribution='Asym_Right')])
        assert filter_phase(asym, 'hard_a', 'opto') == []

    def test_no_matching_sessions_returns_empty(self, rng):
        reg = _regular_animal(rng)
        assert filter_phase(reg, 'uniform', 'opto') == []      # no opto sessions
        assert filter_phase(reg, 'hard_a', 'regular') == []    # no hard_a sessions

    def test_opto_trial_masks_partition_valid_trials(self, rng):
        an = _animal(rng)
        n_all = _n_trials(filter_phase(an, 'uniform', 'opto', trial_type='all', min_trials=1))
        n_opto = _n_trials(filter_phase(an, 'uniform', 'opto', trial_type='opto', min_trials=1))
        n_off = _n_trials(filter_phase(an, 'uniform', 'opto', trial_type='opto_off', min_trials=1))
        n_post = _n_trials(filter_phase(an, 'uniform', 'opto', trial_type='post_opto', min_trials=1))
        assert n_opto > 0 and n_off > 0
        assert n_opto + n_off == n_all          # laser + control == all valid
        assert n_post <= n_off                  # post-opto trials are a subset of controls

    def test_none_trial_type_equals_all(self, rng):
        an = _animal(rng)
        a = filter_phase(an, 'uniform', 'opto', trial_type=None, min_trials=1)
        b = filter_phase(an, 'uniform', 'opto', trial_type='all', min_trials=1)
        assert _n_trials(a) == _n_trials(b)

    def test_invalid_trial_type_raises(self, rng):
        an = _animal(rng)
        with pytest.raises(ValueError):
            filter_phase(an, 'uniform', 'opto', trial_type='laser')

    def test_min_trials_drops_small_sessions(self, rng):
        # NB: select_sessions has its own ~10-trial floor, so use a session above it
        # and exercise filter_phase's own min_trials gate (default MIN_TRIALS=10).
        one = AnimalData(animal_id='S', sessions=[
            _make_session(0, date(2026, 1, 1), _make_trial_data(30, rng),
                          distribution='Uniform')])
        assert len(filter_phase(one, 'uniform', 'regular')) == 1          # 30 kept by default
        assert filter_phase(one, 'uniform', 'regular', min_trials=50) == []  # raised floor drops it
        assert MIN_TRIALS == 10


# ── compute_phase (report assembler) ─────────────────────────────────────────

class TestComputePhase:
    def test_assembles_panels_into_three_dicts(self, rng):
        an = _animal(rng)
        clean, psyc, um = compute_phase(an, 'uniform', cohort='opto')
        assert set(clean) == set(psyc) == set(um)
        assert {'baseline', 'opto_on', 'opto_off'} <= set(clean)
        assert isinstance(clean['opto_on'], list) and len(clean['opto_on']) >= 1

    def test_cohort_autodetected(self, rng):
        clean_opto, _, _ = compute_phase(_animal(rng), 'uniform')          # -> 'opto'
        assert 'opto_on' in clean_opto
        clean_reg, _, _ = compute_phase(_regular_animal(rng), 'uniform')   # -> 'non-opto'
        assert set(clean_reg) == {'baseline'}

    def test_empty_panel_is_none(self, rng):
        # regular-only animal forced through the opto panels -> opto conditions empty
        clean, psyc, um = compute_phase(_regular_animal(rng), 'uniform', cohort='opto')
        assert clean['opto_on'] is None
        assert psyc['opto_on'] is None
        assert um['opto_on'] is None
