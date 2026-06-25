"""Tests for analysis.opto.extract_opto_estimates — the new-layer opto points builder."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from behav_utils.data.structures import (
    TrialData, SessionData, SessionMetadata, AnimalData, ExperimentData,
)
from analysis.opto import extract_opto_estimates


# --------------------------------------------------------------------------- #
# Synthetic opto experiment: two animals (het + wt), each with uniform-opto and
# Hard-A-opto sessions (canonical distribution names, so no config is needed).
# --------------------------------------------------------------------------- #
def _trials(n, rng, noise=0.20, opto_frac=0.0):
    stimuli = rng.uniform(-1, 1, n)
    categories = (stimuli > 0).astype(float)
    choices = categories.copy()
    flip = rng.random(n) < noise
    choices[flip] = 1 - choices[flip]
    opto = np.zeros(n, dtype=bool)
    if opto_frac > 0:
        opto[rng.choice(n, size=int(n * opto_frac), replace=False)] = True
    return TrialData(
        trial_number=np.arange(n), stimulus=stimuli, category=categories,
        choice=choices, outcome=(choices == categories).astype(float),
        correct=(choices == categories), abort=np.zeros(n, dtype=bool), opto_on=opto,
    )


def _session(idx, base, trials, dist='Uniform', masking=False, washout=False):
    return SessionData(
        session_id=f'sess_{idx:03d}', session_idx=idx, date=base + timedelta(days=idx),
        metadata=SessionMetadata(fields={'stage': 'Full_Task_Cont', 'distribution': dist}),
        trials=trials, masking=masking, washout=washout,
    )


def _opto_animal(animal_id, genotype, rng):
    # uniform opto (idx 7-11) + Hard-A opto (idx 14-18): both laser + control trials.
    spec = ([(i, 'Uniform', 0.3, False, False) for i in range(7, 12)] +
            [(i, 'Hard-A', 0.3, False, False) for i in range(14, 19)])
    base = date(2026, 3, 1)
    sessions = [_session(i, base, _trials(350, rng, 0.20, of), dist=d, masking=m, washout=w)
                for i, d, of, m, w in spec]
    animal = AnimalData(animal_id=animal_id, sessions=sessions)
    animal.metadata['genotype'] = genotype
    return animal


@pytest.fixture
def opto_experiment():
    rng = np.random.default_rng(0)
    experiment = ExperimentData()
    experiment.add_animal(_opto_animal('OPTO_HET', 'het', rng))
    experiment.add_animal(_opto_animal('OPTO_WT', 'wt', rng))
    return experiment


class TestExtractOptoEstimates:

    def test_returns_long_frame_with_provenance(self, opto_experiment):
        est = extract_opto_estimates(
            opto_experiment, phases=['uniform', 'hard_a'],
            stats=['recency', 'win_stay'], animals=['OPTO_HET', 'OPTO_WT'])
        assert isinstance(est, pd.DataFrame)
        for col in ('animal', 'genotype', 'distribution', 'session_type',
                    'trial_type', 'stat', 'value', 'n_trials'):
            assert col in est.columns
        # provenance auto-filled from filter_phase; trial-type uses filter_phase names
        assert set(est['trial_type']) == {'opto', 'opto_off'}
        assert set(est['distribution']) == {'uniform', 'hard_a'}
        assert set(est['session_type']) == {'opto'}
        assert set(est['stat']) == {'recency', 'win_stay'}
        assert set(est['genotype']) == {'het', 'wt'}
        # 2 animals x 2 phases x 2 conditions x 2 stats
        assert len(est) == 2 * 2 * 2 * 2

    def test_phases_kept_separate_not_pooled(self, opto_experiment):
        est = extract_opto_estimates(
            opto_experiment, phases=['uniform', 'hard_a'],
            stats=['recency'], animals=['OPTO_HET'])
        # one animal, one stat, two phases x two conditions -> distinct phase rows
        assert len(est) == 1 * 2 * 2
        per_phase = est.groupby('distribution').size()
        assert set(per_phase.index) == {'uniform', 'hard_a'}
        assert (per_phase == 2).all()        # two conditions per phase, never merged

    def test_genotype_normalised(self, opto_experiment):
        opto_experiment.get_animal('OPTO_HET').metadata['genotype'] = 'heterozygous'
        est = extract_opto_estimates(
            opto_experiment, phases='uniform', stats=['recency'], animals=['OPTO_HET'])
        assert set(est['genotype']) == {'het'}

    def test_string_phase_normalised(self, opto_experiment):
        est = extract_opto_estimates(
            opto_experiment, phases='uniform', stats=['recency'], animals=['OPTO_HET'])
        assert set(est['distribution']) == {'uniform'}

    def test_post_opto_trial_type_supported(self, opto_experiment):
        est = extract_opto_estimates(
            opto_experiment, phases='uniform', stats=['recency'], animals=['OPTO_HET'],
            trial_types=('opto', 'opto_off', 'post_opto'))
        assert 'post_opto' in set(est['trial_type'])

    def test_psychometric_curve_with_within_animal_ci(self, opto_experiment):
        est = extract_opto_estimates(
            opto_experiment, phases='uniform', stats=['psychometric'],
            animals=['OPTO_HET'], n_boot=50)
        assert {'mu', 'sigma', 'lapse_low', 'lapse_high'} <= set(est['stat'])
        # n_boot populates the within-animal CI; lapses arrive separate (not averaged)
        assert est['ci_lo_within'].notna().any()
        valid = est.dropna(subset=['ci_lo_within', 'ci_hi_within'])
        assert (valid['ci_lo_within'] <= valid['ci_hi_within']).all()

    def test_history_stats_have_no_ci_when_n_boot_zero(self, opto_experiment):
        est = extract_opto_estimates(
            opto_experiment, phases='uniform', stats=['recency'],
            animals=['OPTO_HET'], n_boot=0)
        assert est['ci_lo_within'].isna().all()

    def test_empty_when_no_sessions(self, opto_experiment):
        # animals have no Hard-B sessions -> empty frame, not an error
        est = extract_opto_estimates(
            opto_experiment, phases='hard_b', stats=['recency'], animals=['OPTO_HET'])
        assert isinstance(est, pd.DataFrame)
        assert len(est) == 0

    def test_animals_default_to_opto_cohort(self, opto_experiment):
        est = extract_opto_estimates(
            opto_experiment, phases='uniform', stats=['recency'])
        assert set(est['animal']) == {'OPTO_HET', 'OPTO_WT'}
