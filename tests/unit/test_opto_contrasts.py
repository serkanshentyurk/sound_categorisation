"""The opto contrasts' reference sets (behaviour/contrasts.py) and the cohort bias rule (adaptation.BiasRule)."""
import json
import warnings

import pandas as pd
import pytest
from behav_utils.analysis import shared_trial_count
from behav_utils.data.ops.filtering import filter_trials
from behav_utils.data.ops.selection import select_sessions
from sound_categorisation.behaviour.adaptation import BiasRule, flag_biased_sessions
from sound_categorisation.behaviour.contrasts import _reference, ppc_contrasts
from sound_categorisation.data.cohort import ensure_presets
from sound_categorisation.data.synthetic import synthetic_experiment
from sound_categorisation.reports.compute import Settings


@pytest.fixture(scope='module')
def sessions():
    ensure_presets()
    animal = next(iter(synthetic_experiment(n_animals=1).animals.values()))
    return (select_sessions(animal, distribution='Uniform', session_type='opto'),
            select_sessions(animal, distribution='Uniform', session_type='masking'))


def _n(phase):
    return sum(s.trials.n_trials for s in phase)


def test_post_opto_reference_is_the_other_laser_off_trials(sessions):
    """Post-opto trials are laser-off trials; the reference is every other laser-off trial, none shared."""
    opto, _ = sessions
    off, post = filter_trials(opto, trial_type='non_opto'), filter_trials(opto, trial_type='post_opto')
    ref = _reference(opto, off, 'post_opto')
    assert _n(post) > 0 and shared_trial_count(post, ref) == 0
    assert _n(post) + _n(ref) == _n(off)
    assert _reference(opto, off, 'opto') is off


def test_post_opto_contrasts_run_with_a_permutation(sessions):
    """behav_utils >= 0.9 refuses to permute overlapping phases; the disjoint reference passes."""
    opto, masking = sessions
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = ppc_contrasts(opto, masking, 'post_opto', ['accuracy'], n_perm=20, n_boot=10)
    assert 'within' in r and 'within_masking' in r and 'dod' in r
    assert r.key == 'post_opto_vs_non_opto'


def test_compensation_reference_is_the_unflagged_masking_trials(sessions):
    opto, masking = sessions
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = ppc_contrasts(opto, masking, 'opto', ['accuracy'], n_perm=0, n_boot=10)
    ph = r['compensation'].phases
    assert ph['masking'].n_trials == _n(filter_trials(masking, trial_type='non_opto'))
    assert ph['masking'].n_trials < _n(filter_trials(masking, trial_type='all'))
    assert ph['laser_off'].n_trials == _n(filter_trials(opto, trial_type='non_opto'))


def _bias_rows():
    """'one_side' leans only on Hard-B: 18 of 30 sessions flagged, all Hard-B. 'sticker' leans the same
    way on both distributions."""
    rows = []
    for k in range(30):
        dist = 'Hard-B' if k < 20 else 'Hard-A'
        rows.append({'animal': 'one_side', 'to_distribution': dist, 'pse_fixed': 0.7 if (dist == 'Hard-B' and k < 18) else 0.05,
                     'lapse_low': 0.05, 'lapse_high': 0.05, 'accuracy': 0.8})
        rows.append({'animal': 'sticker', 'to_distribution': dist, 'pse_fixed': 0.7, 'lapse_low': 0.05,
                     'lapse_high': 0.4, 'accuracy': 0.55})
    return pd.DataFrame(rows)


def test_bias_rule_needs_the_same_lean_on_both_distributions():
    by = flag_biased_sessions(_bias_rows()).groupby('animal')['biased_animal'].first()
    assert by['sticker'] and not by['one_side']


def test_bias_rule_is_changeable():
    rows = _bias_rows()
    lenient = flag_biased_sessions(rows, BiasRule(both_distributions=False)).groupby('animal')['biased_animal'].first()
    assert lenient['one_side']                                   # the rule up to 0.7
    strict_majority = flag_biased_sessions(rows, BiasRule(majority=0.9)).groupby('animal')['biased_animal'].first()
    assert strict_majority['sticker']                            # 30/30 flagged
    assert not flag_biased_sessions(rows, BiasRule(pse=1.0, lapse=0.5, accuracy=0.5))['flagged'].any()


def test_settings_record_the_rule_and_baseline():
    d = Settings().to_dict()
    json.dumps(d)                                                # meta.json must serialise
    assert d['bias_rule']['both_distributions'] is True and d['baseline_last_n'] == 5
