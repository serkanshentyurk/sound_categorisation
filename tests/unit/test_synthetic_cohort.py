"""sc-make-synthetic-cohort: real session shapes, the preset's accuracy rule, validation seeds."""

import pickle

import pandas as pd
import pytest
from sound_categorisation.cli.make_synthetic_cohort import (
    COHORT_SEED,
    make_cohort,
    preset_min_accuracy,
    shapes_from_run,
)
from sound_categorisation.settings import BASE_SEED, SBI_MAX_DRAWS, SBI_REPRESENTATIONS, VALIDATION_SEED


def test_seed_ranges_are_disjoint():
    """Training draw i uses seed BASE_SEED + i, and a network stops at SBI_MAX_DRAWS draws."""
    max_training_seed = BASE_SEED + SBI_MAX_DRAWS
    assert SBI_MAX_DRAWS > 10 * max(r['n_simulations'] for r in SBI_REPRESENTATIONS.values())
    assert VALIDATION_SEED > max_training_seed and COHORT_SEED >= VALIDATION_SEED + 1_000_000


def test_truths_come_from_the_prior_in_use_unless_one_is_named(tmp_path, monkeypatch):
    monkeypatch.setenv('SC_DATA_ROOT', str(tmp_path))
    from sound_categorisation.models.priors import prior_bounds
    seen = {}
    for prior in (None, 'manuscript'):
        path = make_cohort(f'synthetic_uniform_{prior}', n_per_model=6, n_sessions=2, trials=40,
                           burn_in=20, prior=prior)
        with open(path, 'rb') as f:
            cohort = pickle.load(f)
        name = prior or 'learning'
        assert cohort['generation']['prior'] == name
        for animal in cohort['animals']:
            bounds = prior_bounds(animal['true_model'].lower(), name)
            assert list(animal['true_params']) == list(bounds)
            assert all(bounds[k][0] <= v <= bounds[k][1] for k, v in animal['true_params'].items())
        seen[name] = [a['true_params'] for a in cohort['animals'] if a['true_model'] == 'BE']
    assert min(p['eta_learning'] for p in seen['learning']) < 0.05     # below the manuscript floor
    assert min(p['eta_learning'] for p in seen['manuscript']) >= 0.05


def test_preset_threshold():
    assert preset_min_accuracy('uniform') == pytest.approx(0.70)


def test_shapes_and_accuracy_rule(tmp_path, monkeypatch):
    monkeypatch.setenv('SC_DATA_ROOT', str(tmp_path))
    run = tmp_path / 'run'
    (run / 'sessions').mkdir(parents=True)
    pd.DataFrame({'animal_id': ['R1', 'R1', 'R1', 'R2', 'R2', 'R2', 'R2', 'X'],
                  'kept': [True] * 7 + [False], 'session_idx': [0, 1, 2, 0, 1, 2, 3, 0],
                  'n_valid': [60, 70, 80, 90, 65, 75, 85, 99]}).to_csv(run / 'sessions' / 'uniform.csv', index=False)
    shapes = shapes_from_run(run, 'uniform')
    assert shapes == {'R1': [60, 70, 80], 'R2': [90, 65, 75, 85]}        # dropped animals are not shapes

    path = make_cohort('synthetic_uniform_test', n_per_model=3, burn_in=50, shapes=shapes,
                       min_accuracy=0.6)
    with open(path, 'rb') as f:
        cohort = pickle.load(f)
    gen = cohort['generation']
    assert gen['seed'] == COHORT_SEED and gen['min_accuracy'] == 0.6
    assert gen['BE_candidates'] - gen['BE_rejected'] == 3 and gen['SC_candidates'] - gen['SC_rejected'] == 3
    assert len(cohort['animals']) == 6
    for a in cohort['animals']:
        lengths = [len(s['choices']) for s in a['sessions']]
        assert lengths in shapes.values()
        acc = [float((s['choices'] == s['categories']).mean()) for s in a['sessions']]
        assert min(acc) >= 0.6
    assert path.with_suffix('.json').exists()


def test_impossible_rule_stops(tmp_path, monkeypatch):
    monkeypatch.setenv('SC_DATA_ROOT', str(tmp_path))
    with pytest.raises(RuntimeError, match='passed the accuracy rule'):
        make_cohort('synthetic_uniform_none', n_per_model=1, n_sessions=2, trials=30, burn_in=20,
                    min_accuracy=1.01, max_candidates_per_animal=3)
