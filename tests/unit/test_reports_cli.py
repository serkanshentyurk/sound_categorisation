"""sc-reports: which animals run, and --jobs giving the same numbers as one animal at a time."""
import argparse
import warnings

import pandas as pd
import pytest

from sound_categorisation.data.cohort import ensure_presets
from sound_categorisation.data.synthetic import synthetic_experiment
from sound_categorisation.reports import cli
from sound_categorisation.reports.compute import Settings


@pytest.fixture(scope='module')
def experiment():
    ensure_presets()
    return synthetic_experiment(n_animals=2)


def test_cohort_members_without_data_are_left_out_with_a_warning(experiment):
    have = sorted(experiment.animals)
    a = argparse.Namespace(cohort='c', animals=None, limit=None)
    with pytest.warns(UserWarning, match='no data for'):
        assert cli._animals(experiment, {'c': have + ['GONE']}, a) == have


def test_a_named_animal_without_data_still_stops(experiment):
    a = argparse.Namespace(cohort='c', animals=['GONE'], limit=None)
    with pytest.raises(SystemExit):
        cli._animals(experiment, {'c': ['GONE']}, a)


def test_jobs_give_the_same_numbers(experiment, tmp_path):
    ids = sorted(experiment.animals)
    tables = {}
    for jobs in (1, 2):
        out = tmp_path / f'jobs{jobs}'
        a = argparse.Namespace(run_path=out, cohort='synthetic', jobs=jobs, run_id='x', snapshot=None, config=None,
                               fast=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cli.run_animal(experiment, ids, 'Uniform', 'opto', 'ppc', None, a, Settings.fast())
        files = sorted(out.rglob('contrasts.csv'))
        assert len(files) == len(ids)
        tables[jobs] = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pd.testing.assert_frame_equal(tables[1], tables[2])
