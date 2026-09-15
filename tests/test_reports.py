"""Report pipeline: schema, persistence round-trip, and a committed reference on synthetic data.

The reference (tests/reference/selftest_contrasts.csv) pins every number the fast
selftest produces. If a deliberate change moves them, regenerate with
``python -m sound_categorisation.reports selftest`` + copy, and say so in the commit.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sound_categorisation.reports import (
    CONTRAST_COLUMNS, Settings, compute_animal, compute_group, group_tables, read_result, readout_arrays,
    to_tables, write_result,
)
from sound_categorisation.reports.selftest import synthetic_experiment

REFERENCE = Path(__file__).parent / 'reference' / 'selftest_contrasts.csv'


@pytest.fixture(scope='module')
def exp():
    return synthetic_experiment()


@pytest.fixture(scope='module')
def fast_result(exp):
    return compute_animal(exp, 'ST00', 'Uniform', 'opto', cohort='selftest', settings=Settings.fast())


def test_contrast_table_schema(fast_result):
    t = to_tables(fast_result)['contrasts']
    assert list(t.columns) == CONTRAST_COLUMNS
    assert set(t['kind']) == {'within', 'within_masking', 'between', 'dod'}
    assert set(t['unit']) == {'trials', 'sessions'}
    assert t['perm_p'].notna().sum() > 0 and t.loc[t['kind'] == 'between', 'perm_p'].isna().all()


def test_alm_design_has_vs_ppc(exp):
    r = compute_animal(exp, 'ST00', 'Uniform', 'opto', design='alm', site='uni', settings=Settings.fast())
    t = to_tables(r)['contrasts']
    assert 'vs_ppc' in set(t['kind'])
    assert {'reaction_time', 'reaction_time_jitter'} <= set(t['stat'])


def test_write_read_roundtrip(fast_result, tmp_path):
    tables = to_tables(fast_result)
    write_result(tmp_path, tables, readout_arrays(fast_result), {'animal': 'ST00'})
    back, readouts, meta = read_result(tmp_path)
    assert set(back) == {k for k, v in tables.items() if len(v.columns)} and meta['animal'] == 'ST00'
    assert 'behav_utils' in meta['versions']
    num = ['diff', 'ci_lo', 'ci_hi', 'boot_p', 'perm_p']
    pd.testing.assert_frame_equal(back['contrasts'][num], tables['contrasts'][num], check_dtype=False)
    assert list(back['contrasts']['stat']) == list(tables['contrasts']['stat'])
    # '' round-trips as NaN through CSV — site is empty for the PPC design
    assert back['contrasts']['site'].isna().all()


def test_readouts_and_adaptation_present_when_enabled(exp):
    r = compute_animal(exp, 'ST00', 'Hard-A', 'opto', settings=Settings(n_boot=10, n_perm=0, curve_bootstrap=0))
    assert ('opto', 'non_opto') in r.readouts and r.readouts[('opto', 'all')].update_matrix is None
    assert set(r.adaptation) == {'opto', 'masking'}
    arrays = readout_arrays(r)
    assert any(k.endswith('__um') for k in arrays)
    dyn = to_tables(r)['adaptation_dynamics']
    assert {'pse_tau', 'convergence_final', 'trials_to_criterion'} <= set(dyn['stat'])


def test_group_fold(exp):
    g = compute_group(exp, list(exp.animals), 'Uniform', 'opto', settings=Settings.fast())
    assert set(g.rows['kind']) == {'within', 'within_masking', 'between', 'dod'}
    assert set(g.rows['group']) == {'wt', 'het'}
    assert {'kind', 'stat', 'p', 'n_a', 'n_b'} <= set(g.tests.columns)
    gt = group_tables(g)
    assert 'group_rows' in gt and 'group_tests' in gt


def test_reference_numbers(fast_result):
    got = to_tables(fast_result)['contrasts'].sort_values(['kind', 'unit', 'stat']).reset_index(drop=True)
    if not REFERENCE.exists():
        pytest.skip('no reference committed')
    ref = pd.read_csv(REFERENCE).sort_values(['kind', 'unit', 'stat']).reset_index(drop=True)
    assert list(got['stat']) == list(ref['stat'])
    for col in ('diff', 'ci_lo', 'ci_hi', 'boot_p', 'perm_p'):
        np.testing.assert_allclose(got[col].to_numpy(dtype=float), ref[col].to_numpy(dtype=float),
                                   rtol=1e-6, atol=1e-9, equal_nan=True, err_msg=col)
