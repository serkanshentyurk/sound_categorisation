"""End-to-end: the report pipeline on synthetic data — schema, persistence round-trip, the opto run
with summary pages, the switch report, and a committed reference that pins every number the fast
settings produce.

If a deliberate change moves the numbers, regenerate the reference on purpose and say so in the commit:

    pytest tests/e2e -q --regen-reference

To keep the synthetic outputs for inspection: ``pytest tests/e2e -q --basetemp=/tmp/sc_e2e``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.e2e
from sound_categorisation.data.synthetic import synthetic_experiment
from sound_categorisation.reports import (
    CONTRAST_COLUMNS,
    Settings,
    compute_animal,
    compute_group,
    group_tables,
    read_result,
    readout_arrays,
    to_tables,
    write_result,
)

REFERENCE = Path(__file__).parent / 'reference' / 'e2e_contrasts.csv'


@pytest.fixture(scope='module')
def exp():
    return synthetic_experiment()


@pytest.fixture(scope='module')
def fast_result(exp):
    return compute_animal(exp, 'ST00', 'Uniform', 'opto', cohort='selftest', settings=Settings.fast())


def test_contrast_table_schema(fast_result):
    t = to_tables(fast_result)['contrasts']
    assert list(t.columns) == CONTRAST_COLUMNS
    assert set(t['kind']) == {'within', 'within_masking', 'between', 'compensation', 'dod'}
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


def test_readouts_and_trajectory_present_when_enabled(exp):
    r = compute_animal(exp, 'ST00', 'Hard-A', 'opto', settings=Settings(n_boot=10, n_perm=0, curve_bootstrap=0))
    assert ('opto', 'non_opto') in r.readouts and r.readouts[('opto', 'all')].update_matrix is None
    assert r.trajectory is not None and set(r.trajectory.distributions) == {'Hard-A', 'Hard-B'}
    arrays = readout_arrays(r)
    assert any(k.endswith('__um') for k in arrays)
    t = to_tables(r)['trajectory']
    assert {'order', 'distribution', 'phase', 'session_type', 'pse', 'pse_fixed', 'pse_tau', 'delta_from_prev', 'convergence_final'} <= set(t.columns)
    assert set(t['distribution']) == {'Hard-A', 'Hard-B'} and (t['phase'] == 'Hard-A').all()
    assert list(t['order']) == sorted(t['order'])                      # acquisition order
    assert (t['session_type'].iloc[0] == 'opto') and (t['session_type'].iloc[-1] == 'masking')   # laser half first
    curves = to_tables(r)['trajectory_curves']
    assert len(curves) and curves['trial'].max() <= 200                # trial index within a session


def test_group_fold(exp):
    g = compute_group(exp, list(exp.animals), 'Uniform', 'opto', settings=Settings.fast())
    assert set(g.rows['kind']) == {'within', 'within_masking', 'between', 'compensation', 'dod'}
    assert set(g.rows['group']) == {'wt', 'het'}
    assert {'kind', 'stat', 'p', 'n_a', 'n_b'} <= set(g.tests.columns)
    gt = group_tables(g)
    assert 'group_rows' in gt and 'group_tests' in gt


def test_reference_numbers(fast_result, request):
    got = to_tables(fast_result)['contrasts'].sort_values(['kind', 'unit', 'stat']).reset_index(drop=True)
    if request.config.getoption('--regen-reference'):
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        got.to_csv(REFERENCE, index=False)
        pytest.skip(f'reference regenerated: {REFERENCE}')
    if not REFERENCE.exists():
        pytest.fail(f'no reference at {REFERENCE}; run with --regen-reference once and commit it')
    ref = pd.read_csv(REFERENCE).sort_values(['kind', 'unit', 'stat']).reset_index(drop=True)
    assert list(got['stat']) == list(ref['stat'])
    for col in ('diff', 'ci_lo', 'ci_hi', 'boot_p', 'perm_p'):
        np.testing.assert_allclose(got[col].to_numpy(dtype=float), ref[col].to_numpy(dtype=float),
                                   rtol=1e-6, atol=1e-9, equal_nan=True, err_msg=col)


def test_summary_pages_draw(exp, tmp_path):
    """The four summary pages build from written tables (synthetic, fast + one full animal)."""
    import matplotlib
    matplotlib.use('Agg')
    from types import SimpleNamespace

    from sound_categorisation.data.paths import latest_run, start_run
    from sound_categorisation.reports.cli import run_animal, run_group
    from sound_categorisation.reports.summary import write_summary
    ids = list(exp.animals)
    run = start_run('opto_contrasts', 'selftest', fast=True, root=tmp_path)
    assert run.name.endswith('_fast') and (run / 'logs').is_dir()
    assert latest_run('opto_contrasts', 'selftest', root=tmp_path) == run
    a = SimpleNamespace(cohort='selftest', snapshot=None, config=None, fast=True, run_path=run, run_id=run.name)
    per = run_animal(exp, ids, 'Uniform', 'opto', 'ppc', None, a, Settings.fast())
    run_group(exp, ids, 'Uniform', 'opto', 'ppc', None, a, Settings.fast(), per)
    full = Settings(n_boot=10, n_perm=0, curve_bootstrap=0)
    per = run_animal(exp, ids[:2], 'Hard-A', 'opto', 'ppc', None, a, full)
    run_group(exp, ids[:2], 'Hard-A', 'opto', 'ppc', None, a, full, per)
    path = write_summary(run, 'selftest')
    assert path.exists() and (run / 'summary_hard_a.png').exists()
    meta = json.loads((run / 'Uniform' / 'ppc_opto' / ids[0] / 'meta.json').read_text())
    assert meta['run_id'] == run.name and 'argv' in meta and 'git_dirty' in meta


def test_switches_pipeline(exp, tmp_path):
    """Blocked animals: qualifying switches, transition labels, table schema, summary pages."""
    import matplotlib
    matplotlib.use('Agg')
    from sound_categorisation.behaviour.adaptation import compute_switches
    from sound_categorisation.reports.switches import compute_switches_cohort, run_switches, switch_tables
    res = compute_switches(exp.animals['SB00'], min_block_trials=300, max_trials=800)
    assert [r.to_distribution for r in res] == ['Hard-B', 'Hard-A', 'Hard-B', 'Hard-A']
    assert [r.transition for r in res] == ['first', 'novel', 'return', 'return']
    r = res[1]
    assert set(r.convergence['method']) == {'manuscript', 'pinned', 'pinned_running'}
    assert (r.convergence['trial'] >= 1).all()
    assert r.convergence.loc[r.convergence['method'] == 'manuscript', 'convergence_clipped'].dropna().between(0, 1).all()
    assert {'pse_tau', 'trials_to_criterion', 'plateau', 'pse_shape_daic'} <= set(r.dynamics.index)
    assert len(r.sessions) == r.n_sessions_after and len(r.overnight) == r.n_sessions_after - 1
    g = compute_switches_cohort(exp, ['SB00', 'SB01', 'ST00'], cohort='selftest', min_block_trials=300, max_trials=800)
    assert g.animals == ['SB00', 'SB01']                      # ST00 has no blocks
    T = switch_tables(g)
    assert set(T) == {'switches', 'pre_post', 'convergence', 'sessions', 'overnight', 'psychometrics', 'psychometric_curves'}
    assert {'animal', 'switch_idx', 'transition', 'stat', 'value'} <= set(T['switches'].columns)
    from sound_categorisation.data.paths import start_run
    run = start_run('switch_adaptation', 'selftest', root=tmp_path)
    out = run_switches(exp, ['SB00', 'SB01'], run, 'selftest', min_block_trials=300, max_trials=800)
    assert out == run / 'switches'
    assert (out / 'summary_switches.pdf').exists() and (out / 'pdf' / 'SB00_switches.pdf').exists()


def test_bias_rule_flags_distribution_independent_bias():
    """A clean animal is not flagged; an animal with |PSE| > 0.4 of the same sign on A and B blocks is."""
    import pandas as pd
    from sound_categorisation.behaviour.adaptation import flag_biased_sessions
    rows = []
    for k in range(10):
        dist = 'Hard-A' if k % 2 else 'Hard-B'
        rows.append({'animal': 'clean', 'to_distribution': dist, 'pse_fixed': 0.05 * (1 if dist == 'Hard-A' else -1),
                     'lapse_low': 0.05, 'lapse_high': 0.05, 'accuracy': 0.8})
        rows.append({'animal': 'biased', 'to_distribution': dist, 'pse_fixed': 0.7,
                     'lapse_low': 0.05, 'lapse_high': 0.05, 'accuracy': 0.65})
        rows.append({'animal': 'flips', 'to_distribution': dist, 'pse_fixed': 0.7 * (1 if dist == 'Hard-A' else -1),
                     'lapse_low': 0.05, 'lapse_high': 0.05, 'accuracy': 0.65})
    f = flag_biased_sessions(pd.DataFrame(rows))
    by = f.groupby('animal')['biased_animal'].first()
    assert not by['clean'] and by['biased']
    assert not by['flips']          # large but distribution-following PSE is not "independent of the distribution"
    assert f.loc[f['animal'] == 'clean', 'flagged'].sum() == 0
    assert f.loc[f['animal'] == 'biased', 'flagged'].all()


def test_phase_psychometrics_tables(exp):
    from sound_categorisation.reports.switches import compute_switches_cohort, switch_tables
    g = compute_switches_cohort(exp, ['SB00'], cohort='selftest', min_block_trials=300, max_trials=800)
    T = switch_tables(g)
    P = T['psychometrics']
    assert list(P.sort_values('order')['phase']) == ['Uniform', 'Hard-B #1', 'Hard-A #1', 'Hard-B #2', 'Hard-A #2', 'Hard-A all', 'Hard-B all']
    assert {'pse', 'sigma', 'accuracy', 'hard_accuracy', 'n_trials', 'n_sessions'} <= set(P.columns)
    assert {'flagged', 'biased_animal'} <= set(T['sessions'].columns)
    assert len(T['psychometric_curves']) == 7 * 200
