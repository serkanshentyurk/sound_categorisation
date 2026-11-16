"""sound_categorisation.inference.tasks — one task-index convention for every SLURM array."""

import pytest
from sound_categorisation.inference.tasks import CONDITION_GRID, TRAIN_GRID, TaskGrid, gs_grid


def test_decode_encode_roundtrip():
    for grid in (TRAIN_GRID, CONDITION_GRID, gs_grid(['a', 'b', 'c'], 2)):
        for i in range(grid.n):
            assert grid.encode(**grid.decode(i)) == i


def test_last_axis_fastest():
    g = TaskGrid.of(x=('p', 'q'), y=(0, 1, 2))
    assert [g.decode(i)['y'] for i in range(3)] == [0, 1, 2]
    assert g.decode(3) == {'x': 'q', 'y': 0}
    assert g.slurm_range() == '0-5' and g.n == 6


def test_train_grid_shape():
    from sound_categorisation.settings import MODEL_TYPES, SBI_TRAIN_DISTRIBUTIONS, SBI_TRAIN_REPS
    assert TRAIN_GRID.names == ('rep', 'model', 'distribution')
    assert TRAIN_GRID.n == len(SBI_TRAIN_REPS) * len(MODEL_TYPES) * len(SBI_TRAIN_DISTRIBUTIONS)
    assert CONDITION_GRID.n == len(SBI_TRAIN_REPS) * len(MODEL_TYPES)
    assert TRAIN_GRID.n == 4 and CONDITION_GRID.n == 4      # the current run: pooled/moments x BE/SC x uniform


def test_condition_matches_train_prefix():
    """The (rep, model) of CONDITION task k must equal the (rep, model) of TRAIN tasks k*n_d .. k*n_d+n_d-1."""
    n_d = len(TRAIN_GRID.axes[2][1])
    for k in range(CONDITION_GRID.n):
        c = CONDITION_GRID.decode(k)
        for j in range(n_d):
            t = TRAIN_GRID.decode(k * n_d + j)
            assert (t['rep'], t['model']) == (c['rep'], c['model'])


def test_out_of_range():
    with pytest.raises(ValueError):
        TRAIN_GRID.decode(TRAIN_GRID.n)


def test_scripts_use_the_grids():
    import sys
    sys.argv = ['x']
    import sound_categorisation.cli.sbi_condition as R
    import sound_categorisation.cli.sbi_train as T
    assert all(T.decode_task(i) == tuple(TRAIN_GRID.decode(i).values()) for i in range(TRAIN_GRID.n))
    assert all(R.decode_task(i) == tuple(CONDITION_GRID.decode(i).values()) for i in range(CONDITION_GRID.n))


def test_sbc_grid():
    """One task per (representation at one session count, model); the counts differ by
    representation (pooled: full data and the CV halves; single: one session)."""
    import sound_categorisation.cli.sbi_sbc as S
    from sound_categorisation.inference.tasks import SBC_CASES, SBC_GRID, sbc_case
    from sound_categorisation.settings import MODEL_TYPES, SBC_DRAWS, SBI_TRAIN_REPS
    assert SBC_GRID.names == ('case', 'model')
    assert SBC_CASES == ('pooled/5', 'pooled/3', 'pooled/2', 'single/1')
    assert SBC_GRID.n == len(SBC_CASES) * len(MODEL_TYPES) == 8
    seen = set()
    for i in range(SBC_GRID.n):
        assert SBC_GRID.encode(**SBC_GRID.decode(i)) == i
        rep, model, n = S.decode_task(i)
        assert (rep, n) == sbc_case(SBC_GRID.decode(i)['case']) and model in MODEL_TYPES
        assert rep in SBI_TRAIN_REPS and n in SBC_DRAWS[rep]
        seen.add((rep, model, n))
    assert len(seen) == SBC_GRID.n


def test_settings_of_the_representations_agree():
    """The session counts a network is calibrated and conditioned at lie inside what it trains on."""
    from sound_categorisation.settings import (
        MODEL_ID_MIN_SESSIONS,
        SBC_DRAWS,
        SBI_REPRESENTATIONS,
        SBI_SINGLE_T,
        SBI_TRAIN_REPS,
        SBI_TRAIN_T,
    )
    assert SBI_TRAIN_REPS == ('pooled', 'single') and set(SBI_TRAIN_REPS) <= set(SBC_DRAWS)
    lo, hi = SBI_REPRESENTATIONS['pooled']['N']
    assert all(lo <= n <= hi for n in SBC_DRAWS['pooled'])
    assert lo <= MODEL_ID_MIN_SESSIONS // 2 and hi >= 5          # a CV half, and an animal's full data
    assert SBI_REPRESENTATIONS['single']['N'] == 1 and set(SBC_DRAWS['single']) == {1}
    assert SBI_REPRESENTATIONS['single']['T'] == SBI_SINGLE_T
    assert SBI_SINGLE_T[0] <= SBI_TRAIN_T[0] // 2                # half of the shortest selected session
    assert SBI_REPRESENTATIONS['pooled']['accuracy_rule'] and not SBI_REPRESENTATIONS['single']['accuracy_rule']
