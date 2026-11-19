"""
One task-index convention for every SLURM array in the project.

A :class:`TaskGrid` is an ordered product of named axes; index ``i`` in
``range(grid.n)`` decodes to one combination, last axis fastest. The same grid
object decides the array size (``grid.slurm_range()``), the job's arguments
(``grid.decode(task_id)``) and the reverse lookup (``grid.encode(**kw)``), so
train / condition / grid-search cannot drift apart.

    TRAIN_GRID.n                      # 4 networks (SBI_TRAIN_REPS x models x SBI_TRAIN_DISTRIBUTIONS)
    TRAIN_GRID.decode(3)              # {'rep': 'single', 'model': 'SC', 'distribution': 'uniform'}
    gs_grid(animals, n_seeds=3).slurm_range()   # '0-53'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from sound_categorisation.settings import (
    MODEL_TYPES,
    SBC_DRAWS,
    SBI_TRAIN_DISTRIBUTIONS,
    SBI_TRAIN_REPS,
)

__all__ = ['TaskGrid', 'TRAIN_GRID', 'CONDITION_GRID', 'SBC_CASES', 'SBC_GRID', 'sbc_case', 'gs_grid',
           'slurm_array']


@dataclass(frozen=True)
class TaskGrid:
    axes: Tuple[Tuple[str, Tuple], ...]      # ((name, values), ...) in order, last fastest

    @classmethod
    def of(cls, **axes) -> TaskGrid:
        return cls(tuple((k, tuple(v)) for k, v in axes.items()))

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(k for k, _ in self.axes)

    @property
    def n(self) -> int:
        out = 1
        for _, v in self.axes:
            out *= len(v)
        return out

    def decode(self, task_id: int) -> Dict[str, object]:
        if not 0 <= task_id < self.n:
            raise ValueError(f'task_id must be in [0, {self.n}); got {task_id}')
        out, rem = {}, task_id
        for name, values in reversed(self.axes):
            out[name] = values[rem % len(values)]
            rem //= len(values)
        return {k: out[k] for k in self.names}

    def encode(self, **kw) -> int:
        idx = 0
        for name, values in self.axes:
            idx = idx * len(values) + values.index(kw[name])
        return idx

    def all(self):
        for i in range(self.n):
            yield i, self.decode(i)

    def slurm_range(self) -> str:
        return f'0-{self.n - 1}'

    def describe(self) -> str:
        return ' × '.join(f'{k}[{len(v)}]' for k, v in self.axes) + f' = {self.n} tasks'


# network training: one specialist network per (representation, model, distribution). The axes are
# settings.SBI_TRAIN_REPS / SBI_TRAIN_DISTRIBUTIONS -- widen those to train more networks.
TRAIN_GRID = TaskGrid.of(rep=SBI_TRAIN_REPS, model=MODEL_TYPES, distribution=SBI_TRAIN_DISTRIBUTIONS)

# conditioning: one job per (representation, model); the distribution is a run-level argument
CONDITION_GRID = TaskGrid.of(rep=SBI_TRAIN_REPS, model=MODEL_TYPES)

# SBC: one job per (case, model). A case is a representation at one session count,
# '<rep>/<n_sessions>' (settings.SBC_DRAWS): the session counts differ between representations
# (pooled: full data and the CV half sizes; single: one session), so they are not a product axis.
SBC_CASES = tuple(f'{rep}/{n}' for rep in SBI_TRAIN_REPS for n in SBC_DRAWS[rep])
SBC_GRID = TaskGrid.of(case=SBC_CASES, model=MODEL_TYPES)


def sbc_case(case: str) -> Tuple[str, int]:
    """'pooled/5' -> ('pooled', 5)."""
    rep, n = case.split('/')
    return rep, int(n)


def gs_grid(animals: Sequence[str], n_seeds: int, models: Sequence[str] = ('BE', 'SC')) -> TaskGrid:
    """Grid search: one job per (animal, model, seed)."""
    return TaskGrid.of(animal=tuple(animals), model=tuple(models), seed=tuple(range(n_seeds)))


def slurm_array(task_ids) -> str:
    """Task ids as a SLURM ``--array`` list: runs of consecutive ids as ranges, joined by commas
    (``[0, 1, 2, 3, 7, 9, 10]`` -> ``'0-3,7,9-10'``), so a long list of tasks to resubmit stays short."""
    ids = sorted({int(t) for t in task_ids})
    parts, start = [], None
    for i, t in enumerate(ids):
        start = t if start is None else start
        if i + 1 == len(ids) or ids[i + 1] != t + 1:
            parts.append(str(t) if start == t else f'{start}-{t}')
            start = None
    return ','.join(parts)
