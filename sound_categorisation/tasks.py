"""
One task-index convention for every SLURM array in the project.

A :class:`TaskGrid` is an ordered product of named axes; index ``i`` in
``range(grid.n)`` decodes to one combination, last axis fastest. The same grid
object decides the array size (``grid.slurm_range()``), the job's arguments
(``grid.decode(task_id)``) and the reverse lookup (``grid.encode(**kw)``), so
train / condition / grid-search cannot drift apart.

    TRAIN_GRID.n                      # 18 networks
    TRAIN_GRID.decode(7)              # {'rep': 'moments', 'model': 'BE', 'distribution': 'hard_b'}
    gs_grid(animals, n_seeds=3).slurm_range()   # '0-53'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from sound_categorisation.paths import MODEL_TYPES, SBI_REPRESENTATIONS, SBI_TRAIN_DISTRIBUTIONS

__all__ = ['TaskGrid', 'TRAIN_GRID', 'CONDITION_GRID', 'gs_grid']


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


# network training: one specialist network per (representation, model, distribution)
TRAIN_GRID = TaskGrid.of(rep=SBI_REPRESENTATIONS, model=MODEL_TYPES, distribution=SBI_TRAIN_DISTRIBUTIONS)

# conditioning: one job per (representation, model); the distribution is a run-level argument
CONDITION_GRID = TaskGrid.of(rep=SBI_REPRESENTATIONS, model=MODEL_TYPES)


def gs_grid(animals: Sequence[str], n_seeds: int, models: Sequence[str] = ('BE', 'SC')) -> TaskGrid:
    """Grid search: one job per (animal, model, seed)."""
    return TaskGrid.of(animal=tuple(animals), model=tuple(models), seed=tuple(range(n_seeds)))
