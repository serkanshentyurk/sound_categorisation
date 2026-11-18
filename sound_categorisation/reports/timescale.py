"""
Timescale report (thesis plan, analysis 6): how long the criterion remembers the stimulus distribution.

    sc-reports timescale --cohort behaviour1-cohort        # the blocked switches
    sc-reports timescale --cohort opto1-cohort             # the daily alternation of the opto Hard phase

Per animal, ``behaviour.timescale.compute_timescale``: a leaky integrator fitted to the session criterion
(memory N, offset, drift; gain 1 and free), for every variant the animal's sessions allow, and the shift
with the drift taken out. Writes into the run directory: sessions.csv, fits.csv, shifts.csv,
profiles.csv, meta.json, and timescale.pdf (the memory per animal, then each animal's sessions with the
fit). Notebook 13 draws the thesis figure from these tables.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sound_categorisation.behaviour.timescale import (
    LEVEL,
    MEMORY_GRID,
    VARIANTS,
    TimescaleResult,
    compute_timescale,
    timescale_tables,
)
from sound_categorisation.data.cohort import gather_genotypes
from sound_categorisation.plotting.timescale import plot_memory, plot_session_criterion
from sound_categorisation.reports.tables import write_result

__all__ = ['compute_timescale_cohort', 'run_timescale']


def compute_timescale_cohort(experiment, animals: Sequence[str]) -> List[TimescaleResult]:
    """One result per animal with Hard sessions; an animal that fails is left out with a warning."""
    out = []
    for aid in animals:
        if aid not in experiment.animals:
            continue
        try:
            res = compute_timescale(experiment.animals[aid])
        except Exception as exc:                          # one animal must not stop the cohort
            warnings.warn(f'{aid}: timescale failed: {exc!r}', stacklevel=2)
            continue
        if res is not None:
            out.append(res)
    return out


def run_timescale(experiment, animals: Sequence[str], run: Path, cohort: str, meta: dict | None = None) -> Path:
    """Write the timescale report into ``run`` (a run directory from ``start_run``)."""
    matplotlib.use('Agg')
    results = compute_timescale_cohort(experiment, animals)
    genotypes, _ = gather_genotypes(experiment)
    T = timescale_tables(results, genotypes)
    finite = MEMORY_GRID[MEMORY_GRID < float('inf')]
    write_result(run, T, None, {**(meta or {}), 'kind': 'timescale', 'cohort': cohort,
                                'animals': [r.animal for r in results], 'level': LEVEL,
                                'memory_grid': {'min': float(finite.min()), 'max': float(finite.max()),
                                                'n': int(len(finite)), 'plus_infinity': True}})
    with PdfPages(Path(run) / 'timescale.pdf') as pdf:
        if len(T['fits']):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
            for ax, gain in zip(axes, ('fixed', 'free')):
                plot_memory(T['fits'], 'main', gain, ax=ax)
                ax.set_title(f'{cohort}: memory, main sessions, gain {gain}', fontsize=10)
            axes[0].legend(fontsize=8, frameon=False)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        for r in results:
            variants = [v for v in VARIANTS if r.fit(v) is not None]
            fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 3.2), squeeze=False)
            for ax, v in zip(axes[0], variants):
                plot_session_criterion(r.sessions, v, ax=ax)
                f = r.fit(v)
                ax.set_title(f'{r.animal} ({r.design}{", biased" if r.biased else ""}) {v}: '
                             f'N = {f.n_hat:.0f} [{f.n_low:.0f}, {f.n_high:.0f}]', fontsize=9)
            axes[0][0].legend(fontsize=7, frameon=False)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f'timescale: {len(results)} animals -> {run}')
    return Path(run)
