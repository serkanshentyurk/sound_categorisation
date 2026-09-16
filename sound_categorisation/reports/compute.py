"""
Report computation — pure functions, no matplotlib.

    res = compute_animal(experiment, 'SS15', 'Hard-A', 'opto')           # AnimalResult
    grp = compute_group(experiment, ['SS15', ...], 'Hard-A', 'opto')      # GroupResult

An ``AnimalResult`` holds the typed contrasts (``OptoContrasts``), the per-condition
readouts, and the adaptation results; ``tables.to_tables`` flattens it to tidy
frames and ``tables.write_result`` persists them with metadata.

Design-level settings live in :class:`Settings`; ``Settings.fast()`` is the
seconds-long structural check used by ``--fast`` and the selftest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from behav_utils.analysis import collect_rows, compare_groups
from behav_utils.data.arrays import TrialArrays
from behav_utils.data.ops.filtering import filter_trials
from behav_utils.readouts import (
    PsychometricCurve,
    UpdateMatrix,
    compute_psychometric_curve,
    compute_update_matrix,
)

from sound_categorisation.adaptation import Trajectory, compute_trajectory
from sound_categorisation.cohort import collect_sessions_alm, collect_sessions_ppc, gather_genotypes
from sound_categorisation.contrasts import (
    BIAS,
    DUAL_UNITS,
    N_BOOT,
    N_PERM,
    SENSITIVITY,
    STATS,
    STATS_RT,
    OptoContrasts,
    alm_contrasts,
    dod_point,
    ppc_contrasts,
)

__all__ = ['Settings', 'AnimalResult', 'GroupResult', 'compute_animal', 'compute_group', 'DESIGNS',
           'trajectory_distributions']

DESIGNS = ('ppc', 'alm')
PHASES = {'ppc': ('opto', 'masking'), 'alm': ('alm', 'masking', 'opto')}
BETWEEN_KEY = {'ppc': 'opto_vs_masking', 'alm': 'alm_vs_masking'}


@dataclass(frozen=True)
class Settings:
    stats: Tuple[str, ...] = tuple(STATS)
    stats_rt: Tuple[str, ...] = tuple(STATS_RT)
    display: Tuple[str, ...] = tuple(SENSITIVITY + BIAS)
    n_boot: int = N_BOOT
    n_perm: int = N_PERM
    units: Tuple[str, ...] = DUAL_UNITS
    readouts: bool = True          # psychometric curves + update matrices per condition
    trajectory: bool = True        # per-session trajectory + adaptation (pse_dynamics per session)
    curve_bootstrap: int = 200     # bootstrap draws for the psychometric-curve band
    sigma: float | None = None  # sigma for the normative PSE; None = the animal's own psychometric sigma

    @classmethod
    def fast(cls) -> Settings:
        return cls(stats=('accuracy', 'side_bias'),
                   stats_rt=('accuracy', 'side_bias', 'reaction_time', 'reaction_time_jitter'),
                   display=('accuracy', 'side_bias'), n_boot=30, n_perm=30,
                   readouts=False, trajectory=False, curve_bootstrap=0)

    def names(self, design: str) -> List[str]:
        return list(self.stats_rt if design == 'alm' else self.stats)

    def display_for(self, design: str) -> List[str]:
        d = list(self.display)
        if design == 'alm':
            d += ['reaction_time', 'reaction_time_jitter']
        return d

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class ConditionReadouts:
    curve: PsychometricCurve | None
    update_matrix: UpdateMatrix | None


@dataclass(frozen=True)
class AnimalResult:
    cohort: str
    animal: str
    genotype: str
    distribution: str
    design: str
    site: str | None
    toi: str
    contrasts: OptoContrasts
    readouts: Dict[Tuple[str, str], ConditionReadouts] = field(default_factory=dict)   # (phase, trial_type)
    trajectory: Trajectory | None = None                                            # per-session, in order
    n_sessions: Dict[str, int] = field(default_factory=dict)

    @property
    def tag(self) -> str:
        site = f' · ALM-{self.site}' if self.site else ''
        return f'{self.animal} · {self.genotype} · {self.distribution}{site}'

    def __repr__(self) -> str:
        return (f'AnimalResult({self.animal!r}, {self.distribution!r}, {self.design}, toi={self.toi!r}, '
                f'contrasts={[k for k in ("within", "within_masking", "between", "compensation", "dod", "vs_ppc") if k in self.contrasts]})')


@dataclass(frozen=True)
class GroupResult:
    cohort: str
    distribution: str
    design: str
    site: str | None
    toi: str
    rows: pd.DataFrame            # animal, group, kind, stat, value   (per-animal point differences)
    tests: pd.DataFrame           # kind, stat, p, statistic, n_wt, n_het, …  (WT vs HET rank tests)
    animals: Tuple[str, ...]
    by_animal: Dict[str, str]
    trajectories: Dict[str, Trajectory] = field(default_factory=dict)                  # animal → trajectory

    def __repr__(self) -> str:
        return (f'GroupResult({self.distribution!r}, {self.design}, toi={self.toi!r}, '
                f'n_animals={len(self.animals)}, kinds={sorted(self.rows["kind"].unique()) if len(self.rows) else []})')


# ── per animal ──────────────────────────────────────────────────────────────

def trajectory_distributions(distribution: str) -> Tuple[str, ...]:
    """Uniform is blocked and stands alone; the Hard phase alternates A/B, so both are one trajectory."""
    return ('Uniform',) if distribution.lower() == 'uniform' else ('Hard-A', 'Hard-B')


def _trajectory(animal, distribution: str, s: Settings) -> Trajectory | None:
    if not s.trajectory:
        return None
    try:
        return compute_trajectory(animal, trajectory_distributions(distribution), sigma=s.sigma)
    except Exception as exc:                          # never let one animal kill the batch
        import warnings
        warnings.warn(f'{getattr(animal, "animal_id", "?")}: trajectory ({distribution}) failed: {exc}', stacklevel=2)
        return None


def _sessions_for(experiment, aid: str, distribution: str, design: str, site: str | None):
    animal = experiment.animals[aid]
    if design == 'ppc':
        return animal, collect_sessions_ppc(animal, distribution)
    return animal, collect_sessions_alm(animal, distribution, site)


def _readouts(sessions, toi: str, s: Settings) -> Dict[Tuple[str, str], ConditionReadouts]:
    out = {}
    for phase, sess in sessions.items():
        for tt in ('non_opto', toi, 'all'):
            cond = filter_trials(sess, trial_type=tt)
            if not cond:
                continue
            arrays = TrialArrays.from_sessions(cond)
            out[(phase, tt)] = ConditionReadouts(
                compute_psychometric_curve(arrays, n_bootstrap=s.curve_bootstrap),
                compute_update_matrix(arrays) if tt != 'all' else None)
    return out


def compute_animal(experiment, aid: str, distribution: str, toi: str, *, design: str = 'ppc',
                   site: str | None = None, cohort: str = '', settings: Settings = Settings(),
                   genotype: str | None = None) -> AnimalResult:
    """Everything the per-animal report needs, for one animal × distribution × toi."""
    if design not in DESIGNS:
        raise ValueError(f'design must be in {DESIGNS}, got {design!r}')
    if design == 'alm' and site is None:
        raise ValueError("design='alm' needs site='uni' or 'bi'")
    animal, sessions = _sessions_for(experiment, aid, distribution, design, site)
    if genotype is None:
        genotype = gather_genotypes(experiment)[0].get(aid, 'unknown')
    names = settings.names(design)
    if design == 'ppc':
        con = ppc_contrasts(sessions['opto'], sessions['masking'], toi, names,
                            n_boot=settings.n_boot, n_perm=settings.n_perm, units=settings.units)
    else:
        con = alm_contrasts(sessions['alm'], sessions['masking'], sessions['opto'], toi, names,
                            n_boot=settings.n_boot, n_perm=settings.n_perm, units=settings.units)
    readouts = _readouts(sessions, toi, settings) if settings.readouts else {}
    trajectory = _trajectory(animal, distribution, settings) if design == 'ppc' else None
    return AnimalResult(cohort, aid, genotype, distribution, design, site, toi, con, readouts, trajectory,
                        {k: len(v) for k, v in sessions.items()})


# ── group ───────────────────────────────────────────────────────────────────

def _point_rows(con: OptoContrasts, design: str, aid: str, group: str) -> List[dict]:
    rows = []

    def add(kind, series):
        rows.extend(r | {'kind': kind} for r in collect_rows(
            [{'stat': k, 'value': float(v)} for k, v in series.items()], animal=aid, group=group))
    if 'within' in con:
        add('within', con['within'].contrasts[con.key].diff)
    if 'within_masking' in con:
        add('within_masking', con['within_masking'].contrasts[con.key].diff)
    if 'between' in con:
        add('between', con['between'].contrasts[BETWEEN_KEY[design]].diff)
    if 'compensation' in con:
        add('compensation', con['compensation'].contrasts['laser_off_vs_masking'].diff)
    if 'vs_ppc' in con:
        add('vs_ppc', con['vs_ppc'].contrasts['alm_vs_opto'].diff)
    if 'within' in con and 'within_masking' in con:
        add('dod', dod_point(con))
    return rows


def compute_group(experiment, animals: Sequence[str], distribution: str, toi: str, *, design: str = 'ppc',
                  site: str | None = None, cohort: str = '', settings: Settings = Settings()) -> GroupResult:
    """WT-vs-HET fold: per-animal point differences per contrast kind, rank-tested across genotype.

    Two delta-of-deltas appear in ``rows``/``tests``: ``dod`` (opto − masking within
    each animal, the light-artefact-corrected effect) and, in ``tests`` only,
    ``within`` itself compared WT vs HET — the genotype-based correction (HET
    within − WT within), which is the honest silencing estimate if masking
    sessions do not reproduce the laser click.
    """
    by_animal, _ = gather_genotypes(experiment)
    names = settings.names(design)
    rows: List[dict] = []
    trajectories: Dict[str, Trajectory] = {}
    for aid in animals:
        animal, sessions = _sessions_for(experiment, aid, distribution, design, site)
        if design == 'ppc':
            con = ppc_contrasts(sessions['opto'], sessions['masking'], toi, names, n_boot=0, n_perm=0)
        else:
            con = alm_contrasts(sessions['alm'], sessions['masking'], sessions['opto'], toi, names, n_boot=0, n_perm=0)
        rows += _point_rows(con, design, aid, by_animal.get(aid, 'unknown'))
        if design == 'ppc':
            tr = _trajectory(animal, distribution, settings)
            if tr is not None:
                trajectories[aid] = tr
    df = pd.DataFrame(rows, columns=['animal', 'group', 'kind', 'stat', 'value'])
    tests = []
    for kind, sub in df.groupby('kind', sort=False):
        if sub['group'].nunique() < 2:       # one genotype present (e.g. --limit 1): rows only, no test
            continue
        res = compare_groups(sub, group_col='group')
        for stat, r in res.items():
            tests.append({'kind': kind, 'stat': stat, **{k: v for k, v in r.items() if np.isscalar(v)}})
    tests_df = pd.DataFrame(tests, columns=['kind', 'stat', 'p'] if not tests else None)
    return GroupResult(cohort, distribution, design, site, toi, df, tests_df, tuple(animals), by_animal, trajectories)
