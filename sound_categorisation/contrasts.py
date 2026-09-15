"""
The opto contrasts of the project, as typed results.

For one animal at one distribution, given its session sets (``cohort.collect_sessions_*``):

    within          {toi} vs non_opto trials, in the manipulation sessions   (permutation p valid)
    within_masking  {toi} vs non_opto trials, in the masking sessions        (permutation p valid)
    between         manipulation vs masking sessions, all trials             (bootstrap only)
    dod             within − within_masking                                   (bootstrap only)
    vs_ppc          ALM vs PPC-opto sessions, all trials (ALM design only)    (bootstrap only)

``toi`` (trial of interest) is ``'opto'`` or ``'post_opto'``. Every entry is a
``DeltaStats`` (or ``Interaction`` for ``dod``) from ``behav_utils.analysis``;
``dod_point`` gives the per-animal delta-of-deltas as a Series for the group fold.

Units: within-phase contrasts are read on the trial unit (laser randomised per
trial); between-phase and dod carry both the trial and the session unit — the
session interval is the honest one because session type was assigned per session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import pandas as pd

from behav_utils.analysis.comparison import (
    DeltaStats, Interaction, compute_delta_stat, compute_interaction, contrast_key,
)
from behav_utils.data.ops.filtering import filter_trials
from behav_utils.stats import PSYCHOMETRIC

__all__ = ['OptoContrasts', 'ppc_contrasts', 'alm_contrasts', 'dod_point',
           'STATS', 'STATS_RT', 'SENSITIVITY', 'BIAS', 'DUAL_UNITS', 'N_BOOT', 'N_PERM']

STATS = ['accuracy', 'hard_accuracy', 'easy_accuracy', 'recency', 'side_bias',
         *PSYCHOMETRIC, 'lose_shift', 'win_stay']
STATS_RT = STATS + ['reaction_time', 'reaction_time_jitter']
SENSITIVITY = ['accuracy', 'hard_accuracy', 'easy_accuracy', 'sigma', 'recency',
               'lapse_low', 'lapse_high', 'lose_shift', 'win_stay']
BIAS = ['mu', 'side_bias']
DUAL_UNITS = ('trials', 'sessions')
N_BOOT, N_PERM = 1000, 1000


@dataclass(frozen=True)
class OptoContrasts:
    """The contrasts for one animal × distribution × toi. Absent entries mean no sessions."""

    toi: str
    key: str                                     # f'{toi}_vs_non_opto'
    within: Optional[DeltaStats] = None
    within_masking: Optional[DeltaStats] = None
    between: Optional[DeltaStats] = None
    dod: Optional[Interaction] = None
    vs_ppc: Optional[DeltaStats] = None
    labels: Dict[str, str] = field(default_factory=dict)   # {'between': 'opto_vs_masking', ...}

    def __getitem__(self, name: str):
        v = getattr(self, name)
        if v is None:
            raise KeyError(name)
        return v

    def __contains__(self, name: str) -> bool:
        return getattr(self, name, None) is not None

    def table(self, unit: Optional[str] = None) -> pd.DataFrame:
        """All available contrasts stacked: adds a ``kind`` column (within, between, dod, …)."""
        frames = []
        for kind in ('within', 'within_masking', 'between', 'vs_ppc'):
            r = getattr(self, kind)
            if r is not None:
                u = unit if unit in r.units else None
                frames.append(r.table(u).assign(kind=kind))
        if self.dod is not None:
            u = unit if unit in self.dod.units else None
            t = self.dod.table(u).rename(columns={'interaction': 'diff', 'p': 'boot_p'})
            frames.append(t[['stat', 'diff', 'ci_lo', 'ci_hi', 'boot_p', 'unit']].assign(
                perm_p=float('nan'), contrast=f'dod_{self.key}', kind='dod'))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _delta(phases, names, reference, n_perm, n_boot, units):
    return compute_delta_stat(phases, names, reference=reference, curve=False,
                              n_permutations=n_perm, n_bootstrap=n_boot, units=units)


def ppc_contrasts(opto, masking, toi: str, names: Sequence[str] = STATS, *,
                  n_boot: int = N_BOOT, n_perm: int = N_PERM,
                  units: Sequence[str] = DUAL_UNITS) -> OptoContrasts:
    """PPC design. ``n_boot=0`` gives point differences only (the group fold) and no dod."""
    key = contrast_key(toi, 'non_opto')
    o_toi, o_non = filter_trials(opto, trial_type=toi), filter_trials(opto, trial_type='non_opto')
    m_toi, m_non = filter_trials(masking, trial_type=toi), filter_trials(masking, trial_type='non_opto')
    o_all, m_all = filter_trials(opto, trial_type='all'), filter_trials(masking, trial_type='all')
    within = within_masking = between = dod = None
    if o_toi and o_non:
        within = _delta({toi: o_toi, 'non_opto': o_non}, names, 'non_opto', n_perm, n_boot, units)
    if m_toi and m_non:
        within_masking = _delta({toi: m_toi, 'non_opto': m_non}, names, 'non_opto', n_perm, n_boot, units)
    if o_all and m_all:
        between = _delta({'opto': o_all, 'masking': m_all}, names, 'masking', 0, n_boot, units)
    if n_boot > 0 and within is not None and within_masking is not None:
        dod = compute_interaction(within, within_masking, key, label_a='opto', label_b='masking')
    return OptoContrasts(toi, key, within, within_masking, between, dod, None,
                         labels={'between': 'opto_vs_masking'})


def alm_contrasts(alm, masking, opto, toi: str, names: Sequence[str] = STATS_RT, *,
                  n_boot: int = N_BOOT, n_perm: int = N_PERM,
                  units: Sequence[str] = DUAL_UNITS) -> OptoContrasts:
    """ALM design: the PPC four plus ALM vs PPC-opto (all trials)."""
    key = contrast_key(toi, 'non_opto')
    a_toi, a_non = filter_trials(alm, trial_type=toi), filter_trials(alm, trial_type='non_opto')
    m_toi, m_non = filter_trials(masking, trial_type=toi), filter_trials(masking, trial_type='non_opto')
    a_all, m_all, o_all = (filter_trials(alm, trial_type='all'), filter_trials(masking, trial_type='all'),
                           filter_trials(opto, trial_type='all'))
    within = within_masking = between = dod = vs_ppc = None
    if a_toi and a_non:
        within = _delta({toi: a_toi, 'non_opto': a_non}, names, 'non_opto', n_perm, n_boot, units)
    if m_toi and m_non:
        within_masking = _delta({toi: m_toi, 'non_opto': m_non}, names, 'non_opto', n_perm, n_boot, units)
    if a_all and m_all:
        between = _delta({'alm': a_all, 'masking': m_all}, names, 'masking', 0, n_boot, units)
    if n_boot > 0 and within is not None and within_masking is not None:
        dod = compute_interaction(within, within_masking, key, label_a='alm', label_b='masking')
    if a_all and o_all:
        vs_ppc = _delta({'alm': a_all, 'opto': o_all}, names, 'opto', 0, n_boot, units)
    return OptoContrasts(toi, key, within, within_masking, between, dod, vs_ppc,
                         labels={'between': 'alm_vs_masking', 'vs_ppc': 'alm_vs_opto'})


def dod_point(r: OptoContrasts) -> pd.Series:
    """Per-animal delta-of-deltas point: within Δ − within_masking Δ (no resampling needed)."""
    a = r['within'].contrasts[r.key].diff
    b = r['within_masking'].contrasts[r.key].diff
    return (a - b.reindex(a.index)).rename('value')
