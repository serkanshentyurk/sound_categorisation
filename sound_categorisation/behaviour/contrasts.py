"""
The opto contrasts of the project, as typed results.

For one animal at one distribution, given its session sets (``cohort.collect_sessions_*``):

    within          {trial_class} vs non_opto trials, in the manipulation sessions   (permutation p valid)
    within_masking  {trial_class} vs non_opto trials, in the masking sessions        (permutation p valid)
    between         manipulation vs masking sessions, all trials             (bootstrap only)
    compensation    laser-OFF trials of the manipulation sessions vs the unflagged trials of the masking
                    sessions (bootstrap only) — does the animal's baseline move in a session where 30 % of
                    trials are lasered? In WT this is the criterion offset that compensates the light effect.
                    Like for like: the flagged masking trials carry the masking effect, so they stay out.
    dod             within − within_masking                                   (bootstrap only)
    vs_ppc          ALM vs PPC-opto sessions, all trials (ALM design only)    (bootstrap only)

``trial_class`` (trial of interest) is ``'opto'`` or ``'post_opto'``. The reference of ``within`` and
``within_masking`` is the laser-off (unflagged) trials; for ``post_opto`` it leaves the post-opto trials
out, since they are laser-off trials themselves and the two sets must not share trials (the key stays
``post_opto_vs_non_opto``). Every entry is a
``DeltaStats`` (or ``Interaction`` for ``dod``) from ``behav_utils.analysis``;
``dod_point`` gives the per-animal delta-of-deltas as a Series for the group fold.

Units: within-phase contrasts are read on the trial unit (laser randomised per
trial); between-phase and dod carry both the trial and the session unit — the
session unit is the right one because session type was assigned per session, but
with three to six sessions per set its interval is too narrow (an interval meant
to cover 95 % covers roughly 80 to 90 %): read it as rough.

Seeds: every contrast of an animal resamples with its own seed (``SEED`` + ``SEED_OFFSET``). ``dod``
subtracts the draws of ``within_masking`` from those of ``within``; with one seed for both, sets with the
same number of sessions drew the same session positions, which paired the k-th laser session with the k-th
masking session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from behav_utils.analysis import DeltaStats, Interaction, compute_delta_stat, compute_interaction
from behav_utils.data import build_mask, filter_trials, opto_mask
from behav_utils.stats import PSYCHOMETRIC

__all__ = ['OptoContrasts', 'ppc_contrasts', 'alm_contrasts', 'dod_point', 'equivalence_verdict', 'VERDICTS',
           'STATS', 'STATS_RT', 'SENSITIVITY', 'BIAS', 'DUAL_UNITS', 'N_BOOT', 'N_PERM', 'SEED', 'SEED_OFFSET']


def contrast_key(label: str, reference: str) -> str:
    """The key ``DeltaStats.contrasts`` files a contrast under: ``<label>_vs_<reference>``."""
    return f'{label}_vs_{reference}'


STATS = ['accuracy', 'hard_accuracy', 'easy_accuracy', 'recency', 'side_bias',
         *PSYCHOMETRIC, 'lose_shift', 'win_stay']
STATS_RT = STATS + ['reaction_time', 'reaction_time_jitter']
SENSITIVITY = ['accuracy', 'hard_accuracy', 'easy_accuracy', 'sigma', 'recency',
               'lapse_low', 'lapse_high', 'lose_shift', 'win_stay']
BIAS = ['mu', 'side_bias']
DUAL_UNITS = ('trials', 'sessions')
N_BOOT, N_PERM = 1000, 1000
SEED = 42                      # base resampling seed (behav_utils' default, so `within` keeps its numbers)
SEED_OFFSET = {'within': 0, 'within_masking': 1, 'between': 2, 'compensation': 3, 'vs_ppc': 4}


@dataclass(frozen=True)
class OptoContrasts:
    """The contrasts for one animal × distribution × trial_class. Absent entries mean no sessions."""

    trial_class: str
    key: str                                     # f'{trial_class}_vs_non_opto'
    within: DeltaStats | None = None
    within_masking: DeltaStats | None = None
    between: DeltaStats | None = None
    compensation: DeltaStats | None = None
    dod: Interaction | None = None
    vs_ppc: DeltaStats | None = None
    labels: Dict[str, str] = field(default_factory=dict)   # {'between': 'opto_vs_masking', ...}

    def __getitem__(self, name: str):
        v = getattr(self, name)
        if v is None:
            raise KeyError(name)
        return v

    def __contains__(self, name: str) -> bool:
        return getattr(self, name, None) is not None

    def table(self, unit: str | None = None) -> pd.DataFrame:
        """All available contrasts stacked: adds a ``kind`` column (within, between, dod, …)."""
        frames = []
        for kind in ('within', 'within_masking', 'between', 'compensation', 'vs_ppc'):
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


def _delta(kind, phases, names, reference, n_perm, n_boot, units, seed):
    """One contrast, resampled with the seed of its ``kind`` (see the module docstring)."""
    return compute_delta_stat(phases, names, reference=reference, curve=False, n_permutations=n_perm,
                              n_bootstrap=n_boot, units=units, seed=seed + SEED_OFFSET[kind])


def _reference(sessions, laser_off, trial_class: str):
    """The trials ``trial_class`` is compared with: the laser-off trials, minus the post-opto ones when
    ``trial_class`` is ``post_opto`` (they are laser-off trials themselves; up to 0.7 they sat on both
    sides of the contrast, diluting it)."""
    if trial_class != 'post_opto':
        return laser_off
    return filter_trials(sessions, mask_fn=lambda s: build_mask(s.trials) & ~opto_mask(s.trials, delta=1),
                         label='non_opto without post_opto')


def ppc_contrasts(opto, masking, trial_class: str, names: Sequence[str] = STATS, *,
                  n_boot: int = N_BOOT, n_perm: int = N_PERM,
                  units: Sequence[str] = DUAL_UNITS, seed: int = SEED) -> OptoContrasts:
    """PPC design. ``n_boot=0`` gives point differences only (the group fold) and no dod."""
    key = contrast_key(trial_class, 'non_opto')
    o_toi, o_off = filter_trials(opto, trial_type=trial_class), filter_trials(opto, trial_type='non_opto')
    m_toi, m_off = filter_trials(masking, trial_type=trial_class), filter_trials(masking, trial_type='non_opto')
    o_ref, m_ref = _reference(opto, o_off, trial_class), _reference(masking, m_off, trial_class)
    o_all, m_all = filter_trials(opto, trial_type='all'), filter_trials(masking, trial_type='all')
    within = within_masking = between = compensation = dod = None
    if o_toi and o_ref:
        within = _delta('within', {trial_class: o_toi, 'non_opto': o_ref}, names, 'non_opto',
                        n_perm, n_boot, units, seed)
    if m_toi and m_ref:
        within_masking = _delta('within_masking', {trial_class: m_toi, 'non_opto': m_ref}, names,
                                'non_opto', n_perm, n_boot, units, seed)
    if o_all and m_all:
        between = _delta('between', {'opto': o_all, 'masking': m_all}, names, 'masking',
                         0, n_boot, units, seed)
    if o_off and m_off:
        compensation = _delta('compensation', {'laser_off': o_off, 'masking': m_off}, names, 'masking',
                              0, n_boot, units, seed)
    if n_boot > 0 and within is not None and within_masking is not None:
        dod = compute_interaction(within, within_masking, key, label_a='opto', label_b='masking')
    return OptoContrasts(trial_class, key, within, within_masking, between, compensation, dod, None,
                         labels={'between': 'opto_vs_masking', 'compensation': 'laser_off_vs_masking'})


def alm_contrasts(alm, masking, opto, trial_class: str, names: Sequence[str] = STATS_RT, *,
                  n_boot: int = N_BOOT, n_perm: int = N_PERM,
                  units: Sequence[str] = DUAL_UNITS, seed: int = SEED) -> OptoContrasts:
    """ALM design: the PPC four plus ALM vs PPC-opto (all trials)."""
    key = contrast_key(trial_class, 'non_opto')
    a_toi, a_off = filter_trials(alm, trial_type=trial_class), filter_trials(alm, trial_type='non_opto')
    m_toi, m_off = filter_trials(masking, trial_type=trial_class), filter_trials(masking, trial_type='non_opto')
    a_ref, m_ref = _reference(alm, a_off, trial_class), _reference(masking, m_off, trial_class)
    a_all, m_all, o_all = (filter_trials(alm, trial_type='all'), filter_trials(masking, trial_type='all'),
                           filter_trials(opto, trial_type='all'))
    within = within_masking = between = compensation = dod = vs_ppc = None
    if a_toi and a_ref:
        within = _delta('within', {trial_class: a_toi, 'non_opto': a_ref}, names, 'non_opto',
                        n_perm, n_boot, units, seed)
    if m_toi and m_ref:
        within_masking = _delta('within_masking', {trial_class: m_toi, 'non_opto': m_ref}, names,
                                'non_opto', n_perm, n_boot, units, seed)
    if a_all and m_all:
        between = _delta('between', {'alm': a_all, 'masking': m_all}, names, 'masking',
                         0, n_boot, units, seed)
    if a_off and m_off:
        compensation = _delta('compensation', {'laser_off': a_off, 'masking': m_off}, names, 'masking',
                              0, n_boot, units, seed)
    if n_boot > 0 and within is not None and within_masking is not None:
        dod = compute_interaction(within, within_masking, key, label_a='alm', label_b='masking')
    if a_all and o_all:
        vs_ppc = _delta('vs_ppc', {'alm': a_all, 'opto': o_all}, names, 'opto', 0, n_boot, units, seed)
    return OptoContrasts(trial_class, key, within, within_masking, between, compensation, dod, vs_ppc,
                         labels={'between': 'alm_vs_masking', 'compensation': 'laser_off_vs_masking',
                                 'vs_ppc': 'alm_vs_opto'})


VERDICTS = ('within margin', 'effect', 'unclear', 'no interval')


def equivalence_verdict(ci_lo: float, ci_hi: float, margin: float) -> str:
    """Read an interval of a difference against an equivalence margin set in advance.

    ``'within margin'``: the whole interval lies inside ±margin, so any effect is smaller than the margin.
    ``'effect'``: the interval excludes zero and is not inside the margin. ``'unclear'``: neither; these data
    cannot tell. ``'no interval'``: a bound is missing (no resampling, or too few trials).

    With a 95 % interval this is a little stricter than the usual equivalence test, which uses 90 %.
    """
    if not (np.isfinite(ci_lo) and np.isfinite(ci_hi)):
        return 'no interval'
    if -margin <= ci_lo and ci_hi <= margin:
        return 'within margin'
    if ci_lo > 0 or ci_hi < 0:
        return 'effect'
    return 'unclear'


def dod_point(r: OptoContrasts) -> pd.Series:
    """Per-animal delta-of-deltas point: within Δ − within_masking Δ (no resampling needed)."""
    a = r['within'].contrasts[r.key].diff
    b = r['within_masking'].contrasts[r.key].diff
    return (a - b.reindex(a.index)).rename('value')
