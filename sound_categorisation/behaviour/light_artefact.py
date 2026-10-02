"""
Light artefact — what the light does on its own, at each site.

Every session that carries light (PPC opto sessions, PPC sham sessions, ALM sessions) has light-on and
light-off trials randomised per trial, so light-on − light-off is a valid per-trial contrast on each.
For a WT animal (no opsin) every one of those contrasts is a pure light effect; for a HET animal the sham
and ALM contrasts are light effects and the PPC-opto contrast is light + inactivation. The light effect of
interest is a shift of the decision criterion with sensitivity intact: ``criterion`` moves, ``dprime``
does not (behav_utils.stats.sdt), and P(B) moves by a roughly constant amount across stimulus bins.

    sets = light_sets(animal, 'Uniform')                     # {'ppc_opto': [...], 'ppc_sham': [...], 'alm_uni': [...], ...}
    r = compute_light_artefact(animal, 'Uniform')            # LightArtefact
    r.contrasts['ppc_sham'].contrasts['on_vs_off'].table('trials')
    r.bins                                                   # P(B) per stimulus bin, on vs off, per set
    r.site_interaction                                       # ppc_sham − alm: is the light effect site-specific?

Session sets are named by site + what else is on: ``ppc_opto`` (laser), ``ppc_sham`` (blue light, no
laser — the 'masking' session type), ``alm_uni`` / ``alm_bi`` (laser at ALM). The report
(``reports/light_artefact.py``) writes these per animal and folds them across genotypes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from behav_utils import TrialArrays, compute_delta_stat, compute_interaction, filter_trials, select_sessions
from behav_utils.analysis.comparison import DeltaStats, Interaction
from behav_utils.readouts import PsychometricCurve, compute_binned_curve, compute_psychometric_curve
from behav_utils.stats import PSYCHOMETRIC, SDT

from sound_categorisation.data.cohort import SITE_TYPE

LIGHT_STATS = ('side_bias', 'accuracy', *PSYCHOMETRIC[:2], *SDT)     # side_bias, accuracy, mu, sigma, dprime, criterion
LIGHT_SETS = ('ppc_opto', 'ppc_sham', 'alm_uni', 'alm_bi')
KEY = 'on_vs_off'
N_BOOT, N_PERM, N_BINS = 1000, 1000, 8


def light_sets(animal, distribution: str) -> Dict[str, list]:
    """Every light-carrying session set at one distribution, by name; empty lists where absent."""
    return {
        'ppc_opto': select_sessions(animal, distribution=distribution, session_type='opto'),
        'ppc_sham': select_sessions(animal, distribution=distribution, session_type='masking'),
        'alm_uni': select_sessions(animal, distribution=distribution, session_type=SITE_TYPE['uni']),
        'alm_bi': select_sessions(animal, distribution=distribution, session_type=SITE_TYPE['bi']),
    }


@dataclass(frozen=True)
class LightArtefact:
    animal: str
    distribution: str
    contrasts: Dict[str, DeltaStats] = field(default_factory=dict)       # set name -> on vs off
    curves: Dict[str, Dict[str, PsychometricCurve]] = field(default_factory=dict)   # set -> {'on': curve, 'off': curve}
    bins: pd.DataFrame = field(default_factory=pd.DataFrame)            # set, bin, centre, n_on, n_off, p_on, p_off, diff, ci_lo, ci_hi
    site_interaction: Interaction | None = None                         # ppc_sham − alm (first ALM set present)
    n_sessions: Dict[str, int] = field(default_factory=dict)


def _bin_rows(on: TrialArrays, off: TrialArrays, name: str, n_boot: int, rng, n_bins: int = N_BINS) -> list:
    """P(B) per stimulus bin for on and off trials, with a percentile CI on the per-bin difference from
    resampling trials within each bin (valid: light was randomised per trial)."""
    c_on = compute_binned_curve(on, n_bins=n_bins)
    v_on, v_off = on.valid(), off.valid()
    edges = np.quantile(np.concatenate([v_on.stimulus, v_off.stimulus]), np.linspace(0, 1, n_bins + 1))

    def in_bin(v, lo, hi, last):
        return (v.stimulus >= lo) & ((v.stimulus <= hi) if last else (v.stimulus < hi))

    rows = []
    for b in range(n_bins):
        lo, hi, last = edges[b], edges[b + 1], b == n_bins - 1
        ch_on, ch_off = v_on.choice[in_bin(v_on, lo, hi, last)], v_off.choice[in_bin(v_off, lo, hi, last)]
        diff = float(ch_on.mean() - ch_off.mean()) if len(ch_on) and len(ch_off) else np.nan
        ci_lo = ci_hi = np.nan
        if n_boot and len(ch_on) >= 5 and len(ch_off) >= 5:
            d = (rng.choice(ch_on, (n_boot, len(ch_on))).mean(1) - rng.choice(ch_off, (n_boot, len(ch_off))).mean(1))
            ci_lo, ci_hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        rows.append({'set': name, 'bin': b, 'centre': float(c_on.centres[b]) if b < c_on.n_bins else np.nan,
                     'n_on': int(len(ch_on)), 'n_off': int(len(ch_off)),
                     'p_on': float(ch_on.mean()) if len(ch_on) else np.nan,
                     'p_off': float(ch_off.mean()) if len(ch_off) else np.nan,
                     'diff': diff, 'ci_lo': ci_lo, 'ci_hi': ci_hi})
    return rows


def compute_light_artefact(animal, distribution: str, *, names: Sequence[str] = LIGHT_STATS,
                           n_boot: int = N_BOOT, n_perm: int = N_PERM, units: Sequence[str] = ('trials', 'sessions'),
                           curve_boot: int = 200, seed: int = 0) -> LightArtefact:
    """Light-on vs light-off on every light-carrying session set the animal has at this distribution.
    ``curve_boot`` is the band of the on/off psychometric curves drawn on the pages (0 = no band)."""
    rng = np.random.default_rng(seed)
    sets = light_sets(animal, distribution)
    contrasts, curves, bins, n_sessions = {}, {}, [], {}
    for name in LIGHT_SETS:
        sessions = sets[name]
        if not sessions:
            continue
        on, off = filter_trials(sessions, trial_type='opto'), filter_trials(sessions, trial_type='non_opto')
        if not on or not off:
            continue
        a_on, a_off = TrialArrays.from_sessions(on), TrialArrays.from_sessions(off)
        if min(a_on.n_responded, a_off.n_responded) < 20:
            continue
        contrasts[name] = compute_delta_stat({'off': off, 'on': on}, list(names), reference='off', curve=False,
                                             n_permutations=n_perm, n_bootstrap=n_boot, units=list(units))
        curves[name] = {'on': compute_psychometric_curve(a_on, n_bootstrap=curve_boot),
                        'off': compute_psychometric_curve(a_off, n_bootstrap=curve_boot)}
        bins += _bin_rows(a_on, a_off, name, n_boot if n_boot else 0, rng)
        n_sessions[name] = len(sessions)
    interaction = None
    alm = next((k for k in ('alm_bi', 'alm_uni') if k in contrasts), None)
    if n_boot and 'ppc_sham' in contrasts and alm:
        interaction = compute_interaction(contrasts['ppc_sham'], contrasts[alm], KEY, label_a='ppc_sham', label_b=alm)
    return LightArtefact(animal.animal_id, distribution, contrasts, curves,
                         pd.DataFrame(bins, columns=['set', 'bin', 'centre', 'n_on', 'n_off', 'p_on', 'p_off',
                                                     'diff', 'ci_lo', 'ci_hi']),
                         interaction, n_sessions)
