"""
analysis/opto.py — opto-effect statistics.

Three producers feed the opto analysis, all stat-agnostic (summary stats and
psychometric curve params flow through one frame):

  extract_opto_estimates  -> one value per (animal, condition, stat), pooled
                             WITHIN animal, via the estimates-table layer
                             (behav_utils.analysis.extract_stats). Feeds the
                             per-animal tests and the Δ comparison (paired_diff /
                             rank_test). Unit of inference is the animal.
  compute_opto_trajectory -> one value per (animal, condition, stat, session).
                             Feeds the trajectory plots ONLY. These per-session
                             rows must NOT be fed into a test (pseudoreplication).
  compute_opto_comparisons -> per-animal pairwise psychometric comparison of two
                             conditions (wraps behav_utils.compare_conditions).

Conditions map to filter_phase trial-types:
    opto    -> 'opto'      (laser trials)
    nonopto -> 'opto_off'  (interleaved non-laser controls)
    post    -> 'post_opto' (first non-laser trial after each opto run)

The lag-1 summary stats stay correct on these subsets because prev_* are frozen,
abort/block-aware fields sliced with the trials (see analysis/phase.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List

from analysis.phase import filter_phase

DEFAULT_CONDITIONS: Dict[str, str] = {
    'opto': 'opto', 'nonopto': 'opto_off', 'post': 'post_opto'}
DEFAULT_STATS: List[str] = ['win_stay', 'lose_shift', 'recency',
                            'pse', 'slope', 'lapse']

# =============================================================================
# Light-artifact diagnostics (nb30b) — is the WT opto effect a stimulus-independent
# side bias (light-delivery artifact) or a stimulus-dependent decision change?
#
#   compute_choice_by_stimulus -> empirical P(choose B) by stimulus bin + hit/FA per
#                                 animal x condition (opto vs opto_off). Reuses the
#                                 library primitives filter_phase / pool_arrays / _bin_data.
#   compute_side_bias  (PRIMARY) -> net_bias, tail_delta, boundary_delta. The discriminator:
#                                 tail_delta != 0 & flat  => additive offset (artifact);
#                                 boundary_delta only      => horizontal/criterion shift.
#   compute_sdt      (SECONDARY) -> criterion c and d' per condition (+ deltas). Conventional
#                                 language only; carries the same horizontal-shift ambiguity,
#                                 so NOT the discriminator.
# =============================================================================

def _pool_condition(animal, phase, trial_type, session_type):
    """Pooled (stimuli, choices, categories) for one animal x phase x trial_type, via the
    settled filter_phase -> pool_arrays path. None if the selection is empty."""
    from behav_utils.data.ops.filtering import pool_arrays
    sessions = filter_phase(animal, phase, session_type, trial_type=trial_type)
    pooled = pool_arrays(sessions)
    if pooled.get('n_trials', 0) == 0:
        return None
    stim = np.asarray(pooled['stimuli'], float)
    ch = np.asarray(pooled['choices'], float)
    cat = np.asarray(pooled['categories'], float)
    valid = (~pooled['no_response']) & np.isfinite(stim) & np.isfinite(ch)
    if valid.sum() == 0:
        return None
    return stim[valid], ch[valid], cat[valid]


def compute_choice_by_stimulus(experiment, phase='uniform', animals=None,
                               trial_types=('opto', 'opto_off'), session_type='opto',
                               n_bins=8):
    """Empirical P(choose B) by stimulus bin, plus overall P(B) and hit/FA, per animal x
    condition. Model-free (no psychometric fit). Returns {'binned', 'rates', ...}.

    binned: [animal, genotype, condition, stim_centre, p_choose_b, n]
    rates : [animal, genotype, condition, p_b_overall, hit, fa, n, n_signal, n_noise]
            hit = P(choose B | category B), fa = P(choose B | category A).
    """
    from behav_utils.analysis.psychometry import _bin_data   # library binning primitive
    animals = list(animals) if animals is not None else list(experiment.animals)
    binned_rows, rate_rows = [], []
    for aid in animals:
        animal = experiment.animals[aid]
        geno = animal.genotype
        for tt in trial_types:
            pooled = _pool_condition(animal, phase, tt, session_type)
            if pooled is None:
                continue
            stim, ch, cat = pooled
            centres, means, counts = _bin_data(stim, ch, n_bins)
            for c, m, k in zip(centres, means, counts):
                binned_rows.append(dict(animal=aid, genotype=geno, condition=tt,
                                        stim_centre=float(c), p_choose_b=float(m), n=int(k)))
            sig, noise = cat == 1, cat == 0
            rate_rows.append(dict(
                animal=aid, genotype=geno, condition=tt,
                p_b_overall=float(np.mean(ch)),
                hit=float(np.mean(ch[sig])) if sig.any() else np.nan,
                fa=float(np.mean(ch[noise])) if noise.any() else np.nan,
                n=int(len(ch)), n_signal=int(sig.sum()), n_noise=int(noise.sum())))
    return {'binned': pd.DataFrame(binned_rows), 'rates': pd.DataFrame(rate_rows),
            'phase': phase, 'trial_types': tuple(trial_types)}


def compute_side_bias(result, a='opto', b='opto_off',
                      boundary_thresh=0.4, tail_thresh=0.6):
    """PRIMARY discriminator. Per animal, from the binned P(B):

        net_bias      = P(B|a) - P(B|b) overall            (raw choice offset, no binning)
        tail_delta    = mean ΔP(B) in |stim| >= tail_thresh (saturated bins)
        boundary_delta= mean ΔP(B) in |stim| <= boundary_thresh (near-threshold bins)

    Additive side bias (artifact): tail_delta and boundary_delta both non-zero (flat ΔP(B)).
    Criterion/boundary shift (real): boundary_delta only, tail_delta ~ 0.
    Returns [animal, genotype, net_bias, tail_delta, boundary_delta].
    """
    binned, rates = result['binned'], result['rates']
    if binned.empty or 'animal' not in binned.columns:
        return pd.DataFrame(columns=['animal', 'genotype', 'net_bias',
                                     'tail_delta', 'boundary_delta'])
    rows = []
    genos = binned.groupby('animal')['genotype'].first()
    for aid, g in genos.items():
        ba = binned[(binned.animal == aid) & (binned.condition == a)].set_index('stim_centre')['p_choose_b']
        bb = binned[(binned.animal == aid) & (binned.condition == b)].set_index('stim_centre')['p_choose_b']
        common = ba.index.intersection(bb.index)
        d = (ba.loc[common] - bb.loc[common])
        absc = np.abs(common.to_numpy())
        tail = d.to_numpy()[absc >= tail_thresh]
        boundary = d.to_numpy()[absc <= boundary_thresh]
        ra = rates[(rates.animal == aid) & (rates.condition == a)]['p_b_overall']
        rb = rates[(rates.animal == aid) & (rates.condition == b)]['p_b_overall']
        rows.append(dict(
            animal=aid, genotype=g,
            net_bias=float(ra.iloc[0] - rb.iloc[0]) if len(ra) and len(rb) else np.nan,
            tail_delta=float(np.nanmean(tail)) if len(tail) else np.nan,
            boundary_delta=float(np.nanmean(boundary)) if len(boundary) else np.nan))
    return pd.DataFrame(rows)


def compute_sdt(result, a='opto', b='opto_off'):
    """SECONDARY (descriptive). Signal-detection criterion c and sensitivity d' per animal x
    condition, from empirical hit/FA (log-linear / Hautus correction so 0/1 rates stay finite).
    d' = z(hit) - z(fa);  c = -0.5 (z(hit) + z(fa)).  Also Δ = a - b.

    NOT the discriminator: criterion c absorbs both a motor offset and a boundary move (same
    horizontal-shift ambiguity as the psychometric μ). `saturated` flags animals whose raw
    hit/fa hit 0 or 1 (d' there is imprecise). Assumes equal-variance Gaussian.
    """
    from scipy.stats import norm
    rates = result['rates']
    if rates.empty or 'animal' not in rates.columns:
        return pd.DataFrame(columns=['animal', 'genotype', 'd_dprime', 'd_c', 'saturated'])

    def _cd(hit, fa, n_s, n_n):
        h = (hit * n_s + 0.5) / (n_s + 1.0)     # Hautus 1995 log-linear
        f = (fa * n_n + 0.5) / (n_n + 1.0)
        zh, zf = norm.ppf(h), norm.ppf(f)
        return zh - zf, -0.5 * (zh + zf)

    rows = []
    for aid, g in rates.groupby('animal')['genotype'].first().items():
        sub = rates[rates.animal == aid]
        out = dict(animal=aid, genotype=g)
        vals, sat = {}, False
        for cond in (a, b):
            r = sub[sub.condition == cond]
            if not len(r) or not np.isfinite(r.hit.iloc[0]) or not np.isfinite(r.fa.iloc[0]):
                continue
            hit, fa = float(r.hit.iloc[0]), float(r.fa.iloc[0])
            sat = sat or hit in (0.0, 1.0) or fa in (0.0, 1.0)
            dprime, crit = _cd(hit, fa, r.n_signal.iloc[0], r.n_noise.iloc[0])
            vals[cond] = (dprime, crit)
            out[f'dprime_{cond}'], out[f'c_{cond}'] = dprime, crit
        if a in vals and b in vals:
            out['d_dprime'] = vals[a][0] - vals[b][0]
            out['d_c'] = vals[a][1] - vals[b][1]
        out['saturated'] = sat
        rows.append(out)
    return pd.DataFrame(rows)
