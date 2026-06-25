"""
analysis/opto.py — opto-effect statistics.

Two producers feed the opto analysis, both stat-agnostic (summary stats and
psychometric curve params flow through one frame):

  compute_opto_stats      -> one value per (animal, condition, stat), pooled
                             WITHIN animal. Feeds the per-animal tests and the
                             Δ comparison. Unit of inference is the animal.
  compute_opto_trajectory -> one value per (animal, condition, stat, session).
                             Feeds the trajectory plots ONLY. These per-session
                             rows must NOT be fed into a test (pseudoreplication).

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
from typing import Dict, List, Optional, Sequence, Union, TYPE_CHECKING

from analysis.phase import filter_phase, is_opto_cohort, MIN_TRIALS
from behav_utils.analysis.summary_stats import compute_summary_stats
from behav_utils.analysis.psychometry import compute_psychometric
from behav_utils.analysis.stats_table import extract_stats

if TYPE_CHECKING:
    from behav_utils.data.structures import ExperimentData


# Curve-param stat name -> key in compute_psychometric()['params'].
_CURVE_KEY = {'pse': 'mu', 'slope': 'sigma',
              'lapse_low': 'lapse_low', 'lapse_high': 'lapse_high'}
_CURVE_STATS = set(_CURVE_KEY) | {'lapse'}        # 'lapse' = mean(low, high)

DEFAULT_CONDITIONS: Dict[str, str] = {
    'opto': 'opto', 'nonopto': 'opto_off', 'post': 'post_opto'}
DEFAULT_STATS: List[str] = ['win_stay', 'lose_shift', 'recency',
                            'pse', 'slope', 'lapse']


def _norm_genotype(g: Optional[str]) -> Optional[str]:
    if g is None:
        return None
    g = str(g).strip().lower()
    if g in ('het', 'heterozygous', 'hemizygous'):
        return 'het'
    if g in ('wt', 'wildtype', 'wild-type', 'wild type'):
        return 'wt'
    return g                                       # unknown -> keep visible


def _curve_point(params: dict, name: str) -> float:
    """Scalar value of one curve param from a params dict (no CI)."""
    if name == 'lapse':
        return 0.5 * (params['lapse_low'] + params['lapse_high'])
    return params[_CURVE_KEY[name]]


def _curve_value(psy: dict, name: str):
    """(value, ci_lo, ci_hi) for one curve param from a pooled compute_psychometric result."""
    params, ci = psy['params'], psy['params_ci']
    v = _curve_point(params, name)
    if name == 'lapse':
        clo = min(ci['lapse_low'][0], ci['lapse_high'][0])
        chi = max(ci['lapse_low'][1], ci['lapse_high'][1])
    else:
        clo, chi = ci[_CURVE_KEY[name]]
    return v, clo, chi


def _split_stats(stats):
    stats = list(stats or DEFAULT_STATS)
    curve = [s for s in stats if s in _CURVE_STATS]
    summary = [s for s in stats if s not in _CURVE_STATS]
    return summary, curve


def _opto_animals(experiment, animals):
    if animals is not None:
        return animals
    return [aid for aid, a in experiment.animals.items() if is_opto_cohort(a)]


def _filter_phases(animal, phase, session_type, trial_type, min_trials):
    """filter_phase for one phase or a session-level pool of several.

    phase as a list (e.g. ['hard_a', 'hard_b']) concatenates each phase's
    sessions BEFORE any stat is computed, so the pooled value is properly
    trial-weighted. Pool only stats that are invariant to the mirror between
    Hard-A and Hard-B (recency, win/lose history); lateralised curve params
    (pse, lapse) cancel across the mirror and should be read per phase.
    """
    phases = [phase] if isinstance(phase, str) else list(phase)
    sessions = []
    for p in phases:
        sessions += filter_phase(animal, p, session_type,
                                 trial_type=trial_type, min_trials=min_trials)
    return sessions


def extract_opto_estimates(
    experiment: 'ExperimentData',
    *,
    phases: Union[str, List[str]] = 'uniform',
    stats: Sequence[str],
    animals: Optional[List[str]] = None,
    trial_types: Sequence[str] = ('opto', 'opto_off'),
    session_type: str = 'opto',
    n_boot: int = 0,
    ci: float = 0.95,
    exclude_abort: bool = True,
    min_trials: int = MIN_TRIALS,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-animal stat estimates for the opto cohort, via filter_phase -> extract_stats.

    The new-layer replacement for :func:`compute_opto_stats`. For each
    animal x phase x trial_type: select the opto sessions and the laser condition
    (:func:`filter_phase`), pool within the animal (:func:`extract_stats`), and
    stack the per-animal estimate rows into one long frame — the
    :attr:`StatTable.estimates` schema, ready for the Tier-B verbs.

    No differencing or testing happens here: the caller runs ``paired_diff`` /
    ``rank_test`` (:mod:`behav_utils.analysis.group`) on the result. ``distribution``,
    ``session_type`` and ``trial_type`` ride along automatically from filter_phase's
    provenance; ``genotype`` is attached from each animal (normalised to het/wt).

    Phases are kept SEPARATE — each is a single-phase filter_phase call, distinguished
    by the ``distribution`` column. This never pools Hard-A and Hard-B (which would
    presuppose a mirror symmetry that has not been shown); pass them as separate
    phases and contrast per phase downstream.

    Args:
        experiment:   ExperimentData (genotype read from each animal's metadata).
        phases:       phase key or list ('uniform' | 'hard_a' | 'hard_b').
        stats:        summary-stat names, passed straight to extract_stats.
                      ``psychometric`` expands to mu/sigma/lapse_low/lapse_high;
                      rename mu->pse, sigma->slope at the call site if wanted.
        animals:      animal ids; None resolves to the whole opto cohort.
        trial_types:  laser conditions to pull, each a filter_phase trial_type
                      ('opto' | 'opto_off' | 'post_opto'). Note these are the
                      filter_phase names, not the old 'nonopto'/'post' labels.
        session_type: phase session_type to select (default 'opto').
        n_boot:       within-animal bootstrap reps -> ci_lo_within / ci_hi_within.
                      Only the curve-param QC consumes these, so pass 0 for the
                      history stats and a positive value only for ['psychometric'].
        ci:           CI level for the within-animal bootstrap.
        exclude_abort, min_trials: passed to filter_phase.
        seed:         bootstrap seed.

    Returns:
        A long DataFrame concatenated across animal x phase x trial_type, or an
        empty DataFrame if nothing survived the selection.
    """
    if isinstance(phases, str):
        phases = [phases]
    if isinstance(trial_types, str):
        trial_types = [trial_types]
    animals = _opto_animals(experiment, animals)

    frames = []
    for aid in animals:
        animal = experiment.animals[aid]
        genotype = _norm_genotype(animal.genotype)
        for phase in phases:
            for trial_type in trial_types:
                sessions = filter_phase(animal, phase, session_type,
                                        trial_type=trial_type,
                                        exclude_abort=exclude_abort,
                                        min_trials=min_trials)
                if not sessions:
                    continue
                stat_table = extract_stats(
                    sessions, animal_id=aid, stats=stats, mode='pooled',
                    n_boot=n_boot, ci=ci, seed=seed,
                    meta={'genotype': genotype})
                frames.append(stat_table.estimates)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_opto_stats(
    experiment: 'ExperimentData',
    phase: Union[str, List[str]] = 'uniform',
    stats: Optional[List[str]] = None,
    conditions: Optional[Dict[str, str]] = None,
    session_type: str = 'opto',
    min_trials: int = 10,
    n_bootstrap: int = 1000,
    animals: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Aggregate readouts S to the animal level (pooled within animal).

    Returns a tidy DataFrame, one row per (animal, condition, stat): animal,
    genotype, condition, stat, value, ci_lo, ci_hi, n_trials, n_sessions.
    ci_lo/ci_hi are NaN for summary stats and the bootstrap CI for curve params
    (the GoF / reliability gate).
    """
    conditions = conditions or DEFAULT_CONDITIONS
    summary_stats, curve_stats = _split_stats(stats)
    animals = _opto_animals(experiment, animals)

    rows = []
    for aid in animals:
        animal = experiment.animals[aid]
        geno = _norm_genotype(animal.genotype)
        for cond_label, trial_type in conditions.items():
            sessions = _filter_phases(animal, phase, session_type,
                                      trial_type=trial_type, min_trials=min_trials)
            n_sess = len(sessions)
            n_tr = sum(s.trials.n_trials for s in sessions)

            svals = {}
            if summary_stats and sessions:
                svals = compute_summary_stats(
                    sessions, stat_names=summary_stats, mode='pooled').get('stats', {})
            for s in summary_stats:
                rows.append(dict(animal=aid, genotype=geno, condition=cond_label,
                                 stat=s, value=svals.get(s, np.nan),
                                 ci_lo=np.nan, ci_hi=np.nan,
                                 n_trials=n_tr, n_sessions=n_sess))

            psy = None
            if curve_stats and sessions:
                psy = compute_psychometric(sessions, mode='pooled',
                                           n_bootstrap=n_bootstrap)
            ok = bool(psy and psy.get('success'))
            for s in curve_stats:
                v, clo, chi = _curve_value(psy, s) if ok else (np.nan, np.nan, np.nan)
                rows.append(dict(animal=aid, genotype=geno, condition=cond_label,
                                 stat=s, value=v, ci_lo=clo, ci_hi=chi,
                                 n_trials=n_tr, n_sessions=n_sess))

    return pd.DataFrame(rows)


def compute_opto_trajectory(
    experiment: 'ExperimentData',
    phase: Union[str, List[str]] = 'uniform',
    stats: Optional[List[str]] = None,
    conditions: Optional[Dict[str, str]] = None,
    session_type: str = 'opto',
    min_trials: int = 10,
    n_bootstrap: int = 200,
    animals: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Per-session values for the trajectory plots.

    Returns a tidy DataFrame, one row per (animal, condition, stat, session):
    animal, genotype, condition, stat, session_idx, session_id, value, n_trials,
    success. No per-session CI (only the pooled fit carries one).

    NOTE: curve stats (pse/slope/lapse) are fitted per session on the opto subset
    (~tens of trials, sparse at informative levels), so many sessions fail or rail
    — the `success` column flags these. Curve-stat trajectories are a diagnostic
    only; the summary-stat trajectories (recency, win_stay, …) are the reliable
    ones. The tested quantities remain the pooled values from compute_opto_stats.
    """
    conditions = conditions or DEFAULT_CONDITIONS
    summary_stats, curve_stats = _split_stats(stats)
    animals = _opto_animals(experiment, animals)

    rows = []
    for aid in animals:
        animal = experiment.animals[aid]
        geno = _norm_genotype(animal.genotype)
        for cond_label, trial_type in conditions.items():
            sessions = _filter_phases(animal, phase, session_type,
                                      trial_type=trial_type, min_trials=min_trials)
            if not sessions:
                continue

            if summary_stats:
                res = compute_summary_stats(sessions, stat_names=summary_stats,
                                            mode='per_session')
                for entry in res.get('per_session', []):
                    svals = entry.get('stats', {})
                    for s in summary_stats:
                        rows.append(dict(
                            animal=aid, genotype=geno, condition=cond_label, stat=s,
                            session_idx=entry.get('session_idx'),
                            session_id=entry.get('session_id'),
                            value=svals.get(s, np.nan),
                            n_trials=entry.get('n_trials'), success=True))

            if curve_stats:
                res = compute_psychometric(sessions, mode='per_session',
                                           n_bootstrap=n_bootstrap)
                for entry in res.get('per_session', []):
                    ok = bool(entry.get('success', False))
                    params = entry.get('params', {})
                    for s in curve_stats:
                        v = _curve_point(params, s) if (ok and params) else np.nan
                        rows.append(dict(
                            animal=aid, genotype=geno, condition=cond_label, stat=s,
                            session_idx=entry.get('session_idx'),
                            session_id=entry.get('session_id'),
                            value=v,
                            n_trials=entry.get('n_trials'), success=ok))

    return pd.DataFrame(rows)


def compute_opto_delta(stats_df: pd.DataFrame,
                       opto: str = 'opto', nonopto: str = 'nonopto') -> pd.DataFrame:
    """Per-animal Δ = value[opto] − value[nonopto], one row per (animal, stat).

    Returns: animal, genotype, stat, opto, nonopto, delta. This per-animal effect
    is what the genotype comparison (HET-Δ vs WT-Δ) tests.
    """
    wide = (stats_df.pivot_table(index=['animal', 'genotype', 'stat'],
                                 columns='condition', values='value')
            .rename_axis(columns=None).reset_index())
    for c in (opto, nonopto):
        if c not in wide.columns:
            have = [x for x in wide.columns if x not in ('animal', 'genotype', 'stat')]
            raise ValueError(f"condition '{c}' not in stats_df (have {have})")
    wide['delta'] = wide[opto] - wide[nonopto]
    return wide[['animal', 'genotype', 'stat', opto, nonopto, 'delta']]


def compute_opto_comparisons(
    experiment: 'ExperimentData',
    phase: Union[str, List[str]] = 'uniform',
    cond_a: str = 'opto',
    cond_b: str = 'nonopto',
    conditions: Optional[Dict[str, str]] = None,
    session_type: str = 'opto',
    min_trials: int = 10,
    n_permutations: int = 1000,
    n_bootstrap: int = 1000,
    animals: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """Per-animal psychometric comparison of two conditions.

    Thin wrapper over behav_utils.compare_conditions: pools each condition's
    sessions and compares the curves. Returns {animal: result}, where result is
    the compare_conditions dict (params_a/b with mu=PSE, sigma=slope; perm_p on
    the param differences; boot bands) — feed straight to
    behav_utils.plotting.plot_comparison.

    cond_a/cond_b index into `conditions` (default DEFAULT_CONDITIONS): 'opto' vs
    'nonopto' is the pse diagnostic; 'opto' vs 'post' is the recovery curve.
    """
    from behav_utils.data.ops.filtering import pool_arrays
    from behav_utils.analysis.comparison import compare_conditions

    conditions = conditions or DEFAULT_CONDITIONS
    animals = _opto_animals(experiment, animals)
    tt_a, tt_b = conditions[cond_a], conditions[cond_b]

    out = {}
    for aid in animals:
        animal = experiment.animals[aid]
        sa = _filter_phases(animal, phase, session_type, trial_type=tt_a, min_trials=min_trials)
        sb = _filter_phases(animal, phase, session_type, trial_type=tt_b, min_trials=min_trials)
        if not sa or not sb:
            continue
        a, b = pool_arrays(sa), pool_arrays(sb)
        out[aid] = compare_conditions(
            a['stimuli'], a['choices'], a['categories'],
            b['stimuli'], b['choices'], b['categories'],
            n_permutations=n_permutations, n_bootstrap=n_bootstrap,
            label_a=cond_a, label_b=cond_b,
        )
    return out
