"""Phase-level behavioural analysis for the sound-categorisation task.

New architecture:
  filter_phase(animal, dist, session_type, trial_type, ...)  → flat [SessionData]
      select_sessions + filter_trials in one call. Feed to compute_psychometric /
      compute_um, or to compute_ds_x (behav_utils.analysis.downsample) for matched-n.

`compute_phase` is the report's per-condition assembler, now built on filter_phase via the
PANELS specs — it produces the same {clean, psyc, um} dicts the plot helpers consume.
`downsample_phase` wraps compute_ds_x.
"""
from __future__ import annotations

from collections import OrderedDict
import warnings

import numpy as np
import pandas as pd

from behav_utils.data.ops.selection import select_sessions
from behav_utils.data.ops.filtering import filter_trials, opto_mask
from behav_utils.analysis.psychometry import compute_psychometric
from behav_utils.analysis.update_matrix import compute_um
from behav_utils.analysis.summary_stats import compute_summary_stats
from behav_utils.analysis.downsample import calculate_min_n, compute_ds_x
from behav_utils.analysis.session_features import compute_session_features
from behav_utils.data.structures import AnimalData

MIN_TRIALS = 10
N_BOOTSTRAP = 200
PHASE_ORDER = ['uniform', 'hard_a', 'hard_b']

_DIST = {'uniform': 'Uniform', 'hard_a': 'Hard-A', 'hard_b': 'Hard-B'}
# trial_type -> opto_mask delta ('all'/None handled separately as the all-valid mask)
_TRIAL_DELTA = {'opto': 0, 'opto_off': 'control', 'post_opto': 1}


def filter_phase(animal: AnimalData, dist, session_type, trial_type=None,
                 min_accuracy=None, exclude_abort=True, min_trials=MIN_TRIALS,
                 stage=None, last_n=None, first_n=None, last_fraction=None):
    """Select a phase's sessions and filter its trials in one step → [SessionData].

    Args:
        animal:        AnimalData.
        dist:          'uniform' | 'hard_a' | 'hard_b'.
        session_type:  'regular' | 'masking' | 'opto' | 'washout'.
        trial_type:    None/'all' (all valid trials, incl. laser) | 'opto' (laser) |
                       'opto_off' (interleaved non-laser controls) | 'post_opto'
                       (first non-laser trial after each opto run).
        min_accuracy:  session-quality floor (None = no constraint).
        exclude_abort: only affects the 'all'/None mask; the opto masks always drop aborts.
        min_trials:    drop sessions below this many surviving trials.
        stage / last_n / first_n / last_fraction:
                       passed through to the session selection (None = no constraint).

    Note: trial_type other than 'all'/None only isolates trials in opto sessions; on a
    regular session 'opto' returns an empty selection rather than erroring.
    """
    criteria = dict(distribution=_DIST[dist], session_type=session_type)
    for key, val in (('min_accuracy', min_accuracy), ('stage', stage),
                     ('last_n', last_n), ('first_n', first_n), ('last_fraction', last_fraction)):
        if val is not None:
            criteria[key] = val
    sessions = select_sessions(animal, **criteria)
    if not sessions:
        return []

    if trial_type in (None, 'all'):
        return filter_trials(sessions, exclude_opto=False, exclude_abort=exclude_abort,
                             min_trials=min_trials)
    if trial_type not in _TRIAL_DELTA:
        raise ValueError(f"trial_type must be None/'all'/'opto'/'opto_off'/'post_opto', "
                         f"got {trial_type!r}")
    delta = _TRIAL_DELTA[trial_type]
    return filter_trials(sessions, mask_fn=lambda s: opto_mask(s.trials, delta=delta),
                         min_trials=min_trials)


# ── Report panels: label -> filter_phase kwargs (faithful to the old presets) ────
def _opto_panels(dist):
    panels = []
    if dist == 'uniform':
        panels.append(('baseline', dict(dist='uniform', session_type='regular', trial_type=None,
                                         stage='Full_Task_Cont', last_n=5)))
    panels += [
        ('masking',   dict(dist=dist, session_type='masking', trial_type=None)),
        ('all_opto',  dict(dist=dist, session_type='opto', trial_type='all')),
        ('opto_off',  dict(dist=dist, session_type='opto', trial_type='opto_off')),
        ('opto_on',   dict(dist=dist, session_type='opto', trial_type='opto')),
        ('post_opto', dict(dist=dist, session_type='opto', trial_type='post_opto')),
    ]
    return OrderedDict(panels)


PANELS = {
    'opto': {d: _opto_panels(d) for d in PHASE_ORDER},
    'non-opto': {
        'uniform': OrderedDict([('baseline', dict(dist='uniform', session_type='regular',
                                                  trial_type=None, stage='Full_Task_Cont', last_n=5))]),
        'hard_a': OrderedDict([('regular', dict(dist='hard_a', session_type='regular', trial_type=None))]),
        'hard_b': OrderedDict([('regular', dict(dist='hard_b', session_type='regular', trial_type=None))]),
    },
}


def is_opto_cohort(animal: AnimalData) -> bool:
    """True if the animal has any opto or masking sessions."""
    return animal.session_table['session_type'].isin(['opto', 'masking']).any()


def compute_phase(animal: AnimalData, phase, cohort=None, min_accuracy=None):
    """Pooled psychometric + update matrix per condition for one phase (report assembler).

    Returns three dicts keyed by condition label: clean (filtered sessions), psyc, um.
    A condition with no surviving sessions is None in all three.

    min_accuracy is a session-quality floor applied to every panel, unless a panel's own
    spec sets its own min_accuracy (in which case the panel's value wins). None = no floor.
    """
    if cohort is None:
        cohort = 'opto' if is_opto_cohort(animal) else 'non-opto'
    panels = PANELS[cohort][phase]

    clean, psyc, um = {}, {}, {}
    for label, spec in panels.items():
        call = {'min_accuracy': min_accuracy, **spec}   # spec wins if it sets its own
        sessions = filter_phase(animal, **call)
        if not sessions:
            clean[label] = psyc[label] = um[label] = None
            continue
        clean[label] = sessions
        try:
            psyc[label] = compute_psychometric(sessions, mode='pooled', n_bins=8, n_bootstrap=N_BOOTSTRAP)
        except Exception:
            psyc[label] = None
        try:
            um[label] = compute_um(sessions)
        except Exception:
            um[label] = None
    return clean, psyc, um


def downsample_phase(clean_dict, n_repeats=100, n_bins=8, seed=42, with_replacement=True):
    """Matched-n psychometric + update matrix per condition (wraps compute_ds_x).

    Matches every condition to the smallest condition's trial count (psychometric) and pair
    count (UM), then runs compute_ds_x per condition and returns its aggregated result.
    Returns two dicts (psyc, um) keyed by condition label, plot-compatible.
    """
    phases = [s for s in clean_dict.values() if s]
    n_trials = calculate_min_n(phases, unit='trials')
    n_pairs = calculate_min_n(phases, unit='pairs')

    psyc, um = {}, {}
    for label, sessions in clean_dict.items():
        if not sessions:
            psyc[label] = um[label] = None
            continue
        psyc[label] = (compute_ds_x(sessions, 'psychometric', n_trials, n_repeats=n_repeats,
                                    with_replacement=with_replacement, n_bins=n_bins,
                                    seed=seed)['aggregated'] if n_trials > 0 else None)
        um[label] = (compute_ds_x(sessions, 'um', n_pairs, n_repeats=n_repeats,
                                  with_replacement=False, n_bins=n_bins,
                                  seed=seed)['aggregated'] if n_pairs > 0 else None)
    return psyc, um


def build_strip_df(animal: AnimalData, phases=None,
                   stats_of_interest=('accuracy', 'win_stay', 'lose_shift'), cohort=None):
    """Per-session DataFrame for strip plots: one row per session per condition."""
    if phases is None:
        phases = PHASE_ORDER
    rows = []
    for phase in phases:
        clean, _, _ = compute_phase(animal, phase, cohort)
        for condition, sessions in clean.items():
            if sessions is None:
                continue
            for sess in sessions:
                n_trials = int(sess.trials.valid_mask.sum())
                registered = [s for s in stats_of_interest if s != 'n_trials']
                features = compute_session_features(sess, stat_names=registered) if registered else {}
                row = {'phase': phase, 'condition': condition, 'n_trials': n_trials}
                for stat in stats_of_interest:
                    if stat != 'n_trials':
                        row[stat] = features.get(stat, np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Group-level aggregation (the group analog of compute_phase).
#
# Two averaging methods, presented as separate passes (not overlaid):
#   'pooled'   : pool every animal's sessions for a panel into one megasession
#                and compute once. Bands are TRIAL-LEVEL — descriptive only.
#   'per_mouse': compute per animal, then average across animals. The
#                psychometric band is BETWEEN-MOUSE SEM; the update matrix is the
#                element-wise mean of the per-mouse matrices.
# Both return {panel: result} in the same shape compute_phase produces, so the
# same plotters (plot_psychometric, plot_um) draw either pass unchanged.
# ─────────────────────────────────────────────────────────────────────────────

def _average_psychometric(results):
    """Grand-average psychometric from per-mouse compute_psychometric results.

    The bins and x-grid are fixed across calls, so per-mouse curves and binned
    points average directly. Returns a pooled-shape result whose curve_band is
    the BETWEEN-MOUSE SEM (mean ± SEM), with n_fits = number of mice. params are
    the mean of the per-mouse fit params (for annotation only — averaging sigmoid
    params is approximate). None if no mouse fitted.
    """
    results = [r for r in results if r and r.get('success')]
    if not results:
        return None
    x_fit = np.asarray(results[0]['x_fit'], float)
    Y = np.vstack([np.asarray(r['y_fit'], float) for r in results])     # (n_mice, grid)
    mean_y = np.nanmean(Y, axis=0)
    sem_y = (np.nanstd(Y, axis=0, ddof=1) / np.sqrt(len(results))
             if len(results) > 1 else np.zeros_like(mean_y))
    B = np.vstack([np.asarray(r['bin_means'], float) for r in results])  # (n_mice, n_bins)
    keys = ('mu', 'sigma', 'lapse_low', 'lapse_high')
    params = {k: float(np.nanmean([r['params'][k] for r in results])) for k in keys}
    return {
        'mode': 'pooled',
        'params': params,
        'params_ci': {k: (np.nan, np.nan) for k in keys},
        'curve_band': {'x': x_fit, 'median': mean_y,
                       'lo': mean_y - sem_y, 'hi': mean_y + sem_y},
        'bin_centres': np.asarray(results[0]['bin_centres'], float),
        'bin_means': np.nanmean(B, axis=0),
        'bin_counts': np.nansum(np.vstack([np.asarray(r['bin_counts'], float)
                                           for r in results]), axis=0),
        'x_fit': x_fit, 'y_fit': mean_y,
        'n_trials': int(np.nansum([r['n_trials'] for r in results])),
        'n_fits': len(results),
        'success': True,
    }


def _average_um(results):
    """Element-wise mean of per-mouse compute_um results (pooled shape)."""
    results = [r for r in results if r is not None]
    if not results:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)   # all-NaN cells
        um = np.nanmean(np.stack([np.asarray(r['um'], float) for r in results]), axis=0)
        cond = np.nanmean(np.stack([np.asarray(r['conditional_matrix'], float)
                                    for r in results]), axis=0)
    return {
        'mode': 'pooled', 'um': um, 'conditional_matrix': cond,
        'n_bins': results[0].get('n_bins', 8),
        'n_sessions': int(np.nansum([r.get('n_sessions', 0) for r in results])),
        'n_trials': int(np.nansum([r.get('n_trials', 0) for r in results])),
        'info': {'n_mice': len(results)},
    }


def compute_group_phase(experiment, phase, animals, method='per_mouse',
                        cohort='opto', min_accuracy=None, n_bootstrap=None):
    """Group psychometric + update matrix per condition, for one genotype group.

    Args:
        experiment: ExperimentData.
        phase: 'uniform' | 'hard_a' | 'hard_b'.
        animals: animal ids in the group (e.g. the het ids). Pass a single
            genotype — never mix het and wt.
        method: 'per_mouse' (compute per mouse, average across mice — the
            principled view, between-mouse SEM band) or 'pooled' (mega-pool all
            mice's trials, compute once — descriptive, trial-level band).
        cohort, min_accuracy: as compute_phase.
        n_bootstrap: bootstrap reps for the 'pooled' band (default N_BOOTSTRAP).
            Ignored by 'per_mouse', whose band is between-mouse SEM. The pooled
            band is descriptive, so a modest value is fine and much faster.

    Returns:
        (psyc, um): dicts keyed by panel label, each value in compute_phase's
        shape (or None if no data). Draw with plot_psychometric / plot_um.
    """
    if method not in ('per_mouse', 'pooled'):
        raise ValueError(f"method must be 'per_mouse' or 'pooled', got {method!r}")
    nboot = N_BOOTSTRAP if n_bootstrap is None else n_bootstrap
    panels = PANELS[cohort][phase]
    psyc, um = {}, {}
    for label, spec in panels.items():
        call = {'min_accuracy': min_accuracy, **spec}
        if method == 'pooled':
            sessions = []
            for aid in animals:
                sessions += filter_phase(experiment.animals[aid], **call)
            if not sessions:
                psyc[label] = um[label] = None
                continue
            try:
                psyc[label] = compute_psychometric(sessions, mode='pooled',
                                                   n_bins=8, n_bootstrap=nboot)
            except Exception:
                psyc[label] = None
            try:
                um[label] = compute_um(sessions)
            except Exception:
                um[label] = None
        else:  # per_mouse
            pm_psyc, pm_um = [], []
            for aid in animals:
                s = filter_phase(experiment.animals[aid], **call)
                if not s:
                    continue
                try:
                    pm_psyc.append(compute_psychometric(s, mode='pooled',
                                                        n_bins=8, n_bootstrap=0))
                except Exception:
                    pass
                try:
                    pm_um.append(compute_um(s))
                except Exception:
                    pass
            psyc[label] = _average_psychometric(pm_psyc)
            um[label] = _average_um(pm_um)
    return psyc, um


def compute_group_stats(experiment, phase, animals, method='per_mouse',
                        stats=('accuracy', 'win_stay', 'lose_shift', 'recency'),
                        cohort='opto', min_accuracy=None):
    """Group summary stats per condition (the data behind the group bar plots).

    Same two methods as compute_group_phase. Returns a tidy DataFrame:
        method='per_mouse' : one row per (animal, panel, stat) — box/strip these.
        method='pooled'    : one row per (panel, stat) on the megasession — a
                             single value per condition (no between-mouse spread).
    Both carry n_trials. Pass a single genotype.
    """
    if method not in ('per_mouse', 'pooled'):
        raise ValueError(f"method must be 'per_mouse' or 'pooled', got {method!r}")
    panels = PANELS[cohort][phase]
    stats = list(stats)
    rows = []
    for label, spec in panels.items():
        call = {'min_accuracy': min_accuracy, **spec}
        if method == 'pooled':
            sessions = []
            for aid in animals:
                sessions += filter_phase(experiment.animals[aid], **call)
            if not sessions:
                continue
            vals = compute_summary_stats(sessions, stat_names=stats,
                                         mode='pooled').get('stats', {})
            n_tr = sum(s.trials.n_trials for s in sessions)
            for st in stats:
                rows.append(dict(panel=label, stat=st, value=vals.get(st, np.nan),
                                 n_trials=n_tr))
        else:
            for aid in animals:
                s = filter_phase(experiment.animals[aid], **call)
                if not s:
                    continue
                vals = compute_summary_stats(s, stat_names=stats,
                                             mode='pooled').get('stats', {})
                n_tr = sum(ss.trials.n_trials for ss in s)
                for st in stats:
                    rows.append(dict(animal=aid, panel=label, stat=st,
                                     value=vals.get(st, np.nan), n_trials=n_tr))
    return pd.DataFrame(rows)
