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

import numpy as np
import pandas as pd

from behav_utils.data.ops.selection import select_sessions
from behav_utils.data.ops.filtering import filter_trials, opto_mask
from behav_utils.analysis.psychometry import compute_psychometric
from behav_utils.analysis.update_matrix import compute_um
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


def compute_phase(animal: AnimalData, phase, cohort=None):
    """Pooled psychometric + update matrix per condition for one phase (report assembler).

    Returns three dicts keyed by condition label: clean (filtered sessions), psyc, um.
    A condition with no surviving sessions is None in all three.
    """
    if cohort is None:
        cohort = 'opto' if is_opto_cohort(animal) else 'non-opto'
    panels = PANELS[cohort][phase]

    clean, psyc, um = {}, {}, {}
    for label, spec in panels.items():
        sessions = filter_phase(animal, **spec)
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
