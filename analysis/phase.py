"""Phase-level behavioural analysis for the sound-categorisation task.

A *phase* is a (distribution-stage, opto-trial-spec) condition — e.g. ``uniform_opto``
with trial_spec ``opto``. For each requested phase this module selects and cleans the
relevant sessions, then computes the pooled psychometric curve, the update matrix, and
per-session summary statistics.

Everything is built on the canonical behav_utils primitives — ``select_sessions``,
``filter_trials``, ``pool_arrays``, ``compute_summary_stats``, ``fit_psychometric`` and
``fit_update_matrix`` — so this stays a thin, project-specific orchestration layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from behav_utils import select_sessions, filter_trials, pool_arrays, fit_psychometric
from behav_utils.analysis.update_matrix import fit_update_matrix
from behav_utils.analysis.summary_stats import compute_summary_stats
from behav_utils.data.structures import AnimalData

PHASES = ('uniform_training_last5', 'uniform_masking', 'uniform_opto',
          'hard_a_masking', 'hard_a_opto', 'hard_b_masking', 'hard_b_opto')
TRIAL_SPECS = ('all', 'non_opto', 'opto', 'post_opto')

DEFAULT_STATS = ('accuracy', 'win_stay', 'lose_shift', 'recency')


@dataclass
class Phase:
    """One condition plus the analysis filled in by :func:`calculate_phase`.

    The first four fields describe the condition; the last four are populated in place
    (left as ``None`` until then). Read results by name, e.g. ``p.psyc_fit``.
    """
    phase: str
    trial_spec: str = 'all'
    label: str = ''
    color: Optional[str] = None

    pooled: Optional[dict] = None        # pool_arrays / downsample output
    psyc_fit: Optional[dict] = None      # fit_psychometric result
    um_fit: Optional[tuple] = None       # (update_matrix, conditional_matrix, info)
    stats: Optional[dict] = None         # {stat_name: array over sessions, 'n_trials': ...}


def clean_sessions(animal: AnimalData, phase: str, trial_spec: Optional[str] = None,
                   min_accuracy: float = 0.6, min_trials: int = 10,
                   exclude_abort: bool = True):
    """Select sessions for ``phase`` and filter their trials by ``trial_spec``."""
    sessions = select_sessions(animal, phase, min_accuracy=min_accuracy)
    return filter_trials(sessions, min_trials=min_trials,
                         trial_type=trial_spec, exclude_abort=exclude_abort)


def fit_um_psych(pooled: dict, n_bootstrap: int = 200):
    """Psychometric fit + update matrix from a pooled-arrays dict."""
    psyc_fit = fit_psychometric(pooled['stimuli'], pooled['choices'], n_bootstrap=n_bootstrap)
    um_fit = fit_update_matrix(
        pooled['stimuli'], pooled['choices'], pooled['categories'],
        trial_filter='post_correct',
        prev_stimuli=pooled['prev_stimuli'],
        prev_choices=pooled['prev_choices'],
        prev_categories=pooled['prev_categories'],
        no_response=pooled['no_response'],
        not_blockstart=pooled['prev_has_prev'],
    )
    return psyc_fit, um_fit


def calculate_stats(sessions, stat_names=DEFAULT_STATS, mode: str = 'per_session') -> dict:
    """Summary stats for ``sessions``.

    ``per_session`` (default) returns ``{stat: array over sessions, 'n_trials': array}``;
    ``pooled`` returns ``{stat: single pooled value, 'n_trials': int}``.
    Returns ``{}`` if no sessions survived the filter.
    """
    stat_names = list(stat_names)
    result = compute_summary_stats(sessions, stat_names=stat_names, mode=mode)

    if mode == 'pooled':
        out = dict(result['stats'])
        out['n_trials'] = result['n_trials']
        return out

    per = result['per_session']
    if not per:
        return {}
    out = {key: np.array([s['stats'][key] for s in per]) for key in per[0]['stats']}
    out['n_trials'] = np.array([s['n_trials'] for s in per])
    return out


def calculate_min_n_per_phase(animal: AnimalData, phases=None, trial_spec=None,
                              phase_trial_comb=None, min_accuracy: float = 0.6,
                              exclude_abort: bool = True, verbose: bool = True):
    """Minimum pooled trial count across the requested phases (for matched downsampling).

    Pass either ``phases`` (+ optional ``trial_spec``) or ``phase_trial_comb`` — a list of
    ``(phase, trial_spec)`` pairs, which takes precedence.
    """
    if isinstance(phases, str):
        phases = [phases]
    if isinstance(trial_spec, str) or trial_spec is None:
        trial_spec = [trial_spec]

    if phase_trial_comb is None and phases is None:
        raise ValueError("Either `phases` or `phase_trial_comb` must be provided.")

    if phase_trial_comb is not None:
        pairs = phase_trial_comb
    else:
        pairs = [(phase, spec) for phase in phases for spec in trial_spec]

    total_n: dict = {}
    min_n = np.inf
    for pair in pairs:
        phase, spec = pair[0], pair[1]
        sessions = select_sessions(animal, phase, min_accuracy=min_accuracy)
        clean = filter_trials(sessions, min_trials=10, trial_type=spec, exclude_abort=exclude_abort)
        n = sum(sess.n_trials for sess in clean)
        total_n.setdefault(phase, {})[spec] = n
        if verbose:
            print(f"{phase} - {spec}:\t {len(clean)} sessions, {n} trials")
        min_n = min(min_n, n)

    if verbose:
        print(f"\nMinimum n across phases: {int(min_n)}")
    return int(min_n), total_n


def downsample(sessions, n_sample: int = 1000, seed: Optional[int] = None,
               with_replacement: bool = False, n_bins: int = 8) -> dict:
    """Pool ``sessions`` and subsample to ~``n_sample`` trials.

    Stratified by stimulus bin (proportional allocation), so the stimulus marginal is
    preserved. Note rows are grouped by bin, so only order-independent / lag-1 stats stay
    valid on the result; ``session_boundaries`` is invalidated and set to ``None``.
    """
    pooled = pool_arrays(sessions)
    rng = np.random.default_rng(seed)

    edges = np.linspace(-1, 1, n_bins + 1)[1:-1]
    bin_id = np.digitize(pooled['stimuli'], edges)            # 0 .. n_bins-1

    keep = []
    for b in range(n_bins):
        in_bin = np.where(bin_id == b)[0]
        k = min(round(n_sample * len(in_bin) / pooled['n_trials']), len(in_bin))
        if k > 0:
            keep.append(rng.choice(in_bin, k, replace=with_replacement))
    idx = np.concatenate(keep) if keep else np.array([], dtype=int)

    for key, value in list(pooled.items()):
        if key == 'n_trials':
            pooled[key] = len(idx)                # actual kept count, not the request
        elif key == 'n_sessions':
            continue
        elif key == 'session_boundaries':
            pooled[key] = None                    # invalid after bin-reordering
        else:
            pooled[key] = value[idx]
    return pooled


def calculate_phase(animal: AnimalData, phases: List[Phase], min_accuracy: float = 0.6,
                    min_trials: int = 10, exclude_abort: bool = True,
                    stat_names=DEFAULT_STATS, down_sample: bool = False,
                    n_sample_downsample: Optional[int] = None,
                    seed_downsample: Optional[int] = None,
                    downsample_with_replacement: bool = False) -> List[Phase]:
    """Fill each :class:`Phase` in ``phases`` with pooled data, psychometric fit, update
    matrix and per-session stats. Mutates and returns the same list.

    If ``down_sample`` and ``n_sample_downsample`` is ``None``, every phase is matched to
    the smallest phase's trial count.
    """
    for p in phases:
        if p.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {p.phase!r}")
        if p.trial_spec not in TRIAL_SPECS:
            raise ValueError(f"trial_spec must be one of {TRIAL_SPECS}, got {p.trial_spec!r}")

    n_min = None
    if down_sample:
        if n_sample_downsample is not None:
            n_min = n_sample_downsample
        else:
            pairs = [(p.phase, p.trial_spec) for p in phases]
            n_min, _ = calculate_min_n_per_phase(
                animal, phase_trial_comb=pairs, min_accuracy=min_accuracy,
                exclude_abort=exclude_abort, verbose=False)

    for p in phases:
        clean = clean_sessions(animal, p.phase, p.trial_spec, min_accuracy=min_accuracy,
                               min_trials=min_trials, exclude_abort=exclude_abort)
        p.stats = calculate_stats(clean, stat_names=stat_names, mode='per_session')
        if not clean:                            # nothing survived; leave fits as None
            continue
        p.pooled = (downsample(clean, n_sample=n_min, seed=seed_downsample,
                               with_replacement=downsample_with_replacement)
                    if down_sample else pool_arrays(clean))
        p.psyc_fit, p.um_fit = fit_um_psych(p.pooled, n_bootstrap=200)

    return phases
