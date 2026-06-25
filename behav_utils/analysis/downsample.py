"""Matched-n downsampling that returns SessionData, so compute_x is unchanged.

Pipeline:  filter -> calculate_min_n -> compute_ds_x  (or downsample -> compute_x).

`downsample` is `filter_trials` with a stratified n-subset selector instead of a
deterministic mask. It works because `prev_*` are stored frozen fields on TrialData and
`filter_trial_data` slices them with the rest, so a row-subset (or a with-replacement
resample, via repeated integer indices) keeps the lag-1 pairing intact. The clean sessions
are already abort/opto-cleared, so we slice with clear_flags=False — which also lets the
same path carry repeated indices for the bootstrap (with_replacement=True).

Two units: 'trials' (responded trials, for the psychometric) and 'pairs' (post-correct
pairs, for the update matrix — the rows fit_update_matrix actually uses).
"""
from __future__ import annotations

import warnings

import numpy as np

from behav_utils.data.structures import SessionData
from behav_utils.data.ops.filtering import pool_arrays, filter_trial_data
from behav_utils.analysis.psychometry import compute_psychometric, _PARAMS
from behav_utils.analysis.update_matrix import compute_um
from behav_utils.analysis.summary_stats import (
    fit_summary_stats, get_stat_names_expanded, is_exchangeable,
)


def _pair_base_mask(pooled) -> np.ndarray:
    """Rows fit_update_matrix counts as post-correct pairs (mirrors its session-path base)."""
    prev_choices = np.asarray(pooled['prev_choices'], dtype=float)
    prev_categories = np.asarray(pooled['prev_categories'], dtype=float)
    no_response = np.asarray(pooled['no_response'], dtype=bool)
    has_prev = np.asarray(pooled['prev_has_prev'], dtype=bool)
    return ((prev_choices == prev_categories) & (~no_response)
            & (~np.isnan(prev_choices)) & has_prev)


def _pool_index(pooled, unit) -> np.ndarray:
    """Global pooled-row indices eligible for the unit."""
    if unit == 'trials':
        return np.where(~np.asarray(pooled['no_response'], dtype=bool))[0]
    if unit == 'pairs':
        return np.where(_pair_base_mask(pooled))[0]
    raise ValueError(f"unit must be 'trials' or 'pairs', got {unit!r}")


def _draw(pooled, n, unit, n_bins, rng, replace) -> np.ndarray:
    """Stratified draw of ~n global pooled-row indices (stratified by stimulus / prev stimulus)."""
    idx_pool = _pool_index(pooled, unit)
    total = len(idx_pool)
    if total == 0:
        return np.array([], dtype=int)

    edges = np.linspace(-1, 1, n_bins + 1)
    strat_var = 'prev_stimuli' if unit == 'pairs' else 'stimuli'
    strat = np.clip(np.digitize(np.asarray(pooled[strat_var])[idx_pool], edges) - 1, 0, n_bins - 1)

    keep = []
    for b in range(n_bins):
        in_b = idx_pool[strat == b]
        if len(in_b) == 0:
            continue
        kb = round(n * len(in_b) / total)
        if not replace:
            kb = min(kb, len(in_b))
        if kb > 0:
            keep.append(rng.choice(in_b, kb, replace=replace))
    return np.concatenate(keep) if keep else np.array([], dtype=int)


def _slice_session(session: SessionData, local_idx: np.ndarray) -> SessionData:
    """New SessionData with this session's trials at local_idx (repeats allowed)."""
    new_trials = filter_trial_data(session.trials, local_idx, clear_flags=False)
    return SessionData(
        session_id=session.session_id, session_idx=session.session_idx,
        date=session.date, metadata=session.metadata, trials=new_trials,
        masking=session.masking, washout=session.washout, csv_path=session.csv_path,
        filter_info={'label': 'downsampled', 'n_filtered': len(local_idx),
                     'parent_session_id': session.session_id},
        _days_since_first=session._days_since_first,
    )


def downsample(clean, n, unit='trials', with_replacement=True, n_bins=8, rng=None):
    """Subsample clean sessions to ~n of the chosen unit; returns new [SessionData].

    The matched-n draw is pooled across sessions (frozen prev_* make this safe), then split
    back per session and rebuilt, so the output is consumable by compute_psychometric /
    compute_um unchanged. with_replacement=True allows a trial to be drawn more than once
    (a bootstrap resample); False is a clean subsample.
    """
    rng = rng if rng is not None else np.random.default_rng()
    pooled = pool_arrays(clean)
    if pooled['n_trials'] == 0:
        return []

    sel = _draw(pooled, n, unit, n_bins, rng, with_replacement)
    boundaries = pooled['session_boundaries']

    out = []
    for i, session in enumerate(clean):
        lo, hi = boundaries[i], boundaries[i + 1]
        in_session = sel[(sel >= lo) & (sel < hi)] - lo
        if len(in_session) == 0:
            continue
        out.append(_slice_session(session, in_session))
    return out


def calculate_min_n(phases, unit='trials') -> int:
    """Smallest unit-count across a list of clean session-lists (the matched-n target).

    Args:
        phases: list of [SessionData] (each a filtered phase/condition); empties skipped.
        unit:   'trials' (responded trials) or 'pairs' (post-correct pairs).
    """
    counts = []
    for clean in phases:
        if not clean:
            continue
        pooled = pool_arrays(clean)
        if pooled['n_trials'] == 0:
            continue
        c = len(_pool_index(pooled, unit))
        if c > 0:
            counts.append(c)
    return min(counts) if counts else 0


# ── aggregation of K repeats into a plot-compatible result ──────────────────────
def aggregate_psychometric(repeats, n_trials, n_bins=8) -> dict:
    """Mean curve + params + percentile band across successful psychometric repeats."""
    ok = [r for r in repeats if r.get('success')]
    x_fit = repeats[0]['x_fit'] if repeats else np.linspace(-1, 1, 200)
    bin_centres = repeats[0].get('bin_centres') if repeats else None
    if not ok:
        return {'mode': 'pooled', 'params': None, 'params_ci': None, 'curve_band': None,
                'bin_centres': bin_centres, 'bin_means': None, 'bin_counts': None,
                'x_fit': x_fit, 'y_fit': np.full_like(x_fit, np.nan),
                'n_trials': n_trials, 'n_fits': 0, 'success': False}

    Y = np.array([r['y_fit'] for r in ok])
    P = np.array([[r['params'][k] for k in _PARAMS] for r in ok])
    bm = np.array([r['bin_means'] for r in ok])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        params = {k: float(np.nanmean(P[:, i])) for i, k in enumerate(_PARAMS)}
        params_ci = {k: (float(np.nanpercentile(P[:, i], 2.5)),
                         float(np.nanpercentile(P[:, i], 97.5))) for i, k in enumerate(_PARAMS)}
        y_fit = np.nanmean(Y, axis=0)
        band = {'x': x_fit, 'median': np.nanmedian(Y, axis=0),
                'lo': np.nanpercentile(Y, 2.5, axis=0), 'hi': np.nanpercentile(Y, 97.5, axis=0)}
        bin_means = np.nanmean(bm, axis=0)
    return {'mode': 'pooled', 'params': params, 'params_ci': params_ci, 'curve_band': band,
            'bin_centres': bin_centres, 'bin_means': bin_means,
            'bin_counts': np.full(n_bins, n_trials // n_bins), 'x_fit': x_fit,
            'y_fit': y_fit, 'n_trials': n_trials, 'n_fits': len(ok), 'success': True}


def aggregate_um(repeats, n_trials, n_bins=8) -> dict:
    """Mean update + conditional matrix across repeats."""
    ums = [r['um'] for r in repeats if r.get('um') is not None]
    conds = [r['conditional_matrix'] for r in repeats if r.get('conditional_matrix') is not None]
    if not ums:
        empty = np.full((n_bins, n_bins), np.nan)
        return {'mode': 'pooled', 'um': empty, 'conditional_matrix': empty,
                'n_sessions': 0, 'n_trials': 0, 'n_bins': n_bins, 'info': {}}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        um_avg = np.nanmean(np.stack(ums), axis=0)
        cond_avg = np.nanmean(np.stack(conds), axis=0)
    return {'mode': 'pooled', 'um': um_avg, 'conditional_matrix': cond_avg,
            'n_sessions': 0, 'n_trials': n_trials, 'n_bins': n_bins,
            'info': {'method': 'downsampled', 'n_repeats': len(ums)}}


_STAT = {
    'psychometric': {
        'unit': 'trials',
        'compute': lambda s, nb: compute_psychometric(s, mode='pooled', n_bins=nb, n_bootstrap=0),
        'aggregate': aggregate_psychometric,
    },
    'um': {
        'unit': 'pairs',
        'compute': lambda s, nb: compute_um(s, mode='pooled', n_bins=nb),
        'aggregate': aggregate_um,
    },
}


def compute_ds_x(clean, stat, n, n_repeats=100, with_replacement=True, n_bins=8, seed=42) -> dict:
    """Repeat downsample+compute n_repeats times.

    Returns ``{'repeats': [...], 'aggregated': {...}}``: the per-draw compute_x results and
    their aggregate (mean curve + band, or mean matrix). ``n`` is the matched-n target from
    calculate_min_n. ``stat`` selects the unit, fitter and aggregator.
    """
    if stat not in _STAT:
        raise ValueError(f"stat must be one of {list(_STAT)}, got {stat!r}")
    spec = _STAT[stat]
    rng = np.random.default_rng(seed)

    repeats = []
    for _ in range(n_repeats):
        ds = downsample(clean, n, unit=spec['unit'], with_replacement=with_replacement,
                        n_bins=n_bins, rng=rng)
        repeats.append(spec['compute'](ds, n_bins))
    return {'repeats': repeats, 'aggregated': spec['aggregate'](repeats, n, n_bins)}


# ── resample-and-recompute for scalar/param summary stats ───────────────────────
def resample_stat_vectors(
    clean,
    stat_names,
    *,
    n=None,
    n_repeats=1000,
    with_replacement=True,
    unit='trials',
    seed=0,
):
    """Resample trials K times and recompute summary stats — one matrix of replicates.

    The single resample-and-recompute engine for scalar / psychometric-parameter
    stats. It is the scalar analogue of :func:`compute_ds_x` (which targets the
    curve / update matrix), and serves BOTH uses via its arguments:

      * trial bootstrap   — ``with_replacement=True``,  ``n=None`` (natural count)
      * matched-n draw     — ``with_replacement=False``, ``n=target_n``

    Drawing is delegated to :func:`downsample`, so the frozen lag-1 ``prev_*``
    pairing is preserved on every resample (a repeated trial index carries its own
    predecessor). Stats registered ``exchangeable=False`` are refused: trial
    resampling is invalid for order-dependent stats and would return a
    confidently-wrong interval.

    Args:
        clean:           list of [SessionData], abort/opto-cleared (a phase/condition).
        stat_names:      registry stat names (psychometric expands downstream).
        n:               trials/pairs to draw per repeat; None → the natural count
                         of ``unit`` in ``clean`` (the right default for a bootstrap).
        n_repeats:       number of resamples (rows of the returned matrix).
        with_replacement: True for a bootstrap resample, False for a clean subsample.
        unit:            'trials' (responded trials) or 'pairs' (post-correct pairs).
        seed:            RNG seed.

    Returns:
        np.ndarray of shape ``(n_repeats, n_flat)`` where ``n_flat ==
        len(get_stat_names_expanded(stat_names))`` and columns are in that order.
        Rows where the draw was empty are NaN.

    Raises:
        ValueError: if any requested stat is not trial-exchangeable.
    """
    bad = [s for s in stat_names if not is_exchangeable(s)]
    if bad:
        raise ValueError(
            f"trial resampling is invalid for order-dependent stat(s) {bad}; "
            f"exclude them from the bootstrap / downsample (they depend on trial "
            f"order beyond the frozen lag-1 view)."
        )

    rng = np.random.default_rng(seed)
    names = get_stat_names_expanded(stat_names)
    out = np.full((n_repeats, len(names)), np.nan)

    if not clean:
        return out

    if n is None:
        pooled0 = pool_arrays(clean)
        if pooled0['n_trials'] == 0:
            return out
        n = len(_pool_index(pooled0, unit))
    if n <= 0:
        return out

    for r in range(n_repeats):
        drawn = downsample(clean, n, unit=unit,
                           with_replacement=with_replacement, rng=rng)
        if not drawn:
            continue
        pooled = pool_arrays(drawn)
        vals = fit_summary_stats(
            pooled['choices'], pooled['stimuli'], pooled['categories'],
            prev_choices=pooled['prev_choices'],
            prev_stimuli=pooled['prev_stimuli'],
            prev_categories=pooled['prev_categories'],
            stat_names=stat_names, return_dict=False,
        )
        out[r] = np.asarray(vals, dtype=float)

    return out
