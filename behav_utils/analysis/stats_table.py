"""Tier A: sessions → a tidy stat table. The only layer here that touches SessionData.

One animal's pre-filtered sessions go in; a long-form table comes out, one row per
(unit, stat). Everything downstream (combine / paired_diff / bootstrap_units /
rank_test in :mod:`behav_utils.analysis.group`) operates on that table as plain
data — no behavioural objects past this point.

    stat_table = extract_stats(sessions, animal_id='SS15', stats=['recency','win_stay'],
                        mode='pooled', meta={'genotype': 'het'}, n_boot=1000)
    stat_table.estimates   # one row per stat, with within-unit CI columns filled
    stat_table.replicates     # the trial-level bootstrap replicates (None if n_boot=0)

The caller (project side) loops phases/conditions, tags each table with its
labels (``.assign(phase=…, condition=…)``) and concatenates. ``extract_stats``
itself is phase-blind and reusable.

Design notes:
  * Phase/condition labels are NOT produced here — they are the caller's columns.
  * ``SessionData`` carries no animal id / genotype, so both are passed in.
  * Replicates are stored at the ANIMAL level only (``mode='pooled'``); per-session
    bootstrap clouds are intentionally not stored.
  * Update matrices / psychometric curves are matrix-valued and belong on the
    matrix path (``compute_um`` / ``compute_psychometric`` + ``average_arrays``),
    not in this scalar table.
  * The matched-n target comes from
    :func:`behav_utils.analysis.downsample.calculate_min_n` over the session-sets
    being compared, then passed to :func:`extract_matched`.
"""
from __future__ import annotations

from typing import List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from behav_utils.analysis.summary_stats import compute_summary_stats
from behav_utils.analysis.downsample import resample_stat_vectors, calculate_min_n  # noqa: F401 (re-export)

__all__ = ['StatTable', 'extract_stats', 'extract_matched', 'calculate_min_n']

_NUMERIC_POINT = {'value', 'n_trials', 'ci_lo_within', 'ci_hi_within'}


class StatTable(NamedTuple):
    """Output of :func:`extract_stats`.

    Attributes:
        estimates: long frame, one row per (unit, stat). Columns: ``animal``,
            ``session`` (Int64, <NA> at animal level), ``stat``, ``value``,
            ``n_trials``, ``ci_lo_within`` / ``ci_hi_within`` (within-unit /
            trial-level CI; NaN unless ``n_boot>0``), plus any ``meta`` columns.
        replicates: long replicate frame (``animal``, ``stat``, ``rep``, ``value``,
            ``meta``) or None. Keyed to ``estimates`` by (``animal``, ``stat``).
    """
    estimates: pd.DataFrame
    replicates: Optional[pd.DataFrame]


def _flatten_named(results: Mapping) -> List[Tuple[str, float]]:
    """Walk a (possibly nested) stats dict into (name, value) pairs.

    Same traversal order as ``summary_stats.flatten_stats`` (so positions align
    with the resample matrix), but names come from the actual dict keys rather
    than a hardcoded expansion — robust to the order of psychometric params.
    Vector-valued entries get index suffixes.
    """
    out: List[Tuple[str, float]] = []
    for name, value in results.items():
        if isinstance(value, dict):
            for k, v in value.items():
                arr = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
                if arr.size == 1:
                    out.append((str(k), float(arr[0])))
                else:
                    out.extend((f"{k}_{i}", float(x)) for i, x in enumerate(arr))
        else:
            arr = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
            if arr.size == 1:
                out.append((str(name), float(arr[0])))
            else:
                out.extend((f"{name}_{i}", float(x)) for i, x in enumerate(arr))
    return out


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Nullable, parquet-friendly dtypes; non-numeric columns → string."""
    df = df.copy()
    if 'session' in df.columns:
        df['session'] = df['session'].astype('Int64')
    if 'rep' in df.columns:
        df['rep'] = df['rep'].astype('Int64')
    if 'n_trials' in df.columns:
        df['n_trials'] = df['n_trials'].astype('Int64')
    for c in ('value', 'ci_lo_within', 'ci_hi_within'):
        if c in df.columns:
            df[c] = df[c].astype('float64')
    for c in df.columns:
        if c not in (_NUMERIC_POINT | {'session', 'rep'}):
            df[c] = df[c].astype('string')
    return df


# Reserved sub-key of SessionData.filter_info for selection provenance — a flat
# {str: str} dict of the scientific selection that produced the session (e.g.
# distribution / session_type / trial_type). Written by the project's phase
# selector (filter_phase); read here as default meta. The library copies whatever
# keys are present and never inspects their values, so it stays task-agnostic.
_SELECTION_KEY = 'selection'


def _resolve_meta(sessions: Sequence, meta: Optional[Mapping]) -> dict:
    """Merge per-session selection provenance into ``meta`` under the all-or-nothing rule.

    Reads ``session.filter_info['selection']`` from every session. Provenance fills
    gaps only — anything passed explicitly in ``meta`` wins (``setdefault``).

    Guards against pooling unrelated selections into one stat unit:
      * provenance present on all sessions and identical → folded in;
      * present but disagreeing → raise (two different phases/conditions);
      * present on some sessions but not others → raise (mixed tagged/untagged);
      * absent on all → no auto-meta (synthetic or non-``filter_phase`` sessions).
    """
    out = dict(meta or {})
    sels = [(getattr(s, 'filter_info', None) or {}).get(_SELECTION_KEY) for s in sessions]
    present = [s for s in sels if s is not None]

    if present and len(present) != len(sels):
        raise ValueError(
            "extract_*: sessions mix selection-tagged and untagged data (some carry "
            "filter_info['selection'], some do not); refusing to pool them into one "
            "stat unit. Filter the whole unit through filter_phase, or pass meta "
            "explicitly for an untagged list."
        )

    if present:
        first = present[0]
        for other in present[1:]:
            if other != first:
                raise ValueError(
                    f"extract_*: sessions come from different selections "
                    f"({first} vs {other}); one extract_* call is one phase/condition. "
                    f"Filter each selection separately and concatenate the tables."
                )
        for k, v in first.items():
            out.setdefault(k, v)

    return out


def extract_stats(
    sessions: Sequence,
    *,
    animal_id: str,
    stats: Sequence[str],
    mode: str = 'per_session',
    meta: Optional[Mapping] = None,
    n_boot: int = 0,
    ci: float = 0.95,
    unit: str = 'trials',
    seed: int = 0,
) -> StatTable:
    """Extract a tidy stat table for one animal's pre-filtered sessions.

    Args:
        sessions:  list of [SessionData], already filtered to the phase/condition.
        animal_id: id stamped on every row (SessionData carries none).
        stats:     registry stat names. ``psychometric`` expands to
                   ``mu``/``sigma``/``lapse_low``/``lapse_high`` rows.
        mode:      'per_session' (row per session) or 'pooled' (one row per stat,
                   trials pooled within the animal — trial-weighted).
        meta:      animal-level metadata to carry as columns, e.g.
                   ``{'genotype': animal.genotype}``. The selection labels
                   (``distribution`` / ``session_type`` / ``trial_type``) are
                   auto-filled from each session's ``filter_info['selection']``
                   when present (set by ``filter_phase``); anything passed here
                   overrides the auto-filled value.
        n_boot:    trial-level bootstrap resamples. >0 fills ``ci_*_within`` and
                   returns ``replicates``. Only valid with ``mode='pooled'``.
        ci:        central interval width for the within-unit CI.
        unit:      'trials' or 'pairs' (resample unit for the bootstrap).
        seed:      RNG seed for the bootstrap.

    Returns:
        :class:`StatTable`.

    Raises:
        ValueError: bad ``mode``; or ``n_boot>0`` with ``mode!='pooled'``; or an
            order-dependent stat under ``n_boot`` (raised by the resampler).
    """
    meta = _resolve_meta(sessions, meta)
    res = compute_summary_stats(sessions, stat_names=list(stats), mode=mode)

    rows: List[dict] = []
    pooled_named: Optional[List[Tuple[str, float]]] = None

    if mode == 'pooled':
        pooled_named = _flatten_named(res['stats'])
        for nm, v in pooled_named:
            rows.append({'animal': animal_id, 'session': pd.NA, 'stat': nm,
                         'value': v, 'n_trials': res['n_trials']})
    elif mode == 'per_session':
        for entry in res['per_session']:
            for nm, v in _flatten_named(entry['stats']):
                rows.append({'animal': animal_id, 'session': entry['session_idx'],
                             'stat': nm, 'value': v, 'n_trials': entry['n_trials']})
    else:
        raise ValueError(f"mode must be 'pooled' or 'per_session', got {mode!r}")

    estimates = pd.DataFrame(rows)
    for k, val in meta.items():
        estimates[k] = val
    estimates['ci_lo_within'] = np.nan
    estimates['ci_hi_within'] = np.nan

    replicates = None
    if n_boot and n_boot > 0:
        if mode != 'pooled':
            raise ValueError(
                "n_boot is only supported for mode='pooled' (per-session bootstrap "
                "replicates are intentionally not stored — bootstrap at the animal "
                "level, the unit of inference)."
            )
        order_names = [nm for nm, _ in pooled_named]
        mat = resample_stat_vectors(
            sessions, list(stats), n=None, n_repeats=n_boot,
            with_replacement=True, unit=unit, seed=seed,
        )
        if mat.shape[1] != len(order_names):  # pragma: no cover - structural guard
            raise RuntimeError(
                f"resample/point column mismatch ({mat.shape[1]} vs {len(order_names)})"
            )
        alpha = (1.0 - ci) / 2.0
        lo = np.nanpercentile(mat, 100 * alpha, axis=0)
        hi = np.nanpercentile(mat, 100 * (1 - alpha), axis=0)
        ci_lo = {nm: lo[j] for j, nm in enumerate(order_names)}
        ci_hi = {nm: hi[j] for j, nm in enumerate(order_names)}
        estimates['ci_lo_within'] = estimates['stat'].map(ci_lo).astype('float64')
        estimates['ci_hi_within'] = estimates['stat'].map(ci_hi).astype('float64')

        B, K = mat.shape
        replicates = pd.DataFrame({
            'animal': animal_id,
            'stat': np.repeat(np.asarray(order_names, dtype=object), B),
            'rep': np.tile(np.arange(B), K),
            'value': mat.T.reshape(-1),
        })
        for k, val in meta.items():
            replicates[k] = val
        replicates = _coerce(replicates)

    return StatTable(_coerce(estimates), replicates)


def extract_matched(
    sessions: Sequence,
    target_n: int,
    *,
    animal_id: str,
    stats: Sequence[str],
    unit: str = 'trials',
    n_repeats: int = 100,
    meta: Optional[Mapping] = None,
    seed: int = 0,
) -> StatTable:
    """Animal-level stat values with trial count matched to ``target_n``.

    Resamples each animal's trials down to ``target_n`` (without replacement),
    recomputes the stats per draw, and returns the mean over draws as ``value`` —
    so a group difference can't be partly a trial-count artefact. Same engine and
    drawer as the bootstrap; ``replicates`` is None because a matched value is a point,
    not a stored distribution.

    The matched value is *not* an intrinsic property of the animal — it depends on
    ``target_n``, which depends on the comparison — so it is computed on demand,
    not cached. Get ``target_n`` from
    :func:`behav_utils.analysis.downsample.calculate_min_n` over the session-sets
    you are matching.

    Args:
        sessions:  list of [SessionData] for one animal/condition.
        target_n:  matched count of ``unit``.
        animal_id: id stamped on every row.
        stats:     registry stat names (must be trial-exchangeable).
        unit:      'trials' or 'pairs'.
        n_repeats: number of matched draws to average over.
        meta:      animal-level metadata to carry as columns.
        seed:      RNG seed.

    Returns:
        :class:`StatTable` with ``replicates=None``; ``session`` is <NA> (animal level)
        and ``n_trials`` is ``target_n``.
    """
    meta = _resolve_meta(sessions, meta)
    full = compute_summary_stats(sessions, stat_names=list(stats), mode='pooled')
    order_names = [nm for nm, _ in _flatten_named(full['stats'])]

    mat = resample_stat_vectors(
        sessions, list(stats), n=target_n, n_repeats=n_repeats,
        with_replacement=False, unit=unit, seed=seed,
    )
    value = np.nanmean(mat, axis=0)

    rows = [{'animal': animal_id, 'session': pd.NA, 'stat': nm,
             'value': float(value[j]), 'n_trials': target_n}
            for j, nm in enumerate(order_names)]
    estimates = pd.DataFrame(rows)
    for k, val in meta.items():
        estimates[k] = val
    estimates['ci_lo_within'] = np.nan
    estimates['ci_hi_within'] = np.nan
    return StatTable(_coerce(estimates), None)
