"""
Phase comparison for 2AFC tasks.

Compare N pre-filtered trial sets ("phases") against one reference, then compare
those comparisons across session types.

    load_experiment → select_sessions → filter_trials → phase
                                                          ↓
                              compare_phases   (within phase: opto vs non_opto)
                                                          ↓
                              compute_interaction  (between phases: Δ₁ vs Δ₂)

A phase is whatever ``filter_trials`` returned, treated as one pooled trial set:
opto vs non_opto vs post_opto within a session type, masking vs opto sessions,
pre- vs post-shift. One session or forty makes no difference to the machinery.

    phases = {tt: filter_trials(selected, trial_type=tt)
              for tt in ['non_opto', 'opto', 'post_opto']}

    result = compare_phases(phases, stats=['psychometric', 'win_stay', 'um'],
                            reference='non_opto')

    plot_psychometric(result['phases']['opto']['psychometric'])
    plot_um(result['phases']['opto']['um'])
    plot_comparison(result, 'opto_vs_non_opto')
    result['contrasts']['opto_vs_non_opto']['diffs' | 'perm_p' | 'boot_ci']

    # is the opto effect different in masking sessions than in opto sessions?
    interaction = compute_interaction(result_masking, result_opto,
                                      contrast='opto_vs_non_opto')

All resampling is delegated to :mod:`behav_utils.analysis.resampling`; this
module only orchestrates and assembles. Statistics come from pooled arrays via
``fit_summary_stats``, so the observed value, the permutation null and the
bootstrap all travel the same code path. Per-phase display objects are the
untouched outputs of ``compute_psychometric`` / ``compute_um``, so they feed the
plotters directly.

Public API:
    compare_phases      — N phases vs a reference, within-phase contrasts
    compute_interaction — difference of differences across two results
"""

from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from behav_utils.data.structures import SessionData

import numpy as np

from behav_utils.analysis.resampling import (
    bootstrap_phase_stats,
    compute_stats_from_arrays,
    permute_phase_difference,
    pool_phase_arrays,
    summarise_draw_distribution,
)

__all__ = ['compute_delta_stat', 'compare_phases', 'compute_interaction']

_UPDATE_MATRIX_STAT = 'um'


def _contrast_key(phase: str, reference: str) -> str:
    return f'{phase}_vs_{reference}'


def compute_delta_stat(
    phases,
    stats: Sequence[str] = ('psychometric',),
    labels: Optional[List[str]] = None,
    reference: Optional[str] = None,
    downsample: bool = False,
    n_permutations: int = 1000,
    n_bootstrap: int = 1000,
    n_bins: int = 8,
    trial_filter: str = 'post_correct',
    n_repeats: int = 200,
    resample_units: Sequence[str] = ('trials',),
    seed: int = 42,
) -> Dict:
    """Compare N phases against a reference, with permutation p and bootstrap CI.

    Every non-reference phase is contrasted against the reference, so
    ``['non_opto', 'opto', 'post_opto']`` with ``reference='non_opto'`` gives
    opto−non_opto and post_opto−non_opto, and not the opto−post_opto contrast
    nobody asked for.

    Args:
        phases:     ``{label: phase}``, or a list of phases with ``labels``.
                    Each phase is the output of ``filter_trials``.
        stats:      family-level names. ``'psychometric'`` expands to mu, sigma,
                    lapse_low, lapse_high; ``'um'`` adds the update matrix
                    (descriptive only, see Notes); anything else in
                    ``list_available_stats()`` is a scalar.
        reference:  label the others are contrasted against (default: the first).
                    Δ is ``phase − reference``.
        downsample: match trial counts across phases before comparing. The
                    matched n is derived internally per stat unit — trials for
                    scalars and the psychometric, pairs for the update matrix —
                    so the caller cannot mismatch the unit. Corrects bias from
                    unequal precision at the cost of power.
        n_permutations: label shuffles for the p-value (0 to skip). Smallest
                    reportable p is ``1 / (n_permutations + 1)``.
        n_bootstrap: resamples for the CI (0 to skip).
        n_bins:     update-matrix bins.
        trial_filter: 'post_correct' | 'all', for the update matrix.
        n_repeats:  matched-n draws for the display objects when downsampling.
        resample_units: which unit(s) to bootstrap the CI over. The first is the
                    *primary* unit and populates the legacy CI fields
                    (``stats_ci``, ``boot_ci``, ``difference_draws``) unchanged;
                    any further units are added alongside under the ``*_by_unit``
                    fields. ``('trials',)`` (default) reproduces the old output
                    exactly. Pass ``('trials', 'sessions')`` to get both a
                    per-trial CI and a per-session CI on every contrast — the
                    per-session one is the honest interval for a between-phase
                    difference (phases whose session type was not randomised per
                    trial); the per-trial one is the diagnostic that ignores
                    session scatter. ``'sessions'`` needs ≥2 sessions in the
                    phase, else its CI is skipped for that condition.
        seed:       base RNG seed.

    Returns:
        ::

            result['phases'][label]
                ['stats']            {stat: observed value}
                ['stats_ci']         {stat: (lo, hi)}   — display only (primary unit)
                ['stats_ci_by_unit'] {unit: {stat: (lo, hi)}}
                ['bootstrap_draws']  {stat: ndarray}    — primary unit (composable)
                ['bootstrap_draws_by_unit'] {unit: {stat: ndarray}}
                ['psychometric']     → plot_psychometric(...)
                ['um']               → plot_um(...)
                ['n_trials'], ['n_sessions']

            result['contrasts'][f'{phase}_vs_{reference}']
                ['diffs']             {stat: Δ}
                ['perm_p']            {stat: p}
                ['boot_ci']           {stat: (lo, hi)}   — primary unit
                ['boot_ci_by_unit']   {unit: {stat: (lo, hi)}}
                ['difference_draws']  {stat: ndarray} — primary unit; feeds compute_interaction
                ['difference_draws_by_unit'] {unit: {stat: ndarray}}
                ['um_diff'], ['um_rmse'], ['um_corr']
                ['label_a'], ['label_b'], ['n_a'], ['n_b']

            result['meta']  reference, stats, families, scalar_names,
                            downsample, n_matched, seed

    Notes:
        The update matrix is descriptive only — no permutation test. One update
        matrix is nine psychometric fits, so a 1000-shuffle null would be ~18k
        fits per contrast per animal.

        Both phases are fitted separately and subtracted. A joint fit with Δ as
        a free parameter and the lapses shared would be more precise and would
        stop a PSE shift leaking into an asymmetric lapse, but
        ``fit_psychometric`` has no lapse-fixing argument. Read lapse deltas
        with caution.

    Raises:
        ValueError: if fewer than two phases, if ``reference`` is not among
            them, or if an order-dependent stat is requested with resampling on.
    """
    from behav_utils.analysis.psychometry import compute_psychometric
    from behav_utils.analysis.summary_stats import (
        get_stat_names_expanded, is_exchangeable,
    )
    from behav_utils.analysis.update_matrix import compute_um

    # ── phases → ordered {label: phase} ───────────────────────────────────
    if isinstance(phases, dict):
        items = list(phases.items())
    else:
        phases = list(phases)
        if labels is None:
            labels = [f'phase_{i}' for i in range(len(phases))]
        if len(labels) != len(phases):
            raise ValueError("compare_phases: len(labels) != len(phases)")
        items = list(zip(labels, phases))
    if len(items) < 2:
        raise ValueError("compare_phases: need at least two phases")

    names = [name for name, _ in items]
    if reference is None:
        reference = names[0]
    if reference not in names:
        raise ValueError(
            f"compare_phases: reference {reference!r} not among {names}")

    # ── stats vocabulary ──────────────────────────────────────────────────
    stats = list(stats)
    want_update_matrix = _UPDATE_MATRIX_STAT in stats
    families = [s for s in stats if s != _UPDATE_MATRIX_STAT]
    want_psychometric = 'psychometric' in families
    scalar_names = list(get_stat_names_expanded(families)) if families else []

    # Which units to bootstrap over. First is primary (drives the legacy fields).
    resample_units = list(dict.fromkeys(resample_units))  # de-dup, keep order
    bad_units = [u for u in resample_units if u not in ('trials', 'sessions')]
    if bad_units:
        raise ValueError(
            f"compare_phases: resample_units must be 'trials'/'sessions', "
            f"got {bad_units}")
    primary_unit = resample_units[0] if resample_units else 'trials'

    # Order-dependence guard, unit-aware. Permutation shuffles trial labels, and
    # a trial bootstrap reshuffles trials, so a stat that looks beyond the frozen
    # lag-1 view is invalid under either. A *session* bootstrap keeps whole
    # sessions intact (order preserved), so it does not trigger the guard.
    trial_order_resampling = (
        n_permutations > 0
        or (n_bootstrap > 0 and any(u in ('trials',) for u in resample_units))
    )
    if families and trial_order_resampling:
        bad = [s for s in families if not is_exchangeable(s)]
        if bad:
            raise ValueError(
                f"compare_phases: order-dependent stat(s) {bad} cannot be "
                f"permuted or trial-bootstrapped (they depend on trial order "
                f"beyond the frozen lag-1 view, which trial resampling destroys). "
                f"Drop them, use resample_units=('sessions',) with "
                f"n_permutations=0, or set n_permutations=0 and n_bootstrap=0 to "
                f"report observed values only — and note they are unreliable on "
                f"an interleaved subset regardless."
            )

    # ── matched n, derived per unit ───────────────────────────────────────
    n_matched: Dict[str, int] = {}
    if downsample:
        from behav_utils.analysis.downsample import calculate_min_n
        groups = [phase for _, phase in items]
        n_matched['trials'] = calculate_min_n(groups, unit='trials')
        if want_update_matrix:
            n_matched['pairs'] = calculate_min_n(groups, unit='pairs')
    draw_n_trials = n_matched.get('trials') if downsample else None

    # ── per phase: observed values, display objects, bootstrap draws ──────
    pooled_arrays, out_phases = {}, {}
    for i, (name, phase) in enumerate(items):
        arrays = pool_phase_arrays(phase)
        pooled_arrays[name] = arrays

        entry = {
            'n_trials': int(arrays['choices'].size),
            'n_sessions': int(len(phase)),
            'stats': compute_stats_from_arrays(
                arrays, families, rng=np.random.default_rng(seed + i)),
        }

        # Bootstrap each condition ONCE per requested unit. Every contrast and
        # interaction is then a subtraction of these stored arrays, so a shared
        # reference cancels exactly instead of contributing its variance twice.
        # The primary unit keeps the pre-change seed (seed + i), so
        # resample_units=('trials',) reproduces the old draws byte-for-byte.
        draws_by_unit: Dict[str, Dict] = {}
        ci_by_unit: Dict[str, Dict] = {}
        if n_bootstrap > 0 and families:
            for u_idx, unit in enumerate(resample_units):
                if unit == 'trials' and entry['n_trials'] < 10:
                    continue
                if unit == 'sessions' and entry['n_sessions'] < 2:
                    continue  # <2 sessions carries no session-level information
                u_seed = (seed + i) if u_idx == 0 \
                    else (seed + i + u_idx * 1_000_003)
                u_draws = bootstrap_phase_stats(
                    phase, families, n_draws=n_bootstrap,
                    n_trials=draw_n_trials, seed=u_seed, unit=unit,
                )
                draws_by_unit[unit] = u_draws
                ci_by_unit[unit] = {
                    stat: (summarise_draw_distribution(values)['ci_lo'],
                           summarise_draw_distribution(values)['ci_hi'])
                    for stat, values in u_draws.items()
                }
        if draws_by_unit:
            entry['bootstrap_draws_by_unit'] = draws_by_unit
            entry['stats_ci_by_unit'] = ci_by_unit
            # Legacy single-unit fields = primary unit (fall back to whatever
            # was computable, e.g. if the primary was skipped on a gate).
            leg = primary_unit if primary_unit in draws_by_unit \
                else next(iter(draws_by_unit))
            entry['bootstrap_draws'] = draws_by_unit[leg]
            entry['stats_ci'] = ci_by_unit[leg]
            if downsample and 'trials' in draws_by_unit:
                # The null below is built at the matched n, so the observed
                # value compared against it must be too.
                entry['stats_matched'] = {
                    stat: float(np.nanmedian(values))
                    for stat, values in draws_by_unit['trials'].items()
                }

        if want_psychometric:
            if downsample:
                from behav_utils.analysis.downsample import compute_ds_x
                entry['psychometric'] = compute_ds_x(
                    phase, 'psychometric', n_matched['trials'],
                    n_repeats=n_repeats, seed=seed + i)['aggregated']
            else:
                entry['psychometric'] = compute_psychometric(phase, mode='pooled')

        if want_update_matrix:
            if downsample:
                from behav_utils.analysis.downsample import compute_ds_x
                entry['um'] = compute_ds_x(
                    phase, 'um', n_matched['pairs'], n_repeats=n_repeats,
                    seed=seed + i, n_bins=n_bins)['aggregated']
            else:
                entry['um'] = compute_um(phase, mode='pooled', n_bins=n_bins,
                                         trial_filter=trial_filter)

        out_phases[name] = entry

    # ── contrasts: every non-reference phase vs the reference ─────────────
    contrasts = {}
    reference_entry = out_phases[reference]

    for k, name in enumerate([n for n in names if n != reference]):
        entry = out_phases[name]
        observed_key = 'stats_matched' if downsample else 'stats'
        observed_a = entry.get(observed_key, entry['stats'])
        observed_b = reference_entry.get(observed_key, reference_entry['stats'])
        diffs = {stat: observed_a.get(stat, np.nan) - observed_b.get(stat, np.nan)
                 for stat in scalar_names}

        contrast = {
            'diffs': diffs,
            'label_a': name,
            'label_b': reference,
            'n_a': entry['n_trials'],
            'n_b': reference_entry['n_trials'],
            'n_sessions_a': entry['n_sessions'],
            'n_sessions_b': reference_entry['n_sessions'],
            'perm_p': None,
            'boot_ci': None,
            'difference_draws': None,
            'boot_ci_by_unit': {},
            'boot_p_by_unit': {},
            'difference_draws_by_unit': {},
        }

        # Bootstrap CI (and a two-sided bootstrap p) per unit: subtract the stored
        # per-condition draws for that unit. A shared reference cancels exactly
        # because the SAME draw arrays are reused (that is why per-condition draws
        # are stored, not the diffs).
        draws_a_by_unit = entry.get('bootstrap_draws_by_unit', {})
        draws_b_by_unit = reference_entry.get('bootstrap_draws_by_unit', {})
        for unit in resample_units:
            draws_a = draws_a_by_unit.get(unit)
            draws_b = draws_b_by_unit.get(unit)
            if not draws_a or not draws_b:
                continue
            difference_draws, boot_ci, boot_p = {}, {}, {}
            for stat in scalar_names:
                if stat not in draws_a or stat not in draws_b:
                    continue
                n = min(draws_a[stat].size, draws_b[stat].size)
                delta = draws_a[stat][:n] - draws_b[stat][:n]
                difference_draws[stat] = delta
                summary = summarise_draw_distribution(delta)
                boot_ci[stat] = (summary['ci_lo'], summary['ci_hi'])
                boot_p[stat] = summary['p_two_sided']
            contrast['difference_draws_by_unit'][unit] = difference_draws
            contrast['boot_ci_by_unit'][unit] = boot_ci
            contrast['boot_p_by_unit'][unit] = boot_p
        # Legacy single-unit fields = primary unit (fall back to any computed).
        if contrast['difference_draws_by_unit']:
            leg = primary_unit if primary_unit in contrast['difference_draws_by_unit'] \
                else next(iter(contrast['difference_draws_by_unit']))
            contrast['difference_draws'] = contrast['difference_draws_by_unit'][leg]
            contrast['boot_ci'] = contrast['boot_ci_by_unit'][leg]
            contrast['boot_p'] = contrast['boot_p_by_unit'].get(leg)

        # Permutation p: shuffle the phase label, recompute Δ.
        if n_permutations > 0 and families \
                and entry['n_trials'] >= 10 and reference_entry['n_trials'] >= 10:
            null_draws = permute_phase_difference(
                items[names.index(name)][1], items[names.index(reference)][1],
                families, n_draws=n_permutations,
                n_trials=draw_n_trials, seed=seed + 1000 + k,
            )
            contrast['perm_p'] = {
                stat: summarise_draw_distribution(
                    null_draws.get(stat, np.array([])),
                    observed=diffs.get(stat),
                )['p_two_sided']
                for stat in scalar_names
            }

        # Update matrix: descriptive only.
        if want_update_matrix:
            matrix_a = entry.get('um', {}).get('um')
            matrix_b = reference_entry.get('um', {}).get('um')
            if matrix_a is not None and matrix_b is not None:
                difference = matrix_a - matrix_b
                usable = np.isfinite(matrix_a) & np.isfinite(matrix_b)
                contrast['um_diff'] = difference
                if usable.sum() >= 4:
                    from scipy.stats import pearsonr
                    contrast['um_rmse'] = float(np.sqrt(np.mean(difference[usable] ** 2)))
                    contrast['um_corr'] = float(pearsonr(matrix_a[usable], matrix_b[usable])[0])
                else:
                    contrast['um_rmse'] = np.nan
                    contrast['um_corr'] = np.nan

        contrasts[_contrast_key(name, reference)] = contrast

    return {
        'phases': out_phases,
        'contrasts': contrasts,
        'meta': {
            'reference': reference, 'stats': stats, 'families': families,
            'scalar_names': scalar_names, 'downsample': downsample,
            'n_matched': n_matched or None, 'seed': seed,
            'resample_units': resample_units, 'primary_unit': primary_unit,
        },
    }


def compute_interaction(
    result_a: Dict,
    result_b: Dict,
    contrast: str,
    contrast_b: Optional[str] = None,
    label_a: str = 'a',
    label_b: str = 'b',
    ci: float = 0.95,
) -> Dict:
    """Difference of differences between two ``compare_phases`` results.

    Answers "is the opto effect different in *these* sessions than in *those*" —
    for example whether the opto−non_opto shift on masking sessions differs from
    the shift on real opto sessions, which is what separates a genuine
    inactivation effect from the light-delivery artifact.

    Comparing two p-values is not a substitute for this. "Significant here,
    not significant there" is not evidence the two differ; the difference has
    to be tested directly.

    The interaction is estimated by bootstrap only. A permutation would require
    shuffling the session-type label across trials, but session type was not
    randomised per trial — those trials were recorded on different days — so no
    shuffle reproduces the design.

    Args:
        result_a, result_b: outputs of ``compare_phases``. Pass the same object
            twice to compare two contrasts within one result.
        contrast:   contrast key in ``result_a``, e.g. 'opto_vs_non_opto'.
        contrast_b: contrast key in ``result_b``; defaults to ``contrast``.
        label_a, label_b: names for the two results in the output.
        ci:         interval mass.

    Returns:
        ``{stat: {'delta_a', 'delta_b', 'interaction', 'ci_lo', 'ci_hi',
        'p_two_sided', 'n_draws', 'by_unit'}}`` plus a ``'meta'`` entry. The
        top-level fields are the primary unit; ``'by_unit'`` maps each resample
        unit (e.g. 'trials', 'sessions') to the same fields, so a between-phase
        delta-of-deltas can show its trial CI and its session CI together.

    Raises:
        KeyError:   if either contrast is absent.
        ValueError: if either result lacks bootstrap draws (rerun
            ``compare_phases`` with ``n_bootstrap > 0``).

    Note:
        When both results come from the same ``compare_phases`` call, the two
        contrasts share their reference draws and those cancel exactly:
        ``(A − C) − (B − C) = A − B``. That is only true because the draws are
        the same arrays, which is why the per-condition draws are stored rather
        than the differences alone.
    """
    contrast_b = contrast_b or contrast
    for result, key, which in ((result_a, contrast, 'result_a'),
                               (result_b, contrast_b, 'result_b')):
        if key not in result.get('contrasts', {}):
            raise KeyError(
                f"compute_interaction: {which} has no contrast {key!r}; "
                f"available: {sorted(result.get('contrasts', {}))}"
            )

    entry_a = result_a['contrasts'][contrast]
    entry_b = result_b['contrasts'][contrast_b]

    # Difference draws per unit, with a fallback for results produced before the
    # per-unit fields existed (their legacy difference_draws are the trial unit).
    units_a = dict(entry_a.get('difference_draws_by_unit') or {})
    units_b = dict(entry_b.get('difference_draws_by_unit') or {})
    if not units_a and entry_a.get('difference_draws'):
        units_a = {'trials': entry_a['difference_draws']}
    if not units_b and entry_b.get('difference_draws'):
        units_b = {'trials': entry_b['difference_draws']}
    common_units = [u for u in units_a if u in units_b]
    if not common_units:
        raise ValueError(
            "compute_interaction: no shared bootstrap draws — rerun "
            "compare_phases with n_bootstrap > 0 (and matching resample_units)."
        )

    # The unit that fills the legacy top-level fields (back-compat).
    meta_primary = (result_a.get('meta', {}) or {}).get('primary_unit')
    if meta_primary in common_units:
        primary_unit = meta_primary
    elif 'trials' in common_units:
        primary_unit = 'trials'
    else:
        primary_unit = common_units[0]

    def _interaction_for(draws_a, draws_b):
        res = {}
        for stat in sorted(set(draws_a) & set(draws_b)):
            da, db = draws_a[stat], draws_b[stat]
            n = min(da.size, db.size)
            if n < 10:
                continue
            interaction_draws = da[:n] - db[:n]
            summary = summarise_draw_distribution(interaction_draws, ci=ci)
            summary_a = summarise_draw_distribution(da, ci=ci)
            summary_b = summarise_draw_distribution(db, ci=ci)
            res[stat] = {
                # component effects, each with its own interval — context for
                # the interaction, NOT to be compared by eye (overlapping
                # intervals do not mean the difference is null).
                'delta_a': entry_a['diffs'].get(stat, np.nan),
                'ci_a': (summary_a['ci_lo'], summary_a['ci_hi']),
                'delta_b': entry_b['diffs'].get(stat, np.nan),
                'ci_b': (summary_b['ci_lo'], summary_b['ci_hi']),
                # the inferential claim
                'interaction': (entry_a['diffs'].get(stat, np.nan)
                                - entry_b['diffs'].get(stat, np.nan)),
                'ci_lo': summary['ci_lo'],
                'ci_hi': summary['ci_hi'],
                'p_two_sided': summary['p_two_sided'],
                'n_draws': summary['n_draws'],
                'draws': interaction_draws,
            }
        return res

    per_unit = {u: _interaction_for(units_a[u], units_b[u]) for u in common_units}

    out: Dict[str, Dict] = {}
    all_stats = sorted({stat for u in common_units for stat in per_unit[u]})
    for stat in all_stats:
        base = per_unit.get(primary_unit, {}).get(stat)
        if base is None:  # primary didn't yield this stat; take any that did
            base = next((per_unit[u][stat] for u in common_units
                         if stat in per_unit[u]), None)
        if base is None:
            continue
        entry = dict(base)  # legacy top-level fields = primary (or fallback) unit
        entry['by_unit'] = {u: per_unit[u][stat] for u in common_units
                            if stat in per_unit[u]}
        out[stat] = entry

    out['meta'] = {
        'label_a': label_a, 'label_b': label_b,
        'contrast_a': contrast, 'contrast_b': contrast_b,
        'shared_result': result_a is result_b,
        'units': common_units, 'primary_unit': primary_unit,
        'method': 'bootstrap (no permutation: session type is not randomised '
                  'per trial)',
    }
    return out


def compare_phases(*args, **kwargs):
    """Deprecated alias of :func:`compute_delta_stat`.

    Renamed for symmetry with ``compute_stat``: the two public verbs are
    ``compute_stat`` (values) and ``compute_delta_stat`` (differences between
    conditions). ``compare_phases`` still works but will be removed.
    """
    import warnings
    warnings.warn(
        "compare_phases() is deprecated; use compute_delta_stat() "
        "(same signature and return).",
        DeprecationWarning, stacklevel=2)
    return compute_delta_stat(*args, **kwargs)
