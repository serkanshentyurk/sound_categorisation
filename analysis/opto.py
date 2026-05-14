"""
Optogenetic effect analysis.

Phase assignment, within-session opto vs control comparison,
cross-phase stability, adaptation rate comparison,
genotype interaction tests, model-assignment integration,
equivalence testing, and phase × opto interaction.

Core pattern: analysis functions take List[SessionData] or SessionData,
never do their own session selection. Phase assignment takes AnimalData
because it needs the full sequential context. Masking is read from
session.masking (set by load_experiment from config.yaml / CSV column).

Public API:
    OptoPhase               — Enum for experimental phases
    assign_opto_phases      — Label each session with its phase
    opto_relative_mask      — Wrapper: session.trials.opto_mask(delta)
    split_trials_by_opto    — Wrapper: opto + control masks
    get_post_opto_mask      — Wrapper: post-opto mask (delta=1)
    within_session_effect   — Per-session opto vs control stats
    phase_pooled_comparison — Pool across sessions, compare opto vs control
    compute_opto_um         — Update matrix from opto-relative trials (delta API)
    expert_stability        — Backward-compat wrapper for phase_stability
    phase_stability         — Full baseline/masking/opto/washout comparison
    genotype_interaction    — Compare effect sizes across het vs WT
    animal_opto_report      — Run full analysis for one animal
    cohort_opto_report      — Run full analysis for all animals, split by genotype
    opto_by_model_assignment — Group opto effects by BE/SC/unclear consensus
    expert_null_test        — TOST equivalence test for expert-phase null prediction
    expert_um_test          — UM RMSE equivalence test for expert-phase null prediction
    phase_opto_interaction  — Phase × opto interaction (expert vs post-shift)
    simulate_with_opto      — Simulate session with trial-level opto lesion

    Filtering is handled by behav_utils.data.filtering — this module
    does NOT filter internally. Mask wrappers (opto_relative_mask, etc.)
    delegate to filtering.opto_mask.

Usage:
    from analysis.opto import assign_opto_phases, within_session_effect, OptoPhase

    phases = assign_opto_phases(animal)
    for sess, phase in zip(animal.sessions, phases):
        if phase == OptoPhase.EXPERT_OPTO:
            effect = within_session_effect(sess)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from scipy.stats import mannwhitneyu, ttest_1samp, wilcoxon, ttest_rel

from behav_utils.analysis.update_matrix import compute_update_matrix
from behav_utils.data.filtering import (
    build_mask, opto_mask as _opto_mask,
    filter_session, filter_trials, get_arrays, pool_arrays,
)


# ─── Phase enum ──────────────────────────────────────────────────────────────

class OptoPhase(Enum):
    EXPERT_BASELINE = 'expert_baseline'
    EXPERT_OPTO = 'expert_opto'
    EXPERT_WASHOUT = 'expert_washout'
    MASKING = 'masking'
    SHIFT_1_OPTO = 'shift_1_opto'
    SHIFT_1_RECOVERY = 'shift_1_recovery'
    SHIFT_2_OPTO = 'shift_2_opto'
    SHIFT_2_RECOVERY = 'shift_2_recovery'
    PRE_EXPERIMENT = 'pre_experiment'


# ─── Phase assignment ────────────────────────────────────────────────────────

def _session_has_opto(session) -> bool:
    """Check whether a session contains any opto trials."""
    return hasattr(session.trials, 'opto_on') and np.any(session.trials.opto_on)


def assign_opto_phases(animal) -> List[OptoPhase]:
    """
    Label each session with its experimental phase.

    Walks through sessions sequentially, detecting transitions from
    distribution changes and opto presence. Masking is read from
    session.masking (set automatically by load_experiment).

    Args:
        animal: AnimalData object

    Returns:
        List of OptoPhase, one per session, same order as animal.sessions.
    """
    sessions = animal.sessions
    phases = []

    seen_opto = False
    current_dist = None
    n_shifts = 0

    for i, sess in enumerate(sessions):
        dist = sess.distribution
        has_opto = _session_has_opto(sess)

        if getattr(sess, 'masking', False):
            phases.append(OptoPhase.MASKING)
            continue

        if dist == 'Uniform':
            if not seen_opto and not has_opto:
                phases.append(OptoPhase.EXPERT_BASELINE)
            elif has_opto:
                seen_opto = True
                phases.append(OptoPhase.EXPERT_OPTO)
            elif seen_opto and not has_opto:
                phases.append(OptoPhase.EXPERT_WASHOUT)
            else:
                phases.append(OptoPhase.EXPERT_BASELINE)
            current_dist = 'Uniform'

        elif dist in ('Asym_Right', 'Asym_Left'):
            if current_dist != dist:
                n_shifts += 1
                current_dist = dist

            if n_shifts == 1:
                if has_opto:
                    phases.append(OptoPhase.SHIFT_1_OPTO)
                else:
                    phases.append(OptoPhase.SHIFT_1_RECOVERY)
            elif n_shifts >= 2:
                if has_opto:
                    phases.append(OptoPhase.SHIFT_2_OPTO)
                else:
                    phases.append(OptoPhase.SHIFT_2_RECOVERY)
            else:
                phases.append(OptoPhase.PRE_EXPERIMENT)
        else:
            phases.append(OptoPhase.PRE_EXPERIMENT)

    return phases


# ─── Trial masking ───────────────────────────────────────────────────────────

def opto_relative_mask(session, delta=0):
    """Wrapper: delegates to session.trials.opto_mask(delta). See filtering.opto_mask."""
    return _opto_mask(session.trials, delta=delta)


def split_trials_by_opto(session) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper: opto and control masks. See filtering.opto_mask."""
    t = session.trials
    return _opto_mask(t, delta=0), _opto_mask(t, delta='control')


def get_post_opto_mask(session) -> np.ndarray:
    """Wrapper: post-opto trials (delta=1). See filtering.opto_mask."""
    return _opto_mask(session.trials, delta=1)


def extract_trial_arrays(session, mask):
    """
    DEPRECATED — use session.filter(mask).get_arrays() instead.
    
    Kept for backward compatibility. Returns None if < 10 trials.
    """
    if mask.sum() < 10:
        return None
    filtered = filter_session(session, mask, label='extract_trial_arrays')
    arr = get_arrays(filtered.trials)
    arr['n_trials'] = arr['n_trials']  # already there
    return arr


# ─── Within-session comparison ───────────────────────────────────────────────


def within_session_effect(
    session,
    n_permutations: int = 0,
    n_bootstrap: int = 0,
    seed: int = 42,
    min_trials: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Compare opto vs control trials within one session.

    Control = all non-opto valid trials (70%). Post-opto stats are
    computed as an overlay (subset of control) for carry-over analysis.

    Wraps compare_conditions() for the opto/control comparison.
    Set n_permutations/n_bootstrap > 0 for statistical tests.

    Returns dict with:
        opto_stats, control_stats: {accuracy, pse, slope, lapse_low, lapse_high}
        diff: same keys (opto - control)
        n_opto, n_control, n_post_opto: trial counts

        post_opto_stats: same keys (None if < min_trials)
        post_opto_diff: post_opto - control (None if too few)

        perm_p, boot_ci, fisher_p: tests for opto vs control

        um_opto, um_control: update matrices
        um_rmse, um_corr: UM comparison scalars

    Returns None if opto or control split has < min_trials valid trials.
    """
    from behav_utils.analysis.comparison import compare_conditions

    opto_mask, control_mask = split_trials_by_opto(session)
    opto_arrays = extract_trial_arrays(session, opto_mask)
    control_arrays = extract_trial_arrays(session, control_mask)

    if opto_arrays is None or control_arrays is None:
        return None

    valid_o = ~opto_arrays['no_response']
    valid_c = ~control_arrays['no_response']

    comp = compare_conditions(
        opto_arrays['stimuli'][valid_o],
        opto_arrays['choices'][valid_o],
        opto_arrays['categories'][valid_o],
        control_arrays['stimuli'][valid_c],
        control_arrays['choices'][valid_c],
        control_arrays['categories'][valid_c],
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        seed=seed,
        label_a='opto',
        label_b='control',
    )

    post_opto_mask = get_post_opto_mask(session)
    n_post = int(post_opto_mask.sum())

    result = {
        'opto_stats': comp['params_a'],
        'control_stats': comp['params_b'],
        'diff': comp['diffs'],
        'n_opto': comp['n_a'],
        'n_control': comp['n_b'],
        'n_post_opto': n_post,
        'perm_p': comp.get('perm_p'),
        'boot_ci': comp.get('boot_ci'),
        'fisher_p': comp.get('fisher_p', np.nan),
        'um_opto': comp.get('um_a'),
        'um_control': comp.get('um_b'),
        'um_rmse': comp.get('um_rmse', np.nan),
        'um_corr': comp.get('um_corr', np.nan),
    }

    # Post-opto overlay (subset of control)
    post_arrays = extract_trial_arrays(session, post_opto_mask)
    if post_arrays is not None and post_arrays['n_trials'] >= min_trials:
        from behav_utils.analysis.comparison import _fit_params, _accuracy
        valid_p = ~post_arrays['no_response']
        stim_p = post_arrays['stimuli'][valid_p]
        ch_p = post_arrays['choices'][valid_p]
        cat_p = post_arrays['categories'][valid_p]

        post_params = _fit_params(stim_p, ch_p) or {}
        post_params['accuracy'] = _accuracy(ch_p, cat_p)
        result['post_opto_stats'] = post_params

        # Diff relative to full control
        result['post_opto_diff'] = {
            k: post_params.get(k, np.nan) - comp['params_b'].get(k, np.nan)
            for k in ('accuracy', 'pse', 'slope', 'lapse_low', 'lapse_high')
        }
    else:
        result['post_opto_stats'] = None
        result['post_opto_diff'] = None

    return result


# ─── Phase-pooled comparison ─────────────────────────────────────────────────

def _pool_trial_arrays(sessions, mask_fn):
    """Pool trials across sessions via filter_trials + pool_arrays."""
    filtered = filter_trials(sessions, mask_fn=mask_fn, min_trials=10, label='opto pool')
    if not filtered:
        return None
    arr = pool_arrays(filtered)
    if arr['n_trials'] == 0:
        return None
    return arr


def phase_pooled_comparison(
    sessions: list,
    phases: List[OptoPhase],
    target_phase: OptoPhase,
    n_bins: int = 8,
    n_permutations: int = 0,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """
    Pool all sessions in target_phase, compare opto vs control.

    Control = all non-opto trials (70%). Post-opto stats computed
    as an overlay (subset of control).

    Returns dict with:
        opto_stats, control_stats: {accuracy, pse, slope, ...}
        diff: opto - control diffs
        opto_um, control_um: update matrices
        um_rmse, um_corr: UM comparison scalars
        perm_p, boot_ci, fisher_p: statistical tests

        post_opto_stats: same (None if < 10 pooled trials)
        post_opto_diff: post_opto - control

        n_sessions, n_opto, n_control, n_post_opto: counts
    """
    from behav_utils.analysis.comparison import compare_conditions, _fit_params, _accuracy

    phase_sessions = [
        s for s, p in zip(sessions, phases) if p == target_phase
    ]
    if not phase_sessions:
        return None

    opto_arrays = _pool_trial_arrays(
        phase_sessions, lambda s: split_trials_by_opto(s)[0])
    control_arrays = _pool_trial_arrays(
        phase_sessions, lambda s: split_trials_by_opto(s)[1])

    if opto_arrays is None or control_arrays is None:
        return None

    valid_o = ~opto_arrays['no_response']
    valid_c = ~control_arrays['no_response']

    comp = compare_conditions(
        opto_arrays['stimuli'][valid_o],
        opto_arrays['choices'][valid_o],
        opto_arrays['categories'][valid_o],
        control_arrays['stimuli'][valid_c],
        control_arrays['choices'][valid_c],
        control_arrays['categories'][valid_c],
        n_bins=n_bins,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        seed=seed,
        label_a='opto',
        label_b='control',
    )

    result = {
        'opto_stats': comp['params_a'],
        'control_stats': comp['params_b'],
        'diff': comp['diffs'],
        'opto_um': comp['um_a'],
        'control_um': comp['um_b'],
        'um_rmse': comp['um_rmse'],
        'um_corr': comp['um_corr'],
        'perm_p': comp.get('perm_p'),
        'boot_ci': comp.get('boot_ci'),
        'fisher_p': comp.get('fisher_p', np.nan),
        'n_sessions': len(phase_sessions),
        'n_opto': comp['n_a'],
        'n_control': comp['n_b'],
    }

    # Post-opto overlay
    post_arrays = _pool_trial_arrays(
        phase_sessions, lambda s: get_post_opto_mask(s))

    if post_arrays is not None and post_arrays['n_trials'] >= 10:
        valid_p = ~post_arrays['no_response']
        stim_p = post_arrays['stimuli'][valid_p]
        ch_p = post_arrays['choices'][valid_p]
        cat_p = post_arrays['categories'][valid_p]

        post_params = _fit_params(stim_p, ch_p) or {}
        post_params['accuracy'] = _accuracy(ch_p, cat_p)
        result['post_opto_stats'] = post_params
        result['n_post_opto'] = int(valid_p.sum())
        result['post_opto_diff'] = {
            k: post_params.get(k, np.nan) - comp['params_b'].get(k, np.nan)
            for k in ('accuracy', 'pse', 'slope', 'lapse_low', 'lapse_high')
        }
    else:
        result['post_opto_stats'] = None
        result['post_opto_diff'] = None
        result['n_post_opto'] = 0

    return result


# ─── Update matrix from opto-relative trials ─────────────────────────────────

def compute_opto_um(
    sessions: list,
    delta: Optional[Union[int, str]] = 0,
    n_bins: int = 8,
    opto_only: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute pooled update matrix from opto-relative trials.

    Uses opto_relative_mask to select trials at a fixed offset from
    opto events, then pools across sessions and computes the UM.

    Args:
        sessions: List of SessionData
        delta: Trial selection relative to opto events.
            None      → all valid trials
            0         → opto trials only
            1         → post-opto (1st valid non-opto after each opto run)
            2         → 2nd valid non-opto after each opto run
            -1        → pre-opto (last valid non-opto before each opto run)
            'control' → all valid non-opto trials
        n_bins: Number of stimulus bins.
        opto_only: DEPRECATED — backward compat only. If delta is left
            at its default (0) and opto_only is explicitly passed,
            maps True → delta=0, False → delta='control'.

    Returns:
        (update_matrix, counts, info_dict)
        info_dict includes 'n_trials' and 'delta'.
    """
    # Backward compat: old callers pass opto_only=True/False
    if opto_only is not None and delta == 0:
        delta = 0 if opto_only else 'control'

    arrays = _pool_trial_arrays(
        sessions, lambda s: _opto_mask(s.trials, delta=delta))

    if arrays is None:
        empty = np.full((n_bins, n_bins), np.nan)
        return empty, empty, {'n_trials': 0, 'delta': delta}

    um, counts, info = compute_update_matrix(
        arrays['stimuli'], arrays['choices'],
        arrays['categories'], n_bins=n_bins)
    info['delta'] = delta
    return um, counts, info


# ─── Expert stability ────────────────────────────────────────────────────────

def phase_stability(
    sessions: list,
    phases: List[OptoPhase],
    stat_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Track statistics across experimental phases and compare all pairs.

    Replaces expert_stability with full phase coverage including masking.
    Filters each session (exclude abort + opto) before computing stats —
    this tests whether control-trial performance changes across phases.

    Compares all pairs: baseline↔masking, baseline↔opto,
    masking↔opto, baseline↔washout, opto↔washout.
    Uses Mann-Whitney U (two-sided). NaN values are filtered
    before testing.

    Args:
        sessions: List of SessionData
        phases: List of OptoPhase (same length as sessions)
        stat_names: Which stats to track. Default: all 5 psychometric
            params. Must match names returned by sess.stats().

    Returns dict with:
        per_phase: {OptoPhase: {stat_name: np.array of values}}
        phase_means: {OptoPhase: {stat_name: float}}
        phase_n: {OptoPhase: int}
        comparisons: {(phase_a, phase_b): {
            stat_name: {'p': float, 'u': float, 'diff': float}
        }}
    """
    if stat_names is None:
        stat_names = ['accuracy', 'pse', 'slope', 'lapse_low', 'lapse_high']

    target_phases = [
        OptoPhase.EXPERT_BASELINE,
        OptoPhase.MASKING,
        OptoPhase.EXPERT_OPTO,
        OptoPhase.EXPERT_WASHOUT,
    ]

    # Collect per-phase stats
    per_phase = {p: {s: [] for s in stat_names} for p in target_phases}

    for sess, phase in zip(sessions, phases):
        if phase not in target_phases:
            continue

        # Filter once per session (exclude abort + opto → control trials only)
        clean = filter_session(sess)  # standard exclusions
        for sn in stat_names:
            try:
                st = clean.stats(stat_names=[sn])
                val = st[sn]
                if val is not None and not np.isnan(val):
                    per_phase[phase][sn].append(float(val))
            except Exception:
                pass

    # Convert to arrays
    for p in target_phases:
        for sn in stat_names:
            per_phase[p][sn] = np.array(per_phase[p][sn])

    # Means and counts
    phase_means = {}
    phase_n = {}
    for p in target_phases:
        phase_means[p] = {}
        # Count from first stat (all should have same n)
        first_stat = stat_names[0]
        phase_n[p] = len(per_phase[p][first_stat])
        for sn in stat_names:
            vals = per_phase[p][sn]
            phase_means[p][sn] = float(np.mean(vals)) if len(vals) > 0 else np.nan

    # Pairwise comparisons
    pairs = [
        (OptoPhase.EXPERT_BASELINE, OptoPhase.MASKING),
        (OptoPhase.EXPERT_BASELINE, OptoPhase.EXPERT_OPTO),
        (OptoPhase.MASKING, OptoPhase.EXPERT_OPTO),
        (OptoPhase.EXPERT_BASELINE, OptoPhase.EXPERT_WASHOUT),
        (OptoPhase.EXPERT_OPTO, OptoPhase.EXPERT_WASHOUT),
    ]

    comparisons = {}
    for pa, pb in pairs:
        comparisons[(pa, pb)] = {}
        for sn in stat_names:
            vals_a = per_phase[pa][sn]
            vals_b = per_phase[pb][sn]

            comp = {'p': np.nan, 'u': np.nan, 'diff': np.nan}

            if len(vals_a) > 0 and len(vals_b) > 0:
                comp['diff'] = float(np.mean(vals_a) - np.mean(vals_b))

            if len(vals_a) >= 2 and len(vals_b) >= 2:
                try:
                    u, p = mannwhitneyu(
                        vals_a, vals_b, alternative='two-sided')
                    comp['p'] = float(p)
                    comp['u'] = float(u)
                except Exception:
                    pass

            comparisons[(pa, pb)][sn] = comp

    return {
        'per_phase': per_phase,
        'phase_means': phase_means,
        'phase_n': phase_n,
        'comparisons': comparisons,
        'stat_names': stat_names,
    }


# Backward compatibility — thin wrapper
def expert_stability(
    sessions: list,
    phases: List[OptoPhase],
    stat_name: str = 'accuracy',
) -> Dict[str, Any]:
    """
    Track a statistic across expert baseline → opto → washout.

    Backward-compatible wrapper around phase_stability().
    Prefer phase_stability() for new code — it includes masking
    and tests all phase pairs.
    """
    full = phase_stability(sessions, phases, stat_names=[stat_name])

    baseline = full['per_phase'][OptoPhase.EXPERT_BASELINE][stat_name]
    opto = full['per_phase'][OptoPhase.EXPERT_OPTO][stat_name]
    washout = full['per_phase'][OptoPhase.EXPERT_WASHOUT][stat_name]

    comp = full['comparisons'].get(
        (OptoPhase.EXPERT_BASELINE, OptoPhase.EXPERT_OPTO), {})
    comp_stat = comp.get(stat_name, {})

    return {
        'baseline_values': baseline,
        'opto_values': opto,
        'washout_values': washout,
        'baseline_mean': float(np.nanmean(baseline)) if len(baseline) else np.nan,
        'opto_mean': float(np.nanmean(opto)) if len(opto) else np.nan,
        'washout_mean': float(np.nanmean(washout)) if len(washout) else np.nan,
        'p_value': comp_stat.get('p', np.nan),
        'u_statistic': comp_stat.get('u', np.nan),
    }


# ─── Genotype interaction ────────────────────────────────────────────────────

def genotype_interaction(
    het_effects: List[Dict],
    wt_effects: List[Dict],
    metric: str = 'accuracy',
) -> Dict[str, Any]:
    """
    Compare within-animal opto effect sizes across genotypes.

    Tests: (het opto−control) vs (WT opto−control).

    Args:
        het_effects: List of within_session_effect dicts for het animals
        wt_effects: List of within_session_effect dicts for WT animals
        metric: Which diff metric to compare ('accuracy', 'pse', 'slope')

    Returns dict with:
        het_diffs, wt_diffs: arrays of effect sizes
        het_mean, wt_mean: mean effect per genotype
        p_value: Mann-Whitney U comparing effect distributions
        interaction: het_mean - wt_mean
    """
    het_diffs = np.array([
        e['diff'][metric] for e in het_effects
        if e is not None and not np.isnan(e['diff'].get(metric, np.nan))
    ])
    wt_diffs = np.array([
        e['diff'][metric] for e in wt_effects
        if e is not None and not np.isnan(e['diff'].get(metric, np.nan))
    ])

    result = {
        'het_diffs': het_diffs,
        'wt_diffs': wt_diffs,
        'het_mean': float(np.mean(het_diffs)) if len(het_diffs) else np.nan,
        'wt_mean': float(np.mean(wt_diffs)) if len(wt_diffs) else np.nan,
        'metric': metric,
    }
    result['interaction'] = result['het_mean'] - result['wt_mean']

    if len(het_diffs) >= 2 and len(wt_diffs) >= 2:
        try:
            stat, p = mannwhitneyu(
                het_diffs, wt_diffs, alternative='two-sided')
            result['p_value'] = float(p)
        except Exception:
            result['p_value'] = np.nan
    else:
        result['p_value'] = np.nan

    return result


# ─── Convenience: full animal report ─────────────────────────────────────────

def animal_opto_report(animal) -> Dict[str, Any]:
    """
    Run all opto analyses for one animal.

    Masking is read from session.masking (set by load_experiment).

    Returns dict with:
        animal_id: str
        genotype: str
        phases: list of OptoPhase per session
        within_session: list of per-opto-session effect dicts
        expert_stability: expert baseline/opto/washout comparison
        phase_comparisons: dict[OptoPhase, phase_pooled_comparison]
    """
    phases = assign_opto_phases(animal)
    sessions = animal.sessions

    within = []
    for idx, (sess, phase) in enumerate(zip(sessions, phases)):
        if phase in (OptoPhase.EXPERT_OPTO, OptoPhase.SHIFT_1_OPTO,
                     OptoPhase.SHIFT_2_OPTO, OptoPhase.MASKING):
            within.append({
                'phase': phase,
                'session_idx': idx,
                'effect': within_session_effect(sess),
            })

    exp_stab = expert_stability(sessions, phases, stat_name='accuracy')

    phase_comparisons = {}
    for target in (OptoPhase.EXPERT_OPTO, OptoPhase.MASKING,
                   OptoPhase.SHIFT_1_OPTO, OptoPhase.SHIFT_2_OPTO):
        comp = phase_pooled_comparison(sessions, phases, target)
        if comp is not None:
            phase_comparisons[target] = comp

    return {
        'animal_id': animal.animal_id,
        'genotype': animal.genotype,
        'phases': phases,
        'within_session': within,
        'expert_stability': exp_stab,
        'phase_comparisons': phase_comparisons,
    }


# ─── Cohort-level report ─────────────────────────────────────────────────────

def cohort_opto_report(
    experiment,
    target_phase: OptoPhase = OptoPhase.EXPERT_OPTO,
    metric: str = 'accuracy',
) -> Dict[str, Any]:
    """
    Run opto analysis for all animals, split by genotype.

    Reads animal.genotype ('het' or 'wt') to separate groups.

    Args:
        experiment: ExperimentData
        target_phase: Which phase to compare
        metric: Effect metric for interaction test

    Returns dict with:
        reports: {animal_id: animal_opto_report}
        het_effects, wt_effects: separated within-session effects
        interaction: genotype_interaction result
    """
    reports = {}
    het_effects, wt_effects = [], []

    for aid, animal in experiment.animals.items():
        report = animal_opto_report(animal)
        reports[aid] = report

        for entry in report['within_session']:
            if entry['phase'] == target_phase and entry['effect'] is not None:
                if animal.genotype == 'het':
                    het_effects.append(entry['effect'])
                elif animal.genotype == 'wt':
                    wt_effects.append(entry['effect'])

    interaction = genotype_interaction(het_effects, wt_effects, metric=metric)

    return {
        'reports': reports,
        'het_effects': het_effects,
        'wt_effects': wt_effects,
        'interaction': interaction,
    }


# ─── Model-assignment integration ────────────────────────────────────────────

def _collect_phase_effects(
    report: Dict[str, Any],
    target_phase: OptoPhase,
    metric: str = 'accuracy',
) -> List[float]:
    """
    Extract per-session opto effect sizes for a given phase.

    Returns list of diff values (opto - control) for the given metric.
    """
    effects = []
    for entry in report['within_session']:
        if entry['phase'] != target_phase:
            continue
        if entry['effect'] is None:
            continue
        val = entry['effect']['diff'].get(metric, np.nan)
        if not np.isnan(val):
            effects.append(val)
    return effects


def opto_by_model_assignment(
    experiment,
    consensus_df,
    target_phase: OptoPhase = OptoPhase.EXPERT_OPTO,
    metric: str = 'accuracy',
    assignment_col: str = 'consensus',
) -> Dict[str, Any]:
    """
    Group opto effects by model assignment (BE / SC / unclear).

    Bridges consensus model selection (from analysis.consensus) with
    opto analysis. For each animal, looks up its consensus assignment
    and groups effects accordingly.

    Args:
        experiment: ExperimentData
        consensus_df: DataFrame with animal_id index and assignment_col
            column containing 'BE', 'SC', or 'unclear'. Produced by
            consensus_summary() or load_all_assignments().
        target_phase: Which opto phase to analyse.
        metric: Which metric to use for effect sizes.
        assignment_col: Column name in consensus_df for model assignment.

    Returns dict with:
        reports: {animal_id: animal_opto_report} for all opto animals
        groups: {assignment: {
            animal_ids: list,
            effects: list of within_session_effect dicts,
            diffs: np.array of per-session effect sizes,
            mean_diff: float,
            n_animals: int,
            n_sessions: int,
        }}
        comparison: {
            'be_vs_sc_p': float (Mann-Whitney, BE diffs vs SC diffs),
            'be_vs_sc_u': float,
        }
    """
    # Build assignment lookup
    if hasattr(consensus_df, 'index') and consensus_df.index.name == 'animal_id':
        assignment_map = consensus_df[assignment_col].to_dict()
    elif 'animal_id' in consensus_df.columns:
        assignment_map = dict(zip(
            consensus_df['animal_id'],
            consensus_df[assignment_col],
        ))
    else:
        raise ValueError(
            "consensus_df must have 'animal_id' as index or column"
        )

    # Run reports for all opto animals
    reports = {}
    groups = {
        'BE': {'animal_ids': [], 'effects': [], 'diffs': []},
        'SC': {'animal_ids': [], 'effects': [], 'diffs': []},
        'unclear': {'animal_ids': [], 'effects': [], 'diffs': []},
    }

    for aid, animal in experiment.animals.items():
        report = animal_opto_report(animal)

        # Skip animals with no sessions in target phase
        has_target = any(
            e['phase'] == target_phase for e in report['within_session']
        )
        if not has_target:
            continue

        reports[aid] = report
        assignment = assignment_map.get(aid, 'unclear')

        # Normalise assignment label
        if assignment.upper() == 'BE':
            group_key = 'BE'
        elif assignment.upper() == 'SC':
            group_key = 'SC'
        else:
            group_key = 'unclear'

        groups[group_key]['animal_ids'].append(aid)

        for entry in report['within_session']:
            if entry['phase'] != target_phase or entry['effect'] is None:
                continue
            groups[group_key]['effects'].append(entry['effect'])
            val = entry['effect']['diff'].get(metric, np.nan)
            if not np.isnan(val):
                groups[group_key]['diffs'].append(val)

    # Finalise group summaries
    for key, grp in groups.items():
        grp['diffs'] = np.array(grp['diffs'])
        grp['mean_diff'] = (
            float(np.mean(grp['diffs'])) if len(grp['diffs']) else np.nan
        )
        grp['n_animals'] = len(grp['animal_ids'])
        grp['n_sessions'] = len(grp['effects'])

    # BE vs SC comparison
    comparison = {'be_vs_sc_p': np.nan, 'be_vs_sc_u': np.nan}
    be_diffs = groups['BE']['diffs']
    sc_diffs = groups['SC']['diffs']
    if len(be_diffs) >= 2 and len(sc_diffs) >= 2:
        try:
            u, p = mannwhitneyu(be_diffs, sc_diffs, alternative='two-sided')
            comparison['be_vs_sc_p'] = float(p)
            comparison['be_vs_sc_u'] = float(u)
        except Exception:
            pass

    return {
        'reports': reports,
        'groups': groups,
        'comparison': comparison,
        'metric': metric,
        'target_phase': target_phase,
    }


# ─── Expert null prediction testing ─────────────────────────────────────────

def expert_null_test(
    reports: Dict[str, Dict],
    metric: str = 'accuracy',
    equivalence_bound: float = 0.05,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Test the null prediction: PPC inactivation has no effect during expert phase.

    Returns both standard tests (t-test, Mann-Whitney) and TOST equivalence
    testing. The standard tests ask "is there an effect?"; TOST asks "is the
    effect negligibly small?". Both are needed:

    - t-test p < 0.05 → evidence that opto does something (bad for null prediction)
    - TOST p < 0.05 → positive evidence the effect is within ±bound (good for null)
    - Neither significant → inconclusive (underpowered)

    One effect value per animal (mean across that animal's expert opto
    sessions), so each animal contributes one data point. This avoids
    pseudoreplication from pooling sessions.

    The equivalence_bound should be set based on what would constitute
    a meaningful behavioural impairment. For accuracy, 0.05 (5 percentage
    points) is a reasonable default — smaller than typical opto effects
    in similar paradigms (Pinto et al. 2019 report ~10–15% accuracy drops).

    Args:
        reports: {animal_id: animal_opto_report dict}. Only animals with
            EXPERT_OPTO sessions contribute.
        metric: Which diff metric to test.
        equivalence_bound: Symmetric bound for TOST (default ±0.05).
        alpha: Significance level (default 0.05).

    Returns dict with:
        animal_means: {animal_id: mean effect} — one value per animal
        effects: np.array of per-animal mean effects
        grand_mean: float
        grand_sem: float
        n_animals: int

        ttest_p: float — standard two-sided t-test p-value (H0: mean == 0)
        ttest_t: float — t statistic
        mann_whitney_p: float — Mann-Whitney on per-session opto vs control
            accuracy values (unpaired, non-parametric alternative)

        tost_p: float — max of the two one-sided p-values (reject if < alpha)
        tost_reject: bool — True = evidence for equivalence
        tost_lower_p: float — p for H0: mean <= -bound
        tost_upper_p: float — p for H0: mean >= +bound

        equivalence_bound: float — the bound used
        ci_90: (float, float) — 90% CI (appropriate for TOST at alpha=0.05)
    """
    # Compute per-animal mean effect
    animal_means = {}
    for aid, report in reports.items():
        effects = _collect_phase_effects(
            report, OptoPhase.EXPERT_OPTO, metric=metric)
        if effects:
            animal_means[aid] = float(np.mean(effects))

    effects = np.array(list(animal_means.values()))
    n = len(effects)

    result = {
        'animal_means': animal_means,
        'effects': effects,
        'grand_mean': float(np.mean(effects)) if n > 0 else np.nan,
        'grand_sem': (
            float(np.std(effects, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        ),
        'n_animals': n,
        'equivalence_bound': equivalence_bound,
        'metric': metric,
    }

    if n < 3:
        result.update({
            'ttest_p': np.nan, 'ttest_t': np.nan,
            'mann_whitney_p': np.nan,
            'tost_p': np.nan, 'tost_reject': False,
            'tost_lower_p': np.nan, 'tost_upper_p': np.nan,
            'ci_90': (np.nan, np.nan),
            'warning': f'Only {n} animals — need >= 3 for testing.',
        })
        return result

    # ── Standard tests (H0: no effect) ──────────────────────────────────
    # One-sample t-test on per-animal means
    t_stat, p_two = ttest_1samp(effects, 0)
    result['ttest_t'] = float(t_stat)
    result['ttest_p'] = float(p_two)

    # Mann-Whitney: pool per-session opto vs control accuracy across animals
    # (non-parametric alternative, doesn't assume normality)
    opto_accs, ctrl_accs = [], []
    for report in reports.values():
        for entry in report['within_session']:
            if entry['phase'] != OptoPhase.EXPERT_OPTO:
                continue
            if entry['effect'] is None:
                continue
            opto_accs.append(entry['effect']['opto_stats'].get('accuracy', np.nan))
            ctrl_accs.append(entry['effect']['control_stats'].get('accuracy', np.nan))
    opto_accs = np.array([v for v in opto_accs if not np.isnan(v)])
    ctrl_accs = np.array([v for v in ctrl_accs if not np.isnan(v)])

    if len(opto_accs) >= 2 and len(ctrl_accs) >= 2:
        try:
            _, mw_p = mannwhitneyu(opto_accs, ctrl_accs, alternative='two-sided')
            result['mann_whitney_p'] = float(mw_p)
        except Exception:
            result['mann_whitney_p'] = np.nan
    else:
        result['mann_whitney_p'] = np.nan

    # ── TOST (H0: |effect| >= bound) ────────────────────────────────────
    # Lower: H0: mean <= -bound  (test: is mean > -bound?)
    t_lower, p_lower_two = ttest_1samp(effects, -equivalence_bound)
    p_lower = p_lower_two / 2 if t_lower > 0 else 1 - p_lower_two / 2

    # Upper: H0: mean >= +bound  (test: is mean < +bound?)
    t_upper, p_upper_two = ttest_1samp(effects, equivalence_bound)
    p_upper = p_upper_two / 2 if t_upper < 0 else 1 - p_upper_two / 2

    tost_p = max(p_lower, p_upper)

    result['tost_lower_p'] = float(p_lower)
    result['tost_upper_p'] = float(p_upper)
    result['tost_p'] = float(tost_p)
    result['tost_reject'] = tost_p < alpha

    # 90% CI (1 - 2*alpha CI is appropriate for TOST)
    from scipy.stats import t as t_dist
    se = np.std(effects, ddof=1) / np.sqrt(n)
    t_crit = t_dist.ppf(1 - alpha, df=n - 1)
    ci_low = float(np.mean(effects) - t_crit * se)
    ci_high = float(np.mean(effects) + t_crit * se)
    result['ci_90'] = (ci_low, ci_high)

    return result


# ─── Expert UM comparison ────────────────────────────────────────────────────

def expert_um_test(
    experiment_or_animals,
    reports: Dict[str, Dict],
    n_bins: int = 8,
    equivalence_bound: float = 0.02,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Test the null prediction via update matrix comparison.

    For each animal, pools expert-opto sessions and computes separate UMs
    for opto and control trials. Compares them with:
    - UM RMSE: element-wise RMSE between opto and control UMs
    - UM correlation: Pearson r of flattened (non-NaN) UM elements
    - Per-animal UM difference heatmaps (stored for plotting)

    Then runs cohort-level tests on the per-animal UM RMSE values
    (t-test + TOST, same logic as expert_null_test).

    The equivalence bound for UM RMSE should be set based on what
    constitutes a meaningful UM distortion. Default 0.02 is conservative
    — typical inter-session UM variability in expert animals is ~0.01–0.03.

    Args:
        experiment_or_animals: ExperimentData or dict of {aid: AnimalData}
        reports: {animal_id: animal_opto_report dict}
        n_bins: UM bin count (must match analysis defaults)
        equivalence_bound: Symmetric bound for TOST on UM RMSE.
        alpha: Significance level.

    Returns dict with:
        per_animal: {animal_id: {
            opto_um, control_um, diff_um: (n_bins, n_bins) arrays,
            um_rmse: float,
            um_corr: float (Pearson r of flattened elements),
            n_opto_trials, n_control_trials: int,
        }}
        rmse_values: np.array of per-animal UM RMSEs
        corr_values: np.array of per-animal UM correlations
        rmse_mean, corr_mean: float

        ttest_p: float (H0: mean RMSE == 0 — note: RMSE is always >= 0,
            so this tests whether it's significantly above zero)
        tost_p: float (H0: RMSE >= bound — tests whether RMSE is small)
        tost_reject: bool
        equivalence_bound: float
    """
    from scipy.stats import pearsonr as _pearsonr

    # Get animals dict
    if hasattr(experiment_or_animals, 'animals'):
        animals = experiment_or_animals.animals
    else:
        animals = experiment_or_animals

    per_animal = {}

    for aid, report in reports.items():
        animal = animals.get(aid)
        if animal is None:
            continue

        phases = report['phases']
        expert_opto_sessions = [
            s for s, p in zip(animal.sessions, phases)
            if p == OptoPhase.EXPERT_OPTO
        ]
        if not expert_opto_sessions:
            continue

        # Compute pooled UMs for opto and control trials
        opto_um, _, _ = compute_opto_um(
            expert_opto_sessions, opto_only=True, n_bins=n_bins)
        ctrl_um, _, _ = compute_opto_um(
            expert_opto_sessions, opto_only=False, n_bins=n_bins)

        # Element-wise comparison (only where both are valid)
        valid = ~np.isnan(opto_um) & ~np.isnan(ctrl_um)
        if valid.sum() < 4:
            continue

        diff = opto_um - ctrl_um
        rmse = float(np.sqrt(np.nanmean(diff[valid] ** 2)))

        opto_flat = opto_um[valid]
        ctrl_flat = ctrl_um[valid]
        if np.std(opto_flat) > 1e-8 and np.std(ctrl_flat) > 1e-8:
            corr, _ = _pearsonr(opto_flat, ctrl_flat)
        else:
            corr = np.nan

        # Count trials
        n_opto = sum(
            split_trials_by_opto(s)[0].sum() for s in expert_opto_sessions)
        n_ctrl = sum(
            split_trials_by_opto(s)[1].sum() for s in expert_opto_sessions)

        per_animal[aid] = {
            'opto_um': opto_um,
            'control_um': ctrl_um,
            'diff_um': diff,
            'um_rmse': rmse,
            'um_corr': float(corr),
            'n_opto_trials': int(n_opto),
            'n_control_trials': int(n_ctrl),
        }

    rmse_values = np.array([v['um_rmse'] for v in per_animal.values()])
    corr_values = np.array([
        v['um_corr'] for v in per_animal.values()
        if not np.isnan(v['um_corr'])
    ])
    n = len(rmse_values)

    result = {
        'per_animal': per_animal,
        'rmse_values': rmse_values,
        'corr_values': corr_values,
        'rmse_mean': float(np.mean(rmse_values)) if n > 0 else np.nan,
        'corr_mean': float(np.mean(corr_values)) if len(corr_values) > 0 else np.nan,
        'n_animals': n,
        'equivalence_bound': equivalence_bound,
        'n_bins': n_bins,
    }

    if n < 3:
        result.update({
            'ttest_p': np.nan, 'ttest_t': np.nan,
            'tost_p': np.nan, 'tost_reject': False,
            'warning': f'Only {n} animals — need >= 3 for testing.',
        })
        return result

    # t-test: is RMSE significantly above zero?
    t_stat, p_two = ttest_1samp(rmse_values, 0)
    result['ttest_t'] = float(t_stat)
    result['ttest_p'] = float(p_two)

    # TOST: is RMSE within [0, equivalence_bound]?
    # One-sided: H0: mean >= bound → test: is mean < bound?
    t_upper, p_upper_two = ttest_1samp(rmse_values, equivalence_bound)
    p_upper = p_upper_two / 2 if t_upper < 0 else 1 - p_upper_two / 2
    # Lower bound is 0 (RMSE can't be negative), so only one test needed
    result['tost_p'] = float(p_upper)
    result['tost_reject'] = p_upper < alpha

    return result


# ─── Phase × opto interaction ────────────────────────────────────────────────

def phase_opto_interaction(
    reports: Dict[str, Dict],
    expert_phase: OptoPhase = OptoPhase.EXPERT_OPTO,
    shift_phase: OptoPhase = OptoPhase.SHIFT_1_OPTO,
    metric: str = 'accuracy',
) -> Dict[str, Any]:
    """
    Test whether the opto effect differs between expert and post-shift phases.

    This is the key interaction: the hypothesis predicts that PPC inactivation
    impairs performance more during post-shift (when the model is inadequate)
    than during expert (when the model is adequate).

    Design: same animal appears in both phases, so this is a paired
    within-animal comparison. Per-animal mean opto effect is computed
    for each phase, then tested with both paired t-test and Wilcoxon
    signed-rank test.

    Only animals that have data in BOTH phases contribute. With the
    current experimental timeline, this won't be available until
    post-shift opto data arrives — the function returns a warning
    and empty results if no paired animals exist yet.

    Args:
        reports: {animal_id: animal_opto_report dict}
        expert_phase: Phase to use as "expert" (default EXPERT_OPTO)
        shift_phase: Phase to use as "post-shift" (default SHIFT_1_OPTO)
        metric: Which diff metric to compare

    Returns dict with:
        paired_animals: list of animal_ids with data in both phases
        expert_effects: np.array of per-animal mean effects in expert phase
        shift_effects: np.array of per-animal mean effects in shift phase
        interaction_diffs: np.array (shift_effect - expert_effect per animal)
        n_paired: int

        expert_mean, shift_mean: float
        interaction_mean: float (shift_mean - expert_mean)

        paired_ttest_t: float
        paired_ttest_p: float
        wilcoxon_stat: float
        wilcoxon_p: float
    """
    # Collect per-animal means for each phase
    expert_means = {}
    shift_means = {}

    for aid, report in reports.items():
        exp_effs = _collect_phase_effects(report, expert_phase, metric)
        shf_effs = _collect_phase_effects(report, shift_phase, metric)

        if exp_effs:
            expert_means[aid] = float(np.mean(exp_effs))
        if shf_effs:
            shift_means[aid] = float(np.mean(shf_effs))

    # Find animals with data in both phases
    paired_ids = sorted(set(expert_means.keys()) & set(shift_means.keys()))
    n_paired = len(paired_ids)

    expert_arr = np.array([expert_means[aid] for aid in paired_ids])
    shift_arr = np.array([shift_means[aid] for aid in paired_ids])
    interaction_diffs = shift_arr - expert_arr

    result = {
        'paired_animals': paired_ids,
        'expert_effects': expert_arr,
        'shift_effects': shift_arr,
        'interaction_diffs': interaction_diffs,
        'n_paired': n_paired,
        'expert_phase': expert_phase,
        'shift_phase': shift_phase,
        'metric': metric,
        'expert_mean': float(np.mean(expert_arr)) if n_paired > 0 else np.nan,
        'shift_mean': float(np.mean(shift_arr)) if n_paired > 0 else np.nan,
        'interaction_mean': (
            float(np.mean(interaction_diffs)) if n_paired > 0 else np.nan
        ),
        # Also report unpaired animals for completeness
        'expert_only_animals': sorted(
            set(expert_means.keys()) - set(shift_means.keys())),
        'shift_only_animals': sorted(
            set(shift_means.keys()) - set(expert_means.keys())),
    }

    if n_paired < 3:
        result.update({
            'paired_ttest_t': np.nan, 'paired_ttest_p': np.nan,
            'wilcoxon_stat': np.nan, 'wilcoxon_p': np.nan,
            'warning': (
                f'Only {n_paired} animals with data in both phases. '
                f'Need >= 3 for paired testing. '
                f'{len(expert_means)} have expert data, '
                f'{len(shift_means)} have shift data.'
            ),
        })
        return result

    # Paired t-test
    t_stat, p_val = ttest_rel(shift_arr, expert_arr)
    result['paired_ttest_t'] = float(t_stat)
    result['paired_ttest_p'] = float(p_val)

    # Wilcoxon signed-rank (non-parametric paired test)
    try:
        w_stat, w_p = wilcoxon(interaction_diffs, alternative='two-sided')
        result['wilcoxon_stat'] = float(w_stat)
        result['wilcoxon_p'] = float(w_p)
    except ValueError:
        # wilcoxon can fail if all differences are zero
        result['wilcoxon_stat'] = np.nan
        result['wilcoxon_p'] = np.nan

    return result


# ─── Opto simulation (predictions) ──────────────────────────────────────────

def simulate_with_opto(
    model_type, params, stimuli, categories, opto_mask,
    lesion_type='null', lesion_target='choice',
    burn_in=1000, seed=42,
):
    """
    Simulate a session with trial-level opto inactivation.

    Currently uses post-hoc approximation: runs a full control
    simulation, then replaces opto trial choices with lesioned
    choices. This does NOT capture downstream effects (opto trial
    feedback affecting future non-opto trials). For that, the
    model trial loop needs modification (see lesion_target='update').

    Moved from NB 50 to enable reuse across notebooks.

    Args:
        model_type: 'BE' or 'SC'
        params: BEParams or SCParams
        stimuli: (n_trials,) stimulus array
        categories: (n_trials,) category array
        opto_mask: (n_trials,) boolean (True = opto on)
        lesion_type: 'null' (random choice) or 'attenuation' (biased flip)
        lesion_target: 'choice' (affect decision) or 'update' (affect learning)
        burn_in: burn-in trials
        seed: random seed

    Returns:
        choices_opto: simulated choices with opto effects
        choices_ctrl: what choices would have been without opto
    """
    from models.BE_core import BEModel
    from models.SC_core import SCModel

    rng = np.random.default_rng(seed)

    # Control: full simulation without opto
    rng_ctrl = np.random.default_rng(seed)
    if model_type.upper() == 'BE':
        state_ctrl = BEModel.create_initial_state(
            burn_in=burn_in, params=params, seed=seed)
        choices_ctrl, _, _, _ = BEModel.simulate_session(
            params, state_ctrl, stimuli, categories, rng_ctrl)
    else:
        state_ctrl = SCModel.create_initial_state(
            params=params, burn_in=burn_in, seed=seed)
        choices_ctrl, _, _, _ = SCModel.simulate_session(
            params, state_ctrl, stimuli, categories, rng_ctrl)

    # Opto: post-hoc replacement on opto trials
    choices_opto = choices_ctrl.copy()

    if lesion_target == 'choice':
        if lesion_type == 'null':
            choices_opto[opto_mask] = rng.choice(
                [0, 1], size=opto_mask.sum())
        elif lesion_type == 'attenuation':
            flip = rng.random(opto_mask.sum()) < 0.3
            choices_opto[opto_mask] = np.where(
                flip, 1 - choices_ctrl[opto_mask], choices_ctrl[opto_mask])

    elif lesion_target == 'update':
        if lesion_type == 'null':
            # TODO: requires trial-loop modification in BE_core/SC_core
            # to accept per-trial eta=0 on opto trials.
            pass

    return choices_opto, choices_ctrl
