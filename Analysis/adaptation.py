"""
Adaptation Analysis Module

Characterises how animals adapt when task demands change.
Core pattern: every analysis function takes two List[SessionData]
(baseline and post), never does its own session selection.

Note:
    User-friendly stat names like 'pse', 'slope' are resolved to their
    parent registered stats ('psychometric') by SessionData.stats().
    This module does not need its own resolution layer.

Usage:
    from behav_utils.data.selection import select_sessions
    from analysis.adaptation import (
        detect_all_manipulations, adaptation_trajectory,
        fit_recovery_curve, compare_phases,
    )

    baseline = select_sessions(animal, 'expert_uniform')
    post_shift = select_sessions(animal, distribution='Hard-A')

    traj = adaptation_trajectory(baseline, post_shift)
    recovery = fit_recovery_curve(baseline, post_shift, stat='accuracy')
    comparison = compare_phases(baseline, post_shift)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, List, Dict, Tuple, Callable, Union, Any, TYPE_CHECKING
from scipy.optimize import curve_fit
from scipy.stats import wilcoxon, mannwhitneyu

if TYPE_CHECKING:
    from behav_utils.data.structures import AnimalData, SessionData


# =============================================================================
# MANIPULATION DETECTION
# =============================================================================

def detect_all_manipulations(
    animal: 'AnimalData',
    stage: Optional[str] = 'Full_Task_Cont',
) -> List[Dict]:
    """
    Detect all manipulation points in an animal's timeline.

    Returns list of dicts:
        {
            'type': 'distribution_shift' | 'rule_flip' | 'range_change',
            'session_idx': int,           # index within filtered sessions
            'global_session_idx': int,    # animal-wide session_idx
            'details': {...},
        }
    """
    sessions = animal.get_sessions(stage=stage) if stage else animal.sessions
    manipulations = []

    if len(sessions) < 2:
        return manipulations

    for i in range(1, len(sessions)):
        prev, curr = sessions[i - 1], sessions[i]

        if prev.distribution != curr.distribution:
            manipulations.append({
                'type': 'distribution_shift',
                'session_idx': i,
                'global_session_idx': curr.session_idx,
                'details': {'before': prev.distribution, 'after': curr.distribution},
            })
            continue

        prev_cont = prev.metadata.get('sound_contingency', '')
        curr_cont = curr.metadata.get('sound_contingency', '')
        if prev_cont and curr_cont and prev_cont != curr_cont:
            manipulations.append({
                'type': 'rule_flip',
                'session_idx': i,
                'global_session_idx': curr.session_idx,
                'details': {'before': prev_cont, 'after': curr_cont},
            })
            continue

        prev_range = (prev.metadata.get('stim_range_min', -1), prev.metadata.get('stim_range_max', 1))
        curr_range = (curr.metadata.get('stim_range_min', -1), curr.metadata.get('stim_range_max', 1))
        if prev_range != curr_range:
            manipulations.append({
                'type': 'range_change',
                'session_idx': i,
                'global_session_idx': curr.session_idx,
                'details': {'before_range': prev_range, 'after_range': curr_range},
            })

    return manipulations


def detect_first_manipulation(
    animal: 'AnimalData',
    stage: Optional[str] = 'Full_Task_Cont',
) -> Optional[Dict]:
    """Convenience: return only the first manipulation, or None."""
    manips = detect_all_manipulations(animal, stage)
    return manips[0] if manips else None


# =============================================================================
# TRAJECTORY COMPUTATION
# =============================================================================

def _compute_session_stats(
    sessions: List['SessionData'],
    stats: List[str],
) -> pd.DataFrame:
    """Compute stats for a list of sessions, return DataFrame."""
    rows = []
    for sess in sessions:
        row = {
            'session_id': sess.session_id,
            'session_idx': sess.session_idx,
            'date': sess.date,
            'n_trials': sess.trials.valid_mask.sum(),
        }
        computed = sess.stats(stats)
        for s in stats:
            row[s] = computed.get(s, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def adaptation_trajectory(
    baseline: List['SessionData'],
    post: List['SessionData'],
    stats: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Session-by-session stat trajectory for baseline and post phases.

    Args:
        baseline: Pre-manipulation sessions
        post: Post-manipulation sessions
        stats: Stats to compute (default: accuracy, side_bias, recency, pse, slope)

    Returns:
        DataFrame with relative_session, phase, and requested stats.
    """
    if stats is None:
        stats = ['accuracy', 'side_bias', 'recency', 'pse', 'slope']

    bl_df = _compute_session_stats(baseline, stats)
    bl_df['phase'] = 'baseline'
    bl_df['relative_session'] = np.arange(-len(bl_df), 0)

    post_df = _compute_session_stats(post, stats)
    post_df['phase'] = 'post'
    post_df['relative_session'] = np.arange(len(post_df))

    bl_summary = {}
    for stat in stats:
        if stat in bl_df.columns:
            vals = bl_df[stat].dropna()
            bl_summary[f'baseline_{stat}_mean'] = float(vals.mean()) if len(vals) > 0 else np.nan
            bl_summary[f'baseline_{stat}_std'] = float(vals.std()) if len(vals) > 0 else np.nan

    combined = pd.concat([bl_df, post_df], ignore_index=True)
    for k, v in bl_summary.items():
        combined[k] = v

    return combined


# =============================================================================
# RECOVERY CURVE FITTING
# =============================================================================

def _exponential_decay(t, y_init, y_asymp, tau):
    return y_asymp + (y_init - y_asymp) * np.exp(-t / tau)


def fit_recovery_curve(
    baseline: List['SessionData'],
    post: List['SessionData'],
    stat: str = 'accuracy',
    model: str = 'exponential',
    min_post_sessions: int = 3,
) -> Dict[str, Any]:
    """
    Fit a recovery curve to how stat evolves across post sessions.

    Args:
        baseline: Pre-manipulation sessions
        post: Post-manipulation sessions
        stat: Which stat to track (accepts 'pse', 'slope', etc.)
        model: 'exponential' or 'linear'
        min_post_sessions: Minimum post sessions for fitting

    Returns:
        Dict with stat, model, converged, baseline_mean/std, y_initial,
        y_asymptote, tau, r_squared, fitted_values, raw_values,
        relative_sessions, sessions_to_criterion.
    """
    nan_result = {
        'stat': stat, 'model': model, 'converged': False,
        'baseline_mean': np.nan, 'baseline_std': np.nan,
        'y_initial': np.nan, 'y_asymptote': np.nan,
        'tau': np.nan, 'r_squared': np.nan,
        'fitted_values': np.array([]),
        'raw_values': np.array([]),
        'relative_sessions': np.array([]),
        'sessions_to_criterion': np.nan,
    }

    bl_vals = []
    for s in baseline:
        v = s.stats([stat]).get(stat, np.nan)
        if not np.isnan(v):
            bl_vals.append(v)

    if len(bl_vals) < 2:
        return nan_result

    bl_mean = float(np.mean(bl_vals))
    bl_std = float(np.std(bl_vals))

    post_vals = []
    for s in post:
        post_vals.append(s.stats([stat]).get(stat, np.nan))

    post_vals = np.array(post_vals)
    valid = ~np.isnan(post_vals)

    if valid.sum() < min_post_sessions:
        result = dict(nan_result)
        result['baseline_mean'] = bl_mean
        result['baseline_std'] = bl_std
        return result

    t = np.arange(len(post_vals), dtype=float)
    t_valid = t[valid]
    y_valid = post_vals[valid]

    result = {
        'stat': stat, 'model': model,
        'baseline_mean': bl_mean, 'baseline_std': bl_std,
        'y_initial': float(y_valid[0]),
        'raw_values': post_vals, 'relative_sessions': t,
    }

    if model == 'exponential':
        try:
            popt, _ = curve_fit(
                _exponential_decay, t_valid, y_valid,
                p0=[y_valid[0], bl_mean, max(1.0, len(y_valid) / 3.0)],
                bounds=([-np.inf, -np.inf, 0.1], [np.inf, np.inf, 100.0]),
                maxfev=5000,
            )
            y_pred = _exponential_decay(t, *popt)
            ss_res = np.sum((y_valid - _exponential_decay(t_valid, *popt)) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
            result.update({
                'converged': True, 'y_asymptote': float(popt[1]),
                'tau': float(popt[2]), 'r_squared': float(r2),
                'fitted_values': y_pred,
            })
        except (RuntimeError, ValueError, TypeError):
            result.update({
                'converged': False, 'y_asymptote': np.nan,
                'tau': np.nan, 'r_squared': np.nan,
                'fitted_values': np.array([]),
            })
    elif model == 'linear':
        coeffs = np.polyfit(t_valid, y_valid, 1)
        y_pred = np.polyval(coeffs, t)
        ss_res = np.sum((y_valid - np.polyval(coeffs, t_valid)) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        result.update({
            'converged': True, 'y_asymptote': float(y_pred[-1]),
            'tau': np.nan, 'r_squared': float(r2),
            'fitted_values': y_pred,
        })
    else:
        raise ValueError(f"Unknown model: {model}")

    if bl_std > 1e-10:
        for i, v in enumerate(post_vals):
            if not np.isnan(v) and abs(v - bl_mean) <= bl_std:
                result['sessions_to_criterion'] = i
                break
        else:
            result['sessions_to_criterion'] = np.nan
    else:
        result['sessions_to_criterion'] = 0

    return result


# =============================================================================
# PHASE COMPARISON
# =============================================================================

def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return float((more - less) / (n_x * n_y))


def compare_phases(
    phase_a: List['SessionData'],
    phase_b: List['SessionData'],
    stats: Optional[List[str]] = None,
    test: str = 'wilcoxon',
) -> pd.DataFrame:
    """
    Statistical comparison of aggregate stats between two phases.

    Args:
        phase_a: First set of sessions (e.g. baseline)
        phase_b: Second set of sessions (e.g. post-shift)
        stats: Which stats to compare (accepts 'pse', 'slope', etc.)
        test: 'wilcoxon' or 'mannwhitneyu'

    Returns:
        DataFrame with stat, a_mean, a_std, a_n, b_mean, b_std, b_n,
        test_stat, p_value, cliffs_delta, direction
    """
    if stats is None:
        stats = ['accuracy', 'side_bias', 'recency', 'pse', 'slope']

    df_a = _compute_session_stats(phase_a, stats)
    df_b = _compute_session_stats(phase_b, stats)

    rows = []
    for stat in stats:
        if stat not in df_a.columns or stat not in df_b.columns:
            continue

        vals_a = df_a[stat].dropna().values
        vals_b = df_b[stat].dropna().values

        row = {
            'stat': stat,
            'a_mean': float(np.mean(vals_a)) if len(vals_a) > 0 else np.nan,
            'a_std': float(np.std(vals_a)) if len(vals_a) > 0 else np.nan,
            'a_n': len(vals_a),
            'b_mean': float(np.mean(vals_b)) if len(vals_b) > 0 else np.nan,
            'b_std': float(np.std(vals_b)) if len(vals_b) > 0 else np.nan,
            'b_n': len(vals_b),
        }

        if test == 'wilcoxon':
            n_pairs = min(len(vals_a), len(vals_b))
            if n_pairs >= 5:
                try:
                    s, p = wilcoxon(vals_a[-n_pairs:], vals_b[:n_pairs])
                    row['test_stat'] = float(s)
                    row['p_value'] = float(p)
                except ValueError:
                    row['test_stat'] = np.nan
                    row['p_value'] = np.nan
            else:
                row['test_stat'] = np.nan
                row['p_value'] = np.nan
        elif test == 'mannwhitneyu':
            if len(vals_a) >= 3 and len(vals_b) >= 3:
                try:
                    s, p = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
                    row['test_stat'] = float(s)
                    row['p_value'] = float(p)
                except ValueError:
                    row['test_stat'] = np.nan
                    row['p_value'] = np.nan
            else:
                row['test_stat'] = np.nan
                row['p_value'] = np.nan
        else:
            raise ValueError(f"Unknown test: {test}")

        row['cliffs_delta'] = _cliffs_delta(vals_a, vals_b)

        if len(vals_a) > 0 and len(vals_b) > 0:
            row['direction'] = 'increase' if np.mean(vals_b) > np.mean(vals_a) else 'decrease'
        else:
            row['direction'] = 'unknown'

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# SHIFT MAGNITUDE
# =============================================================================

def compute_shift_magnitude(
    baseline: List['SessionData'],
    post: List['SessionData'],
    metric: str = 'accuracy_drop',
    n_post_sessions: int = 3,
) -> Dict[str, float]:
    """
    Quantify how much behaviour changed at the transition.

    Args:
        baseline: Pre-manipulation sessions
        post: Post-manipulation sessions
        metric: 'accuracy_drop', 'pse_shift', or any stat name
        n_post_sessions: How many early post sessions to average

    Returns:
        {'metric': name, 'value': float, 'baseline_mean': float, 'post_mean': float}
    """
    stat = {
        'accuracy_drop': 'accuracy',
        'pse_shift': 'pse',
    }.get(metric, metric)

    bl_vals = []
    for s in baseline:
        v = s.stats([stat]).get(stat, np.nan)
        if not np.isnan(v):
            bl_vals.append(v)

    early_post = post[:n_post_sessions]
    post_vals = []
    for s in early_post:
        v = s.stats([stat]).get(stat, np.nan)
        if not np.isnan(v):
            post_vals.append(v)

    bl_mean = float(np.mean(bl_vals)) if bl_vals else np.nan
    post_mean = float(np.mean(post_vals)) if post_vals else np.nan

    if metric == 'accuracy_drop':
        value = bl_mean - post_mean
    elif metric == 'pse_shift':
        value = post_mean - bl_mean
    else:
        value = abs(post_mean - bl_mean)

    return {
        'metric': metric, 'value': value,
        'baseline_mean': bl_mean, 'post_mean': post_mean,
    }


# =============================================================================
# MULTI-FEATURE CONVERGENCE
# =============================================================================

def compute_convergence_metrics(
    baseline: List['SessionData'],
    post: List['SessionData'],
    stats: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute convergence metrics for multiple stats. Returns DataFrame."""
    if stats is None:
        stats = ['accuracy', 'side_bias', 'recency', 'pse', 'slope']

    rows = []
    for stat in stats:
        recovery = fit_recovery_curve(baseline, post, stat=stat)
        rows.append({
            'stat': stat,
            'converged': recovery['converged'],
            'tau': recovery['tau'],
            'y_initial': recovery['y_initial'],
            'y_asymptote': recovery['y_asymptote'],
            'r_squared': recovery['r_squared'],
            'baseline_mean': recovery['baseline_mean'],
            'baseline_std': recovery['baseline_std'],
            'sessions_to_criterion': recovery['sessions_to_criterion'],
        })

    return pd.DataFrame(rows)


# =============================================================================
# SIMPLE STATE CLASSIFICATION
# =============================================================================

def classify_sessions(
    baseline: List['SessionData'],
    sessions_to_classify: List['SessionData'],
    stats: Optional[List[str]] = None,
    n_sd_threshold: float = 1.5,
    required_deviations: int = 2,
) -> List[str]:
    """
    Threshold-based state classifier: 'updating' if enough stats deviate
    from baseline, otherwise 'inference'.

    Args:
        baseline: Reference sessions for computing thresholds
        sessions_to_classify: Sessions to classify
        stats: Which stats to use (accepts 'pse', 'slope', etc.)
        n_sd_threshold: Number of baseline SDs for deviation
        required_deviations: Minimum deviating stats to classify as 'updating'

    Returns:
        List of 'updating' | 'inference' | 'unknown' per session
    """
    if stats is None:
        stats = ['accuracy', 'recency', 'pse', 'slope']

    bl_stats = {}
    for stat in stats:
        vals = []
        for s in baseline:
            v = s.stats([stat]).get(stat, np.nan)
            if not np.isnan(v):
                vals.append(v)
        if len(vals) >= 2:
            bl_stats[stat] = {'mean': np.mean(vals), 'std': np.std(vals)}

    labels = []
    for s in sessions_to_classify:
        computed = s.stats(stats)
        n_deviant = 0
        n_checked = 0

        for stat, bl in bl_stats.items():
            val = computed.get(stat, np.nan)
            if np.isnan(val) or bl['std'] < 1e-10:
                continue
            n_checked += 1
            if abs(val - bl['mean']) > n_sd_threshold * bl['std']:
                n_deviant += 1

        if n_checked == 0:
            labels.append('unknown')
        elif n_deviant >= required_deviations:
            labels.append('updating')
        else:
            labels.append('inference')

    return labels


# =============================================================================
# GROUP-LEVEL HELPERS
# =============================================================================

def aggregate_trajectories(
    trajectories: List[pd.DataFrame],
    stats: Optional[List[str]] = None,
    session_range: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """Aggregate shift-aligned trajectories across animals."""
    if not trajectories:
        return pd.DataFrame()

    pooled = pd.concat(trajectories, ignore_index=True)

    if session_range is not None:
        pooled = pooled[
            (pooled['relative_session'] >= session_range[0]) &
            (pooled['relative_session'] <= session_range[1])
        ]

    if stats is None:
        metadata_cols = {
            'session_id', 'session_idx', 'date', 'n_trials',
            'phase', 'relative_session', 'animal_id',
        }
        baseline_cols = {c for c in pooled.columns if c.startswith('baseline_')}
        stats = [
            c for c in pooled.columns
            if c not in metadata_cols and c not in baseline_cols
            and pooled[c].dtype in [np.float64, np.float32, np.int64]
        ]

    rows = []
    for rel_sess in sorted(pooled['relative_session'].unique()):
        sess_data = pooled[pooled['relative_session'] == rel_sess]
        for stat in stats:
            if stat not in sess_data.columns:
                continue
            vals = sess_data[stat].dropna().values
            if len(vals) == 0:
                continue
            rows.append({
                'relative_session': int(rel_sess),
                'stat': stat,
                'mean': float(np.mean(vals)),
                'sem': float(np.std(vals) / np.sqrt(len(vals))),
                'median': float(np.median(vals)),
                'std': float(np.std(vals)),
                'n_animals': len(vals),
            })

    return pd.DataFrame(rows)
