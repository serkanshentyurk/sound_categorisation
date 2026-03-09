"""
Adaptation Analysis Module (Project-Specific)

Characterises how animals adapt when task demands change.
Uses behav_utils for data structures and feature computation.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, List, Dict, Tuple, Callable, Union, TYPE_CHECKING
from scipy.optimize import curve_fit
from scipy.stats import wilcoxon

if TYPE_CHECKING:
    from behav_utils.data.structures import AnimalData, SessionData


# =============================================================================
# MANIPULATION DETECTION
# =============================================================================

def detect_manipulation_session(
    animal: 'AnimalData',
    stage: Optional[str] = 'Full_Task_Cont',
) -> Optional[Dict]:
    """Auto-detect manipulation type and session from metadata changes."""
    sessions = animal.get_sessions(stage=stage) if stage else animal.sessions

    if len(sessions) < 2:
        return None

    for i in range(1, len(sessions)):
        prev = sessions[i - 1]
        curr = sessions[i]

        # Distribution shift
        if prev.distribution != curr.distribution:
            return {
                'type': 'distribution_shift',
                'session_idx': i,
                'global_session_idx': curr.session_idx,
                'details': {'before': prev.distribution, 'after': curr.distribution},
            }

        # Rule flip
        prev_cont = prev.metadata.get('sound_contingency', '')
        curr_cont = curr.metadata.get('sound_contingency', '')
        if prev_cont and curr_cont and prev_cont != curr_cont:
            return {
                'type': 'rule_flip',
                'session_idx': i,
                'global_session_idx': curr.session_idx,
                'details': {'before': prev_cont, 'after': curr_cont},
            }

        # Range change
        prev_range = (prev.metadata.get('stim_range_min', -1), prev.metadata.get('stim_range_max', 1))
        curr_range = (curr.metadata.get('stim_range_min', -1), curr.metadata.get('stim_range_max', 1))
        if prev_range != curr_range:
            return {
                'type': 'range_change',
                'session_idx': i,
                'global_session_idx': curr.session_idx,
                'details': {'before_range': prev_range, 'after_range': curr_range},
            }

    return None


def detect_all_manipulations(
    animal: 'AnimalData',
    stage: Optional[str] = 'Full_Task_Cont',
) -> List[Dict]:
    """
    Detect ALL manipulations in an animal's timeline.

    An animal could have multiple (e.g., distribution shift then reversal).
    Returns list of dicts, same format as detect_manipulation_session.
    """
    sessions = animal.get_sessions(stage=stage) if stage else animal.sessions
    manipulations = []

    if len(sessions) < 2:
        return manipulations

    for i in range(1, len(sessions)):
        prev = sessions[i - 1]
        curr = sessions[i]

        manip = None

        # Distribution shift
        if prev.distribution != curr.distribution:
            manip = {
                'type': 'distribution_shift',
                'session_idx': i,
                'global_session_idx': curr.session_idx,
                'details': {'before': prev.distribution, 'after': curr.distribution},
            }

        # Rule flip
        prev_cont = prev.metadata.get('sound_contingency', '')
        curr_cont = curr.metadata.get('sound_contingency', '')
        if not manip and prev_cont and curr_cont and prev_cont != curr_cont:
            manip = {
                'type': 'rule_flip',
                'session_idx': i,
                'global_session_idx': curr.session_idx,
                'details': {'before': prev_cont, 'after': curr_cont},
            }

        # Range change
        prev_range = (prev.metadata.get('stim_range_min', -1), prev.metadata.get('stim_range_max', 1))
        curr_range = (curr.metadata.get('stim_range_min', -1), curr.metadata.get('stim_range_max', 1))
        if not manip and prev_range != curr_range:
            manip = {
                'type': 'range_change',
                'session_idx': i,
                'global_session_idx': curr.session_idx,
                'details': {'before_range': prev_range, 'after_range': curr_range},
            }

        if manip is not None:
            manipulations.append(manip)

    return manipulations

# =============================================================================
# SHIFT-ALIGNED ANALYSIS
# =============================================================================

def align_to_manipulation(
    feature_df: pd.DataFrame,
    shift_session_idx: int,
) -> pd.DataFrame:
    """Add relative_session column centred on manipulation."""
    df = feature_df.copy()
    if 'session_idx' not in df.columns:
        raise ValueError("feature_df must have 'session_idx' column.")
    df['relative_session'] = df['session_idx'] - shift_session_idx
    df['phase'] = 'pre'
    df.loc[df['relative_session'] >= 0, 'phase'] = 'post'
    return df


def align_animal(
    animal: 'AnimalData',
    shift_session_idx: int,
    stage: Optional[str] = 'Full_Task_Cont',
    **kwargs,
) -> pd.DataFrame:
    """Build feature matrix and align in one call."""
    df = animal.feature_matrix(stage=stage, **kwargs)
    if len(df) == 0:
        warnings.warn(f"Empty feature matrix for {animal.animal_id}")
        return df
    return align_to_manipulation(df, shift_session_idx)


# =============================================================================
# CONVERGENCE METRICS
# =============================================================================

def _exponential_decay(t, y_init, y_asymp, tau):
    return y_asymp + (y_init - y_asymp) * np.exp(-t / tau)


def fit_exponential_convergence(
    relative_sessions: np.ndarray,
    values: np.ndarray,
    min_points: int = 3,
) -> Dict[str, float]:
    """Fit exponential convergence to post-manipulation trajectory."""
    nan_result = {'tau': np.nan, 'y_initial': np.nan, 'y_asymptote': np.nan,
                  'r_squared': np.nan, 'converged': False}

    mask = (relative_sessions >= 0) & ~np.isnan(values)
    t = relative_sessions[mask].astype(float)
    y = values[mask]

    if len(t) < min_points:
        return nan_result

    y_init_guess = y[0]
    y_asymp_guess = np.median(y[-max(1, len(y) // 3):])
    tau_guess = max(1.0, len(t) / 3.0)

    try:
        popt, _ = curve_fit(
            _exponential_decay, t, y,
            p0=[y_init_guess, y_asymp_guess, tau_guess],
            bounds=([-np.inf, -np.inf, 0.1], [np.inf, np.inf, 100.0]),
            maxfev=5000,
        )
        y_pred = _exponential_decay(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        return {'tau': float(popt[2]), 'y_initial': float(popt[0]),
                'y_asymptote': float(popt[1]), 'r_squared': float(r2), 'converged': True}
    except (RuntimeError, ValueError, TypeError):
        return nan_result


def trials_to_criterion(
    aligned_df: pd.DataFrame, feature: str,
    baseline_window: int = 5, criterion_n_sd: float = 1.0,
) -> Dict[str, float]:
    """Sessions until feature recovers to within criterion_n_sd of baseline."""
    nan_result = {'sessions_to_criterion': np.nan, 'baseline_mean': np.nan,
                  'baseline_std': np.nan, 'criterion_value': np.nan}

    if feature not in aligned_df.columns:
        return nan_result

    pre = aligned_df[aligned_df['relative_session'] < 0].sort_values('relative_session')
    post = aligned_df[aligned_df['relative_session'] >= 0].sort_values('relative_session')

    if len(pre) < baseline_window or len(post) < 1:
        return nan_result

    baseline_vals = pre[feature].iloc[-baseline_window:].dropna()
    if len(baseline_vals) < 2:
        return nan_result

    bl_mean = float(baseline_vals.mean())
    bl_std = float(baseline_vals.std())
    if bl_std < 1e-10:
        bl_std = 0.01

    criterion_range = criterion_n_sd * bl_std

    for val, rel in zip(post[feature].values, post['relative_session'].values):
        if not np.isnan(val) and abs(val - bl_mean) <= criterion_range:
            return {'sessions_to_criterion': int(rel), 'baseline_mean': bl_mean,
                    'baseline_std': bl_std, 'criterion_value': criterion_range}

    return {'sessions_to_criterion': np.nan, 'baseline_mean': bl_mean,
            'baseline_std': bl_std, 'criterion_value': criterion_range}


def accuracy_recovery_curve(
    aligned_df: pd.DataFrame, baseline_window: int = 5, feature: str = 'accuracy',
) -> Dict[str, np.ndarray]:
    """Normalised recovery curve: 0 = post-shift level, 1 = fully recovered."""
    empty = {'relative_sessions': np.array([]), 'raw_values': np.array([]),
             'normalised_recovery': np.array([])}

    if feature not in aligned_df.columns:
        return empty

    pre = aligned_df[aligned_df['relative_session'] < 0].sort_values('relative_session')
    post = aligned_df[aligned_df['relative_session'] >= 0].sort_values('relative_session')

    if len(pre) < baseline_window or len(post) < 1:
        return empty

    bl_mean = pre[feature].iloc[-baseline_window:].mean()
    first_post = post[feature].iloc[0]
    denom = bl_mean - first_post

    if abs(denom) < 1e-10:
        normalised = np.ones(len(post))
    else:
        normalised = (post[feature].values - first_post) / denom

    return {'relative_sessions': post['relative_session'].values,
            'raw_values': post[feature].values,
            'normalised_recovery': normalised}


def compute_convergence_metrics(
    aligned_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    baseline_window: int = 5,
) -> pd.DataFrame:
    """Compute convergence metrics for multiple features."""
    if features is None:
        features = ['accuracy', 'pse', 'recency', 'stimulus_recency',
                     'choice_entropy', 'slope', 'side_bias']
        features = [f for f in features if f in aligned_df.columns]

    rows = []
    post = aligned_df[aligned_df['relative_session'] >= 0]

    for feat in features:
        exp_fit = fit_exponential_convergence(
            post['relative_session'].values, post[feat].values)
        ttc = trials_to_criterion(aligned_df, feat, baseline_window=baseline_window)

        row = {'feature': feat}
        row.update({f'exp_{k}': v for k, v in exp_fit.items()})
        row.update({f'ttc_{k}': v for k, v in ttc.items()})
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# PHASE COMPARISONS
# =============================================================================

def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return float((more - less) / (n_x * n_y))


def compare_phases(
    aligned_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    pre_window: int = 5, early_post_window: int = 5, late_post_window: int = 5,
) -> pd.DataFrame:
    """Compare pre, early-post, late-post phases."""
    if features is None:
        features = ['accuracy', 'pse', 'recency', 'stimulus_recency',
                     'choice_entropy', 'slope', 'side_bias']
        features = [f for f in features if f in aligned_df.columns]

    pre = aligned_df[aligned_df['relative_session'] < 0].sort_values('relative_session')
    post = aligned_df[aligned_df['relative_session'] >= 0].sort_values('relative_session')

    pre_vals = pre.iloc[-pre_window:] if len(pre) >= pre_window else pre
    early_post_vals = post.iloc[:early_post_window]
    late_post_vals = post.iloc[-late_post_window:] if len(post) >= late_post_window else post

    rows = []
    for feat in features:
        pre_v = pre_vals[feat].dropna().values
        early_v = early_post_vals[feat].dropna().values
        late_v = late_post_vals[feat].dropna().values

        row = {'feature': feat,
               'pre_mean': float(np.mean(pre_v)) if len(pre_v) > 0 else np.nan,
               'pre_std': float(np.std(pre_v)) if len(pre_v) > 0 else np.nan,
               'early_post_mean': float(np.mean(early_v)) if len(early_v) > 0 else np.nan,
               'late_post_mean': float(np.mean(late_v)) if len(late_v) > 0 else np.nan}

        n_pairs = min(len(pre_v), len(early_v))
        if n_pairs >= 5:
            try:
                stat, p = wilcoxon(pre_v[-n_pairs:], early_v[:n_pairs])
                row['wilcoxon_stat'] = float(stat)
                row['wilcoxon_p'] = float(p)
            except ValueError:
                row['wilcoxon_stat'] = np.nan
                row['wilcoxon_p'] = np.nan
        else:
            row['wilcoxon_stat'] = np.nan
            row['wilcoxon_p'] = np.nan

        row['cliffs_delta'] = float(_cliffs_delta(pre_v, early_v)) if len(pre_v) > 0 and len(early_v) > 0 else np.nan

        if len(pre_v) > 0 and len(early_v) > 0 and len(late_v) > 0:
            drop = np.mean(early_v) - np.mean(pre_v)
            if abs(drop) > 1e-10:
                row['recovery_ratio'] = float(np.clip(1.0 - (np.mean(late_v) - np.mean(pre_v)) / drop, -1, 2))
            else:
                row['recovery_ratio'] = 1.0
        else:
            row['recovery_ratio'] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# SIMPLE STATE CLASSIFIER
# =============================================================================

def compute_baseline_stats(
    aligned_df: pd.DataFrame, features: List[str], baseline_window: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Expert baseline from pre-shift sessions."""
    pre = aligned_df[aligned_df['relative_session'] < 0].sort_values('relative_session')
    pre_baseline = pre.iloc[-baseline_window:] if len(pre) >= baseline_window else pre

    stats = {}
    for feat in features:
        if feat in pre_baseline.columns:
            vals = pre_baseline[feat].dropna()
            stats[feat] = {'mean': float(vals.mean()) if len(vals) > 0 else np.nan,
                           'std': float(vals.std()) if len(vals) > 0 else np.nan}
    return stats


def classify_session_simple(
    session_features: Dict[str, float],
    baseline_stats: Dict[str, Dict[str, float]],
    n_sd_threshold: float = 1.5,
    required_deviations: int = 2,
) -> str:
    """Threshold-based state classifier. Returns 'inference' or 'updating'."""
    n_deviant = 0
    n_checked = 0

    for feat, bl in baseline_stats.items():
        if feat not in session_features:
            continue
        val = session_features[feat]
        if np.isnan(val) or np.isnan(bl['mean']) or bl['std'] < 1e-10:
            continue
        n_checked += 1
        if abs(val - bl['mean']) > n_sd_threshold * bl['std']:
            n_deviant += 1

    if n_checked == 0:
        return 'unknown'
    return 'updating' if n_deviant >= required_deviations else 'inference'


def classify_all_sessions(
    aligned_df: pd.DataFrame, features: List[str],
    baseline_window: int = 5, n_sd_threshold: float = 1.5,
    required_deviations: int = 2,
) -> pd.DataFrame:
    """Classify all sessions in an aligned DataFrame."""
    baseline = compute_baseline_stats(aligned_df, features, baseline_window)
    df = aligned_df.copy()
    states = []
    for _, row in df.iterrows():
        session_feats = {f: row[f] for f in features if f in row.index}
        states.append(classify_session_simple(
            session_feats, baseline, n_sd_threshold, required_deviations))
    df['state'] = states
    return df


# =============================================================================
# GROUP-LEVEL AGGREGATION
# =============================================================================

def aggregate_adaptation_curves(
    aligned_dfs: List[pd.DataFrame],
    features: Optional[List[str]] = None,
    session_range: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """Aggregate shift-aligned trajectories across animals."""
    if not aligned_dfs:
        return pd.DataFrame()

    if features is None:
        shared_cols = set(aligned_dfs[0].columns)
        for df in aligned_dfs[1:]:
            shared_cols &= set(df.columns)
        metadata = {'animal_id', 'session_id', 'session_idx', 'date',
                     'stage', 'distribution', 'relative_session', 'phase', 'state'}
        features = [c for c in shared_cols - metadata
                     if aligned_dfs[0][c].dtype in [np.float64, np.float32, np.int64]]

    pooled = pd.concat(aligned_dfs, ignore_index=True)
    if session_range is not None:
        pooled = pooled[(pooled['relative_session'] >= session_range[0]) &
                        (pooled['relative_session'] <= session_range[1])]

    rows = []
    for rel_sess in sorted(pooled['relative_session'].unique()):
        sess_data = pooled[pooled['relative_session'] == rel_sess]
        for feat in features:
            vals = sess_data[feat].dropna().values
            if len(vals) == 0:
                continue
            rows.append({
                'relative_session': int(rel_sess), 'feature': feat,
                'mean': float(np.mean(vals)),
                'sem': float(np.std(vals) / np.sqrt(len(vals))),
                'median': float(np.median(vals)),
                'std': float(np.std(vals)),
                'n_animals': len(vals),
            })

    return pd.DataFrame(rows)

def adaptation_summary_table(
    animals: List['AnimalData'],
    stage: Optional[str] = 'Full_Task_Cont',
    features: Optional[List[str]] = None,
    baseline_window: int = 5,
) -> pd.DataFrame:
    """
    One-row-per-animal summary of adaptation behaviour.

    For each animal, detects the manipulation, aligns, and computes
    key convergence metrics.

    Args:
        animals: List of AnimalData objects
        stage: Stage filter
        features: Features for convergence analysis
        baseline_window: Pre-shift baseline window

    Returns:
        DataFrame with columns: animal_id, manipulation_type,
        shift_session, n_pre_sessions, n_post_sessions,
        plus convergence metrics ({feature}_tau, {feature}_ttc) for each feature.
    """
    rows = []
    for animal in animals:
        manip = detect_manipulation_session(animal, stage=stage)
        if manip is None:
            continue

        df = animal.feature_matrix(stage=stage)
        if len(df) == 0:
            continue

        aligned = align_to_manipulation(df, manip['session_idx'])

        row = {
            'animal_id': animal.animal_id,
            'manipulation_type': manip['type'],
            'shift_session': manip['session_idx'],
            'n_pre_sessions': int((aligned['relative_session'] < 0).sum()),
            'n_post_sessions': int((aligned['relative_session'] >= 0).sum()),
        }

        # Convergence metrics
        metrics = compute_convergence_metrics(
            aligned,
            features=features,
            baseline_window=baseline_window,
        )
        for _, mrow in metrics.iterrows():
            feat = mrow['feature']
            row[f'{feat}_tau'] = mrow['exp_tau']
            row[f'{feat}_ttc'] = mrow['ttc_sessions_to_criterion']

        rows.append(row)

    return pd.DataFrame(rows)
