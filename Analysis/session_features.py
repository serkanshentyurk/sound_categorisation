"""
Session Feature Matrix Builder

Computes a (session × feature) DataFrame from AnimalData, suitable for
HMM epoch categorisation and general behavioural analysis.

Combines:
    - Core summary statistics from Analysis.summary_stats (choice-based)
    - RT-based features (pluggable extraction)
    - Session-to-session dynamics (deltas)
    - Session metadata (distribution, date, trial counts)

RT is handled via a pluggable extractor function so the RT source can be
changed without touching the rest of the pipeline.

Usage:
    from Analysis.session_features import build_feature_matrix

    # Default RT extractor uses Response_Latency
    df = build_feature_matrix(animal_data)

    # Custom RT extractor
    df = build_feature_matrix(animal_data, rt_extractor=my_rt_func)

    # Multiple animals
    df = build_feature_matrix_multi([animal1, animal2])
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, List, Callable, Dict, Union, TYPE_CHECKING

from Analysis.summary_stats import (
    SUMMARY_REGISTRY,
    FEATURE_MATRIX_STATS,
    compute_summary_stats,
)

if TYPE_CHECKING:
    from Data.structures import AnimalData, SessionData, TrialData


# =============================================================================
# RT EXTRACTION (pluggable)
# =============================================================================

def default_rt_extractor(trials: 'TrialData') -> np.ndarray:
    """
    Default RT extractor: uses Response_Latency field (in ms).

    Returns array of shape (n_trials,) with NaN for aborts/no-response.

    IMPORTANT: The interpretation of Response_Latency is currently uncertain.
    RT=0 may mean anticipatory response (licked before/at go cue) or a
    timing issue. This extractor preserves the raw values. When you have
    confirmed the timing, replace this function or adjust downstream.

    To swap in a different RT source (e.g., computed from Trial_End_Time),
    pass a custom function to build_feature_matrix(rt_extractor=my_func).
    """
    rt = trials.reaction_time.copy().astype(float)
    
    # Mark aborts and no-response as NaN
    rt[trials.abort] = np.nan
    rt[trials.no_response] = np.nan

    return rt


# def trial_end_time_rt_extractor(trials: 'TrialData') -> np.ndarray:
#     """
#     Alternative RT extractor using inter-trial-end intervals.

#     Computes time between consecutive Trial_End_Time values.
#     This captures total trial duration (including ITI, sound, response)
#     rather than true reaction time, but tracks relative speed changes
#     across sessions even if absolute calibration is uncertain.

#     Returns array of shape (n_trials,) with NaN for first trial and
#     abort/no-response trials.
#     """
#     if not hasattr(trials, 'extra') or 'Trial_End_Time' not in trials.extra:
#         warnings.warn(
#             "Trial_End_Time not found in trials.extra. "
#             "Falling back to default_rt_extractor."
#         )
#         return default_rt_extractor(trials)

#     tet = trials.extra['Trial_End_Time'].astype(float)
#     rt = np.full(len(tet), np.nan)
#     rt[1:] = (np.diff(tet) - 2.6) * 1000  # convert seconds to ms

#     # Mark aborts and no-response
#     rt[trials.abort] = np.nan
#     rt[trials.no_response] = np.nan

#     return rt


# =============================================================================
# RT FEATURE COMPUTATION
# =============================================================================

def compute_rt_features(
    rt: np.ndarray,
    stimuli: np.ndarray,
    categories: np.ndarray,
    choices: np.ndarray,
    hard_threshold: float = 0.3,
    fast_threshold: float = 50.0,
) -> Dict[str, float]:
    """
    Compute RT-based features from a single session.

    Args:
        rt: Reaction times in ms, NaN for invalid trials
        stimuli: Stimulus values
        categories: True categories
        choices: Category-space choices (0/1/NaN)
        hard_threshold: |stimulus| threshold for hard/easy split
        fast_threshold: RT threshold (ms) for anticipatory/fast responses

    Returns:
        Dict of RT features. All features are NaN-safe.
    """
    valid = ~np.isnan(rt) & ~np.isnan(choices)
    rt_valid = rt[valid]

    result = {}

    if len(rt_valid) < 10:
        result['rt_median'] = np.nan
        result['rt_iqr'] = np.nan
        result['rt_skewness'] = np.nan
        result['proportion_fast'] = np.nan
        result['rt_median_hard'] = np.nan
        result['rt_median_easy'] = np.nan
        result['rt_correct_vs_error'] = np.nan
        return result

    # Basic RT stats
    result['rt_median'] = float(np.median(rt_valid))
    q25, q75 = np.percentile(rt_valid, [25, 75])
    result['rt_iqr'] = float(q75 - q25)

    # Skewness (using scipy-free formula)
    if np.std(rt_valid) > 0:
        result['rt_skewness'] = float(
            np.mean(((rt_valid - np.mean(rt_valid)) / np.std(rt_valid)) ** 3)
        )
    else:
        result['rt_skewness'] = np.nan

    # Proportion fast/anticipatory
    result['proportion_fast'] = float(np.mean(rt_valid <= fast_threshold))

    # RT by difficulty
    s_valid = stimuli[valid]
    hard = np.abs(s_valid) < hard_threshold
    easy = np.abs(s_valid) >= hard_threshold

    result['rt_median_hard'] = float(np.median(rt_valid[hard])) if hard.sum() >= 3 else np.nan
    result['rt_median_easy'] = float(np.median(rt_valid[easy])) if easy.sum() >= 3 else np.nan

    # RT correct vs error
    c_valid = choices[valid]
    cat_valid = categories[valid]
    correct = c_valid == cat_valid
    error = c_valid != cat_valid

    if correct.sum() >= 3 and error.sum() >= 3:
        result['rt_correct_vs_error'] = float(
            np.median(rt_valid[correct]) - np.median(rt_valid[error])
        )
    else:
        result['rt_correct_vs_error'] = np.nan

    return result


# =============================================================================
# SINGLE SESSION FEATURE EXTRACTION
# =============================================================================

def compute_session_features(
    session: 'SessionData',
    rt_extractor: Callable = default_rt_extractor,
    stat_names: Optional[List[str]] = None,
    exclude_abort: bool = True,
    exclude_opto: bool = True,
    hard_threshold: float = 0.3,
    fast_threshold: float = 50.0,
) -> Dict[str, float]:
    """
    Compute all features for a single session.

    Args:
        session: SessionData object
        rt_extractor: Function(TrialData) -> np.ndarray of RT values
        stat_names: Which summary stats to compute (default: FEATURE_MATRIX_STATS)
        exclude_abort: Remove abort trials before computing stats
        exclude_opto: Remove opto trials before computing stats
        hard_threshold: |stimulus| threshold for hard/easy split
        fast_threshold: RT threshold (ms) for fast response classification

    Returns:
        Dict of feature_name -> value (all scalars)
    """
    if stat_names is None:
        stat_names = FEATURE_MATRIX_STATS

    # --- Metadata ---
    features = {
        'animal_id': session.metadata.animal_id,
        'session_id': session.session_id,
        'session_idx': session.session_idx,
        'date': session.date,
        'stage': session.stage,
        'distribution': session.distribution,
    }

    # --- Get filtered arrays ---
    arrays = session.trials.get_model_arrays(
        exclude_abort=exclude_abort,
        exclude_opto=exclude_opto,
    )

    stimuli = arrays['stimuli']
    categories = arrays['categories']
    choices = arrays['choices']
    no_response = arrays['no_response']

    # Trial counts
    features['n_trials_total'] = session.trials.n_trials
    features['n_trials_valid'] = int((~no_response).sum())
    features['n_trials_abort'] = int(session.trials.abort.sum())
    features['abort_rate'] = float(session.trials.abort.mean())

    if features['n_trials_valid'] < 10:
        warnings.warn(
            f"Session {session.session_id} has only {features['n_trials_valid']} "
            f"valid trials. Features will be NaN."
        )

    # --- Core summary stats (choice-based) ---
    stats_dict = compute_summary_stats(
        choices, stimuli, categories,
        stat_names=stat_names,
        return_dict=True,
    )

    # Flatten nested dicts (e.g., psychometric -> pse, slope, ...)
    for stat_name, value in stats_dict.items():
        if isinstance(value, dict):
            for k, v in value.items():
                features[k] = float(v) if not isinstance(v, (str, type(None))) else v
        elif isinstance(value, np.ndarray):
            # Binned stats: store each bin as separate feature
            for i, v in enumerate(value):
                features[f'{stat_name}_{i}'] = float(v)
        else:
            features[stat_name] = float(value) if not isinstance(value, (str, type(None))) else value

    # --- RT features ---
    rt_raw = rt_extractor(session.trials)

    # Apply same filtering as for choices (exclude abort/opto)
    trial_indices = arrays['trial_indices']
    rt_filtered = rt_raw[trial_indices]

    rt_features = compute_rt_features(
        rt_filtered, stimuli, categories, choices,
        hard_threshold=hard_threshold,
        fast_threshold=fast_threshold,
    )
    features.update(rt_features)

    return features


# =============================================================================
# FEATURE MATRIX BUILDER
# =============================================================================

def build_feature_matrix(
    animal: 'AnimalData',
    rt_extractor: Callable = default_rt_extractor,
    stat_names: Optional[List[str]] = None,
    stage: Optional[str] = 'Full_Task_Cont',
    exclude_abort: bool = True,
    exclude_opto: bool = True,
    min_valid_trials: int = 10,
    hard_threshold: float = 0.3,
    fast_threshold: float = 50.0,
    compute_deltas: bool = True,
) -> pd.DataFrame:
    """
    Build session × feature DataFrame for a single animal.

    Args:
        animal: AnimalData object
        rt_extractor: Function(TrialData) -> np.ndarray
        stat_names: Which summary stats to compute
        stage: Filter to this stage (None = all stages)
        exclude_abort: Remove abort trials
        exclude_opto: Remove opto trials
        min_valid_trials: Skip sessions with fewer valid trials
        hard_threshold: |stimulus| threshold for hard/easy split
        fast_threshold: RT threshold (ms) for fast responses
        compute_deltas: Whether to add session-to-session delta features

    Returns:
        DataFrame with one row per session, columns = features.
        Metadata columns: animal_id, session_id, session_idx, date, stage, distribution
        Feature columns: all computed statistics (floats)
    """
    sessions = animal.get_sessions(stage=stage) if stage else animal.sessions

    rows = []
    for session in sessions:
        features = compute_session_features(
            session,
            rt_extractor=rt_extractor,
            stat_names=stat_names,
            exclude_abort=exclude_abort,
            exclude_opto=exclude_opto,
            hard_threshold=hard_threshold,
            fast_threshold=fast_threshold,
        )

        if features['n_trials_valid'] < min_valid_trials:
            warnings.warn(
                f"Skipping session {session.session_id}: "
                f"only {features['n_trials_valid']} valid trials."
            )
            continue

        rows.append(features)

    if not rows:
        warnings.warn(f"No valid sessions for animal {animal.animal_id}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # --- Session-to-session deltas ---
    if compute_deltas and len(df) > 1:
        df = _add_delta_features(df)

    return df


def build_feature_matrix_multi(
    animals: List['AnimalData'],
    rt_extractor: Callable = default_rt_extractor,
    stat_names: Optional[List[str]] = None,
    stage: Optional[str] = 'Full_Task_Cont',
    exclude_abort: bool = True,
    exclude_opto: bool = True,
    min_valid_trials: int = 10,
    hard_threshold: float = 0.3,
    fast_threshold: float = 50.0,
    compute_deltas: bool = True,
) -> pd.DataFrame:
    """
    Build session × feature DataFrame for multiple animals (pooled).

    Deltas are computed within each animal, not across animals.
    Returns a single DataFrame with animal_id column for grouping.
    """
    dfs = []
    for animal in animals:
        df = build_feature_matrix(
            animal,
            rt_extractor=rt_extractor,
            stat_names=stat_names,
            stage=stage,
            exclude_abort=exclude_abort,
            exclude_opto=exclude_opto,
            min_valid_trials=min_valid_trials,
            hard_threshold=hard_threshold,
            fast_threshold=fast_threshold,
            compute_deltas=compute_deltas,
        )
        if len(df) > 0:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# DELTA FEATURES
# =============================================================================

# Features for which session-to-session deltas are informative
DELTA_FEATURES = [
    'pse', 'slope', 'accuracy', 'side_bias', 'recency',
    'choice_entropy', 'rt_median',
]


def _add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add session-to-session delta columns for selected features.

    For each feature in DELTA_FEATURES, adds:
        delta_{feature}: absolute change from previous session
    First session gets NaN for all deltas.
    """
    df = df.copy()

    for feat in DELTA_FEATURES:
        if feat in df.columns:
            delta = df[feat].diff().abs()
            df[f'delta_{feat}'] = delta

    return df


# =============================================================================
# FEATURE DESCRIPTIONS AND UTILITIES
# =============================================================================

# Metadata columns (not features, used for indexing/filtering)
METADATA_COLUMNS = [
    'animal_id', 'session_id', 'session_idx', 'date', 'stage', 'distribution',
    'n_trials_total', 'n_trials_valid', 'n_trials_abort', 'abort_rate',
]


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric feature columns (excluding metadata and deltas)."""
    exclude = set(METADATA_COLUMNS)
    return [
        col for col in df.columns
        if col not in exclude
        and not col.startswith('delta_')
        and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]


def get_delta_columns(df: pd.DataFrame) -> List[str]:
    """Get list of delta feature columns."""
    return [col for col in df.columns if col.startswith('delta_')]


def get_numeric_features(df: pd.DataFrame, include_deltas: bool = False) -> pd.DataFrame:
    """
    Extract numeric feature columns only (for HMM input, PCA, etc.).

    Args:
        df: Full feature matrix from build_feature_matrix
        include_deltas: Whether to include session-to-session delta features

    Returns:
        DataFrame with only numeric feature columns
    """
    cols = get_feature_columns(df)
    if include_deltas:
        cols += get_delta_columns(df)
    return df[cols].copy()


def zscore_features(
    df: pd.DataFrame,
    include_deltas: bool = False,
) -> pd.DataFrame:
    """
    Z-score numeric features (across all sessions in df).

    Useful for HMM fitting where features need comparable scales.
    Metadata columns are preserved unchanged.

    Args:
        df: Feature matrix from build_feature_matrix
        include_deltas: Whether to include delta features

    Returns:
        DataFrame with z-scored numeric features and original metadata
    """
    df_out = df.copy()
    cols = get_feature_columns(df_out)
    if include_deltas:
        cols += get_delta_columns(df_out)

    for col in cols:
        vals = df_out[col].values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() > 1:
            mu = np.nanmean(vals)
            sd = np.nanstd(vals)
            if sd > 1e-10:
                df_out[col] = (vals - mu) / sd
            else:
                df_out[col] = 0.0  # constant feature

    return df_out


def summarise_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics of all numeric features across sessions.

    Returns DataFrame with mean, std, min, max, n_valid per feature.
    Useful for sanity checking before HMM fitting.
    """
    cols = get_feature_columns(df)
    rows = []
    for col in cols:
        vals = df[col].values.astype(float)
        valid = ~np.isnan(vals)
        rows.append({
            'feature': col,
            'n_valid': int(valid.sum()),
            'n_nan': int((~valid).sum()),
            'mean': float(np.nanmean(vals)) if valid.any() else np.nan,
            'std': float(np.nanstd(vals)) if valid.any() else np.nan,
            'min': float(np.nanmin(vals)) if valid.any() else np.nan,
            'max': float(np.nanmax(vals)) if valid.any() else np.nan,
        })
    return pd.DataFrame(rows)


# =============================================================================
# PER-ANIMAL REGRESSION DIAGNOSTICS (Theil-Sen)
# =============================================================================

def _theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Theil-Sen estimator: median of all pairwise slopes.

    Robust to up to ~29% outliers. No parameters to tune.
    """
    n = len(x)
    if n < 2:
        return np.nan

    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))

    if not slopes:
        return np.nan

    return float(np.median(slopes))


def _theil_sen_fit(x: np.ndarray, y: np.ndarray):
    """
    Full Theil-Sen fit returning slope, intercept, and residuals.

    Returns:
        slope, intercept, residuals, r_squared
    """
    slope = _theil_sen_slope(x, y)
    if np.isnan(slope):
        return np.nan, np.nan, np.full(len(y), np.nan), np.nan

    intercept = float(np.median(y - slope * x))
    y_pred = slope * x + intercept
    residuals = y - y_pred

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

    return slope, intercept, residuals, r_squared


def compute_animal_trends(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    min_sessions: int = 5,
    outlier_threshold: float = 2.5,
) -> pd.DataFrame:
    """
    Compute per-animal Theil-Sen regression slope for each feature.

    For each (animal, feature) pair, fits a robust linear trend across sessions
    and reports slope, R², and number of outlier sessions.

    Args:
        df: Feature matrix from build_feature_matrix (or _multi)
        features: List of feature columns to analyse. If None, uses all numeric.
        min_sessions: Minimum sessions required for regression
        outlier_threshold: Residuals beyond this many MADs flagged as outliers

    Returns:
        DataFrame with columns:
            animal_id, feature, n_sessions, n_valid,
            slope, intercept, r_squared, n_outliers,
            mean_value, std_value
    """
    if features is None:
        features = get_feature_columns(df)

    rows = []
    for aid in sorted(df['animal_id'].unique()):
        adf = df[df['animal_id'] == aid].sort_values('session_idx')

        if len(adf) < min_sessions:
            continue

        x = adf['session_idx'].values.astype(float)

        for feat in features:
            if feat not in adf.columns:
                continue

            y = adf[feat].values.astype(float)
            valid = ~np.isnan(y)

            if valid.sum() < min_sessions:
                # Still record the animal-feature pair but with NaN trend
                rows.append({
                    'animal_id': aid,
                    'feature': feat,
                    'n_sessions': len(adf),
                    'n_valid': int(valid.sum()),
                    'slope': np.nan,
                    'intercept': np.nan,
                    'r_squared': np.nan,
                    'n_outliers': 0,
                    'mean_value': float(np.nanmean(y)),
                    'std_value': float(np.nanstd(y)),
                })
                continue

            xv, yv = x[valid], y[valid]
            slope, intercept, residuals, r2 = _theil_sen_fit(xv, yv)

            # Outlier detection via MAD (median absolute deviation)
            mad = np.median(np.abs(residuals - np.median(residuals)))
            if mad > 1e-10:
                # Modified z-score using MAD (0.6745 converts MAD to σ-equivalent)
                mod_z = 0.6745 * np.abs(residuals) / mad
                n_outliers = int(np.sum(mod_z > outlier_threshold))
            else:
                n_outliers = 0

            rows.append({
                'animal_id': aid,
                'feature': feat,
                'n_sessions': len(adf),
                'n_valid': int(valid.sum()),
                'slope': slope,
                'intercept': intercept,
                'r_squared': r2,
                'n_outliers': n_outliers,
                'mean_value': float(np.nanmean(yv)),
                'std_value': float(np.nanstd(yv)),
            })

    return pd.DataFrame(rows)


def summarise_trends(
    trend_df: pd.DataFrame,
    min_animals: int = 3,
) -> pd.DataFrame:
    """
    Summarise per-animal trends across animals for each feature.

    Reports whether slopes are consistent in sign across animals,
    the median slope, and median R².

    Args:
        trend_df: Output from compute_animal_trends
        min_animals: Minimum animals with valid slopes for a feature

    Returns:
        DataFrame sorted by sign consistency (most consistent first), with:
            feature, n_animals, median_slope, iqr_slope,
            sign_consistency (fraction with same sign as median),
            median_r2, total_outliers
    """
    rows = []
    for feat in trend_df['feature'].unique():
        fdf = trend_df[trend_df['feature'] == feat].dropna(subset=['slope'])

        if len(fdf) < min_animals:
            continue

        slopes = fdf['slope'].values
        median_slope = float(np.median(slopes))
        q25, q75 = np.percentile(slopes, [25, 75])

        if median_slope != 0:
            sign_consistency = float(np.mean(np.sign(slopes) == np.sign(median_slope)))
        else:
            sign_consistency = 0.0

        rows.append({
            'feature': feat,
            'n_animals': len(fdf),
            'median_slope': median_slope,
            'iqr_slope': float(q75 - q25),
            'sign_consistency': sign_consistency,
            'median_r2': float(fdf['r_squared'].median()),
            'total_outliers': int(fdf['n_outliers'].sum()),
        })

    result = pd.DataFrame(rows)
    result = result.sort_values('sign_consistency', ascending=False)
    return result


def flag_outlier_sessions(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    min_sessions: int = 5,
    outlier_threshold: float = 2.5,
) -> pd.DataFrame:
    """
    Identify sessions that are outliers from the per-animal trend in any feature.

    Returns a DataFrame of flagged sessions with the deviating feature(s)
    and residual magnitude. Useful for spotting disengagement, equipment
    problems, or data quality issues.

    Args:
        df: Feature matrix from build_feature_matrix
        features: Features to check (default: all numeric)
        min_sessions: Minimum sessions for regression
        outlier_threshold: MAD-based threshold for flagging

    Returns:
        DataFrame with columns:
            animal_id, session_idx, feature, residual, mad_score
    """
    if features is None:
        features = get_feature_columns(df)

    flags = []
    for aid in sorted(df['animal_id'].unique()):
        adf = df[df['animal_id'] == aid].sort_values('session_idx')

        if len(adf) < min_sessions:
            continue

        x = adf['session_idx'].values.astype(float)
        sess_indices = adf['session_idx'].values

        for feat in features:
            if feat not in adf.columns:
                continue

            y = adf[feat].values.astype(float)
            valid = ~np.isnan(y)

            if valid.sum() < min_sessions:
                continue

            xv, yv = x[valid], y[valid]
            slope, intercept, _, _ = _theil_sen_fit(xv, yv)

            if np.isnan(slope):
                continue

            # Compute residuals for ALL sessions (including NaN → skip)
            y_pred = slope * x + intercept
            residuals = y - y_pred

            mad = np.median(np.abs(residuals[valid] - np.median(residuals[valid])))
            if mad < 1e-10:
                continue

            mod_z = 0.6745 * np.abs(residuals) / mad

            for t in range(len(adf)):
                if np.isnan(y[t]):
                    continue
                if mod_z[t] > outlier_threshold:
                    flags.append({
                        'animal_id': aid,
                        'session_idx': sess_indices[t],
                        'feature': feat,
                        'value': float(y[t]),
                        'expected': float(y_pred[t]),
                        'residual': float(residuals[t]),
                        'mad_score': float(mod_z[t]),
                    })

    if not flags:
        return pd.DataFrame(columns=[
            'animal_id', 'session_idx', 'feature',
            'value', 'expected', 'residual', 'mad_score'
        ])

    result = pd.DataFrame(flags)
    result = result.sort_values('mad_score', ascending=False)
    return result
