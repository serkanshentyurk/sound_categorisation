"""
Core Data Structures

Hierarchical containers for behavioural data:

    ExperimentData          All animals in one project
    └── AnimalData          One animal, chronologically ordered sessions
        └── SessionData     One session
            ├── SessionMetadata   Constant within session (stage, contingency, ...)
            └── TrialData         Per-trial arrays (stimulus, choice, correct, ...)

    FittingData             Flat per-session arrays for SBI inference

Convention — three levels per domain:
    Low-level:     fit_psychometric(stim, ch)       — raw arrays
    Session-level: compute_psychometric(sessions)   — pre-filtered sessions → result dict
    Plotting:      plot_psychometric(result)         — result dict → axes

Plot methods on data classes are thin wrappers that call compute_ then plot_.

Usage:
    from behav_utils import (
        load_experiment, select_sessions, filter_trials,
        compute_psychometric, compute_um, plot_psychometric, plot_um, PALETTE,
    )

    experiment = load_experiment('config.yaml')
    animal = experiment.get_animal('SS05')
    sessions = select_sessions(animal, preset='expert_uniform')
    clean = filter_trials(sessions)

    psych = compute_psychometric(clean, mode='pooled', n_bootstrap=200)
    fig, ax = plt.subplots()
    plot_psychometric(psych, ax=ax, color=PALETTE[0])
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    Optional, Dict, List, Tuple, Union, Any, Callable, TYPE_CHECKING,
)
from datetime import date

if TYPE_CHECKING:
    from behav_utils.config.schema import ProjectConfig


_STAT_PARENT = {
    'pse': 'psychometric',
    'slope': 'psychometric',
    'lapse_low': 'psychometric',
    'lapse_high': 'psychometric',
    'win_stay_rate': 'win_stay',
    'lose_shift_rate': 'lose_shift',
    'w_stimulus': 'logistic_history',
    'w_prev_choice_1': 'logistic_history',
    'w_prev_choice_2': 'logistic_history',
    'w_prev_choice_3': 'logistic_history',
    'psychometric_gof': 'psychometric',
}


def _flatten_stats_dict(stats_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested stat dicts into a single-level dict.

    {'psychometric': {'pse': 0.01, 'slope': 0.3}} → {'pse': 0.01, 'slope': 0.3}
    {'accuracy': 0.85} → {'accuracy': 0.85}
    Arrays are left as-is (e.g. update_matrix).
    """
    flat = {}
    for key, value in stats_dict.items():
        if isinstance(value, dict):
            for k, v in value.items():
                flat[k] = v
        else:
            flat[key] = value
    return flat

# =============================================================================
# SESSION METADATA
# =============================================================================

@dataclass
class SessionMetadata:
    """
    Session-level metadata (constant within a session).
    Populated from the config's session_metadata mappings.

    The 'fields' dict holds all metadata key-value pairs.
    Common fields are exposed as properties for convenience.
    """
    fields: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.fields.get(key, default)

    def __getattr__(self, name: str) -> Any:
        # Allow attribute-style access to fields
        if name == 'fields' or name.startswith('_'):
            raise AttributeError(name)
        if name in self.fields:
            return self.fields[name]
        raise AttributeError(
            f"SessionMetadata has no field '{name}'. "
            f"Available: {list(self.fields.keys())}"
        )

    # Common convenience properties
    @property
    def animal_id(self) -> str:
        return self.fields.get('animal_id', '')

    @property
    def stage(self) -> str:
        return self.fields.get('stage', '')

    @property
    def sound_contingency(self) -> str:
        return self.fields.get('sound_contingency', '')

    @property
    def stim_range_min(self) -> float:
        return self.fields.get('stim_range_min', -1.0)

    @property
    def stim_range_max(self) -> float:
        return self.fields.get('stim_range_max', 1.0)


# =============================================================================
# TRIAL DATA
# =============================================================================


@dataclass
class TrialData:
    """
    Per-trial arrays for a single session.

    Required arrays (always present after loading):
        stimulus     float   Raw stimulus values (normalised to [-1, 1])
        choice       float   Category-space choice (0=A, 1=B, NaN=no response)
        outcome      str     Trial outcome label from CSV
        correct      bool    Whether choice matched category
        category     int     Correct category (0 or 1), derived from stimulus
        trial_number int     1-indexed trial number from CSV

    Optional arrays (present if configured, sensible defaults otherwise):
        reaction_time  float   Response latency (NaN if missing)
        abort          bool    Trial aborted (hardware/animal error)
        opto_on        bool    Optogenetic light delivered on this trial
        distribution   str     Stimulus distribution label
        choice_raw     str     Raw choice before category-space mapping

    Extra arrays (unmapped CSV columns):
        optional_fields  Dict[str, np.ndarray]  — mapped optional columns
        extra            Dict[str, np.ndarray]  — unmapped CSV columns

    Filtering:
        Use build_mask() or opto_mask() to create boolean masks,
        then filter(mask) to create a new TrialData with only those trials.
        Analysis and plotting functions receive pre-filtered data —
        they do NOT filter internally.
    """
    # ── Required ────────────────────────────────────────────────────────────
    trial_number: np.ndarray
    stimulus: np.ndarray
    choice: np.ndarray
    outcome: np.ndarray
    correct: np.ndarray
    category: np.ndarray

    choice_raw: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Optional ────────────────────────────────────────────────────────────
    reaction_time: np.ndarray = field(default_factory=lambda: np.array([]))
    abort: np.ndarray = field(default_factory=lambda: np.array([]))
    opto_on: np.ndarray = field(default_factory=lambda: np.array([]))
    distribution: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── All other columns ───────────────────────────────────────────────────
    optional_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    extra: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        """Set defaults for empty optional arrays."""
        n = len(self.stimulus)
        if len(self.reaction_time) == 0:
            self.reaction_time = np.full(n, np.nan)
        if len(self.abort) == 0:
            self.abort = np.zeros(n, dtype=bool)
        if len(self.opto_on) == 0:
            self.opto_on = np.zeros(n, dtype=bool)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def n_trials(self) -> int:
        """Number of trials."""
        return len(self.stimulus)

    @property
    def no_response(self) -> np.ndarray:
        """Boolean mask: True where choice is NaN (animal didn't respond)."""
        return np.isnan(self.choice.astype(float))

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask: non-abort, responded trials."""
        return ~self.abort & ~self.no_response

    # ── Field access ────────────────────────────────────────────────────────

    def get_field(self, name: str) -> Optional[np.ndarray]:
        """
        Get any field by name.

        Checks core fields, then optional_fields, then extra.
        Returns None if not found.
        """
        if hasattr(self, name) and name not in ('optional_fields', 'extra'):
            val = getattr(self, name)
            if isinstance(val, np.ndarray):
                return val
        if name in self.optional_fields:
            return self.optional_fields[name]
        if name in self.extra:
            return self.extra[name]
        return None

    # ── Filtering (thin wrappers — logic lives in filtering.py) ───────────

    def build_mask(self, **kwargs) -> np.ndarray:
        """Build boolean mask from exclusion flags. See filtering.build_mask."""
        from behav_utils.data.filtering import build_mask
        return build_mask(self, **kwargs)

    def opto_mask(self, delta=0) -> np.ndarray:
        """Boolean mask relative to opto events. See filtering.opto_mask."""
        from behav_utils.data.filtering import opto_mask
        return opto_mask(self, delta=delta)

    def filter(self, mask, clear_flags=True) -> 'TrialData':
        """Return new TrialData with only masked trials. See filtering.filter_trial_data."""
        from behav_utils.data.filtering import filter_trial_data
        return filter_trial_data(self, mask, clear_flags=clear_flags)

    # ── Array extraction ────────────────────────────────────────────────────

    def get_arrays(self) -> Dict[str, np.ndarray]:
        """Extract trial arrays (aborts excluded). See filtering.get_arrays."""
        from behav_utils.data.filtering import get_arrays
        return get_arrays(self)

    def get_inputs(self, config: Optional[Any] = None) -> Dict[str, np.ndarray]:
        '''
        Get all input (controlled variable) arrays.

        Args:
            config: ProjectConfig. If None, returns stimulus only (default).

        Returns:
            Dict of {input_name: array}
        '''
        if config is not None and hasattr(config, 'task'):
            input_names = config.task.inputs
        else:
            input_names = ['stimulus']

        result = {}
        for name in input_names:
            arr = self.get_field(name)
            if arr is not None:
                result[name] = arr
            elif name == 'stimulus':
                result[name] = self.stimulus
        return result

    def get_outputs(self, config: Optional[Any] = None) -> Dict[str, np.ndarray]:
        '''
        Get all output (measured variable) arrays.

        Args:
            config: ProjectConfig. If None, returns choice only (default).

        Returns:
            Dict of {output_name: array}
        '''
        if config is not None and hasattr(config, 'task'):
            output_names = config.task.outputs
        else:
            output_names = ['choice']

        result = {}
        for name in output_names:
            arr = self.get_field(name)
            if arr is not None:
                result[name] = arr
            elif name == 'choice':
                result[name] = self.choice
        return result

    # ── Stats ───────────────────────────────────────────────────────────────

    def stats(
        self,
        stat_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute summary statistics on this TrialData.

        No filtering is done here — filter BEFORE calling stats.
        Aborts are excluded (always invalid).

        Accepts both registered stat names ('psychometric', 'accuracy')
        and sub-field names ('pse', 'slope', 'lapse_low').

        Args:
            stat_names: Which stats to compute (default: all registered).

        Returns:
            Flat dict of stat_name → value.
        """
        from behav_utils.analysis.summary_stats import compute_summary_stats

        arrays = self.get_arrays()
        valid = ~arrays['no_response']

        if valid.sum() < 5:
            warnings.warn(f"Only {valid.sum()} valid trials — stats may be unreliable")

        if stat_names is not None:
            registered = list({_STAT_PARENT.get(s, s) for s in stat_names})
        else:
            registered = stat_names

        raw = compute_summary_stats(
            arrays['choices'], arrays['stimuli'], arrays['categories'],
            stat_names=registered,
            return_dict=True,
        )

        return _flatten_stats_dict(raw)



# =============================================================================
# SESSION DATA
# =============================================================================

@dataclass
class SessionData:
    """
    All data for a single behavioural session.

    Attributes:
        session_id:  Unique identifier (e.g. 'SOUND_CAT_SS14_2026_05_06')
        session_idx: Ordinal index within animal (0-based, chronological)
        date:        Session date
        metadata:    SessionMetadata (stage, contingency, protocol, ...)
        trials:      TrialData (per-trial arrays)
        masking:     Whether this is a masking (light-only control) session
        csv_path:    Path to source CSV file
        filter_info: Metadata about filtering applied (None if unfiltered)

    Filtering:
        session.filter()                              # standard: exclude abort + opto
        session.filter(session.trials.opto_mask(0))   # custom: opto trials only
        filtered_session.stats(['accuracy'])           # stats on filtered trials
    """
    session_id: str
    session_idx: int
    date: date
    metadata: SessionMetadata
    trials: TrialData
    masking: bool = False

    csv_path: Optional[str] = None

    # Filter provenance (None = unfiltered raw data)
    filter_info: Optional[Dict[str, Any]] = field(default=None, repr=False)

    # Set by AnimalData after construction
    _days_since_first: Optional[float] = field(default=None, repr=False)

    @property
    def n_trials(self) -> int:
        """Number of trials (reflects filtering if filtered)."""
        return self.trials.n_trials

    @property
    def stage(self) -> str:
        return self.metadata.stage

    @property
    def distribution(self) -> str:
        dist = self.trials.get_field('distribution')
        if dist is not None and len(dist) > 0:
            vals, counts = np.unique(dist[dist != ''], return_counts=True)
            if len(vals) > 0:
                return str(vals[counts.argmax()])
        return self.metadata.get('distribution', 'Unknown')

    @property
    def days_since_first(self) -> Optional[float]:
        return self._days_since_first

    @property
    def is_filtered(self) -> bool:
        """Whether this session has been through filter()."""
        return self.filter_info is not None

    # ── Filtering (thin wrapper — logic in filtering.py) ──────────────────

    def filter(self, mask=None, label='') -> 'SessionData':
        """
        Return new SessionData with filtered trials. See filtering.filter_session.

        If mask is None, applies standard exclusions (abort + opto).
        filter_info metadata records what was done.

        Examples:
            session.filter()                                # standard
            session.filter(session.trials.opto_mask(0))     # opto only
            session.filter(session.trials.correct, 'correct trials')
        """
        from behav_utils.data.filtering import filter_session
        return filter_session(self, mask=mask, label=label)

    # ── Array extraction ───────────────────────────────────────────────────

    def get_arrays(self) -> Dict[str, np.ndarray]:
        """
        Extract trial arrays. Delegates to trials.get_arrays().

        No filtering is done here. Filter before calling:
            filtered = session.filter()
            arr = filtered.get_arrays()
        """
        return self.trials.get_arrays()

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(
        self,
        stat_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute summary stats. Delegates to TrialData.stats().

        No filtering is done here. Filter before calling:
            filtered = session.filter()
            filtered.stats(['accuracy', 'pse'])
        """
        return self.trials.stats(stat_names=stat_names)

    def summary(self) -> Dict[str, Any]:
        """Quick summary dict with basic counts and accuracy."""
        valid = self.trials.valid_mask
        n_valid = valid.sum()
        choices = self.trials.choice[valid].astype(float)
        cats = self.trials.category[valid]
        result = {
            'session_id': self.session_id,
            'session_idx': self.session_idx,
            'date': self.date,
            'stage': self.stage,
            'distribution': self.distribution,
            'n_trials': self.n_trials,
            'n_valid': int(n_valid),
            'n_abort': int(self.trials.abort.sum()),
            'perf': float((choices == cats).mean()) if n_valid > 0 else np.nan,
        }
        if self.filter_info:
            result['filter'] = self.filter_info['label']
        return result

    # ── Plotting ────────────────────────────────────────────────────────────

    def plot_psychometric(self, ax=None, **kwargs):
        """
        Plot psychometric curve for this session.

        Thin wrapper around plot_psychometric(session, ax, **kwargs).

        Common kwargs:
            color:        str — curve colour (default: steelblue)
            label:        str — legend label
            n_bootstrap:  int — bootstrap iterations for CI band (0 = off)
            show_ci:      bool — show CI band (default True if n_bootstrap > 0)
            show_data:    bool — show binned data points (default True)
            show_params:  bool — annotate PSE and slope (default False)
            n_bins:       int — number of stimulus bins (default 8)
            title:        str — axes title (default: session_id)

        Returns:
            (fig, ax) where info is a dict of fit parameters.
        """
        from behav_utils.analysis.psychometry import compute_psychometric
        from behav_utils.plotting.psychometric import plot_psychometric
        result = compute_psychometric([self], mode='pooled')
        kwargs.setdefault('title', self.session_id)
        return plot_psychometric(result, ax=ax, **kwargs)

    def plot_trials(self, **kwargs):
        """
        Plot trial-by-trial raster for this session.

        Thin wrapper: calls compute_session_raster then plot_session_raster.

        Returns:
            (fig, ax)
        """
        from behav_utils.analysis.session_raster import compute_session_raster
        from behav_utils.plotting.session import plot_session_raster
        result = compute_session_raster(self)
        kwargs.setdefault('title', self.session_id)
        return plot_session_raster(result, **kwargs)


# =============================================================================
# FITTING DATA (bridge between AnimalData and SBI inference)
# =============================================================================

@dataclass
class FittingData:
    """
    Structured container for SBI model fitting.

    Provides per-session trial arrays with a consistent interface expected
    by SBIFitter and the ``build_simulator`` function.  Created via
    ``AnimalData.get_fitting_data()``.

    Attributes:
        animal_id: Identifier string.
        session_ids: Per-session identifiers.
        session_dates: Per-session dates.
        session_indices: Ordinal session indices (int array).
        stimuli: List of 1-D stimulus arrays (one per session).
        categories: List of 1-D category arrays.
        choices: List of 1-D choice arrays (NaN = no response).
        no_response: List of boolean masks (True = no response).
        not_blockstart: List of boolean masks (True = not first trial).
        n_sessions: Number of sessions.
        trials_per_session: Array of trial counts.
        time_axis: Alias for ``session_indices`` (used by SBIFitter).
    """
    animal_id: str
    session_ids: List[str]
    session_dates: List[Any]
    session_indices: np.ndarray
    stimuli: List[np.ndarray]
    categories: List[np.ndarray]
    choices: List[np.ndarray]
    no_response: List[np.ndarray]
    not_blockstart: List[np.ndarray]
    n_sessions: int
    trials_per_session: np.ndarray

    @property
    def time_axis(self) -> np.ndarray:
        """Session indices as a float array (for trajectory plotting)."""
        return self.session_indices.astype(float)

    def get_session(self, idx: int) -> Dict[str, np.ndarray]:
        """Return a single session's arrays as a dict."""
        return {
            'stimuli': self.stimuli[idx],
            'categories': self.categories[idx],
            'choices': self.choices[idx],
            'no_response': self.no_response[idx],
            'not_blockstart': self.not_blockstart[idx],
        }

    def pool(self) -> Dict[str, np.ndarray]:
        """
        Concatenate all sessions into single arrays.

        Only valid (responded) trials are included.

        Returns:
            Dict with 'stimuli', 'categories', 'choices' (1-D each).
        """
        all_stim, all_cat, all_choice = [], [], []
        for i in range(self.n_sessions):
            valid = ~self.no_response[i]
            all_stim.append(self.stimuli[i][valid])
            all_cat.append(self.categories[i][valid])
            all_choice.append(self.choices[i][valid])
        return {
            'stimuli': np.concatenate(all_stim),
            'categories': np.concatenate(all_cat),
            'choices': np.concatenate(all_choice),
        }


# =============================================================================
# ANIMAL DATA
# =============================================================================

@dataclass
class AnimalData:
    """
    All data for a single animal. Unit of model fitting.
    Sessions stored chronologically.
    """
    animal_id: str
    sessions: List[SessionData]
    metadata: Dict[str, Any] = field(default_factory=dict)
    _config: Optional[Any] = field(default=None, repr=False)

    # Cache
    _feature_matrix_cache: Optional[pd.DataFrame] = field(
        default=None, repr=False
    )

    def __post_init__(self):
        self._compute_time_axes()

    def _compute_time_axes(self):
        if not self.sessions:
            return
        self.sessions.sort(key=lambda s: s.date)
        first_date = self.sessions[0].date
        for i, sess in enumerate(self.sessions):
            sess.session_idx = i
            sess._days_since_first = (sess.date - first_date).days

    def invalidate_cache(self):
        """Call when sessions change."""
        self._feature_matrix_cache = None

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)
    
    @property
    def genotype(self) -> str:
        return self.metadata.get('genotype', 'unknown')

    @property
    def stages(self) -> List[str]:
        return list(dict.fromkeys(s.stage for s in self.sessions))

    # ── Filtering ───────────────────────────────────────────────────────────

    def get_sessions(
        self,
        stage: Optional[Union[str, List[str]]] = None,
        distribution: Optional[str] = None,
        idx_range: Optional[Tuple[int, int]] = None,
        date_range: Optional[Tuple[date, date]] = None,
        return_indices: bool = False,
    ) -> Union[List[SessionData], Tuple[List[SessionData], List[int]]]:
        """
        Filter sessions by criteria.

        Args:
            stage: Stage filter. A single string matches exactly;
                a list matches any (OR logic). None = no filter.
            distribution: Distribution filter (exact match)
            idx_range: (start, end) session index range (inclusive)
            date_range: (start, end) date range (inclusive)
            return_indices: If True, also return session_idx values.

        Returns:
            If return_indices=False (default):
                List of matching SessionData, in chronological order.
            If return_indices=True:
                (sessions, indices) where indices is a list of
                session_idx values for each matched session.
        """
        sessions = self.sessions

        if stage is not None:
            if isinstance(stage, list):
                stage_set = set(stage)
                sessions = [s for s in sessions if s.stage in stage_set]
            else:
                sessions = [s for s in sessions if s.stage == stage]
        if distribution is not None:
            sessions = [s for s in sessions if s.distribution == distribution]
        if idx_range is not None:
            sessions = [s for s in sessions
                        if idx_range[0] <= s.session_idx <= idx_range[1]]
        if date_range is not None:
            sessions = [s for s in sessions
                        if date_range[0] <= s.date <= date_range[1]]

        if return_indices:
            return sessions, [s.session_idx for s in sessions]
        return sessions

    # ── Stats ───────────────────────────────────────────────────────────────

    def feature_matrix(
        self,
        stage: Optional[str] = None,
        stat_names: Optional[List[str]] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Build session x feature DataFrame.

        Caches the result — call invalidate_cache() if sessions change.
        """
        from behav_utils.analysis.session_features import build_feature_matrix

        cache_key = (stage, tuple(stat_names) if stat_names else None)

        if use_cache and self._feature_matrix_cache is not None:
            return self._feature_matrix_cache

        df = build_feature_matrix(
            self, stage=stage, stat_names=stat_names, **kwargs,
        )
        if use_cache:
            self._feature_matrix_cache = df
        return df

    def stat_trajectory(
        self,
        stat_name: str,
        stage: Optional[str] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract one stat across sessions.

        Returns:
            (session_indices, values) — both np.ndarray
        """
        df = self.feature_matrix(stage=stage, **kwargs)
        if stat_name not in df.columns:
            raise ValueError(
                f"Stat '{stat_name}' not in feature matrix. "
                f"Available: {sorted(df.columns)}"
            )
        return (
            df['session_idx'].values,
            df[stat_name].values,
        )

    def expert_baseline(
        self,
        features: List[str],
        stage: Optional[str] = None,
        last_n: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Mean and std of last N sessions for each feature.

        Returns:
            {feature: {'mean': float, 'std': float}}
        """
        df = self.feature_matrix(stage=stage)
        if len(df) < last_n:
            last_n = len(df)

        tail = df.iloc[-last_n:]
        result = {}
        for feat in features:
            if feat in tail.columns:
                vals = tail[feat].dropna()
                result[feat] = {
                    'mean': float(vals.mean()) if len(vals) > 0 else np.nan,
                    'std': float(vals.std()) if len(vals) > 0 else np.nan,
                }
        return result

    # ── Plotting ────────────────────────────────────────────────────────────

    def plot_psychometric(self, ax=None, mode='pooled', **kwargs):
        """
        Plot psychometric curve for this animal's sessions.

        Thin wrapper: calls compute_psychometric then plot_psychometric.

        Args:
            ax:   matplotlib Axes (creates new if None)
            mode: How to combine sessions:
                'pooled'       — concatenate all trials, single fit (default)
                'overlay'      — per-session curves, colour gradient
                'session_mean' — mean P(B) ± SEM across sessions

        Common kwargs:
            color, label, n_bootstrap, show_ci, show_data, show_params,
            n_bins, title

        Returns:
            (fig, ax)
        """
        from behav_utils.analysis.psychometry import compute_psychometric
        from behav_utils.plotting.psychometric import plot_psychometric
        result = compute_psychometric(self.sessions, mode=mode,
                                    n_bootstrap=kwargs.pop('n_bootstrap', 0))
        kwargs.setdefault('title', self.animal_id)
        return plot_psychometric(result, ax=ax, **kwargs)

    def plot_trajectory(self, stat_name, ax=None, **kwargs):
        """
        Plot a summary stat across this animal's sessions.

        Thin wrapper: calls compute_trajectory then plot_trajectory.

        Args:
            stat_name: Any registered stat name ('accuracy', 'pse', etc.)
            ax:        matplotlib Axes (creates new if None)

        Common kwargs:
            color, label, marker, markersize, title

        Returns:
            (fig, ax)
        """
        from behav_utils.analysis.trajectory import compute_trajectory
        from behav_utils.plotting.trajectory import plot_trajectory
        result = compute_trajectory(self.sessions, stat_names=[stat_name])
        kwargs.setdefault('title', self.animal_id)
        return plot_trajectory(result, stat_name=stat_name, ax=ax, **kwargs)

    def plot_um(self, ax=None, **kwargs):
        """
        Plot pooled update matrix for this animal.

        Thin wrapper: calls compute_um and plot_um internally.

        Args:
            ax: matplotlib Axes (creates new if None)

        Common kwargs:
            n_bins, method, cmap, vmin, vmax, colorbar, title

        Returns:
            (fig, ax)
        """
        from behav_utils.analysis.update_matrix import compute_um
        from behav_utils.plotting.update_matrix import plot_um
        result = compute_um(self.sessions)
        kwargs.setdefault('title', self.animal_id)
        return plot_um(result, ax=ax, **kwargs)

    def plot_overview(
        self,
        stats: Optional[List[str]] = None,
        psych_mode: str = 'pooled',
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        """
        Single-animal summary: psychometric + stat trajectories.

        Layout: [psychometric | stat_1 | stat_2 | stat_3]

        Args:
            stats:      Stat names for trajectory panels.
                        Default: ['accuracy', 'pse', 'recency']
            psych_mode: Mode for psychometric panel ('pooled', 'session_mean')
            figsize:    Figure size (auto-computed if None)

        Returns:
            (fig, axes) — axes is a 1D array of length 1 + len(stats)
        """
        from behav_utils.plotting.psychometric import plot_psychometric
        from behav_utils.plotting.trajectory import plot_trajectory

        if stats is None:
            stats = ['accuracy', 'pse', 'recency']

        n_panels = 1 + len(stats)
        if figsize is None:
            figsize = (4.5 * n_panels, 4)

        fig, axes = plt.subplots(1, n_panels, figsize=figsize)
        if n_panels == 1:
            axes = np.array([axes])

        plot_psychometric(self, ax=axes[0], mode=psych_mode,
                         title=self.animal_id, **kwargs)

        for i, sn in enumerate(stats):
            try:
                plot_trajectory(self, sn, ax=axes[i + 1], title=sn)
            except (ValueError, KeyError):
                axes[i + 1].text(0.5, 0.5, f'{sn}\n(not available)',
                                 transform=axes[i + 1].transAxes,
                                 ha='center', va='center', fontsize=9)

        fig.tight_layout()
        return fig, axes

    def _resolve_sessions(
        self,
        sessions: Union[str, List[int]],
        stage: Optional[str] = None,
    ) -> List[SessionData]:
        """
        Resolve session selector to list of SessionData.

        Args:
            sessions: Selector string or list of indices.
                'all'      — all sessions
                'last_N'   — last N sessions (e.g. 'last_5')
                'first_N'  — first N sessions
                [0, 5, 10] — specific session indices
            stage: Filter to this training stage first.

        Returns:
            List of SessionData in chronological order.
        """
        pool = self.get_sessions(stage=stage) if stage else self.sessions

        if isinstance(sessions, str):
            if sessions == 'all':
                return pool
            elif sessions.startswith('last_'):
                n = int(sessions.split('_')[1])
                return pool[-n:]
            elif sessions.startswith('first_'):
                n = int(sessions.split('_')[1])
                return pool[:n]
            else:
                raise ValueError(f"Unknown session selector: '{sessions}'")
        elif isinstance(sessions, list):
            return [pool[i] for i in sessions if i < len(pool)]
        else:
            raise TypeError(f"sessions must be str or list, got {type(sessions)}")

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'AnimalData':
        with open(path, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# EXPERIMENT DATA
# =============================================================================

@dataclass
class ExperimentData:
    """
    Top-level container for all animals.
    Provides query API for multi-animal analysis and plotting.
    """
    animals: Dict[str, AnimalData] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Optional[Any] = field(default=None, repr=False)  # ProjectConfig

    def add_animal(self, animal: AnimalData) -> None:
        self.animals[animal.animal_id] = animal
        # Propagate config reference
        if self.config is not None:
            animal._config = self.config

    @property
    def animal_ids(self) -> List[str]:
        return sorted(self.animals.keys())

    @property
    def n_animals(self) -> int:
        return len(self.animals)

    def get_animal(self, animal_id: str) -> AnimalData:
        if animal_id not in self.animals:
            raise KeyError(
                f"Animal '{animal_id}' not found. "
                f"Available: {self.animal_ids}"
            )
        return self.animals[animal_id]

    # ── Filtering ───────────────────────────────────────────────────────────

    def get_animals(
        self,
        min_sessions: int = 1,
        stage: Optional[Union[str, List[str]]] = None,
        animal_ids: Optional[List[str]] = None,
    ) -> List[AnimalData]:
        """
        Filter animals by criteria.

        Args:
            min_sessions: Minimum sessions (of given stage) required
            stage: Only count sessions of this stage
            animal_ids: Restrict to these animals

        Returns:
            List of qualifying AnimalData
        """
        result = []
        for animal in self.animals.values():
            if animal_ids is not None and animal.animal_id not in animal_ids:
                continue
            if stage is not None:
                n = len(animal.get_sessions(stage=stage))
            else:
                n = animal.n_sessions
            if n >= min_sessions:
                result.append(animal)
        return result

    def get_sessions(
        self,
        stage: Optional[Union[str, List[str]]] = None,
        min_sessions_per_animal: int = 1,
        return_indices: bool = False,
        **kwargs,
    ) -> Union[List[SessionData], Tuple[List[SessionData], List[int]]]:
        """
        Get all sessions matching criteria across all animals.

        Args:
            stage: Stage filter (str or list)
            min_sessions_per_animal: Minimum sessions per animal
            return_indices: If True, also return session_idx values.
            **kwargs: Passed to AnimalData.get_sessions (e.g. distribution)

        Returns:
            If return_indices=False: List of SessionData
            If return_indices=True: (sessions, indices)
        """
        animals = self.get_animals(
            min_sessions=min_sessions_per_animal, stage=stage,
        )
        sessions = []
        for animal in animals:
            sessions.extend(animal.get_sessions(stage=stage, **kwargs))

        if return_indices:
            return sessions, [s.session_idx for s in sessions]
        return sessions

    # ── Stats ───────────────────────────────────────────────────────────────

    def feature_matrix(
        self,
        stage: Optional[str] = None,
        min_sessions: int = 5,
        stat_names: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Build pooled session x feature DataFrame across all animals.
        """
        from behav_utils.analysis.session_features import build_feature_matrix

        animals = self.get_animals(min_sessions=min_sessions, stage=stage)
        dfs = []
        for animal in animals:
            df = animal.feature_matrix(stage=stage, stat_names=stat_names, **kwargs)
            if len(df) > 0:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def summary(self) -> pd.DataFrame:
        """One-row-per-animal summary."""
        rows = []
        for aid, animal in self.animals.items():
            rows.append({
                'animal_id': aid,
                'n_sessions': animal.n_sessions,
                'stages': animal.stages,
                'date_first': animal.sessions[0].date if animal.sessions else None,
                'date_last': animal.sessions[-1].date if animal.sessions else None,
            })
        return pd.DataFrame(rows)

    # ── Plotting ────────────────────────────────────────────────────────────

    def plot_trajectory(self, stat_name, ax=None, combine='mean_sem',
                        animals='all', min_sessions=5, stage=None, **kwargs):
        """
        Plot stat trajectory across animals.

        Resolves animals, computes per-animal trajectories via
        compute_trajectory, then combines and plots.

        Args:
            stat_name: Registered stat name ('accuracy', 'pse', etc.)
            ax:        matplotlib Axes (creates new if None)
            combine:   How to combine multiple animals:
                'none'       — overlay individual animals
                'mean_sem'   — cohort mean ± SEM (default)
            animals:   'all' or list of animal IDs
            min_sessions: Minimum sessions per animal
            stage:     Stage filter

        Returns:
            (fig, ax)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from behav_utils.analysis.trajectory import compute_trajectory
        from behav_utils.plotting.trajectory import plot_trajectory
        from behav_utils.plotting.styles import PALETTE

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        else:
            fig = ax.get_figure()

        animal_list = self._resolve_animals(animals, min_sessions, stage)

        if not animal_list:
            ax.text(0.5, 0.5, 'No animals', transform=ax.transAxes,
                    ha='center', va='center')
            return fig, ax

        if combine == 'none':
            # Overlay individual animals
            for i, animal in enumerate(animal_list):
                sessions = list(animal.sessions)
                if not sessions:
                    continue
                result = compute_trajectory(sessions, stat_names=[stat_name])
                colour = PALETTE[i % len(PALETTE)]
                plot_trajectory(result, stat_name=stat_name, ax=ax,
                            color=colour, label=animal.animal_id, **kwargs)
            ax.legend(fontsize=8)

        elif combine in ('mean_sem', 'mean_only'):
            # Compute per-animal, then average across animals
            all_values = []
            max_len = 0
            for animal in animal_list:
                sessions = list(animal.sessions)
                if not sessions:
                    continue
                result = compute_trajectory(sessions, stat_names=[stat_name])
                vals = result['values'][stat_name]
                all_values.append(vals)
                max_len = max(max_len, len(vals))

            if not all_values:
                return fig, ax

            # Pad shorter animals with NaN and compute mean/SEM
            padded = np.full((len(all_values), max_len), np.nan)
            for i, vals in enumerate(all_values):
                padded[i, :len(vals)] = vals

            mean = np.nanmean(padded, axis=0)
            x = np.arange(max_len)

            colour = kwargs.pop('color', PALETTE[0])
            label = kwargs.pop('label', f'n={len(all_values)} animals')
            ax.plot(x, mean, '-o', ms=3, color=colour, label=label, **kwargs)

            if combine == 'mean_sem':
                n_valid = np.sum(~np.isnan(padded), axis=0)
                sem = np.nanstd(padded, axis=0, ddof=1) / np.sqrt(
                    np.maximum(n_valid, 1))
                ax.fill_between(x, mean - sem, mean + sem,
                            color=colour, alpha=0.15)

            ax.legend(fontsize=9)

        ax.set_xlabel('Session')
        ax.set_ylabel(stat_name)
        title = kwargs.get('title', f'{stat_name} ({combine})')
        ax.set_title(title)

        return fig, ax

    def plot_psychometric(self, ax=None, mode='pooled', animals='all',
                        min_sessions=5, stage=None, **kwargs):
        """
        Plot psychometric curve across animals.

        Resolves animals, collects their sessions, then calls
        compute_psychometric → plot_psychometric.

        Args:
            ax:   matplotlib Axes (creates new if None)
            mode: 'pooled', 'overlay', or 'session_mean'
            animals: 'all' or list of animal IDs
            min_sessions: Minimum sessions per animal
            stage: Stage filter

        Returns:
            (fig, ax)
        """
        from behav_utils.analysis.psychometry import compute_psychometric
        from behav_utils.plotting.psychometric import plot_psychometric

        animal_list = self._resolve_animals(animals, min_sessions, stage)
        all_sessions = []
        for animal in animal_list:
            all_sessions.extend(animal.sessions)

        if not all_sessions:
            import matplotlib.pyplot as plt
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
            ax.text(0.5, 0.5, 'No sessions', transform=ax.transAxes,
                    ha='center', va='center')
            return fig, ax

        result = compute_psychometric(all_sessions, mode=mode,
                                    n_bootstrap=kwargs.pop('n_bootstrap', 0))
        kwargs.setdefault('title', f'Cohort ({len(animal_list)} animals)')
        return plot_psychometric(result, ax=ax, **kwargs)

    def plot_overview(
        self,
        animals='all',
        stats=None,
        psych_mode='pooled',
        min_sessions=5,
        stage=None,
        figsize_per_panel=(4.0, 3.5),
        **kwargs,
    ):
        """
        Multi-animal overview: one row per animal.

        Layout per row: [psychometric | stat_1 | stat_2 | stat_3]

        Args:
            animals:          'all' or list of animal IDs
            stats:            Stat names for trajectory panels.
                              Default: ['accuracy', 'pse', 'recency']
            psych_mode:       Mode for psychometric panel
            min_sessions:     Minimum sessions per animal
            stage:            Stage filter
            figsize_per_panel: (width, height) per panel

        Returns:
            (fig, axes) — axes is 2D array (n_animals × n_panels)
        """
        from behav_utils.plotting.psychometric import plot_psychometric
        from behav_utils.plotting.trajectory import plot_trajectory

        if stats is None:
            stats = ['accuracy', 'pse', 'recency']

        animal_list = self._resolve_animals(animals, min_sessions, stage)

        n_animals = len(animal_list)
        n_panels = 1 + len(stats)

        fig, axes = plt.subplots(
            n_animals, n_panels,
            figsize=(figsize_per_panel[0] * n_panels,
                     figsize_per_panel[1] * n_animals),
            squeeze=False,
        )

        for row, animal in enumerate(animal_list):
            plot_psychometric(animal, ax=axes[row, 0], mode=psych_mode,
                            title=animal.animal_id, **kwargs)

            for col, sn in enumerate(stats):
                ax = axes[row, col + 1]
                try:
                    plot_trajectory(animal, sn, ax=ax,
                                   title=sn if row == 0 else '')
                except (ValueError, KeyError):
                    ax.text(0.5, 0.5, f'{sn}\n(n/a)',
                            transform=ax.transAxes,
                            ha='center', va='center', fontsize=9)
        plt.tight_layout()
        return fig, axes

    def _resolve_animals(self, animals='all', min_sessions=5, stage=None):
        """Resolve animal selector to list of AnimalData."""
        if animals == 'all':
            return self.get_animals(min_sessions=min_sessions, stage=stage)
        return [self.get_animal(aid) for aid in animals]
        
    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Don't pickle config objects — reload from YAML
        config_backup = self.config
        self.config = None
        animal_configs = {}
        for aid, animal in self.animals.items():
            animal_configs[aid] = getattr(animal, '_config', None)
            animal._config = None
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self.config = config_backup
        for aid, animal in self.animals.items():
            animal._config = animal_configs[aid]

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentData':
        with open(path, 'rb') as f:
            return pickle.load(f)
