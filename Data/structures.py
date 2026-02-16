"""
Behavioural Data Structures

Containers for loading, organising, and extracting behavioural data
from the sound categorisation task. Provides a hierarchical structure:

    ExperimentData
    â””â”€â”€ AnimalData  (one per animal, unit of model fitting)
        â””â”€â”€ SessionData  (ordered chronologically)
            â”œâ”€â”€ SessionMetadata  (constant within session)
            â”œâ”€â”€ BlockInfo  (one per distribution epoch)
            â””â”€â”€ TrialData  (flat arrays, all trials in session)

Key design principles:
    - Store everything at loading, filter at fitting time
    - Raw spatial choices preserved; category-space conversion on demand
    - Block structure supports future within-session distribution switches
    - Session indexing defaults to ordinal but supports calendar-day axis

Usage:
    # Load from CSV directory
    animal = load_animal("SS05", base_path="/Head_Fixed_Behavior/Processed/SS05")
    
    # Or load full experiment
    experiment = load_experiment("/Head_Fixed_Behavior/Processed")
    
    # Extract model-ready data
    fitting_data = animal.get_fitting_data(
        stage="Full_Task_Cont",
        exclude_abort=True,
        exclude_opto=True
    )
    
    # Generate synthetic data for development
    animal = generate_synthetic_animal(
        n_sessions=20, 
        trials_per_session=300,
        true_params={...}
    )

Note:
    This module handles INPUT data. Model simulation outputs use
    the existing TrialHistory class from Models.BE_core.
    
    The existing BE model uses 0/1 encoding:
        categories: 0 = A (stimulus < 0), 1 = B (stimulus > 0)
        choices:    0 = chose A, 1 = chose B, NaN = no response
    
    Raw CSV data uses spatial encoding:
        choices: -1 = left, 1 = right, 0 = no response
    
    Conversion from spatial to category space requires Sound_Contingency:
        Low_Left_High_Right: left=-1 â†’ A=0, right=1 â†’ B=1 (direct mapping)
        Low_Right_High_Left: left=-1 â†’ B=1, right=1 â†’ A=0 (flipped)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    Optional, Dict, List, Tuple, Union, Literal, Any
)
from datetime import datetime, date
import warnings
import glob
import re


# =============================================================================
# CONSTANTS
# =============================================================================

# Columns that vary trial-to-trial (stored in TrialData)
TRIAL_COLUMNS = [
    'Trial_Number', 'Stim_Relative', 'Correct', 'Abort_Trial',
    'Reward_Side', 'First_Lick', 'Correct_Side', 'P_Right',
    'Response_Latency', 'Trial_Outcome', 'Choice',
    'Rolling_perf', 'Rolling_bias', 'Rolling_abort_rate',
    'Opto_On', 'Mask', 'Fiber', 'Perc_Opto_Trials',
    'Opto_Onset_1', 'Opto_Offset_1', 'Opto_Onset_2', 'Opto_Offset_2',
    'Zapit', 'Opto_Duration',
    'Distribution', 'Distribution_Exp_Rate',
]

# Columns constant within a session (stored in SessionMetadata)
SESSION_COLUMNS = [
    'Animal_ID', 'Protocol', 'Stage',
    'Sound_Contingency', 'Stim_Type',
    'Stim_Range_Min', 'Stim_Range_Max',
    'Anti_Bias', 'Nb_Of_Stim', 'Response_Window', 'Timeout_Duration',
    'Sound_Duration', 'Go_Cue_Duration', 'Inter_Trial_Interval',
    'Left_Valve_Time', 'Right_Valve_Time',
    'Working_Memory_Delay',
    'Window_Perf_Size', 'Min_Block_Length', 'Block_Performance_Threshold',
    'Air_Puff_Contingency', 'Air_Puff_Side', 'Air_Puff_Contingency_Rule',
]

# Default contingency: low freq = left, high freq = right
# In this mapping: stimulus > 0 â†’ category B â†’ right side
DEFAULT_CONTINGENCY = 'Low_Left_High_Right'


# =============================================================================
# CHOICE ENCODING CONVERSION
# =============================================================================

def spatial_to_category_choice(
    choice_spatial: np.ndarray,
    contingency: str = DEFAULT_CONTINGENCY
) -> np.ndarray:
    """
    Convert spatial choice (-1=left, 0=none, 1=right) to category choice
    (0=A, 1=B, NaN=no response).
    
    Args:
        choice_spatial: Array of spatial choices {-1, 0, 1}
        contingency: 'Low_Left_High_Right' or 'Low_Right_High_Left'
    
    Returns:
        Array of category choices {0, 1, NaN}
    """
    choice_cat = np.full(len(choice_spatial), np.nan)
    
    if contingency == 'Low_Left_High_Right':
        # left (-1) = low = A (0), right (1) = high = B (1)
        choice_cat[choice_spatial == -1] = 0.0
        choice_cat[choice_spatial == 1] = 1.0
    elif contingency == 'Low_Right_High_Left':
        # left (-1) = high = B (1), right (1) = low = A (0)
        choice_cat[choice_spatial == -1] = 1.0
        choice_cat[choice_spatial == 1] = 0.0
    else:
        raise ValueError(f"Unknown contingency: {contingency}")
    
    # choice_spatial == 0 stays as NaN (no response)
    return choice_cat


def stimulus_to_category(
    stimuli: np.ndarray,
    boundary: float = 0.0
) -> np.ndarray:
    """
    Derive category from stimulus value.
    
    Args:
        stimuli: Stimulus values
        boundary: Category boundary (default 0.0)
    
    Returns:
        Array of categories: 0 = A (stimulus < boundary), 1 = B (stimulus > boundary)
    
    Note:
        Stimulus exactly at boundary is assigned to B (consistent with > 0).
        In practice this is vanishingly rare with continuous stimuli.
    """
    return (stimuli > boundary).astype(int)


# =============================================================================
# DATA CONTAINERS
# =============================================================================

@dataclass
class SessionMetadata:
    """
    Session-level metadata (constant within a session).
    
    Extracted from the first row of the CSV, as these columns
    do not vary trial-to-trial.
    """
    animal_id: str
    protocol: str
    stage: str
    sound_contingency: str
    stim_type: Optional[str] = None
    stim_range_min: float = -1.0
    stim_range_max: float = 1.0
    anti_bias: Optional[Any] = None
    nb_of_stim: Optional[int] = None
    response_window: Optional[float] = None
    timeout_duration: Optional[float] = None
    sound_duration: Optional[float] = None
    go_cue_duration: Optional[float] = None
    iti: Optional[float] = None
    left_valve_time: Optional[float] = None
    right_valve_time: Optional[float] = None
    working_memory_delay: Optional[float] = None
    window_perf_size: Optional[int] = None
    min_block_length: Optional[int] = None
    block_performance_threshold: Optional[float] = None
    air_puff_contingency: Optional[Any] = None
    air_puff_side: Optional[str] = None
    air_puff_contingency_rule: Optional[str] = None
    
    @classmethod
    def from_dataframe_row(cls, row: pd.Series) -> 'SessionMetadata':
        """Extract metadata from a single row of CSV data."""
        def safe_get(key, default=None, dtype=None):
            val = row.get(key, default)
            if pd.isna(val):
                return default
            if dtype is not None:
                try:
                    return dtype(val)
                except (ValueError, TypeError):
                    return default
            return val
        
        return cls(
            animal_id=str(safe_get('Animal_ID', '')),
            protocol=str(safe_get('Protocol', '')),
            stage=str(safe_get('Stage', '')),
            sound_contingency=str(safe_get('Sound_Contingency', DEFAULT_CONTINGENCY)),
            stim_type=safe_get('Stim_Type'),
            stim_range_min=safe_get('Stim_Range_Min', -1.0, float),
            stim_range_max=safe_get('Stim_Range_Max', 1.0, float),
            anti_bias=safe_get('Anti_Bias'),
            nb_of_stim=safe_get('Nb_Of_Stim', dtype=int),
            response_window=safe_get('Response_Window', dtype=float),
            timeout_duration=safe_get('Timeout_Duration', dtype=float),
            sound_duration=safe_get('Sound_Duration', dtype=float),
            go_cue_duration=safe_get('Go_Cue_Duration', dtype=float),
            iti=safe_get('Inter_Trial_Interval', dtype=float),
            left_valve_time=safe_get('Left_Valve_Time', dtype=float),
            right_valve_time=safe_get('Right_Valve_Time', dtype=float),
            working_memory_delay=safe_get('Working_Memory_Delay', dtype=float),
            window_perf_size=safe_get('Window_Perf_Size', dtype=int),
            min_block_length=safe_get('Min_Block_Length', dtype=int),
            block_performance_threshold=safe_get('Block_Performance_Threshold', dtype=float),
            air_puff_contingency=safe_get('Air_Puff_Contingency'),
            air_puff_side=safe_get('Air_Puff_Side'),
            air_puff_contingency_rule=safe_get('Air_Puff_Contingency_Rule'),
        )


@dataclass
class BlockInfo:
    """
    A distribution epoch within a session.
    
    Currently one block per session (single distribution).
    Future: multiple blocks when within-session distribution switches occur.
    """
    block_idx: int
    distribution: str         # 'Uniform', 'Hard-A', 'Hard-B'
    exp_rate: Optional[float] # Distribution parameter
    trial_start: int          # First trial index (0-based within session)
    trial_end: int            # Last trial index (inclusive)
    
    @property
    def n_trials(self) -> int:
        return self.trial_end - self.trial_start + 1


@dataclass
class TrialData:
    """
    Per-trial arrays for a single session.
    
    All arrays have shape (n_trials,). Stores both raw spatial encoding
    and derived category-space encoding.
    
    Encoding conventions:
        choice_spatial: -1 = left, 1 = right, 0 = no response (raw from CSV)
        choice_category: 0 = A, 1 = B, NaN = no response (for BE model)
        category: 0 = A (stim < 0), 1 = B (stim > 0) (derived from stimulus)
    """
    # Core trial data
    trial_number: np.ndarray       # Original trial numbers from CSV
    stimulus: np.ndarray           # Stim_Relative, float in [-1, 1]
    category: np.ndarray           # Derived: 0=A, 1=B (from stimulus sign)
    choice_spatial: np.ndarray     # Raw: -1=left, 0=none, 1=right
    choice_category: np.ndarray    # Converted: 0=A, 1=B, NaN=no response
    correct: np.ndarray            # bool
    outcome: np.ndarray            # str: 'Correct', 'Incorrect', 'Abort'
    abort: np.ndarray              # bool
    reaction_time: np.ndarray      # Response_Latency (float, NaN if abort)
    
    # Block assignment
    block_idx: np.ndarray          # int: which block each trial belongs to
    
    # Rolling performance (kept for quick sanity checks)
    rolling_perf: np.ndarray
    rolling_bias: np.ndarray
    rolling_abort_rate: np.ndarray
    
    # Opto columns (store all for future use)
    opto_on: np.ndarray            # bool
    opto_mask: np.ndarray
    opto_fiber: np.ndarray
    opto_perc_trials: np.ndarray
    opto_onset_1: np.ndarray
    opto_offset_1: np.ndarray
    opto_onset_2: np.ndarray
    opto_offset_2: np.ndarray
    opto_duration: np.ndarray
    opto_zapit: np.ndarray
    
    # Additional columns from CSV (catch-all for future columns)
    extra: Dict[str, np.ndarray] = field(default_factory=dict)
    
    @property
    def n_trials(self) -> int:
        return len(self.stimulus)
    
    @property
    def no_response(self) -> np.ndarray:
        """Boolean mask: True where animal did not respond."""
        return np.isnan(self.choice_category)
    
    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask: True for non-abort trials with a response."""
        return ~self.abort & ~self.no_response
    
    def to_model_trace(
        self,
        p_B: np.ndarray,
        s_hat: np.ndarray,
        beliefs: np.ndarray,
        x: np.ndarray,
        exclude_abort: bool = True,
        exclude_opto: bool = True,
    ) -> 'ModelTrace':
        """
        Create a ModelTrace by combining this trial data with model outputs.
        
        Bridges experimental data with model computation results.
        Requires Models.BE_core.ModelTrace (lazy import to avoid circular dep).
        
        Args:
            p_B: Model's P(choose B) per trial (after filtering)
            s_hat: Perceived stimulus per trial (after filtering)
            beliefs: Belief distributions (n_filtered_trials, n_points)
            x: Discretisation grid
            exclude_abort: Whether model was run with aborts excluded
            exclude_opto: Whether model was run with opto excluded
        
        Returns:
            ModelTrace with matched input and output arrays
        """
        from Models.BE_core import ModelTrace
        return ModelTrace.from_trial_data(
            self, p_B, s_hat, beliefs, x,
            exclude_abort=exclude_abort,
            exclude_opto=exclude_opto,
        )
    
    def get_model_arrays(
        self,
        exclude_abort: bool = True,
        exclude_opto: bool = True,
        exclude_no_response: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Extract arrays compatible with BEModel.simulate_session / compute_log_likelihood.
        
        Args:
            exclude_abort: Remove abort trials
            exclude_opto: Remove opto-on trials  
            exclude_no_response: Remove no-response trials entirely.
                If False (default), no-response trials are kept but marked
                with no_response mask so the model skips choice but still
                updates belief state.
        
        Returns:
            Dict with keys: 'stimuli', 'categories', 'choices', 'no_response',
            'reaction_times', 'trial_indices' (original indices for back-mapping)
        """
        mask = np.ones(self.n_trials, dtype=bool)
        
        if exclude_abort:
            mask &= ~self.abort
        if exclude_opto:
            mask &= ~self.opto_on
        if exclude_no_response:
            mask &= ~self.no_response
        
        stimuli = self.stimulus[mask]
        categories = self.category[mask]
        choices = self.choice_category[mask]
        no_resp = np.isnan(choices)
        rt = self.reaction_time[mask]
        
        # not_blockstart: detect block boundaries within filtered data
        block_ids = self.block_idx[mask]
        not_blockstart = np.ones(len(stimuli), dtype=bool)
        if len(not_blockstart) > 0:
            not_blockstart[0] = False
            # Mark first trial of each new block
            for i in range(1, len(block_ids)):
                if block_ids[i] != block_ids[i - 1]:
                    not_blockstart[i] = False
        
        return {
            'stimuli': stimuli,
            'categories': categories,
            'choices': choices,
            'no_response': no_resp,
            'not_blockstart': not_blockstart,
            'reaction_times': rt,
            'trial_indices': np.where(mask)[0],
        }


@dataclass
class SessionData:
    """
    All data for a single behavioural session.
    
    Contains metadata, block structure, and trial-level data.
    """
    session_id: str               # e.g. "SOUND_CAT_SS05_2026_1_27"
    session_idx: int              # Ordinal index (0-based, chronological)
    date: date                    # Session date
    metadata: SessionMetadata
    blocks: List[BlockInfo]
    trials: TrialData
    
    # Source info
    csv_path: Optional[str] = None
    
    @property
    def n_trials(self) -> int:
        return self.trials.n_trials
    
    @property
    def n_blocks(self) -> int:
        return len(self.blocks)
    
    @property
    def stage(self) -> str:
        return self.metadata.stage
    
    @property
    def distribution(self) -> str:
        """Primary distribution (first block). Use blocks for multi-block sessions."""
        return self.blocks[0].distribution if self.blocks else 'Unknown'
    
    @property
    def days_since_first(self) -> Optional[float]:
        """Days since first session. Set by AnimalData after construction."""
        return self._days_since_first if hasattr(self, '_days_since_first') else None
    
    def summary(self) -> Dict[str, Any]:
        """Quick summary statistics for the session."""
        valid = self.trials.valid_mask
        n_valid = valid.sum()
        return {
            'session_id': self.session_id,
            'session_idx': self.session_idx,
            'date': self.date,
            'stage': self.stage,
            'distribution': self.distribution,
            'n_trials': self.n_trials,
            'n_valid': int(n_valid),
            'n_abort': int(self.trials.abort.sum()),
            'perf': float(self.trials.correct[valid].mean()) if n_valid > 0 else np.nan,
            'bias': float(self.trials.choice_category[valid].mean()) if n_valid > 0 else np.nan,
        }


# =============================================================================
# ANIMAL DATA
# =============================================================================

@dataclass
class AnimalData:
    """
    All data for a single animal. Unit of model fitting.
    
    Sessions are stored chronologically. Provides methods to filter
    sessions and extract model-ready data.
    """
    animal_id: str
    sessions: List[SessionData]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute derived quantities after construction."""
        self._compute_time_axes()
    
    def _compute_time_axes(self):
        """Compute session indices and calendar-day offsets."""
        if not self.sessions:
            return
        
        # Sort by date (should already be sorted, but enforce)
        self.sessions.sort(key=lambda s: s.date)
        
        # Assign ordinal indices
        for i, sess in enumerate(self.sessions):
            sess.session_idx = i
        
        # Compute days since first session
        first_date = self.sessions[0].date
        for sess in self.sessions:
            delta = sess.date - first_date
            sess._days_since_first = delta.days
    
    @property
    def n_sessions(self) -> int:
        return len(self.sessions)
    
    @property
    def stages(self) -> List[str]:
        """Unique stages present in data."""
        return list(dict.fromkeys(s.stage for s in self.sessions))
    
    def get_sessions(
        self,
        stage: Optional[str] = None,
        distribution: Optional[str] = None,
        idx_range: Optional[Tuple[int, int]] = None,
        date_range: Optional[Tuple[date, date]] = None,
    ) -> List[SessionData]:
        """
        Filter sessions by criteria.
        
        Args:
            stage: Keep only this stage (e.g. 'Full_Task_Cont')
            distribution: Keep only this distribution (e.g. 'Uniform')
            idx_range: (start, end) session indices (inclusive)
            date_range: (start_date, end_date) inclusive
        
        Returns:
            Filtered list of SessionData
        """
        sessions = self.sessions
        
        if stage is not None:
            sessions = [s for s in sessions if s.stage == stage]
        if distribution is not None:
            sessions = [s for s in sessions if s.distribution == distribution]
        if idx_range is not None:
            sessions = [s for s in sessions 
                       if idx_range[0] <= s.session_idx <= idx_range[1]]
        if date_range is not None:
            sessions = [s for s in sessions 
                       if date_range[0] <= s.date <= date_range[1]]
        
        return sessions
    
    def get_fitting_data(
        self,
        stage: str = 'Full_Task_Cont',
        exclude_abort: bool = True,
        exclude_opto: bool = True,
        exclude_no_response: bool = False,
        time_axis: Literal['session_idx', 'calendar_days'] = 'session_idx',
        min_valid_trials: int = 10,
    ) -> 'FittingData':
        """
        Extract model-ready data for SBI fitting.
        
        Filters sessions and trials, returning arrays compatible with
        the BE model inference pipeline.
        
        Args:
            stage: Which stage to include
            exclude_abort: Remove abort trials
            exclude_opto: Remove opto trials
            exclude_no_response: Remove no-response trials entirely
            time_axis: 'session_idx' (ordinal) or 'calendar_days' (real time)
            min_valid_trials: Skip sessions with fewer valid trials
        
        Returns:
            FittingData container with per-session arrays and time axis
        """
        sessions = self.get_sessions(stage=stage)
        
        session_arrays = []
        time_values = []
        session_ids = []
        session_dates = []
        
        for sess in sessions:
            arrays = sess.trials.get_model_arrays(
                exclude_abort=exclude_abort,
                exclude_opto=exclude_opto,
                exclude_no_response=exclude_no_response,
            )
            
            # Skip sessions with too few valid trials
            n_valid = (~arrays['no_response']).sum()
            if n_valid < min_valid_trials:
                warnings.warn(
                    f"Session {sess.session_id} has only {n_valid} valid trials "
                    f"(< {min_valid_trials}), skipping."
                )
                continue
            
            session_arrays.append(arrays)
            session_ids.append(sess.session_id)
            session_dates.append(sess.date)
            
            if time_axis == 'session_idx':
                time_values.append(len(session_arrays) - 1)
            elif time_axis == 'calendar_days':
                time_values.append(sess._days_since_first)
            else:
                raise ValueError(f"Unknown time_axis: {time_axis}")
        
        return FittingData(
            animal_id=self.animal_id,
            session_arrays=session_arrays,
            time_axis=np.array(time_values, dtype=float),
            time_axis_type=time_axis,
            session_ids=session_ids,
            session_dates=session_dates,
        )
    
    def summary(self) -> pd.DataFrame:
        """Summary table of all sessions."""
        rows = [s.summary() for s in self.sessions]
        return pd.DataFrame(rows)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save to pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'AnimalData':
        """Load from pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# FITTING DATA (model-ready extraction)
# =============================================================================

@dataclass
class FittingData:
    """
    Model-ready data extracted from AnimalData.
    
    This is what gets passed to the SBI fitting pipeline.
    Contains per-session arrays in BE model encoding (0/1 categories, 0/1/NaN choices).
    
    Attributes:
        animal_id: Source animal
        session_arrays: List of dicts, each from TrialData.get_model_arrays()
        time_axis: Array of time values for between-session structure
        time_axis_type: 'session_idx' or 'calendar_days'
        session_ids: Session identifiers for back-mapping
        session_dates: Session dates
    """
    animal_id: str
    session_arrays: List[Dict[str, np.ndarray]]
    time_axis: np.ndarray
    time_axis_type: str
    session_ids: List[str]
    session_dates: List[date]
    
    @property
    def n_sessions(self) -> int:
        return len(self.session_arrays)
    
    @property
    def trials_per_session(self) -> np.ndarray:
        """Number of trials per session."""
        return np.array([len(sa['stimuli']) for sa in self.session_arrays])
    
    @property
    def valid_trials_per_session(self) -> np.ndarray:
        """Number of valid (non-no-response) trials per session."""
        return np.array([
            (~sa['no_response']).sum() for sa in self.session_arrays
        ])
    
    def get_session(self, idx: int) -> Dict[str, np.ndarray]:
        """Get arrays for a single session."""
        return self.session_arrays[idx]
    
    # Convenience properties for per-session array access
    @property
    def stimuli(self) -> List[np.ndarray]:
        """List of stimulus arrays, one per session."""
        return [sa['stimuli'] for sa in self.session_arrays]
    
    @property
    def categories(self) -> List[np.ndarray]:
        """List of category arrays, one per session."""
        return [sa['categories'] for sa in self.session_arrays]
    
    @property
    def choices(self) -> List[np.ndarray]:
        """List of choice arrays, one per session."""
        return [sa['choices'] for sa in self.session_arrays]
    
    @property
    def no_response(self) -> List[np.ndarray]:
        """List of no_response masks, one per session."""
        return [sa['no_response'] for sa in self.session_arrays]
    
    @property
    def not_blockstart(self) -> List[np.ndarray]:
        """List of not_blockstart masks, one per session."""
        return [sa['not_blockstart'] for sa in self.session_arrays]
    
    def summary(self) -> pd.DataFrame:
        """Summary table of sessions in fitting data."""
        rows = []
        for i, sa in enumerate(self.session_arrays):
            n = len(sa['stimuli'])
            n_valid = (~sa['no_response']).sum()
            choices_valid = sa['choices'][~sa['no_response']]
            cats_valid = sa['categories'][~sa['no_response']]
            perf = (choices_valid == cats_valid).mean() if n_valid > 0 else np.nan
            rows.append({
                'session_idx': i,
                'session_id': self.session_ids[i],
                'date': self.session_dates[i],
                'time_value': self.time_axis[i],
                'n_trials': n,
                'n_valid': int(n_valid),
                'performance': float(perf),
            })
        return pd.DataFrame(rows)


# =============================================================================
# EXPERIMENT DATA (multi-animal)
# =============================================================================

@dataclass
class ExperimentData:
    """
    Container for multiple animals. Thin wrapper for batch operations.
    """
    animals: Dict[str, AnimalData] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_animal(self, animal: AnimalData) -> None:
        """Add an animal to the experiment."""
        self.animals[animal.animal_id] = animal
    
    @property
    def animal_ids(self) -> List[str]:
        return list(self.animals.keys())
    
    @property
    def n_animals(self) -> int:
        return len(self.animals)
    
    def get_animal(self, animal_id: str) -> AnimalData:
        """Get a specific animal's data."""
        return self.animals[animal_id]
    
    def get_animals_with_min_sessions(
        self,
        min_sessions: int,
        stage: str = 'Full_Task_Cont',
    ) -> List[AnimalData]:
        """Filter to animals with enough sessions of a given stage."""
        result = []
        for animal in self.animals.values():
            n = len(animal.get_sessions(stage=stage))
            if n >= min_sessions:
                result.append(animal)
        return result
    
    def summary(self) -> pd.DataFrame:
        """Summary table across animals."""
        rows = []
        for aid, animal in self.animals.items():
            stages = animal.stages
            task_sessions = animal.get_sessions(stage='Full_Task_Cont')
            rows.append({
                'animal_id': aid,
                'n_sessions_total': animal.n_sessions,
                'n_sessions_task': len(task_sessions),
                'stages': stages,
                'date_first': animal.sessions[0].date if animal.sessions else None,
                'date_last': animal.sessions[-1].date if animal.sessions else None,
            })
        return pd.DataFrame(rows)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save to pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentData':
        """Load from pickle."""
        with open(path, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# CSV LOADING
# =============================================================================

def _parse_date_from_path(session_dir: str) -> Optional[date]:
    """
    Extract date from session directory name.
    
    Expected format: SOUND_CAT_SS05_2026_1_27 â†’ 2026-01-27
    Also handles: trial_summary_SS05_20260127.csv â†’ 2026-01-27
    """
    # Try directory name pattern: ..._YYYY_M_D
    match = re.search(r'(\d{4})_(\d{1,2})_(\d{1,2})$', session_dir)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    
    # Try compact date: YYYYMMDD
    match = re.search(r'(\d{4})(\d{2})(\d{2})', session_dir)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    
    return None


def _parse_date_from_csv(df: pd.DataFrame) -> Optional[date]:
    """Extract date from the Date column if present."""
    if 'Date' in df.columns and len(df) > 0:
        date_val = df['Date'].iloc[0]
        if pd.notna(date_val):
            try:
                return pd.to_datetime(date_val).date()
            except (ValueError, TypeError):
                pass
    return None


def _detect_blocks(df: pd.DataFrame) -> List[BlockInfo]:
    """
    Detect distribution blocks within a session.
    
    Currently most sessions have a single block (constant Distribution).
    Future-proofed: detects changes in Distribution column.
    """
    if 'Distribution' not in df.columns:
        return [BlockInfo(
            block_idx=0,
            distribution='Unknown',
            exp_rate=None,
            trial_start=0,
            trial_end=len(df) - 1,
        )]
    
    blocks = []
    distributions = df['Distribution'].values
    exp_rates = df.get('Distribution_Exp_Rate', pd.Series([None] * len(df))).values
    
    current_dist = distributions[0]
    current_rate = exp_rates[0]
    block_start = 0
    block_idx = 0
    
    for i in range(1, len(distributions)):
        if distributions[i] != current_dist:
            blocks.append(BlockInfo(
                block_idx=block_idx,
                distribution=str(current_dist) if pd.notna(current_dist) else 'Unknown',
                exp_rate=float(current_rate) if pd.notna(current_rate) else None,
                trial_start=block_start,
                trial_end=i - 1,
            ))
            block_idx += 1
            current_dist = distributions[i]
            current_rate = exp_rates[i]
            block_start = i
    
    # Final block
    blocks.append(BlockInfo(
        block_idx=block_idx,
        distribution=str(current_dist) if pd.notna(current_dist) else 'Unknown',
        exp_rate=float(current_rate) if pd.notna(current_rate) else None,
        trial_start=block_start,
        trial_end=len(distributions) - 1,
    ))
    
    return blocks


def _safe_array(df: pd.DataFrame, col: str, dtype=None, default=None) -> np.ndarray:
    """Safely extract a column as numpy array with fallback."""
    if col in df.columns:
        arr = df[col].values
        if dtype == bool:
            # Handle string booleans ('TRUE'/'FALSE')
            if arr.dtype == object:
                arr = np.array([
                    str(v).upper() in ('TRUE', '1', 'YES') 
                    for v in arr
                ], dtype=bool)
            else:
                arr = arr.astype(bool)
        elif dtype is not None:
            try:
                arr = arr.astype(dtype)
            except (ValueError, TypeError):
                arr = pd.to_numeric(df[col], errors='coerce').values
        return arr
    else:
        n = len(df)
        if default is not None:
            return np.full(n, default, dtype=dtype if dtype else type(default))
        elif dtype == bool:
            return np.zeros(n, dtype=bool)
        else:
            return np.full(n, np.nan)


def load_session_csv(
    csv_path: Union[str, Path],
    session_id: Optional[str] = None,
    session_idx: int = 0,
) -> SessionData:
    """
    Load a single session from CSV.
    
    Args:
        csv_path: Path to CSV file
        session_id: Session identifier (inferred from path if None)
        session_idx: Session ordinal (typically set later by AnimalData)
    
    Returns:
        SessionData object
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        raise ValueError(f"Empty CSV file: {csv_path}")
    
    # Session ID
    if session_id is None:
        session_id = csv_path.parent.name
    
    # Date: try CSV column first, then path
    session_date = _parse_date_from_csv(df)
    if session_date is None:
        session_date = _parse_date_from_path(str(csv_path.parent))
    if session_date is None:
        session_date = _parse_date_from_path(csv_path.stem)
    if session_date is None:
        warnings.warn(f"Could not parse date from {csv_path}. Using epoch.")
        session_date = date(1970, 1, 1)
    
    # Metadata from first row
    metadata = SessionMetadata.from_dataframe_row(df.iloc[0])
    
    # Blocks
    blocks = _detect_blocks(df)
    
    # Build block_idx array
    block_idx_arr = np.zeros(len(df), dtype=int)
    for block in blocks:
        block_idx_arr[block.trial_start:block.trial_end + 1] = block.block_idx
    
    # Trial data
    stimulus = _safe_array(df, 'Stim_Relative', dtype=float)
    category = stimulus_to_category(stimulus)
    choice_spatial = _safe_array(df, 'Choice', dtype=float)
    choice_category = spatial_to_category_choice(
        choice_spatial, 
        contingency=metadata.sound_contingency
    )
    
    trials = TrialData(
        trial_number=_safe_array(df, 'Trial_Number', dtype=int),
        stimulus=stimulus,
        category=category,
        choice_spatial=choice_spatial,
        choice_category=choice_category,
        correct=_safe_array(df, 'Correct', dtype=bool),
        outcome=_safe_array(df, 'Trial_Outcome'),
        abort=_safe_array(df, 'Abort_Trial', dtype=bool),
        reaction_time=_safe_array(df, 'Response_Latency', dtype=float),
        block_idx=block_idx_arr,
        rolling_perf=_safe_array(df, 'Rolling_perf', dtype=float),
        rolling_bias=_safe_array(df, 'Rolling_bias', dtype=float),
        rolling_abort_rate=_safe_array(df, 'Rolling_abort_rate', dtype=float),
        opto_on=_safe_array(df, 'Opto_On', dtype=bool),
        opto_mask=_safe_array(df, 'Mask'),
        opto_fiber=_safe_array(df, 'Fiber'),
        opto_perc_trials=_safe_array(df, 'Perc_Opto_Trials', dtype=float),
        opto_onset_1=_safe_array(df, 'Opto_Onset_1', dtype=float),
        opto_offset_1=_safe_array(df, 'Opto_Offset_1', dtype=float),
        opto_onset_2=_safe_array(df, 'Opto_Onset_2', dtype=float),
        opto_offset_2=_safe_array(df, 'Opto_Offset_2', dtype=float),
        opto_duration=_safe_array(df, 'Opto_Duration', dtype=float),
        opto_zapit=_safe_array(df, 'Zapit'),
    )
    
    return SessionData(
        session_id=session_id,
        session_idx=session_idx,
        date=session_date,
        metadata=metadata,
        blocks=blocks,
        trials=trials,
        csv_path=str(csv_path),
    )


def load_animal(
    animal_id: str,
    base_path: Union[str, Path],
    csv_pattern: str = 'trial_summary_*.csv',
) -> AnimalData:
    """
    Load all sessions for a single animal from directory structure.
    
    Expected structure:
        base_path/
        â”œâ”€â”€ SESSION_DIR_1/
        â”‚   â””â”€â”€ trial_summary_ANIMAL_DATE.csv
        â”œâ”€â”€ SESSION_DIR_2/
        â”‚   â””â”€â”€ trial_summary_ANIMAL_DATE.csv
        ...
    
    Args:
        animal_id: Animal identifier (e.g. 'SS05')
        base_path: Path to animal's directory containing session folders
        csv_pattern: Glob pattern for CSV files within session dirs
    
    Returns:
        AnimalData object with sessions sorted chronologically
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Animal directory not found: {base_path}")
    
    # Find all session CSV files
    csv_files = sorted(base_path.rglob(csv_pattern))
    
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching '{csv_pattern}' found in {base_path}"
        )
    
    sessions = []
    for csv_path in csv_files:
        try:
            session = load_session_csv(csv_path)
            sessions.append(session)
        except Exception as e:
            warnings.warn(f"Failed to load {csv_path}: {e}")
            continue
    
    if not sessions:
        raise ValueError(f"No sessions successfully loaded for {animal_id}")
    
    animal = AnimalData(
        animal_id=animal_id,
        sessions=sessions,
    )
    
    return animal


def load_experiment(
    base_path: Union[str, Path],
    animal_ids: Optional[List[str]] = None,
    csv_pattern: str = 'trial_summary_*.csv',
) -> ExperimentData:
    """
    Load all animals from a Processed directory.
    
    Expected structure:
        base_path/
        â”œâ”€â”€ SS05/
        â”‚   â”œâ”€â”€ SESSION_DIR_1/ ...
        â”‚   â””â”€â”€ SESSION_DIR_2/ ...
        â”œâ”€â”€ SS06/
        â”‚   â”œâ”€â”€ ...
        ...
    
    Args:
        base_path: Path to Processed directory
        animal_ids: Specific animals to load (None = all)
        csv_pattern: Glob pattern for CSV files
    
    Returns:
        ExperimentData with all animals
    """
    base_path = Path(base_path)
    
    if animal_ids is None:
        # Auto-detect: subdirectories of base_path
        animal_dirs = sorted([
            d for d in base_path.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        animal_ids = [d.name for d in animal_dirs]
    
    experiment = ExperimentData()
    
    for aid in animal_ids:
        animal_path = base_path / aid
        if not animal_path.exists():
            warnings.warn(f"Animal directory not found: {animal_path}")
            continue
        
        try:
            animal = load_animal(aid, animal_path, csv_pattern)
            experiment.add_animal(animal)
        except Exception as e:
            warnings.warn(f"Failed to load animal {aid}: {e}")
    
    return experiment


# =============================================================================
# STIMULUS DISTRIBUTION SAMPLING
# =============================================================================

def _solve_hard_dist_lambda() -> float:
    """
    Solve e^{-lam} + lam = 2 for the default asymmetric distribution parameter.
    
    The density on [0,1] is f(x) = lam*e^{-lam*x} + e^{-lam}, which integrates
    to 1 for ANY lam > 0. The constraint lam + e^{-lam} = 2 gives the default
    where peak density is exactly 2x edge density.
    """
    from scipy.optimize import brentq
    return brentq(lambda x: x + np.exp(-x) - 2, 0.1, 5.0)

# Solve once at import time
_HARD_LAMBDA_DEFAULT = _solve_hard_dist_lambda()


def _sample_hard_B_side(
    n: int,
    rng: np.random.Generator,
    lam: Optional[float] = None,
) -> np.ndarray:
    """
    Sample from f(x) = lam*e^{-lam*x} + e^{-lam} on [0, 1].
    
    Boundary-heavy on B side: density peaks at x=0 (boundary) and
    decays toward x=1. Vectorised rejection sampling.
    
    The density integrates to 1 for any lam > 0. Higher lam = more
    trials near boundary. lam ~ 1.84 (default) gives moderate asymmetry;
    lam = 3 gives strong boundary bias; lam -> 0 approaches uniform.
    
    Args:
        n: Number of samples
        rng: Random generator
        lam: Exponential rate parameter. None = default (~1.84).
    """
    if lam is None:
        lam = _HARD_LAMBDA_DEFAULT
    k = np.exp(-lam)
    M = lam + k  # Maximum of f on [0,1], at x=0
    
    samples = np.empty(n)
    filled = 0
    while filled < n:
        batch_size = int((n - filled) * (M + 0.5)) + 64
        x = rng.uniform(0.0, 1.0, batch_size)
        u = rng.uniform(0.0, 1.0, batch_size)
        f_x = lam * np.exp(-lam * x) + k
        accepted = x[u * M <= f_x]
        take = min(len(accepted), n - filled)
        samples[filled:filled + take] = accepted[:take]
        filled += take
    return samples


def _sample_hard_A_side(
    n: int,
    rng: np.random.Generator,
    lam: Optional[float] = None,
) -> np.ndarray:
    """
    Sample from f(x) = lam*e^{lam*x} + e^{-lam} on [-1, 0].
    
    Boundary-heavy on A side: density peaks at x=0 (boundary) and
    decays toward x=-1. Vectorised rejection sampling.
    
    Args:
        n: Number of samples
        rng: Random generator
        lam: Exponential rate parameter. None = default (~1.84).
    """
    if lam is None:
        lam = _HARD_LAMBDA_DEFAULT
    k = np.exp(-lam)
    M = lam + k
    
    samples = np.empty(n)
    filled = 0
    while filled < n:
        batch_size = int((n - filled) * (M + 0.5)) + 64
        x = rng.uniform(-1.0, 0.0, batch_size)
        u = rng.uniform(0.0, 1.0, batch_size)
        f_x = lam * np.exp(lam * x) + k
        accepted = x[u * M <= f_x]
        take = min(len(accepted), n - filled)
        samples[filled:filled + take] = accepted[:take]
        filled += take
    return samples


def sample_stimuli(
    n_trials: int,
    distribution: str,
    rng: np.random.Generator,
    exp_rate: Optional[float] = None,
) -> np.ndarray:

    """
    Sample stimulus values from a named distribution on [-1, 1].
    
    Each trial: 50/50 chance of sampling from A side [-1,0] or B side [0,1],
    with the density on each side determined by the distribution type.
    
    Distributions:
        'Uniform': Uniform on both sides
        'Hard-A':  Boundary-heavy on A side, uniform on B side
                   (more hard trials where stimulus is near 0 on the A side)
        'Hard-B':  Uniform on A side, boundary-heavy on B side
                   (more hard trials where stimulus is near 0 on the B side)
    
    Mapping to old Sampling.py:
        Hard-A = Asym_right (HardA [-1,0] + Uniform [0,1])
        Hard-B = Asym_left  (Uniform [-1,0] + HardB [0,1])
    """
    # Determine which side each trial is drawn from
    is_B_side = rng.random(n_trials) < 0.5
    n_A = int(np.sum(~is_B_side))
    n_B = int(np.sum(is_B_side))
    
    stimuli = np.empty(n_trials)
    
    if distribution == 'Uniform':
        stimuli[~is_B_side] = rng.uniform(-1.0, 0.0, n_A)
        stimuli[is_B_side] = rng.uniform(0.0, 1.0, n_B)
    
    elif distribution == 'Hard-A':
        # A side: boundary-heavy; B side: uniform
        stimuli[~is_B_side] = _sample_hard_A_side(n_A, rng, lam=exp_rate)
        stimuli[is_B_side] = rng.uniform(0.0, 1.0, n_B)
    
    elif distribution == 'Hard-B':
        # A side: uniform; B side: boundary-heavy
        stimuli[~is_B_side] = rng.uniform(-1.0, 0.0, n_A)
        stimuli[is_B_side] = _sample_hard_B_side(n_B, rng, lam=exp_rate)
    
    else:
        raise ValueError(
            f"Unknown distribution: '{distribution}'. "
            f"Expected one of: 'Uniform', 'Hard-A', 'Hard-B'"
        )
    
    return stimuli

# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_session(
    session_idx: int = 0,
    n_trials: int = 300,
    distribution: str = 'Uniform',
    exp_rate: Optional[float] = None,
    contingency: str = DEFAULT_CONTINGENCY,
    abort_rate: float = 0.05,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> SessionData:
    """
    Generate a single synthetic session.
    
    Args:
        session_idx: Session ordinal index
        n_trials: Number of trials
        distribution: Stimulus distribution ('Uniform', 'Hard-A', 'Hard-B')
        exp_rate: Distribution parameter (for Hard distributions)
        contingency: Sound-side mapping
        abort_rate: Fraction of trials that are aborts
        seed: Random seed
        rng: Random generator (created from seed if None)
    
    Returns:
        SessionData with synthetic trial data (no choices yet â€” use
        generate_synthetic_animal for full simulation with BE model)
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    
    # Generate stimuli from named distribution
    stimuli = sample_stimuli(n_trials, distribution, rng, exp_rate=exp_rate)
    
    categories = stimulus_to_category(stimuli)
    
    # Generate random aborts
    aborts = rng.random(n_trials) < abort_rate
    
    # Placeholder choices (will be filled by model simulation)
    choice_spatial = np.zeros(n_trials, dtype=float)
    choice_spatial[~aborts] = np.nan  # Mark as unfilled (not abort, not yet chosen)
    choice_spatial[aborts] = 0.0      # Aborts have no response
    
    choice_category = np.full(n_trials, np.nan)
    
    # Session date (synthetic: day 0 + session_idx days)
    from datetime import timedelta
    base_date = date(2026, 1, 1)
    session_date = base_date + timedelta(days=session_idx)
    
    metadata = SessionMetadata(
        animal_id='SYN01',
        protocol='synthetic',
        stage='Full_Task_Cont',
        sound_contingency=contingency,
        stim_range_min=-1.0,
        stim_range_max=1.0,
    )
    
    blocks = [BlockInfo(
        block_idx=0,
        distribution=distribution,
        exp_rate=exp_rate,
        trial_start=0,
        trial_end=n_trials - 1,
    )]
    
    trials = TrialData(
        trial_number=np.arange(1, n_trials + 1),
        stimulus=stimuli,
        category=categories,
        choice_spatial=choice_spatial,
        choice_category=choice_category,
        correct=np.zeros(n_trials, dtype=bool),
        outcome=np.where(aborts, 'Abort', ''),
        abort=aborts,
        reaction_time=np.full(n_trials, np.nan),
        block_idx=np.zeros(n_trials, dtype=int),
        rolling_perf=np.full(n_trials, np.nan),
        rolling_bias=np.full(n_trials, np.nan),
        rolling_abort_rate=np.full(n_trials, np.nan),
        opto_on=np.zeros(n_trials, dtype=bool),
        opto_mask=np.full(n_trials, np.nan),
        opto_fiber=np.full(n_trials, np.nan),
        opto_perc_trials=np.full(n_trials, np.nan),
        opto_onset_1=np.full(n_trials, np.nan),
        opto_offset_1=np.full(n_trials, np.nan),
        opto_onset_2=np.full(n_trials, np.nan),
        opto_offset_2=np.full(n_trials, np.nan),
        opto_duration=np.full(n_trials, np.nan),
        opto_zapit=np.full(n_trials, np.nan),
    )
    
    return SessionData(
        session_id=f'SYN_S{session_idx:03d}',
        session_idx=session_idx,
        date=session_date,
        metadata=metadata,
        blocks=blocks,
        trials=trials,
    )



# =============================================================================
# DISTRIBUTION SCHEDULE
# =============================================================================

@dataclass
class DistributionEpoch:
    """
    A distribution epoch specification for synthetic data generation.
    
    Attributes:
        distribution: 'Uniform', 'Hard-A', or 'Hard-B'
        exp_rate: Exponential rate for Hard distributions (None = default ~1.15)
        n_sessions: Number of sessions with this distribution
    """
    distribution: str
    n_sessions: int
    exp_rate: Optional[float] = None
    
    def __post_init__(self):
        valid = {'Uniform', 'Hard-A', 'Hard-B'}
        if self.distribution not in valid:
            raise ValueError(
                f"Unknown distribution: '{self.distribution}'. "
                f"Expected one of: {valid}"
            )


def make_distribution_schedule(
    epochs: List[DistributionEpoch],
) -> List[Tuple[str, Optional[float]]]:
    """
    Expand a list of distribution epochs into a per-session schedule.
    
    Args:
        epochs: List of DistributionEpoch specifications.
    
    Returns:
        List of (distribution, exp_rate) tuples, one per session.
    
    Example:
        # Naive (10 Uniform) -> Expert (10 Uniform) -> Shift (10 Hard-A)
        schedule = make_distribution_schedule([
            DistributionEpoch('Uniform', 20),
            DistributionEpoch('Hard-A', 10),
        ])
        
        # Cycling: A -> B -> A -> B (5 sessions each)
        schedule = make_distribution_schedule([
            DistributionEpoch('Uniform', 10),  # baseline training
            DistributionEpoch('Hard-A', 5),
            DistributionEpoch('Hard-B', 5),
            DistributionEpoch('Hard-A', 5),
            DistributionEpoch('Hard-B', 5),
        ])
    """
    schedule = []
    for epoch in epochs:
        for _ in range(epoch.n_sessions):
            schedule.append((epoch.distribution, epoch.exp_rate))
    return schedule


# =============================================================================
# PARAMETER TRAJECTORY PRESETS
# =============================================================================

def param_trajectory_naive_to_expert(
    n_sessions: int,
    eta_start: float = 0.45,
    eta_end: float = 0.08,
    decay_rate: float = 0.12,
    sigma_percep: float = 0.15,
    A_repulsion: float = 0.10,
    eta_relax: float = 0.12,
) -> Dict[str, Union[float, List[float]]]:
    """
    Exponentially declining eta_learning trajectory.
    
    eta(s) = (eta_start - eta_end) * exp(-decay_rate * s) + eta_end
    
    All other parameters held constant.
    
    Args:
        n_sessions: Number of sessions
        eta_start: Initial learning rate (naive)
        eta_end: Asymptotic learning rate (expert)
        decay_rate: Exponential decay rate (higher = faster learning)
        sigma_percep: Perceptual noise (constant)
        A_repulsion: Serial dependence (constant)
        eta_relax: Relaxation rate (constant)
    
    Returns:
        Dict suitable for generate_synthetic_animal(true_params=...)
    """
    s = np.arange(n_sessions)
    eta = (eta_start - eta_end) * np.exp(-decay_rate * s) + eta_end
    
    return {
        'sigma_percep': sigma_percep,
        'A_repulsion': A_repulsion,
        'eta_learning': eta.tolist(),
        'eta_relax': eta_relax,
    }


def param_trajectory_full(
    n_sessions_naive: int = 15,
    n_sessions_expert: int = 5,
    n_sessions_post_shift: int = 10,
    eta_naive_start: float = 0.45,
    eta_expert: float = 0.08,
    eta_post_shift_peak: float = 0.25,
    naive_decay_rate: float = 0.15,
    post_shift_decay_rate: float = 0.20,
    sigma_percep: float = 0.15,
    A_repulsion: float = 0.10,
    eta_relax: float = 0.12,
) -> Dict[str, Union[float, List[float]]]:
    """
    Full trajectory: naive -> expert -> distribution shift -> readaptation.
    
    Naive phase: exponential decline from eta_naive_start to eta_expert.
    Expert phase: constant at eta_expert.
    Post-shift phase: jumps to eta_post_shift_peak, decays back toward eta_expert.
    
    Returns:
        Dict suitable for generate_synthetic_animal(true_params=...)
    """
    n_total = n_sessions_naive + n_sessions_expert + n_sessions_post_shift
    eta = np.empty(n_total)
    
    # Naive phase
    s = np.arange(n_sessions_naive)
    eta[:n_sessions_naive] = (
        (eta_naive_start - eta_expert) * np.exp(-naive_decay_rate * s) + eta_expert
    )
    
    # Expert phase
    eta[n_sessions_naive:n_sessions_naive + n_sessions_expert] = eta_expert
    
    # Post-shift phase
    s = np.arange(n_sessions_post_shift)
    eta[n_sessions_naive + n_sessions_expert:] = (
        (eta_post_shift_peak - eta_expert) * np.exp(-post_shift_decay_rate * s) + eta_expert
    )
    
    return {
        'sigma_percep': sigma_percep,
        'A_repulsion': A_repulsion,
        'eta_learning': eta.tolist(),
        'eta_relax': eta_relax,
    }


def param_trajectory_cycling(
    n_sessions_baseline: int = 15,
    n_sessions_per_cycle: int = 5,
    n_cycles: int = 4,
    eta_expert: float = 0.08,
    eta_first_shift: float = 0.30,
    eta_later_shift: float = 0.18,
    meta_learning_rate: float = 0.15,
    shift_decay_rate: float = 0.30,
    sigma_percep: float = 0.15,
    A_repulsion: float = 0.10,
    eta_relax: float = 0.12,
    eta_baseline_start: float = 0.45,
    baseline_decay_rate: float = 0.15,
) -> Dict[str, Union[float, List[float]]]:
    """
    Cycling trajectory with meta-learning: A -> B -> A -> B...
    
    Baseline phase: exponential decline to expert.
    Each cycle: eta jumps then decays back within the cycle.
    Meta-learning: successive shift peaks decrease (faster adaptation).
    
    Args:
        n_sessions_baseline: Sessions before first shift
        n_sessions_per_cycle: Sessions per distribution epoch in cycling
        n_cycles: Number of shift cycles
        eta_expert: Baseline expert learning rate
        eta_first_shift: eta peak at first distribution shift
        eta_later_shift: eta peak for later shifts (before meta-learning reduction)
        meta_learning_rate: How fast shift peaks decrease across cycles
        shift_decay_rate: Within-cycle decay back to expert
        sigma_percep, A_repulsion, eta_relax: Constant parameters
        eta_baseline_start: Starting eta for baseline phase
        baseline_decay_rate: Decay during baseline training
    
    Returns:
        Dict suitable for generate_synthetic_animal(true_params=...)
    """
    n_total = n_sessions_baseline + n_cycles * n_sessions_per_cycle
    eta = np.empty(n_total)
    
    # Baseline phase
    s = np.arange(n_sessions_baseline)
    eta[:n_sessions_baseline] = (
        (eta_baseline_start - eta_expert) * np.exp(-baseline_decay_rate * s) + eta_expert
    )
    
    # Cycling phase
    offset = n_sessions_baseline
    for c in range(n_cycles):
        # Meta-learning: shift peak decreases with cycle number
        if c == 0:
            shift_peak = eta_first_shift
        else:
            shift_peak = (
                (eta_later_shift - eta_expert)
                * np.exp(-meta_learning_rate * c)
                + eta_expert
            )
        
        s = np.arange(n_sessions_per_cycle)
        eta[offset:offset + n_sessions_per_cycle] = (
            (shift_peak - eta_expert) * np.exp(-shift_decay_rate * s) + eta_expert
        )
        offset += n_sessions_per_cycle
    
    return {
        'sigma_percep': sigma_percep,
        'A_repulsion': A_repulsion,
        'eta_learning': eta.tolist(),
        'eta_relax': eta_relax,
    }


# =============================================================================
# SYNTHETIC ANIMAL GENERATION
# =============================================================================

def generate_synthetic_animal(
    animal_id: str = 'SYN01',
    n_sessions: Optional[int] = None,
    trials_per_session: Union[int, List[int]] = 300,
    true_params: Optional[Dict[str, Union[float, List[float]]]] = None,
    distribution: str = 'Uniform',
    distribution_schedule: Optional[List[Tuple[str, Optional[float]]]] = None,
    distribution_shift_session: Optional[int] = None,
    shift_distribution: str = 'Uniform',
    abort_rate: float = 0.05,
    burn_in: int = 100,
    seed: int = 42,
) -> Tuple[AnimalData, Dict[str, Any]]:
    """
    Generate a complete synthetic animal with BE model choices.
    
    Simulates the full trajectory: generates stimuli, runs the BE model
    across sessions with state chaining, and fills in choices.
    
    Distribution control (in order of precedence):
        1. distribution_schedule: Full per-session list of (dist_name, exp_rate).
           Use make_distribution_schedule() or build manually.
        2. distribution_shift_session + shift_distribution: Single shift point.
        3. distribution: Constant distribution for all sessions.
    
    Parameter trajectories:
        Use preset functions for common patterns:
            param_trajectory_naive_to_expert(n_sessions)
            param_trajectory_full(...)
            param_trajectory_cycling(...)
        Or pass a custom dict with scalar (constant) or list (per-session) values.
    
    Args:
        animal_id: Animal identifier
        n_sessions: Number of sessions. If None, inferred from true_params
                    or distribution_schedule length. Default: 20.
        trials_per_session: Trials per session (int for constant, list for variable)
        true_params: BE parameters per session. Keys are param names, values are
            either a single float (constant across sessions) or a list of floats
            (one per session). If None, uses naive-to-expert default.
        distribution: Base stimulus distribution (used if no schedule provided)
        distribution_schedule: Per-session list of (distribution, exp_rate) tuples.
            Overrides distribution and distribution_shift_session.
        distribution_shift_session: Session index where distribution changes (legacy)
        shift_distribution: Distribution after shift (legacy, used with
            distribution_shift_session)
        abort_rate: Fraction of abort trials
        burn_in: Number of burn-in trials for initial belief
        seed: Random seed
    
    Returns:
        Tuple of (AnimalData, ground_truth_dict) where ground_truth_dict
        contains the true parameters and any other generation metadata.
    
    Examples:
        # Simple: 20 sessions, default declining eta
        animal, gt = generate_synthetic_animal()
        
        # Full trajectory with shift
        params = param_trajectory_full(
            n_sessions_naive=15, n_sessions_expert=5, n_sessions_post_shift=10
        )
        schedule = make_distribution_schedule([
            DistributionEpoch('Uniform', 20),
            DistributionEpoch('Hard-A', 10),
        ])
        animal, gt = generate_synthetic_animal(
            true_params=params,
            distribution_schedule=schedule,
        )
        
        # Cycling design with meta-learning
        params = param_trajectory_cycling(
            n_sessions_baseline=15, n_sessions_per_cycle=5, n_cycles=4
        )
        schedule = make_distribution_schedule([
            DistributionEpoch('Uniform', 15),
            DistributionEpoch('Hard-A', 5),
            DistributionEpoch('Hard-B', 5),
            DistributionEpoch('Hard-A', 5),
            DistributionEpoch('Hard-B', 5),
        ])
        animal, gt = generate_synthetic_animal(
            true_params=params,
            distribution_schedule=schedule,
        )
    """
    # Lazy import to avoid circular dependency
    from Models.BE_core import BEParams, BEState, BEModel
    
    rng = np.random.default_rng(seed)
    
    # --- Determine n_sessions ---
    # Infer from whichever source provides it
    inferred_n = None
    if true_params is not None:
        for val in true_params.values():
            if isinstance(val, (list, np.ndarray)):
                inferred_n = len(val)
                break
    if distribution_schedule is not None:
        sched_n = len(distribution_schedule)
        if inferred_n is not None and inferred_n != sched_n:
            raise ValueError(
                f"true_params implies {inferred_n} sessions but "
                f"distribution_schedule has {sched_n} entries"
            )
        inferred_n = sched_n
    
    if n_sessions is None:
        n_sessions = inferred_n or 20
    elif inferred_n is not None and n_sessions != inferred_n:
        raise ValueError(
            f"n_sessions={n_sessions} but data implies {inferred_n} sessions"
        )
    
    # --- Build distribution schedule ---
    if distribution_schedule is not None:
        if len(distribution_schedule) != n_sessions:
            raise ValueError(
                f"distribution_schedule has {len(distribution_schedule)} entries "
                f"but n_sessions={n_sessions}"
            )
        dist_schedule = distribution_schedule
    elif distribution_shift_session is not None:
        dist_schedule = []
        for s in range(n_sessions):
            if s < distribution_shift_session:
                dist_schedule.append((distribution, None))
            else:
                dist_schedule.append((shift_distribution, None))
    else:
        dist_schedule = [(distribution, None)] * n_sessions
    
    # --- Handle trials_per_session ---
    if isinstance(trials_per_session, int):
        tps = [trials_per_session] * n_sessions
    else:
        assert len(trials_per_session) == n_sessions
        tps = trials_per_session
    
    # --- Default params: learning rate trajectory (high -> low) ---
    if true_params is None:
        true_params = param_trajectory_naive_to_expert(n_sessions)
    
    # Expand constant params to per-session
    params_per_session = {}
    for key, val in true_params.items():
        if isinstance(val, (list, np.ndarray)):
            assert len(val) == n_sessions, (
                f"{key}: expected {n_sessions} values, got {len(val)}"
            )
            params_per_session[key] = list(val)
        else:
            params_per_session[key] = [val] * n_sessions
    
    # --- Generate sessions ---
    sessions = []
    
    # Initial state with burn-in
    state = BEState.initial_uniform()
    if burn_in > 0:
        burn_stim = rng.uniform(-1.0, 1.0, burn_in)
        burn_cats = stimulus_to_category(burn_stim)
        burn_params = BEParams(
            sigma_percep=params_per_session['sigma_percep'][0],
            A_repulsion=params_per_session['A_repulsion'][0],
            eta_learning=params_per_session['eta_learning'][0],
            eta_relax=params_per_session['eta_relax'][0],
        )
        _, _, state, _ = BEModel.simulate_session(
            burn_params, state, burn_stim, burn_cats, rng
        )
    
    for s_idx in range(n_sessions):
        sess_dist, sess_exp_rate = dist_schedule[s_idx]
        
        # Create session
        session = generate_synthetic_session(
            session_idx=s_idx,
            n_trials=tps[s_idx],
            distribution=sess_dist,
            exp_rate=sess_exp_rate,
            abort_rate=abort_rate,
            rng=rng,
        )
        session.metadata.animal_id = animal_id
        session.session_id = f'{animal_id}_S{s_idx:03d}'
        
        # Build BE params for this session
        params = BEParams(
            sigma_percep=params_per_session['sigma_percep'][s_idx],
            A_repulsion=params_per_session['A_repulsion'][s_idx],
            eta_learning=params_per_session['eta_learning'][s_idx],
            eta_relax=params_per_session['eta_relax'][s_idx],
        )
        
        # Get stimuli and categories
        stimuli = session.trials.stimulus
        categories = session.trials.category
        no_response = session.trials.abort
        
        # Simulate choices using BE model
        choices_01, p_B, state, _ = BEModel.simulate_session(
            params, state, stimuli, categories, rng,
            no_response=no_response,
        )
        
        # Fill in trial data
        session.trials.choice_category = choices_01
        
        # Convert back to spatial for consistency
        contingency = session.metadata.sound_contingency
        choice_spatial = np.zeros(len(choices_01), dtype=float)
        if contingency == 'Low_Left_High_Right':
            choice_spatial[choices_01 == 0] = -1.0
            choice_spatial[choices_01 == 1] = 1.0
        elif contingency == 'Low_Right_High_Left':
            choice_spatial[choices_01 == 0] = 1.0
            choice_spatial[choices_01 == 1] = -1.0
        choice_spatial[np.isnan(choices_01)] = 0.0
        session.trials.choice_spatial = choice_spatial
        
        # Fill correct/outcome
        session.trials.correct = (choices_01 == categories)
        session.trials.correct[np.isnan(choices_01)] = False
        outcomes = np.where(
            no_response, 'Abort',
            np.where(session.trials.correct, 'Correct', 'Incorrect')
        )
        session.trials.outcome = outcomes
        
        sessions.append(session)
    
    animal = AnimalData(
        animal_id=animal_id,
        sessions=sessions,
    )
    
    # Ground truth for recovery testing
    ground_truth = {
        'params_per_session': params_per_session,
        'distribution_schedule': dist_schedule,
        'burn_in': burn_in,
        'seed': seed,
        'n_sessions': n_sessions,
        'trials_per_session': tps,
    }
    
    return animal, ground_truth
