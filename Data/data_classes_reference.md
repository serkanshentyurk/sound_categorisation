# Data Class Reference

## Overview

The data pipeline is built on a hierarchy of dataclasses that mirror the experimental structure: an experiment contains animals, an animal contains sessions, a session contains trials.

```
ExperimentData                        top-level: all animals
  └── AnimalData                      one animal, all its sessions
        └── SessionData               one behavioural session
              ├── SessionMetadata      task parameters (constant within session)
              ├── BlockInfo[]          distribution block(s) within session
              └── TrialData            trial-by-trial arrays

FittingData                           model-ready extraction (from AnimalData)
  └── session_arrays[]                list of dicts with filtered arrays
```

The classes separate **experimental data** (what happened in the rig) from **model data** (what gets fed to the BE model). The bridge between them is `TrialData.get_model_arrays()` at the session level, or `AnimalData.get_fitting_data()` for the whole animal.

---

## ExperimentData

**What it is**: Top-level container. A dictionary of animals keyed by ID, plus optional metadata.

**When you use it**: Loading data, batch operations, getting a summary table of all animals.

| Field | Type | Description |
|---|---|---|
| `animals` | Dict[str, AnimalData] | Animals keyed by animal_id |
| `metadata` | Dict[str, Any] | Optional experiment-level info |

| Property | Returns |
|---|---|
| `.animal_ids` | List of animal ID strings |
| `.n_animals` | Number of animals |

| Method | What it does |
|---|---|
| `.get_animal(id)` | Retrieve one AnimalData by ID |
| `.get_animals_with_min_sessions(n, stage)` | Filter to animals with ≥n sessions of a given stage |
| `.summary()` | DataFrame: one row per animal with session counts, stages, date range |
| `.save(path)` / `.load(path)` | Pickle serialisation |

**Typical usage**:
```python
experiment = ExperimentData.load('experiment.pkl')
good = experiment.get_animals_with_min_sessions(10, stage='Full_Task_Cont')
```

---

## AnimalData

**What it is**: All data for one animal. Sessions are stored in chronological order. This is the unit of model fitting — you fit one animal's learning trajectory at a time.

| Field | Type | Description |
|---|---|---|
| `animal_id` | str | e.g. 'SS01' |
| `sessions` | List[SessionData] | Chronological list of all sessions |
| `metadata` | Dict[str, Any] | Optional animal-level info |

| Property | Returns |
|---|---|
| `.n_sessions` | Total session count (all stages) |
| `.stages` | Set of unique stage names across sessions |

| Method | What it does |
|---|---|
| `.get_sessions(stage, distribution, idx_range, date_range)` | Filter sessions by any combination of criteria |
| `.get_fitting_data(stage, exclude_abort, exclude_opto, ...)` | Extract a FittingData object for the BE model pipeline |
| `.summary()` | DataFrame: one row per session with n_trials, stage, distribution, date |
| `.save(path)` / `.load(path)` | Pickle serialisation |

**Typical usage**:
```python
animal = experiment.get_animal('SS01')
task_sessions = animal.get_sessions(stage='Full_Task_Cont')
fitting = animal.get_fitting_data(stage='Full_Task_Cont')
```

**Key point**: `get_sessions()` returns raw SessionData objects for exploration. `get_fitting_data()` returns a FittingData object with trials already filtered (aborts removed, opto removed, etc.) — ready to go into the model.

---

## SessionData

**What it is**: One behavioural session. Contains three things: metadata (task parameters), block structure (distribution info), and trial data (what actually happened).

| Field | Type | Description |
|---|---|---|
| `session_id` | str | Unique identifier (usually `{animal}_{date}`) |
| `session_idx` | int | Ordinal position within the animal's timeline |
| `date` | datetime.date | When the session was run |
| `metadata` | SessionMetadata | Task parameters |
| `blocks` | List[BlockInfo] | Distribution block(s) |
| `trials` | TrialData | Trial-by-trial arrays |
| `csv_path` | Optional[str] | Path to source CSV |

| Property | Returns |
|---|---|
| `.stage` | Shortcut for `metadata.stage` (e.g. 'Full_Task_Cont') |
| `.distribution` | Shortcut for `blocks[0].distribution` (e.g. 'Uniform') |
| `.n_trials` | Total trials in session |
| `.n_blocks` | Number of distribution blocks |
| `.days_since_first` | Calendar days from animal's first session |

**Note**: Currently there's one block per session (single distribution). The block structure exists so it can handle within-session distribution switches later (the cycling design in Aim 2).

---

## SessionMetadata

**What it is**: Task parameters that are constant within a session. Extracted from the first row of the CSV — these columns don't change trial-to-trial, they're set in the Bonsai protocol.

| Field | Type | Description |
|---|---|---|
| `animal_id` | str | Animal ID |
| `protocol` | str | Bonsai protocol name |
| `stage` | str | Training stage (e.g. 'Full_Task_Cont', 'Habituation') |
| `sound_contingency` | str | Which side maps to which category |
| `stim_type` | Optional[str] | Stimulus type |
| `stim_range_min` | float | Minimum stimulus value (usually -1.0) |
| `stim_range_max` | float | Maximum stimulus value (usually 1.0) |
| `anti_bias` | Optional | Anti-bias correction setting |
| `nb_of_stim` | Optional[int] | Number of discrete stimulus levels (if discrete) |
| `response_window` | Optional[float] | Time allowed for response (seconds) |
| `timeout_duration` | Optional[float] | Timeout penalty duration (seconds) |
| `sound_duration` | Optional[float] | Stimulus sound duration (seconds) |
| `go_cue_duration` | Optional[float] | Go cue duration (seconds) |
| `iti` | Optional[float] | Inter-trial interval (seconds) |
| `left_valve_time` | Optional[float] | Left reward valve open time (seconds) |
| `right_valve_time` | Optional[float] | Right reward valve open time (seconds) |
| `working_memory_delay` | Optional[float] | Delay period (seconds, 0 in standard task) |
| `window_perf_size` | Optional[int] | Rolling performance window size |
| `min_block_length` | Optional[int] | Minimum trials per distribution block |
| `block_performance_threshold` | Optional[float] | Performance threshold for block transitions |
| `air_puff_contingency` | Optional | Air puff settings |
| `air_puff_side` | Optional[str] | Air puff side |
| `air_puff_contingency_rule` | Optional[str] | Air puff rule |

**Note**: Timing fields come from the CSV as Bonsai TimeSpan strings (e.g. `'00:00:10'` for 10 seconds). The loader parses these to float seconds automatically.

---

## BlockInfo

**What it is**: One distribution epoch within a session. Defines what stimulus distribution was active for a range of trials.

| Field | Type | Description |
|---|---|---|
| `block_idx` | int | Block number within session (0-based) |
| `distribution` | str | 'Uniform', 'Hard-A', or 'Hard-B' |
| `exp_rate` | Optional[float] | λ of the exponential shaping the asymmetric distribution. Only meaningful for Hard-A/Hard-B; controls how steeply stimuli concentrate near the boundary. |
| `trial_start` | int | First trial index (0-based, inclusive) |
| `trial_end` | int | Last trial index (inclusive) |

| Property | Returns |
|---|---|
| `.n_trials` | Number of trials in this block |

---

## TrialData

**What it is**: The actual behavioural data. Every field is a numpy array of length n_trials. This is where stimulus values, choices, outcomes, reaction times, and optogenetic flags live.

### Core behavioural arrays

| Field | dtype | Values | Description |
|---|---|---|---|
| `stimulus` | float | [-1, 1] | Relative stimulus value; boundary at 0 |
| `category` | int | 0 or 1 | True category (0 = A: stim < 0, 1 = B: stim > 0) |
| `choice_spatial` | int | -1, 0, 1 | Raw lick: -1 = left, 1 = right, 0 = no response |
| `choice_category` | float | 0, 1, NaN | Derived: 0 = chose A, 1 = chose B, NaN = no response |
| `correct` | bool | True/False | Whether choice matched category |
| `outcome` | int | varies | Trial outcome code from Bonsai |
| `abort` | bool | True/False | Animal broke fixation |
| `reaction_time` | float | ms | Response latency |
| `trial_number` | int | 1, 2, ... | Original trial number from CSV |
| `block_idx` | int | 0, 1, ... | Which BlockInfo this trial belongs to |

### Rolling performance (from Bonsai, computed online during session)

| Field | dtype | Description |
|---|---|---|
| `rolling_perf` | float | Rolling accuracy over recent trials |
| `rolling_bias` | float | Rolling side bias |
| `rolling_abort_rate` | float | Rolling abort rate |

### Optogenetic fields

| Field | dtype | Description |
|---|---|---|
| `opto_on` | bool | Whether opto was active on this trial |
| `opto_mask` | int/float | Opto mask pattern |
| `opto_fiber` | int/float | Which fibre was used |
| `opto_perc_trials` | float | Percentage of trials with opto in this block |
| `opto_onset_1` | float | First pulse onset time |
| `opto_offset_1` | float | First pulse offset time |
| `opto_onset_2` | float | Second pulse onset (if applicable) |
| `opto_offset_2` | float | Second pulse offset |
| `opto_duration` | float | Total opto duration |
| `opto_zapit` | bool/int | Whether Zapit system was used |

### Other

| Field | dtype | Description |
|---|---|---|
| `extra` | Dict[str, ndarray] | Any CSV columns not captured above |

### Derived properties

| Property | Returns |
|---|---|
| `.n_trials` | Total trial count |
| `.no_response` | Boolean mask: True where `choice_category` is NaN |
| `.valid_mask` | Boolean mask: `~abort & ~no_response` |

### Methods

**`get_model_arrays(exclude_abort, exclude_opto, exclude_no_response)`**

This is the main extraction method. Filters trials and returns a dict of arrays ready for the BE model or summary stats computation:

```python
arrays = sess.trials.get_model_arrays(exclude_abort=True, exclude_opto=True)
```

Returns a dict with:

| Key | Description |
|---|---|
| `'stimuli'` | Stimulus values (filtered) |
| `'categories'` | True categories 0/1 (filtered) |
| `'choices'` | Choices 0/1/NaN (filtered; NaN = no response kept but flagged) |
| `'no_response'` | Boolean mask for NaN choice trials |
| `'not_blockstart'` | Boolean mask (False at first trial and block boundaries) |
| `'reaction_times'` | RT values (filtered) |
| `'trial_indices'` | Original indices for back-mapping to the unfiltered TrialData |

**`to_model_trace(p_B, s_hat, beliefs, x)`** — Wraps model outputs together with the trial data into a `ModelTrace` object for update matrix computation and visualisation.

---

## FittingData

**What it is**: Model-ready extraction from an entire animal's trajectory. Produced by `AnimalData.get_fitting_data()`. This is what goes into the SBI pipeline.

It's essentially a list of the `get_model_arrays()` dicts (one per session) plus a time axis and session identifiers.

| Field | Type | Description |
|---|---|---|
| `animal_id` | str | Source animal |
| `session_arrays` | List[Dict[str, ndarray]] | One dict per session (same format as `get_model_arrays()` output) |
| `time_axis` | ndarray | Session indices or calendar days |
| `time_axis_type` | str | 'session_idx' or 'calendar_days' |
| `session_ids` | List[str] | Session identifiers |
| `session_dates` | List[date] | Session dates |

| Property | Returns |
|---|---|
| `.n_sessions` | Number of sessions |
| `.trials_per_session` | Array of total trial counts per session |
| `.valid_trials_per_session` | Array of valid (non-NaN) trial counts |
| `.stimuli` | List of stimulus arrays (one per session) |
| `.choices` | List of choice arrays (one per session) |
| `.categories` | List of category arrays |
| `.no_response` | List of no-response masks |
| `.not_blockstart` | List of block-start masks |

| Method | Returns |
|---|---|
| `.get_session(idx)` | Dict of arrays for one session |
| `.summary()` | DataFrame with per-session trial counts and dates |

**Filtering applied by `get_fitting_data()`**:

1. Stage filter (e.g. only 'Full_Task_Cont' sessions)
2. Abort trials removed (if `exclude_abort=True`)
3. Opto trials removed (if `exclude_opto=True`)
4. Sessions with fewer than `min_valid_trials` valid trials are dropped

**Not filtered**: no accuracy threshold, no side-bias exclusion, no quality gating beyond trial count. Early naive sessions with chance-level performance are deliberately kept — that's where the high-η signal lives.

---

## Data flow

```
CSV files (one per session)
    │
    ▼
load_session_csv()          → SessionData (with metadata, blocks, trials)
    │
    ▼
build_animal_from_csvs()    → AnimalData (chronological list of SessionData)
    │
    ├──► animal.get_sessions()       → List[SessionData]   (for exploration)
    │
    ├──► animal.get_fitting_data()   → FittingData          (for modelling)
    │         │
    │         ▼
    │    fitting.stimuli, .choices    → into BE model / SBI pipeline
    │
    └──► experiment.add_animal()     → ExperimentData       (for batch ops)
```

The separation between SessionData (raw) and FittingData (filtered) means you never accidentally model abort trials or opto-on trials, but you can always go back to the raw data for exploration.
