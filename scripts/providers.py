"""
Source-agnostic animal provider for the model-identification runners.

Yields a uniform AnimalRecord for both synthetic (cohort) and real
(ExperimentData) data, so the GS and SBI runners are identical downstream.
The only real-vs-synthetic difference lives here, at the data-loading boundary:

  - synthetic: cohort pickle -> reconstruct per-session arrays -> SessionData,
               with ground-truth model/params populated.
  - real:      ExperimentData (via the project config) -> expert-uniform session
               selection per animal, with truth set to None.
"""

import pickle
from collections import namedtuple

from scripts.config import cohort_path, load_project_config
from behav_utils.data.synthetic import session_from_arrays
from behav_utils.data.loading import load_experiment
from behav_utils.data.ops.selection import select_sessions

# Uniform record consumed by run_gs / run_sbi. sessions are SessionData objects;
# true_model / true_params are None for real animals.
AnimalRecord = namedtuple('AnimalRecord',
                          ['animal_id', 'sessions', 'true_model', 'true_params'])


def _synthetic_records(cohort):
    with open(cohort_path(cohort), 'rb') as f:
        data = pickle.load(f)
    records = []
    for a in data['animals']:
        sessions = [
            session_from_arrays(
                s['stimuli'], s['choices'], s['categories'],
                animal_id=a['animal_id'],
            )
            for s in a['sessions']
        ]
        records.append(AnimalRecord(
            a['animal_id'], sessions, a['true_model'], a['true_params'],
        ))
    return records


def _real_records(config_path=None, preset='expert_uniform', experiment=None):
    # config_path None -> load_project_config picks cluster vs local config.
    if experiment is None:
        experiment = load_experiment(load_project_config(config_path))
    records = []
    for aid in experiment.animal_ids:
        sessions = select_sessions(experiment.get_animal(aid), preset)
        records.append(AnimalRecord(aid, sessions, None, None))
    return records


def load_animals(source, cohort=None, config_path=None,
                 preset='expert_uniform', experiment=None):
    """Return a list of AnimalRecord for the given source.

    Args:
        source: 'synthetic' or 'real'.
        cohort: cohort name (required for 'synthetic').
        config_path: optional config.yaml path (real; default = project config).
        preset: session-selection preset for real data (default 'expert_uniform').
        experiment: pre-loaded ExperimentData (real; skips loading if supplied).

    Returns:
        List[AnimalRecord]. true_model/true_params are populated for synthetic
        animals and None for real ones.
    """
    if source == 'synthetic':
        if not cohort:
            raise ValueError("source='synthetic' requires a cohort name")
        return _synthetic_records(cohort)
    if source == 'real':
        return _real_records(
            config_path=config_path, preset=preset, experiment=experiment,
        )
    raise ValueError(f"Unknown source '{source}' (use 'synthetic' or 'real')")
