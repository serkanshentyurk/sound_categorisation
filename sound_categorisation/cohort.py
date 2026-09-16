"""
Cohort assembly: load the experiment, read genotypes, collect per-animal session sets.

    experiment = load_experiment_any()                  # snapshot if present, else CSV
    by_animal, groups = gather_genotypes(experiment)    # {'SS15': 'het'}, {'het': [...]}
    sets = collect_sessions_ppc(animal, 'Uniform')      # {'opto': [...], 'masking': [...]}

Torch-free. Session selection is ``behav_utils.select_sessions`` on the session
types defined in ``config.yaml``; this module only names the sets the project
compares.
"""

from __future__ import annotations

import pickle
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple

from behav_utils.data.loading import load_experiment
from behav_utils.data.ops.selection import list_presets, register_presets_from_config, select_sessions
from behav_utils.data.synthetic import session_from_arrays

from sound_categorisation.paths import REPO_ROOT, cohort_path, load_project_config

__all__ = ['load_experiment_any', 'gather_genotypes', 'collect_sessions_ppc',
           'collect_sessions_alm', 'ensure_presets', 'SITE_TYPE', 'GENOTYPES', 'CONTROL_TYPES',
           'AnimalRecord', 'load_animals']

GENOTYPES = ('wt', 'het')
SITE_TYPE = {'uni': 'alm_control_uni', 'bi': 'alm_control_bi'}
CONTROL_TYPES = ('masking', 'washout', 'alm_control_uni', 'alm_control_bi')


def ensure_presets(config_path: Path | None = None) -> None:
    """Register the project's session presets from config.yaml if they aren't yet.

    Presets are project vocabulary (``expert_uniform`` …) and live in the config;
    loading an experiment registers them, but code paths that build data in
    memory (selftest, tests, synthetic cohorts) need this call.
    """
    if 'expert_uniform' in list_presets():
        return
    import yaml
    path = Path(config_path) if config_path else REPO_ROOT / 'config.yaml'
    if path.exists():
        register_presets_from_config(yaml.safe_load(path.read_text()))


def load_experiment_any(config_path: Path | None = None, snapshot_path: Path | None = None):
    """Load the experiment: the snapshot if it exists, else the CSV loader via config.

    ``snapshot_path`` defaults to ``snapshot_dir(repo)/sound_cat_snapshot.pkl``
    (cluster path on Linux, ``<repo>/../../data/behaviour/snapshots/`` locally),
    so a normal call needs no argument. Raises a clear error naming the snapshot
    path tried if neither source is available.
    """
    config_path = Path(config_path) if config_path else REPO_ROOT / 'config.yaml'
    if snapshot_path is None:
        try:
            from sound_categorisation.snapshot import SNAPSHOT_FILENAME, snapshot_dir
            snapshot_path = snapshot_dir(REPO_ROOT) / SNAPSHOT_FILENAME
        except Exception:
            snapshot_path = None
    snapshot_path = Path(snapshot_path) if snapshot_path else None
    if snapshot_path and snapshot_path.exists():
        from sound_categorisation.snapshot import load_snapshot
        experiment, _ = load_snapshot(
            snapshot_path, config_path=config_path if config_path.exists() else None)
        print(f'loaded snapshot: {snapshot_path}')
        return experiment
    if snapshot_path is not None:
        warnings.warn(f'no snapshot at {snapshot_path}; trying CSV (needs raw data mounted)', stacklevel=2)
    if config_path.exists():
        try:
            return load_experiment(config_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f'{exc}\n\nNo snapshot found either (looked for {snapshot_path}). '
                f'Pass snapshot_path=/path/to/sound_cat_snapshot.pkl, or mount the raw '
                f'data dir the config points to.') from None
    raise FileNotFoundError(f'no snapshot at {snapshot_path!r} and no config at {config_path}')


def gather_genotypes(experiment) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """``({animal_id: genotype}, {genotype: [animal_id, ...]})`` from ``animal.genotype``.

    Genotype comes from ``animal_metadata.json`` — the single source of truth.
    Missing -> 'unknown' (warned).
    """
    by_animal = {aid: str(getattr(a, 'genotype', 'unknown') or 'unknown').lower()
                 for aid, a in experiment.animals.items()}
    unknown = sorted(a for a, g in by_animal.items() if g in ('unknown', 'none', ''))
    if unknown:
        warnings.warn(f"{len(unknown)} animal(s) without genotype: {', '.join(unknown)}", stacklevel=2)
    groups: Dict[str, List[str]] = {}
    for aid, g in by_animal.items():
        groups.setdefault(g, []).append(aid)
    for g in groups:
        groups[g].sort()
    return by_animal, groups


def collect_sessions_ppc(animal, distribution: str) -> Dict[str, list]:
    """PPC design: ``{'opto': [...], 'masking': [...]}`` at one distribution."""
    return {
        'opto': select_sessions(animal, distribution=distribution, session_type='opto'),
        'masking': select_sessions(animal, distribution=distribution, session_type='masking'),
    }


def collect_sessions_alm(animal, distribution: str, site: str) -> Dict[str, list]:
    """ALM design: ``{'alm': [...], 'masking': [...], 'opto': [...]}``; ``site`` in SITE_TYPE."""
    return {
        'alm': select_sessions(animal, distribution=distribution, session_type=SITE_TYPE[site]),
        'masking': select_sessions(animal, distribution=distribution, session_type='masking'),
        'opto': select_sessions(animal, distribution=distribution, session_type='opto'),
    }


# ── model-identification records (real or synthetic) ───────────────────────
# One AnimalRecord shape for both sources, so the GS and SBI runners are identical downstream.
# Synthetic: cohort pickle -> per-session arrays -> SessionData, with ground truth populated.
# Real: ExperimentData -> preset session selection per animal, truth None.

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
