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

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from behav_utils.data.loading import load_experiment
from behav_utils.data.ops.selection import select_sessions

from sound_categorisation.paths import REPO_ROOT

__all__ = ['load_experiment_any', 'gather_genotypes', 'collect_sessions_ppc',
           'collect_sessions_alm', 'SITE_TYPE', 'GENOTYPES']

GENOTYPES = ('wt', 'het')
SITE_TYPE = {'uni': 'alm_control_uni', 'bi': 'alm_control_bi'}


def load_experiment_any(config_path: Optional[Path] = None, snapshot_path: Optional[Path] = None):
    """Load the experiment: the snapshot if it exists, else the CSV loader via config.

    ``snapshot_path`` defaults to ``snapshot_dir(repo)/sound_cat_snapshot.pkl``
    (cluster path on Linux, ``<repo>/../../data/behaviour/snapshots/`` locally),
    so a normal call needs no argument. Raises a clear error naming the snapshot
    path tried if neither source is available.
    """
    config_path = Path(config_path) if config_path else REPO_ROOT / 'config.yaml'
    if snapshot_path is None:
        try:
            from sound_categorisation.snapshot import snapshot_dir, SNAPSHOT_FILENAME
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
        warnings.warn(f'no snapshot at {snapshot_path}; trying CSV (needs raw data mounted)')
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
        warnings.warn(f"{len(unknown)} animal(s) without genotype: {', '.join(unknown)}")
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
