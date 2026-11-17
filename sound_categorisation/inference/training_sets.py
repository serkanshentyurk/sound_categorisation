"""A network's training set on disk: the parameters and statistics it was trained on.

``sc-sbi-train`` saves one beside each network as soon as its simulations are done, before the
density estimator trains (``AmortisedSBI.train(training_set=...)``; the path is
``data.paths.snpe_training_set_path``):

    <data root>/snpe_networks/snpe_<rep>_<model>_<distribution>_training.npz

Two uses. A restart: training has no other checkpoint, so a run that stops after simulating (a time
limit, a crash, a laptop that sleeps) would lose its simulations; run again with the same settings,
it reads them from here and goes straight to the density estimator. And the networks' own training
distribution for later analyses (the model classifier), without simulating again.

A saved set is reused only when nothing that shapes it has changed (``config_changes``): the model,
the prior variant and its bounds, the accuracy rule, the schedule, N, T, burn-in, representation,
statistics, design features, the number kept and the seed. Draw ``i`` is simulated with seed
``seed + i`` from one parameter stream (``AmortisedSBI.simulate_training_set``), so a reused set is
the set that simulating again would give with the same code. A change to the simulator code is not
part of the configuration: the network check's fingerprint (``sc-sbi-check``) compares fresh
simulations against what a network was trained on, and the file records the commit it was made at.

The file is plain numpy, read with ``allow_pickle=False``: ``theta`` ``(n, n_params)`` in the
parameters' own units, ``x`` ``(n, n_features)`` as simulated (before median imputation), and
``meta``, a JSON string: the configuration, the column names, draws and passes, the simulation
time, when and where it was made. It is written to a temporary file and renamed, so an interrupted
write never leaves a file behind to reuse.

No torch here: an analysis can read a training set without the fitting stack.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

__all__ = ['TrainingSet', 'save_training_set', 'read_training_set', 'config_changes']

FORMAT = 1          # the file layout; raise it when the layout changes


@dataclass(frozen=True)
class TrainingSet:
    """A saved training set: ``theta`` in the parameters' own units, ``x`` as simulated."""

    theta: np.ndarray
    x: np.ndarray
    meta: Dict[str, Any]

    @property
    def config(self) -> Dict[str, Any]:
        """What shaped the set (``AmortisedSBI.training_config``)."""
        return self.meta['config']

    @property
    def param_names(self) -> List[str]:
        return list(self.meta['param_names'])

    @property
    def feature_names(self) -> List[str]:
        return list(self.meta['feature_names'])

    @property
    def n_drawn(self) -> int:
        """Prior draws simulated to keep ``len(theta)``."""
        return int(self.meta['n_drawn'])

    @property
    def n_passed(self) -> int:
        """Draws that passed the accuracy rule (the last round may pass a few more than were kept)."""
        return int(self.meta['n_passed'])


def _as_stored(value):
    """A value as the JSON in the file stores it (tuples become lists, numpy scalars plain numbers)."""
    return json.loads(json.dumps(value, default=_plain))


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'not storable in a training set file: {type(value).__name__}')


def config_changes(saved: Mapping[str, Any], wanted: Mapping[str, Any]) -> List[str]:
    """The configuration keys whose values differ between a saved set and the one wanted, in sorted
    order; a key present on one side only counts. Values are compared as the file stores them."""
    saved, wanted = _as_stored(dict(saved)), _as_stored(dict(wanted))
    missing = object()
    return sorted(k for k in set(saved) | set(wanted) if saved.get(k, missing) != wanted.get(k, missing))


def save_training_set(path, theta, x, *, config: Mapping[str, Any], param_names, feature_names,
                      n_drawn: int, n_passed: int, simulation_time: float) -> Path:
    """Write a training set to ``path`` (temporary file, then a rename). Returns the path."""
    theta = np.asarray(theta, dtype=float)
    x = np.asarray(x, dtype=float)
    if theta.ndim != 2 or x.ndim != 2 or len(theta) != len(x):
        raise ValueError(f'theta {theta.shape} and x {x.shape} must be 2-D with one row per simulation')
    if theta.shape[1] != len(param_names) or x.shape[1] != len(feature_names):
        raise ValueError(f'theta {theta.shape} / x {x.shape} do not match {len(param_names)} parameters '
                         f'and {len(feature_names)} features')
    from sound_categorisation import __version__
    from sound_categorisation.data.paths import git_state
    meta = {
        'format': FORMAT,
        'config': _as_stored(dict(config)),
        'param_names': list(param_names),
        'feature_names': list(feature_names),
        'n_kept': int(len(theta)),
        'n_drawn': int(n_drawn),
        'n_passed': int(n_passed),
        'simulation_time': float(simulation_time),
        'created_utc': datetime.now(UTC).isoformat(timespec='seconds'),
        'hostname': socket.gethostname(),
        'sound_categorisation': __version__,
        'git': git_state(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    try:
        with open(tmp, 'wb') as f:
            np.savez(f, theta=theta, x=x, meta=np.array(json.dumps(meta)))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_training_set(path) -> TrainingSet:
    """A training set written by ``save_training_set``."""
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data['meta']))
        theta, x = np.array(data['theta']), np.array(data['x'])
    if meta.get('format') != FORMAT:
        raise ValueError(f'{path}: training set format {meta.get("format")!r}, this code reads {FORMAT}')
    return TrainingSet(theta=theta, x=x, meta=meta)
