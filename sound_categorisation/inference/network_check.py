"""Do the saved SBI networks still match the current code and settings? (``sc-sbi-check``)

Each network file is read as saved (``amortised.read_network_file``), without rebuilding the
network, so a mismatch is reported rather than raised. Three groups of rows per network:

    config        stored mode, N, T, burn-in, statistics, design features, simulation count,
                  training seed and distribution against settings (SBI_REPRESENTATIONS,
                  SBI_BURN_IN, SBI_STATS, SBI_DESIGN_FEATURES, BASE_SEED) and the file name
    parameters    stored parameter names against the model's current order, and the prior the
                  network was trained on against the current bounds
    fingerprint   ``n_draws`` fresh prior draws through the CURRENT simulator at the network's
                  own configuration; each feature's median against the median stored at
                  training (``feature_medians``, over every training simulation). z is the
                  difference over the bootstrap SE of the new median; |z| > Z_FLAG flags drift
                  in the model cores, the stimulus sampler or behav_utils' statistics since the
                  network was trained. Medians only: a change that leaves every median in place
                  is left to SBC.

``ok`` is True / False, or None where the file does not record the item. ``sessions_diff``
compares a model-identification run's sessions table with what the current code selects for
the same cohort -- whether the real-data inputs moved since that run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sound_categorisation.inference.amortised import read_network_file, trained_prior_bounds
from sound_categorisation.inference.representation import compute_feature_medians, feature_names
from sound_categorisation.inference.simulator import build_simulator, get_bounds_arrays, get_param_names

Z_FLAG = 4.0                    # |z| above this flags a feature median (~6e-5 two-sided per feature)
FINGERPRINT_DRAWS = 5000
FINGERPRINT_SEED_OFFSET = 500_000   # fingerprint draws use VALIDATION_SEED + this (SBC uses the base)
_NAME = re.compile(r'^snpe_(?P<rep>[a-z]+)_(?P<model>BE|SC)_(?P<distribution>.+)\.pkl$')


def parse_network_name(path) -> Dict[str, str]:
    """``snpe_{rep}_{model}_{distribution}.pkl`` -> {'rep', 'model', 'distribution'}."""
    m = _NAME.match(Path(path).name)
    if not m:
        raise ValueError(f'{Path(path).name}: not a network file name (snpe_<rep>_<BE|SC>_<dist>.pkl)')
    return m.groupdict()


def _norm(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def _fmt(v) -> str:
    v = _norm(v)
    if v is None:
        return 'not recorded'
    if isinstance(v, tuple):
        return '(' + ', '.join(_fmt(x) for x in v) + ')' if len(v) != 1 else f'({_fmt(v[0])},)'
    if isinstance(v, float):
        return f'{v:.6g}'
    return str(v)


def _row(network: str, group: str, check: str, stored, current, ok=None, z=np.nan) -> Dict[str, Any]:
    if ok is None and stored is not None:
        ok = _norm(stored) == _norm(current)
    return {'network': network, 'group': group, 'check': check, 'stored': _fmt(stored),
            'current': _fmt(current), 'z': float(z), 'ok': ok}


def config_rows(data: Dict[str, Any], network: str) -> List[Dict[str, Any]]:
    """Stored training configuration against the current settings."""
    from sound_categorisation.inference.constants import SBI_STATS
    from sound_categorisation.settings import (
        BASE_SEED,
        SBI_BURN_IN,
        SBI_DESIGN_FEATURES,
        SBI_REPRESENTATIONS,
    )
    name = parse_network_name(network)
    spec = SBI_REPRESENTATIONS[name['rep']]
    meta = data.get('training_metadata') or {}
    design = data.get('design_features')
    return [
        _row(network, 'config', 'model', str(data.get('model', '')).upper(), name['model']),
        _row(network, 'config', 'distribution', data.get('dist_schedule'), name['distribution']),
        _row(network, 'config', 'mode', data.get('mode'), spec['mode']),
        _row(network, 'config', 'N (sessions)', data.get('N'), spec['N']),
        _row(network, 'config', 'T (trials per session)', data.get('T'), spec['T']),
        _row(network, 'config', 'burn_in', data.get('burn_in'), SBI_BURN_IN),
        _row(network, 'config', 'stat_names', data.get('stat_names'), list(SBI_STATS)),
        # a network saved before design features existed was trained without them: () not None
        _row(network, 'config', 'design_features', list(design or ()), list(SBI_DESIGN_FEATURES)),
        _row(network, 'config', 'n_simulations', meta.get('n_simulations'), spec['n_simulations']),
        _row(network, 'config', 'training seed', meta.get('seed'), BASE_SEED),
    ]


def parameter_rows(data: Dict[str, Any], network: str) -> List[Dict[str, Any]]:
    """Stored parameter order and trained prior bounds against the model's current ones."""
    model = parse_network_name(network)['model']
    names = get_param_names(model)
    rows = [_row(network, 'parameters', 'param_names', data.get('param_names'), names)]
    lo, hi = get_bounds_arrays(model)
    bounds = trained_prior_bounds(data)
    if bounds is None:
        rows.append(_row(network, 'parameters', 'prior bounds', None, 'current bounds'))
        return rows
    lo_s, hi_s = bounds
    for j, n in enumerate(names):
        stored = (float(lo_s[j]), float(hi_s[j])) if j < len(lo_s) else None
        ok = None if stored is None else bool(np.allclose(stored, (lo[j], hi[j]), atol=1e-6))
        rows.append(_row(network, 'parameters', f'prior bounds {n}', stored,
                         (float(lo[j]), float(hi[j])), ok=ok))
    return rows


def simulate_features(data: Dict[str, Any], n_draws: int = FINGERPRINT_DRAWS,
                      seed: int | None = None, n_jobs: int = 1) -> np.ndarray:
    """(n_draws, n_features) observations from the CURRENT simulator at a saved network's own
    configuration, for prior draws within the bounds it was trained on (current bounds if the
    file does not record them). Draw i is simulated with seed ``seed + 1 + i``; the default seed
    is VALIDATION_SEED + FINGERPRINT_SEED_OFFSET, apart from the SBC draws and every training seed."""
    from sound_categorisation.settings import VALIDATION_SEED
    seed = VALIDATION_SEED + FINGERPRINT_SEED_OFFSET if seed is None else int(seed)
    model = str(data['model']).lower()
    sim_fn, _, names = build_simulator(
        model, dist_schedule=data['dist_schedule'], N=data['N'], T=data['T'],
        burn_in=int(data['burn_in']), mode=data['mode'], stat_names=data['stat_names'],
        design_features=tuple(data.get('design_features') or ()))
    bounds = trained_prior_bounds(data) or get_bounds_arrays(model)
    lo, hi = (np.asarray(b, dtype=float) for b in bounds)
    thetas = np.random.default_rng(seed).uniform(lo, hi, size=(n_draws, len(names)))
    if n_jobs == 1:
        xs = [sim_fn(thetas[i], seed=seed + 1 + i) for i in range(n_draws)]
    else:
        from joblib import Parallel, delayed
        xs = Parallel(n_jobs=n_jobs, batch_size=16)(
            delayed(sim_fn)(thetas[i], seed=seed + 1 + i) for i in range(n_draws))
    return np.vstack(xs).astype(float)


def fingerprint_rows(data: Dict[str, Any], network: str, n_draws: int = FINGERPRINT_DRAWS,
                     seed: int | None = None, n_jobs: int = 1, n_boot: int = 200,
                     X: np.ndarray | None = None) -> List[Dict[str, Any]]:
    """Per-feature median of fresh simulations against the median stored at training."""
    stored = data.get('feature_medians')
    feats = feature_names(data['mode'], data['stat_names'], data.get('design_features') or ())
    if stored is None:
        return [_row(network, 'fingerprint', 'feature medians', None, 'not compared')]
    stored = np.asarray(stored, dtype=float)
    if len(stored) != len(feats):
        return [_row(network, 'fingerprint', 'feature count', len(stored), len(feats))]
    if X is None:
        X = simulate_features(data, n_draws=n_draws, seed=seed, n_jobs=n_jobs)
    current = compute_feature_medians(X)
    rng = np.random.default_rng(0)
    boot = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        boot[b] = np.nanmedian(np.where(np.isfinite(X), X, np.nan)[rng.integers(0, len(X), len(X))],
                               axis=0)
    se = np.nanstd(boot, axis=0, ddof=1)
    rows = []
    for j, f in enumerate(feats):
        diff = current[j] - stored[j]
        if se[j] > 0:
            z = diff / se[j]
        else:
            z = 0.0 if np.isclose(diff, 0.0) else np.inf
        rows.append(_row(network, 'fingerprint', f'median {f}', float(stored[j]),
                         float(current[j]), ok=bool(abs(z) <= Z_FLAG), z=z))
    return rows


def check_network(path, n_draws: int = FINGERPRINT_DRAWS, seed: int | None = None,
                  n_jobs: int = 1) -> pd.DataFrame:
    """Every check for one network file, as tidy rows."""
    path = Path(path)
    data = read_network_file(path)
    rows = config_rows(data, path.name) + parameter_rows(data, path.name)
    rows += fingerprint_rows(data, path.name, n_draws=n_draws, seed=seed, n_jobs=n_jobs)
    return pd.DataFrame(rows)


# =============================================================================
# REAL-DATA INPUTS: did the selected sessions move since a run?
# =============================================================================

def _dates(series: pd.Series) -> List[str]:
    """Session dates as comparable strings, whether they came from a CSV or from SessionData."""
    parsed = pd.to_datetime(series, errors='coerce')
    return [p.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(p) else str(raw)
            for p, raw in zip(parsed, series)]


def _session_key(table: pd.DataFrame) -> Dict[str, Any]:
    out = {}
    for aid, g in table.groupby('animal_id'):
        g = g[g['n_valid'].fillna(0) > 0]
        sessions = sorted(zip(_dates(g['date']), g['n_valid'].astype(int)))
        out[aid] = {'kept': bool(table.loc[table['animal_id'] == aid, 'kept'].iloc[0]),
                    'sessions': sessions}
    return out


def sessions_diff(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Per animal: kept and selected sessions (date, valid trials) in a run's sessions table
    (``old``) against a fresh selection by the current code (``new``). ``same`` is True when
    both the kept flag and the session list match."""
    a, b = _session_key(old), _session_key(new)
    rows = []
    for aid in sorted(set(a) | set(b)):
        o, n = a.get(aid), b.get(aid)
        same = o is not None and n is not None and o == n
        detail = ''
        if o is None or n is None:
            detail = 'only in the run' if n is None else 'only in the current selection'
        elif not same:
            gone = sorted(set(o['sessions']) - set(n['sessions']))
            added = sorted(set(n['sessions']) - set(o['sessions']))
            parts = []
            if o['kept'] != n['kept']:
                parts.append(f"kept {o['kept']} -> {n['kept']}")
            if gone:
                parts.append('dropped ' + ', '.join(f'{d} ({v})' for d, v in gone))
            if added:
                parts.append('added ' + ', '.join(f'{d} ({v})' for d, v in added))
            detail = '; '.join(parts)
        rows.append({'animal_id': aid,
                     'kept_run': None if o is None else o['kept'],
                     'kept_now': None if n is None else n['kept'],
                     'n_sessions_run': None if o is None else len(o['sessions']),
                     'n_sessions_now': None if n is None else len(n['sessions']),
                     'same': same, 'detail': detail})
    return pd.DataFrame(rows)
