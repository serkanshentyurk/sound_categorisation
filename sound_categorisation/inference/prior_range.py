"""Where each animal's statistics fall in what the models simulate (``sc-sbi-range``).

An SBI network can only be trusted for data that look like its training simulations. For every
representation and model this module draws parameters from the prior, runs them through the
simulator at the training configuration (``settings.SBI_REPRESENTATIONS``: session count, session
length, burn-in, statistics, design features), and places each animal's observation vector in the
resulting distribution. Nothing here needs torch or a trained network: ``sc-sbi-check`` is what
shows that the saved networks were trained with this configuration.

The prior is one of the variants of ``models.priors``: ``learning`` (the one in use) or
``manuscript`` (what the networks before mid-November 2026 were trained on). The two are placed side
by side to see what the wider ranges gain in animals covered and cost in telling the models apart
(``comparison_tables``).

A draw is simulated exactly as the training simulator does it (``simulator.build_simulator``): one
generator seeded per draw gives the session count, then the sessions; a test keeps the two equal.
The training set of a network with the accuracy rule is the passing set here.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from sound_categorisation.inference.representation import feature_names, to_stat_vector
from sound_categorisation.inference.simulator import simulate_sessions
from sound_categorisation.models.priors import PRIOR_VARIANTS, resolve, sample_prior

RANGE_SEED_OFFSET = 2_000_000       # draws use VALIDATION_SEED + this: apart from training, SBC, the
                                    # fingerprint (+500,000) and the synthetic cohorts (+1,000,000)
K_NEAREST = 10
CENTRAL = (0.5, 99.5)               # a value outside these percentiles of a set is flagged
MIN_REFERENCE = 50                  # a set with fewer draws gives a percentile but no outside verdict
QUANTILES = (0.005, 0.025, 0.25, 0.5, 0.75, 0.975, 0.995)

SIMULATED_COLUMNS = ['rep', 'model', 'set', 'feature', 'n', 'n_finite', 'min',
                     *(f'q{q * 100:g}' for q in QUANTILES), 'max']
ANIMAL_COLUMNS = ['animal_id', 'rep', 'model', 'feature', 'value', 'percentile', 'outside',
                  'percentile_passing', 'outside_passing', 'n_prior', 'n_passing']
SUMMARY_COLUMNS = ['animal_id', 'true_model', 'rep', 'model', 'n_features', 'n_missing', 'n_outside',
                   'n_outside_passing', 'worst_feature', 'worst_percentile', 'nearest_percentile',
                   'n_prior', 'n_passing', 'acceptance_rate']
RECOGNISED_COLUMNS = ['rep', 'model', 'n_prior', 'n_passing', 'recognised', 'recognised_passing']
NEAR = 99.0                         # nearest_percentile above this: no simulation near the animal
COMPARISON_COLUMNS = ['prior', 'rep', 'model', 'n_animals', 'n_prior', 'n_passing', 'n_inside', 'n_near',
                      'n_covered', 'n_covered_passing', 'recognised', 'recognised_passing']
N_SPLITS = 5                        # folds of the classifier behind ``recognised``


# =============================================================================
# REFERENCE SIMULATIONS
# =============================================================================

def simulate_draw(model: str, params: Dict[str, float], seed: int, *, distribution: str, N, T,
                  burn_in: int, reps: Dict[str, str], stat_names: Sequence[str],
                  design_features: Sequence[str]) -> Dict[str, Any]:
    """One prior draw (``params``: ``{name: value}``) through the simulator: the observation vector
    of every representation in ``reps`` (``{rep: mode}``, all sharing this ``N`` and ``T``) from the
    same simulated sessions, and the lowest session accuracy.

    The generator is used as ``build_simulator`` uses it (session count first when ``N`` is a
    range, then the sessions), so for the four model parameters ``x[rep]`` equals that simulator's
    output for this seed.
    """
    rng = np.random.default_rng(seed)
    if isinstance(N, (tuple, list)):
        n_sessions = int(rng.integers(int(N[0]), int(N[1]) + 1))
    else:
        n_sessions = int(N)
    sessions = simulate_sessions(model, dict(params), [distribution] * n_sessions, T, burn_in, rng)
    x = {rep: to_stat_vector(sessions, mode=mode, stat_names=stat_names, design_features=design_features)
         for rep, mode in reps.items()}
    return {'x': x, 'min_accuracy': min(float(s.summary().accuracy) for s in sessions)}


def simulate_reference(model: str, *, distribution: str, N, T, burn_in: int, reps: Dict[str, str],
                       stat_names: Sequence[str], design_features: Sequence[str], n_draws: int,
                       seed: int, n_jobs: int = 1, prior: str | None = None) -> Dict[str, Any]:
    """``n_draws`` draws of ``model`` from a prior variant (``models.priors.sample_prior``), each
    through ``simulate_draw`` with seed ``seed + 1 + i``.

    Returns ``{'names': [...], 'theta': (n, p), 'min_accuracy': (n,), 'x': {rep: (n, n_features)}}``.
    """
    model = str(model).lower()
    names, thetas = sample_prior(model, prior, n_draws, np.random.default_rng(seed))
    kwargs = dict(distribution=distribution, N=N, T=T, burn_in=burn_in, reps=dict(reps),
                  stat_names=list(stat_names), design_features=tuple(design_features))
    jobs = [(dict(zip(names, map(float, thetas[i]))), seed + 1 + i) for i in range(n_draws)]
    if n_jobs == 1:
        draws = [simulate_draw(model, params, s, **kwargs) for params, s in jobs]
    else:
        from joblib import Parallel, delayed
        draws = Parallel(n_jobs=n_jobs, batch_size=8)(
            delayed(simulate_draw)(model, params, s, **kwargs) for params, s in jobs)
    return {'names': names, 'theta': thetas,
            'min_accuracy': np.array([d['min_accuracy'] for d in draws], dtype=float),
            'x': {rep: np.vstack([d['x'][rep] for d in draws]).astype(float) for rep in reps}}


def group_designs(representations: Dict[str, Dict[str, Any]], reps: Sequence[str]) -> List[Dict[str, Any]]:
    """The representations in ``reps`` grouped by session count and length, so representations that
    differ only in how the statistics are summarised (pooled, moments) share their simulations.

    Returns ``[{'N': ..., 'T': ..., 'reps': {rep: mode}}]`` in the order of ``reps``.
    """
    groups: Dict[Any, Dict[str, Any]] = {}
    for rep in reps:
        cfg = representations[rep]
        key = (_freeze(cfg['N']), _freeze(cfg['T']))
        groups.setdefault(key, {'N': cfg['N'], 'T': cfg['T'], 'reps': {}})['reps'][rep] = cfg['mode']
    return list(groups.values())


def _freeze(v):
    return tuple(v) if isinstance(v, (tuple, list)) else v


# =============================================================================
# PLACING AN OBSERVATION
# =============================================================================

def percentile_of(value: float, reference: np.ndarray) -> float:
    """Percentile of ``value`` among the finite entries of ``reference`` (ties count half), 0 to 100.
    NaN when the value is not finite or there is no finite reference."""
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if not np.isfinite(value) or len(ref) == 0:
        return float('nan')
    return float(100.0 * ((ref < value).sum() + 0.5 * (ref == value).sum()) / len(ref))


def is_outside(percentile: float, n_reference: int, central: Sequence[float] = CENTRAL):
    """True when a percentile lies outside the central range of its set; False when inside or when the
    value is missing; None when the set has fewer than ``MIN_REFERENCE`` draws, too few to say."""
    if n_reference < MIN_REFERENCE:
        return None
    return bool(np.isfinite(percentile) and (percentile < central[0] or percentile > central[1]))


def _scaled(X: np.ndarray, x: np.ndarray):
    """``X`` and ``x`` with each feature centred on the simulated median and divided by the simulated
    interquartile range; non-finite cells take the median (0 after centring), as the networks impute."""
    Xn = np.where(np.isfinite(X), X, np.nan)
    med = np.nanmedian(Xn, axis=0)
    q75, q25 = np.nanpercentile(Xn, 75, axis=0), np.nanpercentile(Xn, 25, axis=0)
    scale = np.where((q75 - q25) > 0, q75 - q25, 1.0)
    Z = np.nan_to_num((Xn - med) / scale, nan=0.0)
    z = np.nan_to_num((np.where(np.isfinite(x), x, np.nan) - med) / scale, nan=0.0)
    return Z, z


def nearest_percentile(x: np.ndarray, X: np.ndarray, k: int = K_NEAREST) -> float:
    """How far ``x`` is from the simulations ``X``, all features together.

    The distance from ``x`` to its ``k``-th nearest row of ``X`` (scaled features), as a percentile of
    the same distance for every row of ``X`` to its own ``k``-th nearest other row. Near 100: no
    simulation looks like ``x``. NaN with fewer than ``k + 1`` simulations.
    """
    X = np.asarray(X, dtype=float)
    if len(X) < k + 1:
        return float('nan')
    Z, z = _scaled(X, np.asarray(x, dtype=float))
    d_obs = np.sort(np.sqrt(((Z - z) ** 2).sum(axis=1)))[k - 1]
    sq = (Z ** 2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T, 0.0)
    np.fill_diagonal(d2, np.inf)
    d_ref = np.sqrt(np.sort(d2, axis=1)[:, k - 1])
    return percentile_of(float(d_obs), d_ref)


def simulated_rows(rep: str, model: str, X: np.ndarray, passing: np.ndarray,
                   feats: Sequence[str]) -> List[dict]:
    """Quantiles of every feature in the prior set and in the passing set."""
    rows = []
    for name, sub in (('prior', X), ('passing', X[passing])):
        for j, feature in enumerate(feats):
            col = sub[:, j] if len(sub) else np.array([])
            col = col[np.isfinite(col)]
            row = {'rep': rep, 'model': model, 'set': name, 'feature': feature, 'n': int(len(sub)),
                   'n_finite': int(len(col)), 'min': float(col.min()) if len(col) else np.nan,
                   'max': float(col.max()) if len(col) else np.nan}
            for q in QUANTILES:
                row[f'q{q * 100:g}'] = float(np.quantile(col, q)) if len(col) else np.nan
            rows.append(row)
    return rows


def animal_rows(animal_id: str, rep: str, model: str, x: np.ndarray, X: np.ndarray, passing: np.ndarray,
                feats: Sequence[str]) -> List[dict]:
    """One row per feature: the animal's value and where it lies in the prior and passing sets."""
    rows = []
    for j, feature in enumerate(feats):
        p_prior = percentile_of(float(x[j]), X[:, j])
        p_pass = percentile_of(float(x[j]), X[passing, j])
        rows.append({'animal_id': animal_id, 'rep': rep, 'model': model, 'feature': feature,
                     'value': float(x[j]), 'percentile': p_prior, 'outside': is_outside(p_prior, len(X)),
                     'percentile_passing': p_pass, 'outside_passing': is_outside(p_pass, int(passing.sum())),
                     'n_prior': int(len(X)), 'n_passing': int(passing.sum())})
    return rows


def _count(flags) -> float:
    """How many flags are True; NaN when the set was too small to flag anything (all None)."""
    flags = list(flags)
    return float('nan') if all(f is None for f in flags) else float(sum(bool(f) for f in flags))


def summary_row(rows: List[dict], x: np.ndarray, X: np.ndarray, n_stats: int, *, true_model=None,
                acceptance_rate: float = float('nan')) -> dict:
    """One line per animal, representation and model from its feature rows. ``n_stats`` is the number
    of statistic features at the front of ``x`` (the design features after them are left out of the
    nearest-simulation distance: every simulation has its own session count and length)."""
    finite = [r for r in rows if np.isfinite(r['percentile'])]
    worst = max(finite, key=lambda r: abs(r['percentile'] - 50.0)) if finite else None
    return {'animal_id': rows[0]['animal_id'], 'true_model': true_model, 'rep': rows[0]['rep'],
            'model': rows[0]['model'], 'n_features': len(rows),
            'n_missing': sum(not np.isfinite(r['value']) for r in rows),
            'n_outside': _count(r['outside'] for r in rows),
            'n_outside_passing': _count(r['outside_passing'] for r in rows),
            'worst_feature': worst['feature'] if worst else None,
            'worst_percentile': worst['percentile'] if worst else np.nan,
            'nearest_percentile': nearest_percentile(x[:n_stats], X[:, :n_stats]),
            'n_prior': rows[0]['n_prior'], 'n_passing': rows[0]['n_passing'],
            'acceptance_rate': acceptance_rate}


# =============================================================================
# CAN THE MODELS BE TOLD APART UNDER THIS PRIOR
# =============================================================================

def recognised(X_by_model: Dict[str, np.ndarray], seed: int = 0) -> Dict[str, np.ndarray]:
    """For every draw of every model, whether a classifier that never saw it names its model.

    A gradient-boosted classifier on the simulated statistics (missing values allowed), predicting
    each draw from ``N_SPLITS`` folds it was not in. It stands in for what a trained network could
    tell apart from the same statistics; it is not that network.

    Args:
        X_by_model: ``{model: (n_draws, n_features)}`` for two or more models.
        seed: Seed of the folds and the classifier.

    Returns:
        ``{model: (n_draws,) bool}``.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    models = list(X_by_model)
    X = np.vstack([np.where(np.isfinite(X_by_model[m]), X_by_model[m], np.nan) for m in models])
    y = np.concatenate([np.full(len(X_by_model[m]), i) for i, m in enumerate(models)])
    predicted = cross_val_predict(HistGradientBoostingClassifier(random_state=seed), X, y,
                                  cv=StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed))
    return {m: predicted[y == i] == i for i, m in enumerate(models)}


def recognised_rows(rep: str, X_by_model: Dict[str, np.ndarray], passing_by_model: Dict[str, np.ndarray],
                    seed: int = 0) -> List[dict]:
    """One row per model: the share of its draws recognised, over every draw and over its passing
    draws. Empty with fewer than two models or fewer than ``MIN_REFERENCE`` draws of one."""
    if len(X_by_model) < 2 or min(len(X) for X in X_by_model.values()) < MIN_REFERENCE:
        return []
    hits = recognised(X_by_model, seed=seed)
    rows = []
    for model, hit in hits.items():
        passing = passing_by_model[model]
        rows.append({'rep': rep, 'model': model, 'n_prior': int(len(hit)), 'n_passing': int(passing.sum()),
                     'recognised': float(hit.mean()),
                     'recognised_passing': float(hit[passing].mean()) if passing.any() else float('nan')})
    return rows


# =============================================================================
# THE SAVED POSTERIORS' ACCEPTANCE RATES
# =============================================================================

class _Placeholder:
    """Stands in for an object whose module is not installed (see ``_TolerantUnpickler``)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _TolerantUnpickler(pickle.Unpickler):
    """Reads a result saved on the cluster on a machine without torch or sbi. The saved result carries
    network metadata that may hold torch objects; anything whose module cannot be imported becomes a
    placeholder. The acceptance rate itself is a plain number."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ImportError:
            return _Placeholder


def read_acceptance(result_dir, animal_id: str, model: str, prior: str | None = None) -> float:
    """The acceptance rate ``sc-sbi-condition`` saved with an animal's posterior
    (``<result_dir>/<animal>_<MODEL>.pkl``): the share of the network's samples that fell inside
    the prior box. NaN when the run holds no such result, the file cannot be read here, or (with
    ``prior``) the posterior came from a network of another prior variant (a result that records
    none came from a ``manuscript`` network).

    Works without torch: objects from modules that are not installed are skipped while reading.
    """
    path = Path(result_dir) / f'{animal_id}_{str(model).upper()}.pkl'
    if not path.exists():
        return float('nan')
    try:
        with open(path, 'rb') as f:
            result = _TolerantUnpickler(f).load()
        posterior = (result or {}).get('posterior') or {}
        rate = posterior.get('acceptance_rate')
        if prior is not None and (posterior.get('prior') or 'manuscript') != prior:
            return float('nan')
    except Exception:           # an unreadable result must not stop the table; the rate is then missing
        return float('nan')
    return float(rate) if rate is not None else float('nan')


# =============================================================================
# THE TABLES
# =============================================================================

def range_tables(records, *, distribution: str, representations: Dict[str, Dict[str, Any]],
                 reps: Sequence[str], models: Sequence[str], stat_names: Sequence[str],
                 design_features: Sequence[str], burn_in: int, min_accuracy: float, n_draws: int,
                 seed: int, n_jobs: int = 1, result_dirs: Dict[str, Path] | None = None,
                 prior: str | None = None, log=print) -> Dict[str, pd.DataFrame]:
    """The three tables for a cohort's animals under one prior variant (see the module docstring).

    Args:
        records: ``AnimalRecord``s (``cohort.select_animals``): ``animal_id``, ``sessions``, ``true_model``.
        distribution: The phase the sessions and the simulations are on.
        representations: ``settings.SBI_REPRESENTATIONS`` (``{rep: {'N', 'T', 'mode', ...}}``).
        reps: The representations to place the animals in.
        models: Model names (``'BE'``, ``'SC'``).
        stat_names: The statistics of the observation vector (``SBI_STATS``).
        design_features: The design features appended to it (``SBI_DESIGN_FEATURES``).
        burn_in: Model burn-in (``SBI_BURN_IN``).
        min_accuracy: The session filter's minimum accuracy; defines the passing set.
        n_draws: Prior draws per model and session design.
        seed: Seed of the draws.
        n_jobs: joblib workers for the simulations.
        result_dirs: ``{rep: directory}`` of saved conditioning results, for the acceptance rates
            (read only from posteriors of networks trained on ``prior``).
        prior: The prior variant the draws come from (``models.priors.PRIOR_VARIANTS``; None = the
            one in use).
        log: Called with progress lines.

    Returns:
        ``{'simulated': ..., 'animals': ..., 'summary': ..., 'recognised': ...}``.
    """
    stat_names, design_features, prior = list(stat_names), tuple(design_features), resolve(prior)
    simulated, animals, summary, told_apart = [], [], [], []
    for design in group_designs(representations, reps):
        refs, passing = {}, {}
        for model in models:
            name = str(model).upper()
            refs[name] = simulate_reference(model, distribution=distribution, N=design['N'], T=design['T'],
                                            burn_in=burn_in, reps=design['reps'], stat_names=stat_names,
                                            design_features=design_features, n_draws=n_draws, seed=seed,
                                            n_jobs=n_jobs, prior=prior)
            passing[name] = refs[name]['min_accuracy'] >= min_accuracy
            log(f'{prior} {model} N={design["N"]} T={design["T"]}: {n_draws} draws, '
                f'{int(passing[name].sum())} with every session at accuracy >= {min_accuracy:g}')
        for rep, mode in design['reps'].items():
            feats = feature_names(mode, stat_names, design_features)
            n_stats = len(feats) - len(design_features)
            told_apart += recognised_rows(rep, {m: ref['x'][rep][:, :n_stats] for m, ref in refs.items()},
                                          passing, seed=seed)
            for name, ref in refs.items():
                X = ref['x'][rep]
                simulated += simulated_rows(rep, name, X, passing[name], feats)
                for record in records:
                    x = np.asarray(to_stat_vector(record.sessions, mode=mode, stat_names=stat_names,
                                                  design_features=design_features), dtype=float)
                    rows = animal_rows(record.animal_id, rep, name, x, X, passing[name], feats)
                    rate = (read_acceptance(result_dirs[rep], record.animal_id, name, prior=prior)
                            if result_dirs and rep in result_dirs else float('nan'))
                    animals += rows
                    summary.append(summary_row(rows, x, X, n_stats, true_model=record.true_model,
                                               acceptance_rate=rate))
    return {'simulated': pd.DataFrame(simulated, columns=SIMULATED_COLUMNS),
            'animals': pd.DataFrame(animals, columns=ANIMAL_COLUMNS),
            'summary': pd.DataFrame(summary, columns=SUMMARY_COLUMNS),
            'recognised': pd.DataFrame(told_apart, columns=RECOGNISED_COLUMNS)}


# =============================================================================
# PRIOR VARIANTS SIDE BY SIDE
# =============================================================================

def _covered(summary: pd.DataFrame) -> pd.DataFrame:
    """``summary`` with the three per-row verdicts the comparison counts.

    inside            no feature outside the central range of the prior set
    near              a simulation near the animal, all statistics together (nearest_percentile <= NEAR)
    covered           both
    covered_passing   near, and no feature outside the passing set (False when that set is too small
                      to say)
    """
    out = summary.copy()
    out['inside'] = out['n_outside'] == 0
    out['near'] = out['nearest_percentile'] <= NEAR
    out['covered'] = out['inside'] & out['near']
    out['covered_passing'] = out['near'] & (out['n_outside_passing'] == 0)
    return out


def comparison_tables(summaries: Dict[str, pd.DataFrame],
                      recognised_tables: Dict[str, pd.DataFrame] | None = None) -> Dict[str, pd.DataFrame]:
    """The prior variants side by side, from the ``summary`` table of each (``{prior: summary}``)
    and, when given, the ``recognised`` table of each.

    Returns:
        ``comparison``: per prior, representation and model, how many animals are inside the prior
            set feature by feature (``n_inside``), have a simulation nearby (``n_near``), both
            (``n_covered``), and the same against the passing set (``n_covered_passing``); and the
            share of the model's draws that are recognised as that model (``recognised``,
            ``recognised_passing``): what the prior gains in animals covered against what it costs
            in telling the models apart.
        ``comparison_animals``: per animal and representation, one column per prior naming the
            models that cover the animal: ``BE+SC``, ``BE``, ``SC`` or ``none``.
    """
    order = [v for v in PRIOR_VARIANTS if v in summaries] + [v for v in summaries if v not in PRIOR_VARIANTS]
    counts, columns = [], []
    for prior in order:
        s = _covered(summaries[prior])
        told = (recognised_tables or {}).get(prior)
        told = told.set_index(['rep', 'model']) if told is not None and len(told) else None
        for (rep, model), g in s.groupby(['rep', 'model'], sort=False):
            row = {'prior': prior, 'rep': rep, 'model': model, 'n_animals': int(len(g)),
                   'n_prior': int(g['n_prior'].iloc[0]), 'n_passing': int(g['n_passing'].iloc[0]),
                   'n_inside': int(g['inside'].sum()), 'n_near': int(g['near'].sum()),
                   'n_covered': int(g['covered'].sum()), 'n_covered_passing': int(g['covered_passing'].sum()),
                   'recognised': np.nan, 'recognised_passing': np.nan}
            if told is not None and (rep, model) in told.index:
                row.update(recognised=float(told.loc[(rep, model), 'recognised']),
                           recognised_passing=float(told.loc[(rep, model), 'recognised_passing']))
            counts.append(row)
        rows = pd.MultiIndex.from_frame(s[['animal_id', 'rep']].drop_duplicates())
        covering = s[s['covered']].groupby(['animal_id', 'rep'], sort=False)['model'].agg('+'.join)
        columns.append(covering.reindex(rows).fillna('none').rename(prior))
    animals = (pd.concat(columns, axis=1).reset_index() if columns
               else pd.DataFrame(columns=['animal_id', 'rep']))
    return {'comparison': pd.DataFrame(counts, columns=COMPARISON_COLUMNS), 'comparison_animals': animals}
