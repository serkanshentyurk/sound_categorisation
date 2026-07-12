"""Synthetic feature diagnostics for the sound-categorisation models.

`compute_param_stat_correlations` samples model parameters from the prior, simulates one
session per draw on a fixed uniform stimulus sequence, computes the flattened summary-stat
vector, and returns the per-parameter / per-summary-stat correlation matrix plus the raw
draws. Built directly on the model simulators and behav_utils stats — no SBI/torch.

Noise is held fixed across draws (same simulator seed every time) so the only thing varying
is the parameter set; the correlations therefore reflect parameter sensitivity rather than
trial noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from behav_utils.data.synthetic import sample_stimuli
from behav_utils.analysis.summary_stats import (
    fit_summary_stats, flatten_stats, get_stat_names_expanded, list_available_stats,
)
from models.BE_core import BEModel, BEParams, BEState
from models.SC_core import SCModel, SCParams, SCState

# model_type -> (Model, Params, initial-state factory)
_MODELS = {
    'be': (BEModel, BEParams, BEState.initial_uniform),
    'sc': (SCModel, SCParams, SCState.initial_default),
}


def compute_param_stat_correlations(model_type, stat_names=None, n_samples=1000,
                                    n_trials=2000, seed=0):
    """Correlate each model parameter with each summary statistic, across the prior.

    Parameters
    ----------
    model_type : {'be', 'sc'}
    stat_names : list of str, optional
        Defaults to ``list_available_stats()`` (the full expanded vector, including the
        update matrix and conditional psychometric).
    n_samples : int
        Number of parameter draws from the prior.
    n_trials : int
        Trials in the (single, shared) simulated session per draw. Keep this high enough
        that the per-session update matrix is well populated, or draws with non-finite
        stat vectors are dropped (see ``n_valid``).
    seed : int

    Returns
    -------
    dict with keys:
        corr_matrix          (n_params, n_stats_expanded) Pearson r
        param_names          list of parameter names
        stat_names_expanded  expanded stat names (columns of corr_matrix and x)
        theta                (n_valid, n_params) sampled parameters
        x                    (n_valid, n_stats_expanded) simulated stat vectors
        n_valid              number of draws with a finite stat vector
    """
    model_type = model_type.lower()
    if model_type not in _MODELS:
        raise ValueError(f"model_type must be 'be' or 'sc', got {model_type!r}")
    Model, Params, initial_state = _MODELS[model_type]

    if stat_names is None:
        stat_names = list_available_stats()
    stat_names = list(stat_names)
    param_names = Params.get_param_names()
    stat_names_expanded = get_stat_names_expanded(stat_names)

    # one fixed stimulus sequence, shared across all draws
    stimuli, categories = sample_stimuli(n_trials=n_trials, rng=np.random.default_rng(seed))
    no_response = np.zeros(n_trials, dtype=bool)
    not_blockstart = np.ones(n_trials, dtype=bool)
    not_blockstart[0] = False

    param_rng = np.random.default_rng(seed + 1)
    theta, x = [], []
    for _ in range(n_samples):
        params = Params.sample_prior(rng=param_rng)
        choices, *_ = Model.simulate_session(
            stimuli=stimuli, categories=categories, params=params,
            initial_state=initial_state(),
            rng=np.random.default_rng(seed + 2),        # fixed noise across draws
            no_response=no_response, not_blockstart=not_blockstart,
        )
        stats = flatten_stats(fit_summary_stats(
            choices, stimuli, categories, stat_names=stat_names, return_dict=True))
        if np.all(np.isfinite(stats)):
            theta.append(params.to_array())
            x.append(stats)

    theta = np.asarray(theta)
    x = np.asarray(x)

    # pairwise-complete Pearson r; constant/degenerate columns come back as NaN (no warning)
    corr = np.full((len(param_names), len(stat_names_expanded)), np.nan)
    if len(theta) > 1:
        full = pd.DataFrame(np.hstack([theta, x])).corr().to_numpy()
        corr = full[:theta.shape[1], theta.shape[1]:]

    return {
        'corr_matrix': corr,
        'param_names': param_names,
        'stat_names_expanded': stat_names_expanded,
        'theta': theta,
        'x': x,
        'n_valid': len(theta),
    }


# =============================================================================
# Summary-statistic selection / attribution (no SBI; sklearn surrogate)
#
# ONE simulated cohort feeds every vary-all view:
#   simulate_selection_cohort  -> simulate BE & SC once for a distribution (vary='all',
#                                 random prior draws, per-draw noise). All views below read it.
#
#   um_scalar_correlation   -> |r| between each UM cell and each scalar stat (what the
#                              scalars do NOT capture; the framing/motivation view).
#   stat_correlation        -> stat × stat |r| (redundancy → manual shortlist).
#   stat_individual_power   -> each stat ALONE: identity AUC, and per-parameter R²
#                              (univariate / individual predictive power).
#   select_stats            -> exhaustive subset search over a hand-picked shortlist;
#                              each subset scored on identity AUC AND min-across-params R².
#
# Also kept (parked, self-simulating; not used by nb10's core flow):
#   stat_contributions         -> joint permutation importance (vary-all × joint).
#   stat_parameter_sensitivity -> one-at-a-time sweep (vary-one isolation).
#
# Predictors default to the scalar named stats; update_matrix is added to the simulated set
# only as a target for um_scalar_correlation. Predictor columns are required finite per draw;
# UM cells may be NaN (sparse at low n_trials) and are handled pairwise-complete.
# =============================================================================
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from behav_utils.analysis.summary_stats import fit_summary_stats
from utils.stimulus_distributions import sample_distribution

_MATRIX_STATS = ('update_matrix', 'conditional_psychometric')


# ----------------------------------------------------------------------------
# Pool / column bookkeeping
# ----------------------------------------------------------------------------
def _resolve_pool(stat_pool: Optional[Sequence[str]], need_um: bool) -> Tuple[List[str], List[str]]:
    """(predictors, sim_stats). predictors = scalar named stats (matrix stats removed);
    sim_stats = predictors (+ update_matrix when a UM target is needed). All column indices
    are taken against get_stat_names_expanded(sim_stats)."""
    if stat_pool is None:
        predictors = [s for s in list_available_stats() if s not in _MATRIX_STATS]
    else:
        predictors = [s for s in stat_pool if s not in _MATRIX_STATS]
    sim_stats = list(dict.fromkeys(list(predictors) + (['update_matrix'] if need_um else [])))
    return predictors, sim_stats


def _groups_and_um(sim_stats: Sequence[str], predictors: Sequence[str]):
    """{predictor: [col idx]}, update-matrix col indices, expanded names (all vs sim_stats)."""
    expanded = get_stat_names_expanded(sim_stats)
    pos = {n: i for i, n in enumerate(expanded)}
    pred_groups = {p: [pos[c] for c in get_stat_names_expanded([p])] for p in predictors}
    um_cols = [pos[c] for c in expanded if c.startswith('um_')]
    return pred_groups, um_cols, expanded


# ----------------------------------------------------------------------------
# Simulate core
# ----------------------------------------------------------------------------
def _reference_array(Params, ref: Optional[np.ndarray]) -> np.ndarray:
    if ref is not None:
        return np.asarray(ref, float)
    b = Params.get_bounds()
    return np.array([(b[n][0] + b[n][1]) / 2.0 for n in Params.get_param_names()], float)


def _simulate_cohort(model_type, distribution, n_sims, n_trials, stat_names, vary='all',
                     ref=None, vary_noise=True, seed=0, finite_idx=None):
    """Simulate; return (theta, X, param_names, expanded). A draw is kept when the columns in
    finite_idx are all finite (default: all columns). vary='all' draws the prior each step;
    vary=<name> sweeps one parameter (others at ref). vary_noise toggles per-draw stimulus +
    choice noise vs a frozen sequence/seed."""
    Model, Params, init = _MODELS[model_type]
    pnames = Params.get_param_names()

    stim0, cat0 = sample_distribution(n_trials, distribution, rng=np.random.default_rng(seed))
    no_resp = np.zeros(n_trials, dtype=bool)
    not_bs = np.ones(n_trials, dtype=bool); not_bs[0] = False

    if vary == 'all':
        theta_iter = (Params.sample_prior(rng=np.random.default_rng(seed + 1 + i)).to_array()
                      for i in range(n_sims))
    else:
        if vary not in pnames:
            raise ValueError(f"vary={vary!r} not a parameter of {model_type}: {pnames}")
        pidx = pnames.index(vary)
        lo, hi = Params.get_bounds()[vary]
        ref_arr = _reference_array(Params, ref)
        sweep = np.linspace(lo, hi, n_sims)

        def _gen():
            for v in sweep:
                a = ref_arr.copy(); a[pidx] = v
                yield a
        theta_iter = _gen()

    theta, X = [], []
    for i, arr in enumerate(theta_iter):
        params = Params.from_array(np.asarray(arr, float))
        if vary_noise:
            stimuli, categories = sample_distribution(
                n_trials, distribution, rng=np.random.default_rng(seed + 10_000 + i))
            choice_rng = np.random.default_rng(seed + 20_000 + i)
        else:
            stimuli, categories = stim0, cat0
            choice_rng = np.random.default_rng(seed + 2)
        choices, *_ = Model.simulate_session(
            stimuli=stimuli, categories=categories, params=params,
            initial_state=init(), rng=choice_rng, no_response=no_resp, not_blockstart=not_bs)
        stats = flatten_stats(fit_summary_stats(
            choices, stimuli, categories, stat_names=stat_names, return_dict=True))
        keep = np.isfinite(stats if finite_idx is None else stats[finite_idx])
        if np.all(keep):
            theta.append(np.asarray(arr, float))
            X.append(stats)

    return np.asarray(theta), np.asarray(X), pnames, get_stat_names_expanded(stat_names)


# ----------------------------------------------------------------------------
# The shared cohort
# ----------------------------------------------------------------------------
@dataclass
class SelectionCohort:
    distribution: str
    predictors: List[str]                 # named stat groups used as regressors
    sim_stats: List[str]                  # actually simulated (predictors + update_matrix)
    expanded: List[str]                   # expanded column names for sim_stats
    pred_groups: Dict[str, List[int]]     # group -> column indices into X / expanded
    um_cols: List[int]                    # update-matrix column indices
    theta: Dict[str, np.ndarray]          # 'be'/'sc' -> (n, n_param)
    X: Dict[str, np.ndarray]              # 'be'/'sc' -> (n, n_expanded); predictor cols finite
    param_names: Dict[str, List[str]]
    n_valid: Dict[str, int]

    def pred_cols(self) -> List[int]:
        return sorted({c for cols in self.pred_groups.values() for c in cols})

    def local_groups(self) -> Dict[str, List[int]]:
        """pred_groups re-indexed against the predictor-only column matrix."""
        remap = {c: i for i, c in enumerate(self.pred_cols())}
        return {g: [remap[c] for c in cols] for g, cols in self.pred_groups.items()}

    def Xp(self, model: str) -> np.ndarray:
        return self.X[model][:, self.pred_cols()]


def simulate_selection_cohort(distribution: str = 'uniform', n_sims: int = 4000,
                              n_trials: int = 400, stat_pool: Optional[Sequence[str]] = None,
                              vary_noise: bool = True, seed: int = 0) -> SelectionCohort:
    """Simulate BE and SC once (vary='all') and bundle everything the views need.

    update_matrix is included as a target for um_scalar_correlation; only the scalar predictor
    columns are required finite, so n_valid stays close to n_sims even at low n_trials (the UM
    cells may be NaN and are handled pairwise-complete downstream).
    """
    predictors, sim_stats = _resolve_pool(stat_pool, need_um=True)
    pred_groups, um_cols, expanded = _groups_and_um(sim_stats, predictors)
    pred_cols = sorted({c for cols in pred_groups.values() for c in cols})
    theta, X, pn, n_valid = {}, {}, {}, {}
    for m in ('be', 'sc'):
        th, x, names, _ = _simulate_cohort(m, distribution, n_sims, n_trials, sim_stats,
                                           vary='all', vary_noise=vary_noise, seed=seed,
                                           finite_idx=np.array(pred_cols))
        theta[m], X[m], pn[m], n_valid[m] = th, x, names, len(th)
    return SelectionCohort(distribution, list(predictors), list(sim_stats), expanded,
                           pred_groups, um_cols, theta, X, pn, n_valid)


# ----------------------------------------------------------------------------
# Surrogate estimators + scoring
# ----------------------------------------------------------------------------
def _make_estimator(scorer: str, task: str, seed: int = 0):
    """task in {'clf','reg','multireg'}. multireg always RF (GBR has no multi-output)."""
    if task == 'multireg':
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    if scorer == 'gbm':
        return (GradientBoostingClassifier(random_state=seed) if task == 'clf'
                else GradientBoostingRegressor(random_state=seed))
    if scorer == 'rf':
        return (RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1) if task == 'clf'
                else RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1))
    if scorer == 'ridge':
        est = LogisticRegression(max_iter=2000) if task == 'clf' else Ridge()
        return make_pipeline(StandardScaler(), est)
    raise ValueError(f"unknown scorer {scorer!r}")


def _cv_auc(scorer, X, y, n_splits, seed) -> float:
    est = _make_estimator(scorer, 'clf', seed)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(est, X, y, cv=kf, scoring='roc_auc')))


def _cv_r2(scorer, X, y, n_splits, seed) -> float:
    est = _make_estimator(scorer, 'reg', seed)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(est, X, y, cv=kf, scoring='r2')))


# ----------------------------------------------------------------------------
# Q (framing): what the scalars do NOT capture
# ----------------------------------------------------------------------------
def um_scalar_correlation(cohort: SelectionCohort, model: str) -> dict:
    """|Pearson r| between each UM cell and each scalar predictor (pairwise-complete, so sparse
    UM cells are tolerated). A UM row dark across all scalars = structure the scalar vector
    misses (e.g. the SC stripe)."""
    Xm = cohort.X[model]
    feat = cohort.pred_cols()
    um_names = [cohort.expanded[c] for c in cohort.um_cols]
    feat_names = [cohort.expanded[c] for c in feat]
    A = pd.DataFrame(Xm[:, cohort.um_cols], columns=um_names)
    B = pd.DataFrame(Xm[:, feat], columns=feat_names)
    C = pd.concat([A, B], axis=1).corr().abs()
    return {'corr': C.loc[um_names, feat_names], 'model_type': model,
            'distribution': cohort.distribution}


# ----------------------------------------------------------------------------
# Q1: stat x stat correlation (redundancy)
# ----------------------------------------------------------------------------
def stat_correlation(cohort: SelectionCohort, model: str) -> dict:
    """|Pearson r| between predictor columns (per model). Read to drop redundant stats by hand."""
    feat = cohort.pred_cols()
    names = [cohort.expanded[c] for c in feat]
    C = pd.DataFrame(cohort.X[model][:, feat], columns=names).corr().abs()
    return {'corr': C, 'model_type': model, 'distribution': cohort.distribution}


# ----------------------------------------------------------------------------
# Q2 + Q3: individual predictive power (each stat alone)
# ----------------------------------------------------------------------------
def stat_individual_power(cohort: SelectionCohort, scorer: str = 'gbm',
                          n_splits: int = 5, seed: int = 0) -> dict:
    """Each stat ON ITS OWN: identity AUC (pooled BE/SC) and per-parameter recovery R² (per
    model). Univariate power — NOT extractable from the joint contribution matrix (a stat can
    be individually strong but jointly redundant). 'stat' = the whole named group (all columns).
    """
    groups = cohort.local_groups()
    Xid = np.vstack([cohort.Xp('be'), cohort.Xp('sc')])
    yid = np.concatenate([np.zeros(cohort.n_valid['be']), np.ones(cohort.n_valid['sc'])])
    identity = {g: _cv_auc(scorer, Xid[:, gc], yid, n_splits, seed) for g, gc in groups.items()}

    recovery = {}
    for m in ('be', 'sc'):
        Xm = cohort.Xp(m); th = cohort.theta[m]; pn = cohort.param_names[m]
        recovery[m] = pd.DataFrame(
            {p: {g: _cv_r2(scorer, Xm[:, gc], th[:, j], n_splits, seed)
                 for g, gc in groups.items()}
             for j, p in enumerate(pn)}).reindex(list(cohort.pred_groups.keys()))
    return {'distribution': cohort.distribution,
            'identity': pd.Series(identity).reindex(list(cohort.pred_groups.keys())),
            'recovery': recovery, 'param_names': cohort.param_names}


# ----------------------------------------------------------------------------
# Q4: subset selection over a hand-picked shortlist (identity AND params)
# ----------------------------------------------------------------------------
def select_stats(cohort: SelectionCohort, shortlist: Sequence[str], scorer: str = 'gbm',
                 method: str = 'auto', max_exhaustive: int = 4096, n_splits: int = 5,
                 eps: float = 0.01, identity_floor: float = 0.9, seed: int = 0) -> dict:
    """Exhaustive (or greedy) search over the hand-picked `shortlist`. Each subset is scored on
    identity AUC (pooled BE/SC) AND min-across-parameters CV R² (both models pooled; min so a
    single weak parameter can't be hidden). Selected = smallest subset whose identity AUC clears
    `identity_floor` and whose min-R² is within `eps` of the best. Curate `shortlist` first from
    stat_correlation / stat_individual_power; exhaustive needs <= ~12 groups.
    """
    shortlist = [s for s in shortlist if s in cohort.pred_groups]
    groups = cohort.pred_groups
    Xid_full = np.vstack([cohort.X['be'], cohort.X['sc']])
    yid = np.concatenate([np.zeros(cohort.n_valid['be']), np.ones(cohort.n_valid['sc'])])

    def cols_of(sub): return sorted({c for g in sub for c in groups[g]})

    def score(sub):
        cols = cols_of(sub)
        auc = _cv_auc(scorer, Xid_full[:, cols], yid, n_splits, seed)
        mins = []
        for m in ('be', 'sc'):
            th = cohort.theta[m]; Xm = cohort.X[m][:, cols]
            mins += [_cv_r2(scorer, Xm, th[:, j], n_splits, seed) for j in range(th.shape[1])]
        return auc, float(np.min(mins))

    G = len(shortlist)
    use = 'exhaustive' if (method == 'exhaustive'
                           or (method == 'auto' and (2 ** G - 1) <= max_exhaustive)) else 'greedy'
    if use == 'exhaustive' and (2 ** G - 1) > max_exhaustive and method == 'exhaustive':
        raise ValueError(f"exhaustive over {G} groups = {2**G-1} subsets > {max_exhaustive}; "
                         "shorten the shortlist or use method='greedy'.")

    rows, best_by_k = [], []
    if use == 'exhaustive':
        for k in range(1, G + 1):
            best = None
            for sub in combinations(shortlist, k):
                auc, r2 = score(sub)
                rows.append({'k': k, 'stats': frozenset(sub), 'identity_auc': auc, 'r2_min': r2})
                if best is None or r2 > best['r2_min']:
                    best = {'k': k, 'stats': frozenset(sub), 'identity_auc': auc, 'r2_min': r2}
            best_by_k.append(best)
    else:
        chosen, remaining = [], list(shortlist)
        while remaining:
            scored = [(score(chosen + [g]), g) for g in remaining]
            (auc, r2), g = max(scored, key=lambda t: t[0][1])
            chosen.append(g); remaining.remove(g)
            best_by_k.append({'k': len(chosen), 'stats': frozenset(chosen),
                              'identity_auc': auc, 'r2_min': r2})

    curve = pd.DataFrame(best_by_k)
    ok = curve[curve['identity_auc'] >= identity_floor]
    pool = ok if len(ok) else curve
    best_r2 = pool['r2_min'].max()
    selected = pool[pool['r2_min'] >= best_r2 - eps].iloc[0]
    return {'distribution': cohort.distribution, 'method': use,
            'best_by_k': curve, 'selected': set(selected['stats']),
            'selected_r2_min': float(selected['r2_min']),
            'selected_identity_auc': float(selected['identity_auc']),
            'per_subset': pd.DataFrame(rows) if rows else None}


# ----------------------------------------------------------------------------
# Parked: joint permutation importance (vary-all × joint)
# ----------------------------------------------------------------------------
def _group_perm_importance(scorer, task, X, y, groups, n_splits=5, n_repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    twod = (y.ndim == 2)

    def sc(est, Xt, yt):
        if task == 'clf':
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(yt, est.predict_proba(Xt)[:, 1])
        from sklearn.metrics import r2_score
        return r2_score(yt, est.predict(Xt), multioutput='uniform_average') if task == 'multireg' \
            else r2_score(yt, est.predict(Xt))

    base_all, imp = [], {g: [] for g in groups}
    for tr, te in kf.split(X):
        est = _make_estimator(scorer, task, seed)
        est.fit(X[tr], y[tr, :] if twod else y[tr])
        yte = y[te, :] if twod else y[te]
        base = sc(est, X[te], yte); base_all.append(base)
        for g, cols in groups.items():
            drops = []
            for _ in range(n_repeats):
                Xp = X[te].copy(); Xp[:, cols] = X[te][rng.permutation(len(te))][:, cols]
                drops.append(base - sc(est, Xp, yte))
            imp[g].append(np.mean(drops))
    return {g: float(np.mean(v)) for g, v in imp.items()}, float(np.mean(base_all))


def stat_contributions(cohort: SelectionCohort, scorer: str = 'gbm', n_splits: int = 5,
                       n_repeats: int = 5, seed: int = 0) -> dict:
    """Joint permutation importance for model_id / UM / each parameter (vary-all × joint).
    Parked: nb10 uses individual power instead, but kept for the 'value given the rest' view.
    Importances are conditional on the other stats — read against stat_correlation."""
    groups = cohort.local_groups()
    cols_out, ceilings = {}, {}
    Xid = np.vstack([cohort.Xp('be'), cohort.Xp('sc')])
    yid = np.concatenate([np.zeros(cohort.n_valid['be']), np.ones(cohort.n_valid['sc'])])
    imp, base = _group_perm_importance(scorer, 'clf', Xid, yid, groups, n_splits, n_repeats, seed)
    cols_out['model_id'] = imp; ceilings['model_id'] = base
    for m in ('be', 'sc'):
        Xp = cohort.Xp(m); th = cohort.theta[m]
        if cohort.um_cols:
            um_local = cohort.X[m][:, cohort.um_cols]
            mask = np.all(np.isfinite(um_local), axis=1)   # drop sparse-UM rows for this target
            if mask.sum() > n_splits:
                imp, base = _group_perm_importance(scorer, 'multireg', Xp[mask], um_local[mask],
                                                   groups, n_splits, n_repeats, seed)
                cols_out[f'um_{m}'] = imp; ceilings[f'um_{m}'] = base
        for j, p in enumerate(cohort.param_names[m]):
            imp, base = _group_perm_importance(scorer, 'reg', Xp, th[:, j], groups,
                                               n_splits, n_repeats, seed)
            cols_out[f'{m}:{p}'] = imp; ceilings[f'{m}:{p}'] = base
    return {'distribution': cohort.distribution,
            'contribution': pd.DataFrame(cols_out).reindex(list(cohort.pred_groups.keys())),
            'ceilings': pd.Series(ceilings), 'scorer': scorer}


# ----------------------------------------------------------------------------
# Parked: one-at-a-time sensitivity (vary-one isolation; self-simulating)
# ----------------------------------------------------------------------------
def stat_parameter_sensitivity(model_type: str, distribution: str = 'uniform',
                               n_sims: int = 400, n_trials: int = 400,
                               ref: Optional[np.ndarray] = None,
                               stat_pool: Optional[Sequence[str]] = None,
                               vary_noise: bool = False, seed: int = 0) -> dict:
    """Sweep each parameter alone (others at ref); |Spearman| with each stat. Isolation view;
    counterfactual and local to ref. Self-simulating (vary-one ≠ the shared vary-all cohort)."""
    predictors, sim_stats = _resolve_pool(stat_pool, need_um=False)
    pred_groups, _, _ = _groups_and_um(sim_stats, predictors)
    pnames = _MODELS[model_type][1].get_param_names()
    out = {}
    for p in pnames:
        theta, X, _, _ = _simulate_cohort(model_type, distribution, n_sims, n_trials, sim_stats,
                                          vary=p, ref=ref, vary_noise=vary_noise, seed=seed)
        pidx = pnames.index(p)
        col = {}
        for g, cols in pred_groups.items():
            rhos = []
            for c in cols:
                if np.std(X[:, c]) == 0:
                    rhos.append(0.0)
                else:
                    r, _ = spearmanr(theta[:, pidx], X[:, c])
                    rhos.append(0.0 if np.isnan(r) else abs(r))
            col[g] = max(rhos) if rhos else 0.0
        out[p] = col
    return {'model_type': model_type, 'distribution': distribution,
            'sensitivity': pd.DataFrame(out).reindex(list(pred_groups.keys())),
            'ref': _reference_array(_MODELS[model_type][1], ref)}
