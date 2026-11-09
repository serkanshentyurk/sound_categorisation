"""Thesis tables for the SBI validation (``sc-validation-tables``).

Everything here reads finished runs and returns tidy DataFrames; nothing simulates or
conditions. Inputs:

    <run>/sbc/<D>/n<N>/<rep>/<model>.pkl     sc-sbi-sbc     -> calibration, coverage, recovery
    <run>/{grid_search,sbi}/...              the model-identification results of a synthetic
                                             cohort (truth known)    -> confusion, contested calls
    a real run's results                     -> real animals' sigma_percep, calls per run

Calibration (Talts et al. 2018): ranks of the truth among L posterior samples are uniform for a
calibrated posterior. They are shown as the ECDF of the fractional ranks minus the uniform CDF,
with a simultaneous 1 - alpha band (Säilynoja, Bürkner & Vehtari 2022, simulation method):
under uniformity the whole curve stays inside the band with probability 1 - alpha.

Recovery and identifiability per parameter (Schad, Betancourt & Vasishth 2021): error and bias
relative to the prior width, R^2 of posterior median against truth, posterior contraction
(1 - posterior variance / prior variance) and posterior z-scores ((mean - truth) / SD).

Model recovery reports every voter and the consensus under both rules (one significant vote,
two significant votes), with Clopper-Pearson intervals per true model.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
RULES = (1, 2)                      # min_significant_votes of the two consensus rules reported
VOTERS = ('GS-UM', 'SBI-pooled-UM', 'SBI-moments-UM')


# =============================================================================
# INTERVALS AND BANDS
# =============================================================================

def clopper_pearson(k: int, n: int, alpha: float = ALPHA):
    """Exact (Clopper-Pearson) 1 - alpha interval for a binomial proportion k / n."""
    if n == 0:
        return np.nan, np.nan
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def binomial_band(n: int, p: float, alpha: float = ALPHA):
    """Central 1 - alpha range of an observed proportion out of n when the true rate is p:
    where a calibrated network's empirical coverage at nominal level p should fall."""
    if n == 0:
        return np.nan, np.nan
    return (float(stats.binom.ppf(alpha / 2, n, p)) / n,
            float(stats.binom.ppf(1 - alpha / 2, n, p)) / n)


def ecdf_band(n: int, z: np.ndarray, alpha: float = ALPHA, n_sim: int = 2000, seed: int = 0):
    """Simultaneous 1 - alpha band for the ECDF of n uniform draws, evaluated at ``z``.

    Simulation method of Säilynoja et al. (2022): for each of ``n_sim`` uniform samples take
    the smallest pointwise two-sided binomial tail probability over ``z``; the alpha-quantile of
    those minima is the pointwise level gamma that makes the band simultaneous.
    Returns (lower, upper) ECDF values at ``z``.
    """
    z = np.asarray(z, dtype=float)
    rng = np.random.default_rng(seed)
    min_p = np.empty(n_sim)
    for s in range(n_sim):
        counts = np.searchsorted(np.sort(rng.uniform(size=n)), z, side='right')
        lower = stats.binom.cdf(counts, n, z)
        upper = stats.binom.sf(counts - 1, n, z)
        min_p[s] = np.min(np.minimum(1.0, 2 * np.minimum(lower, upper)))
    gamma = float(np.quantile(min_p, alpha))
    return stats.binom.ppf(gamma / 2, n, z) / n, stats.binom.ppf(1 - gamma / 2, n, z) / n


# =============================================================================
# SBC: calibration, coverage, recovery
# =============================================================================

def load_sbc(run, distribution: str) -> List[Dict]:
    """Every SBC result under ``<run>/sbc/<distribution>/n<N>/<rep>/<model>.pkl``, each with
    'rep', 'model' and 'n_sessions' set from its path."""
    base = Path(run) / 'sbc' / distribution
    out = []
    for pkl in sorted(base.glob('n*/*/*.pkl')):
        with open(pkl, 'rb') as f:
            res = pickle.load(f)
        res['rep'] = pkl.parent.name
        res['model'] = pkl.stem
        res['n_sessions'] = int(pkl.parent.parent.name[1:])
        out.append(res)
    return out


def _keys(res) -> Dict:
    return {'rep': res['rep'], 'model': res['model'], 'n_sessions': res['n_sessions']}


def sbc_summary_table(results: Iterable[Dict]) -> pd.DataFrame:
    """Per network, session count and parameter: draws used, KS p-value, acceptance."""
    rows = []
    for res in results:
        acc = np.asarray(res.get('acceptance_rate', []), dtype=float)
        hit = np.asarray(res.get('budget_hit', []), dtype=bool)
        for j, p in enumerate(res['param_names']):
            rows.append({**_keys(res), 'param': p, 'n_draws': res['n_draws'],
                         'n_used': int(len(res['ranks'])), 'n_skipped': res['n_skipped'],
                         'ks_p': res['ks_pvalues'][j],
                         'acceptance_median': float(np.nanmedian(acc)) if acc.size else np.nan,
                         'budget_hit_frac': float(hit.mean()) if hit.size else np.nan})
    return pd.DataFrame(rows)


def coverage_table(results: Iterable[Dict], alpha: float = ALPHA) -> pd.DataFrame:
    """Empirical coverage per nominal level, with the range a calibrated network would give."""
    rows = []
    for res in results:
        cov = res['coverage']
        for _, r in cov.iterrows():
            lo, hi = binomial_band(int(r['n']), float(r['nominal']), alpha)
            rows.append({**_keys(res), 'param': r['param'], 'nominal': r['nominal'],
                         'empirical': r['empirical'], 'n': int(r['n']),
                         'band_lo': lo, 'band_hi': hi})
    return pd.DataFrame(rows)


def ecdf_table(results: Iterable[Dict], n_points: int = 50, alpha: float = ALPHA,
               n_sim: int = 2000) -> pd.DataFrame:
    """ECDF of the fractional ranks minus the uniform CDF at ``n_points`` points, with the
    simultaneous band (also as differences)."""
    z = np.arange(1, n_points + 1) / (n_points + 1)
    bands = {}
    rows = []
    for res in results:
        ranks = np.asarray(res['ranks'], dtype=float)
        n = len(ranks)
        if n == 0:
            continue
        if n not in bands:
            bands[n] = ecdf_band(n, z, alpha=alpha, n_sim=n_sim)
        lo, hi = bands[n]
        u = (ranks + 0.5) / (res['n_posterior_samples'] + 1)       # fractional rank in (0, 1)
        for j, p in enumerate(res['param_names']):
            ecdf = np.searchsorted(np.sort(u[:, j]), z, side='right') / n
            for i in range(n_points):
                rows.append({**_keys(res), 'param': p, 'z': z[i], 'ecdf_diff': ecdf[i] - z[i],
                             'band_lo': lo[i] - z[i], 'band_hi': hi[i] - z[i]})
    return pd.DataFrame(rows)


def recovery_draws(results: Iterable[Dict]) -> pd.DataFrame:
    """One row per SBC draw and parameter: truth, posterior median, mean, SD, 5-95 % interval,
    contraction and z-score."""
    frames = []
    for res in results:
        if not len(res['ranks']):
            continue
        q = list(res.get('quantile_levels', []))
        tq = np.asarray(res.get('theta_quantiles'))
        for j, p in enumerate(res['param_names']):
            lo, hi = res['prior_bounds'][p]
            prior_var = (hi - lo) ** 2 / 12
            sd = res['theta_sd'][:, j]
            df = pd.DataFrame({
                'truth': res['theta_true'][:, j], 'median': res['theta_median'][:, j],
                'mean': res['theta_mean'][:, j], 'sd': sd,
                'q05': tq[:, q.index(0.05), j] if 0.05 in q else np.nan,
                'q95': tq[:, q.index(0.95), j] if 0.95 in q else np.nan,
                'contraction': 1 - sd ** 2 / prior_var,
                'zscore': (res['theta_mean'][:, j] - res['theta_true'][:, j]) / sd,
                'prior_lo': lo, 'prior_hi': hi,
            })
            for k, v in {**_keys(res), 'param': p}.items():
                df.insert(0, k, v)
            frames.append(df)
    cols = ['rep', 'model', 'n_sessions', 'param']
    return pd.concat(frames, ignore_index=True)[cols + [c for c in frames[0] if c not in cols]] \
        if frames else pd.DataFrame()


def _recovery_metrics(truth, est, lo, hi) -> Dict[str, float]:
    truth, est = np.asarray(truth, dtype=float), np.asarray(est, dtype=float)
    width = hi - lo
    sst = np.sum((truth - truth.mean()) ** 2)
    return {'n': int(len(truth)),
            'rmse_rel': float(np.sqrt(np.mean((est - truth) ** 2)) / width),
            'bias_rel': float(np.mean(est - truth) / width),
            'r2': float(1 - np.sum((est - truth) ** 2) / sst) if sst > 0 else np.nan}


def recovery_table(draws: pd.DataFrame) -> pd.DataFrame:
    """Per network, session count and parameter: error and bias relative to the prior width,
    R^2 of the posterior median against truth, median contraction, median |z| (about 0.67 for
    a calibrated Gaussian posterior) and the fraction of draws with |z| > 2."""
    rows = []
    for key, g in draws.groupby(['rep', 'model', 'n_sessions', 'param'], sort=False):
        m = _recovery_metrics(g['truth'], g['median'], g['prior_lo'].iloc[0], g['prior_hi'].iloc[0])
        z = g['zscore'].abs()
        rows.append({**dict(zip(['rep', 'model', 'n_sessions', 'param'], key)), **m,
                     'contraction_median': float(g['contraction'].median()),
                     'abs_z_median': float(z.median()), 'frac_abs_z_gt_2': float((z > 2).mean())})
    return pd.DataFrame(rows)


# =============================================================================
# MODEL RECOVERY (synthetic cohort, truth known)
# =============================================================================

def _result_files(run, distribution: str, reps: Sequence[str]):
    run = Path(run)
    yield from sorted((run / 'grid_search' / 'update_matrix' / distribution).glob('*.pkl'))
    for rep in reps:
        yield from sorted((run / 'sbi' / 'update_matrix' / distribution / rep).glob('*.pkl'))


def synthetic_truths(run, distribution: str, reps: Sequence[str] = ('pooled', 'moments')) -> pd.DataFrame:
    """Per synthetic animal: true model and parameters, read from the run's own result files."""
    rows = {}
    for pkl in _result_files(run, distribution, reps):
        with open(pkl, 'rb') as f:
            d = pickle.load(f)
        aid = d.get('animal_id')
        if aid in rows or d.get('true_model') not in ('BE', 'SC'):
            continue
        rows[aid] = {'id': aid, 'true_model': d['true_model'], **(d.get('true_params') or {})}
    return pd.DataFrame(list(rows.values()))


def voter_calls(assign: pd.DataFrame, voters: Sequence[str] = VOTERS, alpha: float = ALPHA) -> pd.DataFrame:
    """Long table, one row per animal and voter: the call (BE / SC when p < alpha, else 'no call')."""
    rows = []
    for _, r in assign.iterrows():
        for v in voters:
            if v not in assign:
                continue
            w, p = r.get(v), r.get(f'{v}_p')
            called = w if (w in ('BE', 'SC') and pd.notna(p) and p < alpha) else 'no call'
            rows.append({'id': r['id'], 'true_model': r.get('true_model'), 'voter': v, 'called': called})
    return pd.DataFrame(rows)


def confusion_table(assign_by_rule: Dict[int, pd.DataFrame], alpha: float = ALPHA) -> pd.DataFrame:
    """Per rule, voter (or 'Consensus'), true model and call: count, fraction of that true
    model's animals, and its Clopper-Pearson interval. Voters do not depend on the rule; they
    are listed under each rule so one table holds everything a figure needs."""
    rows = []
    for rule, assign in assign_by_rule.items():
        calls = voter_calls(assign, alpha=alpha)
        cons = assign[['id', 'true_model', 'Consensus']].rename(columns={'Consensus': 'called'})
        cons['voter'] = 'Consensus'
        both = pd.concat([calls, cons[['id', 'true_model', 'voter', 'called']]], ignore_index=True)
        for (voter, tm), g in both.groupby(['voter', 'true_model'], sort=False):
            n = len(g)
            for called, k in g['called'].value_counts().items():
                lo, hi = clopper_pearson(int(k), n, alpha)
                rows.append({'rule': rule, 'voter': voter, 'true_model': tm, 'called': called,
                             'n': int(k), 'n_true': n, 'frac': k / n, 'ci_lo': lo, 'ci_hi': hi})
    return pd.DataFrame(rows)


def call_summary(confusion: pd.DataFrame) -> pd.DataFrame:
    """Per rule, voter and true model: correct, wrong and no-call rates with intervals."""
    rows = []
    for (rule, voter, tm), g in confusion.groupby(['rule', 'voter', 'true_model'], sort=False):
        n = int(g['n_true'].iloc[0])
        k_right = int(g.loc[g['called'] == tm, 'n'].sum())
        k_wrong = int(g.loc[g['called'].isin(['BE', 'SC']) & (g['called'] != tm), 'n'].sum())
        k_none = n - k_right - k_wrong
        row = {'rule': rule, 'voter': voter, 'true_model': tm, 'n': n}
        for name, k in (('correct', k_right), ('wrong', k_wrong), ('no_call', k_none)):
            lo, hi = clopper_pearson(k, n)
            row.update({name: k, f'{name}_frac': k / n, f'{name}_lo': lo, f'{name}_hi': hi})
        rows.append(row)
    return pd.DataFrame(rows)


def contested_table(assign_by_rule: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Per rule and true model: calls made, how many were contested (a significant vote for the
    other model), and how many of each kind were correct."""
    rows = []
    for rule, a in assign_by_rule.items():
        called = a[a['Consensus'].isin(['BE', 'SC'])]
        for tm, g in called.groupby('true_model'):
            right = g['Consensus'] == tm
            cont = g['contested'].astype(bool)
            rows.append({'rule': rule, 'true_model': tm, 'n_called': len(g),
                         'n_contested': int(cont.sum()), 'contested_correct': int((cont & right).sum()),
                         'uncontested_correct': int((~cont & right).sum())})
    return pd.DataFrame(rows)


def gs_sigma_table(assign: pd.DataFrame, truths: pd.DataFrame, sigma_max: float,
                   alpha: float = ALPHA) -> pd.DataFrame:
    """Grid-search calls split by whether the true sigma_percep lies inside the grid
    (<= sigma_max) or above it, per true model."""
    calls = voter_calls(assign, voters=('GS-UM',), alpha=alpha)
    calls = calls.merge(truths[['id', 'sigma_percep']], on='id', how='left')
    calls['sigma_band'] = np.where(calls['sigma_percep'] <= sigma_max, f'<= {sigma_max:g}',
                                   f'> {sigma_max:g}')
    rows = []
    for (tm, band), g in calls.groupby(['true_model', 'sigma_band']):
        n = len(g)
        k_right = int((g['called'] == tm).sum())
        k_wrong = int(g['called'].isin(['BE', 'SC']).sum()) - k_right
        lo, hi = clopper_pearson(k_right, n, alpha)
        rows.append({'true_model': tm, 'sigma_band': band, 'n': n, 'correct': k_right,
                     'wrong': k_wrong, 'no_call': n - k_right - k_wrong,
                     'correct_frac': k_right / n, 'correct_lo': lo, 'correct_hi': hi})
    return pd.DataFrame(rows)


def synthetic_recovery_table(run, distribution: str, reps: Sequence[str] = ('pooled', 'moments')) -> pd.DataFrame:
    """Parameter recovery on the synthetic cohort itself: each animal's saved all-session
    posterior median from the network of its true model, against its true parameters."""
    rows = []
    for rep in reps:
        d = Path(run) / 'sbi' / 'update_matrix' / distribution / rep
        per = {}
        for pkl in sorted(d.glob('*.pkl')):
            with open(pkl, 'rb') as f:
                r = pickle.load(f)
            if r.get('model') != r.get('true_model') or not r.get('posterior'):
                continue
            post = r['posterior']
            for p, est in post['point_estimate'].items():
                per.setdefault((r['true_model'], p), []).append(
                    (r['true_params'][p], est, *post['prior_bounds'][p]))
        for (tm, p), vals in per.items():
            t, e, lo, hi = (np.array(x) for x in zip(*vals))
            rows.append({'rep': rep, 'true_model': tm, 'param': p,
                         **_recovery_metrics(t, e, float(lo[0]), float(hi[0]))})
    return pd.DataFrame(rows)


# =============================================================================
# REAL ANIMALS
# =============================================================================

def real_sigma_table(run, distribution: str, sigma_max: float,
                     reps: Sequence[str] = ('pooled', 'moments')) -> pd.DataFrame:
    """Per real animal, representation and fitted model: the posterior median of
    sigma_percep and whether it lies above the grid search's largest value."""
    rows = []
    for rep in reps:
        d = Path(run) / 'sbi' / 'update_matrix' / distribution / rep
        for pkl in sorted(d.glob('*.pkl')):
            with open(pkl, 'rb') as f:
                r = pickle.load(f)
            post = r.get('posterior') or {}
            sigma = (post.get('point_estimate') or {}).get('sigma_percep')
            if sigma is None:
                continue
            rows.append({'id': r['animal_id'], 'rep': rep, 'model': r['model'],
                         'sigma_percep_median': float(sigma), 'above_grid': bool(sigma > sigma_max)})
    return pd.DataFrame(rows)


def real_calls_table(assign_by_label: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per animal: the consensus, its contested flag and the significant-vote counts
    in each labelled assignment table (e.g. the October run and the new run under each rule)."""
    out = None
    for label, a in assign_by_label.items():
        cols = {'Consensus': f'{label}', 'contested': f'{label}_contested',
                'n_sig_BE': f'{label}_n_sig_BE', 'n_sig_SC': f'{label}_n_sig_SC'}
        keep = a[['id'] + [c for c in cols if c in a]].rename(columns=cols)
        out = keep if out is None else out.merge(keep, on='id', how='outer')
    return out if out is not None else pd.DataFrame()
