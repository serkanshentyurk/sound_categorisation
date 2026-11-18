"""
How long the criterion remembers the stimulus distribution (thesis plan, analysis 6).

Two designs show it from two sides. In the pre-opto cohort the distribution stayed for blocks of
several sessions (Uniform → Hard-B → Hard-A → Hard-B → Hard-A) and the criterion moved towards each;
in the opto cohort's Hard phase it alternated every day and the criterion did not follow. A leaky
integrator with one memory length can do both.

The model. Each trial the criterion moves 1/N of the way towards the normative PSE of the current
session's distribution (``compute_normative_pse`` at the animal's expert-Uniform psychometric sigma, as
in ``adaptation.compute_trajectory``); N is the memory, in trials. It runs through the animal's real
sequence from its first Hard session to its last (Uniform sessions in between included), starting at
the Uniform normative PSE, with each session's number of responded trials and the sessions back to back
(no overnight change). A session's prediction is the model's mean over that session:

    PSE_s = offset + gain * model_s(N) + drift * k_s  [+ light * laser_s]          k_s: session order

``gain`` is 1, the criterion going all the way to the normative PSE, or fitted (``gain='free'``): a
check on the blocked animals, where a partial shift would otherwise be read as a slow one. With the
daily alternation the gain cannot be told apart from N, so the free-gain fit says nothing there. For
each N on ``MEMORY_GRID`` the linear terms are solved by least squares; N̂ minimises the residual sum of
squares, and its 95 % interval is every N whose F statistic
``(RSS_N - RSS_min) / (RSS_min / (n - p))`` is within the F(1, n - p) 95 % point, p counting N and the
linear terms. When the fitted sessions barely swing, N → ∞ fits as well as anything: only the lower end
of the interval is informative, and ``n_high`` is inf.

The shift, model-free: the same sessions' PSE regressed on distribution (Hard-A against Hard-B), the
session order and, where present, the light constant. The distribution coefficient over the normative
difference is the share of the normative shift with the drift taken out; its interval is the ordinary
least-squares t interval.

Both intervals were chosen by simulation (tests/unit/test_timescale.py): with the 6 to 20 sessions an
animal has, they cover a known truth about 95 % of the time, where a chi-square likelihood-ratio
interval for N and a session bootstrap for the shift covered 86 to 93 %.

Variants, which sessions' criterion is fitted (the model runs through every session either way):

    main        sessions without the laser: every Hard session of a blocked animal, the masking
                sessions of the opto Hard phase; all trials
    laser_off   every Hard session, laser-off trials only, with a constant for the laser sessions
    all_trials  every Hard session, all trials, with the same constant

An animal without laser sessions has main only. The laser variants are checks, not a laser effect:
light alone moves the criterion, and the laser sessions came first.

The PSE is ``pse_fixed`` from ``adaptation.compute_trajectory``: the criterion with slope and lapses
pinned to the animal's expert-Uniform fit, the steadier estimate per session.

In session means, with L trials per distribution in a regular alternation, the share of the normative
shift that shows is ``swing_fraction(L, N) = 1 - 2 tanh(x/2) / x``, x = L/N: about 1 % for L = 400 and
N = 1,000, and 67 % for L = 6,000.

    res = compute_timescale(animal)          # TimescaleResult
    T = timescale_tables([res, ...], genotypes)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from behav_utils.data import TrialArrays, filter_trials, select_sessions
from scipy import stats

from sound_categorisation.behaviour.adaptation import BiasRule, compute_trajectory, flag_biased_sessions
from sound_categorisation.data.cohort import LASER_TYPES
from sound_categorisation.data.stimuli import compute_normative_pse

__all__ = ['VARIANTS', 'GAINS', 'MEMORY_GRID', 'LEVEL', 'IntegratorFit', 'ShiftFit', 'TimescaleResult',
           'swing_fraction', 'integrator_means', 'session_sequence', 'fit_integrator', 'fit_shift',
           'compute_timescale', 'timescale_tables']

HARD = ('Hard-A', 'Hard-B')
VARIANTS = ('main', 'laser_off', 'all_trials')
GAINS = ('fixed', 'free')
PSE_COLUMN = {'main': 'pse_all', 'laser_off': 'pse_laser_off', 'all_trials': 'pse_all'}
# memory in trials: 2 to 100,000 in steps of about 3 % (the interval's ends are read off this grid),
# then no movement at all
MEMORY_GRID = np.append(np.unique(np.round(np.geomspace(2, 100_000, 400))), np.inf)
LEVEL = 0.95              # of every interval

SESSION_COLUMNS = ['animal', 'order', 'session_idx', 'session_id', 'distribution', 'session_type', 'laser',
                   'n_trials', 'normative_pse', 'pse_all', 'pse_laser_off', 'lapse_low', 'lapse_high',
                   'accuracy']


@dataclass(frozen=True)
class IntegratorFit:
    """The leaky integrator fitted to one animal's sessions (one variant, one gain mode)."""

    animal: str
    variant: str
    gain_mode: str             # 'fixed' (gain 1) or 'free'
    n_hat: float               # memory in trials at the best fit (inf: no movement fits best)
    n_low: float               # 95 % profile interval
    n_high: float              # inf when no movement cannot be excluded
    offset: float
    gain: float
    drift: float               # PSE change per session
    light: float               # constant for laser sessions (nan without one)
    rss: float
    n_sessions: int
    profile: pd.DataFrame = field(repr=False)      # memory, rss, f (the interval's statistic)
    predicted: pd.Series = field(repr=False)       # best-fit prediction for every session, by order

    def row(self) -> dict:
        return {k: getattr(self, k) for k in ('animal', 'variant', 'gain_mode', 'n_hat', 'n_low', 'n_high',
                                              'offset', 'gain', 'drift', 'light', 'rss', 'n_sessions')}


@dataclass(frozen=True)
class ShiftFit:
    """Hard-A minus Hard-B in session PSE with the drift taken out (one variant)."""

    animal: str
    variant: str
    shift: float
    shift_low: float
    shift_high: float
    normative_difference: float
    drift: float
    drift_low: float
    drift_high: float
    n_sessions: int

    @property
    def fraction(self) -> float:
        """The shift as a share of the normative one."""
        return self.shift / self.normative_difference if self.normative_difference else np.nan

    def row(self) -> dict:
        d = {k: getattr(self, k) for k in ('animal', 'variant', 'shift', 'shift_low', 'shift_high',
                                           'normative_difference', 'drift', 'drift_low', 'drift_high',
                                           'n_sessions')}
        nd = self.normative_difference
        d.update({'fraction': self.fraction,
                  'fraction_low': self.shift_low / nd if nd else np.nan,
                  'fraction_high': self.shift_high / nd if nd else np.nan})
        return d


@dataclass(frozen=True)
class TimescaleResult:
    animal: str
    design: str                # 'blocked' or 'alternation' (from the Hard sessions' run lengths)
    sigma: float               # psychometric sigma of the normative PSE
    biased: bool               # adaptation.flag_biased_sessions on the Hard sessions
    sessions: pd.DataFrame     # SESSION_COLUMNS + predicted_<variant> (fixed gain)
    fits: List[IntegratorFit]
    shifts: List[ShiftFit]

    def fit(self, variant: str = 'main', gain_mode: str = 'fixed') -> IntegratorFit | None:
        return next((f for f in self.fits if f.variant == variant and f.gain_mode == gain_mode), None)

    def __repr__(self) -> str:
        f = self.fit()
        n = f'N={f.n_hat:.0f} [{f.n_low:.0f}, {f.n_high:.0f}]' if f is not None else 'no fit'
        return f'TimescaleResult({self.animal!r}, {self.design}, {len(self.sessions)} sessions, {n})'


# ── the integrator ──────────────────────────────────────────────────────────

def swing_fraction(trials_per_distribution, memory):
    """Share of the normative A−B shift seen in session means under a regular alternation of
    ``trials_per_distribution`` trials each, at the steady state, for a leaky integrator of
    ``memory`` trials: ``1 - 2 tanh(x/2) / x`` with x = L/N (1 when N is 0, 0 when N is infinite)."""
    x = np.asarray(trials_per_distribution, dtype=float) / np.asarray(memory, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = 1.0 - 2.0 * np.tanh(x / 2.0) / x
    return np.where(x == 0, 0.0, np.where(np.isinf(x), 1.0, f))


def integrator_means(targets, n_trials, memory, start: float = 0.0) -> np.ndarray:
    """Session means of the leaky integrator, ``(n_sessions, n_memory)``.

    Trial t of a session with target m: c_{t+1} = c_t + (m - c_t) / N, the sessions back to back from
    ``start``. In closed form a session of T trials starting at c has mean
    m + (c - m)(1 - r^T) / (T(1 - r)), r = 1 - 1/N, and ends at m + (c - m) r^T. N must be at least 1;
    N = inf never moves. A session without trials has no mean (nan) and leaves the state as it was.
    """
    memory = np.atleast_1d(np.asarray(memory, dtype=float))
    if np.any(memory < 1):
        raise ValueError('memory must be at least 1 trial')
    finite = np.isfinite(memory)
    r = np.ones_like(memory)
    r[finite] = 1.0 - 1.0 / memory[finite]
    c = np.full(memory.shape, float(start))
    out = np.full((len(targets), len(memory)), np.nan)
    for i, (m, t) in enumerate(zip(targets, n_trials)):
        t = int(t)
        if t <= 0 or not np.isfinite(m):
            continue
        rt = r ** t
        frac = np.ones_like(memory)
        moving = r < 1.0
        frac[moving] = (1.0 - rt[moving]) / (t * (1.0 - r[moving]))
        out[i] = m + (c - m) * frac
        c = m + (c - m) * rt
    return out


# ── the sessions ────────────────────────────────────────────────────────────

def _design(hard: pd.DataFrame) -> str:
    """'blocked' if some distribution stays for two or more consecutive Hard sessions, else 'alternation'."""
    d = hard['distribution'].tolist()
    return 'blocked' if any(a == b for a, b in zip(d, d[1:])) else 'alternation'


def session_sequence(animal, *, trajectories: Mapping[str, object] | None = None) -> tuple:
    """``(sessions, sigma)``: every session from the animal's first Hard session to its last, in order.

    Hard sessions carry the criterion on all trials (``pse_all``) and on laser-off trials
    (``pse_laser_off``; the same as ``pse_all`` without laser sessions), both ``pse_fixed`` from
    ``compute_trajectory``. Uniform sessions in between carry only their trial count: they drive the
    model, nothing is fitted to them. ``trajectories`` can pass precomputed ``compute_trajectory``
    results ({'all': ..., 'non_opto': ...}).
    """
    aid = getattr(animal, 'animal_id', '?')
    tr = dict(trajectories or {})
    traj = tr.get('all') or compute_trajectory(animal, HARD, trials='all', dynamics=False)
    hard = traj.sessions
    if not len(hard):
        return pd.DataFrame(columns=SESSION_COLUMNS), float(traj.sigma)
    seq = hard.reindex(columns=['session_idx', 'session_id', 'distribution', 'session_type', 'n_trials',
                                'normative_pse', 'pse_fixed', 'lapse_low', 'lapse_high', 'accuracy'])
    seq = seq.rename(columns={'pse_fixed': 'pse_all'})
    laser = seq['session_type'].isin(LASER_TYPES)
    if laser.any():
        off = tr.get('non_opto') or compute_trajectory(animal, HARD, trials='non_opto', dynamics=False)
        seq['pse_laser_off'] = seq['session_idx'].map(off.sessions.set_index('session_idx')['pse_fixed'])
    else:
        seq['pse_laser_off'] = seq['pse_all']

    lo, hi = seq['session_idx'].min(), seq['session_idx'].max()
    uniform_pse = float(compute_normative_pse('Uniform', traj.sigma)) if np.isfinite(traj.sigma) else 0.0
    rows = []
    for s in select_sessions(animal, distribution='Uniform'):
        if lo < s.session_idx < hi:
            kept = filter_trials([s], trial_type='all')
            rows.append({'session_idx': s.session_idx, 'session_id': s.session_id, 'distribution': 'Uniform',
                         'session_type': s.session_type, 'normative_pse': uniform_pse,
                         'n_trials': TrialArrays.from_sessions(kept).n_responded if kept else 0})
    if rows:
        seq = pd.concat([seq, pd.DataFrame(rows)], ignore_index=True)
    seq = seq.sort_values('session_idx').reset_index(drop=True)
    seq.insert(0, 'order', np.arange(len(seq)))
    seq.insert(0, 'animal', aid)
    seq['laser'] = seq['session_type'].isin(LASER_TYPES)
    seq['n_trials'] = seq['n_trials'].fillna(0).astype(int)
    return seq.reindex(columns=SESSION_COLUMNS), float(traj.sigma)


def _fitted(seq: pd.DataFrame, variant: str) -> pd.Series:
    """Rows whose criterion the variant fits: Hard sessions with a PSE (main: without the laser)."""
    if variant not in VARIANTS:
        raise ValueError(f'unknown variant {variant!r}; choose from {VARIANTS}')
    rows = seq['distribution'].isin(HARD) & seq[PSE_COLUMN[variant]].notna()
    if variant == 'main':
        rows &= ~seq['laser']
    return rows


def _uniform_start(seq: pd.DataFrame) -> float:
    uni = seq.loc[seq['distribution'] == 'Uniform', 'normative_pse']
    return float(uni.iloc[0]) if len(uni) else 0.0


# ── fitting ─────────────────────────────────────────────────────────────────

def fit_integrator(seq: pd.DataFrame, variant: str = 'main', gain_mode: str = 'fixed',
                   grid=MEMORY_GRID) -> IntegratorFit:
    """The leaky integrator fitted to the sessions ``variant`` selects; see the module docstring."""
    if gain_mode not in GAINS:
        raise ValueError(f'unknown gain mode {gain_mode!r}; choose from {GAINS}')
    animal = str(seq['animal'].iloc[0]) if len(seq) else '?'
    grid = np.asarray(grid, dtype=float)
    rows = _fitted(seq, variant)
    y = seq.loc[rows, PSE_COLUMN[variant]].to_numpy(float)
    k = seq.loc[rows, 'order'].to_numpy(float)
    light = seq.loc[rows, 'laser'].to_numpy(float)
    use_light = variant != 'main' and 0 < light.sum() < len(light)
    n_linear = 2 + (gain_mode == 'free') + use_light
    blank = _empty_fit(animal, variant, gain_mode, len(y))
    if len(y) <= n_linear + 1:                       # nothing left to estimate N from
        return blank
    model = integrator_means(seq['normative_pse'].to_numpy(float), seq['n_trials'].to_numpy(int), grid,
                             start=_uniform_start(seq))
    M = model[rows.to_numpy()]
    rss, coefs = np.full(len(grid), np.nan), []
    for j in range(len(grid)):
        cols = [np.ones_like(y), k] + ([light] if use_light else [])
        if gain_mode == 'free':
            X, target = np.column_stack([M[:, j]] + cols), y
        else:
            X, target = np.column_stack(cols), y - M[:, j]
        beta, *_ = np.linalg.lstsq(X, target, rcond=None)
        rss[j] = float(np.sum((target - X @ beta) ** 2))
        coefs.append(beta)
    best = int(np.nanargmin(rss))
    dof = len(y) - (n_linear + 1)
    f_stat = (rss - rss[best]) / max(rss[best] / dof, 1e-300)
    inside = grid[f_stat <= stats.f.ppf(LEVEL, 1, dof)]
    beta = coefs[best]
    if gain_mode == 'free':
        gain, offset, drift = beta[0], beta[1], beta[2]
    else:
        gain, offset, drift = 1.0, beta[0], beta[1]
    light_c = beta[-1] if use_light else np.nan
    # the prediction for every Hard session, the laser ones included, from the fitted terms
    pred = offset + gain * model[:, best] + drift * seq['order'].to_numpy(float)
    if use_light:
        pred = pred + light_c * seq['laser'].to_numpy(float)
    pred = np.where(seq['distribution'].isin(HARD), pred, np.nan)
    if gain_mode == 'free' and not np.isfinite(grid[best]):
        gain = np.nan                                # a model that never moves: its gain is not identified
    return IntegratorFit(animal, variant, gain_mode, float(grid[best]), float(inside.min()),
                         float(inside.max()), float(offset), float(gain), float(drift), float(light_c),
                         float(rss[best]), int(len(y)),
                         pd.DataFrame({'memory': grid, 'rss': rss, 'f': f_stat}),
                         pd.Series(pred, index=seq['order'].to_numpy(), name=f'predicted_{variant}'))


def _empty_fit(animal, variant, gain_mode, n) -> IntegratorFit:
    nan = float('nan')
    return IntegratorFit(animal, variant, gain_mode, nan, nan, nan, nan, nan, nan, nan, nan, int(n),
                         pd.DataFrame(columns=['memory', 'rss', 'f']), pd.Series(dtype=float))


def fit_shift(seq: pd.DataFrame, variant: str = 'main') -> ShiftFit:
    """Hard-A minus Hard-B in the variant's session PSE, with the drift (and the light constant) taken
    out, and t intervals for the shift and the drift."""
    animal = str(seq['animal'].iloc[0]) if len(seq) else '?'
    sel = seq[_fitted(seq, variant)]
    y = sel[PSE_COLUMN[variant]].to_numpy(float)
    is_a = (sel['distribution'] == 'Hard-A').to_numpy(float)
    light = sel['laser'].to_numpy(float)
    use_light = variant != 'main' and 0 < light.sum() < len(light)
    norms = seq.groupby('distribution')['normative_pse'].first()
    nd = float(norms.get('Hard-A', np.nan) - norms.get('Hard-B', np.nan))
    X = np.column_stack([np.ones(len(y)), is_a, sel['order'].to_numpy(float)] + ([light] if use_light else []))
    dof = len(y) - X.shape[1]
    nan = float('nan')
    if min(is_a.sum(), len(y) - is_a.sum()) < 2 or dof < 1 or np.linalg.matrix_rank(X) < X.shape[1]:
        return ShiftFit(animal, variant, nan, nan, nan, nd, nan, nan, nan, int(len(y)))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(np.diag(float(resid @ resid) / dof * np.linalg.inv(X.T @ X)))
    t = stats.t.ppf(0.5 + LEVEL / 2, dof)
    return ShiftFit(animal, variant, float(beta[1]), float(beta[1] - t * se[1]), float(beta[1] + t * se[1]), nd,
                    float(beta[2]), float(beta[2] - t * se[2]), float(beta[2] + t * se[2]), int(len(y)))


# ── per animal and per cohort ───────────────────────────────────────────────

def compute_timescale(animal, *, bias_rule: BiasRule | None = None, grid=MEMORY_GRID) -> TimescaleResult | None:
    """Every variant the animal's sessions allow, both gain modes, and the shift; None without Hard sessions."""
    seq, sigma = session_sequence(animal)
    hard = seq[seq['distribution'].isin(HARD)]
    if not len(hard):
        return None
    variants = VARIANTS if hard['laser'].any() else ('main',)
    fits = [fit_integrator(seq, v, g, grid=grid) for v in variants for g in GAINS]
    shifts = [fit_shift(seq, v) for v in variants]
    for f in fits:
        if f.gain_mode == 'fixed' and len(f.predicted):
            seq[f'predicted_{f.variant}'] = seq['order'].map(f.predicted)
    flags = flag_biased_sessions(hard.rename(columns={'distribution': 'to_distribution', 'pse_all': 'pse_fixed'}),
                                 bias_rule)
    biased = bool(flags['biased_animal'].iloc[0]) if len(flags) else False
    return TimescaleResult(str(seq['animal'].iloc[0]), _design(hard), sigma, biased, seq, fits, shifts)


def timescale_tables(results: Sequence[TimescaleResult],
                     genotypes: Mapping[str, str] | None = None) -> Dict[str, pd.DataFrame]:
    """``{'sessions', 'fits', 'shifts', 'profiles'}`` for a cohort, one row per animal and session /
    variant and gain / variant / memory, with the animal's design, bias flag and genotype."""
    genotypes = dict(genotypes or {})

    def tag(df, r):
        return df.assign(design=r.design, biased_animal=r.biased, genotype=genotypes.get(r.animal, 'unknown'),
                         sigma=r.sigma)

    sessions, fits, shifts, profiles = [], [], [], []
    for r in results:
        if r is None:
            continue
        sessions.append(tag(r.sessions, r))
        fits.append(tag(pd.DataFrame([f.row() for f in r.fits]), r))
        shifts.append(tag(pd.DataFrame([s.row() for s in r.shifts]), r))
        for f in r.fits:
            if len(f.profile):
                profiles.append(f.profile.assign(animal=r.animal, variant=f.variant, gain_mode=f.gain_mode))

    def cat(frames):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return {'sessions': cat(sessions), 'fits': cat(fits), 'shifts': cat(shifts), 'profiles': cat(profiles)}
