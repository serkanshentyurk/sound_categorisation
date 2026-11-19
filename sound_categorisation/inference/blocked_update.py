"""
Is a silenced trial remembered on the next one? What the fitted models say complete blocking would look like.

For every parameter draw of a fitted model, each animal's expert-Uniform opto sessions
(``cohort.collect_sessions_ppc``) are simulated twice: once learning from every trial, once not learning on laser
trials (``models.make_model(..., laser_blocks_update=True)``), with the same random numbers, so the two runs differ
only by the blocking. On each run the post-laser contrast of the opto report is computed (``contrasts.ppc_contrasts``,
trial class ``post_opto``: the trial after a laser trial against the other laser-off trials; point values, no
resampling). Blocked minus normal is the effect complete blocking would have on that contrast.

The animal's observation is the opto report's delta-of-deltas for the same contrast (opto sessions minus masking
sessions, ``<opto run>/Uniform/ppc_post_opto/<animal>/contrasts.csv``, trials unit), which takes out what the light
does on its own. Masking sessions need no simulation: blue light only, nothing is silenced, so a blocked and a normal
run would be the same there and the masking term cancels.

``blocking_verdict`` reads the observed interval against the predicted effect and against zero:

    too weak to test          the predicted effect lies within the margin (settings.EQUIVALENCE_MARGINS)
    not fully blocked         the interval excludes the predicted effect, on the side of zero
    consistent with blocking  the interval includes the predicted effect and excludes zero
    beyond blocking           the interval excludes the predicted effect, on the far side from zero
    unclear                   the interval includes both the predicted effect and zero
    missing                   no interval or no prediction

The light is on from stimulus onset to the response and off during feedback, so complete blocking is an upper bound
on what silencing could do to learning from the silenced trial.

    draws = posterior_draws(fit_run, 'SS15', 'BE', n=50)
    predicted = blocking_table([('SS15', 'het', 'BE', opto_sessions, draws)])
    summary = summarise(predicted, observed_dod(opto_run, 'SS15'))
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from behav_utils.data import opto_mask
from behav_utils.models import Design, simulate

from sound_categorisation.behaviour.contrasts import ppc_contrasts
from sound_categorisation.data.paths import model_id_dir
from sound_categorisation.inference.cv_utils import read_cv_result
from sound_categorisation.models import make_model
from sound_categorisation.settings import BASE_SEED, EQUIVALENCE_MARGINS, SBI_BURN_IN

__all__ = ['STATS', 'MODELS', 'DRAWS', 'VERDICTS', 'PREDICTED_COLUMNS', 'SUMMARY_COLUMNS', 'SESSION_COLUMNS',
           'posterior_path', 'posterior_draws', 'session_design', 'animal_predictions', 'blocking_table',
           'observed_dod', 'blocking_verdict', 'summarise', 'sessions_table']

STATS = ('recency', 'win_stay', 'lose_shift')
MODELS = ('BE', 'SC')
DRAWS = 50
TRIAL_CLASS = 'post_opto'
VERDICTS = ('too weak to test', 'not fully blocked', 'consistent with blocking', 'beyond blocking', 'unclear',
            'missing')

PREDICTED_COLUMNS = ['animal', 'genotype', 'model', 'draw', 'stat', 'within_blocked', 'within_normal', 'effect']
OBSERVED_COLUMNS = ['animal', 'stat', 'observed', 'ci_lo', 'ci_hi', 'n_post_opto']
SUMMARY_COLUMNS = ['animal', 'genotype', 'model', 'stat', 'effect', 'effect_lo', 'effect_hi', 'n_draws',
                   'observed', 'ci_lo', 'ci_hi', 'n_post_opto', 'margin', 'verdict']
SESSION_COLUMNS = ['animal', 'session_id', 'session_idx', 'n_trials', 'n_laser', 'n_post_laser']


# ── the fitted parameters ───────────────────────────────────────────────────

def posterior_path(fit_run, animal: str, model: str, *, representation: str = 'pooled',
                   distribution: str = 'uniform') -> Path:
    """Where ``sc-sbi-condition`` saved an animal's fit (``paths.model_id_dir``)."""
    return (model_id_dir(Path(fit_run), 'sbi', 'update_matrix', distribution, representation)
            / f'{animal}_{str(model).upper()}.pkl')


def posterior_draws(fit_run, animal: str, model: str, n: int = DRAWS, **where) -> List[Dict[str, float]]:
    """The first ``n`` samples of the animal's saved posterior (the all-session conditioning of the pooled network),
    as parameter dicts. The samples are independent draws, so the first ``n`` are a random subset.

    Raises:
        FileNotFoundError: the run holds no fit of this animal and model.
        ValueError: the fit was saved without posterior samples.
    """
    path = posterior_path(fit_run, animal, model, **where)
    if not path.exists():
        raise FileNotFoundError(f'no {str(model).upper()} fit of {animal}: {path}')
    posterior = (read_cv_result(path) or {}).get('posterior') or {}
    samples = posterior.get('posterior_samples')
    if samples is None:
        raise ValueError(f'{path}: no posterior samples saved')
    names = list(posterior['param_names'])
    return [dict(zip(names, map(float, row))) for row in np.asarray(samples, dtype=float)[:n]]


# ── the simulations ─────────────────────────────────────────────────────────

def session_design(session) -> Design:
    """A real session as a behav_utils ``Design``: its task, stimuli, laser flags, aborts and no-response trials, so a
    model plays the trials the animal played."""
    t = session.trials
    bad = ~np.isfinite(t.stimulus) & ~t.abort
    if bad.any():
        raise ValueError(f'{session.session_id}: {int(bad.sum())} trials that were not aborted have no stimulus')
    return Design(session.task, t.stimulus, label=session.distribution or '', laser=t.opto_on, abort=t.abort,
                  no_response=t.no_response)


def _within(sessions, stats: Sequence[str]) -> pd.Series:
    """The post-laser contrast (post-laser − other laser-off trials, trials unit) of simulated sessions."""
    con = ppc_contrasts(sessions, [], TRIAL_CLASS, list(stats), n_boot=0, n_perm=0, units=('trials',))
    if 'within' not in con:
        return pd.Series(np.nan, index=list(stats))
    return con['within'].contrasts[con.key].diff.reindex(list(stats)).astype(float)


def _draw_seed(seed: int, animal: str, draw: int) -> List[int]:
    """One random stream per animal and draw, the same for both models and whatever other animals are run."""
    return [int(seed), zlib.crc32(animal.encode()), int(draw)]


def animal_predictions(sessions, model: str, draws: Sequence[Dict[str, float]], *, animal: str,
                       genotype: str = 'unknown', seed: int = BASE_SEED, burn_in: int = SBI_BURN_IN,
                       stats: Sequence[str] = STATS) -> pd.DataFrame:
    """Every draw of ``model`` on one animal's opto ``sessions``, blocked and normal: one row per draw and statistic
    (``PREDICTED_COLUMNS``). Each session starts from its own burn-in, as in the fits."""
    designs = [session_design(s) for s in sessions]
    rows = []
    for d, params in enumerate(draws):
        session_seeds = np.random.default_rng(_draw_seed(seed, animal, d)).integers(0, 2**31 - 1, len(designs))
        runs = {}
        for blocked in (True, False):
            agent = make_model(model, params, burn_in=burn_in, laser_blocks_update=blocked)
            runs[blocked] = [simulate(agent, design, np.random.default_rng(int(k)), session_idx=s.session_idx,
                                      animal_id=animal)
                             for s, design, k in zip(sessions, designs, session_seeds)]
        blocked, normal = _within(runs[True], stats), _within(runs[False], stats)
        for st in stats:
            rows.append({'animal': animal, 'genotype': genotype, 'model': str(model).upper(), 'draw': d,
                         'stat': st, 'within_blocked': blocked[st], 'within_normal': normal[st],
                         'effect': blocked[st] - normal[st]})
    return pd.DataFrame(rows, columns=PREDICTED_COLUMNS)


def _one(job: Tuple, kw: dict) -> pd.DataFrame:
    animal, genotype, model, sessions, draws = job
    return animal_predictions(sessions, model, draws, animal=animal, genotype=genotype, **kw)


def blocking_table(jobs: Sequence[Tuple], n_jobs: int = 1, **kw) -> pd.DataFrame:
    """``animal_predictions`` for every ``(animal, genotype, model, sessions, draws)`` in ``jobs``, ``n_jobs`` at once
    (the numbers do not depend on it). ``kw`` goes to ``animal_predictions`` (seed, burn_in, stats)."""
    if n_jobs > 1 and len(jobs) > 1:
        from joblib import Parallel, delayed
        frames = Parallel(n_jobs=n_jobs)(delayed(_one)(job, kw) for job in jobs)
    else:
        frames = [_one(job, kw) for job in jobs]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PREDICTED_COLUMNS)


# ── the observation and the verdict ─────────────────────────────────────────

def observed_dod(opto_run, animal: str, stats: Sequence[str] = STATS, distribution: str = 'Uniform') -> pd.DataFrame:
    """The animal's post-laser delta-of-deltas from an opto-contrasts run (trials unit), one row per statistic
    (``OBSERVED_COLUMNS``): the difference, its interval, and the number of post-laser trials (from the ``within``
    row). No rows when the run has no post-laser condition for the animal."""
    f = Path(opto_run) / distribution / f'ppc_{TRIAL_CLASS}' / animal / 'contrasts.csv'
    if not f.exists() or f.stat().st_size == 0:
        return pd.DataFrame(columns=OBSERVED_COLUMNS)
    c = pd.read_csv(f)
    c = c[(c['unit'] == 'trials') & c['stat'].isin(list(stats))]
    dod = c[c['kind'] == 'dod'].set_index('stat')
    within = c[c['kind'] == 'within'].set_index('stat')
    if not len(dod):
        return pd.DataFrame(columns=OBSERVED_COLUMNS)
    out = pd.DataFrame({'observed': dod['diff'], 'ci_lo': dod['ci_lo'], 'ci_hi': dod['ci_hi']})
    out['n_post_opto'] = within['n_a'].reindex(out.index) if 'n_a' in within else np.nan
    out = out.reindex([s for s in stats if s in out.index]).rename_axis('stat').reset_index()
    return out.assign(animal=animal)[OBSERVED_COLUMNS]


def blocking_verdict(effect: float, ci_lo: float, ci_hi: float, margin: float) -> str:
    """Read the observed interval ``[ci_lo, ci_hi]`` against the effect complete blocking would have (see the module
    docstring for the names)."""
    if not (np.isfinite(effect) and np.isfinite(ci_lo) and np.isfinite(ci_hi)):
        return 'missing'
    if abs(effect) <= margin:
        return 'too weak to test'
    covers_effect, covers_zero = ci_lo <= effect <= ci_hi, ci_lo <= 0 <= ci_hi
    if covers_effect:
        return 'unclear' if covers_zero else 'consistent with blocking'
    beyond = ci_lo > effect if effect > 0 else ci_hi < effect
    return 'beyond blocking' if beyond else 'not fully blocked'


def summarise(predicted: pd.DataFrame, observed: pd.DataFrame, margins: Dict[str, float] | None = None,
              q: Tuple[float, float] = (0.1, 0.9)) -> pd.DataFrame:
    """One row per animal, model and statistic (``SUMMARY_COLUMNS``): the median predicted effect over the draws with
    its ``q`` range, the observed delta-of-deltas with its interval, the margin and the verdict."""
    margins = dict(EQUIVALENCE_MARGINS if margins is None else margins)
    if not len(predicted):
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    g = predicted.groupby(['animal', 'genotype', 'model', 'stat'], sort=False)['effect']
    s = pd.DataFrame({'effect': g.median(), 'effect_lo': g.quantile(q[0]), 'effect_hi': g.quantile(q[1]),
                      'n_draws': g.count()}).reset_index()
    obs = observed if len(observed) else pd.DataFrame(columns=OBSERVED_COLUMNS)
    s = s.merge(obs, on=['animal', 'stat'], how='left')
    s['margin'] = s['stat'].map(margins).astype(float)
    s['verdict'] = [blocking_verdict(*v) for v in zip(s['effect'], s['ci_lo'].astype(float),
                                                       s['ci_hi'].astype(float), s['margin'])]
    return s[SUMMARY_COLUMNS]


def sessions_table(sessions_by_animal: Dict[str, list]) -> pd.DataFrame:
    """The opto sessions each animal was simulated on (``SESSION_COLUMNS``)."""
    rows = [{'animal': animal, 'session_id': s.session_id, 'session_idx': s.session_idx,
             'n_trials': int(s.trials.n_trials), 'n_laser': int((s.trials.opto_on & ~s.trials.abort).sum()),
             'n_post_laser': int(opto_mask(s.trials, delta=1).sum())}
            for animal, sessions in sessions_by_animal.items() for s in sessions]
    return pd.DataFrame(rows, columns=SESSION_COLUMNS)
