#!/usr/bin/env python
"""Choose the SBI summary-stat vector: tables and plots saying why each stat is in or out.

Simulates BE and SC animals from the manuscript priors, one long session each (``--n-trials``;
2,500 by default, standing in for the 5 x 500-trial pooled "mega session" the networks see --
the only difference is that the model's state is not reset at session breaks), computes every
candidate stat on each, and scores:

    identity_auc.csv        each stat alone: cross-validated AUC for BE vs SC
    recovery_r2.csv         each stat alone: cross-validated R² for each parameter (tidy rows)
    recovery_r2_{be,sc}.csv the same, wide (stat x parameter)
    stat_correlation_{be,sc}.csv   |r| between stats (redundancy)
    forward_selection.csv   greedy curve: stats added by identity AUC until it plateaus,
                            then by min-across-parameters R²; one row per step
    proposed_stats.txt      the curve's suggested set -- a starting point, not the decision
    cohort.pkl              the simulated cohort (theta, stat vectors), so every table and
                            plot can be remade or extended without resimulating
    *.png                   one plot per table; meta.json

The moments representation uses the same stats (mean and variance of each across sessions), so
one selection serves both networks. ``SBI_STATS`` in ``inference/constants.py`` is set by hand
after reading these outputs.

    sc-feature-selection                       # 2,000 BE + 2,000 SC x 2,500 trials, ~20-30 min
    sc-feature-selection --fast                # 60 + 60 x 400 trials, a minute: pipeline check
    sc-feature-selection --stats accuracy psychometric win_stay ...   # a different pool

Writes to ``<results root>/feature_selection/<distribution>/<run_id>/``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import matplotlib
import pandas as pd

from sound_categorisation.data.paths import build_metadata, start_run
from sound_categorisation.features.feature_diagnostics import (
    forward_selection,
    simulate_selection_cohort,
    stat_correlation,
    stat_individual_power,
)
from sound_categorisation.inference.constants import SBI_STATS
from sound_categorisation.settings import DISTRIBUTIONS

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from sound_categorisation.plotting.sbi_validation import (  # noqa: E402
    plot_forward_selection,
    plot_individual_identity,
    plot_individual_recovery,
    plot_stat_correlation,
)

# Candidate pool: every registry producer that is defined on simulated choices and does not
# depend on within-session order (no pse_dynamics trajectory, no reaction times, no fit
# goodness-of-fit scalars, no sd_profile / sdt which duplicate psychometric / accuracy).
DEFAULT_POOL = (
    'accuracy', 'psychometric', 'stimulus_sensitivity', 'hard_accuracy',
    'easy_accuracy', 'side_bias', 'choice_entropy', 'pse',
    'recency', 'stimulus_recency', 'recency_divergence',
    'win_stay', 'win_stay_rate', 'lose_shift', 'perseveration',
    'choice_autocorr', 'logistic_history',
)
N_SIMS, N_TRIALS = 2000, 2500
FAST_N_SIMS, FAST_N_TRIALS = 60, 400


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_feature_selection(out, distribution, pool, n_sims, n_trials, scorer, seed, meta):
    t0 = time.time()
    print(f'[fs] simulating {n_sims} BE + {n_sims} SC animals x {n_trials} trials ({distribution})')
    cohort = simulate_selection_cohort(distribution=distribution, n_sims=n_sims, n_trials=n_trials,
                                       stat_pool=list(pool), seed=seed)
    (out / 'cohort.pkl').write_bytes(pickle.dumps(cohort))
    print(f'[fs]   n_valid={cohort.n_valid}  predictors={len(cohort.pred_cols())} columns '
          f'in {len(cohort.pred_groups)} groups  ({time.time() - t0:.0f}s)')

    # ── each stat alone: identity and recovery ──────────────────────────────
    t1 = time.time()
    ip = stat_individual_power(cohort, scorer=scorer, seed=seed)
    ip['identity'].rename('identity_auc').rename_axis('stat').sort_values(ascending=False) \
        .to_csv(out / 'identity_auc.csv')
    rows = []
    for m in ('be', 'sc'):
        R = ip['recovery'][m]
        R.to_csv(out / f'recovery_r2_{m}.csv', index_label='stat')
        for stat, r in R.iterrows():
            rows += [{'model': m.upper(), 'stat': stat, 'param': p, 'r2': float(v)} for p, v in r.items()]
    pd.DataFrame(rows).to_csv(out / 'recovery_r2.csv', index=False)
    _save(plot_individual_identity(ip).figure, out / 'identity_auc.png')
    for m in ('be', 'sc'):
        _save(plot_individual_recovery(ip, m).figure, out / f'recovery_r2_{m}.png')
    print(f'[fs]   individual power done ({time.time() - t1:.0f}s)')

    # ── redundancy ──────────────────────────────────────────────────────────
    for m in ('be', 'sc'):
        sc = stat_correlation(cohort, m)
        sc['corr'].to_csv(out / f'stat_correlation_{m}.csv', index_label='stat')
        _save(plot_stat_correlation(sc, annot=True).figure, out / f'stat_correlation_{m}.png')

    # ── forward selection over the whole pool ───────────────────────────────
    t2 = time.time()
    fs = forward_selection(cohort, pool=list(pool), scorer=scorer, seed=seed)
    fs['curve'].to_csv(out / 'forward_selection.csv', index=False)
    (out / 'proposed_stats.txt').write_text('\n'.join(fs['suggested']) + '\n')
    _save(plot_forward_selection(fs).figure, out / 'forward_selection.png')
    print(f'[fs]   forward selection done ({time.time() - t2:.0f}s)')
    print(fs['curve'][['k', 'added', 'phase', 'identity_auc', 'r2_min', 'r2_mean']].round(3).to_string(index=False))
    print(f"[fs] suggested (k={fs['suggested_k']}): {fs['suggested']}")
    print(f'[fs] current SBI_STATS ({len(SBI_STATS)}): {list(SBI_STATS)}')

    meta.update(n_valid=cohort.n_valid, pool=list(pool), scorer=scorer,
                suggested=fs['suggested'], elapsed_s=round(time.time() - t0, 1))
    (out / 'meta.json').write_text(json.dumps(meta, indent=2, default=str))
    print(f'[fs] done in {(time.time() - t0) / 60:.1f} min -> {out}')
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--distribution', default='uniform', choices=list(DISTRIBUTIONS))
    p.add_argument('--stats', nargs='+', default=list(DEFAULT_POOL),
                   help='candidate stat producers (registry names); default: the %(default)s pool')
    p.add_argument('--n-sims', type=int, default=None, help=f'animals per model (default {N_SIMS})')
    p.add_argument('--n-trials', type=int, default=None, help=f'trials per animal (default {N_TRIALS})')
    p.add_argument('--scorer', default='hgb', choices=('hgb', 'gbm', 'rf', 'ridge'),
                   help='surrogate for AUC / R² (default %(default)s)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fast', action='store_true',
                   help=f'{FAST_N_SIMS} animals x {FAST_N_TRIALS} trials: pipeline check, run id gets _fast')
    p.add_argument('--label', default=None, help='free suffix for the run id')
    a = p.parse_args(argv)

    n_sims = a.n_sims or (FAST_N_SIMS if a.fast else N_SIMS)
    n_trials = a.n_trials or (FAST_N_TRIALS if a.fast else N_TRIALS)
    out = start_run('feature_selection', a.distribution, fast=a.fast, label=a.label)
    meta = build_metadata('feature_selection', vars(a), run_id=out.name)
    run_feature_selection(out, a.distribution, a.stats, n_sims, n_trials, a.scorer, a.seed, meta)


if __name__ == '__main__':
    main()
