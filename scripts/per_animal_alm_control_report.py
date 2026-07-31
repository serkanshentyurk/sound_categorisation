#!/usr/bin/env python3
"""Per-animal ALM control report — one PDF per animal, one site per invocation.

Pages: psychometric, within-phase ({toi} vs non_opto), between-phase (ALM vs
masking, all trials), delta-of-deltas, ALM vs PPC-opto (all trials). Adds
reaction_time + reaction_time_jitter to every grid. ALM is Uniform-only. Output:
    {cohort}_{animal}_uniform_alm_{site}_{trial_of_interest}_report.pdf
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts._report as R
from behav_utils.config.schema import load_cohorts

DIST = 'Uniform'   # ALM controls are run on Uniform only


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cohort', default=None)
    ap.add_argument('--site', default=None, choices=['uni', 'bi'], help='ALM site (one per run)')
    ap.add_argument('--trial-of-interest', default='opto', choices=['opto', 'post_opto'])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--snapshot', default=None)
    ap.add_argument('--config', default=None)
    ap.add_argument('--out', default=str(R._ROOT / 'reports'),
                    help='output dir (default: <repo>/reports)')
    a = ap.parse_args()

    if a.selftest:
        R.run_selftest(a.out); return
    if not a.cohort:
        ap.error('--cohort is required')
    if not a.site:
        ap.error('--site is required (uni|bi)')
    if a.fast:
        R.configure(fast=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cohorts = load_cohorts(a.config or (R._ROOT / 'config.yaml'))
    if a.cohort not in cohorts:
        raise SystemExit(f"cohort {a.cohort!r} not found; have {list(cohorts)}")
    exp = R.load_any(a.config, a.snapshot)
    by_animal, _ = R.gather_genotypes(exp)
    toi = a.trial_of_interest
    members = [aid for aid in cohorts[a.cohort] if aid in exp.animals]

    made = 0
    for aid in members:
        sbt = R.collect_sessions_alm(exp.get_animal(aid), DIST, a.site)
        if not sbt['alm']:
            print(f'{aid}: no {R.SITE_TYPE[a.site]} on {DIST}, skipped'); continue
        fn = out / f"{a.cohort}_{aid}_uniform_alm_{a.site}_{toi}_report.pdf"
        R.build_alm_per_animal(sbt, aid, by_animal.get(aid, 'unknown'), DIST, a.site, toi, fn)
        print('wrote', fn.name); made += 1
        if a.limit and made >= a.limit:
            break
    print(f'done: {made} PDF(s) in {out}')


if __name__ == '__main__':
    main()
