#!/usr/bin/env python3
"""Group ALM control report — one PDF, WT vs HET, one site per invocation.

Pages (swarm + rank p + min_p + n): within-phase, between-phase (ALM vs masking,
all trials), delta-of-deltas, ALM vs PPC-opto (all trials), with reaction_time +
reaction_time_jitter. ALM is Uniform-only. Output:
    {cohort}_group_uniform_alm_{site}_{trial_of_interest}_report.pdf
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts._report as R
from behav_utils.config.schema import load_cohorts

DIST = 'Uniform'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cohort', default=None)
    ap.add_argument('--site', default=None, choices=['uni', 'bi'])
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
    if a.limit:
        members = members[:a.limit]

    sba = {}
    for aid in members:
        sbt = R.collect_sessions_alm(exp.get_animal(aid), DIST, a.site)
        if sbt['alm']:
            sba[aid] = sbt
        else:
            print(f'{aid}: no {R.SITE_TYPE[a.site]} on {DIST}, skipped')
    if not sba:
        raise SystemExit('no eligible animals')
    fn = out / f"{a.cohort}_group_uniform_alm_{a.site}_{toi}_report.pdf"
    R.build_alm_group(sba, by_animal, DIST, a.site, toi, fn)
    print(f'wrote {fn.name}  (n={len(sba)} animals) in {out}')


if __name__ == '__main__':
    main()
