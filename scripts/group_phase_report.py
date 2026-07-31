#!/usr/bin/env python3
"""Group PPC opto report — one PDF, WT vs HET across the cohort.

Pages (swarm of per-animal deltas + rank p + min_p + n): within-phase, between-
phase (opto vs masking, all trials), delta-of-deltas. Output:
    {cohort}_group_{distribution}_ppc_{trial_of_interest}_report.pdf
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts._report as R
from behav_utils.config.schema import load_cohorts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cohort', default=None)
    ap.add_argument('--distribution', default='Uniform')
    ap.add_argument('--trial-of-interest', default='opto', choices=['opto', 'post_opto'])
    ap.add_argument('--limit', type=int, default=None, help='first N eligible animals only')
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
        sbt = R.collect_sessions_ppc(exp.get_animal(aid), a.distribution)
        if sbt['opto'] and sbt['masking']:
            sba[aid] = sbt
        else:
            print(f'{aid}: missing opto/masking on {a.distribution}, skipped')
    if not sba:
        raise SystemExit('no eligible animals')
    fn = out / f"{a.cohort}_group_{a.distribution}_ppc_{toi}_report.pdf"
    R.build_ppc_group(sba, by_animal, a.distribution, toi, fn)
    print(f'wrote {fn.name}  (n={len(sba)} animals) in {out}')


if __name__ == '__main__':
    main()
