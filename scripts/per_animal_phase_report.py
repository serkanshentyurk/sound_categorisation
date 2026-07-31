#!/usr/bin/env python3
"""Per-animal PPC opto report — one PDF per animal.

Pages: psychometric, within-phase ({toi} vs non_opto), between-phase (opto vs
masking, all trials), delta-of-deltas. Output:
    {cohort}_{animal}_{distribution}_ppc_{trial_of_interest}_report.pdf

NOTE: the psychometric page and the full stat grid (with `psychometric`) are the
same calls the notebooks run; they were NOT timed end-to-end in the build
sandbox (curve bootstrap is slow there). Use --fast / --selftest for a quick
structural check before the full run.
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts._report as R
from behav_utils.config.schema import load_cohorts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cohort', default=None, help='cohort key in config.yaml cohorts:')
    ap.add_argument('--distribution', default='Uniform')
    ap.add_argument('--trial-of-interest', default='opto', choices=['opto', 'post_opto'])
    ap.add_argument('--limit', type=int, default=None, help='first N eligible animals only')
    ap.add_argument('--fast', action='store_true', help='scalar stats, few draws, no psych page')
    ap.add_argument('--selftest', action='store_true', help='synthetic PDFs, no data load')
    ap.add_argument('--snapshot', default=None, help='snapshot .pkl (else CSV via --config)')
    ap.add_argument('--config', default=None, help='config.yaml (default: repo root)')
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

    made = 0
    for aid in members:
        sbt = R.collect_sessions_ppc(exp.get_animal(aid), a.distribution)
        if not sbt['opto'] or not sbt['masking']:
            print(f'{aid}: missing opto/masking on {a.distribution}, skipped'); continue
        fn = out / f"{a.cohort}_{aid}_{a.distribution}_ppc_{toi}_report.pdf"
        R.build_ppc_per_animal(sbt, aid, by_animal.get(aid, 'unknown'), a.distribution, toi, fn)
        print('wrote', fn.name); made += 1
        if a.limit and made >= a.limit:
            break
    print(f'done: {made} PDF(s) in {out}')


if __name__ == '__main__':
    main()
