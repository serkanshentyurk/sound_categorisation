"""
Report CLI.

    python -m sound_categorisation.reports animal   --distribution Hard-A --toi opto [--design alm --site uni] [--animals SS15 SS16]
    python -m sound_categorisation.reports group    --distribution Hard-A --toi opto [--design alm --site uni]
    python -m sound_categorisation.reports all      [--fast] [--limit N]
    python -m sound_categorisation.reports selftest [--out DIR]

Common options: --cohort (default opto1-cohort), --snapshot PATH, --config PATH, --out DIR
(default <repo>/results/reports), --fast (few draws, scalar stats, no readouts).

Every run writes tables + metadata next to the PDFs; ``group`` also writes the
per-animal tables it folded, so a group run alone is enough for the notebooks.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use('Agg')

from behav_utils.config.schema import load_cohorts

from sound_categorisation.cohort import gather_genotypes, load_experiment_any
from sound_categorisation.paths import REPO_ROOT
from sound_categorisation.reports.compute import Settings, compute_animal, compute_group
from sound_categorisation.reports.pdf import animal_pdf, group_pdf
from sound_categorisation.reports.tables import group_tables, readout_arrays, to_tables, write_result

DISTRIBUTIONS = ('Uniform', 'Hard-A', 'Hard-B')
TOIS = ('opto', 'post_opto')
DEFAULT_COHORT = 'opto1-cohort'


def _run_dir(out: Path, cohort: str, distribution: str, design: str, site: Optional[str], toi: str) -> Path:
    d = f'{design}_{site}_{toi}' if site else f'{design}_{toi}'
    return out / cohort / distribution / d


def _meta(a, **extra) -> dict:
    return {'snapshot': str(a.snapshot) if a.snapshot else None, 'config': str(a.config) if a.config else None,
            'cohort': a.cohort, 'fast': a.fast, **extra}


def _animals(experiment, cohorts: Dict[str, List[str]], a) -> List[str]:
    ids = list(a.animals) if getattr(a, 'animals', None) else list(cohorts.get(a.cohort, []))
    ids = [i for i in ids if i in experiment.animals]
    if getattr(a, 'limit', None):
        ids = ids[:a.limit]
    return ids


def run_animal(experiment, ids, distribution, toi, design, site, a, settings) -> Dict[str, object]:
    out = _run_dir(Path(a.out), a.cohort, distribution, design, site, toi)
    (out / 'pdf').mkdir(parents=True, exist_ok=True)
    by_animal, _ = gather_genotypes(experiment)
    results = {}
    for aid in ids:
        t0 = time.time()
        r = compute_animal(experiment, aid, distribution, toi, design=design, site=site, cohort=a.cohort,
                           settings=settings, genotype=by_animal.get(aid, 'unknown'))
        write_result(out / aid, to_tables(r), readout_arrays(r), _meta(a, animal=aid, distribution=distribution,
                                                                     design=design, site=site, toi=toi,
                                                                     settings=settings.to_dict()))
        animal_pdf(r, out / 'pdf' / f'{aid}_{design}{"_" + site if site else ""}_{toi}.pdf', settings)
        results[aid] = r
        print(f'  {aid}: {time.time() - t0:.0f}s')
    return results


def run_group(experiment, ids, distribution, toi, design, site, a, settings, per_animal=None):
    out = _run_dir(Path(a.out), a.cohort, distribution, design, site, toi)
    (out / 'pdf').mkdir(parents=True, exist_ok=True)
    g = compute_group(experiment, ids, distribution, toi, design=design, site=site, cohort=a.cohort, settings=settings)
    write_result(out / 'group', group_tables(g), None, _meta(a, distribution=distribution, design=design, site=site,
                                                              toi=toi, animals=ids, settings=settings.to_dict()))
    group_pdf(g, per_animal or {}, out / 'pdf' / f'group_{design}{"_" + site if site else ""}_{toi}.pdf', settings)
    return g


def cmd_animal(a):
    experiment = load_experiment_any(a.config, a.snapshot)
    cohorts = load_cohorts(a.config or REPO_ROOT / 'config.yaml')
    settings = Settings.fast() if a.fast else Settings()
    ids = _animals(experiment, cohorts, a)
    print(f'{a.design} {a.distribution} {a.toi}: {len(ids)} animals')
    run_animal(experiment, ids, a.distribution, a.toi, a.design, a.site, a, settings)


def cmd_group(a):
    experiment = load_experiment_any(a.config, a.snapshot)
    cohorts = load_cohorts(a.config or REPO_ROOT / 'config.yaml')
    settings = Settings.fast() if a.fast else Settings()
    ids = _animals(experiment, cohorts, a)
    per_animal = run_animal(experiment, ids, a.distribution, a.toi, a.design, a.site, a, settings) \
        if a.with_animals else {}
    run_group(experiment, ids, a.distribution, a.toi, a.design, a.site, a, settings, per_animal)


def cmd_all(a):
    """The overnight battery: PPC × 3 distributions × 2 tois (+ ALM uni/bi on Uniform), animals then group."""
    experiment = load_experiment_any(a.config, a.snapshot)
    cohorts = load_cohorts(a.config or REPO_ROOT / 'config.yaml')
    settings = Settings.fast() if a.fast else Settings()
    ids = _animals(experiment, cohorts, a)
    jobs = [('ppc', None, d, toi) for toi in TOIS for d in DISTRIBUTIONS]
    jobs += [('alm', site, 'Uniform', toi) for toi in TOIS for site in ('uni', 'bi')]
    failed = []
    for design, site, dist, toi in jobs:
        print(f'>>> {design} {site or ""} {dist} {toi}')
        try:
            per = run_animal(experiment, ids, dist, toi, design, site, a, settings)
            run_group(experiment, ids, dist, toi, design, site, a, settings, per)
        except Exception as exc:                      # one failure must not stop the batch
            failed.append((design, site, dist, toi, repr(exc)))
            print(f'!! FAILED {design} {site} {dist} {toi}: {exc!r}')
    if failed:
        print('\nfailed jobs:')
        for f in failed:
            print('  ', f)
        sys.exit(1)


def cmd_selftest(a):
    from sound_categorisation.reports.selftest import run_selftest
    run_selftest(Path(a.out) / 'selftest')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='python -m sound_categorisation.reports', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp, needs_target=True):
        sp.add_argument('--cohort', default=DEFAULT_COHORT)
        sp.add_argument('--snapshot', type=Path, default=None)
        sp.add_argument('--config', type=Path, default=None)
        sp.add_argument('--out', type=Path, default=REPO_ROOT / 'results' / 'reports')
        sp.add_argument('--fast', action='store_true', help='few draws, scalar stats, no readouts')
        sp.add_argument('--limit', type=int, default=None, help='first N animals only')
        sp.add_argument('--animals', nargs='*', default=None)
        if needs_target:
            sp.add_argument('--distribution', required=True, choices=DISTRIBUTIONS)
            sp.add_argument('--toi', default='opto', choices=TOIS)
            sp.add_argument('--design', default='ppc', choices=('ppc', 'alm'))
            sp.add_argument('--site', default=None, choices=('uni', 'bi'))

    sa = sub.add_parser('animal', help='per-animal tables + PDFs'); common(sa); sa.set_defaults(fn=cmd_animal)
    sg = sub.add_parser('group', help='WT-vs-HET fold'); common(sg)
    sg.add_argument('--with-animals', action='store_true', help='also build the per-animal results (needed for '
                    'the group psychometric/UM pages)')
    sg.set_defaults(fn=cmd_group)
    sl = sub.add_parser('all', help='the full battery'); common(sl, needs_target=False); sl.set_defaults(fn=cmd_all)
    st = sub.add_parser('selftest', help='synthetic end-to-end check')
    st.add_argument('--out', type=Path, default=REPO_ROOT / 'results' / 'reports'); st.set_defaults(fn=cmd_selftest)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if getattr(a, 'design', 'ppc') == 'alm' and getattr(a, 'site', None) is None:
        sys.exit('--design alm needs --site uni|bi')
    a.fn(a)


if __name__ == '__main__':
    main()
