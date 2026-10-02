"""
Report CLI — ``sc-reports`` (also ``python -m sound_categorisation.reports``).

One subcommand per report type; every run writes under

    <results root>/<report>/<cohort>/<run_id>/        (see sound_categorisation.data.paths)

and points ``<report>/<cohort>/latest`` at itself.

    sc-reports opto-contrasts   --distribution Hard-A [--trial-class opto|post_opto] [--site ppc|alm_uni|alm_bi]
                                [--level animal|group|both] [--animals SS15 SS16] [--fast]
    sc-reports opto-contrasts   --all [--fast] [--limit N]    # PPC × 3 distributions × 2 trial classes + ALM uni/bi
    sc-reports switch-adaptation --cohort behaviour1-cohort
    sc-reports summary          [--run latest|<run_id>]      # summary pages from an opto-contrasts run
    sc-reports battery          [--limit N]                  # fast structure check → full opto-contrasts → summary

The synthetic end-to-end check of this pipeline is ``pytest tests/e2e -q`` (not a subcommand).

Common options: --cohort (default opto1-cohort), --snapshot PATH, --config PATH, --run-id ID
(reuse an existing run directory), --root DIR (override the results root), --fast (few draws,
scalar stats, no readouts; run id gets a ``_fast`` suffix).

Subcommand names are the report folder names (``opto_contrasts/``, ``switch_adaptation/``). Within a run,
``opto-contrasts`` writes ``<distribution>/<site>_<trial_class>/{<animal>/, group/, pdf/}``;
``switch-adaptation`` writes ``switches/``; ``summary`` writes ``summary.pdf`` and the generated README at the
run root. Tables + ``meta.json`` sit next to every PDF.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use('Agg')

from behav_utils.config.schema import load_cohorts

from sound_categorisation.data.cohort import (
    SITES,
    gather_genotypes,
    load_experiment_any,
    site_label,
    split_site,
)
from sound_categorisation.data.paths import REPO_ROOT, git_state, resolve_run, start_run
from sound_categorisation.reports.compute import Settings, compute_animal, compute_group
from sound_categorisation.reports.pdf import animal_pdf, group_pdf
from sound_categorisation.reports.tables import group_tables, readout_arrays, to_tables, write_result

DISTRIBUTIONS = ('Uniform', 'Hard-A', 'Hard-B')
TRIAL_CLASSES = ('opto', 'post_opto')
DEFAULT_COHORT = 'opto1-cohort'


# ── run bookkeeping ──────────────────────────────────────────────────────────

def _open_run(report: str, a) -> Path:
    """Create or reopen this command's run directory and remember it on ``a``."""
    run = start_run(report, a.cohort, getattr(a, 'run_id', None), fast=getattr(a, 'fast', False),
                    root=getattr(a, 'root', None))
    a.run_path, a.run_id = run, run.name
    return run


def _condition_dir(run: Path, distribution: str, design: str, site: str | None, trial_class: str) -> Path:
    return run / distribution / f'{site_label(design, site)}_{trial_class}'


def _meta(a, **extra) -> dict:
    g = git_state()
    return {'run_id': a.run_id, 'argv': list(sys.argv), 'git_sha': g['sha'], 'git_dirty': g['dirty'],
            'snapshot': str(a.snapshot) if a.snapshot else None, 'config': str(a.config) if a.config else None,
            'cohort': a.cohort, 'fast': a.fast, **extra}


def _animals(experiment, cohorts: Dict[str, List[str]], a) -> List[str]:
    ids = list(a.animals) if getattr(a, 'animals', None) else list(cohorts.get(a.cohort, []))
    ids = [i for i in ids if i in experiment.animals]
    if getattr(a, 'limit', None):
        ids = ids[:a.limit]
    return ids


def _load(a):
    experiment = load_experiment_any(a.config, a.snapshot)
    cohorts = load_cohorts(a.config or REPO_ROOT / 'config.yaml')
    settings = Settings.fast() if a.fast else Settings()
    return experiment, _animals(experiment, cohorts, a), settings


# ── the opto report ──────────────────────────────────────────────────────────

def run_animal(experiment, ids, distribution, trial_class, design, site, a, settings) -> Dict[str, object]:
    out = _condition_dir(a.run_path, distribution, design, site, trial_class)
    (out / 'pdf').mkdir(parents=True, exist_ok=True)
    by_animal, _ = gather_genotypes(experiment)
    results = {}
    for aid in ids:
        t0 = time.time()
        r = compute_animal(experiment, aid, distribution, trial_class, design=design, site=site, cohort=a.cohort,
                           settings=settings, genotype=by_animal.get(aid, 'unknown'))
        write_result(out / aid, to_tables(r), readout_arrays(r), _meta(a, animal=aid, distribution=distribution,
                                                                     site=site_label(design, site), trial_class=trial_class,
                                                                     settings=settings.to_dict()))
        animal_pdf(r, out / 'pdf' / f'{aid}_{site_label(design, site)}_{trial_class}.pdf', settings)
        results[aid] = r
        print(f'  {aid}: {time.time() - t0:.0f}s')
    return results


def run_group(experiment, ids, distribution, trial_class, design, site, a, settings, per_animal=None):
    out = _condition_dir(a.run_path, distribution, design, site, trial_class)
    (out / 'pdf').mkdir(parents=True, exist_ok=True)
    g = compute_group(experiment, ids, distribution, trial_class, design=design, site=site, cohort=a.cohort, settings=settings)
    write_result(out / 'group', group_tables(g), None, _meta(a, distribution=distribution, site=site_label(design, site),
                                                              trial_class=trial_class, animals=ids, settings=settings.to_dict()))
    group_pdf(g, per_animal or {}, out / 'pdf' / f'group_{site_label(design, site)}_{trial_class}.pdf', settings)
    return g


def _run_condition(experiment, ids, distribution, trial_class, design, site, a, settings, level: str):
    per = run_animal(experiment, ids, distribution, trial_class, design, site, a, settings) if level != 'group' else {}
    if level != 'animal':
        run_group(experiment, ids, distribution, trial_class, design, site, a, settings, per)


def cmd_opto_contrasts(a):
    experiment, ids, settings = _load(a)
    run = _open_run('opto_contrasts', a)
    print(f'opto_contrasts/{a.cohort}/{a.run_id}: {len(ids)} animals -> {run}')
    if a.all:
        jobs = [('ppc', None, d, tc) for tc in TRIAL_CLASSES for d in DISTRIBUTIONS]
        jobs += [('alm', sub, 'Uniform', tc) for tc in TRIAL_CLASSES for sub in ('uni', 'bi')]
    else:
        if not a.distribution:
            sys.exit('--distribution is required (or pass --all)')
        jobs = [(*split_site(a.site), a.distribution, a.trial_class)]
    failed = []
    for design, site, dist, trial_class in jobs:
        print(f'>>> {design} {site or ""} {dist} {trial_class}')
        try:
            _run_condition(experiment, ids, dist, trial_class, design, site, a, settings, a.level)
        except Exception as exc:                      # one failure must not stop the batch
            if not a.all:
                raise
            failed.append((design, site, dist, trial_class, repr(exc)))
            print(f'!! FAILED {design} {site} {dist} {trial_class}: {exc!r}')
    if failed:
        print('\nfailed jobs:')
        for f in failed:
            print('  ', f)
        sys.exit(1)


# ── the switch-adaptation report ─────────────────────────────────────────────

def cmd_switch_adaptation(a):
    from sound_categorisation.reports.switches import run_switches
    experiment, ids, _ = _load(a)
    run = _open_run('switch_adaptation', a)
    print(f'switch_adaptation/{a.cohort}/{a.run_id}: {len(ids)} animals -> {run}')
    run_switches(experiment, ids, run, a.cohort, meta=_meta(a))


# ── summary + battery ────────────────────────────────────────────────────────

def cmd_summary(a):
    from sound_categorisation.reports.summary import write_summary
    run = resolve_run('opto_contrasts', a.cohort, a.run, root=a.root)
    print('summary ->', write_summary(run, a.cohort))


def cmd_battery(a):
    """The overnight sequence: a fast one-animal structure check, then the full opto battery, then
    the summary. Each stage is its own run; the full run is what `latest` points at afterwards."""
    base = dict(cohort=a.cohort, snapshot=a.snapshot, config=a.config, root=a.root, animals=None, run_id=None)
    check = argparse.Namespace(**base, all=True, fast=True, limit=1, level='both', distribution=None,
                               trial_class='opto', site='ppc')
    cmd_opto_contrasts(check)
    full = argparse.Namespace(**base, all=True, fast=False, limit=a.limit, level='both', distribution=None,
                              trial_class='opto', site='ppc')
    cmd_opto_contrasts(full)
    cmd_summary(argparse.Namespace(cohort=a.cohort, run=full.run_id, root=a.root))


# ── parser ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='sc-reports', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp, *, selection=True):
        sp.add_argument('--cohort', default=DEFAULT_COHORT)
        sp.add_argument('--snapshot', type=Path, default=None)
        sp.add_argument('--config', type=Path, default=None)
        sp.add_argument('--root', type=Path, default=None, help='results root (default: paths.results_root())')
        sp.add_argument('--run-id', default=None, help='write into this existing run id instead of a new one')
        if selection:
            sp.add_argument('--fast', action='store_true', help='few draws, scalar stats, no readouts')
            sp.add_argument('--limit', type=int, default=None, help='first N animals only')
            sp.add_argument('--animals', nargs='*', default=None)

    so = sub.add_parser('opto-contrasts', help='opto contrasts: per-animal tables + PDFs and the WT-vs-HET fold')
    common(so)
    so.add_argument('--distribution', default=None, choices=DISTRIBUTIONS)
    so.add_argument('--trial-class', default='opto', choices=TRIAL_CLASSES,
                    help='which laser-related trials are contrasted against laser-off')
    so.add_argument('--site', default='ppc', choices=SITES, help='inactivation site of the sessions')
    so.add_argument('--level', default='both', choices=('animal', 'group', 'both'))
    so.add_argument('--all', action='store_true', help='every site × distribution × trial class (the battery)')
    so.set_defaults(fn=cmd_opto_contrasts)

    sw = sub.add_parser('switch-adaptation', help='switch adaptation across blocks (pre-opto cohorts)')
    common(sw)
    sw.set_defaults(fn=cmd_switch_adaptation)

    ss = sub.add_parser('summary', help='summary pages from the tables of an opto-contrasts run')
    ss.add_argument('--cohort', default=DEFAULT_COHORT)
    ss.add_argument('--run', default='latest', help="run id, or 'latest'")
    ss.add_argument('--root', type=Path, default=None)
    ss.set_defaults(fn=cmd_summary)

    sb = sub.add_parser('battery', help='fast check → full opto-contrasts battery → summary (the overnight run)')
    common(sb, selection=False)
    sb.add_argument('--limit', type=int, default=None)
    sb.set_defaults(fn=cmd_battery)

    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
