"""
sc-daily-tracker — how each animal is doing, from the session files on ceph, without the snapshot or the cluster.

    sc-daily-tracker                  copy what is new from ceph, then write and open the page (last 14 days)
    sc-daily-tracker --days 7
    sc-daily-tracker --cohort opto1-cohort
    sc-daily-tracker --no-copy        rebuild the page from the local copy only: nothing is fetched from ceph
    sc-daily-tracker --dry-run        list what would be copied, copy nothing, write nothing
    sc-daily-tracker --no-open        write the page without opening it

To look at the last page without running anything, open results/daily_tracker/daily_tracker_latest.html.
The files come from $BEHAV_DATA_DIR/Raw if that folder is on this machine (a rig PC with BEHAV_DATA_DIR set
to the mapped drive), otherwise over ssh as $SWC_USER through the login node in $SWC_NODE (hpc-gw2 by
default), like sc-fetch. Only Trial_Summary and Session_Parameters files are copied, to <data root>/raw_mirror.
The page goes to <results root>/daily_tracker/daily_tracker_<date>.html (a rerun the same day replaces it),
with a copy in daily_tracker_latest.html for a bookmark. Every number is in config.yaml, block daily_tracker.
See DAILY_TRACKER.md.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import webbrowser
from datetime import date

from behav_utils.config.schema import load_cohorts, load_config

from sound_categorisation.behaviour.daily_tracker import compute_daily_tracker, load_mirror, load_tracker_settings
from sound_categorisation.data.paths import REPO_ROOT, daily_tracker_dir, raw_mirror_dir
from sound_categorisation.data.raw_mirror import local_copy_status, mirror_sessions
from sound_categorisation.reports.daily_tracker import write_daily_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='sc-daily-tracker', description=__doc__.split('\n\n')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--days', type=int, default=None, help='days shown (default: config.yaml, daily_tracker.days)')
    ap.add_argument('--cohort', default=None, help='only the animals of this config.yaml cohort')
    ap.add_argument('--no-copy', action='store_true',
                    help='build the page from the local copy only; nothing is fetched from ceph')
    ap.add_argument('--dry-run', action='store_true', help='list what would be copied; copy and write nothing')
    ap.add_argument('--no-open', action='store_true', help='write the page without opening it')
    args = ap.parse_args(argv)
    if args.no_copy and args.dry_run:
        raise SystemExit('--no-copy and --dry-run do not go together: --dry-run is about the copy.')

    config_path = REPO_ROOT / 'config.yaml'
    settings = load_tracker_settings(config_path)
    days = args.days or settings.days
    cohorts = load_cohorts(config_path)
    wanted = [*settings.expected_cohorts, *([args.cohort] if args.cohort else [])]
    unknown = [c for c in wanted if c not in cohorts]
    if unknown:
        raise SystemExit(f'unknown cohort(s) {unknown}; cohorts in config.yaml: {sorted(cohorts)}')
    expected = {a for c in wanted for a in cohorts[c]}
    animals = set(cohorts[args.cohort]) if args.cohort else None
    today = date.today()
    config = load_config(config_path)

    if args.no_copy:
        if not raw_mirror_dir().is_dir():
            raise SystemExit(f'There is no local copy in {raw_mirror_dir()} yet: run sc-daily-tracker once '
                             'without --no-copy.')
        shown, last_session = local_copy_status(config, days=days, today=today, animals=animals)
        print(f'not copying: building from the local copy in {raw_mirror_dir()}')
        source = f'the local copy in {raw_mirror_dir()} (not refreshed from ceph this time)'
        notice = ('Built from the local copy, not refreshed from ceph: sessions pushed since the last copy are '
                  'not here, and expected animals were not checked.')
        expected = set()
    else:
        try:
            result = mirror_sessions(config, days=days, summary_days=settings.summary_days, today=today,
                                     animals=animals, dry_run=args.dry_run, progress=print)
        except RuntimeError as e:
            raise SystemExit(str(e)) from None
        print(result.summary())
        if args.dry_run:
            for rel in result.copied[:40]:
                print('  copy', rel)
            if len(result.copied) > 40:
                print(f'  … and {len(result.copied) - 40} more')
            for rel in result.removed_folders[:20]:
                print('  remove', rel)
            return 0
        shown, last_session, source, notice = result.shown_animals, result.last_session, result.source, ''
    if not shown and not expected:
        print(f'No animal has a session from the last {days} days; nothing to report.')
        return 0

    print('loading the copied sessions …')
    experiment, notes = load_mirror(config_path, raw_mirror_dir())
    print('computing …')
    tracker = compute_daily_tracker(experiment, settings, today=today, days=days, notes=notes, expected=expected,
                                    last_seen=last_session, animals=animals)
    print('writing the page …')
    path = write_daily_report(tracker, daily_tracker_dir() / f'daily_tracker_{today.isoformat()}.html',
                              source=source, notice=notice)
    latest = path.with_name('daily_tracker_latest.html')
    shutil.copyfile(path, latest)
    print(f'report: {path}\n        (also {latest.name}, for a bookmark)')
    if not args.no_open:
        webbrowser.open(path.resolve().as_uri())
    return 0


if __name__ == '__main__':
    sys.exit(main())
