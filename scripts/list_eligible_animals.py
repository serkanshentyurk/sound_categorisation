#!/usr/bin/env python3
"""
List animals that qualify for a given analysis.

Applies a SessionFilter preset to each animal and prints those with
enough qualifying sessions.  Used by SLURM scripts to dynamically
determine the array size instead of hardcoding animal lists.

Usage:
    # Which animals have expert uniform sessions?
    python scripts/list_eligible_animals.py --preset expert_uniform

    # Which animals have any Hard-A data?
    python scripts/list_eligible_animals.py --preset all_hard_a

    # Custom criteria (no preset needed)
    python scripts/list_eligible_animals.py --stage Full_Task_Cont \
        --distribution Uniform --min-accuracy 0.70 --min-sessions 3

    # Machine-readable output for SLURM (one line, space-separated)
    python scripts/list_eligible_animals.py --preset expert_uniform --format flat

    # Show session counts alongside IDs
    python scripts/list_eligible_animals.py --preset expert_uniform --format table

Output formats:
    list  (default) — one ID per line, suitable for $(command) capture
    flat  — space-separated on one line, suitable for bash arrays
    table — ID + session count, human-readable
    json  — JSON array, machine-readable
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.config import load_project_config, list_animal_ids, load_animal_data


def main():
    parser = argparse.ArgumentParser(
        description='List animals qualifying for analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Registered presets: use --list-presets to see all available.',
    )

    # Selection mode: preset OR ad-hoc criteria
    parser.add_argument('--preset', type=str, default=None,
                        help='Session filter preset name (e.g. expert_uniform)')
    parser.add_argument('--stage', type=str, default=None,
                        help='Task stage (ad-hoc, ignored if --preset given)')
    parser.add_argument('--distribution', type=str, default=None,
                        help='Distribution (ad-hoc, ignored if --preset given)')
    parser.add_argument('--min-accuracy', type=float, default=None,
                        help='Minimum accuracy (ad-hoc)')
    parser.add_argument('--last-fraction', type=float, default=None,
                        help='Last fraction of sessions (ad-hoc)')

    # Eligibility threshold
    parser.add_argument('--min-sessions', type=int, default=3,
                        help='Minimum qualifying sessions to be eligible (default: 3)')

    # Output control
    parser.add_argument('--format', choices=['list', 'flat', 'table', 'json'],
                        default='list', help='Output format (default: list)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML')

    # Utility
    parser.add_argument('--list-presets', action='store_true',
                        help='Print registered presets and exit')
    parser.add_argument('--animals', type=str, default=None,
                        help='Comma-separated subset of animals to check '
                             '(default: all in data directory)')

    args = parser.parse_args()

    # Import selection after sys.path is set
    from behav_utils.data.selection import (
        select_sessions, SessionFilter, list_presets, get_preset,
    )

    if args.list_presets:
        presets = list_presets()
        for name, desc in sorted(presets.items()):
            print(f'  {name:25s} {desc}')
        return

    # Build the filter
    if args.preset:
        filt = get_preset(args.preset)
        # Allow ad-hoc overrides on top of preset
        overrides = {}
        if args.min_accuracy is not None:
            overrides['min_accuracy'] = args.min_accuracy
        if filt and overrides:
            filt = filt.with_overrides(**overrides)
        filter_desc = f'preset={args.preset}'
        if overrides:
            filter_desc += f' + overrides={overrides}'
    elif args.stage or args.distribution:
        kwargs = {}
        if args.stage:
            kwargs['stage'] = args.stage
        if args.distribution:
            kwargs['distribution'] = args.distribution
        if args.min_accuracy is not None:
            kwargs['min_accuracy'] = args.min_accuracy
        if args.last_fraction is not None:
            kwargs['last_fraction'] = args.last_fraction
        filt = SessionFilter(**kwargs)
        filter_desc = filt.describe()
    else:
        parser.error('Specify --preset or --stage/--distribution criteria')
        return

    # Get candidate animal IDs
    if args.animals:
        candidate_ids = args.animals.split(',')
    else:
        candidate_ids = list_animal_ids(config_path=args.config)

    if not candidate_ids:
        print('No animals found in data directory.', file=sys.stderr)
        sys.exit(1)

    # Apply filter to each animal
    eligible = []
    for aid in sorted(candidate_ids):
        try:
            animal = load_animal_data(aid, config_path=args.config)
            sessions = filt.apply(animal)
            n = len(sessions)
            if n >= args.min_sessions:
                eligible.append({'id': aid, 'n_sessions': n})
        except Exception as e:
            print(f'  {aid}: error ({e})', file=sys.stderr)
            continue

    # Output
    if args.format == 'table':
        print(f'Filter: {filter_desc}')
        print(f'Min sessions: {args.min_sessions}')
        print(f'Eligible: {len(eligible)}/{len(candidate_ids)}')
        print()
        for a in eligible:
            print(f'  {a["id"]:10s} {a["n_sessions"]:3d} sessions')
    elif args.format == 'flat':
        print(' '.join(a['id'] for a in eligible))
    elif args.format == 'json':
        print(json.dumps([a['id'] for a in eligible]))
    else:  # list
        for a in eligible:
            print(a['id'])

    # Exit code: 0 if any eligible, 1 if none
    if not eligible:
        sys.exit(1)


if __name__ == '__main__':
    main()
