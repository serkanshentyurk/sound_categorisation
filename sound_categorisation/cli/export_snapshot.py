#!/usr/bin/env python3
"""
Export the experiment snapshot: read every session's CSVs (the raw data, $BEHAV_DATA_DIR/Raw) and save
the pickled experiment the analyses load, at <data root>/behaviour/snapshots/sound_cat_snapshot.pkl.
Run it on the cluster, where the raw data is.

    sc-export-snapshot
    sc-export-snapshot --check-only
    sc-export-snapshot --config other.yaml --output /some/path.pkl
"""

import argparse
import sys
from pathlib import Path


def _resolve_config_path(explicit_path=None) -> Path:
    """config.yaml at the repo root unless --config names another file."""
    if explicit_path:
        return Path(explicit_path)
    from sound_categorisation.data.paths import DEFAULT_CONFIG
    return DEFAULT_CONFIG


def main(argv=None):
    parser = argparse.ArgumentParser(description='Export experiment snapshot')
    parser.add_argument('--config', type=str, default=None,
                        help='config YAML (default: config.yaml at the repo root)')
    parser.add_argument('--output', type=str, default=None,
                        help='output path (default: <data root>/behaviour/snapshots/sound_cat_snapshot.pkl)')
    parser.add_argument('--check-only', action='store_true',
                        help='Compare existing snapshot against current data')
    args = parser.parse_args(argv)

    from sound_categorisation.data.snapshot import check_staleness, default_output_path, export_snapshot

    config_path = _resolve_config_path(args.config)
    output_path = Path(args.output) if args.output else default_output_path()

    if args.check_only:
        if not output_path.exists():
            print(f'No snapshot at {output_path}. Run without --check-only first.')
            sys.exit(1)
        check_staleness(output_path, config_path)
        return

    export_snapshot(config_path, output_path)


if __name__ == '__main__':
    main()
