#!/usr/bin/env python3
"""
Export experiment snapshot from CSV data.

Reads raw data from the path in the config YAML, saves the snapshot
to the Processed directory alongside it:
    .../Data/Raw/     ← config reads from here
    .../Data/Processed/snapshots/sound_cat_snapshot.pkl  ← saves here

Usage:
    python scripts/export_snapshot.py
    python scripts/export_snapshot.py --config config.yaml
    python scripts/export_snapshot.py --output /custom/path.pkl
    python scripts/export_snapshot.py --check-only
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description='Export experiment snapshot')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to project config YAML (auto-detects if omitted)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: derived from config data_dir)')
    parser.add_argument('--check-only', action='store_true',
                        help='Compare existing snapshot against current CSV data')
    args = parser.parse_args()

    from scripts.config import load_project_config
    from scripts.snapshot import export_snapshot, check_staleness, default_output_path

    # Resolve config (auto-detects cluster vs local)
    config = load_project_config(args.config)
    config_path = Path(args.config) if args.config else None

    # If no config_path was given, figure out which one was used
    if config_path is None:
        from scripts.config import DEFAULT_CONFIG, CLUSTER_CONFIG
        import socket
        hostname = socket.gethostname()
        if CLUSTER_CONFIG.exists() and any(
            x in hostname for x in ('hpc', 'gpu', 'enc')
        ):
            config_path = CLUSTER_CONFIG
        else:
            config_path = DEFAULT_CONFIG

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = default_output_path(config)

    if args.check_only:
        if not output_path.exists():
            print(f'No snapshot at {output_path}. Run without --check-only first.')
            sys.exit(1)
        check_staleness(output_path, config_path)
        return

    export_snapshot(config_path, output_path)


if __name__ == '__main__':
    main()
