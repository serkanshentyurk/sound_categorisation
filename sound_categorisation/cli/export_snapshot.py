#!/usr/bin/env python3
"""
Export experiment snapshot.

Usage:
    sc-export-snapshot                      # auto-detect config
    sc-export-snapshot --config config.yaml
    sc-export-snapshot --output /custom/path.pkl
    sc-export-snapshot --check-only
"""

import argparse
import sys
from pathlib import Path


def _resolve_config_path(explicit_path=None) -> Path:
    """Find the right config file."""
    if explicit_path:
        return Path(explicit_path)

    import socket

    from sound_categorisation.data.paths import CLUSTER_CONFIG, DEFAULT_CONFIG
    hostname = socket.gethostname()
    if CLUSTER_CONFIG.exists() and any(
        x in hostname for x in ('hpc', 'gpu', 'enc', 'sgw')
    ):
        return CLUSTER_CONFIG
    return DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description='Export experiment snapshot')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML (auto-detects if omitted)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: auto per machine)')
    parser.add_argument('--check-only', action='store_true',
                        help='Compare existing snapshot against current data')
    args = parser.parse_args()

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
