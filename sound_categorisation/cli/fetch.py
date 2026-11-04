"""
Copy files from your folder on ceph to the same place on this laptop (rsync over ssh).

    sc-fetch snapshot
    sc-fetch networks
    sc-fetch synthetic
    sc-fetch results model_identification/all-mice
    sc-fetch results model_identification/all-mice/2026-10-23_0951_64a0f89

On ceph the files are under $BEHAV_DATA_DIR/Processed; here they land under data_root() (results under
results_root()), in the same sub-folders, so the code finds them without any further setting. Needs
BEHAV_DATA_DIR (your ceph folder) and SWC_USER (your SWC username) set on this laptop. Goes through
the SWC gateway to a login node, so it asks for your password twice unless you have ssh keys.
--dry-run lists what would be copied.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from sound_categorisation.data.paths import SWC_GATEWAY, SWC_NODE, cluster_data_root, data_root, results_root
from sound_categorisation.data.snapshot import SNAPSHOT_FILENAME

# What can be fetched: name -> (where it lives, sub-path, is it one file). Add a line to add a target.
TARGETS = {
    'snapshot':  ('data', f'behaviour/snapshots/{SNAPSHOT_FILENAME}', True),
    'networks':  ('data', 'snpe_networks', False),
    'synthetic': ('data', 'synthetic_cohorts', False),
    'results':   ('results', '', False),
}


def _swc_user() -> str:
    user = os.environ.get('SWC_USER')
    if not user:
        raise SystemExit('SWC_USER is not set. Add your SWC username to ~/.zshrc:\n'
                         '    export SWC_USER=<your SWC username>\nSee START_HERE.md, "Your own data folder".')
    return user


def paths_for(target: str, sub: str = '') -> tuple[str, Path]:
    """(remote path on ceph, local destination) for a target."""
    where, rel, is_file = TARGETS[target]
    rel = '/'.join(x for x in (rel, sub.strip('/')) if x)
    remote_base = cluster_data_root() / ('results' if where == 'results' else '')
    local_base = results_root() if where == 'results' else data_root()
    remote, local = remote_base / rel, local_base / rel
    if is_file:
        return str(remote), local.parent
    return str(remote) + '/', local


def command(target: str, sub: str = '', dry_run: bool = False) -> list[str]:
    """The rsync command sc-fetch runs."""
    user = _swc_user()
    remote, local = paths_for(target, sub)
    return (['rsync', '-av'] + (['-n'] if dry_run else [])
            + ['-e', f'ssh -J {user}@{SWC_GATEWAY}', f'{user}@{SWC_NODE}:{remote}', f'{local}/'])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('target', choices=sorted(TARGETS))
    p.add_argument('path', nargs='?', default='',
                   help='for results: <report>/<cohort>[/<run id>], e.g. model_identification/all-mice')
    p.add_argument('--dry-run', action='store_true', help='list what would be copied, copy nothing')
    a = p.parse_args(argv)
    if a.target == 'results' and not a.path:
        p.error('results needs a path, e.g. sc-fetch results model_identification/all-mice')
    try:
        cmd = command(a.target, a.path, a.dry_run)
    except RuntimeError as e:                   # BEHAV_DATA_DIR not set
        print(e, file=sys.stderr)
        return 2
    Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
    print('>', ' '.join(cmd), flush=True)
    return subprocess.run(cmd).returncode


if __name__ == '__main__':
    sys.exit(main())
