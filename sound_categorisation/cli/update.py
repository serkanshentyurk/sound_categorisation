"""
Bring this checkout up to date: ``git pull``, then ``pip install -e .``, then print what is installed.

    sc-update
    sc-update --no-deps

``--no-deps`` leaves the dependencies alone, e.g. an editable ``behav_utils`` you develop next to this
repo; without it pip installs the ``behav_utils`` tag pinned in pyproject.toml. The pull is fast-forward
only: if your checkout has commits that are not on GitHub, it stops and you sort that out by hand.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from sound_categorisation.data.paths import REPO_ROOT

_VERSIONS = ("import behav_utils, sound_categorisation; "
             "print('sound_categorisation', sound_categorisation.__version__, '| behav_utils', behav_utils.__version__)")


def commands(no_deps: bool = False) -> list[list[str]]:
    """The commands sc-update runs, in order (all in the repo folder)."""
    pip = [sys.executable, '-m', 'pip', 'install', '-e', '.'] + (['--no-deps'] if no_deps else [])
    return [['git', 'pull', '--ff-only'], pip, ['git', 'log', '-1', '--format=commit %h  %cd  %s', '--date=short'],
            [sys.executable, '-c', _VERSIONS]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--no-deps', action='store_true', help='leave the dependencies (e.g. behav_utils) as they are')
    a = p.parse_args(argv)
    for cmd in commands(a.no_deps):
        print('>', ' '.join(cmd if cmd[0] == 'git' else ['python'] + cmd[1:]), flush=True)
        done = subprocess.run(cmd, cwd=REPO_ROOT)
        if done.returncode != 0:
            print(f'sc-update: stopped, the step above failed (exit {done.returncode}).', file=sys.stderr)
            return done.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
