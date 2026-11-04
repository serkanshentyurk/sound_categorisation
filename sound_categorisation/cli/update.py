"""
Bring this checkout up to date: ``git pull``, then ``pip install -e .``, then print what is installed.

    sc-update
    sc-update --keep-behav-utils

``--keep-behav-utils`` is for working on ``behav_utils`` next to this repository (an editable install):
it installs the project and every other dependency, but leaves your ``behav_utils`` as it is. Without
it, pip installs the ``behav_utils`` tag pinned in pyproject.toml. The pull is fast-forward only: if your
checkout has commits that are not on GitHub, it stops and you sort that out by hand.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import subprocess
import sys

from sound_categorisation.data.paths import REPO_ROOT

_DIST = 'sound-categorisation'
_VERSIONS = ("import behav_utils, sound_categorisation; "
             "print('sound_categorisation', sound_categorisation.__version__, '| behav_utils', behav_utils.__version__)")


def other_requirements() -> list[str]:
    """The project's dependencies (pyproject.toml, as installed) except behav_utils and the extras."""
    out = []
    for r in importlib.metadata.requires(_DIST) or []:
        if 'extra ==' in r:
            continue
        name = re.split(r'[\s<>=!~;@\[]', r.strip(), maxsplit=1)[0].lower().replace('_', '-')
        if name != 'behav-utils':
            out.append(r.split(';')[0].strip())
    return out


def _install_other_requirements() -> int:
    reqs = other_requirements()
    return subprocess.call([sys.executable, '-m', 'pip', 'install', *reqs]) if reqs else 0


def commands(keep_behav_utils: bool = False) -> list[list[str]]:
    """The commands sc-update runs, in order (all in the repo folder)."""
    pip = [sys.executable, '-m', 'pip', 'install', '-e', '.']
    steps = [['git', 'pull', '--ff-only']]
    if keep_behav_utils:
        steps += [pip + ['--no-deps'],
                  [sys.executable, '-c', 'from sound_categorisation.cli.update import _install_other_requirements as f; '
                                         'raise SystemExit(f())']]
    else:
        steps += [pip]
    return steps + [['git', 'log', '-1', '--format=commit %h  %cd  %s', '--date=short'],
                    [sys.executable, '-c', _VERSIONS]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--keep-behav-utils', action='store_true',
                   help='install the project and its other dependencies, leave behav_utils as it is')
    a = p.parse_args(argv)
    for cmd in commands(a.keep_behav_utils):
        shown = cmd if cmd[0] == 'git' else ['python'] + cmd[1:]
        print('>', ' '.join(shown) if len(' '.join(shown)) < 100 else 'python: install the other dependencies',
              flush=True)
        done = subprocess.run(cmd, cwd=REPO_ROOT)
        if done.returncode != 0:
            print(f'sc-update: stopped, the step above failed (exit {done.returncode}).', file=sys.stderr)
            return done.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
