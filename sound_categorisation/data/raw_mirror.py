"""
The daily tracker's local copy of the raw session files: only the two small files it reads.

    from sound_categorisation.data.raw_mirror import mirror_sessions
    result = mirror_sessions(config, days=14, summary_days=None)
    print(result.summary())

Which sessions: every animal with a session folder dated in the last ``days`` days is shown, and for those
animals every session within ``summary_days`` (None: the whole history) is copied, because a condition's
last sessions can be weeks back. Only ``Trial_Summary*.csv`` and ``Session_Parameters*.csv`` are copied;
videos, lick files and logs never leave ceph.

Two routes to the same copy (data.paths.raw_source):
    local   $BEHAV_DATA_DIR/Raw is on this machine (cluster, rig PC with the mapped drive): shutil.copy2
    ssh     a laptop: one remote `find` lists the files with size and time, one remote `tar` sends the ones
            needed; through the SWC gateway as $SWC_USER, one password prompt pair per run without keys

The copies keep the server's timestamps, so a file is fetched again only when its size or time (to the
second) differs. Folders that are no longer selected are deleted from the copy.
"""

from __future__ import annotations

import csv
import fnmatch
import os
import re
import shlex
import shutil
import subprocess
import tarfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path, PurePosixPath

from sound_categorisation.data.paths import SWC_GATEWAY, raw_mirror_dir, raw_source

FILE_PATTERNS = ('Trial_Summary*.csv', 'Session_Parameters*.csv')


@dataclass(frozen=True)
class SourceFile:
    relpath: str          # '<animal>/<session folder>/<file>', '/'-separated
    size: int
    mtime: float

    @property
    def folder(self) -> tuple:
        p = PurePosixPath(self.relpath).parts
        return p[0], p[1]


@dataclass(frozen=True)
class MirrorResult:
    route: str                  # 'local' or 'ssh'
    source: str
    dest: Path
    shown_animals: tuple        # animals with a session in the daily window
    n_folders: int              # session folders selected
    copied: tuple               # relpaths fetched this run (or that would be, with dry_run)
    unchanged: int
    removed_folders: tuple      # '<animal>/<folder>' deleted from the copy
    dry_run: bool
    last_session: dict = field(default_factory=dict)   # every listed animal: its newest folder date

    def summary(self) -> str:
        verb = 'would copy' if self.dry_run else 'copied'
        return (f'{self.route} {self.source}: {len(self.shown_animals)} animals, {self.n_folders} session folders; '
                f'{verb} {len(self.copied)} files, {self.unchanged} unchanged, '
                f'{len(self.removed_folders)} old folders {"to remove" if self.dry_run else "removed"}')


# ── selection ────────────────────────────────────────────────────────────────

def _wanted(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in FILE_PATTERNS)


def select_folders(folders, *, today: date, days: int, summary_days: int | None,
                   animals=None) -> tuple:
    """(shown animals, selected (animal, folder) pairs) from (animal, folder, date) triples.

    Shown: an animal with a folder dated in [today - days + 1, today] (and in ``animals`` if given).
    Selected: every folder of a shown animal, or those within ``summary_days`` of today.
    """
    start = today - timedelta(days=days - 1)
    since = None if summary_days is None else today - timedelta(days=summary_days - 1)
    shown = sorted({a for a, _, d in folders
                    if start <= d <= today and (animals is None or a in animals)})
    keep = {(a, f) for a, f, d in folders
            if a in shown and d <= today and (since is None or d >= since)}
    return tuple(shown), keep


def folder_date(name: str, regex: str) -> date | None:
    """A session folder's date from its name, as behav_utils' csv_folders reader reads it: the regex's
    groups ``year``, ``month``, ``day`` (or its first three groups); None if it does not match."""
    match = re.search(regex, PurePosixPath(str(name)).name)
    if not match:
        return None
    groups = match.groupdict()
    try:
        if groups.get('year') is not None:
            return date(int(groups['year']), int(groups['month']), int(groups['day']))
        return date(*(int(g) for g in match.groups()[:3]))
    except (ValueError, TypeError, IndexError, KeyError):
        return None


def _dated(pairs, regex: str) -> list:
    out = []
    for a, f in pairs:
        d = folder_date(f, regex)
        if d is not None:
            out.append((a, f, d))
    return out


def _needs_copy(f: SourceFile, dest: Path) -> bool:
    local = dest / f.relpath
    try:
        st = local.stat()
    except FileNotFoundError:
        return True
    return st.st_size != f.size or int(st.st_mtime) != int(f.mtime)


def _prune(dest: Path, keep: set, dry_run: bool, animals=None) -> tuple:
    """Delete copied folders that are no longer selected; with ``animals``, only among those animals."""
    removed = []
    if not dest.is_dir():
        return ()
    for animal_dir in sorted(p for p in dest.iterdir() if p.is_dir()):
        if animals is not None and animal_dir.name not in animals:
            continue
        for folder in sorted(p for p in animal_dir.iterdir() if p.is_dir()):
            if (animal_dir.name, folder.name) not in keep:
                removed.append(f'{animal_dir.name}/{folder.name}')
                if not dry_run:
                    shutil.rmtree(folder)
        if not dry_run and not any(animal_dir.iterdir()):
            animal_dir.rmdir()
    return tuple(removed)


# ── local route ──────────────────────────────────────────────────────────────

def _local_folders(raw: Path) -> list:
    pairs = []
    with os.scandir(raw) as animals:
        for a in animals:
            if a.is_dir():
                with os.scandir(a.path) as folders:
                    pairs += [(a.name, f.name) for f in folders if f.is_dir()]
    return pairs


def _local_files(raw: Path, keep: set) -> list:
    files = []
    for a, f in sorted(keep):
        with os.scandir(raw / a / f) as entries:
            for e in entries:
                if e.is_file() and _wanted(e.name):
                    st = e.stat()
                    files.append(SourceFile(f'{a}/{f}/{e.name}', st.st_size, st.st_mtime))
    return files


def _copy_local(raw: Path, dest: Path, files: list) -> None:
    for f in files:
        target = dest / f.relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + '.partial')
        shutil.copy2(raw / f.relpath, tmp)
        os.replace(tmp, target)


# ── ssh route ────────────────────────────────────────────────────────────────

def _ssh_command(host: str, remote_command: str) -> list:
    """The ssh call for one remote command. Shares one connection between calls where ssh can (not on
    Windows), so a run without keys asks for the password once per hop, not once per call."""
    user = host.split('@')[0]
    cmd = ['ssh', '-J', f'{user}@{SWC_GATEWAY}']
    if os.name != 'nt':
        cmd += ['-o', 'ControlMaster=auto', '-o', 'ControlPath=~/.ssh/sc-%C', '-o', 'ControlPersist=120']
    return cmd + [host, remote_command]


def _ssh_files(host: str, raw: str) -> list:
    names = ' -o '.join(f'-name {shlex.quote(p)}' for p in FILE_PATTERNS)
    remote = (f'find {shlex.quote(raw)} -mindepth 3 -maxdepth 3 -type f \\( {names} \\) '
              "-printf '%P\\t%s\\t%T@\\n'")
    out = subprocess.run(_ssh_command(host, remote), capture_output=True, text=True)
    if out.returncode != 0:
        hint = ('\nIf the login node does not answer, set SWC_NODE to another one (e.g. export SWC_NODE=hpc-gw1).'
                if out.returncode == 255 else '')
        raise RuntimeError(f'listing {host}:{raw} failed (exit {out.returncode}):\n{out.stderr.strip()}{hint}')
    files = []
    for line in out.stdout.splitlines():
        parts = line.split('\t')
        if len(parts) == 3:
            files.append(SourceFile(parts[0], int(parts[1]), float(parts[2])))
    return files


def _copy_ssh(host: str, raw: str, dest: Path, files: list) -> None:
    if not files:
        return
    wanted = {f.relpath for f in files}
    remote = f'tar -C {shlex.quote(raw)} -cf - -T -'
    proc = subprocess.Popen(_ssh_command(host, remote), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    proc.stdin.write(('\n'.join(sorted(wanted)) + '\n').encode())
    proc.stdin.close()
    kwargs = {'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}
    got = set()
    with tarfile.open(fileobj=proc.stdout, mode='r|') as tar:
        for member in tar:
            if member.isfile() and member.name in wanted:
                target = dest / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                tar.extract(member, path=dest, **kwargs)
                got.add(member.name)
    err = proc.stderr.read().decode(errors='replace')
    if proc.wait() != 0 or got != wanted:
        missing = sorted(wanted - got)[:5]
        raise RuntimeError(f'fetching from {host}:{raw} failed (exit {proc.returncode}); '
                           f'missing {len(wanted - got)} files, e.g. {missing}\n{err.strip()}')


# ── reading the copied files ─────────────────────────────────────────────────

def local_copy_status(config, *, days: int, today: date | None = None, animals=None,
                      dest: Path | None = None) -> tuple:
    """(shown animals, {animal: newest session date}) from the local copy alone, without ceph
    (sc-daily-tracker --no-copy). Empty when there is no copy yet."""
    dest = Path(dest) if dest is not None else raw_mirror_dir()
    if not dest.is_dir():
        return (), {}
    today = today or date.today()
    dated = _dated(_local_folders(dest), config.reader.date_regex)
    shown, _ = select_folders(dated, today=today, days=days, summary_days=None, animals=animals)
    last: dict = {}
    for a, _, d in dated:
        if d <= today and (animals is None or a in animals) and (a not in last or d > last[a]):
            last[a] = d
    return shown, last


def session_parameters(folder: Path) -> dict:
    """The rig's Session_Parameters file in a session folder, as {name: value} (strings).

    The file has a header row to ignore and one data row that alternates name and value (behaviour_rig
    README, "Reading the outputs"). With several files (a restart), the first one's values win and a name
    whose value differs between files keeps them all, joined with '/'. Empty when there is no file.
    """
    merged: dict = {}
    for path in sorted(Path(folder).glob('Session_Parameters*.csv')):
        with open(path, newline='') as fh:
            rows = list(csv.reader(fh))
        if len(rows) < 2:
            continue
        row = rows[1]
        for name, value in zip(row[0::2], row[1::2]):
            if name not in merged:
                merged[name] = value
            elif value not in merged[name].split('/'):
                merged[name] = f'{merged[name]}/{value}'
    return merged


# ── entry point ──────────────────────────────────────────────────────────────

def mirror_sessions(config, *, days: int, summary_days: int | None, today: date | None = None,
                    animals=None, dry_run: bool = False, source: tuple | None = None,
                    dest: Path | None = None, progress=None) -> MirrorResult:
    """Bring the shown animals' session files to the local copy; see the module docstring.

    ``source`` defaults to ``raw_source()`` and ``dest`` to ``raw_mirror_dir()``; ``animals`` limits the
    animals considered (e.g. a cohort). ``dry_run`` lists what would be copied and removed, and changes
    nothing. ``progress`` (e.g. print) receives a line before the listing and before the copying.
    """
    today = today or date.today()
    route, where = source or raw_source()
    dest = Path(dest) if dest is not None else raw_mirror_dir()
    regex = config.reader.date_regex
    say = progress or (lambda message: None)
    if route == 'local':
        raw = Path(where)
        label = str(raw)
        say(f'listing {label} …')
        dated = _dated(_local_folders(raw), regex)
        shown, keep = select_folders(dated, today=today, days=days, summary_days=summary_days, animals=animals)
        files = _local_files(raw, keep)
    elif route == 'ssh':
        host, raw = where
        label = f'{host}:{raw}'
        say(f'listing {label} …')
        listed = _ssh_files(host, raw)
        dated = _dated({f.folder for f in listed}, regex)
        shown, keep = select_folders(dated, today=today, days=days, summary_days=summary_days, animals=animals)
        files = [f for f in listed if f.folder in keep]
    else:
        raise ValueError(f'unknown route {route!r}')
    todo = [f for f in files if _needs_copy(f, dest)]
    last_session: dict = {}
    for a, _, d in dated:
        if d <= today and (a not in last_session or d > last_session[a]):
            last_session[a] = d
    if todo and not dry_run:
        say(f'copying {len(todo)} files …')
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        if route == 'local':
            _copy_local(raw, dest, todo)
        else:
            _copy_ssh(host, raw, dest, todo)
    removed = _prune(dest, keep, dry_run, animals)
    return MirrorResult(route=route, source=label, dest=dest, shown_animals=shown, n_folders=len(keep),
                        copied=tuple(f.relpath for f in todo), unchanged=len(files) - len(todo),
                        removed_folders=removed, dry_run=dry_run, last_session=last_session)
