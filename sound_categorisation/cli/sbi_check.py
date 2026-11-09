#!/usr/bin/env python
"""Check saved SBI networks against the current code and settings (``inference.network_check``).

For every ``snpe_*.pkl`` in a networks directory: the stored configuration against settings,
the stored parameter order and prior bounds against the model's, and a fingerprint (feature
medians of fresh simulations from the current simulator against those stored at training).
With ``--real-run``, also whether the sessions a model-identification run used are still the
ones the current code selects.

Writes into the run it is given (normally the synthetic validation run):

    <run>/network_check/<label>.csv                        one row per check per network
    <run>/network_check/sessions_<cohort>_<run_id>_<D>.csv  with --real-run

``<label>`` defaults to the networks directory's name: ``snpe_networks`` for the current set,
``2026-10`` for the October networks once archived to ``snpe_networks/2026-10/``.

    sc-sbi-check --cohort synthetic_uniform_k50 --run-id ID
    sc-sbi-check --cohort synthetic_uniform_k50 --run-id ID \\
        --networks-dir <data root>/snpe_networks/2026-10 --real-run all-mice/<october run id>

Cluster (one node; the fingerprint simulates 5,000 draws per network)::

    bash slurm/submit.sh sbi-check --cohort synthetic_uniform_k50 --run-id ID [same options]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from sound_categorisation.data.paths import build_metadata, resolve_run, snpe_networks_dir, start_run
from sound_categorisation.settings import DISTRIBUTIONS


def check_directory(networks_dir: Path, n_draws: int, seed: int, n_jobs: int,
                    fingerprint: bool = True) -> pd.DataFrame:
    from sound_categorisation.inference.amortised import read_network_file
    from sound_categorisation.inference.network_check import (
        config_rows,
        fingerprint_rows,
        parameter_rows,
    )
    files = sorted(Path(networks_dir).glob('snpe_*.pkl'))
    if not files:
        raise FileNotFoundError(f'no snpe_*.pkl in {networks_dir}')
    rows = []
    for path in files:
        t0 = time.time()
        data = read_network_file(path)
        net_rows = config_rows(data, path.name) + parameter_rows(data, path.name)
        if fingerprint:
            net_rows += fingerprint_rows(data, path.name, n_draws=n_draws, seed=seed, n_jobs=n_jobs)
        bad = [r['check'] for r in net_rows if r['ok'] is False]
        print(f'[check] {path.name}: {len(bad)} difference(s){": " + ", ".join(bad) if bad else ""} '
              f'({(time.time() - t0) / 60:.1f} min)')
        rows += net_rows
    return pd.DataFrame(rows)


def check_sessions(real_run: str, distribution: str) -> tuple[pd.DataFrame, str]:
    from sound_categorisation.data.cohort import select_animals
    from sound_categorisation.inference.network_check import sessions_diff
    if '/' not in real_run:
        raise ValueError(f'--real-run takes <cohort>/<run_id>, got {real_run!r}')
    cohort, run_id = real_run.split('/', 1)
    run = resolve_run('model_identification', cohort, run_id)
    old = pd.read_csv(run / 'sessions' / f'{distribution}.csv')
    _, new = select_animals(cohort, preset=f'model_id_{distribution}')
    diff = sessions_diff(old, new)
    n_changed = int((~diff['same']).sum())
    print(f'[check] sessions of {cohort}/{run.name} ({distribution}): {n_changed} animal(s) differ '
          'from the current selection')
    for _, r in diff[~diff['same']].iterrows():
        print(f'    {r["animal_id"]}: {r["detail"]}')
    return diff, f'sessions_{cohort}_{run.name}_{distribution}.csv'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cohort', default=None, help='the run to write into (normally the validation cohort)')
    p.add_argument('--run-id', default=None)
    p.add_argument('--networks-dir', default=None, help='default: <data root>/snpe_networks')
    p.add_argument('--label', default=None, help='output name; default the networks directory name')
    p.add_argument('--real-run', default=None, help='<cohort>/<run_id> of a run whose sessions to re-check')
    p.add_argument('--distribution', default='uniform', choices=DISTRIBUTIONS,
                   help='the phase whose sessions --real-run compares')
    p.add_argument('--n-draws', type=int, default=None, help='fingerprint draws per network (default 5000)')
    p.add_argument('--n-jobs', type=int, default=-1)
    p.add_argument('--seed', type=int, default=None,
                   help='fingerprint seed (default VALIDATION_SEED + 500,000)')
    p.add_argument('--no-fingerprint', action='store_true', help='config and parameter checks only')
    p.add_argument('--task-id', type=int, default=None, help='accepted for slurm/submit.sh; one task')
    p.add_argument('--print-array', action='store_true', help='print the SLURM --array range and exit')
    p.add_argument('--count', action='store_true', help='print the number of array tasks and exit')
    args = p.parse_args(argv)
    if args.print_array:
        print('0-0')
        return
    if args.count:
        print(1)
        return
    if not args.cohort:
        p.error('--cohort is required (the run the check is written into)')

    from sound_categorisation.inference.network_check import FINGERPRINT_DRAWS
    networks_dir = Path(args.networks_dir) if args.networks_dir else snpe_networks_dir()
    label = args.label or networks_dir.name
    run = start_run('model_identification', args.cohort, args.run_id)
    out = run / 'network_check'
    out.mkdir(parents=True, exist_ok=True)
    meta = build_metadata('sbi_check', vars(args), run_id=run.name)

    table = check_directory(networks_dir, args.n_draws or FINGERPRINT_DRAWS, args.seed, args.n_jobs,
                            fingerprint=not args.no_fingerprint)
    table.insert(0, 'networks_dir', str(networks_dir))
    table.to_csv(out / f'{label}.csv', index=False)
    print(f'[check] {len(table)} rows -> {out / (label + ".csv")}')

    if args.real_run:
        diff, name = check_sessions(args.real_run, args.distribution)
        diff.to_csv(out / name, index=False)
        print(f'[check] -> {out / name}')
    (out / f'{label}_meta.json').write_text(json.dumps(meta, indent=2, default=str))


if __name__ == '__main__':
    main()
