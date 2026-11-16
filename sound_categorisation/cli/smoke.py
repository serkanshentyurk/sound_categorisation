#!/usr/bin/env python
"""Smoke run of the model-identification chain: every stage, small, in a scratch folder.

Trains the networks of the training array with a few hundred simulations each, makes a small
synthetic cohort with known truth, and runs each later stage on it through the same commands a real
run uses:

    sc-sbi-train            the networks of the array (settings.SBI_TRAIN_REPS x BE / SC)
    sc-make-synthetic-cohort
    sc-sbi-condition        held-out errors and the saved posteriors
    sc-grid-search          the tiny grid
    sc-sbi-sbc              calibration draws for every (representation, session count)
    sc-sbi-check            the networks against the current settings and code
    sc-consensus            the votes, and the reported single-session errors
    sc-validation-tables

After each stage it checks what the stage wrote: that the networks carry the prior in use, their
session and trial ranges and their accuracy rule; that posteriors are in the parameters' own units
inside that prior; that calibration results are in flat space; that the tables have their columns.
It ends with ``SMOKE PASSED`` or stops at the first check that fails.

What it does not say: whether the networks are any good. They are far too small. It says the code
runs end to end with the current settings, on this machine, with this environment.

Everything goes under one folder (``--dir``; default ``<data root>/smoke/<time>``): its ``data/``
is the data root (networks, cohort) and its ``results/`` the results root for the duration of the
run, so nothing real is read or written. Needs torch and sbi.

    sc-smoke                                   # local: about 500 kept simulations per network
    sc-smoke --n-simulations 300 --dir /tmp/sc_smoke
    sbatch slurm/smoke.sh                      # the same on the cluster, as one job

Time: mostly the pooled BE network, which simulates about ten prior draws for each one it keeps.
With the defaults, up to about 30 minutes on 8 cores.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

COHORT = 'smoke_uniform'
DISTRIBUTION = 'uniform'


class SmokeFailure(AssertionError):
    """A check on a stage's output failed."""


class _Checks:
    def __init__(self):
        self.n = 0

    def __call__(self, condition, message: str):
        self.n += 1
        if not condition:
            raise SmokeFailure(message)


def _stage(name: str):
    print(f'\n[smoke] === {name} ===', flush=True)
    return time.time()


def _done(t0: float):
    print(f'[smoke] ok ({(time.time() - t0) / 60:.1f} min)', flush=True)


def check_networks(check, n_simulations: int) -> None:
    """Every network of the array exists and carries the current settings."""
    from sound_categorisation.data.paths import snpe_net_path
    from sound_categorisation.inference.amortised import network_prior, read_network_file
    from sound_categorisation.inference.simulator import accuracy_rule
    from sound_categorisation.settings import MODEL_TYPES, SBI_PRIOR, SBI_REPRESENTATIONS, SBI_TRAIN_REPS

    print('[smoke] rep     model  kept   drawn  pass%  simulate  train')
    for rep in SBI_TRAIN_REPS:
        cfg = SBI_REPRESENTATIONS[rep]
        for model in MODEL_TYPES:
            path = snpe_net_path(rep, model, DISTRIBUTION)
            check(path.exists(), f'{path} was not written')
            data = read_network_file(path)
            meta = data['training_metadata']
            name = f'{rep}/{model}'
            check(network_prior(data) == SBI_PRIOR, f'{name}: prior {network_prior(data)!r}, not {SBI_PRIOR!r}')
            check(tuple(data['N']) == tuple(cfg['N']) if isinstance(cfg['N'], tuple) else data['N'] == cfg['N'],
                  f'{name}: N {data["N"]}, settings say {cfg["N"]}')
            check(tuple(data['T']) == tuple(cfg['T']), f'{name}: T {data["T"]}, settings say {cfg["T"]}')
            rule = accuracy_rule(rep, DISTRIBUTION)
            check(data['min_accuracy'] == rule, f'{name}: accuracy rule {data["min_accuracy"]}, expected {rule}')
            check(meta['n_simulations'] == n_simulations, f'{name}: {meta["n_simulations"]} kept')
            check(meta['n_drawn'] >= n_simulations and 0 < meta['pass_rate'] <= 1, f'{name}: draws {meta}')
            check((rule is None) == (meta['n_drawn'] == n_simulations),
                  f'{name}: {meta["n_drawn"]} draws for {n_simulations} kept, rule {rule}')
            print(f'[smoke] {rep:<7} {model:<5} {meta["n_simulations"]:>5} {meta["n_drawn"]:>7} '
                  f'{100 * meta["pass_rate"]:>5.1f}  {meta["simulation_time"] / 60:>6.1f} m '
                  f'{meta["fit_time"] / 60:>5.1f} m')


def check_conditioning(check, run: Path, n_animals: int) -> None:
    """Every (representation, animal, model) has a result with finite held-out errors and estimates
    in the parameters' own units inside the prior in use; the multi-session representations also
    have the saved all-session posterior, the single-session one has none."""
    import numpy as np

    from sound_categorisation.data.paths import model_id_dir
    from sound_categorisation.models.priors import bounds_arrays
    from sound_categorisation.settings import MODEL_TYPES, SBI_PRIOR, SBI_REPRESENTATIONS, SBI_TRAIN_REPS

    for rep in SBI_TRAIN_REPS:
        files = sorted(model_id_dir(run, 'sbi', 'update_matrix', DISTRIBUTION, rep).glob('*.pkl'))
        check(len(files) == n_animals * len(MODEL_TYPES), f'{rep}: {len(files)} result files')
        for path in files:
            with open(path, 'rb') as f:
                result = pickle.load(f)
            name = f'{rep}/{path.stem}'
            errors = [r['test_error'] for r in result['results']]
            check(len(errors) > 0 and np.all(np.isfinite(errors)), f'{name}: held-out errors {errors}')
            low, high = bounds_arrays(result['model'].lower(), SBI_PRIOR)
            estimates = np.array([list(r['best_params'].values()) for r in result['results']], dtype=float)
            check(bool(np.all(estimates >= low - 1e-9) and np.all(estimates <= high + 1e-9)),
                  f'{name}: estimates outside the {SBI_PRIOR} prior')
            post = result.get('posterior')
            if SBI_REPRESENTATIONS[rep]['N'] == 1:
                check(post is None, f'{name}: a single-session result carries an all-session posterior')
                continue
            check(post is not None and post.get('prior') == SBI_PRIOR,
                  f'{name}: saved posterior {post and post.get("prior")}')
            samples = np.asarray(post['posterior_samples'], dtype=float)
            check(samples.ndim == 2 and samples.shape[1] == len(low), f'{name}: posterior shape {samples.shape}')
            check(bool(np.all(samples >= low - 1e-9) and np.all(samples <= high + 1e-9)),
                  f'{name}: posterior samples outside the {SBI_PRIOR} prior')


def check_sbc(check, run: Path, n_draws: int) -> None:
    """Every calibration case of the array was written, in flat space, with the network's rule."""
    import numpy as np

    from sound_categorisation.inference.simulator import accuracy_rule
    from sound_categorisation.inference.tasks import SBC_CASES, sbc_case
    from sound_categorisation.models.priors import flat_bounds, log_uniform_params
    from sound_categorisation.settings import MODEL_TYPES, SBI_PRIOR

    for case in SBC_CASES:
        rep, n_sessions = sbc_case(case)
        for model in MODEL_TYPES:
            path = run / 'sbc' / DISTRIBUTION / f'n{n_sessions}' / rep / f'{model}.pkl'
            check(path.exists(), f'{path} was not written')
            with open(path, 'rb') as f:
                res = pickle.load(f)
            name = f'sbc {case}/{model}'
            check(res['prior'] == SBI_PRIOR and res['min_accuracy'] == accuracy_rule(rep, DISTRIBUTION),
                  f'{name}: prior {res["prior"]}, rule {res["min_accuracy"]}')
            check(res['log_uniform'] == list(log_uniform_params(model.lower(), SBI_PRIOR)),
                  f'{name}: log_uniform {res["log_uniform"]}')
            check(len(res['ranks']) + res['n_skipped'] == n_draws, f'{name}: {len(res["ranks"])} draws')
            low, high = flat_bounds(model.lower(), SBI_PRIOR)
            truth = np.asarray(res['theta_true'], dtype=float)
            check(bool(np.all(truth >= low - 1e-9) and np.all(truth <= high + 1e-9)),
                  f'{name}: truths outside the flat-space box')


def check_network_table(check, run: Path) -> None:
    """The configuration and parameter rows of sc-sbi-check agree with the settings, apart from
    the simulation count (small here). Fingerprint rows are printed, not checked: the stored
    medians of a network this small are too noisy for that."""
    import pandas as pd

    files = sorted((run / 'network_check').glob('*.csv'))
    check(len(files) == 1, f'network check tables: {files}')
    table = pd.read_csv(files[0])
    structural = table[table['group'].isin(['config', 'parameters']) & (table['check'] != 'n_simulations')]
    bad = structural[structural['ok'].astype(str) != 'True']
    check(len(bad) == 0, 'sc-sbi-check disagrees with the settings:\n'
          + bad[['network', 'check', 'stored', 'current']].to_string(index=False))
    flagged = table[(table['group'] == 'fingerprint') & (table['ok'].astype(str) == 'False')]
    print(f'[smoke] network check: {len(structural)} config and parameter rows agree; '
          f'{len(flagged)} fingerprint rows flagged (not checked at this size)')


def check_tables(check, run: Path, n_animals: int) -> None:
    """The consensus has its two votes and the reported single-session errors; the validation
    tables are there with the scale column."""
    import pandas as pd

    assign = pd.read_csv(run / 'consensus' / DISTRIBUTION / 'assignments.csv')
    check(len(assign) == n_animals, f'consensus: {len(assign)} animals')
    for column in ('GS-UM', 'GS-UM_p', 'SBI-pooled-UM', 'SBI-pooled-UM_p', 'SBI-single-UM_be',
                   'SBI-single-UM_sc', 'SBI-single-UM_lower', 'Consensus', 'true_model'):
        check(column in assign, f'consensus: no column {column}')
    check('SBI-single-UM' not in assign and 'SBI-single-UM_p' not in assign,
          'consensus: the single-session network has a vote column')
    tables = run / 'tables'
    for name in ('sbc_summary.csv', 'coverage.csv', 'recovery.csv', 'confusion.csv', 'synthetic_recovery.csv'):
        check((tables / name).exists(), f'{tables / name} was not written')
    recovery = pd.read_csv(tables / 'recovery.csv')
    check(set(recovery['scale']) <= {'log', 'linear'} and 'log' in set(recovery['scale']),
          f'recovery.csv scales: {set(recovery["scale"])}')
    check(set(recovery['rep']) == {'pooled', 'single'}, f'recovery.csv reps: {set(recovery["rep"])}')


def run(directory, n_simulations: int = 500, n_per_model: int = 2, sbc_draws: int = 4,
        fingerprint_draws: int = 200) -> Path:
    """The smoke run under ``directory``; returns the run directory. Raises ``SmokeFailure`` at the
    first failed check. The data and results roots are pointed at ``directory`` for the duration
    and restored afterwards."""
    from sound_categorisation.cli import (
        consensus,
        grid_search,
        make_synthetic_cohort,
        sbi_check,
        sbi_condition,
        sbi_sbc,
        sbi_train,
        validation_tables,
    )
    from sound_categorisation.data.paths import start_run

    directory = Path(directory).resolve()
    before = {k: os.environ.get(k) for k in ('SC_DATA_ROOT', 'SC_RESULTS_ROOT')}
    os.environ['SC_DATA_ROOT'] = str(directory / 'data')
    os.environ['SC_RESULTS_ROOT'] = str(directory / 'results')
    check, t_all, n_animals = _Checks(), time.time(), 2 * n_per_model
    print(f'[smoke] everything under {directory}')
    try:
        t0 = _stage(f'sc-sbi-train: {n_simulations} kept simulations per network')
        sbi_train.main(['--n-simulations', str(n_simulations)])
        check_networks(check, n_simulations)
        _done(t0)

        t0 = _stage(f'sc-make-synthetic-cohort: {n_per_model} BE + {n_per_model} SC animals')
        make_synthetic_cohort.main(['--name', COHORT, '--distribution', DISTRIBUTION,
                                    '--n-per-model', str(n_per_model), '--accuracy-rule'])
        _done(t0)

        run_dir = start_run('model_identification', COHORT, fast=True)
        common = ['--cohort', COHORT, '--distribution', DISTRIBUTION, '--run-id', run_dir.name]

        t0 = _stage('sc-sbi-condition --fast')
        sbi_condition.main([*common, '--fast'])
        check_conditioning(check, run_dir, n_animals)
        _done(t0)

        t0 = _stage('sc-grid-search --fast')
        grid_search.main([*common, '--fit-target', 'update_matrix', '--fast', '--n-jobs', '-1'])
        _done(t0)

        t0 = _stage(f'sc-sbi-sbc: {sbc_draws} draws per case')
        sbi_sbc.main([*common, '--n-draws', str(sbc_draws), '--n-posterior-samples', '100'])
        check_sbc(check, run_dir, sbc_draws)
        _done(t0)

        t0 = _stage(f'sc-sbi-check: {fingerprint_draws} fingerprint draws per network')
        sbi_check.main(['--cohort', COHORT, '--run-id', run_dir.name, '--n-draws', str(fingerprint_draws)])
        check_network_table(check, run_dir)
        _done(t0)

        t0 = _stage('sc-consensus and sc-validation-tables')
        consensus.main(['--cohort', COHORT, '--distribution', DISTRIBUTION, '--run-id', run_dir.name])
        validation_tables.main(['--cohort', COHORT, '--distribution', DISTRIBUTION, '--run-id', run_dir.name])
        check_tables(check, run_dir, n_animals)
        _done(t0)
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(f'\n[smoke] SMOKE PASSED: {check.n} checks in {(time.time() - t_all) / 60:.1f} min -> {run_dir}')
    return run_dir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir', default=None, help='scratch folder (default <data root>/smoke/<time>)')
    p.add_argument('--n-simulations', type=int, default=500, help='simulations kept per network')
    p.add_argument('--n-per-model', type=int, default=2, help='synthetic animals per model')
    p.add_argument('--sbc-draws', type=int, default=4, help='calibration draws per case')
    p.add_argument('--fingerprint-draws', type=int, default=200, help='sc-sbi-check draws per network')
    args = p.parse_args(argv)

    try:
        import sbi  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        p.exit(2, f'sc-smoke needs torch and sbi (pip install -e ".[fit]"): {e}\n')
    if args.dir:
        directory = Path(args.dir)
    else:
        from sound_categorisation.data.paths import data_root
        directory = data_root() / 'smoke' / time.strftime('%Y-%m-%d_%H%M%S')
    try:
        run(directory, n_simulations=args.n_simulations, n_per_model=args.n_per_model,
            sbc_draws=args.sbc_draws, fingerprint_draws=args.fingerprint_draws)
    except SmokeFailure as e:
        p.exit(1, f'\n[smoke] SMOKE FAILED: {e}\n')


if __name__ == '__main__':
    main()
