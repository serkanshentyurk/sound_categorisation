"""Synthetic ExperimentData with the shape of the real opto and blocked-switch designs.

Used by the end-to-end tests (``tests/e2e``) and anywhere an in-memory experiment is wanted without
the snapshot. Ground truth is by construction: HET animals shift their criterion on laser trials, the
blocked animals drift toward each new distribution across a block.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
from behav_utils.data.structures import AnimalData, ExperimentData, SessionData, SessionMetadata, TrialData

__all__ = ['synthetic_experiment', 'write_synthetic_runs']


def _trials(n, rng, bias, rt_mean, p_opto=0.3):
    s = rng.uniform(-1, 1, n)
    c = (s > 0).astype(float)
    ch = (s > bias).astype(float)
    flip = rng.random(n) < 0.15
    ch[flip] = 1 - ch[flip]
    return TrialData(trial_number=np.arange(n), stimulus=s, category=c, choice=ch,
                     outcome=(ch == c).astype(float), correct=(ch == c), abort=np.zeros(n, bool),
                     opto_on=rng.random(n) < p_opto,
                     reaction_time=np.clip(rng.normal(rt_mean, 40, n), 80, None))


def synthetic_experiment(n_animals=4, seed=0, n_trials=180) -> ExperimentData:
    """WT/HET animals with expert-Uniform, opto, masking and ALM sessions on Uniform
    plus a Hard-A block (opto + masking) — enough to exercise every page."""
    rng = np.random.default_rng(seed)
    exp = ExperimentData(metadata={'cohort': 'selftest'})
    for k in range(n_animals):
        geno = 'het' if k % 2 else 'wt'
        sessions, idx = [], 0

        def add(stype, dist, bias, rt, n=3, sessions=sessions):
            nonlocal idx
            for _ in range(n):
                sessions.append(SessionData(
                    session_id=f'{stype}{idx}', session_idx=idx, date=date(2024, 1, 1) + timedelta(days=idx),
                    metadata=SessionMetadata(fields={'stage': 'Full_Task_Cont', 'distribution': dist}),
                    trials=_trials(n_trials, rng, rng.normal(bias, 0.1), rt,
                                   p_opto=0.0 if stype == 'regular' else 0.3),
                    session_type=stype))
                idx += 1
        add('regular', 'Uniform', 0.0, 260, n=6)
        add('masking', 'Uniform', 0.05, 270)
        add('opto', 'Uniform', 0.35 if geno == 'het' else 0.1, 250)
        add('alm_control_uni', 'Uniform', 0.2, 300)
        add('alm_control_bi', 'Uniform', 0.2, 300)
        # Hard phase as in the real design: A/B alternating daily, laser sessions first, then masking
        for _ in range(3):
            add('opto', 'Hard-A', 0.4 if geno == 'het' else 0.15, 250, n=1)
            add('opto', 'Hard-B', -0.3 if geno == 'het' else -0.1, 250, n=1)
        for _ in range(3):
            add('masking', 'Hard-A', 0.15, 270, n=1)
            add('masking', 'Hard-B', -0.1, 270, n=1)
        exp.add_animal(AnimalData(animal_id=f'ST{k:02d}', sessions=sessions, metadata={'genotype': geno}))
    # two blocked animals (no laser): Uniform → Hard-B → Hard-A → Hard-B → Hard-A, several sessions per block,
    # the criterion drifting toward the new distribution across each block
    for k in range(2):
        sessions, idx = [], 0
        schedule = [('Uniform', 0.0)] * 4 + [('Hard-B', -0.15)] * 4 + [('Hard-A', 0.15)] * 4 + [('Hard-B', -0.15)] * 3 + [('Hard-A', 0.15)] * 3
        cur = 0.0
        for dist, target in schedule:
            cur = cur + 0.5 * (target - cur)            # approach the new criterion by half each session
            sessions.append(SessionData(
                session_id=f'regular{idx}', session_idx=idx, date=date(2024, 1, 1) + timedelta(days=idx),
                metadata=SessionMetadata(fields={'stage': 'Full_Task_Cont', 'distribution': dist}),
                trials=_trials(n_trials, rng, cur + rng.normal(0, 0.03), 260, p_opto=0.0), session_type='regular'))
            idx += 1
        exp.add_animal(AnimalData(animal_id=f'SB{k:02d}', sessions=sessions, metadata={'genotype': 'wt'}))
    return exp


def write_synthetic_runs(results_root, data_root, *, with_model_id: bool = True) -> dict:
    """Build the runs the notebooks and CI read, from synthetic data only: an ``opto_contrasts`` run
    (fast on every animal, one full condition), a ``switch_adaptation`` run, their summary, and — with
    ``with_model_id`` — a synthetic cohort, a fast grid search and its consensus under
    ``model_identification/synthetic_uniform``. Returns the run directories. ``sc-make-synthetic-run``
    is the CLI; ``SC_NB_SYNTHETIC=1`` makes the notebooks read these instead of the real cohorts.
    """
    import os
    from types import SimpleNamespace

    results_root, data_root = Path(results_root), Path(data_root)
    os.environ['SC_RESULTS_ROOT'] = str(results_root)
    os.environ['SC_DATA_ROOT'] = str(data_root)
    results_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    from sound_categorisation.data.paths import start_run
    from sound_categorisation.reports.cli import run_animal, run_group
    from sound_categorisation.reports.compute import Settings
    from sound_categorisation.reports.summary import write_summary
    from sound_categorisation.reports.switches import run_switches

    exp = synthetic_experiment()
    ids = list(exp.animals)
    run = start_run('opto_contrasts', 'selftest', fast=True, root=results_root)
    a = SimpleNamespace(cohort='selftest', snapshot=None, config=None, fast=True, run_path=run, run_id=run.name)
    fast = Settings.fast()
    for dist in ('Uniform', 'Hard-B'):
        per = run_animal(exp, ids, dist, 'opto', 'ppc', None, a, fast)
        run_group(exp, ids, dist, 'opto', 'ppc', None, a, fast, per)
    per = run_animal(exp, ids[:2], 'Uniform', 'opto', 'alm', 'uni', a, fast)
    run_group(exp, ids[:2], 'Uniform', 'opto', 'alm', 'uni', a, fast, per)
    full = Settings(n_boot=20, n_perm=20, curve_bootstrap=10)   # the slow pages once, on one animal
    per = run_animal(exp, ids[:1], 'Hard-A', 'opto', 'ppc', None, a, full)
    run_group(exp, ids, 'Hard-A', 'opto', 'ppc', None, a, full, per)
    write_summary(run, 'selftest')
    srun = start_run('switch_adaptation', 'selftest', fast=True, root=results_root)
    run_switches(exp, ['SB00', 'SB01'], srun, 'selftest', min_block_trials=300, max_trials=800)
    out = {'opto_contrasts': run, 'switch_adaptation': srun}

    if with_model_id:
        from sound_categorisation.cli import consensus, make_synthetic_cohort, run_gs
        make_synthetic_cohort.main(['--distribution', 'uniform', '--n-per-model', '1', '--n-sessions', '2',
                                    '--trials', '200'])
        mrun = start_run('model_identification', 'synthetic_uniform', fast=True, root=results_root)
        base = ['--source', 'synthetic', '--cohort', 'synthetic_uniform', '--distribution', 'uniform',
                '--fit-target', 'update_matrix', '--fast', '--run-id', mrun.name]
        for t in range(8):
            run_gs.main(base + ['--task-id', str(t)])
        run_gs.main(base + ['--gather'])
        consensus.main(['--cohort', 'synthetic_uniform', '--distribution', 'uniform', '--run-id', mrun.name])
        out['model_identification'] = mrun
    print('synthetic runs ->', results_root)
    return out
