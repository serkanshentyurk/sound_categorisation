"""Synthetic end-to-end check of the report pipeline (no data load, seconds)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
from behav_utils.data.structures import AnimalData, ExperimentData, SessionData, SessionMetadata, TrialData

__all__ = ['synthetic_experiment', 'run_selftest']


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
    return exp


def run_selftest(out_dir: Path):
    from types import SimpleNamespace

    from sound_categorisation.reports.cli import run_animal, run_group
    from sound_categorisation.reports.compute import Settings
    exp = synthetic_experiment()
    ids = list(exp.animals)
    a = SimpleNamespace(out=Path(out_dir), cohort='selftest', snapshot=None, config=None, fast=True)
    s = Settings.fast()
    per = run_animal(exp, ids, 'Uniform', 'opto', 'ppc', None, a, s)
    run_group(exp, ids, 'Uniform', 'opto', 'ppc', None, a, s, per)
    per = run_animal(exp, ids[:2], 'Uniform', 'opto', 'alm', 'uni', a, s)
    run_group(exp, ids[:2], 'Uniform', 'opto', 'alm', 'uni', a, s, per)
    # the slow pages once, on one animal, with readouts + adaptation on
    full = Settings(n_boot=20, n_perm=20, curve_bootstrap=10)
    per = run_animal(exp, ids[:1], 'Hard-A', 'opto', 'ppc', None, a, full)
    run_group(exp, ids, 'Hard-A', 'opto', 'ppc', None, a, full, per)
    print(f'selftest OK -> {out_dir}')
    return Path(out_dir)
