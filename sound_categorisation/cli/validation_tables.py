#!/usr/bin/env python
"""Write the thesis tables of an SBI validation run (``inference.validation``).

Reads a finished validation run (a synthetic cohort's model identification plus sc-sbi-sbc,
and sc-sbi-check if it was run) and, optionally, real runs; writes CSVs to ``<run>/tables/``:

    from sc-sbi-sbc          sbc_summary.csv, coverage.csv, sbc_ecdf.csv,
                             recovery.csv, recovery_draws.csv
    from the synthetic run   assignments_rule1.csv, assignments_rule2.csv, confusion.csv,
                             call_summary.csv, contested.csv, gs_by_sigma.csv,
                             synthetic_recovery.csv
    with --real              real_sigma.csv (posterior sigma_percep against the grid's largest
                             value), real_calls.csv (columns current_rule1 / current_rule2, and with
                             --october also october_rule1: both runs side by side)

Rule 1 calls the model with more significant votes (at least one; a call with a significant
vote against it is flagged contested); rule 2 needs two significant votes. A part whose inputs
are missing is skipped with a note, so the command can run while a run is still filling.

    sc-validation-tables --cohort synthetic_uniform_k50 --run-id ID \\
        --real all-mice/<new run id> --october all-mice/<october run id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sound_categorisation.data.paths import build_metadata, resolve_run
from sound_categorisation.settings import DISTRIBUTIONS, SBI_TRAIN_REPS


def _run(spec: str) -> Path:
    if '/' not in spec:
        raise ValueError(f'expected <cohort>/<run_id>, got {spec!r}')
    cohort, run_id = spec.split('/', 1)
    return resolve_run('model_identification', cohort, run_id)


def _write(df, out: Path, name: str):
    if df is None or len(df) == 0:
        print(f'  {name}: nothing to write')
        return
    df.to_csv(out / name, index=False)
    print(f'  {name}: {len(df)} rows')


def write_tables(run: Path, distribution: str, real: Path | None = None,
                 october: Path | None = None) -> Path:
    from sound_categorisation.inference import validation as V
    from sound_categorisation.inference.consensus import load_all_assignments
    from sound_categorisation.inference.grid_search import DEFAULT_GRID

    out = run / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    reps = tuple(SBI_TRAIN_REPS)
    sigma_max = float(max(DEFAULT_GRID[m].sigma_percep_values.max() for m in ('BE', 'SC')))
    print(f'[tables] {run} -> {out}')

    sbc = V.load_sbc(run, distribution)
    if sbc:
        _write(V.sbc_summary_table(sbc), out, 'sbc_summary.csv')
        _write(V.coverage_table(sbc), out, 'coverage.csv')
        _write(V.ecdf_table(sbc), out, 'sbc_ecdf.csv')
        draws = V.recovery_draws(sbc)
        _write(draws, out, 'recovery_draws.csv')
        _write(V.recovery_table(draws), out, 'recovery.csv')
    else:
        print(f'  no SBC results under {run / "sbc" / distribution}')

    assign = {rule: load_all_assignments(run, distribution, min_significant_votes=rule)
              for rule in V.RULES}
    if len(assign[1]) and 'true_model' in assign[1]:
        for rule, a in assign.items():
            _write(a, out, f'assignments_rule{rule}.csv')
        conf = V.confusion_table(assign)
        _write(conf, out, 'confusion.csv')
        _write(V.call_summary(conf), out, 'call_summary.csv')
        _write(V.contested_table(assign), out, 'contested.csv')
        truths = V.synthetic_truths(run, distribution, reps)
        _write(V.gs_sigma_table(assign[1], truths, sigma_max), out, 'gs_by_sigma.csv')
        _write(V.synthetic_recovery_table(run, distribution, reps), out, 'synthetic_recovery.csv')
    else:
        print('  no model-identification results with ground truth in this run')

    if real is not None:
        _write(V.real_sigma_table(real, distribution, sigma_max, reps), out, 'real_sigma.csv')
        # column prefixes: 'october' (the earlier run, primary rule) and 'current' (this run, both
        # rules); the run ids are in tables/meta.json
        by_label = {f'current_rule{rule}': load_all_assignments(real, distribution,
                                                                 min_significant_votes=rule)
                    for rule in V.RULES}
        if october is not None:
            by_label = {'october_rule1': load_all_assignments(october, distribution), **by_label}
        _write(V.real_calls_table(by_label), out, 'real_calls.csv')
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cohort', required=True, help='the synthetic validation cohort')
    p.add_argument('--run-id', default='latest')
    p.add_argument('--distribution', default='uniform', choices=DISTRIBUTIONS)
    p.add_argument('--real', default=None, help='<cohort>/<run_id> of the real run to tabulate')
    p.add_argument('--october', default=None, help='<cohort>/<run_id> of an earlier real run to compare')
    args = p.parse_args(argv)

    run = resolve_run('model_identification', args.cohort, args.run_id)
    real = _run(args.real) if args.real else None
    october = _run(args.october) if args.october else None
    out = write_tables(run, args.distribution, real=real, october=october)
    meta = build_metadata('validation_tables', vars(args), run_id=run.name)
    (out / 'meta.json').write_text(json.dumps(meta, indent=2, default=str))


if __name__ == '__main__':
    main()
