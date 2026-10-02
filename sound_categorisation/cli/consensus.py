"""
Consensus BE/SC assignment across methods (grid search + SBI representations).

    sc-consensus --cohort opto1-cohort --distribution uniform              # latest run of that cohort
    sc-consensus --cohort synthetic_uniform --distribution uniform --run-id 2026-10-01_abc1234 --min-votes 2

Reads the finals sc-grid-search (``--gather``) and sc-sbi-condition wrote into one run
(``model_identification/<cohort>/<run_id>/``), computes one row per animal with each method's call
and the consensus, and writes ``consensus/<distribution>/{assignments.csv, summary.txt, meta.json}``
into the same run.
"""

from __future__ import annotations

import argparse
import json

from sound_categorisation.data.cohort import is_synthetic_cohort
from sound_categorisation.data.paths import build_metadata, resolve_run
from sound_categorisation.inference.consensus import compute_consensus_summary, load_all_assignments
from sound_categorisation.settings import DISTRIBUTIONS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cohort', required=True, help='the cohort the run was made for (synthetic or config.yaml)')
    p.add_argument('--distribution', required=True, choices=list(DISTRIBUTIONS))
    p.add_argument('--run-id', default='latest', help="run id under model_identification/<cohort>, or 'latest'")
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--min-votes', type=int, default=1, help='significant votes needed for a consensus call')
    p.add_argument('--with-experiment', action='store_true',
                   help='list cohort animals absent from the results (real cohorts only)')
    a = p.parse_args(argv)

    try:
        run = resolve_run('model_identification', a.cohort, a.run_id)
    except FileNotFoundError as e:
        p.exit(2, f'{e}\n')
    experiment = None
    if a.with_experiment and not is_synthetic_cohort(a.cohort):
        from sound_categorisation.data.cohort import load_experiment_any
        experiment = load_experiment_any()
    df = load_all_assignments(run, a.distribution, experiment=experiment, alpha=a.alpha,
                              min_significant_votes=a.min_votes)
    out = run / 'consensus' / a.distribution
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'assignments.csv', index=False)
    summary = compute_consensus_summary(df)
    (out / 'summary.txt').write_text(summary + '\n')
    meta = build_metadata('consensus', {'cohort': a.cohort, 'distribution': a.distribution, 'alpha': a.alpha,
                                        'min_votes': a.min_votes}, run_id=run.name)
    (out / 'meta.json').write_text(json.dumps(meta, indent=2, default=str))
    print(summary)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
