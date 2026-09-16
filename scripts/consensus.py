"""
Consensus BE/SC assignment across methods (grid search + SBI representations).

    python -m scripts.consensus --run full --cohort real --out results/consensus/full_real
    python -m scripts.consensus --run full --cohort synth_v3 --alpha 0.05 --min-votes 2

Reads the finals written by run_gs (``--gather``) and run_sbi, computes one
row per animal with each method's call and the consensus, writes
``assignments.csv`` + ``summary.txt`` + ``meta.json``, and prints the summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sound_categorisation.consensus import compute_consensus_summary, load_all_assignments
from sound_categorisation.paths import REPO_ROOT, build_metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default='full', help='run label used by run_gs / run_sbi')
    p.add_argument('--cohort', required=True, help="synthetic cohort name, or 'real'")
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--min-votes', type=int, default=1, help='significant votes needed for a consensus call')
    p.add_argument('--with-experiment', action='store_true',
                   help='list animals present in the experiment but absent from results (real only)')
    p.add_argument('--out', type=Path, default=None, help='default results/consensus/<run>_<cohort>')
    a = p.parse_args(argv)

    experiment = None
    if a.with_experiment and a.cohort == 'real':
        from sound_categorisation.cohort import load_experiment_any
        experiment = load_experiment_any()
    df = load_all_assignments(a.run, a.cohort, experiment=experiment, alpha=a.alpha,
                              min_significant_votes=a.min_votes)
    out = a.out or (REPO_ROOT / 'results' / 'consensus' / f'{a.run}_{a.cohort}')
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'assignments.csv', index=False)
    summary = compute_consensus_summary(df)
    (out / 'summary.txt').write_text(summary + '\n')
    meta = build_metadata('consensus', {'run': a.run, 'cohort': a.cohort, 'alpha': a.alpha, 'min_votes': a.min_votes})
    (out / 'meta.json').write_text(json.dumps(meta, indent=2, default=str))
    print(summary)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
