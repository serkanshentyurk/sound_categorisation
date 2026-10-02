"""
Build synthetic runs for the notebooks and CI — no data, no cluster, a few minutes.

    sc-make-synthetic-run --results /tmp/sc_results --data /tmp/sc_data [--no-model-id]
    SC_NB_SYNTHETIC=1 SC_RESULTS_ROOT=/tmp/sc_results SC_DATA_ROOT=/tmp/sc_data jupyter lab notebooks/

Writes ``opto_contrasts/selftest``, ``switch_adaptation/selftest`` and (unless --no-model-id)
``model_identification/synthetic_uniform`` under --results, in exactly the layout real runs use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sound_categorisation.data.synthetic import write_synthetic_runs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, required=True, help='results root to write into')
    p.add_argument('--data', type=Path, required=True, help='data root (the synthetic cohort goes here)')
    p.add_argument('--no-model-id', action='store_true', help='skip the synthetic grid search + consensus')
    a = p.parse_args(argv)
    write_synthetic_runs(a.results, a.data, with_model_id=not a.no_model_id)


if __name__ == '__main__':
    main()
