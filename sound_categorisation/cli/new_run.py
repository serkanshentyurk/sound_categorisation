"""
Create a run directory and print its id — for job submission, where every array task must write
into the same run.

    RUN=$(sc-new-run --report model_identification --cohort real)
    RUN=$(sc-new-run --report opto_contrasts --cohort opto1-cohort --fast)

Creates ``<results root>/<report>/<cohort>/<run_id>/logs/`` and points ``latest`` at it. Prints the
run id only (``--path`` prints the full path), so it can be captured in a shell variable.
"""

from __future__ import annotations

import argparse

from sound_categorisation.data.paths import REPORTS, start_run


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--report', required=True, choices=REPORTS)
    p.add_argument('--cohort', required=True)
    p.add_argument('--fast', action='store_true', help='suffix the run id with _fast')
    p.add_argument('--label', default=None, help='free suffix for the run id')
    p.add_argument('--path', action='store_true', help='print the full path instead of the id')
    a = p.parse_args(argv)
    run = start_run(a.report, a.cohort, fast=a.fast, label=a.label)
    print(run if a.path else run.name)


if __name__ == '__main__':
    main()
