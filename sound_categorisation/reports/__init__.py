"""
Report pipeline: compute → tables (+ readouts, meta) → figures → PDF.

    sc-reports opto     --distribution Hard-A --toi opto [--level animal|group|both]
    sc-reports opto     --all                        # the whole battery into one run
    sc-reports switches --cohort behaviour1-cohort
    sc-reports summary                               # pages from the latest opto run
    sc-reports battery                               # fast check → full battery → summary

Outputs land in ``<results root>/opto_contrasts/<cohort>/<run_id>/<distribution>/<design>[_<site>]_<toi>/``
(``switch_adaptation/<cohort>/<run_id>/switches/`` for the switch report; see data/paths.py):
``<animal>/{contrasts.csv, adaptation_*.csv, readouts.npz, meta.json}``,
``group/{group_rows.csv, group_tests.csv, adaptation_*.csv, meta.json}`` and
``pdf/*.pdf``. Notebooks read the CSVs; nothing in a notebook re-runs a bootstrap.
"""

from sound_categorisation.reports.compute import (
    DESIGNS,
    AnimalResult,
    GroupResult,
    Settings,
    compute_animal,
    compute_group,
)
from sound_categorisation.reports.tables import (
    CONTRAST_COLUMNS,
    group_tables,
    read_result,
    readout_arrays,
    to_tables,
    write_result,
)

__all__ = ['AnimalResult', 'GroupResult', 'Settings', 'compute_animal', 'compute_group', 'DESIGNS',
           'to_tables', 'group_tables', 'readout_arrays', 'write_result', 'read_result', 'CONTRAST_COLUMNS']
