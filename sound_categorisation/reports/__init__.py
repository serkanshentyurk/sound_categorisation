"""
Report pipeline: compute → tables (+ readouts, meta) → figures → PDF.

    python -m sound_categorisation.reports animal --distribution Hard-A --toi opto
    python -m sound_categorisation.reports group  --distribution Hard-A --toi opto
    python -m sound_categorisation.reports all                      # the overnight battery
    python -m sound_categorisation.reports selftest                 # synthetic, seconds

Outputs land in ``results/reports/<cohort>/<distribution>/<design>[_<site>]_<toi>/``:
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
