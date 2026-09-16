#!/usr/bin/env bash
# Overnight report battery. From the repo root:   bash run_reports.sh
# Snapshot is found automatically (see sound_categorisation.snapshot); override with
#   SNAP=/path/to/sound_cat_snapshot.pkl bash run_reports.sh
# Outputs: results/reports/<cohort>/<distribution>/<design>_<toi>/{<animal>/,group/,pdf/}
set -uo pipefail
snap_args=""; [ -n "${SNAP:-}" ] && snap_args="--snapshot $SNAP"
python -m sound_categorisation.reports selftest || exit 1
python -m sound_categorisation.reports all --fast --limit 1 $snap_args || exit 1     # structure check on real data
python -m sound_categorisation.reports all $snap_args
python -m sound_categorisation.reports summary
