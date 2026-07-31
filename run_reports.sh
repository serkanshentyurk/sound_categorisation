#!/usr/bin/env bash
# Overnight report runner.  From the repo root:   bash run_reports.sh
#
# The snapshot is found automatically (<repo>/../../data/behaviour/snapshots/ on
# a laptop, /ceph/... on the cluster), so nothing else is needed. To point at a
# snapshot elsewhere:   SNAP=/path/to/sound_cat_snapshot.pkl bash run_reports.sh
#
# One failed run prints "!! FAILED" and the batch keeps going; everything lands
# in reports/. Filenames encode distribution / site / opto|post_opto, so nothing
# overwrites.

COHORT=opto1-cohort
SNAP="${SNAP:-}"
snap_args=""
[ -n "$SNAP" ] && snap_args="--snapshot $SNAP"

run() {                       # run <script> <args...>  (adds snapshot + cohort)
    echo ">>> $*"
    python $@ $snap_args --cohort "$COHORT" || echo "!! FAILED: $*"
}

# ── checks first (a pass here means the rest will run) ───────────────────────
python scripts/per_animal_phase_report.py --selftest $snap_args
run scripts/per_animal_phase_report.py --distribution Hard-A --fast --limit 1

# ── full battery: opto then post_opto ────────────────────────────────────────
for TOI in opto post_opto; do
    for D in Uniform Hard-A Hard-B; do
        run scripts/per_animal_phase_report.py --distribution "$D" --trial-of-interest "$TOI"
        run scripts/group_phase_report.py      --distribution "$D" --trial-of-interest "$TOI"
    done
    for SITE in uni bi; do
        run scripts/per_animal_alm_control_report.py --site "$SITE" --trial-of-interest "$TOI"
        run scripts/group_alm_control_report.py      --site "$SITE" --trial-of-interest "$TOI"
    done
done

echo "ALL DONE -> reports/"
