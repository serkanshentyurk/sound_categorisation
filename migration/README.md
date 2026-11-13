# migration: behav_utils 0.10 → 1.0

Temporary. `capture.py` records what the project computes; `compare.py` matches two captures number by
number and names what changed underneath every number that moved. A capture on behav_utils 0.10.0 is the
baseline; a capture after each step of the move is compared with it. Delete this folder when the move is
closed.

## What a capture holds

| file | one row per |
|---|---|
| `manifest.json` | versions, git state, arguments, statistic names, sets, counts |
| `sessions.csv` | session: folder, id, index, date, type, counts, hashes of its trials |
| `trials/<animal>.csv.gz` | trial: stimulus, choice, category, score, recorded outcome, laser, lag-1 view, stimulus check |
| `sets.csv` | session in a set: all sessions, every config preset, the opto designs' distribution × type sets |
| `stats_pooled.csv.gz` | statistic × animal × set × trial class (all, non_opto, opto, post_opto) |
| `stats_sessions.csv.gz` | statistic × session × trial class (all, non_opto, opto) |
| `readouts.csv.gz` | readout value: psychometric curve (no bootstrap), update matrix, conditional psychometric, SD profile, binned curves, per animal × set × trial class |
| `errors.csv` | anything that raised, and where |
| `model_id/` | the model-ID sessions (`all-mice`, `model_id_uniform`), empirical UM and CM targets, SBI observation vectors, grid-search matrices at six fixed points on two animals, simulator outputs at fixed parameters and seeds, simulated sessions per distribution |

Sessions are keyed by their folder name, the identity both versions share: 1.0 renames sessions to
`<animal>_<YYYYMMDD>` and hashes trial keys. Every `compute_stats` call gets a fresh generator with seed 0, so
a capture is deterministic. The same `capture.py` runs on both library versions.

## What the comparison says

Each number is equal, close (a fitted value within 1e-6 relative; `--loose` widens this for another machine),
moved, or on one side only. A number that is not equal carries the causes found in the trials beneath it:

- expected with 1.0: `gone`, `added` (the session is in one capture only), `date`, `type`, `boundary`,
  `nan_stimulus`, `score` (a trial's score changed), `prev_score` (a previous-trial score changed),
  `membership` (the set holds other sessions, with one of these somewhere in the animal);
- not expected: `rows` (a session's trial count changed), `data` (stimulus, choice, abort or laser
  changed), `history` (the lag-1 view changed), or nothing at all.

`summary.md` counts each class and lists the unexplained numbers; the CSVs hold every row that is not equal.
Exit status 1 means something moved without an expected cause.

## Baseline, on the cluster, with behav_utils 0.10.0

From the repository root, after `load-env`:

```bash
python -c "import behav_utils; print(behav_utils.__version__)"
OUT=$BEHAV_DATA_DIR/Processed/migration/baseline_0.10.0
mkdir -p "$OUT"
sbatch -o "$OUT/capture_%j.out" migration/capture.sh "$OUT"
```

When the capture job has finished (its snapshot is needed):

```bash
sbatch -o "$OUT/reports_%j.out" migration/reports.sh "$OUT"
```

## After a step

1.0 can export the snapshot only once steps 1 and 2 are both done, so the first comparison on real data is
after step 2; step 1 alone is checked by the unit and e2e tests and `pytest migration -q`. Then the same two
jobs into a new folder, and the comparison (add `--runs` pairs once both sides have report runs):

```bash
OUT=$BEHAV_DATA_DIR/Processed/migration/after_step2
mkdir -p "$OUT"
sbatch -o "$OUT/capture_%j.out" migration/capture.sh "$OUT"
```

```bash
BASE=$BEHAV_DATA_DIR/Processed/migration/baseline_0.10.0
python -m migration.compare "$BASE/capture" "$OUT/capture" --out "$OUT/compare"
```

Report runs are compared folder by folder, e.g.
`--runs "$BASE/results/opto_contrasts/opto1-cohort/<run id>" "$OUT/results/opto_contrasts/opto1-cohort/<run id>"`,
one `--runs` pair per report.

## What it does not cover

- Figures and PDFs.
- SBI posteriors: conditioning needs the networks and torch. If the model-ID sessions or observation vectors
  move, rerun conditioning and compare the calls.
- The daily tracker, whose 1.0 traps are covered by unit tests in step 1.
- In report runs, a table with no animal column gets the union of every animal's causes, a weaker
  attribution than the per-animal rows.
