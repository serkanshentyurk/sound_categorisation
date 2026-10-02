"""Write the README.md of an opto run — the layout, tables, columns and page conventions.

Regenerated on every ``summary`` run so it always matches the code
that produced the folder. It describes what the files are; it makes no claims
about what the numbers show (that belongs in the notebooks and the thesis).
"""

from __future__ import annotations

from pathlib import Path

README = '''# Report outputs

Produced by `sc-reports` (behav_utils {bu}, sound_categorisation {sc}). This folder is one run:
`<results root>/opto_contrasts/<cohort>/<run_id>/`; `../latest` points at the newest run of this cohort.
Every subfolder carries a `meta.json` with the run id, command line, snapshot, config, settings, versions
and git state (sha + dirty flag) that produced it.

## Layout

```
<run_id>/
  logs/                                  job logs, when run on the cluster
  summary.pdf, summary_<page>.png        four pages: overall, uniform, hard_a, hard_b
  README.md                              this file
  <distribution>/<site>_<trial_class>/
    <animal>/contrasts.csv               every contrast x unit, one schema (below)
    <animal>/levels.csv                  the observed value of every stat in every condition the contrasts use
    <animal>/trajectory.csv              one row per session of the phase, in acquisition order
    <animal>/trajectory_curves.csv       rolling PSE within each session
    <animal>/readouts.npz                psychometric curves + update matrices per (phase, trial_type)
    <animal>/meta.json
    group/group_rows.csv                 per-animal point differences per contrast kind (the fold)
    group/group_tests.csv                WT-vs-HET rank tests per kind x stat
    group/trajectory.csv, trajectory_curves.csv   all animals' trajectories, with genotype
    pdf/<animal>_<site>_<trial_class>.pdf, pdf/group_<site>_<trial_class>.pdf
```

`distribution` in {{Uniform, Hard-A, Hard-B}}; `site` in {{ppc, alm_uni, alm_bi}} (older runs: empty = ppc, plus a
`design` column — readers normalise them with `reports.tables.normalise_site`);
`trial_class` (trial of interest) in {{opto, post_opto}}: which trials are contrasted with the non-laser trials.

## Session types and sites

- `opto`: laser sessions (30 % of trials laser-on, randomised per trial). WT = VGAT-ChR2-negative: light, no silencing.
  HET = light + silencing.
- `masking`: masking-light sessions, no laser; the `opto_on` flag marks the trials the rig would have lasered
  ("fake opto"). Uniform: one masking block and one laser block per animal. Hard: six laser sessions alternating
  A/B daily, then six masking sessions alternating A/B.
- `alm_control_uni` / `alm_control_bi`: laser on ALM instead of PPC (site control), Uniform only.

## Contrasts (`contrasts.csv`)

| kind | what is compared | unit | p |
|---|---|---|---|
| `within` | `trial_class` trials vs non-laser trials, in the laser sessions | trials (and sessions) | permutation (laser randomised per trial) |
| `within_masking` | fake-opto trials vs the rest, in masking sessions (null control for the trial flag) | trials | permutation |
| `between` | laser sessions vs masking sessions, all trials | sessions (and trials) | none: session type was not randomised; the masking sessions came later |
| `compensation` | laser-OFF trials of the laser sessions vs masking sessions, all trials: does the baseline criterion move in a session where 30 % of trials are lasered? | sessions (and trials) | none (same reason) |
| `dod` | `within` minus `within_masking` (masking-based delta-of-deltas) | sessions/trials | bootstrap only |
| `vs_ppc` | ALM sessions vs PPC-laser sessions, all trials (ALM sites) | sessions/trials | none |

Columns: `cohort animal genotype distribution site trial_class kind contrast unit stat diff ci_lo ci_hi boot_p perm_p
n_a n_b n_sessions_a n_sessions_b`. `diff` is a - b for the contrast named in `contrast` (e.g. `opto_vs_non_opto`).
`unit` is the bootstrap resampling unit: `trials` (stratified by stimulus bin; assumes trials exchangeable) or
`sessions` (whole sessions; the honest interval for anything that varies by session). `boot_p` is the two-sided
bootstrap p against 0; `perm_p` the permutation p where the randomisation allows one, else NaN.

Stats: `mu` (criterion / PSE of the cumulative-Gaussian fit; positive = more evidence needed to choose B),
`sigma` (slope), `lapse_low`, `lapse_high`, `accuracy`, `hard_accuracy`, `easy_accuracy`, `side_bias`
(P(B) - 0.5), `recency`, `win_stay`, `lose_shift`; ALM adds `reaction_time`, `reaction_time_jitter`.
`mu`/`sigma` are NaN where the fit was unreliable (sigma > 5 or |mu| > 0.99).

## Levels (`levels.csv`)

The observed value of every stat in every condition a contrast was built from, so a level can be read
without reconstructing it from differences. `(kind, phase)` names the condition: `within/opto` = laser-on
trials of opto sessions, `within/non_opto` = laser-off trials of opto sessions, `within_masking/*` the same
on sham sessions, `between/opto|masking` = all trials of each session set, `compensation/laser_off` =
laser-off trials of opto sessions. Columns: `kind phase n_trials n_sessions stat value` plus the labels.

## Trajectory (`trajectory.csv`)

One row per session of the phase (Hard-A and Hard-B together, since the phase alternates them), in
acquisition order (`order`). `distribution` and `session_type` are the session's own; `phase` is the folder.
Per session on all trials: the stats above, `pse` (lapse-corrected PSE), `pse_fixed` (PSE with sigma and lapses
pinned to the animal's expert-Uniform fit - only the criterion free), the `pse_dynamics` fit (`pse_start`,
`pse_end`, `pse_final`, `pse_tau`, `pse_trials_to_90`, `pse_censored`, `pse_shape_daic` = AIC(exponential) -
AIC(step), `pse_step_switch`), and the adaptation anchors: `prev_pse` (previous session's PSE; the expert-
Uniform PSE for the first session), `normative_pse` (at the animal's own psychometric `sigma`),
`convergence_final` = (end PSE - prev_pse) / (normative_pse - prev_pse), `delta_from_prev` = PSE - prev_pse.
`trajectory_curves.csv`: rolling PSE within each session (`trial` = trial within session; 100-trial windows on
Hard, 50 on Uniform), with the same convergence index.

Caveats that apply to every trajectory number: one session (~500 trials) at sigma 0.3-0.7 pins the PSE to about
+/-0.1 and tau only loosely; `pse_censored` = 1 means the fit did not plateau within the session; |shape_daic| < 2
is no evidence either way.

## Switches (`<cohort>/switches/`, pre-opto cohorts)

Adaptation across whole blocks after a distribution switch (blocks ≥ 1000 trials; short excursions are
folded into their neighbours). `switches.csv`: one row per animal × switch × stat — `pre_*` (last 250 trials
before the switch, 4-parameter fit), `normative_pse` (constant-σ observer at the animal's pre-switch σ),
the `pse_dynamics` fit on the whole block with shape pinned (`pse_tau`, `pse_final`, `pse_censored`,
`pse_shape_daic`, …), `trials_to_criterion` (first running window at ≥ 80 % convergence), `plateau` (mean
convergence over the last third of the block). `convergence.csv`: one row per animal × switch × method ×
bin — `method` ∈ `manuscript` (4-parameter fit per 50-trial bin, `convergence_clipped` to [0, 1], the
Fig. 5C recipe), `pinned` (only the criterion fitted per bin, unclipped), `pinned_running` (running 50-trial
windows, step 10). `sessions.csv`: per session within the block. `overnight.csv`: PSE over the last 100
trials of a session vs the first 100 of the next; `toward_pre` > 0 means the overnight change went back
toward the pre-switch criterion. `transition` ∈ first (Uniform→Hard), novel, return. Pages:
`psychometrics.csv` / `psychometric_curves.csv`: one 4-parameter fit per animal × phase (the pre-switch
Uniform block, each Hard block in order — `Hard-B #1`, `Hard-A #1`, … — and all Hard-A / all Hard-B pooled).
`sessions.csv` carries the QC flags: `flagged` (|PSE| > 0.4, a lapse > 0.3, or accuracy < 0.6) and
`biased_animal` (majority of Hard sessions flagged with the same PSE sign on A and B blocks — a side bias that
ignores the distribution; thresholds in `adaptation.py`). `pre_post.csv`: PSE over the last 250 trials before each switch vs the last 250 of the block. Pages of
`summary_switches.pdf`, in order: cohort QC; psychometrics by phase (clean animals; Uniform = last 5 sessions
before the first switch); PSE before/after each switch; convergence (manuscript recipe, clipped, beside the
shape-pinned unclipped version); speed/shape/overnight/drift by transition; then psychometrics and convergence
for all animals. `pdf/<animal>_switches.pdf`: convergence per switch; psychometrics by phase.

## Reading the summary pages

- One marker per animal, WT green, HET red, vertical line = 95 % CI.
- Within-session rows: trial bootstrap; filled marker = permutation p < 0.05.
- Between-session rows: session bootstrap; filled marker = CI excludes 0. No p-value is shown because none is
  valid; the masking sessions were recorded after the laser sessions, so a between-session difference includes
  whatever changed over that time.
- "WT vs HET p" is a rank test on the per-animal differences; with 4 vs 5 animals its floor is 0.016
  (3 vs 4: 0.057), so per-animal consistency carries more weight than the group p.
- Hard pages rows 3-5: the whole A/B alternation session by session; dotted line = laser half | masking half;
  thin lines = animals, thick = genotype mean.
- Not on the pages: delta-of-deltas (in `contrasts.csv`), post-opto contrasts for Hard, readouts (in the PDFs).

## Regenerating

`sc-reports battery` (fast structure check, full battery, summary — each a run; `latest` ends on the full one).
Single pieces: `sc-reports opto-contrasts --distribution Hard-A --trial-class opto [--site alm_uni] [--level animal|group]`,
`sc-reports summary [--run <run_id>]`. See `docs/runs.md` for every command.
'''


def write_readme(root: Path) -> Path:
    import behav_utils

    import sound_categorisation
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'README.md'
    path.write_text(README.format(bu=behav_utils.__version__, sc=sound_categorisation.__version__))
    guide = Path(__file__).resolve().parents[2] / 'docs' / 'results_guide.md'
    if guide.exists():
        (root / 'GUIDE.md').write_text(guide.read_text())
    return path
