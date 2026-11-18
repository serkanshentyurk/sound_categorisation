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
    <animal>/trajectory_curves.csv       the criterion within each session (running 50-trial windows)
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
For `post_opto` the comparison is with the other non-laser trials (the post-opto trials are left out).

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
| `compensation` | laser-OFF trials of the laser sessions vs the unflagged trials of the masking sessions: does the baseline criterion move in a session where 30 % of trials are lasered? | sessions (and trials) | none (same reason) |
| `dod` | `within` minus `within_masking` (masking-based delta-of-deltas) | sessions/trials | bootstrap only |
| `vs_ppc` | ALM sessions vs PPC-laser sessions, all trials (ALM sites) | sessions/trials | none |

Columns: `cohort animal genotype distribution site trial_class kind contrast unit stat diff ci_lo ci_hi boot_p perm_p
n_a n_b n_sessions_a n_sessions_b`. `diff` is a - b for the contrast named in `contrast` (e.g. `opto_vs_non_opto`).
`unit` is the bootstrap resampling unit: `trials` (stratified by stimulus bin; assumes trials exchangeable) or
`sessions` (whole sessions; the right unit for anything that varies by session). A session interval from few
sessions is too narrow: with the three to six sessions a set has here, an interval meant to cover 95 % covers
roughly 80 to 90 %, so read it as rough and rest claims on the per-animal values and the tests across animals.
Sessions two trial sets share (`within`, `within_masking`) are drawn once for both, so those intervals are
paired. `boot_p` is the two-sided bootstrap p against 0; `perm_p` the permutation p where the randomisation
allows one, else NaN.

Stats: `mu` (criterion / PSE of the cumulative-Gaussian fit; positive = more evidence needed to choose B),
`sigma` (slope), `lapse_low`, `lapse_high`, `accuracy`, `hard_accuracy`, `easy_accuracy`, `side_bias`
(P(B) - 0.5), `recency`, `win_stay`, `lose_shift`; ALM adds `reaction_time`, `reaction_time_jitter`.
`mu`/`sigma` are NaN where the fit was unreliable (sigma > 5 or |mu| > 0.99).

## Equivalence (`equivalence.csv`)

Is the laser effect absent? A p-value cannot show that; an interval inside a margin fixed beforehand can. One
row per statistic that has a margin: the `within` contrast on the trials unit (`diff`, `ci_lo`, `ci_hi`), the
`margin` and the `verdict`: `within margin` (the whole 95 % interval inside +/- margin: any effect is smaller
than the margin), `effect` (the interval excludes zero and is not inside), `unclear` (neither: these data
cannot tell) or `no interval`. The margins are `settings.EQUIVALENCE_MARGINS`, recorded in `meta.json` under
`settings.margins`: `mu` and `sigma` 0.10, the other statistics 0.05. For HET animals on Uniform this is
prediction 1, PPC dispensable in experts.

## Levels (`levels.csv`)

The observed value of every stat in every condition a contrast was built from, so a level can be read
without reconstructing it from differences. `(kind, phase)` names the condition: `within/opto` = laser-on
trials of opto sessions, `within/non_opto` = laser-off trials of opto sessions, `within_masking/*` the same
on sham sessions, `between/opto|masking` = all trials of each session set, `compensation/laser_off` =
laser-off trials of opto sessions, `compensation/masking` = unflagged trials of masking sessions. Columns: `kind phase n_trials n_sessions stat value` plus the labels.

## Trajectory (`trajectory.csv`)

One row per session of the phase (Hard-A and Hard-B together, since the phase alternates them), in
acquisition order (`order`). `distribution` and `session_type` are the session's own; `phase` is the folder.
Per session on all trials: the stats above, `pse` (lapse-corrected PSE, all four parameters free), the
`pse_dynamics` fit (`pse_start`, `pse_end`, `pse_final`, `pse_tau`, `pse_trials_to_90`, `pse_censored`,
`pse_shape_daic` = AIC(exponential) - AIC(step), `pse_step_switch`) and the same with `_fixed` (shape fixed at
the session's phase). The criterion, measured as in the switch report (`behaviour/adaptation.py`): `pse_pinned`
is the 50 % point with sigma and lapses fixed at the fit of the session's phase — all the days of its
distribution and session type pooled, in the daily alternation (its block, for blocks) — given as
`shape_sigma`, `shape_lapse_low`, `shape_lapse_high`; missing when the fit ends at ±1 (the mouse chose mostly
one side). `pse_fixed` is the location parameter with the expert-Uniform shape, used by the bias check only.
`baseline_pse` is the criterion before the session (over the previous session's last 250 trials, with that
session's shape; the expert-Uniform PSE before the first), `normative_pse` the normative criterion at the
animal's expert-Uniform `sigma`, `pse_last250` the criterion over the session's last 250 trials. `convergence_100_300` = the mean over trials 100-300 of (criterion - baseline_pse)
/ (normative_pse - baseline_pse), not clipped; `convergence_final` the same at the session's end; `span` =
|normative_pse - baseline_pse|, the weight when sessions are combined. `prev_pse` is the previous session's
`pse` and `delta_from_prev` = pse - prev_pse (the day-to-day amplitude). `flagged` and `biased_animal` come
from the one bias check (`adaptation.bias_sessions`, as in the switch report). `trajectory_curves.csv`: the
criterion within each session in running 50-trial windows every 10 trials (`trial` = the window's centre),
with its convergence.

Caveats that apply to every trajectory number: one session (~500 trials) at sigma 0.3-0.7 pins the PSE to about
+/-0.1 and tau only loosely; `pse_censored` = 1 means the fit did not plateau within the session; |shape_daic| < 2
is no evidence either way.

The expert-Uniform baseline (`expert_pse`, the sigma of `normative_pse`, and the sigma and lapses used for
`pse_fixed` and for the Uniform phase) comes from the `expert_uniform` preset: Uniform sessions except masking, washout and ALM, the last half of them,
accuracy >= 0.70, then the last 5, laser-off trials only. For the opto cohort these are mostly laser sessions.
The plan was to take the baseline from masking sessions only; that is not done yet. `Settings.baseline_*`
changes it.

## Switches (`<cohort>/switches/`)

How fast the criterion follows a change of distribution, in both Hard designs: each switch into Hard-A or Hard-B
is the start of a block of at least 1000 trials (pre-opto cohorts; short excursions folded into their
neighbours) or, in the daily alternation (opto Hard phase), each day (`design` = blocked | daily, decided per
animal by the median run of days on one distribution). The curve: the criterion with sigma and lapses fixed at
the fit of its phase (one 4-parameter fit on the phase's pooled trials: the block, or every day of one
distribution and session type in the daily alternation; the Uniform phase takes the expert-Uniform fit), as its
50 % point, in running 50-trial windows; 0 = the criterion over the last 250 trials before the switch (with the
previous phase's shape), 1 = the normative criterion at the animal's expert-Uniform σ; not clipped; a window at
the ±1 edge is missing. One shape for every phase would show a change of slope or lapses between phases as a
criterion shift (up to 0.09 in the blocked mice). In the daily design a laser day is measured on laser-off
trials (its own and the day before's; `trials` = non_opto). `switch_kind`: blocks — the transition (first =
Uniform→Hard, novel, return); days — first day, laser day (after a laser day), masking day (after a masking day)
or mixed day.

The result per mouse depends on the design. Days: `convergence_100_300`, the curve averaged over trials 100-300,
five or six days of one kind combined. Blocks: `block_convergence` = (the block's pooled PSE − the previous
phase's) / (the normative shift), from 4-parameter fits on thousands of trials: 1 = moved with the distribution by
the normative amount, 0 = did not move. A mouse has one block switch of each kind, the distance to go can be
0.01-0.06 and a 50-trial window moves by 0.3-0.6, so the curve's numbers do not hold for one block. A slow
upward drift of the criterion across days adds to Hard-B → Hard-A and works against Hard-A → Hard-B, so a mouse
that moves with Hard-A → Hard-B does so despite it.

`switches.csv`: one row per animal × switch × stat — `pre_*` (last 250 trials before the switch: the
4-parameter fit, `pre_pse_pinned`, the criterion, and `pre_shape_*`, the previous phase's shape),
`normative_pse`, `normative_from` (before the switch), the `pse_dynamics` fit on the block (the phase's shape,
`shape_*`), `trials_to_criterion` (first running window at ≥ 80 % convergence), `plateau` (mean over the last
third), `convergence_100_300` (mean over windows centred on trials 100-300), `pse_post` and `convergence_final`
(the last 250 trials of the block or day), `span` (|normative - before|). `animals.csv`: one number per animal ×
kind of switch — days: `convergence_100_300` and `convergence_final`, the days combined weighted by span (the
distance moved over the distance to go, summed), which the plain mean of ratios is not: a switch with almost
nothing to converge to would swamp it; blocks: `block_convergence` (the mean over the kind's switches), the
curve's numbers left empty. `convergence.csv`: one row per animal × switch × method × window — `manuscript`
(4-parameter fit per 50-trial bin, `convergence_clipped` to [0, 1], the Fig. 5C recipe, for comparison),
`pinned` (the criterion per bin), `pinned_running` (the curve the pages and the thesis use). `sessions.csv`: per
session in the block. `overnight.csv`: the criterion over the last 100 trials of a session vs the first 100 of
the next; `toward_pre` signs the change by the direction of the switch (> 0 = back towards the previous
distribution). `psychometrics.csv` / `psychometric_curves.csv`: one 4-parameter fit per animal × phase (the
pre-switch Uniform block, each block in order for the blocked design, all Hard-A / all Hard-B pooled).
`bias.csv`: the one bias check — every Hard session on laser-off trials, `flagged` when |criterion| > 0.4 (the
location with the expert-Uniform shape), a lapse > 0.3 or accuracy < 0.6, and `biased_animal` when more than
half are flagged leaning the same way on Hard-A and Hard-B (settings in `adaptation.BiasRule`); `sessions.csv`
carries the same flags. `pre_post.csv` (blocks): the block-pooled 4-parameter PSEs before (`pse_pre`; the last
5 Uniform sessions before the first switch) and after (`pse_post`), `delta`, `normative_from`, `normative_pse`,
`normative_shift`, `block_convergence` (empty when either PSE lies outside ±1, the stimulus range: the mouse chose
mostly one side), and the criterion over the last 250 trials on each side.
Pages of `summary_switches.pdf`, in order: the bias check; psychometrics by phase (clean animals); one page per
design (the curves: for days with 95 % t intervals over mice, for blocks the mean alone, a description; one number per mouse: days
`convergence_100_300`, blocks `block_convergence` by direction); for blocks the PSE before/after each switch,
the manuscript recipe and speed/shape/overnight/drift by transition; then the same for all animals.
`pdf/<animal>_switches.pdf`: convergence per switch; psychometrics by phase.

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

`sc-reports battery` (check, opto-contrasts --all, light-artefact, summary — each a run; `latest` ends on the full ones).
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
