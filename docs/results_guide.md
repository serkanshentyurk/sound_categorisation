# Guide to the opto results folder

This is the plain-language companion to the auto-generated `README.md` in the same folder (that one lists
every column; this one explains what the analyses mean and where to look).

## 1. Where things are

```
results/opto_contrasts/opto1-cohort/<run_id>/        (../latest points at the newest run)
│
├── logs/                       job logs (cluster runs)
├── summary.pdf                 ← START HERE. Four pages: overall, Uniform, Hard-A, Hard-B.
├── summary_*.png               the same four pages as images
├── README.md                   technical reference (columns, conventions)
│
├── Uniform/                    one folder per distribution the animals were tested on
│   ├── ppc_opto/               laser on PPC, contrasting laser trials
│   ├── ppc_post_opto/          laser on PPC, the trial AFTER a laser trial vs the other laser-off trials
│   ├── alm_uni_opto/           laser on ALM, one hemisphere (site control)
│   ├── alm_bi_opto/            laser on ALM, both hemispheres
│   └── …post_opto variants
├── Hard-A/ppc_opto/ …          same layout; ALM was only run on Uniform
└── Hard-B/ppc_opto/ …
```

Inside each `<distribution>/<design>_<trials>/` folder:

| path | what it is |
|---|---|
| `pdf/<animal>_….pdf` | one animal, all pages: psychometric curves, update matrices, every contrast, the session trajectory |
| `pdf/group_….pdf` | all animals folded together: WT vs HET on every contrast, group curves, group trajectory |
| `<animal>/contrasts.csv` | every number behind that animal's contrast pages |
| `<animal>/equivalence.csv` | the laser effect read against margins set in advance: within margin, effect or unclear, per statistic |
| `<animal>/trajectory.csv` | one row per session, in the order they were run |
| `<animal>/readouts.npz` | the fitted curves and matrices (arrays) |
| `group/group_rows.csv` | the per-animal numbers that went into each WT-vs-HET test |
| `group/group_tests.csv` | the WT-vs-HET test results |
| `*/meta.json` | which snapshot, config and code version produced the folder |

If a summary panel looks odd, open the matching `group_….pdf`, then the animal's PDF, then the CSV. Nothing on
a summary page is computed there; it is all read from the CSVs, which are all computed from the snapshot.

## 2. The design, as the analysis sees it

Each animal has sessions of four kinds:

- **laser sessions** ("opto"): on 30 % of trials, chosen at random by the rig, a 473 nm laser hits PPC. In
  **WT** animals there is no opsin, so a laser trial is light in the tissue and nothing else. In **HET** animals
  (VGAT-ChR2) the same light silences PPC by driving inhibitory neurons.
- **masking sessions**: the masking LED runs as always, but the laser is set to zero power. The rig still flags
  which trials "would have been" laser trials. This controls for the masking light and for the trial flag; it
  does **not** control for what a non-zero laser does at the tissue — that is what the WT animals are for.
- **ALM sessions**: the same laser aimed at ALM (a motor area) instead of PPC. A site control: does the same
  light do the same thing somewhere else?
- **regular sessions**: expert Uniform sessions with no light at all; used as each animal's baseline.

Uniform was run in blocks (a masking block and a laser block). The Hard phase alternated Hard-A and Hard-B
**every day**: six laser sessions (A B A B A B), then six masking sessions (A B A B A B). So a Hard session is
one switch from yesterday's distribution, and the masking sessions were all recorded after the laser ones.

## 3. The contrasts, in words

Every contrast is a difference of the same set of behavioural statistics between two conditions. The
statistics are: **μ** (the criterion, or PSE — the stimulus value at which the animal is indifferent; positive
means it needs more B-like evidence to choose B, i.e. a bias toward A), **σ** (the slope of the psychometric
curve; larger = less sensitive), **lapses** (errors on the easiest stimuli), **accuracy**, **side bias**
(P(choose B) − 0.5), and the history statistics (recency, win-stay, lose-shift).

**Within-session contrast** — "laser trials minus non-laser trials, inside the laser sessions".
The cleanest comparison: the two trial types are interleaved at random in the same session, so nothing but the
laser differs. Because the rig randomised which trials got the laser, a permutation test is valid here: we
shuffle the laser labels many times and ask how often a difference this large appears by chance. A **filled**
marker on the summary pages means that permutation p < 0.05.

**Masking-session contrast** — "flagged trials minus the rest, inside the masking sessions".
The same computation on sessions where the laser was off. It should show nothing; it checks that the trial flag
and the masking light are inert. (They are.)

**Between-session contrast** — "laser sessions minus masking sessions, using all trials of each".
This asks whether whole sessions with the laser in play differ from whole sessions without it — for instance,
whether 30 % of trials being silenced changes how the other 70 % go, or how the animal adapts over a session.
It is a weaker comparison than the within-session one because the two session sets were recorded at different
times, and anything that changed between those times (practice, drift) ends up in the difference. No
permutation test is valid for it; the interval shown comes from resampling whole sessions, and a filled marker
means that interval excludes zero.

**Compensation contrast** — "laser-off trials of the laser sessions minus the unflagged trials of the masking
sessions".
Asks whether the animal's baseline moves in a session where a third of trials are lasered. In WT it does: the
laser-off trials sit below the masking sessions by about as much as the laser-on trials sit above them — the
animal offsets the light-driven push. Same statistical status as the between-session contrast (session
bootstrap, no permutation, order caveat).

**Delta of deltas** (in the CSVs, not on the summary pages) — "the within-session laser effect minus the
within-masking flagged-trial effect". The idea was to subtract whatever the trial flag or masking light does
from the laser effect. Since the masking-session contrast is null, this ends up equal to the within-session
contrast and adds nothing; it is kept in the tables for completeness.

**Genotype comparison** — every contrast is computed per animal, then WT and HET animals are compared with a
rank test on those per-animal numbers ("WT vs HET p" in the panel corner). With 4 WT and 5 HET the smallest p
this test can produce is 0.016; with 3 vs 4 it is 0.057. So a p of 0.016 means "as different as 4 vs 5 animals
can be", not "very strong evidence"; the strength of a result is in how many animals show the same thing.

## 4. The session trajectory (Hard pages, rows 4–6)

Instead of pooling the six Hard-A sessions, these rows show every session in the order it was run:

- **Row 4** — the criterion (PSE) fitted to each session's trials. Blue dots are Hard-A days, orange are
  Hard-B days. If the animals were tracking the daily switch, blue and orange would alternate up and down.
- **Row 5** — the change in PSE from the previous day: the size of each switch. A tracking animal alternates
  sign day by day.
- **Row 6** — the criterion again (the 50 % point), but fitted with the curve's slope and lapses held fixed, so
  only the criterion can move. The slope and lapses are those of the day's kind: all the animal's days of that
  distribution and session type pooled (Hard-A masking days, say), because some mice change their lapses between
  the distributions, and one shape for all of them would show that as a criterion shift. This is the steadier
  estimate when sessions are short and noisy. A session where the fit ends at the edge of the stimulus range (the
  mouse chose mostly one side) has no value here.
- The dotted vertical line is where the laser sessions end and the masking sessions begin.

Two further numbers in `trajectory.csv` are **not** on the summary pages because they turned out to be
uninformative at this cohort's trial counts and noise levels (they are on the per-animal PDFs):

- **τ (tau)** — from a fit of how the PSE moves *within* one session, τ is the number of trials it takes the
  criterion to get about two-thirds of the way to where it settles. A value near or above the session length
  means it never settled ("censored").
- **convergence** — on a scale from 0 (the criterion at the end of yesterday's session) to 1 (the criterion an
  ideal observer would adopt for today's distribution): `convergence_100_300` over trials 100–300 of the day,
  `convergence_final` over its last 250 trials. It is the switch report's measure (§5), here per day, laser days on their laser-off trials as there.

## 5. The switch report (both Hard designs)

`<cohort>/switches/` asks how fast the criterion follows a change of distribution. In the pre-opto cohort the
animals lived in each distribution for many sessions (Uniform → Hard-B → Hard-A → Hard-B → Hard-A), so a switch
is the start of a block; in the opto cohort's Hard phase the distribution alternated every day, so each day is a
switch (laser days measured on their laser-off trials). After each switch the criterion moves from where it was
(0: over the last 250 trials before the switch) towards the value an ideal observer would adopt (1). The criterion
is fitted with the curve's slope and lapses held at those of its phase — the block, or all the days of one
distribution and session type — in running 50-trial windows, and not clipped; a window where the fit ends at the
edge of the stimulus range (the mouse chose mostly one side) is left out. Holding one shape (the Uniform one) for
every phase would read a change of lapses between phases as a criterion shift: up to 0.09 in the blocked mice.

The one number per mouse and kind of switch depends on the design. For days, `convergence_100_300` averages
trials 100–300 after the switch: inside the first day, so the slow drift of the criterion across days hardly
enters; a mouse's five or six days of one kind are combined weighted by how far there was to go, so a day with
almost nothing to converge to cannot swamp the average. For blocks, `block_convergence` compares the whole
block's PSE with the previous phase's (each a 4-parameter fit on thousands of trials) and divides by how far the
ideal observer's criterion moved: 1 = moved by the normative amount, 0 = did not move. The curve's numbers do not
hold for one block: a mouse has one switch of each kind, the distance still to go can be 0.01–0.06, and a
50-trial window moves by 0.3–0.6 either way. A block whose PSE lies outside the stimulus range is not a
measurement and is left empty.

Page 1 is the bias check (`bias.csv`): every animal's Hard sessions in order, with accuracy, the criterion and
lapses, and a flag where the psychometric has collapsed. An animal whose Hard sessions are mostly flagged and lean
the same way on A and B days sticks to one side; it is marked BIASED, and the same list is used by every report.
Then the psychometrics by phase without the biased animals. Then one page per design: the curve after a switch
per kind of switch (mean over mice, with a 95 % interval for days; for blocks the mean alone, a description) and the one number per mouse —
for blocks by direction of switch. The criterion drifts slowly upwards across days, which works against
Hard-A → Hard-B, so a mouse above 0 there moved with the distribution despite the drift. For blocks there follow
the criterion before and after each switch, the manuscript's recipe (a full
4-parameter fit on each 50-trial bin, clipped to 0–1, as Fig. 5C did; the clipping pushes noisy bins upward), and
per switch τ, trials to 80 %, the plateau, gradual-vs-step, the overnight change and the drift within a block.
The overnight change is signed by the direction of the switch (> 0 = back towards the previous distribution);
it used to be signed by the end-of-session value itself, which made it positive from noise alone. The last pages
repeat these with every animal.

## 6. What is deliberately not shown

The summary pages leave out post-laser contrasts for Hard, the delta of deltas, the update matrices and the
psychometric curves themselves. All are in the per-animal and group PDFs and in the CSVs.

## 7. Regenerating

`sc-reports battery` rebuilds everything from the snapshot into a new run (a few hours). Single pieces:
`sc-reports opto-contrasts --distribution Hard-A --trial-class opto`, `sc-reports summary [--run <run_id>]`. Every command is
in `docs/runs.md`.
