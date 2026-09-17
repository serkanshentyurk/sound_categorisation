# Guide to the opto results folder

This is the plain-language companion to the auto-generated `README.md` in the same folder (that one lists
every column; this one explains what the analyses mean and where to look).

## 1. Where things are

```
results/reports/opto1-cohort/
│
├── summary.pdf                 ← START HERE. Four pages: overall, Uniform, Hard-A, Hard-B.
├── summary_*.png               the same four pages as images
├── README.md                   technical reference (columns, conventions)
│
├── Uniform/                    one folder per distribution the animals were tested on
│   ├── ppc_opto/               laser on PPC, contrasting laser trials
│   ├── ppc_post_opto/          laser on PPC, contrasting the trial AFTER a laser trial
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

**Compensation contrast** — "laser-off trials inside the laser sessions minus masking sessions, using all trials".
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
- **Row 6** — the PSE again, but fitted with the curve's slope and lapses held fixed at the animal's Uniform
  values, so only the criterion can move. This is the steadier estimate when sessions are short and noisy.
- The dotted vertical line is where the laser sessions end and the masking sessions begin.

Two further numbers in `trajectory.csv` are **not** on the summary pages because they turned out to be
uninformative at this cohort's trial counts and noise levels (they are on the per-animal PDFs):

- **τ (tau)** — from a fit of how the PSE moves *within* one session, τ is the number of trials it takes the
  criterion to get about two-thirds of the way to where it settles. A value near or above the session length
  means it never settled ("censored").
- **convergence** — where the session ended on a scale from 0 (yesterday's PSE) to 1 (the PSE an ideal observer
  would adopt for today's distribution).

## 5. What is deliberately not shown

The summary pages leave out post-laser contrasts for Hard, the delta of deltas, the update matrices and the
psychometric curves themselves. All are in the per-animal and group PDFs and in the CSVs.

## 6. Regenerating

`bash run_reports.sh` rebuilds everything from the snapshot (a few hours). Single pieces:
`python -m sound_categorisation.reports group --with-animals --distribution Hard-A --toi opto`,
`python -m sound_categorisation.reports summary`.
