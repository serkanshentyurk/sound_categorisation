# Start here

This is the way into the sound-categorisation project of the Akrami Lab (Sainsbury Wellcome Centre,
UCL): head-fixed mice categorise sounds, the stimulus distribution is switched, and posterior parietal
cortex (PPC) is silenced optogenetically. The question is whether PPC is needed while the animal updates
its statistical model of the sounds, and not once that model fits. The project uses three
repositories; this page says how they fit together and how to get going.

## The three repositories

| repository | what it does |
|---|---|
| `behaviour_rig` | runs the experiments: the Bonsai workflow, the operator GUI, the presets; pushes each session's files to ceph |
| `sound_categorisation` (this one) | the analysis of this project: the snapshot, the reports, BE/SC model identification |
| [`behav_utils`](https://github.com/serkanshentyurk/behav_utils) | a general 2-AFC analysis library the project is built on |

How the data moves:

```
rig PC (Bonsai)            Push Data                  →  ceph  $BEHAV_DATA_DIR/Raw
cluster                    sc-export-snapshot         →  ceph  $BEHAV_DATA_DIR/Processed/behaviour/snapshots/
laptop                     sc-fetch snapshot          →  <X>/data/behaviour/snapshots/
laptop                     sc-reports ...             →  <repo>/results/   →  notebooks/
cluster (Slurm)            model identification       →  ceph  $BEHAV_DATA_DIR/Processed/results/
```

## Who to ask

- HPC and computing: the `#ithpc` Slack channel, and lab mates.
- The rig: Serkan Shentyurk and [Quentin Pajot-Moric](https://www.sainsburywellcome.org/people/quentin-pajot-moric).
- Other access: [Lillianne Teachen](https://www.sainsburywellcome.org/people/lillianne-teachen) or other lab members.

## First steps

1. Get an SWC account with access to the cluster and to ceph (`/ceph/akrami/`), and access to the three
   repositories on GitHub.
2. Make your data folder on ceph: `/ceph/akrami/<Name>/Head_Fixed_Behavior/Data`. The rig writes into
   its `Raw` folder, the analysis into its `Processed` folder.
3. Install the project on the cluster and on your laptop: [SETUP.md](SETUP.md).
4. Do the one-time settings below.

## Your own data folder

Every path the analysis uses hangs off one setting, `BEHAV_DATA_DIR`. Nothing in the repositories needs
editing. Set it once on each machine.

On the cluster, add to `~/.bashrc`:

```bash
export BEHAV_DATA_DIR=/ceph/akrami/<Name>/Head_Fixed_Behavior/Data
alias load-env='cd ~/repos/sound_categorisation && source slurm/env.sh'
```

On your laptop, add to `~/.zshrc`:

```bash
export BEHAV_DATA_DIR=/ceph/akrami/<Name>/Head_Fixed_Behavior/Data
export SWC_USER=<your SWC username>
```

`sc-fetch` and `sc-daily-tracker` reach ceph through the gateway and the login node `hpc-gw2`. If that
node does not answer, add `export SWC_NODE=hpc-gw1` to `~/.zshrc` to go through `hpc-gw1` instead, and
remove the line once `hpc-gw2` is back.

If you run experiments, the rig needs to know your folder too: in the `behaviour_rig` repository copy
`GUI/paths/serkan.json` to `GUI/paths/<name>.json` and set `"data_path"` to
`/<Name>/Head_Fixed_Behavior/Data/Raw`. It must end in `/Raw`, because that is where the analysis looks.

To work on Serkan's cohorts, use his folder: `BEHAV_DATA_DIR=/ceph/akrami/Serkan/Head_Fixed_Behavior/Data`.
Runs are stamped with a date and a commit, so they do not overwrite each other; re-exporting replaces the
snapshot.

If `BEHAV_DATA_DIR` is not set on the cluster, the code stops and says so. It never falls back to
someone else's folder.

## Everyday routines

### Get the latest code

On the cluster:

```bash
load-env
sc-update
```

On the laptop, `sc-update` does the same. If you work on `behav_utils` next to the project (an editable
install), use `sc-update --keep-behav-utils`: it installs everything else and leaves your copy alone.

### New sessions → snapshot (cluster)

The snapshot is the pickled experiment every analysis reads. Export it after new sessions, or after
changing column mappings in `config.yaml`:

```bash
load-env
sc-export-snapshot
```

### Snapshot → laptop

```bash
sc-fetch snapshot
```

It goes through the SWC gateway to a login node, so it asks for your password twice unless you have
ssh keys set up (ask in `#ithpc`).

### Reports (laptop)

```bash
sc-reports switch-adaptation --cohort behaviour1-cohort
caffeinate -i sc-reports battery --jobs 6
```

The switch report takes minutes. The battery takes about six hours one animal at a time; `--jobs 6` runs six
animals at once (use your number of cores minus two) and gives the same numbers. `caffeinate -i` only keeps
the Mac awake. Results land in `<repo>/results/`, one folder per run; `latest` points at the newest. Then
open the notebooks:

```bash
jupyter lab notebooks/
```

### Daily behaviour tracker (laptop or rig PC)

How each animal is doing, read straight from the session files on ceph: no snapshot, no cluster job.
[DAILY_TRACKER.md](DAILY_TRACKER.md) is the step-by-step version for someone who only wants to look at the
animals (setting it up, reading the page, changing whose data it reads, what to do when it fails).

```bash
sc-daily-tracker
```

It shows every animal with a session in the last 14 days, writes
`results/daily_tracker/daily_tracker_<date>.html` and opens it. Sessions reach ceph only when Push Data is
pressed, so run it after the day's push (the next morning, say). Options: `--days 7`, `--cohort <name>`,
`--no-open`, `--no-copy` (rebuild the page from the local copy, without contacting ceph), and `--dry-run`,
which lists what would be copied and changes nothing; use it on the first run.

The page opens on an overview: the animals that need a look first (a flag on or since their latest
session), each with a 14-day trend, its latest session (date and session number) and the move-on mark.
Click an animal for its own page: every day of the window, newest first, values against date with
preset changes marked, the latest psychometric curve, and for every condition (preset) its last five
sessions, pooled and as mean ± SD next to the individual sessions, light trials on their own rows. The
arrow keys step through the animals and Esc goes back; `daily_tracker_latest.html` is always the newest
report, for a bookmark. Early training steps show trials only. Flags: side bias, low trials, accuracy
drop, and a missing day ("not run, or not pushed"). Animals of the cohorts listed in
`daily_tracker.expected_cohorts` that have no session on ceph are listed at the top.

It copies only `Trial_Summary` and `Session_Parameters` files (never videos, lick files or logs) to
`<X>/data/raw_mirror/`: the shown animals' whole history the first time, then new files only. On a laptop
the files come over ssh as `SWC_USER`, like `sc-fetch` (the password twice per run without ssh keys;
`SWC_NODE` picks the login node, see "Your own data folder"). On a
rig PC they come from the mapped drive: install the project in its own environment as in SETUP.md (a
plain `pip install -e .`, no `[fit]`; never into the rig's environment) and set `BEHAV_DATA_DIR` to the
data folder on that drive, e.g. `setx BEHAV_DATA_DIR "Z:\Serkan\Head_Fixed_Behavior\Data"`, then open a
new terminal.

Every number it uses (the 14 days, the five sessions, the flag thresholds, the move-on criteria taken from
`behaviour_rig/TRAINING.md`) is in `config.yaml`, block `custom.daily_tracker`, with a comment on each. To change
one: edit it, commit, then run `sc-update` on the PC that runs the tracker. The page's last section lists
the values it used. The side-bias thresholds are `BiasRule`'s in `sound_categorisation/behaviour/adaptation.py`,
shared with the switch report.

### Model identification (cluster)

The SBI networks are trained once (`bash slurm/submit.sh sbi-train`) and kept in
`$BEHAV_DATA_DIR/Processed/snpe_networks/`; a network trained with a different representation is not overwritten
(move it into a dated subfolder first). How far the networks and the calls can be trusted is a separate validation
run on a synthetic cohort (`docs/runs.md`, "SBI validation"; notebook 91). A model-identification run:

```bash
load-env
RUN=$(sc-new-run --report model_identification --cohort all-mice)
bash slurm/submit.sh grid-search --cohort all-mice --distribution uniform --fit-target update_matrix --run-id $RUN
bash slurm/submit.sh sbi-condition --cohort all-mice --distribution uniform --run-id $RUN
```

When the jobs have finished:

```bash
load-env
sc-grid-search --cohort all-mice --distribution uniform --fit-target update_matrix --run-id $RUN --gather
sc-consensus --cohort all-mice --distribution uniform --run-id $RUN
```

To look at it on the laptop:

```bash
sc-fetch results model_identification/all-mice
```

Every command, its options and what it writes: [docs/runs.md](docs/runs.md).

## New animals

1. On the rig: a row for the mouse in `Params/Mouse_Room_Params.xlsx`, in the `behaviour_rig` repository
   (its INSTRUCTIONS.md).
2. Its cohort: add it to the cohort's list under `cohorts:` in `config.yaml`, or add a new cohort line.
3. Its genotype: add it to `$BEHAV_DATA_DIR/Raw/animal_metadata.json`, for example
   `"SS24": {"genotype": "het"}`. An animal without an entry is left out of WT vs HET comparisons; the
   reports warn "animal(s) without genotype".
4. Session types the data cannot show (masking, ALM, washout) go by date under `session_types:` in
   `config.yaml`; sessions to leave out go under `sessions_to_ignore:`.
5. Export the snapshot again. Genotypes, `sessions_to_ignore` and session types are read at export;
   presets are read from `config.yaml` every time the snapshot is loaded.

A cohort member with no data yet is left out of the reports with a warning.

## Where things are

| what | on ceph (cluster) | on the laptop |
|---|---|---|
| raw sessions | `$BEHAV_DATA_DIR/Raw/<animal>/SOUND_CAT_<animal>_<Y>_<M>_<D>/` | not needed |
| genotypes | `$BEHAV_DATA_DIR/Raw/animal_metadata.json` | in the snapshot |
| snapshot | `$BEHAV_DATA_DIR/Processed/behaviour/snapshots/sound_cat_snapshot.pkl` | `<X>/data/behaviour/snapshots/` |
| SBI networks | `$BEHAV_DATA_DIR/Processed/snpe_networks/` | `<X>/data/snpe_networks/` |
| synthetic cohorts | `$BEHAV_DATA_DIR/Processed/synthetic_cohorts/` | `<X>/data/synthetic_cohorts/` |
| results | `$BEHAV_DATA_DIR/Processed/results/<report>/<cohort>/<run>/` | `<repo>/results/<report>/<cohort>/<run>/` |
| cohorts, session types, presets | `config.yaml` in this repository | same |

`<X>` is the folder that holds `repos/`: clone this repository into `<X>/repos/sound_categorisation` and
the laptop data goes to `<X>/data`. `SC_DATA_ROOT` and `SC_RESULTS_ROOT` move them elsewhere.

Every results folder has a `meta.json` (run id, command, snapshot, settings, versions, commit) and a
`README.md` describing every table and page.

## What to read, in order

1. The project [README](README.md), then [docs/runs.md](docs/runs.md) and
   [docs/results_guide.md](docs/results_guide.md).
2. The notebooks, in order: `00` → `10` → `11` → `12` → `20` → `21` ([notebooks/README.md](notebooks/README.md)).
3. [ARCHITECTURE.md](ARCHITECTURE.md), before changing code.
4. For experiments, the `behaviour_rig` documents: README and INSTRUCTIONS first, then TRAINING,
   PARAMETERS, HARDWARE, ZAPIT, SURGERY and PERFUSION.
5. The `behav_utils` README, if you change library code.

The BE and SC models and the model-identification recipe come from the lab's sound-categorisation
manuscript; ask Serkan or Athena Akrami for the current draft.

## Before you analyse: caveats in the data

- Data recorded before 5 October 2026 is compromised by rig faults fixed that day. On rigs 31104 and
  31105 the right reward was about 26 % larger than the left. In Hard-A/B sessions about 6 % of trials
  played no sound and the response window opened as the sound ended. The same animal ran on different rigs
  on different days, so no session is dropped and the data carry no rig id: state it as a caveat.
- The silent trials are removed at export where each session's logs allow the check. They don't for the end
  of every run (the logs stop when Bonsai is killed) or for 13 whole Hard sessions of the pre-opto cohort
  (SS01, SS06, SS07, SS08, SS09, SS11), where the trial table and the logs don't line up. Those trials are
  kept: 15 % of Hard trials are unchecked, so roughly 1 % of Hard trials may still be silent.
- The first 20 trials of a session (the stable start) play easy sounds, at least 0.8 from the boundary,
  with at most 3 on the same side in a row.
- The first trial of a session has unreliable timing. It is kept.
- Sessions on the same day share a folder and load as one session.
- Masking sessions run the Zapit protocol at laser power 0: the LED is on, the brain is not lit. Their
  flagged trials are the control for the laser trials.
- On laser trials the sound and the delay can end tens of ms early (the trial waits for Zapit). Masking
  trials presumably do the same, since they also run through Zapit; if so, the laser-minus-masking
  difference (`dod`) controls for it, while a within-session laser contrast alone does not.
- Lick times on the Arduino rigs lag by about 25 ms.
- There is no go cue; only licks that start after the response window opens count.

## Decisions, and why

- Within-session contrasts use a trial bootstrap and a permutation p (the laser was randomised per
  trial); between-session contrasts use a session bootstrap and no p; genotype comparisons are rank tests
  on per-animal effects.
- The light-controlled effect is `dod`: the laser effect minus the masking effect.
- Animals that stick to one side, on both Hard-A and Hard-B, are dropped by the bias check
  (`adaptation.BiasRule`).
- The BE/SC call copies the manuscript's model selection: repeated cross-validation over seeds, the
  lower-error model if p < 0.05.
- The Hard-session baseline comes from the `expert_uniform` preset, not from masking sessions only as
  first planned; the report README says so.

## Status (November 2026)

- Expert phase, PPC (opto1 cohort, 5 HET and 4 WT): no large effect of silencing in HET animals, but the null
  is only partly shown: against margins set in advance, about 2 of 5 HET animals are within the margin for the
  criterion, accuracy and lapses, and every HET animal is unclear for the history statistics. Light alone
  shifts the criterion in WT animals, and the side bias in both genotypes (notebook 20).
- Adaptation: the daily Hard-A/B alternation (opto1) produced none. The blocked switches (behaviour1) show a
  partial shift, about half the ideal size, which the order of the blocks may partly explain. The opto cohort
  did not adapt, so the post-shift prediction could not be tested.
- Model identification on all mice (October 2026 run): 12 BE, 3 SC, 4 unclear; a description of the animals,
  too few for a BE/SC split of the opto effects. Superseded in Nov 2026: the networks were conditioned on CV halves
  (2–3 sessions) outside their 4–6-session training design, so they are retrained with the amount of data as an
  input and the run is redone, with a validation run (notebook 91).
- Not done: the learning phase (state-space models), imaging. Networks for the Hard distributions and the
  single-session representation are not planned: the opto cohort's Hard sessions all carry light, the blocked
  shift is partial and may be drift, and a single-session voter cannot reach p < 0.05 with ≤ 5 sessions.
