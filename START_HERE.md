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

### Model identification (cluster)

The SBI networks are trained once (`bash slurm/submit.sh sbi-train`) and kept in
`$BEHAV_DATA_DIR/Processed/snpe_networks/`. A model-identification run:

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
5. Export the snapshot again. Genotypes and `sessions_to_ignore` are read at export; session types and
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
2. The notebooks, in order: `00` → `10` → `11` → `20` → `21` → `31` ([notebooks/README.md](notebooks/README.md)).
3. [ARCHITECTURE.md](ARCHITECTURE.md), before changing code.
4. For experiments, the `behaviour_rig` documents: README and INSTRUCTIONS first, then TRAINING,
   PARAMETERS, HARDWARE, ZAPIT, SURGERY and PERFUSION.
5. The `behav_utils` README, if you change library code.

The BE and SC models and the model-identification recipe come from the lab's sound-categorisation
manuscript; ask Serkan or Athena Akrami for the current draft.

## Before you analyse: caveats in the data

- Data recorded before 5 October 2026 is compromised by rig faults fixed that day. On rigs 31104 and
  31105 the right reward was about 26 % larger than the left. In Hard-A/B sessions some trials played no
  sound (they are removed at export) and the response window opened as the sound ended. The same animal
  ran on different rigs on different days, so no session is dropped and the data carry no rig id: state
  it as a caveat.
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

- Expert phase (opto1 cohort, 5 HET and 4 WT): analysed.
- After switches: the daily Hard-A/B alternation showed no measurable adaptation; a blocked design is needed.
- Model identification on all mice (October 2026 run): 12 BE, 3 SC, 4 unclear. The SLDS and disRNN
  analyses are not written; networks for the Hard distributions are not trained.
- Imaging: not started.
