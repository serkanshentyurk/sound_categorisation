# Checking the animals: the daily tracker

For anyone who wants to see how the mice are doing without working with the code. One command copies the
recent session files from ceph and opens a page in the browser with every animal's last 14 days. It only
reads from ceph; nothing there is changed.

## What you need

- A Mac (Linux works the same way; for a Windows rig PC see the end).
- [conda](https://github.com/conda-forge/miniforge) and git.
- An SWC account that can log in to `ssh.swc.ucl.ac.uk` and read the experimenter's folder on ceph
  (`/ceph/akrami/<Name>/Head_Fixed_Behavior/Data`).
- Access to the repository on GitHub. If `git clone` below asks you to log in, the repository is private:
  ask to be added (see "Who to ask" in [START_HERE.md](START_HERE.md)).

## Set it up once

In a terminal:

```bash
conda create -n sound_cat python=3.11 -y
conda activate sound_cat
mkdir -p ~/sound_cat/repos
cd ~/sound_cat/repos
git clone https://github.com/serkanshentyurk/sound_categorisation.git
cd sound_categorisation
pip install -e .
```

Then tell it whose data to read and who you are on the SWC system. Open `~/.zshrc` in a text editor and add
these two lines, with the experimenter's folder on ceph and your own SWC username:

```bash
export BEHAV_DATA_DIR=/ceph/akrami/Serkan/Head_Fixed_Behavior/Data
export SWC_USER=<your SWC username>
```

Save the file, then run `source ~/.zshrc` (or open a new terminal).

## Every day

```bash
conda activate sound_cat
sc-daily-tracker
```

It asks for your SWC password twice (the gateway, then the cluster's login node), prints a line for each
step, and opens the page. Run it after the day's sessions are on ceph: the rigs send them there when someone
presses Push Data, usually at the end of the day, so the next morning is a good time.

The first run copies each animal's whole history, which takes a few minutes; after that only the new
day's files are copied. To open the newest page later without running anything, bookmark
`~/sound_cat/repos/sound_categorisation/results/daily_tracker/daily_tracker_latest.html`.

To rebuild the page without contacting ceph at all, for example when the cluster cannot be reached or after
changing a number in `config.yaml`, run `sc-daily-tracker --no-copy`: it uses the local copy from the last
run, and the page says at the top that it was not refreshed.

Options: `--days 7` (a shorter window), `--cohort <name>` (one cohort only), `--no-copy` (build from the
local copy only), `--no-open` (write the page without opening it), `--dry-run` (list what would be
copied, and stop).

## Reading the page

**Overview.** One row per animal, the ones that need a look first:

- the badge: *flag* (something on or since its latest session), *earlier* (flags earlier in the 14 days
  only) or *ok*;
- *latest*: the date of the latest session and its number (the animal's 23rd session reads `#23`);
- *trend*: accuracy over the 14 days (trials, while the animal is still on the early steps);
- the numbers of the latest session; hover over a column header for what it means;
- the flags, with their dates, and *move on*: whether the animal meets the criterion in the rig's
  `TRAINING.md` to go to the next training step (hover for why).

Animals that should be running but have no session on ceph are listed above the overview (see "Expected
animals" below).

**Flags.**

| Flag | Means |
|---|---|
| side bias | the session leans to one side: a shifted psychometric curve, a large lapse, or accuracy below 0.6 |
| low trials | fewer than 200 correct trials |
| accuracy drop | accuracy at least 0.10 below the animal's earlier sessions on the same preset |
| missing day | no session on ceph for a day when other animals had one: not run, or not pushed |

**An animal's page.** Click the animal. Each day of the window, newest first, with its session number;
plots against date, with a dashed line where the preset changed; the latest psychometric curve; and for each
condition (preset) its last five sessions, pooled and as mean ± SD next to each session, with trials that had
light on their own rows. ← and → move between animals, Esc goes back to the overview, and ↑ top returns
to the top of the page. The page's address ends in `#<animal>`, so it can be bookmarked or sent.

Statistics use trials without light. Right and left are the rig's right and left spouts.

## Where things are, and how to change them

| What | Where | Change it with |
|---|---|---|
| The session files it reads (on ceph) | `$BEHAV_DATA_DIR/Raw/<animal>/<session folder>/` | `BEHAV_DATA_DIR` |
| Who logs in to the SWC system | your SWC username | `SWC_USER` |
| The cluster login node it goes through | `hpc-gw2` | `SWC_NODE` (e.g. `hpc-gw1`) |
| The local copy | `~/sound_cat/data/raw_mirror/Raw/` | `SC_DATA_ROOT` (the copy goes in `$SC_DATA_ROOT/raw_mirror`) |
| The pages | `<repository>/results/daily_tracker/` | `SC_RESULTS_ROOT` (pages in `$SC_RESULTS_ROOT/daily_tracker`) |
| The window, thresholds, move-on criteria, expected cohorts | `config.yaml`, block `daily_tracker` | edit it (the comments explain each number) |

All of these are lines like `export NAME=value` in `~/.zshrc`; for a single run, put the setting in front of
the command, e.g. `BEHAV_DATA_DIR=/ceph/akrami/<Name>/Head_Fixed_Behavior/Data sc-daily-tracker`.

- **Another experimenter's data:** set `BEHAV_DATA_DIR` to their folder; the animals' folders must be in its
  `Raw` folder. To keep two people's copies and pages apart, give each its own `SC_DATA_ROOT` and
  `SC_RESULTS_ROOT`. These two also move the rest of the analysis's data and results, which matters only if
  you use the rest of the repository.
- **Only these two files are copied:** `Trial_Summary` and `Session_Parameters`; never videos, lick files or
  logs. The local copy can be deleted at any time: the next run copies it again.
- **After someone changes `config.yaml` or the code:** run `sc-update` in the repository folder to get the
  new version.

**Expected animals.** List the cohorts that should be running in `config.yaml`, under
`daily_tracker: expected_cohorts:` (a cohort is a named list of animals in the `cohorts:` block of the same
file). Their animals then appear at the top of the page whenever they have no session on ceph in the window.

## If something goes wrong

| You see | What to do |
|---|---|
| `sc-daily-tracker: command not found` | run `conda activate sound_cat` first |
| `SWC_USER is not set` | add `export SWC_USER=...` to `~/.zshrc` (see above) |
| `listing … failed (exit 255)` | the login node is not answering: run `SWC_NODE=hpc-gw1 sc-daily-tracker`; if that fails too, ask in `#ithpc`, and meanwhile `sc-daily-tracker --no-copy` shows the last copy |
| the password asked twice on every run | expected without ssh keys; `#ithpc` can help set keys up |
| `No animal has a session from the last 14 days` | nothing has been pushed recently; check Push Data on the rigs, or try `--days 30` |
| a mouse is missing from the page | its sessions are not on ceph (not run, or not pushed); list its cohort as expected to see it flagged |
| the page looks out of date | the newest file's time is at the top of the page; anything after it was not pushed yet |

## On a rig PC instead (Windows, ceph mapped as a drive)

Install the same way, in its own conda environment (never in the rig's). Then point `BEHAV_DATA_DIR` at the
experimenter's folder on the mapped drive (your drive letter may differ):

```bat
setx BEHAV_DATA_DIR "Z:\Serkan\Head_Fixed_Behavior\Data"
```

`setx` takes effect in terminals opened afterwards: open a new one, activate the environment and run
`sc-daily-tracker`. No SWC username or password is needed: the files are read from the drive.
