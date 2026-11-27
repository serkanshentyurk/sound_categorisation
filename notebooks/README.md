# Notebooks

Each notebook does two things: it says what produced its data and how to run that (the `sc-*` commands,
what they write), and it shows the results — one animal through the library pipeline first, then the
cohort. Notebooks read the tables an analysis run wrote and do not resample; the one exception is `10`,
which computes point estimates from the snapshot. `nb_setup.py` holds the loaders and the cohort/run
switches.

They follow the argument: the data and the task, how expert animals behave, which strategy explains their
updates, whether they adapt when the distribution changes, and then what silencing PPC does — in experts and
on the Hard phases.

| notebook | story | reads |
|---|---|---|
| `00_data_and_task` | what is in the snapshot; the task from the trials | the snapshot (no run) |
| `10_expert_behaviour` | expert Uniform behaviour without light: psychometrics, history effects, update matrices, all mice | the snapshot (computes point estimates) |
| `11_model_identification` | BE or SC per animal (descriptive): the all-mice run, synthetic recovery as the check; each model's held-out error over the noise floor, beside a no-history reference (absolute fit); which fitted model predicts the Hard sessions (blocks and the daily alternation) | `model_identification/<cohort>/latest`, `noise_floor/<cohort>/latest`, `hard_predictions/<cohort>/latest` for both cohorts |
| `12_switch_adaptation` | the criterion after a distribution switch (blocks): shift, convergence (Fig. 5C), overnight, did it follow the distribution | `switch_adaptation/<cohort>/latest` |
| `13_adaptation_speed` | whether and how fast the criterion follows the distribution in both designs (chapter 3, Fig 3.6): the steadied Fig. 5C curve for days, one number per mouse (days: trials 100–300; blocks: the whole block's share of the normative shift, by direction), the blocks switch by switch, overnight | `switch_adaptation/<cohort>/latest` for both cohorts |
| `20_opto_expert` | PPC inactivation in experts, the light itself (light-artefact report), ALM specificity, WT vs HET, equivalence; whether a silenced trial is remembered on the next one (each animal's fitted models with learning blocked on laser trials, against the observed post-laser effect) | `opto_contrasts/…/latest`, `light_artefact/…/latest`, `blocked_update/…/latest` |
| `21_opto_hard` | opto on the Hard phases and the per-session trajectory (partial adaptation at most to disrupt) | `opto_contrasts/<cohort>/latest` |
| `90_sbi_feature_selection` | appendix: choosing the SBI summary-statistic vector (simulates; slow; cached) | nothing |
| `91_sbi_validation` | appendix: SBI networks against the current code, calibration and coverage, parameter and model recovery (synthetic cohort, both consensus rules), the real animals' σ and calls | `model_identification/synthetic_uniform_learning_k100/latest` (`tables/`, `network_check/`) |
| `dev/` | model explorers, not maintained against the current API | — |

## Running them

On real data: fetch the snapshot, run the reports, open the notebooks. For `11` and the last section of `20`,
fetch the model-identification run from the cluster first.

```bash
sc-fetch snapshot
sc-reports battery
sc-reports switch-adaptation --cohort behaviour1-cohort
sc-reports switch-adaptation --cohort opto1-cohort
sc-fetch results model_identification/all-mice
sc-noise-floor --cohort all-mice
sc-blocked-update --fits all-mice/<run id> --jobs 4
sc-hard-predictions --fits all-mice/<run id> --cohort behaviour1-cohort --jobs 8
sc-hard-predictions --fits all-mice/<run id> --cohort opto1-cohort --jobs 8
jupyter lab notebooks/
```

Without data, or to see them on known-good synthetic input (this is what CI does):

```bash
sc-make-synthetic-run --results /tmp/sc_results --data /tmp/sc_data
SC_NB_SYNTHETIC=1 SC_RESULTS_ROOT=/tmp/sc_results SC_DATA_ROOT=/tmp/sc_data jupyter lab notebooks/
```

To execute them all, as CI does:

```bash
SC_NB_SYNTHETIC=1 SC_RESULTS_ROOT=/tmp/sc_results SC_DATA_ROOT=/tmp/sc_data pytest --nbmake notebooks/[0-2]*.ipynb -q
```

`SC_NB_RUN=<run_id>` pins a run instead of `latest`. Outputs are not committed (strip before commit:
`jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`, or `nbstripout`).
