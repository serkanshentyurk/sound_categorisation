# Notebooks

Each notebook does two things: it says what produced its data and how to run that (the `sc-*` commands,
what they write), and it shows the results — one animal through the library pipeline first, then the
cohort. Notebooks read the tables an analysis run wrote; they do not recompute (no bootstraps in
notebooks). `nb_setup.py` holds the loaders and the cohort/run switches.

| notebook | story | reads |
|---|---|---|
| `00_data_and_task` | what is in the snapshot; the task from the trials | the snapshot (no run) |
| `10_expert_behaviour` | expert behaviour by phase: psychometrics, update matrices, trajectories; cohort by genotype | `opto_contrasts/<cohort>/latest` |
| `11_switch_adaptation` | PSE shift and convergence after a distribution switch (Fig. 5C), overnight, per-phase fits | `switch_adaptation/<cohort>/latest` |
| `20_opto_expert` | PPC inactivation in experts, the light itself (light-artefact report), ALM specificity, WT vs HET | `opto_contrasts/…/latest`, `light_artefact/…/latest` |
| `21_opto_hard` | opto on the Hard phases and the per-session trajectory (the design null) | `opto_contrasts/<cohort>/latest` |
| `30_sbi_feature_selection` | choosing the SBI summary-statistic vector (simulates; slow; cached) | nothing |
| `31_model_identification` | BE/SC recovery on synthetic data, consensus; real-data section pending the cluster run | `model_identification/<cohort>/latest` |
| `dev/` | model explorers, not maintained against the current API | — |

## Running them

On real data: fetch the snapshot, run the reports, open the notebooks. For `31`, fetch the
model-identification run from the cluster first.

```bash
sc-fetch snapshot
sc-reports battery
sc-reports switch-adaptation --cohort behaviour1-cohort
sc-fetch results model_identification/all-mice
jupyter lab notebooks/
```

Without data, or to see them on known-good synthetic input (this is what CI does):

```bash
sc-make-synthetic-run --results /tmp/sc_results --data /tmp/sc_data
SC_NB_SYNTHETIC=1 SC_RESULTS_ROOT=/tmp/sc_results SC_DATA_ROOT=/tmp/sc_data jupyter lab notebooks/
```

To execute them all, as CI does:

```bash
SC_NB_SYNTHETIC=1 SC_RESULTS_ROOT=/tmp/sc_results SC_DATA_ROOT=/tmp/sc_data pytest --nbmake notebooks/[0-2]*.ipynb notebooks/31_*.ipynb -q
```

`SC_NB_RUN=<run_id>` pins a run instead of `latest`. Outputs are not committed (strip before commit:
`jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`, or `nbstripout`).
