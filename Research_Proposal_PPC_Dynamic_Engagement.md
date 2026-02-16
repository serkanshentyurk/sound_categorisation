# Dynamic Engagement of Posterior Parietal Cortex During Statistical Context Learning Versus Inference

## Updated Research Proposal

---

## Overarching Hypothesis

The posterior parietal cortex (PPC) is necessary for statistical context learning — when animals actively update beliefs about the distributional structure of task stimuli — but becomes dispensable for inference from learned context, when animals apply stable statistical knowledge to make decisions without online updating. When the statistical context changes, PPC re-engages for context adaptation: updating the existing statistical model, not re-learning from scratch.

The primary measure distinguishing these computational states is the effective learning rate (η). We predict that η transitions coherently with PPC necessity, neural population dynamics, and behavioural strategy: high η during context learning (PPC necessary), low η during inference (PPC dispensable), and moderate η during context adaptation (PPC necessary again, but less than during naive learning).

**Critical distinction:** "Dispensable" refers to loss of causal necessity — no behavioural deficit when PPC is inactivated — not loss of neural representation. PPC neurons may remain active during inference, but this activity is not required for behaviour (activity ≠ necessity; Lohuis et al. 2022). This maintained activity constitutes a "monitoring mode" that enables detection of distributional changes and subsequent re-engagement.

**Context learning versus category learning:** To distinguish PPC's role in statistical context learning from simpler category learning, we employ repeated distribution switches. If PPC is needed only for learning novel statistical structure, then early distribution switches (when the animal has not yet experienced context changes) should require PPC, while late switches (when the animal has meta-learned the structure of context changes) should not — even though the sensory and motor demands are identical.

---

## Behavioural Paradigm

All experiments use a 2-AFC sound categorisation task based on Akrami et al. (in prep). Mice categorise sound stimuli as belonging to category A or B, with a fixed decision boundary throughout.

**Training trajectory:**

1. **Uniform distribution** — train from naive to expert
2. **Repeated cycling** — alternate between hard-A asymmetric and hard-B asymmetric distributions, each until the animal has adapted

The category boundary remains constant; only the statistical distribution of stimuli around the boundary changes. This isolates statistical context learning from rule learning or sensorimotor remapping.

**Adaptation criterion:** An animal is considered adapted when its psychometric curve (specifically the PSE) is not statistically different from the normative model prediction (Akrami et al., in prep), sustained for 1–2 additional sessions. Convergence is quantified as:

Convergence(t) = (PSE(t) − PSE_pre-switch) / (PSE_normative − PSE_pre-switch)

When Convergence ≈ 1.0 and stable, the animal has reached the adapted/expert state. This criterion is applied per-animal to respect individual learning speeds and is grounded in the published normative model rather than arbitrary thresholds.

---

## Aim 1: Modelling Animal Behaviour Across Learning

**Goal:** Characterise the full behavioural trajectory from naive to expert to post-shift adaptation, develop both model-free and model-based methods to define learning epochs, and validate that the effective learning rate from the BE/SC model tracks with model-free behavioural signatures.

### 1.1 Track behavioural trajectory from naive to expert to post-adaptation

Train mice on the 2-AFC sound categorisation task with the distribution manipulation schedule described above. Track the following session-level behavioural measures across the full training trajectory:

**Psychometric measures:**

- Psychometric curve parameters (PSE, slope, lapse rates) from cumulative Gaussian fits
- Goodness-of-fit (R²) of psychometric fits
- Session-to-session changes in psychometric parameters

**Choice strategy measures:**

- Overall accuracy and binned accuracy (by stimulus distance from boundary)
- Side bias
- Win-stay and lose-switch indices
- Choice randomness / choice entropy
- Recency effect (regression weight of recent trials on current choice)

**Temporal measures:**

- Reaction time distribution (median, variance, skewness)
- Session-to-session changes in all of the above

### 1.2 Model-free learning epoch categorisation

Develop a data-driven method to categorise training sessions into learning epochs — naive, intermediate, expert, post-shift, adapting, adapted — without relying on the BE/SC model. This provides a model-independent definition of "when is the animal learning versus performing," which is essential for interpreting Aims 2 and 3 without circular reasoning (i.e., not defining epochs by the very model we are trying to validate).

**Candidate approaches:**

- Changepoint detection on the behavioural metric trajectories from 1.1
- Clustering of sessions in multi-dimensional behavioural feature space
- The normative model convergence metric as a principled anchor point (Convergence ≈ 1.0 = adapted)

**Validation:** Split-half within animals — define epochs using one set of behavioural metrics (e.g., psychometric parameters and accuracy), then show that held-out metrics (e.g., recency index, choice entropy, RT distribution) also change at those boundaries. If epochs defined by performance metrics predict transitions in history-dependence metrics that were not used to define them, this provides convergent evidence.

### 1.3 Fit BE/SC model to multisession data using SBI

Use Simulation-Based Inference (SBI) to fit the Boundary Estimation and/or Stimulus-Category model to multisession behavioural data. The model-free behavioural statistics from 1.1 serve as summary statistics for the SBI procedure. A subset of the statistics listed above will be selected based on their informativeness for distinguishing model parameters (to be determined through sensitivity analysis and parameter recovery validation).

**Extract session-by-session parameter trajectories, primarily:**

- η_learning (learning rate for belief updating) — the primary parameter of interest
- η_relax (relaxation rate)
- σ_percep (perceptual noise)
- A_repulsion (serial dependence / repulsion)

### 1.4 Relate model parameters to model-free statistics and learning epochs

**Core prediction:** η_learning decreases as animals transition from naive to expert epochs (as defined by the model-free method in 1.2). After each distribution shift, η_learning shows a moderate increase — higher than expert but lower than naive — reflecting context adaptation rather than naive re-learning. Across repeated A↔B cycles, the post-shift η increase should diminish as the animal meta-learns the context structure.

**Additional predictions:**

- High η_learning sessions correspond to high recency effects, high session-to-session parameter changes, and high choice entropy
- Low η_learning sessions correspond to central tendency bias, stable psychometric curves, and low session-to-session variability
- The model-free epoch boundaries from 1.2 align with transition points in the η_learning trajectory

**Statistics:**

- Correlation between η and model-free metrics across sessions within animals
- Mixed-effects models testing whether η differs significantly across model-free epochs
- Comparison of epoch transition points from model-free classification versus η trajectory changepoints

**Why this matters:** This aim establishes the analytical foundation for Aims 2 and 3. If η reliably tracks with model-free behavioural signatures and learning epochs, it validates using η (and the model-free epoch definitions) as the independent variable for interpreting optogenetic and imaging results.

---

## Aim 2: Optogenetic Inactivation of PPC Across Learning and Trial Epochs

**Goal:** Test whether PPC is causally necessary at different stages of learning (training epochs) and for different within-trial computations (trial epochs). Use the repeated distribution cycling design to distinguish context learning from context switching, and model the effect of inactivation within the BE/SC framework.

### 2.1 Inactivate during different training epochs

**Design:**

- 3 batches of animals (n per batch to be determined by power analysis):
    - Batch 1: No optogenetic inhibition (control trajectory)
    - Batch 2: Inhibit during choice window (stimulus onset → reward delivery)
    - Batch 3: Inhibit during update window (reward delivery → end of ITI)
- All batches undergo the same training trajectory: uniform to expert, then repeated hard-A ↔ hard-B cycling
- 30% of trials within each session are interleaved optogenetic trials (within-session control)
- Test at multiple training epochs:
    - **Expert phase** (low η, stable psychometric matching normative prediction)
    - **Early distribution shifts** (first 1–2 A↔B transitions; novel contexts)
    - **Late distribution shifts** (after 3+ transitions; familiar contexts)
    - **Naive phase** (tentative — low baseline performance complicates interpretation)

**Core predictions — training epochs:**

*Expert phase (low η):*
No behavioural impairment in either Batch 2 or Batch 3 relative to Batch 1. PPC is dispensable for both choice computation and belief updating when the statistical context is stable.

*Early distribution shifts (moderate-to-high η, novel contexts):*
Prolonged adaptation in Batch 3 (update window) relative to Batch 1: more sessions to reach convergence criterion, slower η decrease. Batch 2 (choice window) may also show impairment if PPC is needed to override the old A1→striatum mapping. The stronger prediction from the framework is that the update window effect should dominate, because PPC's hypothesised role is specifically updating the statistical context.

*Late distribution shifts (low η, familiar contexts):*
No behavioural impairment in either batch. The animal has meta-learned the context structure and can switch between known distributions via inference/recall, without requiring PPC. This is the critical test distinguishing context learning from mere context switching — if PPC is dispensable for late shifts but necessary for early shifts, PPC's role is specifically in learning new statistical structure, not in switching between contexts per se.

*Adapted phase (low η):*
Effects should disappear as animals return to inference mode with the current statistics.

### 2.2 Inactivate during different trial epochs

The within-trial temporal dissection is implemented via the batch structure:

- **Choice window (stimulus onset → reward delivery):** Tests whether PPC is needed to use the current belief to make a decision. Encompasses stimulus encoding, evidence accumulation, and decision.
- **Update window (reward delivery → end of ITI):** Tests whether PPC is needed to update the belief based on feedback. Encompasses the computation that adjusts the statistical context model.

If only the update window shows effects during early distribution shifts, it supports the specific claim that PPC's role is in belief updating (the η computation), not in reading out beliefs for choices. If both windows show effects, PPC may be involved in both using and updating context during adaptation.

### 2.3 Model the optogenetic effect in the BE/SC framework

If inactivation during a specific trial epoch lengthens adaptation, simulate the effect within the BE/SC model by "lesioning" that computation on 30% of trials (matching the experimental design). For example, if the update window is critical: on optogenetic trials, set η_learning = 0 (no belief update occurs), and test whether the model reproduces the observed slowing of adaptation.

This creates a closed loop between Aims 1 and 2: the model predicts which trial epoch should matter, the optogenetics tests it, and the model accounts for the magnitude of the effect.

**Primary measures:**

- Accuracy difference between optogenetic and control trials (within-session)
- Psychometric curve parameters on optogenetic versus control trials
- Sessions to reach adaptation criterion (between-batch, per shift)
- η trajectory across post-shift sessions (between-batch)
- Convergence trajectory (between-batch, per shift)
- Adaptation speed as a function of cycle number (early vs late shifts)
- Reaction time differences
- Model-derived KL divergence between session belief distributions (optogenetic vs control sessions)

**Statistics:**

- Within-session: mixed-effects logistic regression on trial-level choices with optogenetic condition as predictor, controlling for stimulus difficulty
- Between-batch: comparison of adaptation trajectories using survival analysis (sessions to criterion) and mixed-effects models on η trajectories
- Critical comparison: interaction between batch (opto window) and shift number (early vs late) — tests whether PPC necessity diminishes with meta-learning

**Key falsification:**

- If PPC remains necessary during late distribution shifts (when η is low and contexts are familiar), this challenges the context learning versus inference distinction
- If PPC remains necessary during expert phase when η is stably low, the core hypothesis fails
- If neither window shows effects during early shifts, PPC may not be necessary for adaptation at all (challenges the framework)

---

## Aim 3: Imaging PPC Dynamics Across Learning

**Goal:** Track excitatory and inhibitory PPC neurons across the full learning trajectory using chronic two-photon calcium imaging. Model population dynamics as a switching linear dynamical system (SLDS) fitted blind to behavioural labels, and test whether neural regime transitions precede behavioural transitions defined in Aim 1.

This aim uses a separate cohort from Aim 2. The link between PPC necessity (Aim 2) and PPC dynamics (Aim 3) is established through the shared behavioural framework from Aim 1: the same model-free epoch definitions and model-based η trajectories characterise both cohorts. If feasible and time permits, a subset of imaging animals will receive optogenetic inactivation during expert and early post-shift sessions to directly bridge Aims 2 and 3.

### 3.1 Image excitatory and inhibitory cells over the course of training

**Design:**

- Chronic two-photon calcium imaging through cranial window (3–4 mm) over PPC
- Imaging system: Mesoscope (large field of view, multi-area capability)
- Genetic strategy: VGAT-Cre × GtACR1-flox (R26-LNL-GtACR1-Fred-Kv2.1) + AAV-hSyn-soma-jGCaMP8s
    - jGCaMP8s in all neurons (soma-targeted for cleaner signals)
    - Fred in inhibitory neurons (red fluorescent marker for identification)
    - GtACR1 in inhibitory neurons (for optional manipulation, connecting to Aim 2 logic)
- Track 150–250 neurons across 12–14 weeks
- Image every session for the full training trajectory (naive → expert → repeated A↔B cycling)
- Cell extraction via Suite2P or CaImAn; manual verification of cross-session tracking (>80% neuron tracking threshold)

**Measures per session:**

- Single-cell encoding via GLM with stimulus, choice, previous stimulus, previous choice, and interaction terms
- Classification of neurons as mixed-selective (significant interactions) versus stimulus-selective (main effect only)
- Excitatory versus inhibitory identity from Fred expression
- E/I activity ratio
- Pairwise functional connectivity (noise correlations, cross-correlograms) between E-E, E-I, and I-I pairs
- Opponent inhibition motif strength: functional connectivity between opposite-preference E-I pairs (Kuan et al. 2024)
- Choice-selective inhibition strength: trial-by-trial choice consistency, attractor stability metrics (Roach et al. 2023)
- Population dimensionality (PCA participation ratio)

### 3.2 Model population dynamics as a switching linear dynamical system

Fit a switching linear dynamical system (SLDS) to the neural population data **blind to behavioural epoch labels**. The SLDS infers discrete latent states (dynamical regimes), continuous latent dynamics within each regime, and transition probabilities between regimes.

**Core prediction:** The SLDS identifies discrete regime transitions that align with — and critically, *precede* — the behavioural epoch transitions defined in Aim 1.

Specifically:

- Neural dynamics shift from a high-dimensional regime (mixed selectivity, weak inhibitory structure) to a low-dimensional regime (stimulus selectivity, strong inhibitory structure) **before** the behavioural measures stabilise into the expert pattern
- After a distribution shift, neural dynamics return to a higher-dimensional regime **before** the behavioural η increases
- Across repeated A↔B cycles, the duration of the "learning regime" shortens in parallel with faster behavioural adaptation

**Why "precede" matters:** If neural regime changes lead behavioural changes, it provides evidence that the neural dynamics are driving the behavioural transition rather than merely reflecting it. This is the difference between PPC tracking computational state changes versus PPC implementing them.

**Fallback:** If transitions are simultaneous rather than sequential, this is still consistent with PPC implementing the transition (behavioural readout may simply be fast). The falsification is if behavioural changes consistently *precede* neural regime changes, which would suggest PPC is following rather than leading. If SLDS struggles with gradual transitions (discrete-state assumption may not hold), continuous-state models (e.g., Gaussian Process latent variable models) serve as a fallback.

### 3.3 Relate neural dynamics to learning epochs

**Within the SLDS framework, test the following predictions:**

*E-I circuit dynamics:*

- Opponent inhibition motif strength (functional connectivity between anti-selective E-I pairs) increases during the transition to inference and decreases during post-shift adaptation
- Choice-selective inhibition strength tracks with SLDS regime identity: strong in the inference regime, weak in the learning regime
- E/I activity ratio correlates with the η trajectory from Aim 1: high ratio during learning, low (more balanced) ratio during inference

*Single-cell encoding:*

- The same individual neurons transition from mixed selectivity (context learning) to stimulus selectivity (inference) to mixed selectivity (context adaptation) — flexible remapping, not population recruitment
- Individual neurons that transition encoding properties are the same neurons whose dynamics change regime in the SLDS

*Population-level:*

- Population dimensionality (participation ratio) is high during learning regimes and low during inference regimes
- SLDS learning regime corresponds to: high dimensionality, weak opponent inhibition motif, weak choice-selective inhibition, high E/I ratio
- SLDS inference regime corresponds to: low dimensionality, strong opponent inhibition motif, strong choice-selective inhibition, low E/I ratio

**Statistics:**

- Cross-validated comparison of SLDS transition timepoints versus Aim 1 behavioural epoch boundaries
- Permutation tests for whether alignment exceeds chance
- Temporal precedence analysis (Granger causality or cross-correlation between neural regime probability and behavioural metrics)
- Within-regime analysis of E-I metrics with bootstrap confidence intervals
- Within-neuron transition matrices: what proportion of mixed-selective neurons become stimulus-selective (and vice versa), tested against null model where transitions are random

---

## Integration Across Aims

The three aims are designed to provide converging evidence at three levels:

**Aim 1 (Behaviour + Model)** defines when the animal is learning versus performing, providing the temporal scaffold for Aims 2 and 3.

**Aim 2 (Optogenetics)** tests causal necessity at each learning phase and trial epoch, asking *when* and *for what computation* PPC is required.

**Aim 3 (Imaging)** reveals the neural dynamics underlying these transitions, asking *how* PPC implements context learning and *why* it becomes dispensable during inference.

The repeated distribution cycling design runs through all three aims, providing the critical test of context learning versus context switching: PPC is necessary for early shifts (novel contexts, high η) but dispensable for late shifts (familiar contexts, low η), and the neural dynamics reflect this distinction.

---

## Key Falsifications

**The framework fails if:**

- PPC remains necessary during expert phase when learning rates are stably low
- PPC remains necessary during late distribution shifts when contexts are familiar
- Learning rate is low but neural dimensionality remains high (η does not track with neural state)
- Different neuronal populations are recruited for different phases rather than the same neurons remapping
- E-I dynamics show no systematic changes across learning phases
- Neural regime transitions consistently follow rather than precede or co-occur with behavioural transitions

**Requires nuanced interpretation:**

- Post-shift η fully returns to naive levels (suggests complete re-learning, not adaptation)
- Both trial epoch windows show equal impairment (PPC involved in both choice and update, need richer model)
- Neither trial epoch window shows impairment during early shifts (PPC not necessary for adaptation, challenges framework)
- E and I populations do not transition together (challenges Najafi et al. 2020 predictions)
