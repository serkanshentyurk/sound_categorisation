"""
sound_categorisation — project code for the PPC / auditory-categorisation PhD.

Layered on ``behav_utils`` (the generic 2-AFC library):

    cohort       load the experiment, genotypes, per-animal session sets, AnimalRecord providers
    contrasts    the PPC / ALM opto contrasts as typed DeltaStats / Interaction
    stimuli      Hard-A / Hard-B stimulus densities, normative PSE
    models       BE and SC generative models
    inference    amortised SBI (network training, conditioning, representation)
    grid_search  CV grid search over model parameters, update-matrix MSE
    consensus    GS + SBI model-identification consensus
    tasks        one task-index grid per SLURM array (train / condition / grid search)
    validation   SBI feature selection diagnostics
    reports      per-animal / group PDF report builders
    plotting     project-specific plotters (CV, opto swarms, assignment)
    paths        data / results locations and run metadata
    snapshot     experiment snapshot export / load
"""

__version__ = '0.3.0'
