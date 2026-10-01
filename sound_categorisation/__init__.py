"""
sound_categorisation — project code for the PPC / auditory-categorisation PhD.

Layered on ``behav_utils`` (the generic 2-AFC library). Subpackages follow the aims:

    settings     project constants: distributions, model types, fit sizes, thresholds, seeds
    data/        cohort (experiment, genotypes, session sets, AnimalRecord), snapshot, stimuli, paths
    behaviour/   contrasts (the PPC / ALM opto contrasts), adaptation (switch and per-session trajectories)
    models/      BE and SC generative models, perception, simulation, traces
    inference/   amortised SBI, simulator, representation, selection, grid search, CV utilities,
                 block-aware folds, GS+SBI consensus, the SLURM task grids
    features/    SBI feature-selection diagnostics
    reports/     compute → tables → figures → PDF → summary, and the report CLI
    plotting/    project plotters (CV, opto swarms, assignment, SBI diagnostics)
    cli/         entry points (sc-*): reports, export_snapshot, make_synthetic_cohort, new_run, train_sbi,
                 run_sbi, run_gs, consensus
"""

__version__ = '0.4.0'
