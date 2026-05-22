"""
inference — Simulation-based inference for BE and SC models.

AmortisedSBI: Train once on curriculum data, condition on many animals.
            Used for static BE vs SC selection, matches GS-CV protocol.
SBIFitter:    Per-animal training using the animal's stimulus sequence.
            Supports time-varying parameters (ConstantSpec, GPSpec,
            RandomWalkSpec). Used for dynamic parameter trajectories.
"""
from inference.types import (
    ModelType, ParamConfig, ThetaLayout, LinkSpec,
    ConstantSpec, GPSpec, RandomWalkSpec,
    get_default_param_configs, get_default_links,
)
from inference.fitting import (
    SBIResult, SBIFitter,
    train_sbi, sample_posterior,
    build_prior, build_simulator, compute_observed_stats,
    train_per_animal_snpe,
)
from inference.amortised import (
    AmortisedSBI, compute_pooled_stats, compute_observed_stats_from_sessions,
)
from inference.comparison import compute_cv_comparison, compute_model_comparison