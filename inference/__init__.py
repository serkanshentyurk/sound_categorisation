"""
inference — Simulation-based inference for BE and SC models.

AmortisedSBI: Train once on curriculum data, condition on many animals.
              Used for static BE vs SC selection, matches GS-CV protocol.
SBIFitter:    Per-animal training using the animal's stimulus sequence.
              Supports time-varying parameters (ConstantSpec, GPSpec,
              RandomWalkSpec). Used for dynamic parameter trajectories.

Module structure (after fitting.py split):
    sbi_core.py — SBIResult, train_sbi, sample_posterior
    builders.py — build_prior, build_simulator, compute_observed_stats
    fitter.py   — SBIFitter class
    train.py    — train_per_animal_snpe (script-style entry)
"""
from inference.types import (
    ModelType, ParamConfig, ThetaLayout, LinkSpec,
    ConstantSpec, GPSpec, RandomWalkSpec,
    get_default_param_configs, get_default_links,
)
from inference.sbi_core import (
    SBIResult, train_sbi, sample_posterior,
)
from inference.builders import (
    build_prior, build_simulator, compute_observed_stats,
    DEFAULT_SUMMARY_STATS, DEFAULT_BE_PARAM_LINKS, DEFAULT_SC_PARAM_LINKS,
)
from inference.fitter import SBIFitter
from inference.train import train_per_animal_snpe
from inference.amortised import (
    AmortisedSBI, compute_pooled_stats, compute_observed_stats_from_sessions,
)
from inference.comparison import compute_cv_comparison, compute_model_comparison
