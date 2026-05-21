"""
inference — Simulation-based inference modules

SBIFitter is the canonical high-level API. Low-level building blocks
(train_sbi, build_prior, build_simulator) are available for custom
pipelines.

Modules:
    fitting             — SBIFitter + training + posterior sampling
    comparison          — Cross-validated BE vs SC comparison
    simulation          — Model simulation for visualisation + timing
    types               — Parameter specification types (ConstantSpec, etc.)
    simulator           — Model simulators compatible with sbi package
    priors              — Multi-session prior construction
    amortised           — Amortised SBI for static model comparison
    diagnostics         — SBC, parameter recovery, calibration
"""

# ── Types ────────────────────────────────────────────────────────────────────
from inference.types import (
    ConstantSpec, GPSpec, RandomWalkSpec, IndependentSpec, HierarchicalSpec,
    ThetaLayout, LinkSpec, PARAM_CLAMP,
)

# ── Fitting (canonical API) ──────────────────────────────────────────────────
from inference.fitting import (
    SBIResult,
    SBIFitter,
    train_sbi,
    sample_posterior,
    train_per_animal_snpe,
    build_prior,
    build_simulator,
    compute_observed_stats,
)

# ── Comparison ───────────────────────────────────────────────────────────────
from inference.comparison import (
    compute_cv_comparison,
    compute_model_comparison,
)

# ── Simulation ───────────────────────────────────────────────────────────────
from inference.simulation import (
    simulate_all_sessions,
    simulate_example_session,
    estimate_timing,
    print_timing_report,
)

# ── Simulator factories ──────────────────────────────────────────────────────
# Lazy — only import if needed (avoids torch dependency at module load)
# from inference.simulator import create_be_simulator, create_sc_simulator

__all__ = [
    # Types
    'ConstantSpec', 'GPSpec', 'RandomWalkSpec', 'IndependentSpec',
    'HierarchicalSpec', 'ThetaLayout', 'LinkSpec',
    # Fitting
    'SBIResult', 'SBIFitter',
    'train_sbi', 'sample_posterior', 'train_per_animal_snpe',
    'build_prior', 'build_simulator', 'compute_observed_stats',
    # Comparison
    'compute_cv_comparison', 'compute_model_comparison',
    # Simulation
    'simulate_all_sessions', 'simulate_example_session',
    'estimate_timing', 'print_timing_report',
]
