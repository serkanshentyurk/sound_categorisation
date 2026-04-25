"""
Inference Module for BE and SC Models.

Provides simulation-based inference infrastructure:
- Summary statistics computation
- SBI-compatible simulators (BE and SC)
- Prior definitions (single and multi-session)
- SBI training and posterior sampling
- Diagnostics (SBC, parameter recovery, posterior predictive)

Quick Start:
    from inference import (
        create_be_simulator,
        create_prior,
        train_sbi,
    )

    # Setup
    simulator = create_be_simulator(stimuli, categories, burn_in=100)
    prior = create_prior()

    # Train
    result = train_sbi(simulator, prior, observed_stats, method='NPE')

Note:
    Full functionality requires: pip install torch sbi
    Basic simulator and summary_stats work without torch.
"""

# Summary statistics (no torch dependency)
from behav_utils.analysis.summary_stats import (
    compute_summary_stats,
    compute_stats_for_sbi,
    list_available_stats,
    DEFAULT_STATS,
    DEFAULT_N_BINS,
)

# Simulator (no torch dependency at import)
from inference.simulator import (
    create_be_simulator,
    Simulator,
    SimulatorConfig,
    ModelType,
)

# Try to import torch-dependent modules
_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

if _TORCH_AVAILABLE:
    # Priors (requires torch)
    from inference.priors import (
        create_prior,
        create_multisession_prior,
        UniformPrior,
        MultiSessionPrior,
        LinkingConfig,
        DEFAULT_BE_BOUNDS,
    )
    
    # SBI wrapper (requires torch + sbi)
    try:
        from inference.fitting import (
            train_sbi,
            sample_posterior,
            quick_posterior,
            compare_methods,
            train_multisession_sbi,
            SBIResult,
        )
        
        # Diagnostics (requires torch)
        from inference.diagnostics import (
            run_sbc,
            parameter_recovery,
            plot_sbc_ranks,
            plot_sbc_ecdf,
            plot_recovery_scatter,
            plot_recovery_bias,
            recovery_summary_table,
        )
    except ImportError as e:
        import warnings
        warnings.warn(f"SBI functionality not available: {e}. Install with: pip install sbi")


__all__ = [
    # Summary stats (always available)
    'compute_summary_stats',
    'compute_stats_for_sbi',
    'list_available_stats',
    'DEFAULT_STATS',
    'DEFAULT_N_BINS',
    # Simulator (always available)
    'create_be_simulator',
    'Simulator',
    'SimulatorConfig',
    'ModelType',
]

if _TORCH_AVAILABLE:
    __all__.extend([
        # Priors
        'create_prior',
        'create_multisession_prior',
        'UniformPrior',
        'MultiSessionPrior',
        'LinkingConfig',
        'DEFAULT_BE_BOUNDS',
        # SBI
        'train_sbi',
        'sample_posterior',
        'quick_posterior',
        'compare_methods',
        'train_multisession_sbi',
        'SBIResult',
        # Diagnostics
        'run_sbc',
        'parameter_recovery',
        'plot_sbc_ranks',
        'plot_sbc_ecdf',
        'plot_recovery_scatter',
        'plot_recovery_bias',
        'recovery_summary_table',
    ])