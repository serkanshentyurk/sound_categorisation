"""
inference — Simulation-based inference for BE and SC models.

AmortisedSBI: Train once on curriculum data, condition on many animals.
              Used for static BE vs SC selection (matches GS-CV protocol),
              and for per-SLDS-state fits by conditioning on each state's
              pooled trials.
compute_cv_comparison / compute_model_comparison:
              Held-out UM/CP evaluation of the trained posterior — the
              SBI-UM / SBI-CP votes in the four-method consensus.

Module structure:
    types.py      — ModelType, ParamConfig, get_default_param_configs
    constants.py  — SBI_STATS (the ten heuristic summary stats)
    simulator.py  — Simulator, create_be/sc_simulator, wrap_for_sbi (CV path)
    amortised.py  — AmortisedSBI, build_curriculum_simulator, pooled stats
    comparison.py — held-out CV / model comparison
"""
from inference.types import (
    ModelType, ParamConfig, get_default_param_configs,
)
from inference.constants import SBI_STATS
from inference.simulator import (
    Simulator, SimulatorConfig,
    create_be_simulator, create_sc_simulator,
    get_sbi_prior, wrap_for_sbi,
)
from inference.amortised import (
    AmortisedSBI,
    compute_pooled_stats,
    compute_observed_stats_from_sessions,
    simulate_choices_from_params,
    build_curriculum_simulator,
)
from inference.comparison import compute_cv_comparison, compute_model_comparison
