"""analysis — Real-data behavioural and model-selection analyses.

For synthetic-data validation: see validation/
For shared utilities and math primitives: see utils/
"""
from analysis.consensus import compute_consensus_summary, load_all_assignments
from analysis.grid_search import (
    compute_grid_search_cv,
    simulate_model_matrices, ParameterGrid, DEFAULT_GRID, COARSE_GRID,
)
from analysis.adaptation import detect_shifts