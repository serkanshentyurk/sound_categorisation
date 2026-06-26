"""Shared plotting style constants for the project plotting modules."""

# Per-parameter colours, used across SBI posterior / diagnostic plots.
PARAM_COLOURS = {
    'sigma_percep': '#1f77b4',   # blue
    'A_repulsion':  '#ff7f0e',   # orange
    'eta_learning': '#2ca02c',   # green
    'eta_relax':    '#d62728',   # red
    'gamma':        '#9467bd',   # purple (SC)
    'sigma_update': '#8c564b',   # brown  (SC)
}

# Per-condition colours for the opto / masking session breakdown.
CONDITION_COLOURS = {
    'baseline': '#1f77b4', 'regular': '#1f77b4', 'masking': '#ff7f0e',
    'all_opto': '#2ca02c', 'opto_off': '#9467bd', 'opto_on': '#d62728',
    'post_opto': '#8c564b',
}

# Per-distribution colours and per-session-type markers for the experiment
# overview / timeline plots. Hard-A / Hard-B are this project's distributions,
# so these live project-side, not in the task-agnostic behav_utils.
DIST_COLOURS = {'Uniform': '#1f77b4', 'Hard-A': '#d62728', 'Hard-B': '#2ca02c'}
TYPE_MARKERS = {'regular': 'o', 'masking': 's', 'opto': '^', 'washout': 'D'}
