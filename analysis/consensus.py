"""
Model Assignment Consensus

Combines the per-method BE/SC calls into a single consensus label per animal.
This is the *canonical* consensus logic — notebooks import from here, never
reimplement it.

Each "method" is a (source, rep, fit_target) triple resolving to a results
directory written in the neutral CV schema (save_cv_result). load_cv_results +
compare_models turn that directory into per-animal winners + p-values, so this
module never re-reads pickles or recomputes winners by hand — it only loads,
joins, and votes. That keeps it in lock-step with run_gs / run_sbi via the
shared results_dir() convention and the shared on-disk schema.

Methods (configurable):
    ('grid_search', None, ft) -> GS-UM / GS-CP      dir: grid_search/{run}/{cohort}_{ft}
    ('sbi', rep, ft)          -> SBI-{rep}-UM / -CP  dir: sbi/{run}/{cohort}_{ft}/{rep}

DEFAULT_METHODS mirrors the old four-method scheme (GS + SBI `pooled`, both fit
targets); add 'moments'/'single' once NB12 shows a rep is trustworthy enough to
vote.

Consensus rule (configurable):
    1. Collect significant (p < alpha) BE/SC votes across methods.
    2. Fewer than min_significant_votes significant -> 'Unclear'.
    3. Strict majority on one model -> that model.
    4. Tie / no majority -> 'Split'.
"""

import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING

from utils.cv_utils import load_cv_results
from scripts.config import results_dir

if TYPE_CHECKING:
    from behav_utils.data.structures import ExperimentData


FT_LABEL = {'update_matrix': 'UM', 'conditional_psych': 'CP'}

# A method is (source, rep, fit_target). rep is None for grid search.
Method = Tuple[str, Optional[str], str]

DEFAULT_METHODS: List[Method] = [
    ('grid_search', None, 'update_matrix'),
    ('grid_search', None, 'conditional_psych'),
    ('sbi', 'pooled', 'update_matrix'),
    ('sbi', 'pooled', 'conditional_psych'),
]


# ── method -> label / directory ──────────────────────────────────────────────

def _method_label(source: str, rep: Optional[str], fit_target: str) -> str:
    ft = FT_LABEL[fit_target]
    if source == 'grid_search':
        return f'GS-{ft}'
    if source == 'sbi':
        return f'SBI-{rep}-{ft}'
    return f'{source}-{rep}-{ft}'


def _method_dir(source: str, rep: Optional[str], fit_target: str,
                run: str, cohort: str) -> Path:
    d = results_dir(source, run, cohort, fit_target)
    return d / rep if (source == 'sbi' and rep) else d


# ── consensus rule (row-based; auto-detects method columns) ──────────────────

def _compute_consensus(row: dict, alpha: float = 0.05,
                       min_significant_votes: int = 1) -> str:
    """Majority vote over a row's method columns.

    A method column K is any key with a sibling '{K}_p'. K contributes a vote
    when its value is 'BE'/'SC' and its p-value is < alpha. The auto-detection
    keeps the rule agnostic to which methods are present.
    """
    votes = []
    for key in list(row):
        p_key = f'{key}_p'
        if p_key not in row:
            continue
        val, p = row.get(key), row.get(p_key)
        if val in ('BE', 'SC') and pd.notna(p) and p < alpha:
            votes.append(val)

    if len(votes) < min_significant_votes:
        return 'Unclear'
    top_model, top_count = Counter(votes).most_common(1)[0]
    return top_model if top_count > len(votes) / 2 else 'Split'


# ── public API ───────────────────────────────────────────────────────────────

def load_all_assignments(
    run: str,
    cohort: str,
    methods: Optional[List[Method]] = None,
    experiment: Optional['ExperimentData'] = None,
    alpha: float = 0.05,
    min_significant_votes: int = 1,
) -> pd.DataFrame:
    """Load each method's BE/SC call and compute a consensus per animal.

    Args:
        run: Run label (e.g. 'full'), as passed to run_gs / run_sbi.
        cohort: Cohort label (synthetic cohort name, or e.g. 'real').
        methods: (source, rep, fit_target) triples; defaults to DEFAULT_METHODS.
        experiment: If given, animals present in it but absent from results are
            still listed (an all-missing row -> 'Unclear').
        alpha, min_significant_votes: consensus-rule parameters.

    Returns:
        One row per animal: 'id', then per method '{label}', '{label}_p',
        '{label}_be', '{label}_sc'; 'true_model' if any method carried ground
        truth; 'Consensus'; and 'consensus_correct' when truth is known.
    """
    methods = methods or DEFAULT_METHODS

    per_method = {}                 # label -> {animal_id: (winner, p, be, sc)}
    true_model_by_animal = {}
    for source, rep, ft in methods:
        label = _method_label(source, rep, ft)
        cv = load_cv_results(_method_dir(source, rep, ft, run, cohort))
        comp = cv.comparison
        rows = {}
        if comp is not None and len(comp):
            for _, r in comp.iterrows():
                rows[r['animal_id']] = (r.get('winner'), r.get('p_value'),
                                        r.get('be_mean'), r.get('sc_mean'))
                tm = r.get('true_model')
                if tm in ('BE', 'SC'):
                    true_model_by_animal[r['animal_id']] = tm
        per_method[label] = rows

    ids = set().union(*[set(d) for d in per_method.values()]) if per_method else set()
    if experiment is not None:
        exp_ids = getattr(experiment, 'animal_ids', None)
        if exp_ids is None:
            exp_ids = list(getattr(experiment, 'animals', {}))
        ids |= set(exp_ids)

    has_truth = bool(true_model_by_animal)
    out_rows = []
    for aid in sorted(ids):
        row = {'id': aid}
        for label, rows in per_method.items():
            winner, p, be, sc = rows.get(aid, (None, np.nan, np.nan, np.nan))
            row[label] = winner
            row[f'{label}_p'] = p
            row[f'{label}_be'] = be
            row[f'{label}_sc'] = sc
        if has_truth:
            row['true_model'] = true_model_by_animal.get(aid)
        row['Consensus'] = _compute_consensus(
            row, alpha=alpha, min_significant_votes=min_significant_votes)
        if has_truth:
            tm = true_model_by_animal.get(aid)
            row['consensus_correct'] = (row['Consensus'] == tm
                                        if tm in ('BE', 'SC') else None)
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def compute_consensus_summary(assign_df: pd.DataFrame) -> str:
    """Printable consensus summary (counts per label; accuracy if truth known)."""
    if 'Consensus' not in assign_df or len(assign_df) == 0:
        return 'no assignments'
    counts = assign_df['Consensus'].value_counts()
    lines = [f'{len(assign_df)} animals total', 'Consensus:']
    for label, n in counts.items():
        lines.append(f'  {label}: {n}')
    if 'consensus_correct' in assign_df:
        valid = assign_df['consensus_correct'].dropna()
        if len(valid):
            lines.append(f'consensus accuracy (vs truth): '
                         f'{valid.mean():.2f} (n={len(valid)})')
    return '\n'.join(lines)
