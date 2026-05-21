"""
Model Assignment Consensus

Loads GS + SBI assignments for all animals and computes a single
consensus label per animal.  This is the *canonical* consensus logic —
notebooks should import from here, not reimplement.

Consensus rule (configurable):
    1. Collect significant (p < alpha) assignments from all methods.
    2. If ≥ min_significant_votes are significant AND a strict majority
       agree on one model → that model.
    3. If votes are tied or no majority → 'Split'.
    4. If fewer than min_significant_votes are significant → 'Unclear'.

Usage:
    from analysis.consensus import load_all_assignments

    assign_df = load_all_assignments(results_dir, experiment)
    # Returns DataFrame with columns:
    #   id, GS-UM, GS-UM_p, GS-CP, GS-CP_p,
    #   SBI-UM, SBI-UM_p, SBI-CP, SBI-CP_p, Consensus
"""

import pickle
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING
from scipy.stats import wilcoxon

if TYPE_CHECKING:
    from behav_utils.data.structures import ExperimentData


# ── Constants ────────────────────────────────────────────────────────────────

FIT_TARGETS = ['update_matrix', 'conditional_psych']
FT_LABEL = {'update_matrix': 'UM', 'conditional_psych': 'CP'}
METHOD_COLS = ['GS-UM', 'GS-CP', 'SBI-UM', 'SBI-CP']


# ── Internal helpers ─────────────────────────────────────────────────────────

def _load_gs_assignment(
    animal_id: str,
    cv_dir: Path,
    distribution: str,
    fit_target: str,
) -> dict:
    """Load GS assignment for one animal × one fit target."""
    ft_short = FT_LABEL[fit_target]
    be_path = cv_dir / f'{distribution}_{fit_target}' / f'cv_{animal_id}_BE.pkl'
    sc_path = cv_dir / f'{distribution}_{fit_target}' / f'cv_{animal_id}_SC.pkl'

    if not be_path.exists() or not sc_path.exists():
        return {}

    with open(be_path, 'rb') as f:
        be_data = pickle.load(f)
    with open(sc_path, 'rb') as f:
        sc_data = pickle.load(f)

    be_errors = [
        r['avg_test_error'] for r in be_data['results']
        if not np.isnan(r.get('avg_test_error', np.nan))
    ]
    sc_errors = [
        r['avg_test_error'] for r in sc_data['results']
        if not np.isnan(r.get('avg_test_error', np.nan))
    ]

    if not be_errors or not sc_errors:
        return {}

    be_mean = np.mean(be_errors)
    sc_mean = np.mean(sc_errors)
    winner = 'BE' if be_mean < sc_mean else 'SC'

    n = min(len(be_errors), len(sc_errors))
    try:
        _, p_val = wilcoxon(be_errors[:n], sc_errors[:n])
    except Exception:
        p_val = np.nan

    return {
        f'GS-{ft_short}': winner,
        f'GS-{ft_short}_p': p_val,
        f'GS-{ft_short}_be': be_mean,
        f'GS-{ft_short}_sc': sc_mean,
    }


def _load_sbi_assignment(
    animal_id: str,
    sbi_dir: Path,
    distribution: str,
    fit_target: str,
) -> dict:
    """Load SBI comparison assignment for one animal × one fit target."""
    ft_short = FT_LABEL[fit_target]
    sbi_path = sbi_dir / 'comparisons' / f'{distribution}_{fit_target}' / f'animal_{animal_id}.pkl'

    if not sbi_path.exists():
        return {}

    with open(sbi_path, 'rb') as f:
        comp = pickle.load(f)

    winner = comp.get('winner')
    p_val = comp.get('p', comp.get('p_value', np.nan))

    out = {
        f'SBI-{ft_short}_p': p_val,
        f'SBI-{ft_short}_be': comp.get('be_mean'),
        f'SBI-{ft_short}_sc': comp.get('sc_mean'),
    }
    if winner:
        out[f'SBI-{ft_short}'] = winner

    return out


def _compute_consensus(
    row: dict,
    alpha: float = 0.05,
    min_significant_votes: int = 1,
) -> str:
    """
    Compute consensus from method assignments.

    Rule:
        1. Collect methods where p < alpha AND assignment is BE or SC.
        2. If count < min_significant_votes → 'Unclear'.
        3. If strict majority agrees → that model.
        4. Otherwise → 'Split'.
    """
    votes = []
    for mc in METHOD_COLS:
        val = row.get(mc)
        p = row.get(f'{mc}_p', np.nan)
        if val in ('BE', 'SC') and pd.notna(p) and p < alpha:
            votes.append(val)

    if len(votes) < min_significant_votes:
        return 'Unclear'

    counts = Counter(votes)
    top_model, top_count = counts.most_common(1)[0]

    if top_count > len(votes) / 2:
        return top_model
    return 'Split'


# ── Public API ───────────────────────────────────────────────────────────────

def _collect_animal_ids(
    cv_dir: Path,
    sbi_dir: Path,
    experiment: Optional['ExperimentData'],
    distribution: str,
) -> set:
    """Gather all animal IDs from GS, SBI, and experiment."""
    ids = set()
    for ft in FIT_TARGETS:
        gs_path = cv_dir / f'{distribution}_{ft}'
        if gs_path.exists():
            for p in gs_path.glob('cv_*_BE.pkl'):
                ids.add(p.stem.replace('cv_', '').replace('_BE', ''))

        sbi_path = sbi_dir / 'comparisons' / f'{distribution}_{ft}'
        if sbi_path.exists():
            for p in sbi_path.glob('animal_*.pkl'):
                ids.add(p.stem.replace('animal_', ''))

    if experiment is not None:
        ids |= set(experiment.animals.keys())

    return ids


def load_all_assignments(
    results_dir: Path,
    experiment: Optional['ExperimentData'] = None,
    distribution: str = 'uniform',
    alpha: float = 0.05,
    min_significant_votes: int = 1,
) -> pd.DataFrame:
    """
    Load GS + SBI assignments for all animals and compute consensus.

    Parameters
    ----------
    results_dir : Path
        Root results directory containing cv/ and sbi_static/.
    experiment : ExperimentData, optional
        If given, also includes animal IDs from the experiment even if
        they have no results (shows as missing in the strip).
    distribution : str
        Which distribution phase to load (default 'uniform').
    alpha : float
        Significance threshold for including a method in consensus.
    min_significant_votes : int
        Minimum number of significant methods required before assigning
        a consensus label (otherwise 'Unclear').

    Returns
    -------
    pd.DataFrame
        One row per animal.  Columns include:
        id, GS-UM, GS-UM_p, GS-CP, GS-CP_p,
        SBI-UM, SBI-UM_p, SBI-CP, SBI-CP_p, Consensus
    """
    results_dir = Path(results_dir)
    cv_dir = results_dir / 'cv'
    sbi_dir = results_dir / 'sbi_static'

    all_ids = _collect_animal_ids(cv_dir, sbi_dir, experiment, distribution)

    rows = []
    for aid in sorted(all_ids):
        row = {'id': aid}
        for ft in FIT_TARGETS:
            row.update(_load_gs_assignment(aid, cv_dir, distribution, ft))
            row.update(_load_sbi_assignment(aid, sbi_dir, distribution, ft))
        row['Consensus'] = _compute_consensus(
            row, alpha=alpha, min_significant_votes=min_significant_votes,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def compute_consensus_summary(assign_df: pd.DataFrame) -> str:
    """Return a printable consensus summary."""
    counts = assign_df['Consensus'].value_counts()
    lines = [f'{len(assign_df)} animals total', 'Consensus:']
    for label, n in counts.items():
        lines.append(f'  {label}: {n}')
    return '\n'.join(lines)
