"""SBI conditioning for model identification (BE vs SC).

``condition_sbi(sessions, net, model)`` produces ONE model's held-out MSE
results -- a list of per-rep ``{rep, test_error, best_params[, n_valid]}`` entries
-- ready for ``utils.cv_utils.save_cv_result``. It conditions a pre-trained net
and scores held-out matrices; it never trains and never picks a winner. The
BE-vs-SC decision is ``utils.cv_utils.compare_models`` (method-agnostic, shared
with grid_search); the cross-method consensus is ``analysis.consensus``.

Two rep paths, chosen by the net's session count:

  multi-session (``net.N > 1``; the pooled / moments reps):
      Block-level CV over sessions (first vs second half, the same split
      grid_search uses). Condition on the train fold, simulate on the test fold,
      score the held-out update-matrix (or conditional-psychometric) MSE.
      rep-axis = repeats (posterior resampling + simulation seed -> the spread
      the Wilcoxon consumes). ``best_params`` is one all-session conditioning
      (stable, for the recovery plot), identical across reps.

  single-session (``net.N == 1``):
      Per session, a within-session first/second-half TRIAL split: condition on
      one half, score the UM on the other, swap, mean -> that session's MSE.
      rep-axis = sessions. A session whose stats are degenerate (e.g. an
      error-free expert session -> NaN win_stay, which ``condition`` rejects) is
      skipped, not imputed. Each entry carries ``n_valid`` (the session's
      valid-trial count) so downstream averaging can weight by it. The held-out
      UM here comes from a single ~half-session and is correspondingly noisy --
      this rep leans on aggregation across sessions, by design.

Reuses (no metric reimplemented):
    utils.fold_utils.split_folds_by_block            block-level CV folds
    behav_utils.data.ops.filtering.pool_arrays       block-aware pooling
    behav_utils.data.synthetic.session_from_arrays   half-session construction
    behav_utils.analysis.update_matrix.fit_update_matrix, matrix_error
    models.simulate.simulate_choices                 params -> choices
"""

import numpy as np
from typing import Any, Dict, List, Optional

from inference.types import ModelType
from utils.fold_utils import split_folds_by_block
from behav_utils.data.ops.filtering import pool_arrays
from behav_utils.data.synthetic import session_from_arrays
from behav_utils.analysis.update_matrix import fit_update_matrix, matrix_error
from models.simulate import simulate_choices


# ── shared helpers ───────────────────────────────────────────────────────────

def _as_model(model) -> ModelType:
    if isinstance(model, ModelType):
        return model
    return ModelType(str(getattr(model, 'value', model)).lower())


def _block_ids(sessions: List) -> np.ndarray:
    """Per-trial session(block) index, in session order."""
    pooled = pool_arrays(sessions)
    sizes = np.diff(pooled['session_boundaries'])
    return np.repeat(np.arange(len(sizes)), sizes)


def _empirical_target(pooled: Dict[str, np.ndarray], fit_target: str, n_bins: int):
    um, cm, _ = fit_update_matrix(
        pooled['stimuli'], pooled['choices'], pooled['categories'],
        n_bins=n_bins, trial_filter='post_correct',
        no_response=pooled['no_response'],
        not_blockstart=pooled['prev_has_prev'],
    )
    return um if fit_target == 'update_matrix' else cm


def _simulated_target(model, params: Dict[str, float],
                      pooled: Dict[str, np.ndarray], fit_target: str,
                      n_bins: int, burn_in: int, seed: int):
    sim_ch = simulate_choices(
        model, params, pooled['stimuli'], pooled['categories'],
        burn_in=burn_in, seed=seed)
    um, cm, _ = fit_update_matrix(
        pooled['stimuli'], sim_ch, pooled['categories'],
        n_bins=n_bins, trial_filter='post_correct',
        no_response=pooled['no_response'],
        not_blockstart=pooled['prev_has_prev'],
    )
    return um if fit_target == 'update_matrix' else cm


def _safe_condition(net, sessions: List, n_posterior_samples: int
                    ) -> Optional[Dict[str, float]]:
    """Condition; return the point-estimate dict, or None if the obs is
    degenerate (``condition`` raises on non-finite stats) or theta is non-finite.
    """
    try:
        cond = net.condition(sessions, n_samples=n_posterior_samples)
    except ValueError:
        return None
    theta = cond['point_estimate']
    if not all(np.isfinite(v) for v in theta.values()):
        return None
    return theta


# ── multi-session path (pooled / moments) ────────────────────────────────────

def _condition_multi(sessions, net, model, fit_target, n_folds,
                     n_repeats, n_posterior_samples, n_bins, seed):
    if len(sessions) < 2:
        raise ValueError(f'Need >= 2 sessions for CV, got {len(sessions)}')

    burn_in = getattr(net, 'burn_in', 1000)
    block_ids = _block_ids(sessions)
    folds = split_folds_by_block(block_ids, n_folds)

    fold_data = []
    for train_mask, test_mask in folds:
        train_blocks = np.unique(block_ids[train_mask])
        test_blocks = np.unique(block_ids[test_mask])
        train_sessions = [sessions[int(b)] for b in train_blocks]
        test_sessions = [sessions[int(b)] for b in test_blocks]
        test_pooled = pool_arrays(test_sessions)
        emp = _empirical_target(test_pooled, fit_target, n_bins)
        fold_data.append((train_sessions, test_pooled, emp))

    # One all-session conditioning -> stable recovered params (recovery plot).
    best = _safe_condition(net, sessions, n_posterior_samples)

    results = []
    for rep in range(n_repeats):
        rep_seed = seed + rep
        fold_errs = []
        for train_sessions, test_pooled, emp in fold_data:
            theta = _safe_condition(net, train_sessions, n_posterior_samples)
            if theta is None:
                fold_errs = []
                break
            sim = _simulated_target(model, theta, test_pooled,
                                    fit_target, n_bins, burn_in, rep_seed)
            fold_errs.append(float(matrix_error(emp, sim)))
        if not fold_errs:
            continue
        err = float(np.mean(fold_errs))
        if not np.isfinite(err):
            continue
        results.append({'rep': rep, 'test_error': err, 'best_params': best})
    return results


# ── single-session path ──────────────────────────────────────────────────────

def _condition_single(sessions, net, model, fit_target,
                      n_posterior_samples, n_bins, seed):
    burn_in = getattr(net, 'burn_in', 1000)
    results = []
    for si, session in enumerate(sessions):
        a = session.get_arrays()
        stim, ch, cat = a['stimuli'], a['choices'], a['categories']
        n = len(stim)
        if n < 8:                       # too short to split into two usable halves
            continue
        mid = n // 2

        def _half(lo, hi):
            return session_from_arrays(
                stim[lo:hi], ch[lo:hi], cat[lo:hi], session_idx=si)

        half_a, half_b, full = _half(0, mid), _half(mid, n), _half(0, n)

        # Conditioning must stay in-distribution for all three; skip on any NaN.
        theta_a = _safe_condition(net, [half_a], n_posterior_samples)
        theta_b = _safe_condition(net, [half_b], n_posterior_samples)
        theta_full = _safe_condition(net, [full], n_posterior_samples)
        if theta_a is None or theta_b is None or theta_full is None:
            continue

        pooled_a, pooled_b = pool_arrays([half_a]), pool_arrays([half_b])
        emp_a = _empirical_target(pooled_a, fit_target, n_bins)
        emp_b = _empirical_target(pooled_b, fit_target, n_bins)
        sess_seed = seed + si
        # condition on A -> predict B, condition on B -> predict A, then mean.
        sim_on_b = _simulated_target(model, theta_a, pooled_b,
                                     fit_target, n_bins, burn_in, sess_seed)
        sim_on_a = _simulated_target(model, theta_b, pooled_a,
                                     fit_target, n_bins, burn_in, sess_seed)
        err = 0.5 * (float(matrix_error(emp_b, sim_on_b)) +
                     float(matrix_error(emp_a, sim_on_a)))
        if not np.isfinite(err):
            continue

        results.append({
            'rep': si,
            'test_error': err,
            'best_params': theta_full,
            'n_valid': int(np.sum(~np.isnan(ch))),
        })
    return results


# ── public entry point ───────────────────────────────────────────────────────

def condition_sbi(
    sessions: List,
    net,
    model,
    fit_target: str = 'update_matrix',
    n_folds: int = 2,
    n_repeats: int = 64,
    n_posterior_samples: int = 50,
    n_bins: int = 8,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Produce one model's held-out MSE results for the BE-vs-SC comparison.

    Args:
        sessions: One animal's pre-filtered SessionData list.
        net: A trained AmortisedSBI. ``net.N == 1`` selects the single-session
            path; otherwise the block-CV (pooled / moments) path.
        model: ModelType or 'be'/'sc' -- the model whose net this is.
        fit_target: 'update_matrix' or 'conditional_psych'.
        n_folds: Block-level CV folds (multi-session path only).
        n_repeats: Repeats over posterior resampling + sim seed (multi-session
            path only; the single path's spread is across sessions).
        n_posterior_samples: Samples per conditioning (median = point estimate).
        n_bins: Update-matrix bins.
        seed: Base seed (BE and SC must share it so the comparison pairs).

    Returns:
        List of ``{rep, test_error, best_params[, n_valid]}`` -- the ``results``
        argument for ``save_cv_result``. rep is the repeat index (multi) or the
        session index (single).
    """
    if fit_target not in ('update_matrix', 'conditional_psych'):
        raise ValueError(f"Unknown fit_target {fit_target!r}")
    model = _as_model(model)

    if getattr(net, 'N', None) == 1:
        return _condition_single(sessions, net, model, fit_target,
                                 n_posterior_samples, n_bins, seed)
    return _condition_multi(sessions, net, model, fit_target, n_folds,
                            n_repeats, n_posterior_samples, n_bins, seed)
