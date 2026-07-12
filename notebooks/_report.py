"""
notebooks/_report.py — shared multi-panel display helpers for the validation notebooks.

These compose the single-panel `plotting.cv` primitives (plot_cv_comparison, plot_confusion,
plot_recovery) into cohort-level figures. They live here rather than in `plotting/` because
they loop and compose multiple panels — notebook territory — whereas every `plot_x` in
`plotting/` stays single-panel and draw-only.

Used by 11_gs_validation and 12_sbi_validation (previously each carried its own, drifted copy
of show_model_id / show_recovery).

Both take a CVResults (from utils.cv_utils.load_cv_results), which exposes:
    .comparison  — per-animal winner/p-value table
    .long        — per-seed errors (animal_id, model, seed, avg_test_error)
    .recovery    — true-vs-recovered (param, true_model, true_value, recovered_value)
"""

import numpy as np
import matplotlib.pyplot as plt

from plotting.cv import plot_cv_comparison, plot_confusion, plot_recovery


def show_model_id(cv, title='', per_animal=False, fit_target=None):
    """Cohort confusion matrix, optionally preceded by per-animal CV comparison plots.

    Args:
        cv: CVResults (uses .comparison, and .long when per_animal).
        title: Prefix for the confusion-matrix title.
        per_animal: If True, draw plot_cv_comparison for each animal first
            (the GS-validation view); needs fit_target for the y-axis label.
        fit_target: Axis label passed through to plot_cv_comparison (e.g. 'UM', 'CP').
    """
    if cv.comparison is None or len(cv.comparison) == 0:
        print(f'{title}: no comparison rows'.strip())
        return
    if per_animal:
        for aid in cv.comparison['animal_id']:
            plot_cv_comparison(cv.long, cv.comparison, aid, fit_target=fit_target or 'UM')
            plt.show()
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_confusion(cv.comparison, ax=ax)
    if title:
        ax.set_title(f'{title} {ax.get_title()}'.strip())
    plt.show()


def show_recovery(cv, title='', ncol=4):
    """Grid of true-vs-recovered scatter, one panel per recovered parameter.

    Args:
        cv: CVResults (uses .recovery).
        title: Figure suptitle.
        ncol: Max columns; a single row when params <= ncol (the common 4-param case).
    """
    if cv.recovery is None or len(cv.recovery) == 0:
        print(f'{title}: no recovery rows'.strip())
        return
    params = list(cv.recovery['param'].unique())
    if not params:
        print(f'{title}: no recovery data'.strip())
        return
    ncol = max(1, min(ncol, len(params)))
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, params):
        plot_recovery(cv.recovery, p, ax=ax)
    for ax in axes[len(params):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    plt.show()
