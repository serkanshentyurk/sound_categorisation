"""
Legacy code from the inherited codebase.

Modules implement the original grid-search CV pipeline from the manuscript.
Kept as-is aside from import path updates. Wrapped by analysis/cv_utils.py.

Import convention:
    from legacy.fitter import k_fold_CV, post_correct_update_matrix
    from legacy.be import BE_model, Noise_generator, Delta_repulsion
    from legacy.sc import SC_model
"""
