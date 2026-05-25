import pandas as pd 
import numpy as np 

# =============================================================================
# MODEL IDENTIFICATION RUNNERS
# =============================================================================

def run_gs_model_id(
    animals, sessions_key='sessions', grid=None, n_seeds=2, burn_in=1000,
    fit_target='update_matrix',
):
    """
    Run grid-search model identification on synthetic animals.

    Args:
        fit_target: 'update_matrix' or 'conditional_psych'.

    Returns DataFrame with gs_winner, gs_correct, gs_be_mean, gs_sc_mean,
    gs_recovered_params, fit_target.
    """
    from analysis.grid_search import compute_grid_search_cv, COARSE_GRID
    if grid is None:
        grid = COARSE_GRID

    rows = []
    for sa in animals:
        aid = sa['animal_id']
        sessions = sa[sessions_key]
        print(f'  GS [{fit_target}] {aid} [{sa["true_model"]}]...', end=' ')

        errors = {'BE': [], 'SC': []}
        errors_detail = {'BE': [], 'SC': []}
        for seed in range(1, n_seeds + 1):
            for mt in ['BE', 'SC']:
                try:
                    r = compute_grid_search_cv(
                        sessions, mt, grid=grid[mt],
                        n_folds=2, seed=seed, burn_in=burn_in,
                        fit_target=fit_target,
                    )
                    errors[mt].append(r['avg_test_error'])
                    errors_detail[mt].append(r)
                except (KeyError, TypeError):
                    pass

        be_mean = np.mean(errors['BE']) if errors['BE'] else np.nan
        sc_mean = np.mean(errors['SC']) if errors['SC'] else np.nan
        winner = 'BE' if be_mean < sc_mean else 'SC'
        correct = winner == sa['true_model']
        # Store recovered params from winning model's best seed
        recovered = {}
        winner_results = errors_detail.get(winner, [])
        if winner_results:
            best_seed_idx = int(np.argmin([r['avg_test_error'] for r in winner_results]))
            recovered = winner_results[best_seed_idx].get('best_params_single', {})

        rows.append({
            'animal_id': aid, 'true_model': sa['true_model'],
            'gs_winner': winner, 'gs_correct': correct,
            'gs_be_mean': be_mean, 'gs_sc_mean': sc_mean,
            'gs_recovered_params': recovered,
            'fit_target': fit_target,
        })
        print(f'{winner} {"✓" if correct else "✗"}')

    return pd.DataFrame(rows)
