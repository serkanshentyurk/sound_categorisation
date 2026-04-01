#!/usr/bin/env python3
"""
run_cv_single.py — Run BE/SC grid-search CV for one animal, one seed.

Designed to be called as a SLURM array job:
    python run_cv_single.py --config config.yaml --animal SS01 --seed 1 --output-dir ./cv_results

Each invocation fits both BE and SC for one seed, saves a pickle.
A separate gather script merges all seeds into the final DataFrame.
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PARSE ARGS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='BE/SC grid-search CV for one animal, one seed')
    p.add_argument('--config', required=True, help='Path to behav_utils config.yaml')
    p.add_argument('--animal', required=True, help='Animal ID (e.g. SS01)')
    p.add_argument('--seed', required=True, type=int, help='CV seed (1-indexed)')
    p.add_argument('--output-dir', required=True, help='Directory for output pickles')

    # Grid resolution
    p.add_argument('--grid', default='full', choices=['full', 'coarse'],
                   help='Grid resolution: full (manuscript) or coarse (fast)')

    # Session selection
    p.add_argument('--stage', default='Full_Task_Cont')
    p.add_argument('--distribution', default='Uniform')
    p.add_argument('--min-accuracy', default=0.70, type=float)
    p.add_argument('--last-fraction', default=0.50, type=float)
    p.add_argument('--min-expert-sessions', default=5, type=int)

    # CV settings
    p.add_argument('--n-folds', default=2, type=int)
    p.add_argument('--fit-with', default='conditional',
                   choices=['conditional', 'update'])
    p.add_argument('--mode-pre', default='simulated',
                   choices=['simulated', 'real'])


    return p.parse_args()


# =============================================================================
# DATA CONVERSION
# =============================================================================

def select_expert_sessions(animal, stage, distribution,
                           min_accuracy, last_fraction):
    sessions = animal.get_sessions(stage=stage, distribution=distribution)
    if len(sessions) == 0:
        raise ValueError(f"No sessions for {animal.animal_id}")

    n_total = len(sessions)
    start_idx = int(n_total * (1.0 - last_fraction))
    candidate_sessions = sessions[start_idx:]

    expert = []
    for s in candidate_sessions:
        acc = s.stats(['accuracy'])['accuracy']
        if acc >= min_accuracy:
            expert.append(s)

    if len(expert) == 0:
        raise ValueError(f"No expert sessions for {animal.animal_id}")
    return expert


def sessions_to_old_df(sessions, animal_id=None):
    all_rows = []
    for block_id, session in enumerate(sessions):
        trials = session.trials
        valid = ~trials.abort
        if hasattr(trials, 'opto_on') and trials.opto_on is not None:
            valid = valid & ~trials.opto_on

        stim = trials.stimulus[valid]
        choice = trials.choice[valid]
        correct = trials.correct[valid]
        n = len(stim)
        if n == 0:
            continue

        no_response = np.isnan(choice)
        choice_clean = np.where(no_response, 0, choice).astype(int)
        correct_clean = np.where(no_response, 0, correct).astype(int)

        df_block = pd.DataFrame({
            'stim_relative': stim,
            'choice': choice_clean,
            'correct': correct_clean,
            'No_response': no_response,
            'block': block_id,
            'Trial': np.arange(1, n + 1),
        })
        all_rows.append(df_block)

    if len(all_rows) == 0:
        raise ValueError("No valid trials found")

    df = pd.concat(all_rows, ignore_index=True)
    df['is_not_start_of_block'] = df['block'].eq(df['block'].shift())

    if animal_id is not None:
        df['Participant_ID'] = animal_id
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    from Old.Fitter import k_fold_CV, post_correct_update_matrix
    from Old.BE import BE_model
    from Old.SC import SC_model
    from behav_utils.data.loading import load_experiment

    # ── Grid parameters ───────────────────────────────────────────────────
    if args.grid == 'full':
        sigma_noise_values     = np.linspace(0.05, 0.30, 10)
        A_repulsion_values     = np.linspace(0.0, 0.5, 4)
        be_eta_learning_values = np.linspace(0.1, 0.9, 20)
        be_eta_relax_values    = np.linspace(0.05, 0.4, 10)
        sc_gamma_values        = np.linspace(0.1, 1.0, 20)
        sc_sigma_update_values = np.linspace(0.1, 1.0, 10)
    else:  # coarse
        sigma_noise_values     = np.linspace(0.05, 0.30, 4)
        A_repulsion_values     = np.array([0.0, 0.25, 0.5])
        be_eta_learning_values = np.linspace(0.1, 0.9, 8)
        be_eta_relax_values    = np.linspace(0.05, 0.4, 4)
        sc_gamma_values        = np.linspace(0.1, 1.0, 8)
        sc_sigma_update_values = np.linspace(0.1, 1.0, 4)

    # ── Load data ─────────────────────────────────────────────────────────
    experiment = load_experiment(args.config)
    animal = experiment.get_animal(args.animal)

    expert_sessions = select_expert_sessions(
        animal, args.stage, args.distribution,
        args.min_accuracy, args.last_fraction,
    )

    if len(expert_sessions) < args.min_expert_sessions:
        print(f"SKIP: {args.animal} has only {len(expert_sessions)} expert sessions")
        sys.exit(0)

    df = sessions_to_old_df(expert_sessions, animal_id=args.animal)
    print(f"{args.animal}: {len(df)} trials, {df['block'].nunique()} blocks, seed={args.seed}")

    # ── Run CV for both models ────────────────────────────────────────────
    results = {}
    for model_name, model_func, x_vals, y_vals in [
        ('BE', BE_model, be_eta_relax_values, be_eta_learning_values),
        ('SC', SC_model, sc_sigma_update_values, sc_gamma_values),
    ]:
        t0 = time.time()
        try:
            cv_out = k_fold_CV(
                df=df,
                model=model_func,
                func=post_correct_update_matrix,
                sigma_noise_values=sigma_noise_values,
                A_repulsion_values=A_repulsion_values,
                x_axis_values=x_vals,
                y_axis_values=y_vals,
                seed=args.seed,
                k=args.n_folds,
                mode_pre=args.mode_pre,
                fit_with=args.fit_with,
                show_progress=False,
            )
            test_errors = cv_out[1]
            avg_error = float(np.mean(test_errors))
            best_params = cv_out[0]

            results[model_name] = {
                'avg_test_error': avg_error,
                'test_errors': test_errors,
                'best_params': best_params,
                'model': model_name,
                'seed': args.seed,
            }
            elapsed = time.time() - t0
            print(f"  {model_name}: avg_error={avg_error:.6f} ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {model_name}: FAILED after {elapsed:.1f}s — {e}")
            results[model_name] = {
                'avg_test_error': np.nan,
                'test_errors': [],
                'best_params': None,
                'model': model_name,
                'seed': args.seed,
            }

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f'cv_{args.animal}_seed{args.seed:03d}.pkl'
    )
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
