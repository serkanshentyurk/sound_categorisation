import ast
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import seaborn as sns
import os
import sys
from scipy.stats import norm, uniform
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import UnivariateSpline
from scipy.integrate import trapezoid


# from Modules.Fitter import (
#     post_correct_update_matrix,
#     post_correct_conditional_stimuli,
#     post_correct_total_stimuli,
#     fit_Psych_curve
# )

# Seed randomness once for reproducibility
np.random.seed(42)

# Configure logging for warnings
logging.basicConfig(level=logging.INFO)

# Set up module paths
cwd = os.getcwd()
dataset_dir = os.path.join(cwd, 'Dataset')
modules_dir = os.path.join(cwd, 'Modules')
fitter_dir = os.path.join(cwd, 'Modules/Fitter')
models_dir = os.path.join(cwd, 'Modules/Models')
models_generative_path = os.path.join(cwd, 'Synthetic_Data/Modules/Models/')

sys.path.extend([modules_dir, models_dir, fitter_dir, models_generative_path])

from Fitter import (
    psychometric_model,
    post_correct_total_stimuli, 
    fit_Psych_curve, 
    post_correct_update_matrix, 
    post_correct_conditional_stimuli
)
from BE_Generative import BE_model, Noise_generator, Delta_repulsion  
from SC_Generative import SC_model


def dummy_array(x, *args, **kwargs):
    """
    Dummy replacement for numpy.array to safely evaluate string representations
    of arrays inside eval. Used to prevent unwanted execution during parsing.
    """
    return x

def convert_str_to_dict(x):
    """
    Safely evaluates a string representing a dictionary containing model fit results.
    Replaces numpy.array with dummy_array and restricts eval scope for safety.

    Parameters:
        x (str or dict): Stringified dictionary or already-parsed dictionary.

    Returns:
        dict or None: Parsed dictionary, or None if parsing fails.
    """
    if isinstance(x, str):
        try:
            safe_globals = {"__builtins__": {"__import__": __import__}, "array": dummy_array}
            return eval(x, safe_globals)
        except Exception as e:
            print(f"Error converting cell:\n{x}\nError: {e}")
            return None
    return x

def extract_list_literal(s, key):
    """
    Find the first bracketed literal after "key" in string s,
    return it (including brackets) or None if not found.
    """
    idx = s.find(key)
    if idx < 0:
        return None
    # find the opening '['
    start = s.find('[', idx)
    if start < 0:
        return None
    # walk forward to match brackets
    depth = 0
    for i, ch in enumerate(s[start:], start):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def plot_individual_seed_errors(df, species_label, marker="D"):
    """
    Plots individual model fit errors for each seed per participant.

    Parameters:
        df (pd.DataFrame): Long-form DataFrame with columns: Participant_ID, Model, Test_Error.
        species_label (str): Label for plot title.
        marker (str): Matplotlib marker style (e.g., 'o', 's', 'D') for species.
    """


    fig, ax = plt.subplots(figsize=(14, 7))
    color_map = {'BE': 'blue', 'SC': 'orange'}

    for _, row in df.iterrows():
        ax.scatter(
            row["Participant_ID"],
            row["Test_Error"],
            color=color_map[row["Model"]],
            marker=marker,
            s=50,
            alpha=0.8,
            # edgecolors='k',
            linewidths=0.4
        )

    # Add legend
    be_patch = mpatches.Patch(color='blue', label='Fitted: BE')
    sc_patch = mpatches.Patch(color='orange', label='Fitted: SC')
    ax.legend(handles=[be_patch, sc_patch])

    ax.set_title(f"Best Error by Participant and Model Fit ({species_label})", fontsize=14)
    ax.set_xlabel("Participant", fontsize=12)
    ax.set_ylabel("Test Error", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def sort_by_participant_id(df, id_column: str, prefix: str) -> pd.DataFrame:
    """
    Sort the dataframe by the numeric portion of the IDs in `id_column`
    that appear after a given `prefix`. For example, if `prefix='EM'`,
    we expect IDs like 'EM28', 'EM31', etc., and the numeric portion is
    extracted and converted to int to sort numerically.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    id_column : str
        The name of the column containing the IDs (e.g. 'Participant_ID').
    prefix : str
        The prefix in front of the numeric portion (e.g. 'EM', 'QP', 'Human_').

    Returns
    -------
    pd.DataFrame
        A new dataframe, sorted by the numeric ID portion after the prefix.
    """
    # Make a copy so we don’t modify original DataFrame in place
    df_sorted = df.copy()

    # Extract the substring *after* the prefix, then convert to int
    # For example, if prefix="EM" and ID="EM41", the numeric part is "41".
    df_sorted["numeric_id"] = df_sorted[id_column].str.replace(prefix, "", regex=False)
    df_sorted["numeric_id"] = df_sorted["numeric_id"].astype(int)

    # Sort and then drop the helper column
    df_sorted = df_sorted.sort_values(by="numeric_id", ascending=True)
    df_sorted.drop(columns="numeric_id", inplace=True)

    # Optionally reset the index
    df_sorted.reset_index(drop=True, inplace=True)

    return df_sorted

def offset_xtick_labels(ax, offset_points=-3):
    """
    Offsets the x-tick labels horizontally by a given number of points.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object containing the tick labels.
    offset_points : float
        The horizontal offset in points. Use a negative number to move left,
        or a positive number to move right.
    """
    # Convert the desired offset from points to inches (1 inch = 72 points)
    offset_inch = offset_points / 72.0
    # Create a translation transformation.
    offset = mtransforms.ScaledTranslation(offset_inch, 0, ax.figure.dpi_scale_trans)
    
    # Apply the offset to each x-tick label.
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + offset)

def plot_with_seed_and_average(
    long_df, avg_df, species_label, marker, figsize=(7, 5), dpi = 150, y_lim=None,
    jitter_std=0.1, plot_type='both', legend_pos='upper right',
    marker_style_config=None, save_path=None
):
    """
    Plots individual seed-level model fit errors and/or per-participant averages with jitter.
    Includes a clear legend for average vs. individual seeds.

    Parameters:
        long_df (pd.DataFrame): Long-form df with columns: Participant_ID, Model, Test_Error, Species
        avg_df (pd.DataFrame): Averaged df with columns: Participant_ID, Model, Avg_Test_Error, Species
        species_label (str): Label for the plot title
        marker (str): Matplotlib marker symbol (e.g. 'o', 's', 'D')
        jitter_std (float): Std dev of jitter applied to x-axis of individual seeds
        plot_type (str): One of {'both', 'seeds', 'average'} to control which data to plot
        marker_style_config (dict): Dictionary mapping 'average' and 'seed' to style dicts (e.g. size, alpha)
    """
    np.random.seed(42)  # or whatever seed you choose
    if marker_style_config is None:
        marker_style_config = {
            "average": {"s": 100, "alpha": 0.2},
            "seed": {"s": 40, "alpha": 0.5}
        }

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    color_map = {'BE': 'blue', 'SC': 'orange'}
    participant_ids = avg_df["Participant_ID"].unique()
    pid_to_x = {pid: i for i, pid in enumerate(participant_ids)}

    # Plot individual seed errors
    if plot_type in ['both', 'seeds']:
        for _, row in long_df.iterrows():
            x_base = pid_to_x[row["Participant_ID"]]
            x_jittered = x_base + np.random.normal(0, jitter_std)

            ax.scatter(
                x_jittered,
                row["Test_Error"],
                color=color_map[row["Model"]],
                marker=marker,
                **marker_style_config["seed"],
                linewidths=0.3
            )

    # Plot averages
    if plot_type in ['both', 'average']:
        for _, row in avg_df.iterrows():
            x = pid_to_x[row["Participant_ID"]]
            ax.scatter(
                x,
                row["Avg_Test_Error"],
                color=color_map[row["Model"]],
                marker=marker,
                **marker_style_config["average"]
            )

    # Legend
    legend_handles = []
    if plot_type in ['both', 'average']:
        legend_handles += [
            mlines.Line2D([], [], color=color_map['BE'], marker=marker, linestyle='None',
                          markersize=marker_style_config['average']['s'] ** 0.5,
                          alpha=marker_style_config['average']['alpha'], label='BE: Average'),
            mlines.Line2D([], [], color=color_map['SC'], marker=marker, linestyle='None',
                          markersize=marker_style_config['average']['s'] ** 0.5,
                          alpha=marker_style_config['average']['alpha'], label='SC: Average')
        ]
    if plot_type in ['both', 'seeds']:
        legend_handles += [
            mlines.Line2D([], [], color=color_map['BE'], marker=marker, linestyle='None',
                          markersize=marker_style_config['seed']['s'] ** 0.5,
                          alpha=marker_style_config['seed']['alpha'], label='BE: Individual Seeds'),
            mlines.Line2D([], [], color=color_map['SC'], marker=marker, linestyle='None',
                          markersize=marker_style_config['seed']['s'] ** 0.5,
                          alpha=marker_style_config['seed']['alpha'], label='SC: Individual Seeds')
        ]

    if legend_pos is not None:
        ax.legend(handles=legend_handles, loc=legend_pos, fontsize=12, frameon=True)
    else:
        # remove legend
        ax.legend().remove()

    ax.set_title(f"Model Fit Error per Participant\nSpecies: {species_label}", fontsize=14)
    ax.set_xlabel("Participant", fontsize=12)
    ax.set_ylabel("Test Error", fontsize=12)
    # ax.set_xticks(list(pid_to_x.values()))
    # ax.set_xticklabels(participant_ids, rotation=45)
    ax.set_xticks(list(pid_to_x.values()))
    ax.set_xticklabels(
        participant_ids,
        rotation=45,
        ha='right',             # ← here: align right edge of label on the tick
        rotation_mode='anchor'  # ← here: ensure the ‘ha’ anchor is used during rotation
    )

    ax.set_autoscale_on(False)

    if y_lim is not None:
        ax.set_ylim(y_lim)
    else:
        ax.set_ylim(-0.01, 1)

    # set yticks
    # ax.set_yticks(np.arange(0, 1.1, 0.2))

    offset_xtick_labels(ax, offset_points=-2)  # shifts labels 3 points to the left

    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300)

    plt.show()

def _to_array(x):
    """
    Internal helper to convert a list-like to a NumPy array of floats,
    returning None on failure.
    """
    try:
        return np.array(x, float)
    except:
        return None


def compute_seed_best_fold(cell, matrix_type='conditional'):
    """
    Parse one seed's CV result, find the lowest-error fold, and
    return (avg_error, best_params, best_fold,
            best_train_matrix, best_model_train_matrix,
            best_test_matrix, best_model_test_matrix).

    matrix_type determines which set of keys to look up:
      - 'conditional' (default): uses *_conditional_matrix keys
      - 'update':                uses *_update_matrix keys
    """
    # Convert string-encoded dict to actual dict
    d = convert_str_to_dict(cell)
    if not isinstance(d, dict):
        return (np.nan, None, None, None, None, None, None)

    # 1) errors & params
    errs = d.get("test_errors_list", [])
    avg_error = float(np.mean(errs)) if errs else np.nan
    best_params = d.get("best_params") or d.get("optimal_params_list")

    # 2) best-fold index
    try:
        best_fold = int(np.argmin(errs))
    except:
        best_fold = None

    # 3) choose key suffixes based on matrix_type
    if matrix_type == 'conditional':
        train_key       = 'training_conditional_matrix'
        model_train_key = 'Model_training_conditional_matrix'
        test_key        = 'test_conditional_matrix'
        model_test_key  = 'Model_test_conditional_matrix'
    elif matrix_type == 'update':
        train_key       = 'training_update_matrix'
        model_train_key = 'Model_training_update_matrix'
        test_key        = 'test_update_matrix'
        model_test_key  = 'Model_test_update_matrix'
    else:
        raise ValueError(f"Unknown matrix_type '{matrix_type}'. Use 'conditional' or 'update'.")

    # 4) grab & convert each matrix stack
    train_stack       = _to_array(d.get(train_key))
    model_train_stack = _to_array(d.get(model_train_key))
    test_stack        = _to_array(d.get(test_key))
    model_test_stack  = _to_array(d.get(model_test_key))

    # 5) slice out the best-fold matrices
    if best_fold is not None and train_stack is not None:
        best_train_matrix       = train_stack[best_fold]
        best_model_train_matrix = model_train_stack[best_fold]
        best_test_matrix        = test_stack[best_fold]
        best_model_test_matrix  = model_test_stack[best_fold]
    else:
        best_train_matrix = best_model_train_matrix = best_test_matrix = best_model_test_matrix = None

    return (
        avg_error,
        best_params,
        best_fold,
        best_train_matrix,
        best_model_train_matrix,
        best_test_matrix,
        best_model_test_matrix
    )


def extract_long_seed_df_with_all_params(
    cv_df,
    species_label,
    matrix_type='conditional',
    n_seeds=8
):
    """
    Unpack each seed’s best-fold results into a long-form DataFrame,
    using the specified matrix_type ('conditional' or 'update').

    Column names for the matrix columns will be suffixed with CM or UM
    depending on matrix_type.

    Returns a DataFrame with columns:
      Participant_ID, Model, Seed, Best_Fold, Test_Error, Species,
      Best_Params, Best_Train_CM/UM, Best_Model_Train_CM/UM,
      Best_Test_CM/UM, Best_Model_Test_CM/UM
    """
    rows = []
    for seed in range(1, n_seeds + 1):
        col = f"seed_{seed}_results"
        if col not in cv_df.columns:
            continue

        for _, r in cv_df.iterrows():
            pid   = r["Participant_ID"]
            model = r["Model"]
            cell  = r[col]

            (err, params, best_fold,
             best_train_matrix, best_model_train_matrix,
             best_test_matrix, best_model_test_matrix
            ) = compute_seed_best_fold(cell, matrix_type=matrix_type)

            # choose output column labels based on matrix_type
            if matrix_type == 'conditional':
                train_label       = 'Best_Train_CM'
                model_train_label = 'Best_Model_Train_CM'
                test_label        = 'Best_Test_CM'
                model_test_label  = 'Best_Model_Test_CM'
            else:
                train_label       = 'Best_Train_UM'
                model_train_label = 'Best_Model_Train_UM'
                test_label        = 'Best_Test_UM'
                model_test_label  = 'Best_Model_Test_UM'

            row = {
                "Participant_ID":      pid,
                "Model":               model,
                "Seed":                seed,
                "Best_Fold":           best_fold,
                "Test_Error":          err,
                "Species":             species_label,
                "Best_Params":         params,
                train_label:            best_train_matrix,
                model_train_label:      best_model_train_matrix,
                test_label:             best_test_matrix,
                model_test_label:       best_model_test_matrix
            }
            rows.append(row)

    return pd.DataFrame(rows)


def select_best_seed(long_df):
    idx = long_df.groupby(["Participant_ID","Model"])["Test_Error"].idxmin()
    return long_df.loc[idx].reset_index(drop=True)

def expand_best_params(best_df):
    """
    Expand the Best_Params dictionary into separate columns.
    """
    params_df = best_df["Best_Params"].apply(pd.Series)
    return pd.concat([best_df.drop(columns=["Best_Params"]), params_df], axis=1)

def expand_best_params_V2(df, params_col="Best_Params"):
    """
    Expand 4-tuple/list in `params_col` to named columns:
      Common: sigma_noise, A_repulsion
      BE:     eta_relax, eta_learning
      SC:     sigma_update, gamma
    Works if params are real lists/tuples or stringified lists.
    """
    def to_list(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = list(v)
        elif pd.isna(v):
            arr = [np.nan]*4
        elif isinstance(v, str):
            try:
                arr = list(ast.literal_eval(v))
            except Exception:
                # ultra-fallback
                arr = [float(x) for x in v.strip("[]").split(",")]
        else:
            arr = [np.nan]*4
        # pad/trim to length 4
        return (arr + [np.nan]*4)[:4]

    # explode to 4 temp columns
    p = df[params_col].apply(to_list).apply(pd.Series)
    p.columns = ["p0","p1","p2","p3"]

    out = df.drop(columns=[params_col]).copy()

    # shared
    out["sigma_noise"] = p["p0"]
    out["A_repulsion"] = p["p1"]

    # init
    for c in ["eta_relax","eta_learning","sigma_update","gamma"]:
        out[c] = np.nan

    be = out["Model"].eq("BE")
    sc = out["Model"].eq("SC")

    # BE gets (p2,p3) -> (eta_relax, eta_learning)
    out.loc[be, ["eta_relax","eta_learning"]] = p.loc[be, ["p2","p3"]].to_numpy()

    # SC gets (p2,p3) -> (sigma_update, gamma)
    out.loc[sc, ["sigma_update","gamma"]] = p.loc[sc, ["p2","p3"]].to_numpy()

    return out


def rename_param_columns(final_df):
    """
    Rename columns [0, 1, 2, 3] depending on the model type (BE or SC).
    """

    # Rename temp columns
    final_df = final_df.rename(columns={0: "param_0", 1: "param_1", 2: "param_2", 3: "param_3"})

    # Create new properly named columns
    final_df["sigma_noise"] = np.where(
        final_df["Model"] == "BE",
        final_df["param_0"],
        final_df["param_0"]
    )

    final_df["A_repulsion"] = np.where(
        final_df["Model"] == "BE",
        final_df["param_1"],
        final_df["param_1"]
    )

    final_df["eta_relax"] = np.where(
        final_df["Model"] == "BE",
        final_df["param_2"],
        np.nan
    )

    final_df["eta_learning"] = np.where(
        final_df["Model"] == "BE",
        final_df["param_3"],
        np.nan
    )

    final_df["sigma_update"] = np.where(
        final_df["Model"] == "SC",
        final_df["param_2"],
        np.nan
    )

    final_df["gamma"] = np.where(
        final_df["Model"] == "SC",
        final_df["param_3"],
        np.nan
    )

    # Drop the old unnamed parameter columns
    final_df = final_df.drop(columns=["param_0", "param_1", "param_2", "param_3"])

    return final_df

# def simulate_subject_trials(participant_df, model_type, param_dict, seed=42):
#     """
#     Simulate one subject's trial-by-trial choices & rewards under either the BE or SC generative model,
#     given their optimal parameters and the empirical stimulus sequence.

#     Args:
#         participant_df (pd.DataFrame): columns must include
#             'stim_relative', 'No_response', 'block', 'Trial', 'Participant_ID'.
#         model_type (str): 'BE' or 'SC'.
#         param_dict (dict): contains the required parameters for the chosen model:
#             for BE: {'A_repulsion', 'sigma_noise', 'eta_learning', 'eta_relax'}
#             for SC: {'A_repulsion', 'sigma_noise', 'gamma', 'sigma_update'}
#         seed (int): base random seed for reproducibility.

#     Returns:
#         pd.DataFrame with columns ['Participant_ID','Trial','choice','correct',
#                                    'No_response','stim_relative','block'].
#     """
#     # unpack data
#     s           = participant_df['stim_relative'].to_numpy()
#     no_response = participant_df['No_response'].to_numpy()
#     block       = participant_df['block'].to_numpy()
#     trial       = participant_df['Trial'].to_numpy()
#     pid         = participant_df['Participant_ID'].iloc[0]

#     # construct stimulus grid x
#     A_repulsion = param_dict['A_repulsion']
#     sigma_noise = param_dict['sigma_noise']
#     max_range   = 1 + 6*sigma_noise + 2*A_repulsion*(1 + 6*sigma_noise)
#     min_range   = -1 - 6*sigma_noise - 2*A_repulsion*(1 + 6*sigma_noise)
#     num_points  = round((max_range - min_range)*1000)
#     x           = np.linspace(min_range, max_range, num_points)

#     burn_in_seed = 5 * seed  # consistent with BE/SC burn-in

#     # compute perceptual estimate & categories
#     s_tilde = Noise_generator(len(s), seed, sigma_noise)
#     s_hat   = Delta_repulsion(A_repulsion, s_tilde)
#     categories = (s > 0).astype(int)

#     if model_type == 'BE':
#         # uniform prior for boundary PDF
#         y = uniform.pdf(x, loc=min_range, scale=(max_range - min_range))
#         eta_learning = param_dict['eta_learning']
#         eta_relax    = param_dict['eta_relax']
#             # compute perceptual estimate & categories
#         s_tilde = Noise_generator(len(s), seed, sigma_noise)
#         s_hat   = Delta_repulsion(A_repulsion, s_tilde)
#         categories = (s > 0).astype(int) 

#         choices, rewards = BE_model(
#             x, y, s_hat, categories,
#             sigma_noise, A_repulsion,
#             eta_learning, eta_relax,
#             no_response, seed, burn_in_seed, mode = 'simulated'
#         )

#     elif model_type == 'SC':
#         # # start with unbiased (uniform) beliefs for categories A & B
#         # A_dist = np.ones_like(x)
#         # B_dist = np.ones_like(x)
#         # # normalize so these integrate to 1 over x
#         # A_dist /= trapezoid(A_dist, x)
#         # B_dist /= trapezoid(B_dist, x)

#         # gamma        = param_dict['gamma']
#         # sigma_update = param_dict['sigma_update']

#         # choices, rewards = SC_model(
#         #     x, A_dist, B_dist,    # initial category distributions
#         #     s,
#         #     sigma_noise, A_repulsion,
#         #     gamma, sigma_update,
#         #     no_response, seed, burn_in_seed,
#         # )
#         gamma        = param_dict['gamma']
#         sigma_update = param_dict['sigma_update']
#         # initial category priors for SC
#         A_dist = np.ones_like(x); A_dist /= trapezoid(A_dist, x)
#         B_dist = np.ones_like(x); B_dist /= trapezoid(B_dist, x)
#         # <<< CHANGED: pass s_hat & categories, include mode >>>
#         choices, rewards = SC_model(
#             x,
#             A_dist,
#             B_dist,
#             s_hat,
#             categories,
#             sigma_noise,
#             A_repulsion,
#             gamma,
#             sigma_update,
#             no_response,
#             seed,
#             burn_in_seed,
#             mode='simulated'
#         )

#     else:
#         raise ValueError("Invalid model_type: must be 'BE' or 'SC'")

#     # assemble output
#     df_sim = pd.DataFrame({
#         'Participant_ID': pid,
#         'Trial':         trial,
#         'choice':        choices,
#         'correct':       rewards,
#         'No_response':   no_response,
#         'stim_relative': s,
#         'block':         block
#     })

#     return df_sim

def simulate_subject_trials(participant_df, model_type, param_dict, seed=42):
    """
    Simulate one subject's trial-by-trial choices & rewards under either the BE or SC generative model,
    given their optimal parameters and the empirical stimulus sequence.

    Args:
        participant_df (pd.DataFrame): columns must include
            'stim_relative', 'No_response', 'block', 'Trial', 'Participant_ID'.
        model_type (str): 'BE' or 'SC'.
        param_dict (dict): contains the required parameters for the chosen model:
            for BE: {'A_repulsion', 'sigma_noise', 'eta_learning', 'eta_relax'}
            for SC: {'A_repulsion', 'sigma_noise', 'gamma', 'sigma_update'}
        seed (int): base random seed for reproducibility.

    Returns:
        pd.DataFrame with columns ['Participant_ID','Trial','choice','correct',
                                   'No_response','stim_relative','block'].
    """
    # --- unpack data ---
    s           = participant_df['stim_relative'].to_numpy()
    no_response = participant_df['No_response'].to_numpy()
    block       = participant_df['block'].to_numpy()
    trial       = participant_df['Trial'].to_numpy()
    pid         = participant_df['Participant_ID'].iloc[0]
    n_trials    = len(s)

    # --- unpack parameters ---
    A_repulsion = param_dict['A_repulsion']
    sigma_noise = param_dict['sigma_noise']
    burn_in_seed = 5 * seed  # consistent burn-in

    # --- stimulus space grid (x) ---
    max_range   = 1 + 6*sigma_noise + 2*A_repulsion*(1 + 6*sigma_noise)
    min_range   = -1 - 6*sigma_noise - 2*A_repulsion*(1 + 6*sigma_noise)
    num_points  = round((max_range - min_range) * 1000)
    x_space     = np.linspace(min_range, max_range, num_points)

    # --- initial distributions ---
    if model_type == 'BE':
        # uniform prior over boundary
        y_pdf = np.ones_like(x_space) / len(x_space)

    elif model_type == 'SC':
        # uniform priors for categories A and B
        A_dist = np.ones_like(x_space) / len(x_space)
        B_dist = np.ones_like(x_space) / len(x_space)

    # --- per-trial inputs ---
    categories = (s > 0).astype(int)
    s_tilde    = s + Noise_generator(n_trials, seed, sigma_noise)  # noisy perception
    s_hat      = Delta_repulsion(A_repulsion, s_tilde)              # perceived boundary

    # --- run the model ---
    if model_type == 'BE':
        eta_learning = param_dict['eta_learning']
        eta_relax    = param_dict['eta_relax']

        choices, rewards = BE_model(
            x_space,
            y_pdf,
            s_hat,
            categories,
            sigma_noise,
            A_repulsion,
            eta_learning,
            eta_relax,
            no_response,
            seed,
            burn_in_seed,
            mode='simulated'
        )

    elif model_type == 'SC':
        gamma        = param_dict['gamma']
        sigma_update = param_dict['sigma_update']

        choices, rewards = SC_model(
            x_space,
            A_dist,
            B_dist,
            s_hat,
            categories,
            sigma_noise,
            A_repulsion,
            gamma,
            sigma_update,
            no_response,
            seed,
            burn_in_seed,
            mode='simulated'
        )

    else:
        raise ValueError("Invalid model_type: must be 'BE' or 'SC'")

    # --- assemble output ---
    df_sim = pd.DataFrame({
        'Participant_ID': pid,
        'Trial':         trial,
        'choice':        choices,
        'correct':       rewards,
        'No_response':   no_response,
        'stim_relative': s,
        'block':         block
    })

    return df_sim


def plot_update_matrix(update_matrix, title='', annot = False,
                       ax=None, vmin=-0.2, vmax=0.2, save_path=None):
    """
    Plots a heatmap based on the update matrix, formatted to match the given screenshot.

    Parameters:
    update_matrix (numpy.ndarray): An update matrix with shape (num_bins, num_bins).
    title (str): A string to use as the title.
    ax (matplotlib.axes.Axes, optional): An existing matplotlib axis to plot on.
    save_path (str, optional): Path to save the figure.
    """
    num_bins = update_matrix.shape[0]

    # Define midpoints for labeling
    midpoints = np.linspace(-1, 1, num_bins)

    # Flip rows so lowest stimulus at bottom
    data = update_matrix[::-1, :]

    # Create a custom diverging colormap
    cvals = [-0.2, 0, 0.2]
    colors = ['darkorange', 'white', 'blueviolet']
    norm2 = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(norm2(cvals), colors))
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('', tuples)

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # choose fmt for annotations: .3g = max 3 significant figures
    fmt = '.2g' if annot else ''

    # Plot heatmap
    g = sns.heatmap(
        data,
        ax=ax,
        cmap=cmap2,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
        cbar_kws={'label': 'Δ Bias Towards A'},
        square=True,
        xticklabels=[f"{m:.1f}" for m in midpoints],
        yticklabels=[f"{m:.1f}" for m in midpoints[::-1]]
    )

    # rotate the colorbar label
    cbar = g.collections[0].colorbar
    cbar.set_label('Δ Bias Towards A', rotation=270, labelpad=15)

    # Set labels and title
    ax.set_title(title or 'Post correct')
    ax.set_xlabel('Previous stimulus')
    ax.set_ylabel('Current stimulus')

    # plt.tight_layout()

    # Save figure if requested
    if save_path and ax is None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Show plot if standalone
    if ax is None:
        plt.show()

def plot_conditional_psychometric(
    s, chooseB, reward, No_response, Not_Blockstart,
    ax, cmap_def=None, plot_kwargs=None
):
    # 1) compute tables & lists
    _, table     = post_correct_update_matrix(s, chooseB, reward, No_response, Not_Blockstart)
    cond_s_list, cond_choiceB_list = post_correct_conditional_stimuli(
        s, chooseB, reward, No_response, Not_Blockstart
    )
    s_post, cb_post = post_correct_total_stimuli(s, chooseB, reward, No_response, Not_Blockstart)

    # 2) midpoints & dense grid
    intervals = np.linspace(-1, 1, 9)
    midpoints = (intervals[:-1] + intervals[1:]) / 2
    x_dense   = np.linspace(-0.875, 0.875,200)
    total_curve, _ = fit_Psych_curve(s_post[0], cb_post[0], x_dense)

    # 3) colormap setup
    if cmap_def is None:
        cdict = [
            (0.00, "mistyrose"),
            (0.33, "darkred"),
            (0.66, "darkblue"),
            (1.00, "lightblue"),
        ]
        cmap_def = LinearSegmentedColormap.from_list("RdBl_custom", cdict)
    norm = Normalize(vmin=-1, vmax=1)

    # prepare the ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_def)
    sm.set_array([])

    plot_kwargs = plot_kwargs or {}

    # 4) draw onto the passed-in ax
    for j, prev in enumerate(midpoints):
        color = cmap_def(norm(prev))
        ax.plot(midpoints, table[:, j], 'o', color=color, alpha=0.8, **plot_kwargs)
        y_fit, _ = fit_Psych_curve(cond_s_list[j], cond_choiceB_list[j], x_dense)
        ax.plot(x_dense, y_fit, '-', color=color, alpha=0.7, **plot_kwargs)

    # 5) overlay grand average
    ax.plot(x_dense, total_curve, 'k-', linewidth=3, label='Grand average')

    # 6) colorbar on the same figure
    cbar = ax.get_figure().colorbar(
        sm,
        ax=ax,
        label='Previous stimulus',
        pad=0.02    # default is 0.15, so 0.02 pulls it in much tighter
    )   
    cbar.set_label('Previous stimulus', rotation=270, labelpad=10)
    # set the tick locations on the colorbar
    ticks = [-1, -0.5, 0, 0.5, 1]
    cbar.set_ticks(ticks)

    # (optional) explicitly set how they’re rendered
    cbar.set_ticklabels([f"{t:.1f}" for t in ticks])

    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 7) styling
    ax.set_xlim(-1, 1)
    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.set_xlabel('Stimulus Relative')
    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks(np.arange(0, 1.1, 0.5))
    ax.set_ylabel('P(choose B)')
    ax.set_title('Conditional Psychometric Curves')
    ax.legend(loc='lower right', frameon=False)

    return ax


# def plot_conditional_psychometrics_from_table(
#     table,
#     *,
#     ax=None,
#     x_min=-0.875,
#     x_max=0.875,
#     n_dense=200,
#     cmap_def=None,
#     plot_kwargs=None,
#     flip_horizontal=False
# ):
#     """
#     Mirror‐around‐zero psychometric curves from a raw 8×8 "table" of P(choose B).
#     If `ax` is given, draws into it; otherwise makes a new fig/ax.
#     Returns (fig, ax, cbar) where fig is None if ax was passed in.
#     """

#     # --- 1) intervals → midpoints
#     intervals = np.linspace(-1, 1, 9)
#     midpoints = (intervals[:-1] + intervals[1:]) / 2   # length 8

#     # --- 2) mirrored grids
#     x_dense   = np.linspace(x_min, x_max, n_dense)
#     x_mirror  = -x_dense
#     mid_mirror = -midpoints

#     # --- 3) colormap
#     if cmap_def is None:
#         cdict = [
#             (0.00, "mistyrose"),  # -1.0
#             (0.33, "darkred"),
#             (0.66, "darkblue"),
#             (1.00, "lightblue"),  # +1.0
#         ]
#         cmap_def = LinearSegmentedColormap.from_list("RdBl_custom", cdict)
#     norm = Normalize(vmin=-1, vmax=1)

#     # --- 4) fig/ax setup
#     fig = None
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8,6))
#     plot_kwargs = plot_kwargs or {}

#     # --- 5) plot each bin
#     for j, prev in enumerate(midpoints):
#         y = table[:, j]
#         color = cmap_def(norm(prev))

#         ax.plot(mid_mirror,        y,  'o', color=color, alpha=0.8)
#         spline = UnivariateSpline(midpoints, y, s=0)
#         ax.plot(x_mirror, spline(x_dense), '-', color=color, alpha=0.7, **plot_kwargs)

#     # --- 6) grand-average
#     grand   = table.mean(axis=1)
#     spline_g = UnivariateSpline(midpoints, grand, s=0)
#     ax.plot(x_mirror, spline_g(x_dense),
#             'k-', linewidth=3, label='Grand average')

#     # --- 7) colorbar
#     sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_def)
#     sm.set_array([])
#     cbar = ax.figure.colorbar(sm, ax=ax, label="Previous stimulus")
#     cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
#     cbar.set_label("Previous stimulus", rotation=270, labelpad=15)

#     # --- 8) styling
#     ax.set_xlim(-1, 1)
#     if flip_horizontal:
#     # flip across y-axis
#         ax.invert_xaxis()
#     ax.set_xticks(np.arange(-1, 1.1, 0.5))
#     ax.set_ylim(-0.01, 1.01)
#     ax.set_yticks(np.arange(0, 1.1, 0.5))
#     ax.set_xlabel('Stimulus Relative')
#     ax.set_ylabel('P(choose B)')
#     ax.set_title('Conditional Psychometric Curves')

#     if fig is not None:
#         fig.tight_layout()

#     return fig, ax, cbar

def plot_conditional_psychometrics_from_table(
    table,
    *,
    ax=None,
    x_min=-0.875,
    x_max=0.875,
    n_dense=200,
    cmap_def=None,
    plot_kwargs=None,
    flip_horizontal=False,  # Optionally mirror flip along the y-axis
    save_path=None         # Optional path to save the figure
):
    """
    Mirror‑around‑zero psychometric curves from an 8×8 table of P(choose B).
    If `ax` is provided, plots into it; otherwise creates a new figure & axis.
    Returns (fig, ax, cbar) where fig is None if an external ax was provided.

    If `flip_horizontal` is True, inverts the x-axis to mirror plots over x=0.
    If `save_path` is given and a new figure was created, saves the figure there.
    """

    # --- 1) compute midpoints
    intervals   = np.linspace(-1, 1, table.shape[1] + 1)
    midpoints   = (intervals[:-1] + intervals[1:]) / 2  # length equal to num cols

    # --- 2) prepare dense & mirrored grids
    x_dense     = np.linspace(x_min, x_max, n_dense)
    x_mirror    = -x_dense
    mid_mirror  = -midpoints

    # --- 3) default colormap
    if cmap_def is None:
        cdict = [
            (0.00, "mistyrose"),  # -1.0
            (0.33, "darkred"),
            (0.66, "darkblue"),
            (1.00, "lightblue")   # +1.0
        ]
        cmap_def = LinearSegmentedColormap.from_list("RdBl_custom", cdict)
    norm = Normalize(vmin=-1, vmax=1)

    # --- 4) figure/axis setup
    fig_created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig_created = True
    else:
        fig = None
    plot_kwargs = plot_kwargs or {}

    # --- 5) plot each conditional curve
    for j, prev in enumerate(midpoints):
        y      = table[:, j]
        color  = cmap_def(norm(prev))

        # scatter points and spline line
        ax.plot(mid_mirror, y, 'o', color=color, alpha=0.8)
        spline = UnivariateSpline(midpoints, y, s=0)
        ax.plot(x_mirror, spline(x_dense), '-', color=color, alpha=0.7, **plot_kwargs)

    # --- 6) grand-average curve
    grand     = table.mean(axis=1)
    spline_g  = UnivariateSpline(midpoints, grand, s=0)
    ax.plot(x_mirror, spline_g(x_dense), 'k-', linewidth=3, label='Grand average')

    # --- 7) colorbar
    sm   = plt.cm.ScalarMappable(norm=norm, cmap=cmap_def)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, label="Previous stimulus")
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_label("Previous stimulus", rotation=270, labelpad=15)

    # --- 8) styling & optional flip
    ax.set_xlim(-1, 1)
    if flip_horizontal:
        ax.invert_xaxis()
    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.5))
    ax.set_xlabel('Stimulus Relative')
    ax.set_ylabel('P(choose B)')
    ax.set_title('Conditional Psychometric Curves')
    ax.legend(loc='upper left')

    # --- 9) save or show if standalone
    if save_path and fig_created:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if fig_created:
        plt.tight_layout()
        plt.show()

    return fig, ax, cbar


def plot_conditional_psychometrics_from_table_V2(
    table,
    *,
    ax=None,
    x_min=-0.875,
    x_max=0.875,
    n_dense=200,
    cmap_def=None,
    plot_kwargs=None,
    flip_horizontal=False,
    save_path=None
):
    """
    Mirror‑around‑zero psychometric curves from an 8×8 table of P(choose B).
    If `ax` is provided, plots into it; otherwise creates a new figure & axis.
    Returns (fig, ax, cbar) where fig is None if an external ax was provided.

    If `flip_horizontal` is True, inverts the x-axis to mirror plots over x=0.
    If `save_path` is given and a new figure was created, saves the figure there.
    """
    # 1) midpoints and dense grid
    num_bins = table.shape[1]
    intervals = np.linspace(-1, 1, num_bins + 1)
    midpoints = (intervals[:-1] + intervals[1:]) / 2
    x_dense = np.linspace(x_min, x_max, n_dense)
    x_mirror = -x_dense
    mid_mirror = -midpoints

    # 2) colormap setup
    if cmap_def is None:
        cdict = [
            (0.00, "mistyrose"),
            (0.33, "darkred"),
            (0.66, "darkblue"),
            (1.00, "lightblue"),
        ]
        cmap_def = LinearSegmentedColormap.from_list("RdBl_custom", cdict)
    norm = Normalize(vmin=-1, vmax=1)

    # 3) figure/axis setup
    fig_created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig_created = True
    else:
        fig = None
    plot_kwargs = plot_kwargs or {}

    # 4) plot each conditional fit using fit_Psych_curve
    for j, prev in enumerate(midpoints):
        y_scatter = table[:, j]
        color = cmap_def(norm(prev))

        # scatter actual table points (mirrored)
        ax.plot(mid_mirror, y_scatter, 'o', color=color, alpha=0.8)
        # fit a psychometric curve to the binned probabilities
        y_fit, _ = fit_Psych_curve(midpoints, y_scatter, x_dense)
        ax.plot(x_mirror, y_fit, '-', color=color, alpha=0.7, **plot_kwargs)

    # 5) grand-average fit from table means
    grand = table.mean(axis=1)
    grand_fit, _ = fit_Psych_curve(midpoints, grand, x_dense)
    ax.plot(x_mirror, grand_fit, 'k-', linewidth=3, label='Grand average')

    # 6) colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_def)
    sm.set_array([])
    cbar = ax.figure.colorbar(
        sm,
        ax=ax,
        label="Previous stimulus",
        pad=0.02
    )
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_label("Previous stimulus", rotation=270, labelpad=10)

    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 7) styling & optional flip
    ax.set_xlim(-1, 1)
    if flip_horizontal:
        ax.invert_xaxis()
        ax.set_xticks(np.arange(1, -1.1, -0.5))  # reverse ticks
        # set ticklabels
        ax.set_xticklabels([f"{-t:.1f}" for t in np.arange(1, -1.1, -0.5)])
    else:
        ax.set_xticks(np.arange(-1, 1.1, 0.5))
    # ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.set_ylim(-0.025, 1.025)
    ax.set_yticks(np.arange(0, 1.1, 0.5))
    ax.set_xlabel('Stimulus Relative')
    ax.set_ylabel('P(choose B)')
    ax.set_title(' ')
    ax.legend(loc='upper left', frameon=False)

    # 8) save or show
    if save_path and fig_created:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if fig_created:
        plt.tight_layout()
        plt.show()

    return fig, ax, cbar


from scipy.optimize import minimize

def plot_all_conditional_psychometrics_from_table(
    data_df,
    best_params_df,
    participant_ids,
    output_path
):
    """
    For each participant, page in PDF with three panels:
      [ empirical | BE model | SC model ]
    using the *conditional* table (mirrored) routine.
    """

    with PdfPages(output_path) as pdf:
        for pid in participant_ids:
            # 1×3 panel
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # extract raw subject data
            subj_df = data_df[data_df['Participant_ID'] == pid].reset_index(drop=True)
            subj_df['is_not_start_of_block'] = subj_df['block'].eq(subj_df['block'].shift())
            s           = subj_df['stim_relative'].to_numpy()
            chooseB     = subj_df['choice'].to_numpy()
            reward      = subj_df['correct'].to_numpy()
            No_response = subj_df['No_response'].to_numpy()
            Not_Block   = subj_df['is_not_start_of_block'].to_numpy()
            
            # 3) empirical
            plot_conditional_psychometric(
                s, chooseB, reward, No_response, Not_Block,
                ax=axes[0]
            )
            axes[0].set_title('Empirical')
            

            # -- BE model table from best_params_df
            be_row = best_params_df.query(
                "Participant_ID==@pid and Model=='BE'"
            ).iloc[0]
            table_be = be_row['Best_Model_Test_CM']
            plot_conditional_psychometrics_from_table(
                table_be, ax=axes[1]
            )
            axes[1].set_title('BE Model')

            # -- SC model table
            sc_row = best_params_df.query(
                "Participant_ID==@pid and Model=='SC'"
            ).iloc[0]
            table_sc = sc_row['Best_Model_Test_CM']
            plot_conditional_psychometrics_from_table(
                table_sc, ax=axes[2]
            )
            axes[2].set_title('SC Model')

            fig.suptitle(f'Conditional Psychometric Curves — {pid}', fontsize=16)
            fig.tight_layout(rect=[0,0,1,0.95])
            pdf.savefig(fig)
            plt.close(fig)

def plot_all_update_matrices(data_df, param_df, participant_ids, output_path):
    with PdfPages(output_path) as pdf:
        for pid in participant_ids:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # empirical, BE, SC

            participant_df = data_df[data_df['Participant_ID'] == pid].reset_index(drop=True)
            participant_params = param_df[param_df['Participant_ID'] == pid].reset_index(drop=True)
            participant_df['is_not_start_of_block'] = participant_df['block'].eq(participant_df['block'].shift())

            s = participant_df['stim_relative'].to_numpy()
            chooseB = participant_df['choice'].to_numpy()
            reward = participant_df['correct'].to_numpy()
            No_response = participant_df['No_response'].to_numpy()
            Not_Blockstart = participant_df['is_not_start_of_block'].to_numpy()

            emp_update_matrix, _ = post_correct_update_matrix(s, chooseB, reward, No_response, Not_Blockstart)

            be_params = participant_params[participant_params['Model'] == 'BE'].iloc[0].to_dict()
            be_df = simulate_subject_trials(participant_df, 'BE', be_params)
            be_update_matrix, _ = post_correct_update_matrix(
                be_df['stim_relative'], be_df['choice'], be_df['correct'],
                be_df['No_response'], Not_Blockstart
            )

            sc_params = participant_params[participant_params['Model'] == 'SC'].iloc[0].to_dict()
            sc_df = simulate_subject_trials(participant_df, 'SC', sc_params)
            sc_update_matrix, _ = post_correct_update_matrix(
                sc_df['stim_relative'], sc_df['choice'], sc_df['correct'],
                sc_df['No_response'], Not_Blockstart
            )

            # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            plot_update_matrix(emp_update_matrix, title=f'Empirical', ax=axes[0])
            plot_update_matrix(be_update_matrix, title='BE Model', ax=axes[1])
            plot_update_matrix(sc_update_matrix, title='SC Model', ax=axes[2])
            # participant_id as suptitle
            fig.suptitle(f'Update Matrices for {pid}', fontsize=16)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.show()

def plot_all_psychometrics(data_df, best_params_df, participant_ids, output_pdf):
    x_vals = np.linspace(-1, 1, 300)

    with PdfPages(output_pdf) as pdf:
        for pid in participant_ids:
            # load and preprocess real data
            df = data_df[data_df['Participant_ID']==pid].reset_index(drop=True)
            participant_params = best_params_df[best_params_df['Participant_ID'] == pid].reset_index(drop=True)
            df['is_not_start_of_block'] = df['block'].eq(df['block'].shift())
            s = df['stim_relative'].to_numpy()
            c = df['choice'].to_numpy()
            r = df['correct'].to_numpy()
            nr = df['No_response'].to_numpy()
            nbs = df['is_not_start_of_block'].to_numpy()

            # empirical
            sel_s, sel_c = post_correct_total_stimuli(s, c, r, nr, nbs)
            stim_emp = np.concatenate(sel_s)
            choice_emp = np.concatenate(sel_c)
            y_emp, _, = fit_Psych_curve(stim_emp, choice_emp, x_vals)

            be_params = participant_params[participant_params['Model'] == 'BE'].iloc[0].to_dict()

            df_be = simulate_subject_trials(df, 'BE', be_params)
            sel_s, sel_c = post_correct_total_stimuli(
                df_be['stim_relative'], df_be['choice'], df_be['correct'],
                df_be['No_response'], nbs)
            stim_be = np.concatenate(sel_s)
            choice_be = np.concatenate(sel_c)
            y_be, _, = fit_Psych_curve(stim_be, choice_be, x_vals)
    
            sc_params = participant_params[participant_params['Model'] == 'SC'].iloc[0].to_dict()
            df_sc = simulate_subject_trials(df, 'SC', sc_params)
            sel_s, sel_c = post_correct_total_stimuli(
                df_sc['stim_relative'], df_sc['choice'], df_sc['correct'],
                df_sc['No_response'], nbs)
            stim_sc = np.concatenate(sel_s)
            choice_sc = np.concatenate(sel_c)
            y_sc, _, = fit_Psych_curve(stim_sc, choice_sc, x_vals)

            # plot
            fig, axes = plt.subplots(1,3,figsize=(15,4), sharey=True)
            axes[0].plot(x_vals, y_emp, color='black')
            axes[0].scatter(stim_emp, choice_emp, alpha=0.3, s=10)
            axes[0].set_title(f"Empirical")
            axes[0].set_xlabel("Stimulus")
            axes[0].set_ylabel("P(Choose B)")

            axes[1].plot(x_vals, y_be, color='blue')
            axes[1].scatter(stim_be, choice_be, alpha=0.3, s=10)
            axes[1].set_title("BE model")
            axes[1].set_xlabel("Stimulus")

            axes[2].plot(x_vals, y_sc, color='green')
            # if not sc_row.isnull().any():
            axes[2].scatter(stim_sc, choice_sc, alpha=0.3, s=10)
            axes[2].set_title("SC model")
            axes[2].set_xlabel("Stimulus")
            fig.suptitle(f'Psychometric Curves for {pid}', fontsize=16)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def plot_parameters_across_seeds(seed_df, species, save_path=None):
    """
    Plot individual‐seed estimates and mean±std for each model parameter.
    
    Parameters
    ----------
    seed_df : pd.DataFrame
        One row per seed with columns:
        - Participant_ID
        - Model    (values "BE" or "SC")
        - Best_Params   (list or array of length 4, or None)
    output_dir : str
        Directory where PDFs will be saved.
    species : str
        Species label to include in filenames.
    seed_color : str
        Color for individual‐seed scatter points.
    mean_color : str
        Color for mean±std errorbars.
    """
    # 1. clean Best_Params → always 4‐element list
    seed_df = seed_df.copy()
    seed_df['Best_Params_clean'] = seed_df['Best_Params'].apply(
        lambda x: x if isinstance(x, (list, tuple, np.ndarray)) else [np.nan]*4
    )
    # 2. expand into p1,p2,p3,p4
    params = pd.DataFrame(
        seed_df['Best_Params_clean'].tolist(),
        index=seed_df.index,
        columns=['p1','p2','p3','p4']
    )
    df = pd.concat([seed_df, params], axis=1)
    # 3. name parameters
    df['sigma_noise']  = df['p1']
    df['A_repulsion']  = df['p2']
    df['eta_learning'] = np.where(df['Model']=='BE', df['p3'], np.nan)
    df['eta_relax']    = np.where(df['Model']=='BE', df['p4'], np.nan)
    df['sigma_update'] = np.where(df['Model']=='SC', df['p3'], np.nan)
    df['gamma']        = np.where(df['Model']=='SC', df['p4'], np.nan)
    # 4. compute summary
    summary = (
        df
        .groupby(['Participant_ID','Model'])[
            ['sigma_noise','A_repulsion','eta_learning','eta_relax','sigma_update','gamma']
        ]
        .agg(['mean','std'])
        .reset_index()
    )
    participants = sorted(df['Participant_ID'].unique())
    x = np.arange(len(participants))
    
    # 5. for each model, plot and save
    for model in ['BE','SC']:
        keys = (['sigma_noise','A_repulsion','eta_learning','eta_relax']
                if model=='BE'
                else ['sigma_noise','A_repulsion','sigma_update','gamma'])
        fig, axes = plt.subplots(1, len(keys), figsize=(5*len(keys), 5), sharey=True)
        
        for ax, param in zip(axes, keys):
            # individual seeds
            for i, pid in enumerate(participants):
                vals = df.loc[
                    (df['Participant_ID']==pid) & (df['Model']==model),
                    param
                ].values
                ax.scatter(np.full_like(vals, i), vals,
                           color='darkslategray', alpha=0.3,
                             zorder=1)
            # mean ± std
            means = summary.loc[summary['Model']==model, (param,'mean')].values
            errs  = summary.loc[summary['Model']==model, (param,'std')].values
            ax.errorbar(x, means, yerr=errs,
                        fmt='o-', color='darkslategray', capsize=5, zorder=2)
            ax.set_title(param)
            ax.set_xticks(x)
            ax.set_xticklabels(participants, rotation=45)
            ax.set_xlabel("Participant")
        
        axes[0].set_ylabel("Parameter value")
        fig.suptitle(f"{model} parameters across seeds ({species})", fontsize=16)
        plt.tight_layout(rect=[0,0,1,0.95])
        if save_path:
            plt.savefig(f'{save_path}/params_across_seeds_{species}_{model}.pdf', dpi=300)
            plt.close(fig)
        else:
            plt.show()

def plot_psycho_params_over_trials(
    df,
    participant_ids,
    pdf_path,
    n_splits: int = 5,
    max_trials: int = None,
    y_lims: dict = None
):
    """
    Generates psychometric parameter trajectories for each participant
    and saves all figures to a multipage PDF, with optional hard-coded y-limits.

    Naming conventions for arrays are preserved:
        s, chooseB, reward, No_response, Not_Blockstart

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'Participant_ID','stim_relative','choice',
        'No_response','block'.
    participant_ids : list of str
        List of Participant_ID values to include.
    pdf_path : str
        File path to save the multipage PDF.
    n_splits : int
        Number of trial checkpoints.
    max_trials : int or None
        Cap on number of trials; if None, uses full length.
    y_lims : dict or None
        Optional dict mapping parameter names to (ymin, ymax) tuples.
        Keys should be among {"mu","sd","gamma","lambda"}.
    """
    with PdfPages(pdf_path) as pdf:
        for pid in participant_ids:
            pid_df = df[df['Participant_ID'] == pid].reset_index(drop=True)
            pid_df['is_not_start_of_block'] = pid_df['block'].eq(pid_df['block'].shift())
            # Naming conventions
            s              = pid_df['stim_relative'].values
            chooseB        = pid_df['choice'].astype(int).values
            reward         = pid_df['correct'].astype(int).values
            No_response    = pid_df['No_response'].astype(bool).values
            Not_Blockstart = pid_df['is_not_start_of_block'].astype(bool).values

            N = len(s) if max_trials is None else min(max_trials, len(s))
            step = N // n_splits
            cuts = np.arange(step, N + 1, step)

            intervals = np.linspace(-1, 1, 9)
            midpoints = (intervals[:-1] + intervals[1:]) / 2

            params_over_time = []
            for cut in cuts:
                mask = (~No_response[:cut]) & (Not_Blockstart[:cut])
                _, popt, = fit_Psych_curve(
                    s[:cut][mask],
                    chooseB[:cut][mask],
                    midpoints
                )
                params_over_time.append(popt)
            params_over_time = np.array(params_over_time)

            fig, axs = plt.subplots(1, 4, figsize=(16, 4), tight_layout=True)
            names = ["mu", "sd", "gamma", "lambda"]
            for i, name in enumerate(names):
                ax = axs[i]
                ax.plot(cuts, params_over_time[:, i], marker='o')
                ax.axhline(0, color='k', linestyle=':')  # black dotted line at zero
                ax.set_xlabel("Trials used")
                ax.set_ylabel(name)
                ax.set_title(name)
                if y_lims is not None and name in y_lims:
                    ax.set_ylim(*y_lims[name])

            fig.suptitle(f"{pid}", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Saved psychometric parameters to {pdf_path}")

def plot_update_matrices_over_trials(
    df,
    participant_ids,
    pdf_path,
    n_splits: int = 5,
    max_trials: int = None
):
    """
    Generates post-correct update matrices for each participant
    and saves all figures to a multipage PDF.

    Naming conventions for arrays are preserved:
        s, chooseB, reward, No_response, Not_Blockstart
    """
    with PdfPages(pdf_path) as pdf:
        for pid in participant_ids:
            pid_df = df[df['Participant_ID'] == pid].reset_index(drop=True)
            # ensure block-start flag
            pid_df['is_not_start_of_block'] = pid_df['block'].eq(pid_df['block'].shift())
            # naming conventions
            s              = pid_df['stim_relative'].values
            chooseB        = pid_df['choice'].astype(int).values
            reward         = pid_df['correct'].astype(int).values
            No_response    = pid_df['No_response'].astype(bool).values
            Not_Blockstart = pid_df['is_not_start_of_block'].astype(bool).values

            # define cuts
            N = len(s) if max_trials is None else min(max_trials, len(s))
            step = N // n_splits
            cuts = np.arange(step, N + 1, step)

            # create figure with one subplot per cut
            fig, axs = plt.subplots(1, len(cuts),
                                    figsize=(6 * len(cuts), 5),
                                    tight_layout=True)
            for ax, cut in zip(axs, cuts):
                U, _ = post_correct_update_matrix(
                    s[:cut],
                    chooseB[:cut],
                    reward[:cut],
                    No_response[:cut],
                    Not_Blockstart[:cut]
                )
                plot_update_matrix(
                    U,
                    title=f"First {cut} trials",
                    ax=ax
                )

            fig.suptitle(f"{pid}", fontsize=16)
            # reserve space for suptitle
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Saved update matrices to {pdf_path}")

def compute_update_matrix_frobnorms(
    data_df: pd.DataFrame,
    param_df: pd.DataFrame,
    participant_ids: list[str]
) -> pd.DataFrame:
    """
    For each participant in `participant_ids`, compute:
      ‒ Frob_norm(empirical ‒ BE_model)
      ‒ Frob_norm(empirical ‒ SC_model)

    Returns a DataFrame with columns:
      ['Participant_ID', 'frob_emp_be', 'frob_emp_sc']
    """
    records = []

    for pid in participant_ids:
        # 1) slice subject data and build arrays
        pid_df = data_df[data_df['Participant_ID'] == pid].reset_index(drop=True)
        pid_df['is_not_start_of_block'] = pid_df['block'].eq(pid_df['block'].shift())

        s               = pid_df['stim_relative'].to_numpy()
        chooseB         = pid_df['choice'].to_numpy()
        reward          = pid_df['correct'].to_numpy()
        No_response     = pid_df['No_response'].to_numpy()
        Not_Blockstart  = pid_df['is_not_start_of_block'].to_numpy()

        # 2) empirical update matrix
        emp_U, _ = post_correct_update_matrix(
            s, chooseB, reward, No_response, Not_Blockstart
        )

        # 3) simulate BE and compute its update matrix
        be_params = param_df.loc[
            (param_df['Participant_ID']==pid)&(param_df['Model']=='BE')
        ].iloc[0].to_dict()
        be_df = simulate_subject_trials(pid_df, 'BE', be_params)
        be_U, _ = post_correct_update_matrix(
            be_df['stim_relative'].to_numpy(),
            be_df['choice'].to_numpy(),
            be_df['correct'].to_numpy(),
            be_df['No_response'].to_numpy(),
            Not_Blockstart  # reuse the same block‐start mask
        )

        # 4) simulate SC and compute its update matrix
        sc_params = param_df.loc[
            (param_df['Participant_ID']==pid)&(param_df['Model']=='SC')
        ].iloc[0].to_dict()
        sc_df = simulate_subject_trials(pid_df, 'SC', sc_params)
        sc_U, _ = post_correct_update_matrix(
            sc_df['stim_relative'].to_numpy(),
            sc_df['choice'].to_numpy(),
            sc_df['correct'].to_numpy(),
            sc_df['No_response'].to_numpy(),
            Not_Blockstart
        )

        # 5) compute Frobenius norms
        frob_emp_be = np.linalg.norm(emp_U - be_U, ord='fro')
        frob_emp_sc = np.linalg.norm(emp_U - sc_U, ord='fro')

        records.append({
            'Participant_ID': pid,
            'frob_emp_be': frob_emp_be,
            'frob_emp_sc': frob_emp_sc
        })

    return pd.DataFrame.from_records(records)

def plot_frobenius_norms(
    frob_df,
    y_max: float = None,
    pdf_path: str = None,
    show_plot: bool = True
):
    """
    Plots Empirical vs BE and Empirical vs SC Frobenius norms
    for each participant as a grouped bar chart.

    Parameters
    ----------
    frob_df : pandas.DataFrame
        Must have columns ['Participant_ID', 'frob_emp_be', 'frob_emp_sc'].
    pdf_path : str or None
        If given, saves the figure to this path.
    show_plot : bool
        Whether to call plt.show() at the end.
    """
    participants = frob_df['Participant_ID'].tolist()
    frob_emp_be  = frob_df['frob_emp_be'].to_numpy()
    frob_emp_sc  = frob_df['frob_emp_sc'].to_numpy()

    x = np.arange(len(participants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(5, len(x)*0.5), 4), dpi = 150)
    ax.bar(x - width/2, frob_emp_be, width, label='Emp vs BE')
    ax.bar(x + width/2, frob_emp_sc, width, label='Emp vs SC')

    ax.set_xticks(x)
    ax.set_xticklabels(participants, rotation=45, ha='right')
    ax.set_ylabel('Frobenius Norm')
    ax.set_title('Empirical vs. Model Update Matrix Distances')
    ax.set_ylim(0, y_max) if y_max else None
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()

    if pdf_path:
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"Saved Frobenius-norm plot to {pdf_path}")
    if show_plot:
        plt.show()

    plt.close(fig)

def plot_update_mat_emp_minus_model(
    data_df: pd.DataFrame,
    param_df: pd.DataFrame,
    participant_ids: list[str],
    output_path: str
):
    """
    For each participant, compute and plot:
      1. Empirical minus BE update matrix
      2. Empirical minus SC update matrix
    Saves all figures into a single PDF at output_path.
    """
    with PdfPages(output_path) as pdf:
        for pid in participant_ids:
            participant_df    = data_df[data_df['Participant_ID'] == pid].reset_index(drop=True)
            participant_params = param_df[param_df['Participant_ID'] == pid].reset_index(drop=True)
            participant_df['is_not_start_of_block'] = participant_df['block'].eq(participant_df['block'].shift())

            s              = participant_df['stim_relative'].to_numpy()
            chooseB        = participant_df['choice'].to_numpy()
            reward         = participant_df['correct'].to_numpy()
            No_response    = participant_df['No_response'].to_numpy()
            Not_Blockstart = participant_df['is_not_start_of_block'].to_numpy()

            # empirical update matrix
            emp_update_matrix, _ = post_correct_update_matrix(
                s, chooseB, reward, No_response, Not_Blockstart
            )

            # BE model update matrix
            be_params       = participant_params[participant_params['Model'] == 'BE'].iloc[0].to_dict()
            be_df           = simulate_subject_trials(participant_df, 'BE', be_params)
            be_update_matrix, _ = post_correct_update_matrix(
                be_df['stim_relative'].to_numpy(),
                be_df['choice'].to_numpy(),
                be_df['correct'].to_numpy(),
                be_df['No_response'].to_numpy(),
                Not_Blockstart
            )

            # SC model update matrix
            sc_params       = participant_params[participant_params['Model'] == 'SC'].iloc[0].to_dict()
            sc_df           = simulate_subject_trials(participant_df, 'SC', sc_params)
            sc_update_matrix, _ = post_correct_update_matrix(
                sc_df['stim_relative'].to_numpy(),
                sc_df['choice'].to_numpy(),
                sc_df['correct'].to_numpy(),
                sc_df['No_response'].to_numpy(),
                Not_Blockstart
            )

            # compute differences
            diff_be = emp_update_matrix - be_update_matrix
            diff_sc = emp_update_matrix - sc_update_matrix

            # plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            plot_update_matrix(diff_be, title='Empirical − BE', ax=axes[0])
            plot_update_matrix(diff_sc, title='Empirical − SC', ax=axes[1])
            fig.suptitle(f'Difference Matrices for {pid}', fontsize=16)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)