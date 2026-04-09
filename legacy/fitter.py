import numpy as np
from scipy.stats import norm, uniform
from scipy.optimize import minimize
import pandas as pd
from joblib import Parallel, delayed
from legacy.be import Noise_generator, Delta_repulsion
from tqdm import tqdm

import logging

# ───────────────────────────────────────────────────────────────────────────────
# Configure logging for Fitter.py:
logging.basicConfig(
    filename="fitter_errors.log",   # ← this is the file that will collect errors
    filemode="w",                        # overwrite on each run
    level=logging.WARNING,               # capture WARNING, ERROR, and CRITICAL
    format="%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s"
)
logging.warning("🔍 Logging initialized in Fitter.py")
# ───────────────────────────────────────────────────────────────────────────────


"""
x_axis variable refers to the parameter corresponding to x-axis when plotting the loss surface heatmap
y_axis variable refers to the parameter corresponding to y-axis when plotting the loss surface heatmap

In BE model: 
    x_axis : eta_relax
    y_axis : eta_learning

In SC model:
    x_axis : sigma_update
    y_axis : gamma
"""


def psychometric_model(x, mu, sd, gamma, lamda):
    # Inputs:
    # x: Array representing stimulus values.
    # mu: Mean of the psychometric function, representing the point of subjective equality (PSE).
    # sd: Standard deviation, representing the sensitivity or slope of the psychometric curve.
    # gamma: Lapse rate for lower asymptote, which accounts for guessing or error at low stimulus values.
    # lamda: Lapse rate for upper asymptote, accounting for guessing or error at high stimulus values.

    # The psychometric function calculates the probability of a subject making a certain choice
    # based on the given stimulus value (x), subject's PSE (mu), sensitivity (sd), and lapse rates (gamma, lamda).

    # The formula:
    # y = gamma + (1 - gamma - lamda) * norm.cdf(x, mu, sd)
    # gamma: Defines the lower bound of the probability (minimum response probability).
    # (1 - gamma - lamda): Defines the range between the lower bound (gamma) and the upper bound (1 - lamda).
    # norm.cdf(x, mu, sd): Calculates the cumulative probability of the subject making a certain choice as x increases.
    x = np.asarray(x, dtype=np.float64)
    y = gamma + (1 - gamma - lamda) * norm.cdf(x, mu, sd)

    return y


def neg_log_likelihood(params, stimulus, choiceB):
    # Inputs:
    # params   : list or array, [mu, sd, gamma, lamda].
    # stimulus: Array of stimulus values presented to the subject.
    # choiceB: Array of binary responses (0 or 1) representing the subject's choices (0 for A, 1 for B).

    stimulus = np.asarray(stimulus , dtype=np.float64)
    choiceB = np.asarray(choiceB , dtype=np.float64)

    # Extract the fitted parameters: mean (mu), standard deviation (sd), lapse rates (gamma, lambda)
    mu, sd, gamma, lamda = params

    # Use the parameters to generate the psychometric curve y
    y = psychometric_model(stimulus, mu, sd, gamma, lamda)
    # To prevent issues with log(0), we clip the predicted probabilities.
    eps = np.finfo(float).eps
    y = np.clip(y, eps, 1 - eps)

    # Calculate the log likelihood for each observation and sum it up.
    logL = choiceB * np.log(y) + (1 - choiceB) * np.log(1 - y)
    return -np.sum(logL)


def fit_Psych_curve(stimulus, choiceB, x, p0=None):
    # Inputs:
    # stimulus: Array of stimulus values presented to the subject.
    # choiceB: Array of binary responses (0 or 1) representing the subject's choices (0 for A, 1 for B).
    # x: Array representing the range of stimulus values for which we want to fit the psychometric curve.
    # p0: Optional initial parameter estimates (mu, sd, gamma, lambda) for the psychometric model.

    stimulus = np.asarray(stimulus , dtype=np.float64)
    choiceB = np.asarray(choiceB , dtype=np.float64)

    # If no initial guess is provided, use some reasonable defaults.
    if p0 is None:
        p0 = [0.0, 1.0, 0.05, 0.05]

    # Set bounds for the parameters: adjust these based on your specific problem.
    bounds = [(-1., 1.), (0.01, 10.), (0., 0.5), (0., 0.5)]

    # Use minimize to optimize the negative log likelihood.
    result = minimize(neg_log_likelihood, p0, args=(stimulus, choiceB), bounds=bounds, method='L-BFGS-B')

    if result.success:
        # If the optimization was successful, extract the optimal parameters
        popt = result.x  # 'result.x' contains the best-fit values for (mu, sd, gamma, lambda)

        # Unpack the optimized parameters into individual variables for readability
        mu_fit, sd_fit, gamma_fit, lamda_fit = popt

        # Use the optimized parameters to compute the fitted psychometric curve
        y_fit = psychometric_model(x, mu_fit, sd_fit, gamma_fit, lamda_fit)

        # Return the fitted curve and the optimized parameters
        return y_fit, popt
    else:
        # If the optimization fails, print the error message provided by the optimizer
        print("Optimization failed:", result.message)

        # Return None to indicate that fitting was unsuccessful
        return None, None


def post_correct_total_stimuli(s, chooseB, reward, No_response, Not_Blockstart):
    # Inputs:
    # s: Array of stimulus values presented during the trials.
    # chooseB: Array of binary subject choices (0 for A, 1 for B).
    # reward: Array of rewards (1 for correct, 0 for incorrect).
    # No_response: Boolean array indicating whether the subject responded in the trial (False = responded).
    # Not_Blockstart: Boolean array indicating whether the trial is not the start of a session (True = not start).

    # Convert inputs to numpy arrays for easier manipulation
    reward = np.array(reward)
    chooseB = np.array(chooseB)
    s = np.array(s)
    No_response = np.array(No_response)
    Not_Blockstart = np.array(Not_Blockstart)

    # Prepare lists to store the filtered stimuli and choices
    selected_stimuli = []
    choiceB = []

    # Identify trials where the subject was rewarded (reward == 1) in the previous trial
    previous_reward_1 = (reward[:-1] == 1)

    # Condition to ensure that the subject responded in both the current and previous trials
    no_response_condition = (No_response[1:] == False) & (No_response[:-1] == False)

    # Ensure the current trial is not the start of a session (block)
    no_session_start = (Not_Blockstart[1:] == True)

    # Combine the conditions: previous trial was rewarded, subject responded, and not a session start
    condition = previous_reward_1 & no_response_condition & no_session_start

    # Select the stimulus values and corresponding choices for trials meeting all conditions
    selected_s = s[1:][condition]
    selected_choiceB = chooseB[1:][condition]

    # Append the filtered results to the output lists
    selected_stimuli.append(selected_s)
    choiceB.append(selected_choiceB)

    # Return the lists of selected stimuli and corresponding choices after rewarded trials
    return selected_stimuli, choiceB


def post_correct_conditional_stimuli(s, chooseB, reward, No_response, Not_Blockstart):
    # Inputs:
    # s: Array of stimulus values presented during the trials.
    # chooseB: Array of binary subject choices (0 for A, 1 for B).
    # reward: Array of rewards (1 for correct, 0 for incorrect).
    # No_response: Boolean array indicating whether the subject responded in the trial (False = responded).
    # Not_Blockstart: Boolean array indicating whether the trial is not the start of a session (True = not start).

    # Convert inputs to numpy arrays for easier manipulation
    reward = np.array(reward)
    chooseB = np.array(chooseB)
    s = np.array(s)
    No_response = np.array(No_response)
    Not_Blockstart = np.array(Not_Blockstart)

    # Create 8 equal intervals between -1 and 1
    intervals = np.linspace(-1, 1, num=9)  # 9 points define 8 intervals

    # Assign elements in s to intervals using digitization (binning)
    indices = np.digitize(s, intervals) - 1  # Subtract 1 for 0-based indexing

    # Prepare lists to store the selected stimuli and choices for each interval
    selected_stimuli = []
    choiceB = []

    # Loop through each interval to filter data based on the interval of the previous trial's stimulus
    for interval_idx in range(len(intervals) - 1):  # Loop through the 8 intervals (0 to 7)
        # Condition to check if the previous trial's stimulus falls within the current interval
        previous_in_interval = (indices[:-1] == interval_idx)

        # Condition to check if the previous trial was rewarded (reward == 1)
        previous_reward_1 = (reward[:-1] == 1)

        # Condition to check if No_response for current and previous trials is False (i.e., subject responded)
        no_response_condition = (No_response[1:] == False) & (No_response[:-1] == False)

        # Condition to check if the current trial is not the start of a session (block)
        no_session_start = (Not_Blockstart[1:] == True)

        # Combine all conditions to filter trials
        condition = previous_in_interval & previous_reward_1 & no_response_condition & no_session_start

        # Select stimuli and choices for trials that meet the combined condition
        selected_s = s[1:][condition]
        selected_choiceB = chooseB[1:][condition]

        # Append the filtered stimuli and choices to the respective lists
        selected_stimuli.append(selected_s)
        choiceB.append(selected_choiceB)

    # Return the lists of selected stimuli and corresponding choices for each interval
    return selected_stimuli, choiceB


def post_correct_update_matrix(s, chooseB, reward, No_response, Not_Blockstart):
    # Inputs:
    # s: Array of stimulus values for each trial.
    # chooseB: Array of binary choices (0 = chose A, 1 = chose B) for each trial.
    # reward: Array of rewards for each trial (1 = rewarded, 0 = not rewarded).
    # No_response: Boolean array indicating whether the subject responded (False = responded).
    # Not_Blockstart: Boolean array indicating whether the trial is not the start of a block (True = not start).

    # Convert inputs to numpy arrays for easier manipulation
    reward = np.array(reward)
    chooseB = np.array(chooseB)
    s = np.array(s)
    No_response = np.array(No_response)
    Not_Blockstart = np.array(Not_Blockstart)

    # Create 8 equal intervals between -1 and 1 to group stimuli, and calculate the midpoints of each interval
    intervals = np.linspace(-1, 1, num=9)  # 9 points define 8 intervals between -1 and 1
    midpoints = (intervals[:-1] + intervals[1:]) / 2  # Midpoints of each interval for psychometric fitting

    # Get post-correct total psychometric values using all post-corrected stimuli and choices
    s_postCorrect, chooseB_postCorrect = post_correct_total_stimuli(s, chooseB, reward, No_response, Not_Blockstart)
    total_pc_Psychometric, _ = fit_Psych_curve(s_postCorrect[0], chooseB_postCorrect[0], midpoints)

    # If the total fit failed, fill with NaNs and log a warning
    if total_pc_Psychometric is None:
        logging.warning(
            "post_correct_update_matrix: total_pc_Psychometric is None—filling all with NaNs"
        )
        total_pc_Psychometric = np.full(8, np.nan)  

    # Get stimuli and choices conditioned on the previous trial's stimulus being in a specific interval
    conditional_pc_stimuli, conditional_pc_choiceB = post_correct_conditional_stimuli(
        s, chooseB, reward, No_response, Not_Blockstart
    )

    # Initialize matrices to store the conditional psychometric values and the update matrix
    table = np.zeros((8, 8))  # To store conditional post-correct psychometric values
    update_matrix = np.zeros((8, 8))  # To store the difference from total psychometric values

    # Loop over each interval to calculate psychometric curves and update matrix
    for i in range(8):
        if len(conditional_pc_stimuli[i]) == 0:  # If no stimuli for this interval
            y1 = np.full(8, np.nan)  # Fill with NaN values if no data
        else:
            y1, _ = fit_Psych_curve(conditional_pc_stimuli[i], conditional_pc_choiceB[i],
                                    midpoints)  # Fit psychometric curve

            # If curve fit failed (returned None), replace with NaNs and log a warning
            if y1 is None:  
                logging.warning(
                    f"post_correct_update_matrix: fit_Psych_curve returned None at interval {i}, filling with NaNs"
                )
                y1 = np.full(8, np.nan) 

        # Store the psychometric values for this condition (previous stimulus in interval i) in the table
        table[:, i] = np.array(y1)

        # Compute the update matrix: difference between conditional psychometric values and total psychometric values
        update_matrix[:, i] = np.array(y1) - np.array(total_pc_Psychometric)

    # Return the update matrix and the table of conditional psychometric values
    return update_matrix, table


def PsychometricCurves(s, chooseB, reward, No_response, Not_Blockstart):
    # Inputs:
    # s: Array of stimulus values for each trial.
    # chooseB: Array of binary choices (0 = chose A, 1 = chose B) for each trial.
    # reward: Array of rewards for each trial (1 = rewarded, 0 = not rewarded).
    # No_response: Boolean array indicating whether the subject responded (False = responded).
    # Not_Blockstart: Boolean array indicating whether the trial is not the start of a block (True = not start).

    # Convert inputs to numpy arrays for easier manipulation
    reward = np.array(reward)
    chooseB = np.array(chooseB)
    s = np.array(s)
    No_response = np.array(No_response)

    # Define a finely spaced array of stimulus values for psychometric curve fitting
    x = np.linspace(-1, 1, 100000)  # 100,000 points between -1 and 1

    # Post-correct total psychometric values (over all trials)
    s_postCorrect, chooseB_postCorrect = post_correct_total_stimuli(s, chooseB, reward, No_response, Not_Blockstart)

    # Fit the psychometric curve for post-corrected total data using all valid stimuli and choices
    total_pc_Psychometric, total_opt = fit_Psych_curve(s_postCorrect[0], chooseB_postCorrect[0], x)

    # Get post-corrected stimuli and choices conditioned on the previous stimulus' interval
    conditional_pc_stimuli, conditional_pc_choiceB = post_correct_conditional_stimuli(
        s, chooseB, reward, No_response, Not_Blockstart
    )

    # Initialize lists to store fitted psychometric curves and optimal parameters for each condition
    y_fit = []  # To store the fitted psychometric curve for each condition
    optimum_params = []  # To store the optimal parameters of the psychometric curve for each condition

    # Loop over the 8 stimulus intervals (from the conditional analysis)
    for i in range(8):
        if len(conditional_pc_stimuli[i]) == 0:  # If no stimuli for this condition (previous stimulus in this interval)
            y1 = np.full(8, np.nan)  # Fill with NaN if there is no data for this interval
            opt = np.nan  # No optimal parameters if no data
        else:
            # Fit the psychometric curve for the post-corrected stimuli and choices conditioned on the previous interval
            y1, opt = fit_Psych_curve(conditional_pc_stimuli[i], conditional_pc_choiceB[i], x)

        # Append the fitted psychometric curve and the optimal parameters for the current condition
        y_fit.append(y1)
        optimum_params.append(opt)

    # Return the total post-corrected psychometric curve, its optimal parameters,
    # as well as the fitted curves and optimal parameters for each conditional interval
    return total_pc_Psychometric, total_opt, y_fit, optimum_params


def total_psychometric(s, chooseB, No_response):
    """
    Computes the psychometric curve using trials where the subject responded.

    Parameters:
        s (array-like): Stimulus values per trial.
        chooseB (array-like): Binary choices made by the subject (0 = A, 1 = B).
        No_response (array-like): Boolean array indicating no-response trials (True = no response, False = responded).

    Returns:
        psych_curve (ndarray): The fitted psychometric curve evaluated over a fine stimulus grid.
        fit_params (ndarray): Optimal parameters of the psychometric curve.
    """
    s = np.asarray(s)
    chooseB = np.asarray(chooseB)
    No_response = np.asarray(No_response)

    # Select only trials where the subject responded
    valid_trials = ~No_response
    s_valid = s[valid_trials]
    chooseB_valid = chooseB[valid_trials]

    # Define finely spaced stimulus values for evaluating the psychometric curve
    x = np.linspace(-1, 1, 100000)

    # Fit and return the psychometric curve and optimal parameters
    psych_curve, fit_params = fit_Psych_curve(s_valid, chooseB_valid, x)
    return psych_curve, fit_params


def compute_pse(mu, sigma, gamma, lapse):
    denominator = 1 - gamma - lapse
    if denominator <= 0:
        raise ValueError("Invalid parameters: 1 - gamma - lapse must be > 0")

    phi_inv_input = (0.5 - gamma) / denominator

    if not (0 < phi_inv_input < 1):
        raise ValueError("Invalid input to inverse CDF. Check gamma and lapse values.")

    return norm.ppf(phi_inv_input, loc=mu, scale=sigma)


def rolling_PSE_time_series(s, chooseB, No_response, m, overlap_pct=80):
    """
    Computes PSE in a rolling window with specified overlap percentage.

    Parameters:
    - s, chooseB, reward, No_response, Not_Blockstart : 1D arrays of length N (trials)
    - m : int, window size in trials
    - overlap_pct : float (0 to <100), percent overlap between consecutive windows

    Returns:
    - trial_centers : center trial indices of windows
    - pse_vals : computed PSE values per window
    """
    if not (0 <= overlap_pct < 100):
        raise ValueError("overlap_pct must be in the range [0, 100).")

    step = max(1, round(m * (1 - overlap_pct / 100)))
    N = len(s)
    
    start_indices = list(range(0, N - m + 1, step))
    pse_vals = []
    trial_centers = []

    for i in start_indices:
        win_s = s[i:i + m]
        win_choice = chooseB[i:i + m]
        win_no_response = No_response[i:i + m]

        try:
            _, fit_params = total_psychometric(win_s, win_choice, win_no_response)
            mu, sd, gamma, lamda = fit_params
            pse = compute_pse(mu, sd, gamma, lamda)
            pse_vals.append(pse)
            trial_centers.append(i + m // 2)
        except:
            pse_vals.append(np.nan)
            trial_centers.append(i + m // 2)

    return np.array(trial_centers), np.array(pse_vals)



# def matrix_error(Model_matrix, data_matrix):
#     # Inputs:
#     # Model_matrix: Update matrix obtained from the model.
#     # data_matrix: Update matrix obtained from the actual data.

#     # Calculate the squared differences between the two matrices
#     squared_diff = (Model_matrix - data_matrix) ** 2

#     # Count the number of non-NaN elements in the squared differences for each column
#     non_nan_count = np.sum(~np.isnan(squared_diff), axis=0)

#     # Calculate the number of columns that contain at least one non-NaN value
#     non_nan_columns = np.sum(non_nan_count > 0)

#     # Compute the sum of squared differences column-wise, ignoring NaN elements
#     columnwise_sum = np.nansum(squared_diff, axis=0)

#     # Calculate the total least square error by summing the column-wise sums
#     # and normalizing by the number of columns with non-NaN elements
#     cost = np.sum(columnwise_sum) / non_nan_columns

#     return cost

def matrix_error(Model_matrix, data_matrix):
    """Mean squared error between two matrices, ignoring NaNs.

    Matches behav_utils.analysis.update_matrix.matrix_error.
    """
    diff = Model_matrix - data_matrix
    valid = ~np.isnan(diff)
    if np.sum(valid) == 0:
        return np.nan
    return np.mean(diff[valid] ** 2)


def cost_function(data_matrix, model, func, x, y, s, s_hat, categories, sigma_noise, A_repulsion, y_axis_value,
                  x_axis_value, no_response, Not_Blockstart, seed, mode_pre, fit_with):
    # Inputs:
    # data_matrix: The actual observed data update matrix to compare against.
    # model: The model function that generates the predicted update matrix. (BE model or SC model)
    # func: Additional function passed to the model to calculate update matrix from model predictions.
    # x, y, s, s_hat, categories, sigma_noise, A_repulsion, y_axis_value, x_axis_value, no_response, Not_Blockstart, seed:
    # Various parameters required by the model to generate the update matrix (Model_matrix).

    # Generate the Model_matrix using the provided model and its parameters.
    Model_Up_matrix,Model_Cond_matrix = model(func, x, y, s, s_hat, categories, sigma_noise, A_repulsion, y_axis_value, x_axis_value,
                         no_response,
                         Not_Blockstart, seed, mode_pre)

    # Compute the cost (error) by comparing the corresponding Model_matrix with the data_matrix
    # using the matrix_error function, which calculates the error between two update matrices.
    if fit_with=='conditional':
        cost = matrix_error(Model_Cond_matrix, data_matrix)
    elif fit_with=='update':
        cost = matrix_error(Model_Up_matrix, data_matrix)
    else:
        raise ValueError("Invalid fit_with. Use 'conditional' or 'update'.")


    # Return the calculated cost value (lower values indicate a better model fit).
    return cost


def parameter_sweep(data_matrix, model, func, x, y, s, s_hat, categories, sigma_noise, A_repulsion,
                    x_axis_values, y_axis_values, no_response, Not_Blockstart, seed, mode_pre, fit_with):
    """
    This function performs a parameter sweep by iterating over a grid of `x_axis_values` and `y_axis_values`,
    computing a cost/error for each pair, and identifying the parameters that minimize the error.

    Args:
        data_matrix: The update matrix from the observed data to compare against.
        model: The model to evaluate and optimize.(BE or SC)
        func: The function used to create update matrix.
        x, y, s, s_hat: Variables or matrices representing data points or predictions used in model fitting.
        categories: The classification categories used in the model.
        sigma_noise: A parameter that defines noise in the model.
        A_repulsion: A parameter defining repulsion in the model.
        x_axis_values: Array of values to sweep over the x-axis (one model parameter).
        y_axis_values: Array of values to sweep over the y-axis (another model parameter).
        no_response: A condition indicating whether a response was made (used to filter data).
        Not_Blockstart: A condition indicating if the trial is not a starting trial of a block/session.
        seed: Random seed for reproducibility.

    Returns:
        errors: A 2D array of error values corresponding to each combination of `x_axis_values` and `y_axis_values`.
        optimal_params: The (x, y) parameters that yielded the lowest error.
        min_error: The minimum error obtained from the parameter sweep.
    """

    # Initialize a matrix to store the errors for each combination of parameters
    errors = np.zeros((len(y_axis_values), len(x_axis_values)))  # Error matrix initialized to 0s

    # Start with a very high minimum error to track the lowest encountered error during the sweep
    min_error = float('inf')

    # This will store the optimal (x_axis_value, y_axis_value) that minimizes the error
    optimal_params = None

    def compute_error(y_axis_value, x_axis_value, i, j):
        """
        Computes the error for a given pair of x and y axis values, representing two parameters.

        Args:
            y_axis_value: The value of the parameter on the y-axis.
            x_axis_value: The value of the parameter on the x-axis.
            i, j: Indices of the current combination in the parameter grid.

        Returns:
            i, j: The indices of the current combination.
            error: The calculated error for the current parameter combination.
            (x_axis_value, y_axis_value): The parameter values used for this error computation.
        """
        # Call the cost function to calculate the error for the given parameter combination
        error = cost_function(data_matrix, model, func, x, y, s, s_hat, categories, sigma_noise, A_repulsion,
                              y_axis_value, x_axis_value, no_response, Not_Blockstart, seed, mode_pre,fit_with)
        return i, j, error, (x_axis_value, y_axis_value)

    # Perform the parameter sweep in parallel to speed up computation.
    # For each combination of x and y axis values, compute the error.
    results = Parallel(n_jobs=-1)(delayed(compute_error)(y_axis_value, x_axis_value, i, j)
                                  for i, y_axis_value in enumerate(y_axis_values)
                                  for j, x_axis_value in enumerate(x_axis_values))

    # Iterate through the results to populate the error matrix and track the minimum error
    for i, j, error, params in results:
        errors[i, j] = error  # Store the calculated error in the appropriate location in the matrix
        # If the current error is smaller than the previously tracked minimum error, update min_error and optimal_params
        if error < min_error:
            min_error = error
            optimal_params = params

    # Return the error matrix, the optimal parameters, and the minimum error
    return errors, optimal_params, min_error


# Calculate error tensor
def calculate_tensor(model, func, sigma_noise_values, A_repulsion_values, y_axis_values, x_axis_values, s,
                     categories, no_response, Not_Blockstart, data_matrix, seed, mode_pre,fit_with):
    """
    This function computes a 4D error tensor for different combinations of model parameters like `sigma_noise`,
    `A_repulsion`, and additional parameter sweeps over x and y axis values. This is useful for finding the best
    parameter set in models like SC (Stimulus Category) and BE (Boundary Estimation).

    Args:
        model: The model being used (could be either SC or BE model).
        func: A function that calculates the update matrix.
        sigma_noise_values: List of `sigma_noise` values (sensory noise) to sweep over.
        A_repulsion_values: List of `A_repulsion` values.
        y_axis_values: The y-axis values over which parameter sweep is performed, representing a secondary parameter.
        x_axis_values: The x-axis values over which parameter sweep is performed, representing a primary parameter.
        s: Sensory stimuli inputs.
        categories: Array of true categorical labels (0 or 1) for each trial (Category A or Category B).
        no_response: Boolean array indicating trials with no response, used to filter relevant trials.
        Not_Blockstart: Boolean array marking trials that are not the start of a block/session.
        data_matrix: The observed upadte matrix against which the model error is computed.
        seed: Random seed for reproducibility.

    Returns:
        errors_tensor: A 4D tensor where each entry corresponds to the error value for a specific combination
                       of (sigma_noise, A_repulsion, x_axis_value, y_axis_value).
        best_params_tensor: A 3D tensor where each entry stores the best x and y axis parameter values
                            for each combination of sigma_noise and A_repulsion.
                            Shape: (len(sigma_noise_values), len(A_repulsion_values), 2).
        global_min_error: The minimum error found across all parameter combinations.
        global_optimal_params: The parameters corresponding to the global minimum error (sigma_noise, A_repulsion,
                               x_axis_value, y_axis_value).
        global_min_error_matrix: The matrix of error values corresponding to the global minimum error.
    """

    # Initialize a tensor to store errors for each combination of (sigma_noise, A_repulsion, y_axis_value, x_axis_value).
    # The shape of the tensor is (number of sigma_noise values, number of A_repulsion values,
    # number of y_axis values, number of x_axis values).
    errors_tensor = np.zeros(
        (len(sigma_noise_values), len(A_repulsion_values), len(y_axis_values), len(x_axis_values)))

    # Tensor to store the best x_axis_value and y_axis_value for each (sigma_noise, A_repulsion) combination.
    best_params_tensor = np.zeros(
        (len(sigma_noise_values), len(A_repulsion_values), 2))  # 2 for storing x and y axis values.

    # Variables to track the global minimum error and corresponding optimal parameters.
    global_min_error = float('inf')  # Initialize with a very large value.
    global_optimal_params = None  # To store the optimal parameters corresponding to the minimum error.

    # Loop through all combinations of sigma_noise and A_repulsion values.
    for noise_idx, sigma_noise in enumerate(sigma_noise_values):
        for repulsion_idx, A_repulsion in enumerate(A_repulsion_values):
            # Apply noise to the stimuli (s) based on the current sigma_noise value.
            s_tilde = s + Noise_generator(len(s), seed, sigma_noise)

            # Apply boundary repulsion to adjust the stimulus estimate (s_hat).
            s_hat = Delta_repulsion(A_repulsion, s_tilde)

            # Generate uniform distribution for x-axis stimulus range, widened by 6 standard deviations of noise.
            # And also maximum Delta repulsion value and increase resolution with range increase
            # Also round the number of points to the closest integer
            max_range = 1 + 6 * sigma_noise + 2 * A_repulsion * (1 + 6 * sigma_noise)
            min_range = -1 - 6 * sigma_noise - 2 * A_repulsion * (1 + 6 * sigma_noise)
            num_points = round((max_range - min_range) * 1000)
            x = np.linspace(min_range, max_range, num_points)
            y = uniform.pdf(x, loc=min_range, scale=max_range - min_range)

            # Perform a parameter sweep over x_axis_values and y_axis_values, and calculate the error matrix.
            errors, optimal_params, min_error = parameter_sweep(data_matrix, model, func, x, y, s, s_hat, categories,
                                                                sigma_noise, A_repulsion, x_axis_values,
                                                                y_axis_values, no_response, Not_Blockstart, seed, mode_pre,fit_with)

            # Store the error values in the errors_tensor for the current combination of sigma_noise and A_repulsion.
            errors_tensor[noise_idx, repulsion_idx, :, :] = errors

            # Store the best (x, y) axis values that yielded the minimum error for this sigma_noise and A_repulsion.
            best_params_tensor[noise_idx, repulsion_idx, :] = optimal_params

            # If this combination results in a global minimum error, update the global optimal parameters.
            if min_error < global_min_error:
                global_min_error = min_error
                global_optimal_params = [sigma_noise, A_repulsion, optimal_params[0], optimal_params[1]]
                global_min_error_matrix = errors

    return errors_tensor, best_params_tensor, global_min_error, global_optimal_params, global_min_error_matrix


### cross-validation
def merge_smallest_adjacent(blocks, labels=None, num_folds=5):
    # If no labels are provided, default to using the indices of the blocks as labels
    if labels is None:
        labels = list(range(len(blocks)))

    # Initialize labeled_blocks as a list of single-element lists (each containing the label for the corresponding block)
    # This will help in tracking the merging process.
    labeled_blocks = [[label] for label in labels]

    # Create a copy of block sizes to track the size of each block during merging
    sizes = list(blocks)

    # Continue merging blocks until the number of blocks equals the desired number of folds
    while len(sizes) > num_folds:
        # Find the index of the smallest block
        min_idx = min(range(len(sizes)), key=lambda i: sizes[i])

        # If the smallest block is at the start, merge it with the adjacent block to the right
        if min_idx == 0:
            adjacent_idx = 1
        # If the smallest block is at the end, merge it with the adjacent block to the left
        elif min_idx == len(sizes) - 1:
            adjacent_idx = len(sizes) - 2
        else:
            # Otherwise, merge it with the smaller of the two adjacent blocks (left or right)
            adjacent_idx = min_idx - 1 if sizes[min_idx - 1] < sizes[min_idx + 1] else min_idx + 1

        # Merge the two blocks: combine labels and sizes, and remove the merged block
        if min_idx < adjacent_idx:
            # If merging with the right, append the adjacent block's labels to the current block
            labeled_blocks[min_idx].extend(labeled_blocks.pop(adjacent_idx))
            # Add the size of the adjacent block to the current block and remove the adjacent block's size
            sizes[min_idx] += sizes.pop(adjacent_idx)
        else:
            # If merging with the left, append the current block's labels to the adjacent block
            labeled_blocks[adjacent_idx].extend(labeled_blocks.pop(min_idx))
            # Add the size of the current block to the adjacent block and remove the current block's size
            sizes[adjacent_idx] += sizes.pop(min_idx)

    # Return the list of labeled blocks after merging
    return labeled_blocks


def select_and_concatenate(df, folds_idx, test_fold_idx):
    """
    Selects the DataFrame corresponding to the `test_fold_idx` from the list of folds and
    concatenates the remaining DataFrames into one DataFrame.

    Parameters:
    df (pd.DataFrame): The original DataFrame containing the data.
    folds_idx (list of lists): A list of lists where each inner list contains the block indices for a fold.
    test_fold_idx (int): The index of the fold to select as the test fold.

    Returns:
    selected_df (pd.DataFrame): The DataFrame corresponding to the test fold (test data).
    remaining_dfs (pd.DataFrame): A concatenated DataFrame of all other folds (training data).
    """

    # Create a list to hold the DataFrames corresponding to each fold
    grouped_dfs = []

    # Iterate through each fold (list of block indices) in folds_idx
    for block in folds_idx:
        # Extract the rows from df where the 'block' column values match the indices in the current fold
        group_df = df[df['block'].isin(block)]
        # Append the resulting DataFrame (group_df) to the list of grouped DataFrames
        grouped_dfs.append(group_df)

    # Select the DataFrame corresponding to the test_fold_idx
    selected_df = grouped_dfs[test_fold_idx]

    # Concatenate all the DataFrames except the one at test_fold_idx into a single DataFrame
    # This will be used as the training data
    remaining_dfs = pd.concat([df for idx, df in enumerate(grouped_dfs) if idx != test_fold_idx], ignore_index=True)

    return selected_df, remaining_dfs


def k_fold_CV(df, model, func, sigma_noise_values, A_repulsion_values, x_axis_values, y_axis_values, seed, k=5, mode_pre = 'simulated',fit_with = 'conditional', show_progress=False):
    """
    Perform k-fold cross-validation on a model to evaluate performance across different parameter combinations.

    Args:
        df: DataFrame containing experimental data.
        model: The model function for behavior simulation.
        func: A function used to calculate update matrix.
        sigma_noise_values: List of sigma_noise parameter values.
        A_repulsion_values: List of A_repulsion parameter values.
        x_axis_values: Values for the x-axis parameter in grid search.
        y_axis_values: Values for the y-axis parameter in grid search.
        seed: Random seed for reproducibility.
        k: Number of cross-validation folds (default 5).
        mode_pre: Mode of the model, either 'simulated' or 'empirical'.
        show_progress: Whether to show a progress bar during cross-validation (default True).

    Returns:
        optimal_params_list, test_errors_list, best_params, best_error, training_errors_list,
        Error_tensors_list, Errors_tensor_params_list, best_error_matrix, training_update_matrix,
        Model_training_update_matrix, test_update_matrix, Model_test_update_matrix
    """
    # Step 1: Extract label (block identifier) and sizes of each block from dataframe
    block_sizes = df.groupby('block')['Trial'].count().reset_index(name='count')
    sizes = block_sizes['count'].to_numpy()
    label = block_sizes['block'].to_numpy()

    assert fit_with in ['update', 'conditional'], "fit_with must be 'update' or 'conditional'"

    # Check if the number of blocks is less than k
    if len(sizes) == 1:
        raise ValueError("Only one block found. Cannot perform k-fold cross-validation.")
    elif len(sizes) < k:
        k = len(sizes)

    # Step 2: Merge adjacent blocks into k approximately equal folds
    blocks_in_folds = merge_smallest_adjacent(sizes, label, k)

    # Step 3: Initialize lists to store outputs for each fold
    optimal_params_list = []  # Optimal parameters for training data in each fold
    training_errors_list = []  # In-sample error (training set) for each fold
    test_errors_list = []  # Out-of-sample error (test set) for each fold
    Error_tensors_list = []  # Error tensors for each fold (grid search results)
    Errors_tensor_params_list = []  # Best parameters tensor for each fold (from grid search)
    best_error_matrix = []  # Best error matrix for each fold

    # Additional matrices for model update matrix comparisons
    training_update_matrix = []  # Real training data update matrix
    Model_training_update_matrix = []  # Model's training data update matrix
    test_update_matrix = []  # Real test data update matrix
    Model_test_update_matrix = []  # Model's test data update matrix

    # Matrices needed for fitting
    training_conditional_matrix = []  # Real training data conditional matrix
    Model_training_conditional_matrix = []  # Model's training data conditional matrix
    test_conditional_matrix = []  # Real test data conditional matrix
    Model_test_conditional_matrix = []  # Model's test data conditional matrix

    # Update matrix error, needed for better understanding
    Test_update_matrix_distance=[]  # The error between test (empirical) fold and model update matrix
    Train_update_matrix_distance=[] # The error between train (empirical) fold and model update matrix

    # Conditional matrix error, needed for better understanding
    Test_conditional_matrix_distance=[]  # The error between test (empirical) fold and model conditional matrix
    Train_conditional_matrix_distance=[] # The error between train (empirical) fold and model conditional matrix


    # Sanity-check error list----- I want to check if the training error coming out of 'calculate_tensor' is equal
    # to what we're getting from finding the distance between model fitted training matrix and empirical training matrix
    Sanity_check_fitted_training_error=[]
   
    if show_progress:
       fold_iterator = tqdm(range(k), desc="Cross-validation Progress", unit="fold")
    else:
       fold_iterator = range(k)

    for test_fold_idx in fold_iterator: 
        # Separate the test set and training set for the current fold
        test_df, training_df = select_and_concatenate(df, blocks_in_folds, test_fold_idx)

        # Step 5: Extract relevant columns for analysis (stimuli, choices, rewards, etc.)
        test_choice_correct = test_df[
            ['stim_relative', 'choice', 'correct', 'No_response', 'is_not_start_of_block']].to_numpy()
        training_choice_correct = training_df[
            ['stim_relative', 'choice', 'correct', 'No_response', 'is_not_start_of_block']].to_numpy()

        # Step 6: Extract arrays for training data
        s_training = training_choice_correct[:, 0]  # Stimulus values
        chooseB_training = training_choice_correct[:, 1]  # Choices
        rewards_training = training_choice_correct[:, 2]  # Rewards
        No_response_training = training_choice_correct[:, 3]  # No response
        Not_Blockstart_training = training_choice_correct[:, 4]  # Non-start of block indicator

        # Step 7: Extract arrays for test data (same as above)
        s_test = test_choice_correct[:, 0]
        chooseB_test = test_choice_correct[:, 1]
        rewards_test = test_choice_correct[:, 2]
        No_response_test = test_choice_correct[:, 3]
        Not_Blockstart_test = test_choice_correct[:, 4]

        # Step 8: Compute the real update matrices for training and test data
        data_training, psychs_training = post_correct_update_matrix(s_training, chooseB_training, rewards_training,
                                                                    No_response_training, Not_Blockstart_training)
        data_training_update_matrix = data_training[::-1]  # Reverse the update matrix
        training_update_matrix.append(data_training_update_matrix)  # Store it

        data_training_conditional_matrix = psychs_training[::-1]  # Reverse the conditional matrix
        training_conditional_matrix.append(data_training_conditional_matrix)  # Store it


        data_test, psychs_test = post_correct_update_matrix(s_test, chooseB_test, rewards_test,
                                                            No_response_test, Not_Blockstart_test)
        data_test_update_matrix = data_test[::-1]  # Reverse the update matrix
        test_update_matrix.append(data_test_update_matrix)  # Store it

        data_test_conditional_matrix = psychs_test[::-1]  # Reverse the conditional matrix
        test_conditional_matrix.append(data_test_conditional_matrix)  # Store it

        # Step 9: Categorize training and test stimuli (binary classification based on sign of stimulus)
        categories_training = np.where(s_training > 0, 1, 0)
        categories_test = np.where(s_test > 0, 1, 0)

        # Step 10: Perform grid search on training data to find optimal model parameters
        errors_tensor, best_params_tensor, global_min_error, global_optimal_params, global_min_error_matrix = calculate_tensor(
            model, func, sigma_noise_values, A_repulsion_values, y_axis_values, x_axis_values, s_training,
            categories_training, No_response_training, Not_Blockstart_training,
            data_training_conditional_matrix if fit_with=='conditional' else data_training_update_matrix, seed, mode_pre, fit_with=fit_with)

        # Step 11: Store results from grid search
        optimal_params_list.append(global_optimal_params)  # Best params for this fold
        training_errors_list.append(global_min_error)  # Training error
        Error_tensors_list.append(errors_tensor)  # Error tensor for this fold
        Errors_tensor_params_list.append(best_params_tensor)  # Best params tensor for this fold
        best_error_matrix.append(global_min_error_matrix)  # Best error matrix

        # Step 12: Extract the optimal parameters from grid search
        sigma_noise, A_repulsion, x_axis_value, y_axis_value = global_optimal_params

        # Step 13: Define the range of stimulus for update calculation
        max_range = 1 + 6 * sigma_noise + 2 * A_repulsion * (1 + 6 * sigma_noise)
        min_range = -1 - 6 * sigma_noise - 2 * A_repulsion * (1 + 6 * sigma_noise)
        num_points = round((max_range - min_range) * 1000)
        x = np.linspace(min_range, max_range, num_points)
        y = uniform.pdf(x, loc=min_range, scale=max_range - min_range)

        # Step 14: Apply the optimal model parameters to training and test stimuli
        s_training_tilde = s_training + Noise_generator(len(s_training), seed, sigma_noise)
        s_training_hat = Delta_repulsion(A_repulsion,
                                        s_training_tilde)  # Apply repulsion transformation to training data

        s_test_tilde = s_test + Noise_generator(len(s_test), seed, sigma_noise)
        s_test_hat = Delta_repulsion(A_repulsion, s_test_tilde)  # Apply repulsion transformation to test data

        # Step 15: Generate model-predicted update matrices for training and test sets using optimal parameters
        training_model_update, training_model_cond = model(func, x, y, s_training, s_training_hat, categories_training, sigma_noise,
                                    A_repulsion, y_axis_value, x_axis_value, No_response_training,
                                    Not_Blockstart_training, seed, mode_pre)
        test_model_update, test_model_cond = model(func, x, y, s_test, s_test_hat, categories_test, sigma_noise, A_repulsion,
                                y_axis_value, x_axis_value, No_response_test, Not_Blockstart_test, seed, mode_pre)

        # Step 16: Store the model-generated update & conditional matrices
        Model_training_update_matrix.append(training_model_update)  # Model's training data update matrix
        Model_test_update_matrix.append(test_model_update)  # Model's test data update matrix

        Model_training_conditional_matrix.append(training_model_cond)
        Model_test_conditional_matrix.append(test_model_cond)


        # Step 17: Compute out-of-sample (test) error by comparing model's predictions with actual test data
        if fit_with=='conditional':
            test_error = matrix_error(test_model_cond, data_test_conditional_matrix)
            tr_error = matrix_error(training_model_cond, data_training_conditional_matrix)
        elif fit_with=='update':
            test_error = matrix_error(test_model_update, data_test_update_matrix)
            tr_error = matrix_error(training_model_update, data_training_update_matrix )
        else:
            raise ValueError("Invalid fit_with. Use 'conditional' or 'update'.")

        test_errors_list.append(test_error)  # Store test error
        Sanity_check_fitted_training_error.append(tr_error) # Sanity check

        # Compute error between model and groundtruth update and conditional matrices
        up_mat_error_test = matrix_error(test_model_update,data_test_update_matrix)
        Test_update_matrix_distance.append(up_mat_error_test)

        up_mat_train = matrix_error(training_model_update, data_training_update_matrix)
        Train_update_matrix_distance.append(up_mat_train)


        cond_mat_error_test = matrix_error(test_model_cond,data_test_conditional_matrix )
        Test_conditional_matrix_distance.append(cond_mat_error_test)

        cond_mat_train = matrix_error(training_model_cond, data_training_conditional_matrix)
        Train_conditional_matrix_distance.append(cond_mat_train)



    # Step 18: After all folds are processed, identify the best fold (minimum test error)
    best_fold_idx = np.argmin(test_errors_list)
    best_params = optimal_params_list[best_fold_idx]  # Best parameters from best fold
    best_error = test_errors_list[best_fold_idx]  # Best (lowest) test error

    # Step 19: Return the results
    return (optimal_params_list, test_errors_list, best_params, best_error, training_errors_list,
            best_error_matrix, training_update_matrix, Model_training_update_matrix, test_update_matrix,
            Model_test_update_matrix,training_conditional_matrix,Model_training_conditional_matrix,
            test_conditional_matrix,Model_test_conditional_matrix,Test_update_matrix_distance,
            Train_update_matrix_distance,Test_conditional_matrix_distance,Train_conditional_matrix_distance,
            Sanity_check_fitted_training_error)


def grid_search_fit(df, model, func, sigma_noise_values, A_repulsion_values,
                     x_axis_values, y_axis_values, seed, mode_pre='simulated',
                     fit_with='conditional', show_progress=False):
    """
    Perform a single-grid search (no cross-validation) over model parameters,
    returning the same output structure as k_fold_CV but without any train/test split.

    Returns:
        optimal_params_list: list of best params (one element)
        test_errors_list: empty list
        best_params: best params tuple
        best_error: minimized error
        training_errors_list: list of training error (one element)
        best_error_matrix: list of error matrix (one element)
        training_update_matrix: list of empirical update matrix (one element)
        Model_training_update_matrix: list of model update matrix (one element)
        test_update_matrix: empty list
        Model_test_update_matrix: empty list
        training_conditional_matrix: list of empirical conditional matrix (one element)
        Model_training_conditional_matrix: list of model conditional matrix (one element)
        test_conditional_matrix: empty list
        Model_test_conditional_matrix: empty list
        Test_update_matrix_distance: empty list
        Train_update_matrix_distance: empty list
        Test_conditional_matrix_distance: empty list
        Train_conditional_matrix_distance: empty list
        Sanity_check_fitted_training_error: list of trained-model distance (one element)
    """
    # Extract all trials as "training"
    arr = df[['stim_relative', 'choice', 'correct', 'No_response', 'is_not_start_of_block']].to_numpy()
    s = arr[:, 0]
    chooseB = arr[:, 1]
    rewards = arr[:, 2]
    no_resp = arr[:, 3]
    not_block = arr[:, 4]
    categories = np.where(s > 0, 1, 0)

    # Empirical matrices
    up_mat, cond_mat = post_correct_update_matrix(s, chooseB, rewards, no_resp, not_block)
    emp_update = up_mat[::-1]
    emp_cond = cond_mat[::-1]

    categories = np.where(s > 0, 1, 0)

    # Grid search
    errors_tensor, best_params_tensor, min_error, opt_params, best_err_mat = calculate_tensor(
        model, func, sigma_noise_values, A_repulsion_values,
        y_axis_values, x_axis_values, s,
        categories, no_resp, not_block,
        emp_cond if fit_with == 'conditional' else emp_update,
        seed, mode_pre, fit_with)

    # Prepare output lists
    optimal_params_list = [opt_params]
    training_errors_list = [min_error]
    test_errors_list = []

    # Compute model matrices at optimal
    sigma_noise, A_repulsion, x_val, y_val = opt_params
    max_range = 1 + 6 * sigma_noise + 2 * A_repulsion * (1 + 6 * sigma_noise)
    min_range = -1 - 6 * sigma_noise - 2 * A_repulsion * (1 + 6 * sigma_noise)
    num = round((max_range - min_range) * 1000)
    x = np.linspace(min_range, max_range, num)
    y = uniform.pdf(x, loc=min_range, scale=max_range - min_range)
    s_tilde = s + Noise_generator(len(s), seed, sigma_noise)
    s_hat = Delta_repulsion(A_repulsion, s_tilde)

    model_up, model_cond = model(
        func, x, y, s, s_hat, categories,
        sigma_noise, A_repulsion, y_val, x_val,
        no_resp, not_block, seed, mode_pre)

    Model_training_update_matrix = [model_up]
    Model_training_conditional_matrix = [model_cond]

    # Distances
    tr_up_dist = matrix_error(model_up, emp_update)
    tr_cond_dist = matrix_error(model_cond, emp_cond)
    Sanity_check_fitted_training_error = [tr_cond_dist if fit_with == 'conditional' else tr_up_dist]

    # Empty placeholders for test-related outputs
    return (
        optimal_params_list,
        test_errors_list,
        opt_params,
        min_error,
        training_errors_list,
        [best_err_mat],
        [emp_update],
        Model_training_update_matrix,
        [],
        [],
        [emp_cond],
        Model_training_conditional_matrix,
        [],
        [],
        [],
        [tr_up_dist],
        [],
        [tr_cond_dist],
        Sanity_check_fitted_training_error
    )


