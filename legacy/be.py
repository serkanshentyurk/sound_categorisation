import numpy as np
import math
from scipy.stats import norm
from scipy.integrate import trapezoid


###########################################################################################
# Models
# BE Model


'''
original stimulus <0  ---> category A
original stimulus >0  ---> category B
'''

def Noise_generator(n, seed, sigma_noise):
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Mean of the noise is set to 0.0
    mu_noise = 0.0

    # Create a normal distribution with the specified mean and standard deviation (sigma_noise)
    noise_dist = norm(loc=mu_noise, scale=sigma_noise)

    # Generate n random noise values from the normal distribution
    noise = noise_dist.rvs(n)
    return noise


def Delta_repulsion(A_repulsion, s_tilde):
    # Get the length of the input sequence s_tilde
    n = len(s_tilde)

    # Initialize the output array s_hat with zeros, same length as s_tilde
    s_hat = np.zeros(n)

    # Set the first element of s_hat to match the first element of s_tilde
    s_hat[0] = s_tilde[0]

    # Loop through the rest of the elements in s_tilde
    for t in range(1, n):
        # Calculate the repulsion effect using A_repulsion, the difference between the current and previous elements,
        # and an exponential decay factor based on the absolute difference
        Delta_rep = A_repulsion * (s_tilde[t] - s_hat[t - 1]) * math.exp(- abs(s_tilde[t] - s_hat[t - 1]))

        # Update the current element in s_hat by adding the repulsion effect to s_tilde[t]
        s_hat[t] = s_tilde[t] + Delta_rep
    return s_hat


def Delta_learning(x, s_hat, category, sigma_boundary):
    # x: Array representing the stimulus space.
    #    This is the range of values over which the learning updates are calculated.

    # s_hat: Array of perceived stimulus values for each trial.
    #        It contains the model's estimate of the stimulus for each trial, against which the update will be calculated.

    # category: Array indicating the correct category (e.g., 0 for Category A and 1 for Category B) for each trial.
    #           This determines how the learning updates are applied based on the category of the stimulus.

    # sigma_boundary: Scalar value controlling the sensitivity of the learning update.
    #                 A higher value makes the model more sensitive to small differences between x and s_hat.

    # Get the length of the input sequence s_hat
    n = len(s_hat)

    # Initialize the Delta_learning matrix with zeros, dimensions (n, len(x))
    Delta_learning = np.zeros((n, len(x)))

    # Loop through each element in the sequence
    for i in range(n):
        # Set the category feedback factor in sigmoid: +1 if category is 1 (Category B), -1 if category is 0 (Category A)
        if category[i] == 1:
            C = 1
        elif category[i] == 0:
            C = -1

        # Calculate Delta_learning and filling the matrix row-wisely
        # i-th row of this matrix corresponds to Delta_learning that is going to be applied to model due to i-th stimulus

        # The logistic function is influenced by the category (C), sigma_boundary, and the difference (x - s_hat[i])
        Delta_learning[i, :] = 1 / (1 + np.exp(-sigma_boundary * C * (x - s_hat[i])))

    return Delta_learning


def find_closest_element(array, target):
    """
    Find the closest element of an array to a target number.

    Parameters:
        array (array-like): The input array.
        target (float): The target number.

    Returns:
        float: The closest element in the array to the target.
    """
    array = np.asarray(array)
    idx = np.abs(array - target).argmin()
    return idx


# Function to find the CDF value at a specific x
def calculate_cdf(x_value, x_array, pdf_array):
    # x_value: The specific point at which you want to calculate the CDF.
    # The function will integrate the PDF values up to this point.

    # x_array: A 1D array containing the x-values where the PDF is defined.
    # This should be sorted in ascending order and corresponds to the x-values of the PDF.

    # pdf_array: A 1D array of the same length as x_array, containing the PDF values.
    # Each element represents the probability density at the corresponding x-value.

    # Mask the array to include only values up to x_value
    mask = x_array <= x_value
    # Perform numerical integration using the trapezoidal rule
    # Only integrate over the portion of the array where the mask is True (i.e., up to x_value)
    integral = np.trapezoid(pdf_array[mask], x_array[mask])
    return integral


# Function to update the distribution
def update_distribution(x, y, s_hat, categories, Delta_learning, eta_learning, eta_relax, no_response, seed):
    # x: Array of x-values where the distribution is defined (e.g., stimulus space).
    # y: The current PDF (probability density function of boundary) values associated with x.
    # s_hat: Perceived stimulus values (array of n elements).
    # categories: Array indicating the correct category for each trial (0 for A, 1 for B).
    # Delta_learning: Learning update matrix (n x len(x)) that each row of it modifies the PDF based on learning.
    # eta_learning: Learning rate controlling the impact of Delta_learning on y.
    # eta_relax: Relaxation rate controlling how y moves back to equilibrium (uniform) after learning.
    # no_response: Boolean array indicating whether the subject provided a response for each trial.
    # seed: Random seed for reproducibility.


    np.random.seed(2 * seed)     # Set random seed for reproducibility
    Response = []    # Store subject's choices (0 for A, 1 for B)
    ProbB = []     # Store the probability of choosing category B
    n = len(s_hat) # Number of trials

    for i in range(n):
        # Check if the subject responded in the i-th trial
        if no_response[i] == False:
            # If the subject has responded:
            # Get the perceived stimulus (s_hat) for the i-th trial
            s = s_hat[i].astype(np.float32)

            # Find the closest element in x to s_hat[i]
            j = find_closest_element(x, s)

            # Calculate the cumulative probability (CDF) up to the closest x-value
            P_B = calculate_cdf(x[j], x, y)
            ProbB.append(P_B)     # Store the probability of choosing category B

            # Calculating subjects choice- 0 for A, 1 for B
            choice = np.random.binomial(1, P_B)
            Response.append(choice)     # Store the subject's response

            # Update the PDF by applying the learning rate and Delta_learning for the i-th trial
            y_prime = y - eta_learning * Delta_learning[i, :]

        else:
            # If there was no response, keep the PDF unchanged
            y_prime = y
            ProbB.append(np.nan)  # Mark no response with NaN
            Response.append(np.nan)  # No response is stored as NaN

        # Relaxation: Move the distribution back toward equilibrium after learning
        Delta_relax = y_prime - 0.5
        y = y_prime - eta_relax * Delta_relax

        # Check if there are any negative values in the updated PDF and shift them upward if needed
        min_y = np.min(y)
        if min_y < 0:
            y = y + np.abs(min_y)   # Ensure the PDF stays non-negative

        # Normalize the updated PDF to ensure its integral equals 1
        integral_y = trapezoid(y, x)
        y = y / integral_y

    # Calculate rewards based on whether the subject's response matched the correct category
    rewards = [1 if categories[i] == Response[i] else 0 for i in range(len(categories))]

    # Return the following outputs:
    # ProbB: A list containing the probability of choosing category B in each trial by subject.
    # Response: A list containing the subject's choice (0 for A, 1 for B), or NaN if no response.
    # rewards: A binary list indicating whether the subject's response was correct (1) or incorrect (0).
    # y: The updated PDF (probability distribution of boundary) after applying learning and relaxation for all trials
    return ProbB, Response, rewards, y



def BE_model(func, x, y, s, s_hat, categories, sigma_noise, A_repulsion, eta_learning, eta_relax,
             no_response, Not_Blockstart, seed, mode):
    """
    # Inputs:
    # func: A function to compute update matrix based on model choices and rewards.
    # x: Array representing the stimulus space.
    # y: Array representing the probability density function (PDF) of boundary over the stimulus space.
    # s: Original stimulus values after pretraining.
    # s_hat: Array of perceived stimulus values.
    # categories: Array of true categorical labels (0 or 1) for each trial (Category A or Category B).
    # sigma_noise: Standard deviation of the noise applied to the stimulus, determining uncertainty in the model.
    # A_repulsion: Scalar controlling the repulsion effect in the model, which adjusts the difference between s_tilde and s_hat.
    # eta_learning: Scalar controlling the learning rate for the Delta_learning updates.
    # eta_relax: Scalar controlling the relaxation parameter applied to the distribution in `update_distribution`.
    # no_response: Boolean array indicating whether the subject has responded or not in each trial.
    # Not_Blockstart: Array of booleans indicating trials that are not starting trial of a session (block).
    # seed: Integer used for seeding the random number generator for reproducibility.
    BE model that supports two modes:
    - 'simulated': Adds 1000 simulated expert trials to the beginning.
    - 'real': Uses only actual data and skips the first n_burn trials.
    """
    sigma_boundary = 1 / sigma_noise

    if mode == 'simulated':
        # simulated mode: simulate expert behavior
        np.random.seed(5 * seed)
        m = 1000
        s_pre = np.random.uniform(low=-1, high=1, size=m)
        categories_pre = np.where(s_pre > 0, 1, 0)
        s_tilde_pre = s_pre + Noise_generator(m, seed, sigma_noise)
        s_hat_pre = Delta_repulsion(A_repulsion, s_tilde_pre)
        no_response_pre = np.full(m, False)

        # Concatenate simulated expert trials with actual trials
        s_hat = np.concatenate((s_hat_pre, s_hat))
        categories = np.concatenate((categories_pre, categories))
        no_response = np.concatenate((no_response_pre, no_response))

        # Learning and update
        Delta_learning_matrix = Delta_learning(x, s_hat, categories, sigma_boundary)
        ProbB, chooseB, rewards, y = update_distribution(x, y, s_hat, categories, Delta_learning_matrix,
                                                         eta_learning, eta_relax, no_response, seed)

        # Exclude simulated trials from analysis
        data, psychs = func(s, chooseB[m:], rewards[m:], no_response[m:], Not_Blockstart)

    elif mode == 'real':
        # real mode: skip early real trials
        n_burn = 200
        Delta_learning_matrix = Delta_learning(x, s_hat, categories, sigma_boundary)
        ProbB, chooseB, rewards, y = update_distribution(x, y, s_hat, categories, Delta_learning_matrix,
                                                         eta_learning, eta_relax, no_response, seed)
        data, psychs = func(s[n_burn:], chooseB[n_burn:], rewards[n_burn:], no_response[n_burn:], Not_Blockstart[n_burn:])

    else:
        raise ValueError("Invalid mode. Use 'simulated' or 'real'.")

    # Reverse matrices to match plotting format
    Model_update_matrix = data[::-1]
    Model_conditional_matrix = psychs[::-1]

    return Model_update_matrix, Model_conditional_matrix


