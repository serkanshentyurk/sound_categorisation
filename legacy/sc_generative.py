import numpy as np
from scipy.stats import norm
from scipy.integrate import trapezoid
from legacy.be_generative import Noise_generator,Delta_repulsion,find_closest_element


###########################################################################################
# Models
# SC Model


def calculate_choice_and_rewards(x, A_initial_distribution, B_initial_distribution, gamma, sigma_update, s_hat, categories, no_response, seed):

    # Inputs:
    # x: Array representing the stimulus space.
    # A_initial_distribution: Initial belief of agent about category A
    # B_initial_distribution: Initial belief of agent about category B
    # gamma: A scalar that controls how much the subject's belief is updated after receiving feedback.
    # sigma_update: A scalar for the standard deviation of the Gaussian window used to update the belief distributions.
    # s_hat: Array of perceived stimulus values.
    # categories: Array of true categorical labels (0 for Category A, 1 for Category B) for each trial.
    # no_response: Boolean array indicating whether the subject has responded or not in each trial.
    # seed: Integer used for seeding the random number generator for reproducibility.

    # Set random seed for reproducibility.
    np.random.seed(2 * seed)


    # Copy the initial distributions to use for updates.
    A_distribution = A_initial_distribution.copy()
    B_distribution = B_initial_distribution.copy()

    # Initialize a list to store the subject's responses (choices).
    Response = []
    # Get the number of trials.
    n = len(s_hat)

    # Iterate through each trial to compute choice and update belief distributions.
    for i in range(n):
        # Proceed only if the subject responded in the trial (no_response is False).
        if no_response[i] == False:
            # Get the perceived stimulus for the current trial and find the closest element in x.
            s = s_hat[i].astype(np.float32)
            j = find_closest_element(x, s)

            # Calculate the probability of choosing Category A (P_A) and Category B (P_B).
            # Safe division guard: if the sum of distributions is zero, set P_A and P_B to 0.5.
            denom = A_distribution[j] + B_distribution[j]
            if denom <= 0:
                print(f"[calc_choice] WARNING trial={i}, stim={s:.4f}, bin_idx={j}, "
                      f"x[{j}]={x[j]:.4f}, A={A_distribution[j]:.3e}, "
                      f"B={B_distribution[j]:.3e}, sum={denom:.3e}")
                P_B = 0.5
            else:
                P_B = B_distribution[j] / denom            # Use a Bernoulli (binomial with n=1) distribution to make a choice, where 1 = Category B, 0 = Category A.
            choice = np.random.binomial(1, P_B)
            Response.append(choice)

            # Apply a Gaussian window centered at x[j] to update beliefs.
            g = norm.pdf(x, loc=x[j], scale=sigma_update)

            # Set a negative feedback coefficient (for incorrect choices).
            neg_gamma = gamma

            # Update belief distributions based on the subject's choice and feedback:

            # Case 1: Subject chose Category A (choice == 0) and the correct category is A.
            if choice == 0 and categories[i] == 0:
                # Strengthen belief in Category A using positive feedback.
                A_distribution = A_distribution * gamma + g * (1 - gamma)
                # Normalize the belief distribution.
                A_distribution = A_distribution / trapezoid(A_distribution, x)

            # Case 2: Subject chose Category A but the correct category is B.
            elif choice == 0 and categories[i] == 1:
                # Reduce belief in Category A using negative feedback.
                A_distribution = A_distribution * neg_gamma - g * (1 - neg_gamma)

                # Check for negative values in the distribution and shift it if necessary.
                min_yA = np.min(A_distribution)
                if min_yA < 0:
                    A_distribution = A_distribution + np.abs(min_yA)

                # Normalize the belief distribution.
                A_distribution = A_distribution / trapezoid(A_distribution, x)

            # Case 3: Subject chose Category B (choice == 1) and the correct category is B.
            elif choice == 1 and categories[i] == 1:
                # Strengthen belief in Category B using positive feedback.
                B_distribution = B_distribution * gamma + g * (1 - gamma)
                # Normalize the belief distribution.
                B_distribution = B_distribution / trapezoid(B_distribution, x)

            # Case 4: Subject chose Category B but the correct category is A.
            elif choice == 1 and categories[i] == 0:
                # Reduce belief in Category B using negative feedback.
                B_distribution = B_distribution * neg_gamma - g * (1 - neg_gamma)

                # Check for negative values in the distribution and shift it if necessary.
                min_yB = np.min(B_distribution)
                if min_yB < 0:
                    B_distribution = B_distribution + np.abs(min_yB)

                # Normalize the belief distribution.
                B_distribution = B_distribution / trapezoid(B_distribution, x)

        # If the subject did not respond, append NaN for the response.
        else:
            Response.append(np.nan)

    # Final belief distributions for Category A and B after all updates.
    yA = A_distribution
    yB = B_distribution

    # Calculate rewards: 1 for correct choice (category matches response), otherwise 0.
    rewards = [1 if categories[i] == Response[i] else 0 for i in range(len(categories))]

    # Return the subject's responses, rewards, and final belief distributions for A and B.
    return Response, rewards, yA, yB

def SC_model(x, A_initial_distribution, B_initial_distribution, s_hat, categories, sigma_noise, A_repulsion, gamma, 
                                sigma_update, no_response, seed, burn_in_seed, mode):
                 
    """
    # Inputs:
    # x: Array representing the stimulus space.
    # A_initial_distribution: Initial belief of agent about category A
    # B_initial_distribution: Initial belief of agent about category B
    # s_hat: Perceived stimulus values for the current trials.
    # categories: Array of true categorical labels (0 for Category A, 1 for Category B) for each trial.
    # sigma_noise: Noise level added to the stimulus values.
    # A_repulsion: Repulsion parameter applied to the stimulus values.
    # gamma: A scalar controlling how much the subject's belief is updated after feedback.
    # sigma_update: Standard deviation of the Gaussian window used to update the belief distributions.
    # no_response: Boolean array indicating whether the subject has responded or not in each trial.
    # seed: Integer for seeding the random number generator for reproducibility.
    # burn_in_seed: Integer used for seeding the random number generator for reproducibility.(Corresponds to randomness of burn-in phase)

        BE model that supports two modes:
    - 'simulated': Adds 1000 simulated expert trials to the beginning.
    - 'real': Uses only actual data and skips the first n_burn trials.
    """

    if mode == 'simulated':
        # simulated mode: simulate expert behavior
        np.random.seed(burn_in_seed)
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

        # Calculate model choices and rewards using the choice and reward calculation function.
        chooseB, rewards, yA, yB = calculate_choice_and_rewards(x, A_initial_distribution, B_initial_distribution,
                                         gamma, sigma_update, s_hat, categories, no_response, seed)

        output_choiceB = np.array(chooseB[m:])
        output_rewards = np.array(rewards[m:])

    elif mode == 'real':
        # real mode: skip early real trials
        n_burn = 200
        
        # Calculate model choices and rewards using the choice and reward calculation function.
        chooseB, rewards, yA, yB = calculate_choice_and_rewards(x, A_initial_distribution, B_initial_distribution,
                                         gamma, sigma_update, s_hat, categories, no_response, seed)


        output_choiceB = np.array(chooseB[n_burn:])
        output_rewards = np.array(rewards[n_burn:])

    else:
        raise ValueError("Invalid mode. Use 'simulated' or 'real'.")


    # Output update matrix
    return output_choiceB, output_rewards

