import numpy as np
from scipy.optimize import fsolve
import pandas as pd

# As explained in readme.md file, for both of asymmetric distributions
# We have to solve these two equations (i) k = e^-λ  (ii) k+λ=2
# Define the equation
def equation(x):
    return x + np.exp(-x) - 2


def HardB(lam, n_samples):
    """
    Sample from the density
      f(x) = lam * exp(-lam * x) + exp(-lam)
    on the interval [0, 1] using rejection sampling.

    Parameters:
        lam      : parameter lambda (λ)
        n_samples: desired number of samples
    Returns:
        A NumPy array of samples.
    """
    samples = []

    while len(samples) < n_samples:
        # Generate candidate from Uniform(0, 1)
        x_candidate = np.random.uniform(0, 1)
        # Generate uniform random number for acceptance decision
        u = np.random.uniform(0, 1)
        # Compute target density at x_candidate
        f_x = lam * np.exp(-lam * x_candidate) + np.exp(-lam)
        # Accept the candidate if u is below the ratio f(x_candidate) / 2
        if u <= f_x / 2:
            samples.append(x_candidate)

    return np.array(samples)


def HardA(lam, n_samples):
    """
    Sample from the density
      f(x) = lam * exp(lam * x) + exp(-lam)
    on the interval [-1, 0] using rejection sampling.

    Parameters:
        lam      : parameter lambda (λ)
        n_samples: desired number of samples
    Returns:
        A NumPy array of samples.
    """
    samples = []

    while len(samples) < n_samples:
        # Generate candidate from Uniform(-1, 0)
        x_candidate = np.random.uniform(-1, 0)
        # Generate uniform random number for acceptance decision
        u = np.random.uniform(0, 1)
        # Compute target density at x_candidate
        f_x = lam * np.exp(lam * x_candidate) + np.exp(-lam)
        # Accept the candidate if u is below the ratio f(x_candidate) / 2
        if u <= f_x / 2:
            samples.append(x_candidate)

    return np.array(samples)


def uniform_B(n_samples):
    """
    Generate n_samples from Uniform(0, 1).

    Parameters:
        n_samples: desired number of samples
    Returns:
        A NumPy array of uniform samples from the interval [0, 1].
    """

    return np.random.uniform(0, 1, n_samples)


def uniform_A(n_samples):
    """
    Generate n_samples from Uniform(-1, 0).

    Parameters:
        n_samples: desired number of samples
    Returns:
        A NumPy array of uniform samples from the interval [-1, 0].
    """

    return np.random.uniform(-1, 0, n_samples)


def Asym_left(n_samples, seed):
    """
    At each trial choose between uniform_A and HardB with equal chance and sample from the chosen one .

    Parameters:
        n_samples: desired number of samples
        seed      : random seed for reproducibility
    Returns:
        A NumPy array of samples.
    """
    np.random.seed(seed)

    # Use fsolve to find the root
    root = fsolve(equation, 0)  # 0 is the initial guess
    lam = root[0]

    samples = []
    for _ in range(n_samples):
        # Choose between uniform_A and HardB with equal chance
        if np.random.rand() < 0.5:
            # Sample from uniform_A
            samples.append(uniform_A(1)[0])
        else:
            # Sample from HardB
            samples.append(HardB(lam, 1)[0])

    return np.array(samples)


def Asym_right(n_samples, seed):
    """
    At each trial choose between uniform_B and HardA with equal chance and sample from the chosen one.

    Parameters:
        n_samples: desired number of samples
        seed      : random seed for reproducibility
    Returns:
        A NumPy array of samples.
    """
    np.random.seed(seed)

    # Use fsolve to find the root
    root = fsolve(equation, 0)  # 0 is the initial guess
    lam = root[0]

    samples = []
    for _ in range(n_samples):
        # Choose between uniform_B and HardA with equal chance
        if np.random.rand() < 0.5:
            # Sample from uniform_B
            samples.append(uniform_B(1)[0])
        else:
            # Sample from HardA
            samples.append(HardA(lam, 1)[0])

    return np.array(samples)


def Uniform(n_samples, seed):
    """
    At each trial choose between uniform_A and uniform_B with equal chance and sample from the chosen one.

    Parameters:
        n_samples: desired number of samples
        seed      : random seed for reproducibility
    Returns:
        A NumPy array of samples.
    """
    np.random.seed(seed)
    samples = []
    for _ in range(n_samples):
        # Choose between uniform_A and uniform_B with equal chance
        if np.random.rand() < 0.5:
            # Sample from uniform_A
            samples.append(uniform_A(1)[0])
        else:
            # Sample from uniform_B
            samples.append(uniform_B(1)[0])

    return np.array(samples)


def generate_trials_and_blocks(total_trials, min_block_size, max_block_size, seed=None):
    """
    Generates a block and trial array for the specified number of total trials and block size range.

    Args:
    - total_trials (int): The total number of trials to be divided into blocks.
    - min_block_size (int): The minimum size of a block.
    - max_block_size (int): The maximum size of a block.
    - seed (int, optional): Seed for reproducibility (default is None).

    Returns:
    - block (numpy.ndarray): An array with block IDs corresponding to each trial.
    - trial (numpy.ndarray): An array with trial numbers within each block.
    """
    # Initialize block sizes and trials
    block_sizes = []
    remaining_trials = total_trials

    # Set seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # Create blocks with random sizes
    while remaining_trials > 0:
        block_size = np.random.randint(min_block_size, max_block_size + 1)
        block_sizes.append(block_size)
        remaining_trials -= block_size

    # Adjust the last block to ensure the total number of trials matches
    if remaining_trials < 0:
        block_sizes[-1] += remaining_trials  # Add the excess to the last block

    # Create block array and trials array
    block = []
    trial = []

    # Trial numbering resets at the start of each block
    for block_id, size in enumerate(block_sizes):
        block.extend([block_id] * size)  # Add the block id
        trial.extend(range(1, size + 1))  # Add trial numbers starting from 1

    # Convert to numpy arrays
    block = np.array(block)
    trial = np.array(trial)

    return block, trial


def generate_and_merge_trials(total_trials, min_block_size, max_block_size, iterations=3, seed=None):
    """
    Generates trials and block arrays multiple times and merges them.

    Args:
        total_trials (int): Total number of trials for each generation.
        min_block_size (int): Minimum block size.
        max_block_size (int): Maximum block size.
        iterations (int): How many times to generate the arrays.
        seed (int, optional): If provided, will be used to seed the random generator for each iteration with an offset in order to make each iteration reproducible.

    Returns:
        merged_block (np.ndarray): Merged block array with unique block numbers.
        merged_trial (np.ndarray): Merged trial array (trial counter resets per block).
    """
    blocks_list = []
    trials_list = []
    block_offset = 0


    for i in range(iterations):
        # If a seed is provided, use it with i offset for reproducibility of each iteration independently
        current_seed = seed + i if seed is not None else None
        # Generate arrays. For subsequent iterations, we do re-seed
        block, trial = generate_trials_and_blocks(total_trials, min_block_size, max_block_size, seed=current_seed)

        # Adjust block numbers so that block IDs across iterations are unique.
        block = block + block_offset

        # Determine number of blocks generated in this iteration.
        num_blocks = block.max() - block_offset + 1
        block_offset += num_blocks

        blocks_list.append(block)
        trials_list.append(trial)


    merged_block = np.concatenate(blocks_list)
    merged_trial = np.concatenate(trials_list)

    return merged_block, merged_trial



def switch(n_LtoR, n_RtoL, chunk_first, chunk_second, seed):
    
    # Calculate required number of samples for L and R
    samples_L = n_LtoR * chunk_first + n_RtoL * chunk_second
    samples_R = n_LtoR * chunk_second + n_RtoL * chunk_first
        
    # Generate base arrays
    total_asymR = Asym_right(samples_R, seed)
    total_asymL = Asym_left(samples_L, seed)
    
    # First halves for LtoR
    first_half_asymR = total_asymR[:n_LtoR * chunk_second]
    first_half_asymL = total_asymL[:n_LtoR * chunk_first]
    
    # Second halves for RtoL
    second_half_asymR = total_asymR[-n_RtoL * chunk_first:]
    second_half_asymL = total_asymL[-n_RtoL * chunk_second:]
    
    def build_blocks(A, B, n_blocks, chunk_a_size, chunk_b_size, switch_label, dist_a, dist_b):
        records = []
        a_index, b_index = 0, 0
        for block_num in range(1, n_blocks + 1):
            # First chunk
            a_chunk = A[a_index:a_index + chunk_a_size]
            a_index += chunk_a_size
            for trial_num, val in enumerate(a_chunk, start=1):
                records.append([switch_label, block_num, trial_num, dist_a, val, False])
            
            # Second chunk
            b_chunk = B[b_index:b_index + chunk_b_size]
            b_index += chunk_b_size
            for trial_num, val in enumerate(b_chunk, start=chunk_a_size + 1):
                records.append([switch_label, block_num, trial_num, dist_b, val, False])
        return records
    
    # LtoR blocks
    LtoR_records = build_blocks(first_half_asymL, first_half_asymR, n_LtoR, 
                                 chunk_first, chunk_second, "LtoR", "Asym_left", "Asym_right")
    
    # RtoL blocks
    RtoL_records = build_blocks(second_half_asymR, second_half_asymL, n_RtoL, 
                                 chunk_first, chunk_second, "RtoL", "Asym_right", "Asym_left")
    
    # Combine into DataFrame
    all_records = LtoR_records + RtoL_records
    df = pd.DataFrame(all_records, columns=[
        "Switch", "switch block", "Trial", "Distribution", "stim_relative", "No_response"
    ])
    
    return df


